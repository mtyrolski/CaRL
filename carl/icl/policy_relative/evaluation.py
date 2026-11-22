"""Evaluate structured generator outputs with explicit failure accounting."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from carl.icl.intent_data import validate_prediction
from carl.icl.intent_domain import SokobanObservation, decode_intent, validate_intent_proposal
from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import generate
from carl.icl.intent_generator_experiment import initialise
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_representation_training import expired
from carl.icl.intents.generation.checkpoint import (
    has_adapter_contract, predicate_prompt_contract, verify_adapter_artifact, verify_held_out_records,
    verify_validation_selection,
)
from carl.icl.intents.generation.data import predicate_record
from carl.icl.intents.representation.normalization import remove_one_outer_fence
from carl.icl.policy_relative.checks import _identity
from carl.icl.policy_relative.configuration import _model_revision
from carl.icl.policy_relative.prompts import generator_messages, prompt_scale_arm
from carl.icl.policy_relative.provenance import code_provenance
from carl.icl.policy_relative.types import Arm
from carl.icl.policy_relative.types import PredictionStatus
from carl.icl.policy_relative.types import SCALED_ARMS, VERSIONED_SCALE_ARMS


def _strict_json(raw: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)


def assess_output(
    raw: str, board: list[str], *, normalize: bool = False, predicate: bool = False,
) -> dict[str, Any]:
    """Parse complete JSON, optionally removing one enclosing Markdown fence."""
    try:
        _strict_json(raw)
        strict_valid = True
    except (TypeError, ValueError, json.JSONDecodeError):
        strict_valid = False
    normalized, removed = remove_one_outer_fence(raw) if normalize else (raw, False)
    try:
        intent = _strict_json(normalized)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": PredictionStatus.INVALID_JSON,
            "json_valid": False,
            "strict_json_valid": strict_valid,
            "normalized_json_valid": False,
            "fence_removed": removed,
            "parsed": None,
            "validation_errors": [str(error)],
        }
    if predicate:
        try:
            errors = list(validate_intent_proposal(decode_intent(intent), SokobanObservation(tuple(board))))
        except ValueError as error:
            errors = [str(error)]
    else:
        errors = validate_prediction(intent, board)
    return {
        "status": PredictionStatus.INVALID_INTENT if errors else PredictionStatus.VALID,
        "json_valid": True,
        "strict_json_valid": strict_valid,
        "normalized_json_valid": True,
        "fence_removed": removed,
        "parsed": intent,
        "validation_errors": errors,
    }


def test_contexts(path: str | Path) -> list[dict[str, Any]]:
    """Read one frozen query per context without consulting any target label."""
    rows = read_records(path)
    if not rows:
        raise ValueError("no test contexts")
    identifiers: set[str] = set()
    contexts: set[str] = set()
    for row in rows:
        record_id, context_id, _ = _identity(row)
        if (
            row.get("split") != DataSplit.TEST
            or record_id in identifiers
            or context_id in contexts
        ):
            raise ValueError("test contexts require unique IDs and split=test")
        identifiers.add(record_id)
        contexts.add(context_id)
    return rows


def verify_adapter(
    adapter: Path, model: Path, revision: str, arm: Arm, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind evaluation to this arm's validation-selected adapter and layouts."""
    training_path = adapter.parent / "configuration.json"
    best_path = adapter.parent / "best.json"
    if not training_path.is_file() or not best_path.is_file():
        raise ValueError(
            "adapter must come from a completed validation-selected training run"
        )
    training: dict[str, Any] = json.loads(training_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    saved = best.get("adapter")
    if not isinstance(saved, str) or Path(saved).resolve() != adapter.resolve():
        raise ValueError("adapter does not match the validation-selected checkpoint")
    expected = {
        "arm": arm,
        "model_revision": revision,
        "model_weights_sha256": file_sha256(model / "model.safetensors"),
        "model_config_sha256": file_sha256(model / "config.json"),
        "tokenizer_sha256": file_sha256(model / "tokenizer.json"),
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "adapter arm, model revision or source files differ from training"
        )
    known_layouts = set(training.get("train_layout_ids", [])) | set(
        training.get("validation_layout_ids", [])
    )
    if not known_layouts:
        raise ValueError("training configuration does not record its layouts")
    if known_layouts & {row["layout_id"] for row in rows}:
        raise ValueError("test layouts overlap generator training or validation")
    return training


def evaluate(args: argparse.Namespace) -> None:
    rows = test_contexts(args.data)
    arm = Arm(args.arm)
    is_scaled = arm in SCALED_ARMS
    requested = args.requested_scale if is_scaled else None
    if not is_scaled and args.requested_scale is not None:
        raise ValueError("unscaled evaluation must not receive --requested-scale")
    if is_scaled and requested is None:
        raise ValueError("scaled evaluation requires --requested-scale")
    model_path = Path(args.model)
    _model_revision(model_path, args.revision)
    manifest = None
    if arm == Arm.BASE:
        if args.adapter is not None:
            raise ValueError("the base arm must not load an adapter")
        adapter = None
        training = None
    else:
        if not args.adapter:
            raise ValueError(
                "adapted evaluation requires a validation-selected adapter"
            )
        adapter = Path(args.adapter)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            if not (adapter / name).is_file():
                raise ValueError(f"adapter is missing {name}")
        if has_adapter_contract(adapter) or arm in VERSIONED_SCALE_ARMS or any(predicate_record(row) for row in rows):
            manifest = verify_adapter_artifact(adapter, base=model_path, revision=args.revision, role="generator")
            verify_validation_selection(adapter, manifest)
            verify_held_out_records(rows, manifest)
            training = manifest["training"]
            if training.get("arm") != arm or manifest["prompt"] != predicate_prompt_contract(prompt_scale_arm(arm)):
                raise ValueError("adapter arm or prompt differs from the pinned checkpoint")
        else:
            training = verify_adapter(adapter, model_path, args.revision, arm, rows)
    predicate = manifest is not None or any(predicate_record(row) for row in rows)
    messages = [generator_messages(row, arm, requested, predicate=predicate) for row in rows]
    output = Path(args.output)
    meta_path = output.with_suffix(".run.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError("use a fresh evaluation output")
    output.parent.mkdir(parents=True, exist_ok=True)
    code_sha256, code_files_sha256 = code_provenance()
    weights = model_path / "model.safetensors"
    if not weights.is_file():
        weights = model_path / "model.safetensors.index.json"
    configuration = {
        "data_sha256": file_sha256(args.data),
        "model": str(Path(args.model).resolve()),
        "revision": args.revision,
        "model_weights_sha256": file_sha256(weights),
        "model_config_sha256": file_sha256(Path(args.model) / "config.json"),
        "tokenizer_sha256": file_sha256(Path(args.model) / "tokenizer.json"),
        "adapter": str(adapter.resolve()) if adapter is not None else None,
        "adapter_weights_sha256": file_sha256(adapter / "adapter_model.safetensors")
        if adapter
        else None,
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json")
        if adapter
        else None,
        "adapter_fingerprint": manifest["fingerprint"] if manifest else None,
        "code_sha256": code_sha256,
        "code_files_sha256": code_files_sha256,
        "arm": arm,
        "requested_scale": requested,
        "training_configuration_sha256": file_sha256(
            adapter.parent / "configuration.json"
        )
        if adapter
        else None,
        "training_comparability_sha256": training["comparability_sha256"]
        if training is not None
        else None,
        "controller_hash": training["controller_hash"]
        if training is not None
        else None,
        "record_ids": [row["record_id"] for row in rows],
        "dtype": args.dtype,
        "device": getattr(args, "device", "auto"),
        "seed": args.seed,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "output_normalisation": (
            "remove_one_outer_fence"
            if arm not in (Arm.UNSCALED, Arm.SCALED)
            else "strict"
        ),
        "enable_thinking": False,
        "command": sys.argv,
    }
    atomic_json(meta_path, {**configuration, "status": "initializing"})
    started, failure = time.monotonic(), None
    model_tuple = None
    try:
        if args.require_cuda:
            import torch as torch_module

            if not torch_module.cuda.is_available():
                raise RuntimeError(
                    "CUDA required but unavailable; no silent CPU fallback"
                )
        model_tuple = initialise(
            args, adapter=str(adapter) if adapter is not None else None, train=False
        )
        torch, tokenizer, model, device = model_tuple
        for prompt in messages:
            tokens = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            if len(tokens) + args.max_new_tokens > args.max_length:
                raise ValueError(
                    "prompt plus generation exceeds max_length; no truncation"
                )
    # Record initialization failures from any supported backend in the evaluation manifest.
    except Exception as error:  # pylint: disable=broad-exception-caught
        failure = f"{type(error).__name__}: {error}"
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        for record, prompt in zip(rows, messages, strict=True):
            row: dict[str, Any] = {
                key: record.get(key)
                for key in ("record_id", "context_id", "layout_id", "family")
            }
            row.update(
                arm=arm,
                requested_scale=requested,
                prompt=prompt,
                raw_output=None,
                parsed=None,
                json_valid=None,
                strict_json_valid=None,
                normalized_json_valid=None,
                fence_removed=None,
                validation_errors=[],
                generated_tokens=None,
                generation_seconds=None,
            )
            if failure is not None:
                row.update(status=PredictionStatus.NOT_ATTEMPTED, error=failure)
            elif expired(args):
                failure = "evaluation deadline reached"
                row.update(status=PredictionStatus.NOT_ATTEMPTED, error=failure)
            else:
                try:
                    assert model_tuple is not None
                    torch, tokenizer, model, device = model_tuple
                    raw, tokens, seconds = generate(
                        torch, model, tokenizer, record, args, device, messages=prompt
                    )
                    row.update(
                        raw_output=raw,
                        generated_tokens=tokens,
                        generation_seconds=seconds,
                        **assess_output(
                            raw,
                            record["start_observation"],
                            normalize=arm not in (Arm.UNSCALED, Arm.SCALED),
                            predicate=predicate,
                        ),
                    )
                # Preserve every unfinished row after an unexpected backend failure.
                except Exception as error:  # pylint: disable=broad-exception-caught
                    failure = f"{type(error).__name__}: {error}"
                    row.update(
                        status=PredictionStatus.INFRASTRUCTURE_ERROR, error=failure
                    )
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    atomic_json(
        meta_path,
        {
            **configuration,
            "status": "failed" if failure else "complete",
            "error": failure,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    if failure:
        raise RuntimeError(f"evaluation stopped: {failure}")
