"""Measure complete local training steps and representative generation."""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from carl.icl.intent_data import validate_prediction
from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import generate
from carl.icl.intent_generator_experiment import initialise
from carl.icl.intent_generator_experiment import memory
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_generator_experiment import sync_gpu
from carl.icl.intent_generator_experiment import tensor_example
from carl.icl.intent_representation_training import expired
from carl.icl.policy_relative.checks import _identity
from carl.icl.policy_relative.configuration import _model_revision
from carl.icl.policy_relative.prompts import encode_target
from carl.icl.policy_relative.prompts import generator_messages
from carl.icl.policy_relative.provenance import code_provenance
from carl.icl.policy_relative.types import Arm


def resource_records(path: str | Path) -> list[dict[str, Any]]:
    """Select checked, unmeasured fitting candidates without using test records."""
    rows = [
        row
        for row in read_records(path)
        if row.get("split") in (DataSplit.TRAIN, DataSplit.VALIDATION)
    ]
    layouts: dict[str, set[str]] = defaultdict(set)
    identifiers: set[str] = set()
    for row in rows:
        record_id, _, layout = _identity(row)
        if record_id in identifiers:
            raise ValueError("resource-check record IDs must be unique")
        identifiers.add(record_id)
        layouts[row["split"]].add(layout)
        validation = row.get("validation")
        if not isinstance(validation, dict) or validation.get("accepted") is not True:
            raise ValueError("resource-check requires checked candidate intents")
        if row.get("policy_measurements") is not None:
            raise ValueError("resource-check accepts unmeasured candidates only")
        errors = validate_prediction(row.get("intent"), row["start_observation"])
        if errors:
            raise ValueError(f"resource-check candidate intent is invalid: {errors}")
    if not layouts[DataSplit.TRAIN] or not layouts[DataSplit.VALIDATION]:
        raise ValueError("resource-check requires both train and validation candidates")
    if layouts[DataSplit.TRAIN] & layouts[DataSplit.VALIDATION]:
        raise ValueError("resource-check train and validation layouts overlap")
    return rows


def resource_encodings(
    tokenizer: Any, rows: list[dict[str, Any]], max_length: int, max_new_tokens: int
) -> list[tuple[dict[str, Any], dict[str, list[int]], dict[str, Any]]]:
    """Measure complete prompt and target lengths at a diagnostic scale of 0.5."""
    result = []
    for row in rows:
        encoded = encode_target(tokenizer, row, Arm.SCALED, 0.5, max_length)
        prompt_length = sum(value == -100 for value in encoded["labels"])
        if prompt_length + max_new_tokens > max_length:
            raise ValueError(
                "resource-check prompt plus generation exceeds max_length; no truncation"
            )
        lengths = {
            "record_id": row["record_id"],
            "context_id": row["context_id"],
            "layout_id": row["layout_id"],
            "split": row["split"],
            "prompt_tokens": prompt_length,
            "response_tokens": len(encoded["input_ids"]) - prompt_length,
            "complete_example_tokens": len(encoded["input_ids"]),
            "truncated": False,
        }
        result.append((row, encoded, lengths))
    return result


def resource_check(args: argparse.Namespace) -> None:
    """Benchmark a real checked intent with a synthetic scale input, not a label."""
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a fresh resource-check output")
    started = time.monotonic()
    code_sha256, code_files_sha256 = code_provenance()
    metadata: dict[str, Any] = {
        "measurement_kind": "resource_only_no_luna_labels",
        "synthetic_requested_scale": 0.5,
        "scale_interpretation": "Diagnostic input only. It is not measured u and must never be used as a label.",
        "data_sha256": file_sha256(args.data),
        "model": str(Path(args.model).resolve()),
        "revision": args.revision,
        "dtype": args.dtype,
        "seed": args.seed,
        "threads": args.threads,
        "rank": args.rank,
        "lora_targets": ["q_proj", "v_proj"],
        "accumulation": args.accumulation,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "gradient_checkpointing": args.gradient_checkpointing,
        "enable_thinking": False,
        "code_sha256": code_sha256,
        "code_files_sha256": code_files_sha256,
        "command": sys.argv,
    }
    atomic_json(output, {**metadata, "status": "initializing"})
    torch = None
    try:
        rows = resource_records(args.data)
        _model_revision(Path(args.model), args.revision)
        if expired(args):
            raise TimeoutError(
                "resource-check deadline reached before loading the model"
            )
        if args.require_cuda:
            import torch as torch_module

            if not torch_module.cuda.is_available():
                raise RuntimeError(
                    "CUDA required but unavailable; no silent CPU fallback"
                )
        torch, tokenizer, model, device = initialise(args, train=True)
        prepared = resource_encodings(
            tokenizer, rows, args.max_length, args.max_new_tokens
        )
        training = [item for item in prepared if item[0]["split"] == DataSplit.TRAIN]
        # Repeated use of one long example stresses the real microbatch without
        # presenting repeated examples as policy measurements or new data.
        longest = max(training, key=lambda item: len(item[1]["input_ids"]))
        generation_case = max(prepared, key=lambda item: item[2]["prompt_tokens"])
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
        )
        model.train()
        sync_gpu(torch)
        step_started = time.monotonic()
        tokens = 0
        for _ in range(args.accumulation):
            tensors = tensor_example(torch, longest[1], device)
            tokens += len(longest[1]["input_ids"])
            loss = model(**tensors).loss / args.accumulation
            if not math.isfinite(float(loss.detach().cpu())):
                raise ValueError("resource-check loss is not finite")
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync_gpu(torch)
        seconds = time.monotonic() - step_started
        if expired(args):
            raise TimeoutError("resource-check deadline reached before generation")
        raw, generated, generation_seconds = generate(
            torch,
            model,
            tokenizer,
            generation_case[0],
            args,
            device,
            messages=generator_messages(generation_case[0], Arm.SCALED, 0.5),
        )
        summary = {
            **metadata,
            "status": "complete",
            "device": device,
            "model_weights_sha256": file_sha256(Path(args.model) / "model.safetensors"),
            "model_config_sha256": file_sha256(Path(args.model) / "config.json"),
            "tokenizer_sha256": file_sha256(Path(args.model) / "tokenizer.json"),
            "token_lengths": [item[2] for item in prepared],
            "training_record_id": longest[0]["record_id"],
            "generation_record_id": generation_case[0]["record_id"],
            "complete_step_seconds": seconds,
            "training_tokens": tokens,
            "training_tokens_per_second": tokens / seconds,
            "generation_tokens": generated,
            "generation_seconds": generation_seconds,
            "generation_tokens_per_second": generated / generation_seconds
            if generation_seconds
            else None,
            "generation_sample": raw,
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "Scale 0.5 is a diagnostic input, not a Luna measurement.",
                "Later measured scales can tokenise differently and require their own length check.",
                "No adapter checkpoint or policy-relative labels are produced.",
            ],
            **memory(torch),
        }
        atomic_json(output, summary)
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "status",
                        "measurement_kind",
                        "complete_step_seconds",
                        "generation_seconds",
                        "gpu_peak_reserved_bytes",
                    )
                },
                indent=2,
            )
        )
    except Exception as error:
        try:
            measured_memory = memory(torch) if torch is not None else {}
        except Exception:
            measured_memory = {}
        atomic_json(
            output,
            {
                **metadata,
                "status": "failed",
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                **measured_memory,
            },
        )
        raise
