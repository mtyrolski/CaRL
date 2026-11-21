"""Frozen evaluation and resumable output for intent generators."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from carl.icl.intent_domain import SokobanObservation, decode_intent, validate_intent_proposal
from carl.icl.intents.generation.backend import generate
from carl.icl.intents.generation.backend import initialise
from carl.icl.intents.generation.backend import local_snapshot
from carl.icl.intents.generation.backend import memory
from carl.icl.intents.generation.checkpoint import (
    has_adapter_contract, predicate_prompt_contract, verify_adapter_artifact, verify_held_out_records,
    verify_validation_selection,
)
from carl.icl.intents.generation.data import SYSTEM
from carl.icl.intents.generation.data import DataSplit
from carl.icl.intents.generation.data import GeneratorMethod
from carl.icl.intents.generation.data import atomic_json
from carl.icl.intents.generation.data import evaluation_records
from carl.icl.intents.generation.data import file_sha256
from carl.icl.intents.generation.data import parse_json
from carl.icl.intents.generation.data import predicate_record
from carl.icl.intents.generation.data import read_records
from carl.icl.predicate_prompts import GENERATOR_PROMPT_VERSION, generator_messages


def evaluation_configuration(args: argparse.Namespace) -> dict[str, Any]:
    adapter = Path(args.adapter) if args.adapter else None
    return {
        "model": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "adapter_weights_sha256": file_sha256(adapter / "adapter_model.safetensors")
        if adapter
        else None,
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json")
        if adapter
        else None,
        "data_sha256": file_sha256(args.data),
        "test_ids_sha256": file_sha256(args.test_ids) if args.test_ids else None,
        "prompt_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "dtype": args.dtype,
        "device": getattr(args, "device", "auto"),
        "seed": args.seed,
        "test_boards": args.test_boards,
        "decoding": "greedy",
        "enable_thinking": False,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
    }


def prepare_evaluation_output(
    output: Path,
    configuration: dict[str, Any],
    records: list[dict[str, Any]],
    resume: bool,
) -> list[dict[str, Any]]:
    """Verify an interrupted output belongs to this exact frozen experiment."""
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".run.json")
    if not resume and (output.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Evaluation output already exists: {output}; use --resume"
        )
    if output.exists() and not metadata_path.exists():
        raise ValueError("Cannot resume output without its original configuration")
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Evaluation configuration changed; refusing to resume")
    else:
        atomic_json(metadata_path, configuration)
    if not output.exists():
        return []
    raw = output.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        tail = raw[boundary:]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            saved_tail = output.with_name(
                f"{output.name}.partial-{hashlib.sha256(tail).hexdigest()[:12]}"
            )
            saved_tail.write_bytes(tail)
            raw = raw[:boundary]
        else:
            raw += b"\n"
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, output)
    rows = read_records(output)
    method = configuration.get("method") or (
        GeneratorMethod.ADAPTED if configuration["adapter"] else GeneratorMethod.BASE
    )
    if len(rows) > len(records):
        raise ValueError("Output contains more contexts than the frozen selection")
    for row, record in zip(rows, records, strict=False):
        if (
            row.get("record_id") != record["record_id"]
            or row.get("layout_id") != record["layout_id"]
            or row.get("split") != DataSplit.TEST
            or row.get("method") != method
        ):
            raise ValueError("Output does not match the ordered frozen test selection")
        if not isinstance(row.get("raw_output"), str) or not isinstance(
            row.get("json_valid"), bool
        ):
            raise ValueError("Output has malformed generation fields")
        try:
            parsed = parse_json(row["raw_output"])
            valid = True
        except (json.JSONDecodeError, ValueError):
            parsed, valid = None, False
        if row["json_valid"] != valid or row.get("parsed") != parsed:
            raise ValueError("Stored JSON result disagrees with its raw output")
    return rows


def evaluate(args: argparse.Namespace) -> None:
    records = evaluation_records(
        read_records(args.data, DataSplit.TEST), args.test_ids, args.test_boards
    )
    if not records:
        raise ValueError("no held-out test boards")
    output = Path(args.output)
    configuration = evaluation_configuration(args)
    predicate_prompt = any(predicate_record(record) for record in records)
    if args.adapter:
        adapter = Path(args.adapter)
        if predicate_prompt or has_adapter_contract(adapter):
            snapshot = local_snapshot(args.model, args.revision)
            manifest = verify_adapter_artifact(adapter, base=snapshot, revision=args.revision, role="generator")
            verify_validation_selection(adapter, manifest)
            verify_held_out_records(records, manifest)
            prompt = manifest["prompt"]
            predicate_prompt = prompt["version"] == GENERATOR_PROMPT_VERSION
            if predicate_prompt and prompt != predicate_prompt_contract("unscaled"):
                raise ValueError("use policy_relative_lora for an adapter trained with a controller scale")
            configuration.update(adapter_fingerprint=manifest["fingerprint"], prompt_sha256=prompt["prompt_sha256"])
    if predicate_prompt:
        configuration["prompt_sha256"] = predicate_prompt_contract("unscaled")["prompt_sha256"]
    previous = prepare_evaluation_output(output, configuration, records, args.resume)
    torch, tokenizer, model, device = initialise(
        args, adapter=args.adapter, train=False
    )
    started = time.monotonic()
    with output.open("a", encoding="utf-8", buffering=1) as dest:
        for record in records[len(previous) :]:
            messages = generator_messages(record["start_observation"], record["task_goal"], None, "unscaled") \
                if predicate_prompt else None
            output_text, tokens, elapsed = generate(
                torch, model, tokenizer, record, args, device, messages=messages,
            )
            try:
                parsed = parse_json(output_text)
                valid_json = True
            except (json.JSONDecodeError, ValueError):
                parsed = None
                valid_json = False
            row = {
                "record_id": record["record_id"],
                "layout_id": record["layout_id"],
                "split": DataSplit.TEST,
                "method": GeneratorMethod.ADAPTED
                if args.adapter
                else GeneratorMethod.BASE,
                "raw_output": output_text,
                "json_valid": valid_json,
                "parsed": parsed,
                "generated_tokens": tokens,
                "generation_seconds": elapsed,
            }
            if predicate_prompt:
                try:
                    intent = decode_intent(output_text)
                    errors = list(validate_intent_proposal(intent, SokobanObservation(tuple(record["start_observation"]))))
                except ValueError as error:
                    errors = [str(error)]
                row.update(intent_valid=not errors, validation_errors=errors)
            dest.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "record_id": record["record_id"],
                        "json_valid": valid_json,
                        "seconds": elapsed,
                    }
                ),
                flush=True,
            )
    atomic_json(
        output.with_suffix(".summary.json"),
        {
            "status": "complete",
            "n": len(records),
            **configuration,
            "resumed_records": len(previous),
            "elapsed_seconds": time.monotonic() - started,
            **memory(torch),
        },
    )
