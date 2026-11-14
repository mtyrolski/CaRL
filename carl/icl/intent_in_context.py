"""Evaluate a fixed two-example in-context prompt for Sokoban intent generation.

Both demonstrations come from checked train layouts. Only their current board,
task and checked intent are shown; the held-out query contains no target or
future observation. The frozen support manifest makes selection reproducible.
"""

import argparse
import json
import os
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import GeneratorMethod
from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import evaluation_configuration
from carl.icl.intent_generator_experiment import evaluation_records
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import generate
from carl.icl.intent_generator_experiment import initialise
from carl.icl.intent_generator_experiment import memory
from carl.icl.intent_generator_experiment import parse_json
from carl.icl.intent_generator_experiment import prepare_evaluation_output
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_generator_experiment import target_text


class Mode(StrEnum):
    FREEZE_SUPPORT = "freeze-support"
    INSPECT_PROMPTS = "inspect-prompts"
    EVALUATE = "evaluate"


SUPPORT_EFFECT_TYPES = ("box", "goal_conjunction")


def select_support(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one checked training example of each prespecified effect type."""
    candidates = sorted(
        (row for row in records
         if row.get("split") == DataSplit.TRAIN and row.get("validation", {}).get("accepted") is True),
        key=lambda row: (row["layout_id"], row["record_id"]),
    )
    selected: list[dict[str, Any]] = []
    for effect_type in SUPPORT_EFFECT_TYPES:
        used_layouts = {row["layout_id"] for row in selected}
        match = next((row for row in candidates
                      if row.get("effect_type") == effect_type and row["layout_id"] not in used_layouts), None)
        if match is None:
            raise ValueError(f"No checked training example for effect type {effect_type} on a distinct layout")
        selected.append(match)
    return selected


def freeze_support(data: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Support manifest already exists: {destination}")
    selected = select_support(read_records(data))
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        destination, {
            "schema_version":
                1,
            "dataset_sha256":
                file_sha256(data),
            "selection":
                "First checked train example of each specified effect type by (layout_id, record_id), on distinct layouts",
            "effect_types":
                list(SUPPORT_EFFECT_TYPES),
            "record_ids": [row["record_id"] for row in selected],
            "layout_ids": [row["layout_id"] for row in selected],
        })


def load_support(data: Path, manifest_path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = manifest.get("record_ids")
    if manifest.get("dataset_sha256") != file_sha256(data):
        raise ValueError("Support manifest belongs to a different dataset")
    if not isinstance(ids, list) or len(ids) != len(SUPPORT_EFFECT_TYPES) or len(set(ids)) != len(ids):
        raise ValueError("Support manifest must contain two distinct record IDs")
    lookup = {row["record_id"]: row for row in records}
    if len(lookup) != len(records) or any(record_id not in lookup for record_id in ids):
        raise ValueError("Support record IDs are missing or duplicated in the dataset")
    support = [lookup[record_id] for record_id in ids]
    expected = select_support(records)
    if [row["record_id"] for row in expected] != ids:
        raise ValueError("Support IDs do not match the prespecified deterministic selection")
    if manifest.get("layout_ids") != [row["layout_id"] for row in support]:
        raise ValueError("Support layout metadata does not match selected records")
    return support


def in_context_messages(record: dict[str, Any], support: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Show checked support labels, never the held-out query's future or label."""
    query = prompt_messages(record)
    messages = [query[0]]
    for example in support:
        if example.get("split") != DataSplit.TRAIN or example.get("validation", {}).get("accepted") is not True:
            raise ValueError("In-context demonstrations must be checked training records")
        if example["layout_id"] == record["layout_id"]:
            raise ValueError("A demonstration cannot share the query layout")
        messages.extend([prompt_messages(example)[1], {"role": "assistant", "content": target_text(example)}])
    messages.append(query[1])
    return messages


def build_prompts(tokenizer: Any, records: list[dict[str, Any]], support: list[dict[str, Any]], max_length: int,
                  max_new_tokens: int) -> list[dict[str, Any]]:
    """Record exact chat messages and rendered prompts, without truncating."""
    result = []
    for record in records:
        messages = in_context_messages(record, support)
        options = {"add_generation_prompt": True, "enable_thinking": False}
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, **options)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        count = len(token_ids)
        if count + max_new_tokens > max_length:
            raise ValueError(f"Prompt {record['record_id']} needs {count} + {max_new_tokens} tokens; "
                             f"limit is {max_length}; no truncation performed")
        result.append({
            "record_id": record["record_id"],
            "layout_id": record["layout_id"],
            "messages": messages,
            "rendered_prompt": rendered,
            "prompt_tokens": count,
        })
    return result


def save_prompts(path: Path, prompts: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("Existing prompt file differs; refusing to overwrite")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], Path]:
    from transformers import AutoTokenizer

    all_records = read_records(args.data)
    records = evaluation_records([row for row in all_records if row.get("split") == DataSplit.TEST], args.test_ids, 0)
    if not records:
        raise ValueError("No frozen held-out contexts")
    frozen = json.loads(Path(args.test_ids).read_text(encoding="utf-8"))
    if frozen.get("dataset_sha256") != file_sha256(args.data):
        raise ValueError("Frozen test IDs belong to a different dataset")
    support_data = Path(args.support_data or args.data)
    support = load_support(support_data, Path(args.support), read_records(support_data))
    if {row["layout_id"] for row in support} & {row["layout_id"] for row in records}:
        raise ValueError("Support and test layouts overlap")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer does not provide a chat template")
    prompts = build_prompts(tokenizer, records, support, args.max_length, args.max_new_tokens)
    prompt_path = Path(args.output).with_suffix(".prompts.jsonl")
    save_prompts(prompt_path, prompts)
    return records, support, prompts, prompt_path


def stop_if_deadline(output: Path, stop_monotonic: float | None, processed: int, planned: int, started: float) -> None:
    """Preserve partial progress and fail the stage when its deadline expires."""
    if stop_monotonic is None or time.monotonic() < stop_monotonic:
        return
    atomic_json(
        output.with_suffix(".partial-summary.json"), {
            "status": "stopped",
            "reason": "deadline",
            "processed": processed,
            "planned": planned,
            "elapsed_seconds_since_initialisation_start": time.monotonic() - started,
        })
    raise SystemExit("Stopped at configured deadline; partial predictions retained")


def mark_partial_superseded(output: Path) -> None:
    """Retain interruption details while identifying a successful resumption."""
    partial = output.with_suffix(".partial-summary.json")
    if not partial.exists():
        return
    record = json.loads(partial.read_text(encoding="utf-8"))
    record["superseded_by"] = output.with_suffix(".summary.json").name
    record["final_status"] = "complete"
    atomic_json(partial, record)


def run(args: argparse.Namespace) -> None:
    records, support, prompts, prompt_path = prepare(args)
    if args.mode == Mode.INSPECT_PROMPTS:
        counts = [row["prompt_tokens"] for row in prompts]
        print(
            json.dumps(
                {
                    "contexts": len(counts),
                    "min_prompt_tokens": min(counts),
                    "max_prompt_tokens": max(counts),
                    "generation_tokens": args.max_new_tokens,
                    "max_length": args.max_length,
                    "support_record_ids": [row["record_id"] for row in support],
                    "prompts_path": str(prompt_path)
                },
                indent=2))
        return
    args.adapter = None
    args.test_boards = 0
    configuration = evaluation_configuration(args)
    configuration.update({
        "method": GeneratorMethod.IN_CONTEXT,
        "support_sha256": file_sha256(args.support),
        "support_data_sha256": file_sha256(args.support_data or args.data),
        "support_record_ids": [row["record_id"] for row in support],
        "prompts_sha256": file_sha256(prompt_path),
    })
    output = Path(args.output)
    previous = prepare_evaluation_output(output, configuration, records, args.resume)
    initialisation_started = time.monotonic()
    stop_if_deadline(output, args.stop_monotonic, len(previous), len(records), initialisation_started)
    torch, tokenizer, model, device = initialise(args, train=False)
    started = time.monotonic()
    with output.open("a", encoding="utf-8", buffering=1) as dest:
        pairs = zip(records[len(previous):], prompts[len(previous):], strict=True)
        for index, (record, prompt) in enumerate(pairs, start=len(previous)):
            stop_if_deadline(output, args.stop_monotonic, index, len(records), initialisation_started)
            output_text, tokens, elapsed = generate(torch,
                                                    model,
                                                    tokenizer,
                                                    record,
                                                    args,
                                                    device,
                                                    messages=prompt["messages"])
            try:
                parsed = parse_json(output_text)
                valid_json = True
            except (json.JSONDecodeError, ValueError):
                parsed, valid_json = None, False
            dest.write(
                json.dumps(
                    {
                        "record_id": record["record_id"],
                        "layout_id": record["layout_id"],
                        "split": DataSplit.TEST,
                        "method": GeneratorMethod.IN_CONTEXT,
                        "raw_output": output_text,
                        "json_valid": valid_json,
                        "parsed": parsed,
                        "generated_tokens": tokens,
                        "generation_seconds": elapsed,
                    },
                    ensure_ascii=False) + "\n")
            print(json.dumps({
                "record_id": record["record_id"],
                "json_valid": valid_json,
                "seconds": elapsed
            }),
                  flush=True)
    atomic_json(
        output.with_suffix(".summary.json"), {
            "status": "complete",
            "n": len(records),
            **configuration,
            "resumed_records": len(previous),
            "elapsed_seconds": time.monotonic() - started,
            **memory(torch),
        })
    mark_partial_superseded(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=list(Mode))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--support")
    parser.add_argument("--support-data", help="Original checked training dataset, if test data is stored separately")
    parser.add_argument("--test-ids")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == Mode.FREEZE_SUPPORT:
        freeze_support(Path(args.data), Path(args.output))
        return
    if not all((args.support, args.test_ids, args.model, args.revision)):
        parser.error("--support, --test-ids, --model and --revision are required")
    if args.max_length < 1 or args.max_new_tokens < 1:
        parser.error("Token limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
