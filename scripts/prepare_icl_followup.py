#!/usr/bin/env python3
"""Freeze a reproducible in-context evaluation plan without running the model."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import evaluation_records
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_in_context import in_context_messages
from carl.icl.intent_in_context import load_support

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "analysis/intent_generator_20261114T032232Z"
TRANSFER = ROOT / "analysis/lora_matrix_20261114T065804Z/eval_transfer"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def validate_hidden_fields(query: dict[str, Any], support: list[dict[str, Any]]) -> None:
    """Reject prompt builders that accidentally read hidden trajectory fields."""
    altered = dict(query)
    altered.update({
        "record_id": "hidden-query-id", "trajectory_id": "hidden-trajectory-id",
        "source_path": "hidden-source", "source_sha256": "hidden-hash",
        "later_observation": ["hidden-future"], "fragment_actions": [999],
        "k": 999, "intent": {"hidden": "test-target"}, "validation": {"hidden": True},
    })
    if in_context_messages(query, support) != in_context_messages(altered, support):
        raise ValueError("ICL query prompt depends on hidden fields")
    support_without_hidden = [
        dict(row, later_observation=["hidden-support-future"], fragment_actions=[999], k=999,
             source_path="hidden-support-source", trajectory_id="hidden-support-id")
        for row in support
    ]
    if in_context_messages(query, support) != in_context_messages(query, support_without_hidden):
        raise ValueError("ICL support prompt depends on hidden trajectory fields")


def make_plan(data: Path, canonical_ids: Path, icl_ids: Path, support_data: Path, support_manifest: Path,
              model: Path, revision: str, output: Path, expected_contexts: int = 40) -> dict[str, Any]:
    """Validate frozen data and produce command arrays; never access an API or GPU."""
    paths = [data, canonical_ids, icl_ids, support_data, support_manifest]
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("Plan output must be separate from frozen inputs")
    canonical = json.loads(canonical_ids.read_text(encoding="utf-8"))
    enriched = json.loads(icl_ids.read_text(encoding="utf-8"))
    ids = canonical.get("record_ids")
    if (not isinstance(ids, list) or len(ids) != expected_contexts
            or not all(isinstance(identifier, str) for identifier in ids) or len(set(ids)) != len(ids)
            or enriched.get("record_ids") != ids):
        raise ValueError("ICL IDs must match the exact frozen order and expected count")
    if enriched.get("dataset_sha256") != file_sha256(data):
        raise ValueError("ICL ID manifest has a different dataset hash")
    test_rows = read_records(data, DataSplit.TEST)
    if len({row["record_id"] for row in test_rows}) != len(test_rows):
        raise ValueError("Duplicate test record IDs")
    queries = evaluation_records(test_rows, str(icl_ids), 0)
    support = load_support(support_data, support_manifest, read_records(support_data))
    if {row["layout_id"] for row in support} & {row["layout_id"] for row in queries}:
        raise ValueError("Support and test layouts overlap")
    for query in queries:
        validate_hidden_fields(query, support)
    metadata = json.loads((model / "experiment_revision.json").read_text(encoding="utf-8"))
    if metadata.get("revision") != revision or metadata.get("model") != "Qwen/Qwen3-0.6B":
        raise ValueError("Model revision metadata differs from the frozen plan")
    model_files = ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"]
    model_hashes = {name: file_sha256(model / name) for name in model_files}
    destination = output.resolve().parent / "icl_transfer.jsonl"
    if output.resolve() == destination:
        raise ValueError("Plan and prediction outputs must be separate")
    code_files = [ROOT / "carl/icl/intent_generator_experiment.py", ROOT / "carl/icl/intent_in_context.py",
                  Path(__file__).resolve()]
    common = [
        "uv", "run", "--offline", "--no-sync", "python", "-m", "carl.icl.intent_in_context",
        "{mode}", "--data", str(data.resolve()), "--test-ids", str(icl_ids.resolve()),
        "--support", str(support_manifest.resolve()), "--support-data", str(support_data.resolve()),
        "--output", str(destination), "--model", str(model.resolve()), "--revision", revision,
        "--dtype", "bfloat16", "--seed", "1729", "--max-length", "768", "--max-new-tokens", "128",
    ]
    return {
        "schema_version": 1,
        "status": "prepared_not_evaluated",
        "workdir": str(ROOT),
        "inputs": {str(path.resolve()): file_sha256(path) for path in paths},
        "model": {"path": str(model.resolve()), "revision": revision, "sha256": model_hashes},
        "code_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in code_files},
        "test_record_ids": ids,
        "test_layout_count": len({row["layout_id"] for row in queries}),
        "test_families": dict(Counter(str(row.get("evaluation_family", row.get("family", "unknown")))
                                       for row in queries)),
        "support_record_ids": [row["record_id"] for row in support],
        "support_layout_ids": [row["layout_id"] for row in support],
        "input_check": "Hidden query and support trajectory fields were perturbed without changing the prompts.",
        "decoding": {"mode": "greedy", "enable_thinking": False, "max_length": 768, "max_new_tokens": 128},
        "commands": {mode.replace("-", "_"): [mode if part == "{mode}" else part for part in common]
                     for mode in ("inspect-prompts", "evaluate")},
        "notes": ["Inspect prompt lengths before running evaluation; prompts are never truncated.",
                  "The test is exploratory; some x10 layouts overlap previous evaluation.",
                  "Use a supervised process group and an explicit deadline when evaluating."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", default=TRANSFER / "dataset.jsonl", type=Path)
    parser.add_argument("--canonical-ids", default=TRANSFER / "test_ids.json", type=Path)
    parser.add_argument("--icl-ids", default=TRANSFER / "icl_test_ids.json", type=Path)
    parser.add_argument("--support-data", default=ORIGINAL / "dataset.jsonl", type=Path)
    parser.add_argument("--support-manifest", default=ORIGINAL / "icl_followup/support.json", type=Path)
    parser.add_argument("--model", default=ORIGINAL / "models/Qwen3-0.6B", type=Path)
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    plan = make_plan(args.data, args.canonical_ids, args.icl_ids, args.support_data, args.support_manifest,
                     args.model, args.revision, args.output)
    content = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text(encoding="utf-8") != content:
        raise FileExistsError("Refusing to overwrite a different experiment plan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
