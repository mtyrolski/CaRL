#!/usr/bin/env python3
"""Run one frozen local controller on explicitly supplied Sokoban intents.

The caller supplies current-board records, not future demonstrations.  One
complete Markdown fence around an action is removed symmetrically for every
model; the unmodified answer is retained alongside the action actually parsed.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from carl.icl.intent_fence_audit import remove_one_outer_fence
from carl.icl.local_intent_policy import LocalIntentPolicy
from carl.icl.local_intent_policy import PolicyConfig
from carl.icl.local_intent_policy import config_hash
from carl.icl.local_intent_policy import run_record
from carl.icl.local_intent_policy import sha256_file


def evaluate_record(record: dict[str, Any], policy: Any, action_cap: int) -> dict[str, Any]:
    """Retain raw and normalised answers while reusing the real environment."""
    answers: list[dict[str, Any]] = []

    def decide(prompt: str) -> str:
        raw = policy(prompt)
        normalised, removed = remove_one_outer_fence(raw)
        answers.append({"raw": raw, "normalised": normalised, "fence_removed": removed})
        return normalised

    result = run_record(record, decide, action_cap)
    result["raw_policy_responses"] = answers
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--action-cap", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action_cap < 1 or args.max_new_tokens < 1:
        raise ValueError("Action and generation limits must be positive")
    if not args.model.is_dir() or args.model.resolve().name != args.revision:
        raise ValueError("Model must be a complete local directory for the stated revision")
    records = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["record_id"] for row in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError("Records must have nonempty, unique identifiers")
    config = PolicyConfig(str(args.model.resolve()), args.revision, args.device, args.max_new_tokens,
                          args.max_context_tokens, args.action_cap)
    manifest_path = args.output.with_suffix(".manifest.json")
    summary_path = args.output.with_suffix(".summary.json")
    metadata = {
        "schema_version": 1,
        "records_sha256": sha256_file(args.records),
        "weights_sha256": sha256_file(args.model / "model.safetensors"),
        "tokenizer_sha256": sha256_file(args.model / "tokenizer.json"),
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): sha256_file(path)
                          for path in (Path(__file__).resolve(),
                                       Path(__file__).resolve().parents[1] / "carl/icl/local_intent_policy.py",
                                       Path(__file__).resolve().parents[1] / "carl/icl/luna_intent_rollout.py",
                                       Path(__file__).resolve().parents[1] / "carl/icl/intent_fence_audit.py")},
        "config": asdict(config),
        "policy_config_sha256": config_hash(config),
        "action_parser": "Remove at most one complete outer plain or json Markdown fence, then require strict action JSON.",
        "record_ids": ids,
        "api_calls": 0,
        "error_interpretation": "api_error denotes local inference failure",
    }
    metadata["configuration_sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    previous: list[dict[str, Any]] = []
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != metadata:
            raise ValueError("Cannot resume with changed inputs or controller configuration")
        if args.output.exists():
            previous = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
        if [row["record_id"] for row in previous] != ids[:len(previous)]:
            raise ValueError("Partial results do not match the frozen input order")
    else:
        if args.output.exists() or manifest_path.exists():
            raise FileExistsError("Use a fresh output or explicitly resume")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, metadata)

    import torch
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no implicit CPU fallback")
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    model: Any = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype)
    model = model.to(args.device)
    policy = LocalIntentPolicy(tokenizer, model, torch, config)
    with args.output.open("a", encoding="utf-8", buffering=1) as destination:
        for record in records[len(previous):]:
            item = evaluate_record(record, policy, config.action_cap)
            destination.write(json.dumps(item, ensure_ascii=False) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            print(json.dumps({"record_id": item["record_id"], "status": item["result"]["status"],
                              "actions": len(item["result"]["actions"]), "elapsed": item["elapsed_seconds"]}),
                  flush=True)
    summary = {"status": "complete", "records": len(records), "resumed": len(previous),
               "elapsed_seconds": time.monotonic() - started,
               "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
               "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
               "configuration_sha256": metadata["configuration_sha256"], "command": sys.argv}
    atomic_json(summary_path, summary)


if __name__ == "__main__":
    main()
