"""Audit frozen stage-three inputs without loading model weights or outputs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shlex
from collections import Counter
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_in_context import in_context_messages
from carl.icl.intent_in_context import load_support

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "analysis/intent_generator_20261114T032232Z"
MODEL = HISTORICAL / "models/Qwen3-0.6B"
SUPPORT = HISTORICAL / "icl_followup/support.json"
SUPPORT_DATA = HISTORICAL / "dataset.jsonl"
GOAL = "Put every box on a goal."


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def digest(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        values = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError("Expected JSON objects")
    return values


def prompts_match_contract(prompts: list[dict[str, Any]], contract: dict[str, Any]) -> bool:
    """The generation contract hashes canonical JSON, not JSONL file bytes."""
    return digest(prompts) == contract["prompts_sha256"]


def transform_rows(board: list[str], transform: str) -> list[str]:
    """Recompute orientations independently using row/column indexing."""
    n = len(board)
    if n != 12 or any(len(row) != n for row in board):
        raise ValueError("Expected a 12x12 board")
    if transform == "identity":
        return board
    if transform == "rot90_ccw":
        return ["".join(board[col][n - 1 - row] for col in range(n)) for row in range(n)]
    if transform == "horizontal_reflection":
        return [row[::-1] for row in board]
    raise ValueError("Unknown board transform")


def inspect(plan: dict[str, Any], freeze: dict[str, Any], contract: dict[str, Any],
            prompts: list[dict[str, Any]], tokenizer: Any, support: list[dict[str, Any]],
            known_train_layouts: set[str] | None = None) -> dict[str, Any]:
    """Check the frozen grid, full prompts, exact token IDs and no truncation."""
    errors: list[str] = []
    frozen = freeze["contexts"]
    contexts, cases = plan["context_manifest"], plan["cases"]
    fields = ("context_id", "layout_id", "family", "source_family", "board_rank", "adapter_training_exposure")
    frozen_rows = [{key: row[key] for key in fields} for row in frozen]
    actual_rows = [{key: row[key] for key in fields} for row in contexts]
    if frozen_rows != actual_rows:
        errors.append("Frozen context identities, metadata or order differ")
    if len({row["context_id"] for row in contexts}) != len(contexts):
        errors.append("Context IDs are duplicated")
    if len({row["layout_id"] for row in contexts}) != len(contexts):
        errors.append("Canonical layouts are duplicated")
    source_counts = dict(Counter(row["source_family"] for row in contexts))
    family_counts = dict(Counter(row["family"] for row in contexts))
    if source_counts != freeze["source_counts"] or any(row["adapter_training_exposure"] for row in contexts):
        errors.append("Frozen source distribution or known adapter exposure differs")
    train_overlap = {row["layout_id"] for row in contexts} & (known_train_layouts or set())
    if train_overlap:
        errors.append("Evaluation includes layouts from the known adapter-training dataset")
    if digest(plan) != contract["plan_sha256"]:
        errors.append("Plan hash differs from the controller contract")
    if freeze["pairs_sha256"] not in contract["files_sha256"].values():
        errors.append("Frozen stage-one input hash is not present in the contract")
    methods, transforms = freeze["methods"], freeze["transforms"]
    expected = set(itertools.product((row["context_id"] for row in contexts), methods, transforms))
    actual = [(case["context_id"], case["method"], case["transform"]) for case in cases]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        errors.append("Planned cases are not the unique frozen context/method/transform grid")
    if any(set(row["expected_methods"]) != set(methods) or set(row["expected_transforms"]) != set(transforms)
           for row in contexts):
        errors.append("A context does not declare the complete grid")
    case_by_id = {case["case_id"]: case for case in cases}
    if len(case_by_id) != len(cases):
        errors.append("Case IDs are duplicated")
    original_by_id = {context["context_id"]: context for context in contexts}
    replay = plan.get("replay_checks", [])
    replay_keys = [(row.get("context_id"), row.get("transform")) for row in replay]
    expected_replay = set(itertools.product(original_by_id, transforms))
    if len(replay_keys) != len(expected_replay) or set(replay_keys) != expected_replay:
        errors.append("Transform replay metadata does not cover the frozen grid")
    expected_support = [{"record_id": row["record_id"], "layout_id": row["layout_id"], "intent": row["intent"]}
                        for row in support]
    if plan.get("support_intents") != expected_support:
        errors.append("ICL support differs from the frozen historical train selection")
    if {row["layout_id"] for row in support} & {row["layout_id"] for row in contexts}:
        errors.append("ICL support overlaps an evaluation layout")
    seen, lengths = set(), []
    token_mismatches = prompt_mismatches = orientation_mismatches = metadata_mismatches = 0
    for entry in prompts:
        identifier = entry.get("case_id")
        if identifier in seen or identifier not in case_by_id:
            errors.append("Duplicate or unplanned prompt")
            continue
        seen.add(identifier)
        case = case_by_id[identifier]
        context = original_by_id[case["context_id"]]
        metadata_fields = ("cohort_id", "context_id", "layout_id", "family", "source_family", "split",
                           "board_rank", "adapter_training_exposure")
        if any(case.get(field) != context.get(field) for field in metadata_fields):
            metadata_mismatches += 1
        computed_id = digest({"cohort_id": plan["cohort_id"], "context_id": case["context_id"],
                              "method": case["method"], "transform": case["transform"]})
        if (identifier != computed_id or case["original_rows"] != context["original_rows"]
                or case["source_rows"] != transform_rows(context["original_rows"], case["transform"])):
            orientation_mismatches += 1
        record = {"layout_id": case["layout_id"], "start_observation": case["source_rows"], "task_goal": GOAL}
        expected_messages = (in_context_messages(record, support) if case["method"] == "in_context"
                             else prompt_messages(record))
        expected_query = {"role": "user", "content": f"Task: {GOAL}\nCurrent board:\n" + "\n".join(case["source_rows"])}
        rendered = tokenizer.apply_chat_template(expected_messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        ids = tokenizer.apply_chat_template(expected_messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False)
        encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        lengths.append(len(ids))
        if ids != encoded:
            token_mismatches += 1
        prompt_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        if (expected_messages[-1] != expected_query or entry.get("messages") != expected_messages
                or entry.get("rendered_prompt") != rendered
                or entry.get("prompt_sha256") != prompt_hash or case.get("prompt_sha256") != prompt_hash
                or case.get("prompt_tokens") != len(ids)):
            prompt_mismatches += 1
    if len(seen) != len(cases):
        errors.append("Prompt cases do not cover the frozen plan")
    if token_mismatches:
        errors.append("Rendered prompt token IDs differ from the chat template")
    if prompt_mismatches:
        errors.append("Prompt messages, hashes or token counts differ from whitelisted inputs")
    if orientation_mismatches:
        errors.append("Case identity or D4 orientation differs")
    if metadata_mismatches:
        errors.append("Case metadata differs from the corresponding frozen context")
    max_length, max_new = contract["max_length"], contract["max_new_tokens"]
    max_tokens = max(lengths) if lengths else None
    if max_tokens is None or max_tokens + max_new > max_length:
        errors.append("A prompt and output allowance do not fit without truncation")
    return {"state": "passed" if not errors else "failed", "errors": errors,
            "frozen_contexts": len(frozen), "planned_contexts": len(contexts),
            "frozen_order_matches": frozen_rows == actual_rows, "source_counts": source_counts,
            "family_counts": family_counts, "known_adapter_training_layouts": len(train_overlap),
            "historical_train_registry_checked": known_train_layouts is not None,
            "planned_cases": len(cases), "unique_case_ids": len(case_by_id),
            "grid_size": len(expected), "prompt_count": len(prompts), "unique_prompt_cases": len(seen),
            "token_id_mismatches": token_mismatches, "prompt_or_hash_mismatches": prompt_mismatches,
            "orientation_or_case_id_mismatches": orientation_mismatches,
            "case_metadata_mismatches": metadata_mismatches,
            "prompt_tokens_min": min(lengths) if lengths else None, "prompt_tokens_max": max_tokens,
            "max_new_tokens": max_new, "max_length": max_length,
            "model_weights_loaded": False, "evaluation_outputs_read": False, "network_used": False}


def audit(run: Path, freeze_path: Path) -> dict[str, Any]:
    """Read only the plan, prompts, contract, frozen context list and support."""
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1")
    from transformers import AutoTokenizer

    paths = {"plan": run / "plan.json", "contract": run / "contract.json", "prompts": run / "prompts.jsonl",
             "freeze": freeze_path, "support_manifest": SUPPORT, "support_data": SUPPORT_DATA,
             "tokenizer": MODEL / "tokenizer.json", "tokenizer_config": MODEL / "tokenizer_config.json",
             "auditor": Path(__file__)}
    plan = json.loads(paths["plan"].read_text())
    freeze = json.loads(paths["freeze"].read_text())
    contract = json.loads(paths["contract"].read_text())
    prompts = rows(paths["prompts"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    historical_records = read_records(SUPPORT_DATA)
    support = load_support(SUPPORT_DATA, SUPPORT, historical_records)
    train_layouts = {row["layout_id"] for row in historical_records if row.get("split") == "train"}
    result = inspect(plan, freeze, contract, prompts, tokenizer, support, train_layouts)
    result["source_sha256"] = {name: sha(path) for name, path in paths.items()}
    result["canonical_prompts_sha256"] = digest(prompts)
    if not prompts_match_contract(prompts, contract):
        result["errors"].append("Canonical prompts hash differs from the generation contract")
        result["state"] = "failed"
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="rerun the audit without replacing the saved record")
    args = parser.parse_args()
    run, freeze, output = args.run.resolve(), args.freeze.resolve(), args.output.resolve()
    result = audit(run, freeze)
    arguments = ["-m", "scripts.audit_overnight_stage03_input", "--run", str(run), "--freeze", str(freeze),
                 "--output", str(output), "--verify"]
    result["reproduce_command"] = shlex.join(["uv", "run", "--offline", "--no-sync", "python", *arguments])
    result["reproduce_fallback_command"] = shlex.join([str(ROOT / ".venv/bin/python"), *arguments])
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        if args.verify:
            previous = json.loads(output.read_text())
            if {k: v for k, v in previous.items() if k != "created_utc"} != {
                    k: v for k, v in result.items() if k != "created_utc"}:
                raise ValueError("Reproduced audit differs from the saved record")
        else:
            if output.exists():
                raise FileExistsError("Audit result already exists; use --verify")
            temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("w", encoding="utf-8") as target:
                    json.dump(result, target, indent=2, sort_keys=True, allow_nan=False)
                    target.write("\n")
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
    print(json.dumps({"state": result["state"], "frozen_contexts": result["frozen_contexts"],
                      "planned_cases": result["planned_cases"], "errors": result["errors"]}))
    if result["state"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
