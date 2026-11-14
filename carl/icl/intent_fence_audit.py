"""Post-hoc audit of exactly one enclosing Markdown fence around JSON.

Remove at most one complete outer JSON or plain Markdown fence. The primary
strict result and original responses stay unchanged. This audit uses the same
schema and initial-state checks as the main assessment; it does not establish
feasibility or report solver success.
"""

import argparse
import csv
import io
import json
import random
import re
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_assessment import Method
from carl.icl.intent_assessment import atomic_write
from carl.icl.intent_assessment import board_means
from carl.icl.intent_assessment import bootstrap
from carl.icl.intent_assessment import compare_metadata
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_assessment import predictions_for
from carl.icl.intent_assessment import read_jsonl
from carl.icl.intent_assessment import recover_progress
from carl.icl.intent_assessment import test_contexts
from carl.icl.intent_data import validate_prediction
from carl.icl.intent_generator_experiment import parse_json

FENCE = re.compile(r"\A```(?:json)?\r?\n(?P<body>.*?)\r?\n```\Z", re.DOTALL)


class AuditStatus(StrEnum):
    EVALUATED = "evaluated"
    MISSING = "missing_infrastructure_output"


def remove_one_outer_fence(output: str) -> tuple[str, bool]:
    match = FENCE.fullmatch(output)
    if match is None or "```" in match.group("body"):
        return output, False
    return match.group("body"), True


def inspect_output(output: str, board: list[str]) -> dict[str, Any]:
    normalised, fence_removed = remove_one_outer_fence(output)
    try:
        parsed = parse_json(normalised)
    except ValueError:
        return {
            "fence_removed": fence_removed,
            "json_valid": False,
            "valid_nontrivial": False,
            "validation_errors": ["invalid JSON"],
            "parsed": None,
        }
    errors = validate_prediction(parsed, board)
    return {
        "fence_removed": fence_removed,
        "json_valid": True,
        "valid_nontrivial": not errors,
        "validation_errors": errors,
        "parsed": parsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    expected_hash = json.loads(args.test_ids.read_text(encoding="utf-8")).get("dataset_sha256")
    if expected_hash is not None and expected_hash != file_hash(args.data):
        raise ValueError("Frozen contexts belong to a different dataset")
    contexts = test_contexts(read_jsonl(args.data), args.test_ids)
    if not contexts:
        raise ValueError("no frozen contexts")
    predictions, metadata = {}, {}
    for method, path in ((Method.BASE, args.base), (Method.ADAPTED, args.adapted)):
        predictions[method], metadata[method] = predictions_for(path, method, contexts, args.data, args.test_ids)
    compare_metadata(metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "analysis": "posthoc_fence_only",
        "data_sha256": file_hash(args.data),
        "test_ids_sha256": file_hash(args.test_ids),
        "prediction_sha256": {
            method: file_hash(path) if path.exists() else None for method, path in (
                (Method.BASE, args.base),
                (Method.ADAPTED, args.adapted),
            )
        },
    }
    output_jsonl = args.output / "fence_audit.jsonl"
    rows = recover_progress(output_jsonl, config, contexts, args.resume, "fence_audit_config.json")
    seen = {(row["method"], row["record_id"]) for row in rows}
    with output_jsonl.open("a", encoding="utf-8", buffering=1) as stream:
        for method in (Method.BASE, Method.ADAPTED):
            for record in contexts:
                record_id = record["record_id"]
                if (method, record_id) in seen:
                    continue
                pred = predictions[method].get(record_id)
                row = {
                    "method": method,
                    "record_id": record_id,
                    "layout_id": record["layout_id"],
                }
                if pred is None:
                    row.update({
                        "status": AuditStatus.MISSING,
                        "json_valid": None,
                        "valid_nontrivial": None,
                        "fence_removed": None,
                    })
                else:
                    result = inspect_output(pred["raw_output"], record["start_observation"])
                    row.update({
                        "status":
                            AuditStatus.EVALUATED,
                        "strict_json_valid":
                            pred["json_valid"],
                        "strict_valid_nontrivial":
                            bool(pred["json_valid"]
                                 and not validate_prediction(pred.get("parsed"), record["start_observation"])),
                        "raw_output":
                            pred["raw_output"],
                        **result,
                    })
                    if (not result["fence_removed"] and result["json_valid"] != pred["json_valid"]):
                        raise ValueError(f"strict JSON parsing disagrees with generator for {record_id}")
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = []
    for method in (Method.BASE, Method.ADAPTED):
        planned = [row for row in rows if row["method"] == method]
        evaluated = [row for row in planned if row["status"] == AuditStatus.EVALUATED]
        n = len(evaluated)
        entry = {
            "method":
                method,
            "planned_contexts":
                len(planned),
            "planned_boards":
                len({row["layout_id"] for row in planned}),
            "outputs":
                n,
            "observed_boards":
                len({row["layout_id"] for row in evaluated}),
            "missing_infrastructure_outputs":
                len(planned) - n,
            "single_outer_fence_removed":
                sum(row["fence_removed"] for row in evaluated),
            "strict_valid_json":
                sum(row["strict_json_valid"] for row in evaluated),
            "strict_valid_nontrivial":
                sum(row["strict_valid_nontrivial"] for row in evaluated),
            "posthoc_valid_json":
                sum(row["json_valid"] for row in evaluated),
            "posthoc_valid_nontrivial":
                sum(row["valid_nontrivial"] for row in evaluated),
            "posthoc_valid_json_fraction":
                sum(row["json_valid"] for row in evaluated) / n if n else None,
            "posthoc_valid_nontrivial_fraction":
                sum(row["valid_nontrivial"] for row in evaluated) / n if n else None,
            "valid_nontrivial_recovered_by_fence_removal":
                sum(row["fence_removed"] and not row["strict_json_valid"] and row["valid_nontrivial"]
                    for row in evaluated),
        }
        summary.append(entry)
    by_method = {
        method: {
            row["record_id"]: row for row in rows if row["method"] == method and row["status"] == AuditStatus.EVALUATED
        } for method in (Method.BASE, Method.ADAPTED)
    }
    paired = []
    for record_id in sorted(by_method[Method.BASE].keys() & by_method[Method.ADAPTED].keys()):
        before, after = (
            by_method[Method.BASE][record_id],
            by_method[Method.ADAPTED][record_id],
        )
        paired.append({
            "layout_id": before["layout_id"],
            "delta": int(after["valid_nontrivial"]) - int(before["valid_nontrivial"]),
        })
    board_deltas = board_means(paired, lambda row: row["delta"])
    comparison = {
        "contexts": len(paired),
        "boards": len(board_deltas),
        "board_mean_valid_nontrivial_delta": sum(board_deltas) / len(board_deltas) if board_deltas else None,
        "delta_95_board_bootstrap": bootstrap(board_deltas, random.Random(20261114)),
    }
    meta = {
        "analysis":
            "post-hoc formatting audit; primary strict result is unchanged",
        "transformation":
            "remove one full outer ```json or ``` fence, only with newline-delimited body and no nested fence",
        "same_checker_as_primary":
            "carl.icl.intent_data.validate_prediction",
        "feasibility":
            "not evaluated",
        "results":
            summary,
        "paired_adapted_minus_base":
            comparison,
    }
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    atomic_write(args.output / "fence_audit_summary.csv", stream.getvalue())
    # The summary is the completion marker and must be written last.
    atomic_write(args.output / "fence_audit_summary.json", json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
