"""Evaluate generated Sokoban intents using a bounded, replay-checked solver."""

import argparse
import csv
import json
import os
from pathlib import Path

from carl.icl.intent_data import SokobanPredicate
from carl.icl.intent_data import predicate_value
from carl.icl.intent_data import validate_prediction
from carl.icl.intents.assessment.io import AssessmentStatus
from carl.icl.intents.assessment.io import Method
from carl.icl.intents.assessment.io import Record
from carl.icl.intents.assessment.io import _reject_non_json
from carl.icl.intents.assessment.io import atomic_write
from carl.icl.intents.assessment.io import board_means
from carl.icl.intents.assessment.io import bootstrap
from carl.icl.intents.assessment.io import check_raw_prediction
from carl.icl.intents.assessment.io import compare_metadata
from carl.icl.intents.assessment.io import file_hash
from carl.icl.intents.assessment.io import missing_comparison_metadata
from carl.icl.intents.assessment.io import predictions_for
from carl.icl.intents.assessment.io import read_jsonl
from carl.icl.intents.assessment.io import recover_progress
from carl.icl.intents.assessment.io import test_contexts
from carl.icl.intents.assessment.reporting import has_witness
from carl.icl.intents.assessment.reporting import paired_comparison
from carl.icl.intents.assessment.reporting import summarise
from carl.icl.intents.assessment.reporting import write_examples
from carl.icl.intents.assessment.search import DIRECTIONS
from carl.icl.intents.assessment.search import Coordinate
from carl.icl.intents.assessment.search import Positions
from carl.icl.intents.assessment.search import SearchStatus
from carl.icl.intents.assessment.search import State
from carl.icl.intents.assessment.search import board_parts
from carl.icl.intents.assessment.search import bounded_witness
from carl.icl.intents.assessment.search import goal_satisfied
from carl.icl.intents.assessment.search import local_rule
from carl.icl.intents.assessment.search import local_rule_with_stats
from carl.icl.intents.assessment.search import real_replay
from carl.icl.intents.assessment.search import step

__all__ = [
    "AssessmentStatus",
    "Coordinate",
    "DIRECTIONS",
    "Method",
    "Positions",
    "Record",
    "SearchStatus",
    "SokobanPredicate",
    "State",
    "_reject_non_json",
    "assess_context",
    "atomic_write",
    "board_means",
    "board_parts",
    "bootstrap",
    "bounded_witness",
    "check_raw_prediction",
    "compare_metadata",
    "file_hash",
    "goal_satisfied",
    "has_witness",
    "local_rule",
    "local_rule_with_stats",
    "main",
    "missing_comparison_metadata",
    "paired_comparison",
    "predicate_value",
    "predictions_for",
    "read_jsonl",
    "real_replay",
    "recover_progress",
    "step",
    "summarise",
    "test_contexts",
    "validate_prediction",
    "write_examples",
]


# Preserve the positional assessment API for existing experiment scripts.
# pylint: disable-next=too-many-positional-arguments
def assess_context(
    record: Record,
    method: Method,
    predictions: dict[Method, dict[str, Record]],
    max_depth: int,
    max_nodes: int,
    seconds: float,
) -> Record:
    prediction: Record | None
    if method == Method.LOCAL_RULE:
        local_intent = local_rule(record["start_observation"])
        prediction = {
            "parsed": local_intent,
            "raw_output": json.dumps(local_intent),
            "json_valid": True,
        }
    else:
        prediction = predictions[method].get(record["record_id"])
    row = {
        "record_id": record["record_id"],
        "layout_id": record["layout_id"],
        "method": method,
    }
    if prediction is None:
        row.update(
            {
                "status": AssessmentStatus.MISSING,
                "json_valid": None,
                "valid_nontrivial": None,
                "solver": None,
            }
        )
        return row
    intent = prediction.get("parsed")
    errors = (
        validate_prediction(intent, record["start_observation"])
        if prediction["json_valid"]
        else ["invalid JSON"]
    )
    valid = not errors
    solver = None
    if valid:
        if not isinstance(intent, dict):
            raise ValueError("A valid intent must be an object")
        solver = bounded_witness(
            record["start_observation"], intent, max_depth, max_nodes, seconds
        )
    effect_value = intent.get("effects") if isinstance(intent, dict) else None
    effects = effect_value if isinstance(effect_value, list) else []
    row.update(
        {
            "status": AssessmentStatus.EVALUATED,
            "json_valid": prediction["json_valid"],
            "valid_nontrivial": valid,
            "validation_errors": errors,
            "intent": intent,
            "raw_output": prediction.get("raw_output"),
            "effect_types": [
                effect.get("predicate")
                for effect in effects
                if isinstance(effect, dict)
            ],
            "conjunction_size": len(effects),
            "solver": solver,
            "generation_seconds": prediction.get("generation_seconds"),
            "start_observation": record["start_observation"],
            "demonstration_later_observation": record["later_observation"],
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "assess"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--adapted", type=Path)
    parser.add_argument("--in-context", type=Path)
    parser.add_argument("--test-ids", type=Path)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_depth < 0 or args.max_nodes < 1 or args.seconds <= 0:
        parser.error("search depth must be nonnegative and node/time limits positive")
    if args.test_ids:
        expected_hash = json.loads(args.test_ids.read_text(encoding="utf-8")).get(
            "dataset_sha256"
        )
        if expected_hash is not None and expected_hash != file_hash(args.data):
            raise ValueError("Frozen contexts belong to a different dataset")
    contexts = test_contexts(read_jsonl(args.data), args.test_ids)
    if not contexts:
        raise ValueError("No held-out contexts")
    if args.mode == "freeze":
        result = {
            "selection": "all preselected checked test records, ordered by frozen dataset",
            "dataset_sha256": file_hash(args.data),
            "record_ids": [r["record_id"] for r in contexts],
            "layout_ids": sorted({r["layout_id"] for r in contexts}),
            "count": len(contexts),
            "independent_boards": len({r["layout_id"] for r in contexts}),
        }
        atomic_write(args.output, json.dumps(result, indent=2) + "\n")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    predictions: dict[Method, dict[str, Record]] = {}
    metadata: dict[Method, Record | None] = {}
    inputs = [(Method.BASE, args.base), (Method.ADAPTED, args.adapted)]
    if args.in_context is not None:
        inputs.append((Method.IN_CONTEXT, args.in_context))
    for method, path in inputs:
        predictions[method], metadata[method] = predictions_for(
            path, method, contexts, args.data, args.test_ids
        )
    compare_metadata(metadata)
    config = {
        "schema_version": 1,
        "data_sha256": file_hash(args.data),
        "test_ids_sha256": file_hash(args.test_ids) if args.test_ids else None,
        "prediction_sha256": {
            method: file_hash(path) if path and path.exists() else None
            for method, path in inputs
        },
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "seconds": args.seconds,
    }
    output_jsonl = args.output / "assessment.jsonl"
    results = recover_progress(output_jsonl, config, contexts, args.resume)
    seen = {(row["method"], row["record_id"]) for row in results}
    with output_jsonl.open("a", encoding="utf-8", buffering=1) as stream:
        methods = [method for method, _ in inputs] + [Method.LOCAL_RULE]
        for method in methods:
            for record in contexts:
                if (method, record["record_id"]) in seen:
                    continue
                row = assess_context(
                    record,
                    method,
                    predictions,
                    args.max_depth,
                    args.max_nodes,
                    args.seconds,
                )
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            "method": method,
                            "record_id": record["record_id"],
                            "valid": row["valid_nontrivial"],
                            "solver": row["solver"]["status"]
                            if row["solver"]
                            else None,
                        }
                    ),
                    flush=True,
                )
    summary, comparison = summarise(results)
    result = {
        "results": summary,
        "paired_adapted_minus_base": comparison,
        "prediction_metadata": metadata,
        "solver": {
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "seconds": args.seconds,
            "meaning": "bounded solver witness; not controller reachability",
        },
    }
    if args.in_context is not None:
        result["paired_in_context_minus_base"] = paired_comparison(
            results, Method.BASE, Method.IN_CONTEXT
        )
        result["paired_in_context_minus_adapted"] = paired_comparison(
            results, Method.ADAPTED, Method.IN_CONTEXT
        )
        result["comparison_metadata_missing"] = missing_comparison_metadata(metadata)
    write_examples(results, args.output)
    columns = [
        "method",
        "planned_contexts",
        "planned_boards",
        "outputs",
        "boards_with_outputs",
        "missing_infrastructure_outputs",
        "valid_json",
        "valid_nontrivial",
        "witness_found",
        "unresolved_among_valid",
        "valid_json_fraction",
        "valid_fraction",
        "witness_fraction_all_outputs",
        "witness_fraction_valid",
        "board_mean_valid_fraction",
        "board_mean_witness_fraction",
        "solver_expanded_nodes",
        "solver_seconds",
    ]
    output_csv = args.output / "summary.csv"
    temporary = output_csv.with_name(f".{output_csv.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    os.replace(temporary, output_csv)
    # The summary is the completion marker and must be written last.
    atomic_write(args.output / "summary.json", json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
