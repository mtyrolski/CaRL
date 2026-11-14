"""Post-hoc bounded feasibility after removing one complete outer fence.

All generators use the same normalisation, frozen contexts and search budget.
The primary strict-JSON comparison is unchanged. A found path is a bounded
solver witness, not an action sequence executed by a learned controller.
"""
import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

from carl.icl.intent_assessment import Method
from carl.icl.intent_assessment import assess_context
from carl.icl.intent_assessment import atomic_write
from carl.icl.intent_assessment import compare_metadata
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_assessment import missing_comparison_metadata
from carl.icl.intent_assessment import paired_comparison
from carl.icl.intent_assessment import predictions_for
from carl.icl.intent_assessment import read_jsonl
from carl.icl.intent_assessment import recover_progress
from carl.icl.intent_assessment import summarise
from carl.icl.intent_assessment import test_contexts
from carl.icl.intent_assessment import write_examples
from carl.icl.intent_fence_audit import inspect_output

Record = dict[str, Any]


def normalise_prediction(prediction: Record, board: list[str]) -> tuple[Record, Record]:
    """Apply exactly the same one-fence rule used by the format audit."""
    audit = inspect_output(prediction["raw_output"], board)
    normalised = {
        "raw_output": prediction["raw_output"],
        "json_valid": audit["json_valid"],
        "parsed": audit["parsed"],
        "generation_seconds": prediction.get("generation_seconds"),
    }
    provenance = {
        "strict_json_valid": prediction["json_valid"],
        "fence_removed": audit["fence_removed"],
    }
    return normalised, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--in-context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_depth < 0 or args.max_nodes < 1 or args.seconds <= 0:
        parser.error("search depth must be nonnegative and node/time limits positive")
    expected_hash = json.loads(args.test_ids.read_text(encoding="utf-8")).get("dataset_sha256")
    if expected_hash is not None and expected_hash != file_hash(args.data):
        raise ValueError("Frozen contexts belong to a different dataset")
    contexts = test_contexts(read_jsonl(args.data), args.test_ids)
    if not contexts:
        raise ValueError("No held-out contexts")

    predictions: dict[Method, dict[str, Record]] = {}
    metadata: dict[Method, Record | None] = {}
    inputs = [(Method.BASE, args.base), (Method.ADAPTED, args.adapted)]
    if args.in_context is not None:
        inputs.append((Method.IN_CONTEXT, args.in_context))
    for method, path in inputs:
        predictions[method], metadata[method] = predictions_for(path, method, contexts, args.data, args.test_ids)
    compare_metadata(metadata)
    normalised: dict[Method, dict[str, Record]] = {method: {} for method, _ in inputs}
    provenance: dict[tuple[Method, str], Record] = {}
    for method, _ in inputs:
        for record in contexts:
            record_id = record["record_id"]
            prediction = predictions[method].get(record_id)
            if prediction is not None:
                normalised[method][record_id], provenance[(method, record_id)] = normalise_prediction(
                    prediction, record["start_observation"])

    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "analysis": "posthoc_one_outer_fence_bounded_feasibility",
        "data_sha256": file_hash(args.data),
        "test_ids_sha256": file_hash(args.test_ids),
        "prediction_sha256": {method: file_hash(path) if path.exists() else None for method, path in inputs},
        "max_depth": args.max_depth,
        "max_nodes": args.max_nodes,
        "seconds": args.seconds,
    }
    output_jsonl = args.output / "feasibility.jsonl"
    results = recover_progress(output_jsonl, config, contexts, args.resume, "feasibility_config.json")
    seen = {(row["method"], row["record_id"]) for row in results}
    with output_jsonl.open("a", encoding="utf-8", buffering=1) as stream:
        for method, _ in inputs:
            for record in contexts:
                key = method, record["record_id"]
                if key in seen:
                    continue
                row = assess_context(record, method, normalised, args.max_depth, args.max_nodes, args.seconds)
                row.update(provenance.get(key, {"strict_json_valid": None, "fence_removed": None}))
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps({
                    "method": method,
                    "record_id": record["record_id"],
                    "valid": row["valid_nontrivial"],
                    "solver": row["solver"]["status"] if row["solver"] else None
                }),
                      flush=True)
    summary, comparison = summarise(results)
    summary = [row for row in summary if row["method"] != Method.LOCAL_RULE]
    for row in summary:
        selected = [
            result for result in results
            if result["method"] == row["method"] and result["strict_json_valid"] is not None
        ]
        row["strict_valid_json"] = sum(result["strict_json_valid"] for result in selected)
        row["single_outer_fence_removed"] = sum(result["fence_removed"] for result in selected)
    result = {
        "analysis": "post-hoc: remove at most one full outer Markdown fence, symmetrically for all models",
        "results": summary,
        "paired_adapted_minus_base": comparison,
        "prediction_metadata": metadata,
        "solver": {
            "max_depth": args.max_depth,
            "max_nodes": args.max_nodes,
            "seconds": args.seconds,
            "meaning": "bounded solver witness replayed in the environment; not controller reachability"
        },
    }
    if args.in_context is not None:
        result["paired_in_context_minus_base"] = paired_comparison(results, Method.BASE, Method.IN_CONTEXT)
        result["paired_in_context_minus_adapted"] = paired_comparison(results, Method.ADAPTED, Method.IN_CONTEXT)
        result["comparison_metadata_missing"] = missing_comparison_metadata(metadata)
    write_examples(results, args.output)
    columns = [
        "method", "planned_contexts", "planned_boards", "outputs", "boards_with_outputs",
        "missing_infrastructure_outputs", "strict_valid_json", "single_outer_fence_removed", "valid_json",
        "valid_nontrivial", "witness_found", "unresolved_among_valid", "valid_json_fraction", "valid_fraction",
        "witness_fraction_all_outputs", "witness_fraction_valid", "board_mean_valid_fraction",
        "board_mean_witness_fraction", "solver_expanded_nodes", "solver_seconds"
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(summary)
    atomic_write(args.output / "summary.csv", stream.getvalue())
    # Written last, after all other outputs, as the stage completion marker.
    atomic_write(args.output / "summary.json", json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
