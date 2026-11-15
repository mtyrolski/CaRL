#!/usr/bin/env python3
"""Freeze the stage-three intervention recommendation from development only.

The rule was specified before examining fresh outputs.  Each original layout
has one vote; transformed pairs on the same layout are averaged first.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.policy_relative_lora import assess_output
from scripts.overnight_stage03_generate import METHODS
from scripts.overnight_stage03_generate import TRANSFORMS
from scripts.overnight_stage03_generate import case_id
from scripts.overnight_stage03_generate import transform_rows
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl

THRESHOLD = 0.5
MIN_BOARDS = 4
COORDINATE_PREDICATES = {"box_at", "player_at"}
ROOT = Path(__file__).resolve().parents[1]


def canonical_intent(intent: dict[str, Any]) -> str:
    """Canonicalise conjunctions without treating instruction prose as structure."""
    def ordered(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))

    return json.dumps({"effects": ordered(intent["effects"]), "preserve": ordered(intent["preserve"])},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def literal_coordinates(intent: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    """Keep the literal predicate-array order; do not transform coordinates."""
    return tuple((predicate["row"], predicate["col"]) for group in ("effects", "preserve")
                 for predicate in intent[group] if predicate["predicate"] in COORDINATE_PREDICATES)


def analyse(plan: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen threshold to valid development predictions per board."""
    cases = plan.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("A nonempty frozen development plan is required")
    planned: dict[str, dict[str, Any]] = {}
    by_context: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for case in cases:
        identifier, context = case["case_id"], case["context_id"]
        if identifier in planned or case.get("split") != "development":
            raise ValueError("Duplicate case or non-development case in decision input")
        if case.get("cohort_id") != plan.get("cohort_id"):
            raise ValueError("Case belongs to another cohort")
        if identifier != case_id(plan["cohort_id"], context, case["method"], case["transform"]):
            raise ValueError("Case identity hash differs from its frozen coordinates")
        if (case.get("method") not in METHODS or case.get("transform") not in TRANSFORMS
                or transform_rows(case["original_rows"], case["transform"]) != case["source_rows"]):
            raise ValueError("Case method, transform, or transformed board is invalid")
        metadata = {key: case[key] for key in ("layout_id", "family", "source_family", "original_rows",
                                                "adapter_training_exposure")}
        if context not in by_context and canonical_layout_id(from_rows(case["original_rows"])) != case["layout_id"]:
            raise ValueError("Original board does not match its canonical layout ID")
        if by_context.setdefault(context, metadata) != metadata:
            raise ValueError("Inconsistent board identity within a context")
        planned[identifier] = case
        by_key[(context, case["method"], case["transform"])] = case
    expected = {(context, str(method), str(transform)) for context in by_context
                for method in METHODS for transform in TRANSFORMS}
    actual = {(case["context_id"], case["method"], case["transform"]) for case in cases}
    if actual != expected or len(actual) != len(cases):
        raise ValueError("Development plan is not a complete method/transform grid")
    seen: set[str] = set()
    valid: dict[tuple[str, str, str], dict[str, Any]] = {}
    observed: dict[tuple[str, str, str], str] = {}
    for record in records:
        identifier = record.get("case_id")
        if identifier in seen or identifier not in planned:
            raise ValueError("Duplicate or unplanned prediction")
        seen.add(identifier)
        case = planned[identifier]
        if any(record.get(key) != value for key, value in case.items()):
            raise ValueError("Prediction differs from the frozen case")
        key = (case["context_id"], case["method"], case["transform"])
        status = record.get("prediction_status")
        if status == "infra_error":
            observed[key] = "infra_error"
            continue
        if status != "complete" or not isinstance(record.get("raw_output"), str):
            raise ValueError("Malformed development prediction")
        assessed = assess_output(record["raw_output"], case["source_rows"], normalize=True)
        strict = assess_output(record["raw_output"], case["source_rows"], normalize=False)
        semantic = bool(assessed["json_valid"] and not assessed["validation_errors"])
        if (record.get("json_valid") != assessed["json_valid"] or record.get("valid_nontrivial") != semantic
                or record.get("strict_json_valid") != strict["json_valid"]
                or record.get("parsed") != assessed["parsed"]):
            raise ValueError("Stored parser metadata disagrees with the frozen parser")
        observed[key] = "valid" if semantic else "invalid"
        if semantic:
            valid[key] = assessed["parsed"]
    # Select one context per canonical layout before consulting prediction outcomes.
    context_by_layout: dict[str, str] = {}
    for context, row in sorted(by_context.items()):
        context_by_layout.setdefault(row["layout_id"], context)
    selected = sorted(context_by_layout.values())
    metrics = {}
    for method in METHODS:
        identity_keys = [(context, str(method), "identity") for context in selected]
        identities = [valid[key] for key in identity_keys if key in valid]
        signatures = Counter(canonical_intent(intent) for intent in identities)
        frequency = max(signatures.values(), default=0)
        leading = min((signature for signature, n in signatures.items() if n == frequency), default=None)
        boards = []
        pairs = copied_pairs = skipped_symmetric = 0
        for context in selected:
            key = (context, str(method), "identity")
            if key not in valid:
                continue
            literal = literal_coordinates(valid[key])
            matches = []
            identity_case = by_key[key]
            for transform in ("rot90_ccw", "horizontal_reflection"):
                transformed_key = (context, str(method), transform)
                transformed_case = by_key[transformed_key]
                if identity_case["source_rows"] == transformed_case["source_rows"]:
                    skipped_symmetric += 1
                    continue
                if transformed_key not in valid:
                    continue
                transformed = literal_coordinates(valid[transformed_key])
                if not literal and not transformed:
                    continue
                same = literal == transformed
                pairs += 1
                copied_pairs += int(same)
                matches.append(int(same))
            if matches:
                boards.append(sum(matches) / len(matches))
        identity_n, coordinate_n = len(identities), len(boards)
        mode_share = frequency / identity_n if identity_n else None
        copy_share = sum(boards) / coordinate_n if coordinate_n else None
        triggers = ((identity_n >= MIN_BOARDS and mode_share is not None and mode_share >= THRESHOLD)
                    or (coordinate_n >= MIN_BOARDS and copy_share is not None and copy_share >= THRESHOLD))
        enough = identity_n >= MIN_BOARDS or coordinate_n >= MIN_BOARDS
        metrics[str(method)] = {
            "planned_layouts": len(selected), "identity_outputs_observed": sum(key in observed for key in identity_keys),
            "invalid_identity_outputs": sum(observed.get(key) == "invalid" for key in identity_keys),
            "identity_infrastructure_errors": sum(observed.get(key) == "infra_error" for key in identity_keys),
            "identity_outputs_missing": sum(key not in observed for key in identity_keys),
            "valid_identity_layouts": identity_n, "most_common_canonical_intent": json.loads(leading) if leading else None,
            "most_common_intent_layouts": frequency, "most_common_intent_share": mode_share,
            "eligible_coordinate_layouts": coordinate_n, "eligible_transform_pairs": pairs,
            "literal_coordinate_copy_pairs": copied_pairs, "literal_coordinate_copy_share_equal_layout": copy_share,
            "identical_transformed_boards_skipped": skipped_symmetric, "at_least_four_layouts": enough,
            "threshold_reached": triggers,
            "recommendation": ("train_only_d4_augmentation_and_coordinate_balancing" if triggers else
                               "additional_independent_train_layouts_with_balanced_predicates" if enough else
                               "insufficient_development_evidence"),
        }
    triggered = any(row["threshold_reached"] for row in metrics.values())
    sufficient = any(row["at_least_four_layouts"] for row in metrics.values())
    result = ("train_only_d4_augmentation_and_coordinate_balancing" if triggered else
              "additional_independent_train_layouts_with_balanced_predicates" if sufficient else
              "insufficient_development_evidence")
    return {"methods": metrics, "recommendation": result, "provisional": not sufficient,
            "planned_cases": len(cases), "observed_cases": len(records), "unobserved_cases": len(cases) - len(seen),
            "distinct_layouts": len(selected), "deduplication": "first context ID per original canonical layout",
            "known_adapter_training_layouts": sum(by_context[context]["adapter_training_exposure"] is True
                                                   for context in selected),
            "coordinate_aggregation": "mean over eligible transforms within each layout, then equal-layout mean",
            "identity_selection": "semantic identity ignores instruction and sorts conjunction predicates",
            "threshold": THRESHOLD, "minimum_distinct_layouts": MIN_BOARDS}


def freeze_decision(run: Path, frozen_plan: Path, output: Path) -> dict[str, Any]:
    """Check that evaluation has not produced output, then write once atomically."""
    evaluation_output = run.parent / "fresh_evaluation/predictions.jsonl"
    if evaluation_output.exists():
        raise ValueError("Fresh evaluation output already exists; development decision was not frozen in time")
    paths = {"plan": run / "plan.json", "contract": run / "contract.json", "predictions": run / "predictions.jsonl",
             "status": run / "status.json", "stage_plan": frozen_plan}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Development run is missing a frozen artifact")
    status = json.loads(paths["status"].read_text())
    if status.get("state") not in {"complete", "partial"}:
        raise ValueError("Development run has not reached a terminal saved status")
    contract = json.loads(paths["contract"].read_text())
    plan = json.loads(paths["plan"].read_text())
    protocol = json.loads(frozen_plan.read_text())
    if (contract.get("split") != "development" or contract.get("plan_sha256") != digest(plan)
            or set(protocol.get("methods", [])) != set(METHODS)
            or set(protocol.get("transformations", [])) != set(TRANSFORMS)):
        raise ValueError("Development contract or prescribed method/transform grid differs")
    rule = protocol.get("fresh_diagnostic_intervention_rule", "")
    if not isinstance(rule, str) or any(fragment not in rule for fragment in (
        "first averaging over valid transforms within each layout", "at least 0.5", "at least four observations",
        "Do not use evaluation outputs")):
        raise ValueError("Frozen development decision rule differs from the implemented rule")
    contract_sha = digest(contract)
    records = read_jsonl(paths["predictions"])
    if any(row.get("contract_sha256") != contract_sha for row in records):
        raise ValueError("Development prediction contract hash differs")
    decision = analyse(plan, records)
    decision.update({"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                     "created_monotonic": time.monotonic(), "cohort_id": plan["cohort_id"],
                     "stage_plan_path": str(frozen_plan.resolve()), "development_run": str(run.resolve()),
                     "source_sha256": {name: file_sha(path) for name, path in paths.items()},
                     "decision_code_sha256": file_sha(Path(__file__)),
                     "test_results_accessed": False, "evaluation_guard_path": str(evaluation_output.resolve())})
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        if output.exists():
            raise FileExistsError("A frozen development decision already exists")
        if evaluation_output.exists():
            raise ValueError("Fresh evaluation output appeared before the development decision was saved")
        atomic_json(output, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--frozen-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision = freeze_decision(args.run.resolve(), args.frozen_plan.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output), "recommendation": decision["recommendation"],
                      "provisional": decision["provisional"]}))


if __name__ == "__main__":
    main()
