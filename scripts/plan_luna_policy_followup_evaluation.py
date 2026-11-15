#!/usr/bin/env python3
"""Freeze the follow-up arm/slot matrix before generating test outputs.

Only checked demonstrations, completed calibration and frozen test *inputs*
are read. The prediction paths in the resulting matrix must not yet exist.
No model or API call is made.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from decimal import ROUND_HALF_UP
from decimal import Decimal
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import read_records
from carl.icl.policy_relative_lora import Arm
from carl.icl.policy_relative_lora import FitData
from carl.icl.policy_relative_lora import load_demo_data
from carl.icl.policy_relative_lora import load_study_data
from carl.icl.policy_relative_lora import require_scale_information
from carl.icl.policy_relative_lora import test_contexts

NO_MEASURED_SUCCESS = "train and validation each need at least one complete context with a measured success"
TIERS = (0.25, 0.75)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def absolute_levels(data: FitData) -> tuple[list[float], dict[str, Any]]:
    """Use nearest-rank quartiles, rounded half-up to 0.1 primitive action.

    If the quartiles tie, use the observed minimum and maximum. These values
    come exclusively from successful train/validation calibration attempts.
    """
    values = sorted(candidate.mean_success_actions for context in (*data.train, *data.validation)
                    for candidate in context.candidates if candidate.mean_success_actions is not None)
    if not values:
        raise ValueError("No successful train/validation durations")

    def rounded(value: float) -> float:
        return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

    indices = [max(0, math.ceil(fraction * len(values)) - 1) for fraction in TIERS]
    levels = [rounded(values[index]) for index in indices]
    method = "nearest-rank quartiles; round half up to 0.1 action"
    if levels[0] == levels[1]:
        levels = [rounded(values[0]), rounded(values[-1])]
        method += "; tied quartiles fall back to observed minimum and maximum"
    if levels[0] <= 0 or levels[0] == levels[1]:
        raise ValueError("Cannot choose two distinct positive absolute-duration levels")
    return levels, {"method": method, "successful_candidates": len(values), "minimum": values[0],
                    "maximum": values[-1], "absolute_levels": levels}


def arm_decision(data: FitData | None, minimum_train: int, minimum_validation: int,
                 include_rule: bool, include_base: bool = False) -> tuple[list[str], list[float] | None, dict[str, Any]]:
    """Use D only, D/N, or D/N/T/U according to measured scale evidence."""
    arms = ["B", "D"] if include_base else ["D"]
    details: dict[str, Any] = {"measured_available": data is not None, "scale_information": False,
                               "reason": "no complete successful train and validation contexts"}
    levels = None
    if data is not None:
        arms.append("N")
        details["reason"] = "insufficient distinct successful durations for a scale comparison"
        try:
            require_scale_information(data, Arm.RELATIVE, minimum_train, minimum_validation)
            levels, duration_selection = absolute_levels(data)
        except ValueError as error:
            details["scale_diagnostic"] = str(error)
        else:
            arms.extend(("T", "U"))
            details.update(scale_information=True, reason="measured scale thresholds satisfied",
                           duration_selection=duration_selection)
    if include_rule:
        arms.append("R")
    details["arms"] = arms
    details["matrix_compatible"] = len(arms) >= 2
    return arms, levels, details


def planned_matrix(arms: list[str], levels: list[float] | None, predictions: Path) -> dict[str, Any]:
    """Name separate physical slots without reading prediction files."""
    slots = {}
    for arm in arms:
        if arm == "T" and (levels is None or len(levels) != 2):
            raise ValueError("T requires two frozen absolute-duration levels")
        paths = (predictions / "local_rule" / f"R_slot{i}.jsonl" if arm == "R"
                 else predictions / f"{arm}_slot{i}.jsonl" for i in range(2))
        inputs = list(TIERS) if arm == "U" else levels if arm == "T" else [None, None]
        assert inputs is not None
        slots[arm] = [{"path": str(path.resolve()), "requested_input": value}
                      for path, value in zip(paths, inputs, strict=True)]
    return {"schema_version": 1, "tiers": list(TIERS), "arms": slots}


def _check_no_test_outputs(predictions: Path) -> None:
    """Do not decide arms after any of the contemplated test files exist."""
    for arm in ("B", "D", "N", "T", "U", "R"):
        for index in range(2):
            path = (predictions / "local_rule" / f"R_slot{index}.jsonl" if arm == "R"
                    else predictions / f"{arm}_slot{index}.jsonl")
            if path.exists():
                raise FileExistsError(f"Test generation has already started: {path}")


def calibration_inventory(path: Path) -> dict[str, Any] | None:
    """Count all saved records, including incomplete contexts, without relabelling them."""
    if not path.is_file():
        return None
    rows = read_records(path)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        context = row.get("context_id")
        if not isinstance(context, str) or not context:
            raise ValueError("Calibration record has no context_id")
        groups.setdefault(context, []).append(row)
    incomplete = []
    for context, group in sorted(groups.items()):
        measured = [row.get("policy_measurements") for row in group]
        if (any(not isinstance(value, dict) or value.get("complete") is not True for value in measured)
                or any(isinstance(value, dict) and value.get("expected_candidates") != len(group)
                       for value in measured)):
            incomplete.append({"context_id": context, "split": group[0].get("split"),
                               "record_ids": [row.get("record_id") for row in group]})
    return {"records": len(rows), "contexts": len(groups), "incomplete_contexts": incomplete,
            "complete_record_count": sum(isinstance(row.get("policy_measurements"), dict)
                                         and row["policy_measurements"].get("complete") is True for row in rows)}


def freeze(calibration: Path, status: Path, demo_path: Path, contexts_path: Path, predictions: Path, output: Path,
           minimum_train: int, minimum_validation: int, expected_boards: int, include_rule: bool,
           allow_partial: bool = False, include_base: bool = False) -> dict[str, Any]:
    """Persist a decision and compatible matrix without opening test outputs."""
    output, predictions = output.resolve(), predictions.resolve()
    if output.exists():
        raise FileExistsError(f"Will not replace a frozen decision: {output}")
    _check_no_test_outputs(predictions)
    has_calibration, has_status = calibration.is_file(), status.is_file()
    state = json.loads(status.read_text(encoding="utf-8")) if has_status else None
    if state is not None and not isinstance(state, dict):
        raise ValueError("Calibration status must be an object")
    status_name = state.get("state") if state is not None else None
    terminal = {"complete"} | ({"interrupted", "stopped_api_error"} if allow_partial else set())
    if state is not None and status_name not in terminal:
        raise ValueError("Calibration must be terminal and complete unless --allow-partial is declared")
    if state is not None and status_name == "complete":
        planned = state.get("planned_attempts")
        if (isinstance(planned, bool) or not isinstance(planned, int) or planned < 1
                or state.get("saved_attempts") != planned or state.get("not_started") != 0
                or state.get("started_without_saved_result") != 0):
            raise ValueError("Calibration claims completion but attempt counts disagree")
    demonstration = load_demo_data(demo_path)
    contexts = test_contexts(contexts_path)
    layouts = {row["layout_id"] for row in contexts}
    if len(contexts) != expected_boards or len(layouts) != expected_boards:
        raise ValueError(f"Expected {expected_boards} distinct held-out boards")
    training_layouts = (set(demonstration.audit["layout_ids_by_split"]["train"])
                        | set(demonstration.audit["layout_ids_by_split"]["validation"]))
    if layouts & training_layouts:
        raise ValueError("Held-out layouts overlap demonstration train/validation")
    if not has_calibration or not has_status:
        data = None
        no_success_diagnostic = "calibration data or its terminal status is missing"
    else:
        try:
            data, _ = load_study_data(argparse.Namespace(arm=Arm.MEASURED, data=str(calibration),
                                                        demo_data=str(demo_path)))
        except ValueError as error:
            if not str(error).startswith(NO_MEASURED_SUCCESS):
                raise
            data = None
            no_success_diagnostic = str(error)
        else:
            no_success_diagnostic = None
    arms, levels, result = arm_decision(data, minimum_train, minimum_validation, include_rule, include_base)
    matrix = planned_matrix(arms, levels, predictions)
    result.update({
        "schema_version": 1,
        "calibration_sha256": digest(calibration) if has_calibration else None,
        "calibration_status_sha256": digest(status) if has_status else None,
        "calibration_status": state, "partial_calibration_allowed": allow_partial,
        "demo_sha256": digest(demo_path), "test_contexts_sha256": digest(contexts_path),
        "test_contexts": len(contexts), "test_layout_ids": sorted(layouts),
        "relative_levels": list(TIERS), "absolute_levels": levels,
        "minimum_informative_contexts": {"train": minimum_train, "validation": minimum_validation},
        "base_protocol": "B uses the same input and greedy decoding as D, with no adapter" if include_base else None,
        "no_success_diagnostic": no_success_diagnostic,
        "calibration_inventory": calibration_inventory(calibration),
        "calibration_audit": data.audit if data is not None else None,
        "note": "Arm/slot paths are frozen before generation. Costs and run feasibility require separate assessment.",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".freeze-matrix-", dir=output.parent))
    try:
        (temporary / "matrix.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        result["matrix_sha256"] = digest(temporary / "matrix.json")
        (temporary / "decision.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
        # Close the preflight race immediately before publishing the decision.
        _check_no_test_outputs(predictions)
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-status", type=Path, required=True)
    parser.add_argument("--demo-data", type=Path, required=True)
    parser.add_argument("--test-contexts", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-train-contexts", type=int, default=4)
    parser.add_argument("--minimum-validation-contexts", type=int, default=1)
    parser.add_argument("--expected-boards", type=int, default=12)
    parser.add_argument("--include-rule", action="store_true")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if min(args.minimum_train_contexts, args.minimum_validation_contexts, args.expected_boards) < 1:
        parser.error("Context and board counts must be positive")
    decision = freeze(args.calibration, args.calibration_status, args.demo_data, args.test_contexts,
                      args.predictions_dir, args.output_dir, args.minimum_train_contexts,
                      args.minimum_validation_contexts, args.expected_boards, args.include_rule, args.allow_partial,
                      args.include_base)
    print(json.dumps({"output": str(args.output_dir), "arms": decision["arms"],
                      "reason": decision["reason"], "absolute_levels": decision["absolute_levels"]}, indent=2))


if __name__ == "__main__":
    main()
