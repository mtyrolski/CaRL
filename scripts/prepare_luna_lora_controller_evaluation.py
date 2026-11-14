#!/usr/bin/env python3
"""Prepare matched Luna inputs from saved generator outputs without API calls."""

import argparse
import hashlib
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_data import validate_prediction
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records


class Arm(StrEnum):
    UNSCALED = "unscaled"
    SCALED = "scaled"


GENERATION_STATUSES = frozenset({"valid_intent", "invalid_json", "invalid_intent", "infrastructure_error",
                                 "not_attempted"})


def _index(rows: list[dict[str, Any]], arm: Arm, requested: float | None,
           contexts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {}
    for row in rows:
        identifier = row.get("record_id")
        if (identifier not in contexts or identifier in by_id or row.get("arm") != arm
                or row.get("requested_scale") != requested or row.get("status") not in GENERATION_STATUSES):
            raise ValueError("Prediction does not match the frozen context, arm, scale or status")
        context = contexts[identifier]
        if any(row.get(key) != context[key] for key in ("context_id", "layout_id", "family")):
            raise ValueError("Prediction context provenance differs from the frozen selection")
        if row["status"] == "valid_intent":
            if row.get("json_valid") is not True or validate_prediction(row.get("parsed"), context["start_observation"]):
                raise ValueError("An allegedly valid prediction does not pass the current-board checker")
        by_id[identifier] = row
    if set(by_id) != set(contexts):
        raise ValueError("Every planned context needs one saved prediction or an explicit missing status")
    return by_id


def _case_id(identifier: str, arm: Arm, scale: float) -> str:
    text = f"luna-lora-controller-case-v1|{identifier}|{arm}|{scale:.8g}"
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def prepare_cases(context_rows: list[dict[str, Any]], unscaled: list[dict[str, Any]],
                  scaled: dict[float, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep every generation outcome; send only checked intents to the controller.

    The unscaled generator is run once per context and reused across the two
    requested-scale conditions. Each valid case later gets its own policy
    attempt; duplicate outputs are not counted as independent boards.
    """
    contexts = {row["record_id"]: row for row in context_rows}
    if (not contexts or len(contexts) != len(context_rows)
            or any(row.get("split") != "test" for row in context_rows)):
        raise ValueError("Distinct frozen test contexts are required")
    if len(scaled) < 2 or any(isinstance(scale, bool) or not 0 <= scale <= 1 for scale in scaled):
        raise ValueError("At least two distinct requested scales in [0,1] are required")
    predictions: dict[tuple[Arm, float | None], dict[str, dict[str, Any]]] = {
        (Arm.UNSCALED, None): _index(unscaled, Arm.UNSCALED, None, contexts)
    }
    for scale, rows in scaled.items():
        predictions[Arm.SCALED, scale] = _index(rows, Arm.SCALED, scale, contexts)
    cases, controller = [], []
    for context in context_rows:
        identifier = context["record_id"]
        for scale in sorted(scaled):
            for arm in Arm:
                prediction = predictions[arm, None if arm == Arm.UNSCALED else scale][identifier]
                case = {
                    "record_id": _case_id(identifier, arm, scale), "source_record_id": identifier,
                    "context_id": context["context_id"], "layout_id": context["layout_id"],
                    "family": context["family"], "split": "test", "phase": "test",
                    "arm": arm.value, "requested_scale": scale,
                    "prediction_status": prediction["status"], "raw_output": prediction.get("raw_output"),
                    "parsed": prediction.get("parsed"), "json_valid": prediction.get("json_valid"),
                    "validation_errors": prediction.get("validation_errors", []),
                    "start_observation": context["start_observation"], "task_goal": context["task_goal"],
                }
                cases.append(case)
                if prediction["status"] == "valid_intent":
                    controller.append({**case, "intent": prediction["parsed"]})
    if len({row["record_id"] for row in cases}) != len(cases):
        raise ValueError("Evaluation case identity collision")
    return cases, controller


def _write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite a different evaluation selection: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--unscaled", type=Path, required=True)
    parser.add_argument("--scaled", type=Path, nargs=2, required=True, metavar=("LOW", "HIGH"))
    parser.add_argument("--scales", type=float, nargs=2, default=(0.25, 0.75))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scales[0] == args.scales[1]:
        raise ValueError("The two requested scales must differ")
    inputs = [args.contexts, args.unscaled, *args.scaled]
    cases, controller = prepare_cases(read_records(args.contexts), read_records(args.unscaled),
                                      {scale: read_records(path) for scale, path in zip(args.scales, args.scaled,
                                                                                      strict=True)})
    for name, rows in (("evaluation_plan.jsonl", cases), ("controller_inputs.jsonl", controller)):
        _write_new(args.output / name,
                   "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows))
    manifest = {"schema_version": 1, "input_sha256": {str(path.resolve()): file_sha256(path) for path in inputs},
                "planned_cases": len(cases), "controller_cases": len(controller),
                "cases_sha256": file_sha256(args.output / "evaluation_plan.jsonl"),
                "controller_inputs_sha256": file_sha256(args.output / "controller_inputs.jsonl"),
                "unscaled_policy": "The same saved unscaled generation is evaluated at both requested scales.",
                "missing_policy": "Invalid or missing generation remains in the plan but is not sent to Luna."}
    _write_new(args.output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"planned_cases": len(cases), "controller_cases": len(controller)}))


if __name__ == "__main__":
    main()
