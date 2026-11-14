#!/usr/bin/env python3
"""Save the chosen frozen Qwen policy's observed within-context scale u.

The existing calibration routine calculates midpoint percentiles from actual
successful controller actions. Demonstration offsets are never used as costs.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from carl.icl.intent_fence_audit import remove_one_outer_fence
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.luna_intent_rollout import POLICY_PROMPT
from scripts.audit_luna_rollouts import audit_rollout
from scripts.run_luna_intent_experiment import read_jsonl
from scripts.run_luna_intent_experiment import save_calibration

ROOT = Path(__file__).resolve().parents[1]
CAP = 16


def validate_identities(records: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    """Require exactly one saved attempt for each distinct calibration candidate."""
    record_ids = [item["record_id"] for item in records]
    result_ids = [item["record_id"] for item in results]
    if not records or len(record_ids) != len(set(record_ids)) or len(result_ids) != len(set(result_ids)):
        raise ValueError("Calibration record and result IDs must be nonempty and distinct")
    if record_ids != result_ids or any(record.get("phase") != "calibration" for record in records):
        raise ValueError("Calibration attempts must match the exact frozen record order")
    contexts: dict[str, tuple[Any, ...]] = {}
    candidates: set[tuple[str, str]] = set()
    for record in records:
        context = record["context_id"]
        condition = (record["layout_id"], record["start_observation"], record["task_goal"])
        if context in contexts and contexts[context] != condition:
            raise ValueError("Candidates in one context must share the same current board and task")
        contexts[context] = condition
        intent = record["intent"]
        signature = json.dumps({key: sorted(json.dumps(item, sort_keys=True) for item in intent[key])
                                for key in ("effects", "preserve")}, sort_keys=True)
        key = context, signature
        if key in candidates:
            raise ValueError("A context must not repeat a calibration intent")
        candidates.add(key)


def calculate(records: list[dict[str, Any]], results: list[dict[str, Any]], directory: Path) -> dict[str, Any]:
    """Call the existing policy-relative calibration and test offset independence."""
    validate_identities(records, results)
    first, second = directory / "calibration.json", directory / "changed_offsets.json"
    save_calibration(records, results, first)
    altered = [{**record, "k": 987654, "i": 123456, "fragment_actions": [999],
                "later_observation": ["hidden-future"]} for record in records]
    save_calibration(altered, results, second)
    output = json.loads(first.read_text(encoding="utf-8"))
    if output != json.loads(second.read_text(encoding="utf-8")):
        raise ValueError("Relative scale unexpectedly depends on demonstration fields")
    for context in output["contexts"]:
        add_distribution(context)
    return output


def add_distribution(context: dict[str, Any]) -> None:
    """Record the observed weights and tied CDF intervals underlying each u."""
    candidates = context["candidates"]
    for candidate in candidates:
        candidate.update(weight=None, cdf_before=None, cdf_at=None)
    reached = [candidate for candidate in candidates if candidate["u"] is not None]
    if not reached:
        return
    total = sum(candidate["success_fraction"] for candidate in reached)
    if total <= 0:
        raise ValueError("Labelled candidates must have positive measured success fractions")
    for candidate in reached:
        candidate["weight"] = candidate["success_fraction"] / total
    for candidate in reached:
        cost = candidate["mean_success_actions"]
        before = sum(row["weight"] for row in reached if row["mean_success_actions"] < cost)
        tied = sum(row["weight"] for row in reached if row["mean_success_actions"] == cost)
        candidate["cdf_before"], candidate["cdf_at"] = before, before + tied
        if abs(candidate["u"] - (before + tied / 2)) > 1e-12:
            raise ValueError("Observed label differs from its weighted tied midpoint")


def check_result(record: dict[str, Any], result: dict[str, Any]) -> dict[str, int]:
    """Recheck every saved action and response against the real environment."""
    responses = result.get("raw_policy_responses")
    answered = [step for step in result["result"]["steps"] if step.get("raw_response") is not None]
    if not isinstance(responses, list) or len(responses) != len(answered):
        raise ValueError("Missing saved raw or normalised policy responses")
    for original, step in zip(responses, answered, strict=True):
        normalised, removed = remove_one_outer_fence(original["raw"])
        if (original.get("normalised") != normalised or original.get("fence_removed") != removed
                or step["raw_response"] != normalised):
            raise ValueError("Saved controller response normalisation differs")
    row = {**{key: record.get(key) for key in ("record_id", "phase", "family", "layout_id", "context_id")},
           **result, "cap": CAP}
    return audit_rollout(record, row, POLICY_PROMPT)


def build(run: Path) -> dict[str, Any]:
    records_path, results_path = run / "policy_calibration.jsonl", run / "policy_calibration_05b.jsonl"
    manifest_path, selection_path = run / "policy_calibration_05b.manifest.json", run / "controller_selection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records, results = read_jsonl(records_path), read_jsonl(results_path)
    validate_identities(records, results)
    if (selection.get("selected") != "05b" or selection.get("selected_config") != manifest.get("config")
            or selection.get("selected_configuration_sha256") != manifest.get("configuration_sha256")
            or selection.get("parser") != manifest.get("action_parser")
            or manifest.get("config", {}).get("action_cap") != CAP):
        raise ValueError("Controller config, budget or selection do not match")
    if (manifest.get("records_sha256") != file_sha256(records_path)
            or selection.get("calibration_records_sha256") != file_sha256(records_path)
            or selection.get("result_sha256", {}).get("05b") != file_sha256(results_path)
            or manifest.get("record_ids") != [row["record_id"] for row in records]):
        raise ValueError("Frozen calibration data or results changed")
    metadata = {key: value for key, value in manifest.items() if key != "configuration_sha256"}
    calculated_config = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if calculated_config != manifest["configuration_sha256"]:
        raise ValueError("Saved controller configuration fingerprint differs")
    model = Path(manifest["config"]["model"])
    if (file_sha256(model / "model.safetensors") != manifest["weights_sha256"]
            or file_sha256(model / "tokenizer.json") != manifest["tokenizer_sha256"]):
        raise ValueError("Selected controller weights or tokenizer changed")
    for relative, digest in manifest["source_sha256"].items():
        if file_sha256(ROOT / relative) != digest:
            raise ValueError(f"Selected controller source changed: {relative}")
    counts: Counter[str] = Counter()
    for record, result in zip(records, results, strict=True):
        counts.update(check_result(record, result))
    successes = sum(result["result"]["achieved"] is True for result in results)
    if successes != selection["scores"]["05b"]["successes"]:
        raise ValueError("Calibration successes differ from the selected controller summary")
    with tempfile.TemporaryDirectory(prefix=".relative-u-", dir=run) as temporary:
        data = calculate(records, results, Path(temporary))
    candidates_by_id = {row["record_id"]: row for row in results}
    for context in data["contexts"]:
        for candidate in context["candidates"]:
            actual = candidates_by_id[candidate["record_id"]]["result"]
            candidate.update({"controller_status": actual["status"], "primitive_actions_attempted": len(actual["actions"]),
                              "model_calls": actual["model_calls"],
                              "no_ops": sum(step.get("no_op") is True for step in actual["steps"])})
    data.update({
        "provenance": {"records_path": str(records_path), "records_sha256": file_sha256(records_path),
                       "results_path": str(results_path), "results_sha256": file_sha256(results_path),
                       "controller_manifest_sha256": file_sha256(manifest_path),
                       "controller_selection_sha256": file_sha256(selection_path),
                       "controller_configuration_sha256": manifest["configuration_sha256"],
                       "policy_config_sha256": manifest["policy_config_sha256"],
                       "weights_sha256": manifest["weights_sha256"], "tokenizer_sha256": manifest["tokenizer_sha256"],
                       "controller_source_sha256": manifest["source_sha256"], "controller_config": manifest["config"],
                       "policy_prompt_sha256": hashlib.sha256(POLICY_PROMPT.encode()).hexdigest(),
                       "calculation_source_sha256": file_sha256(ROOT / "scripts/run_luna_intent_experiment.py"),
                       "percentile_source_sha256": file_sha256(ROOT / "carl/icl/calibration.py"),
                       "budget_primitive_actions": CAP, "attempts_per_candidate": 1, "seed": None,
                       "decoding": "greedy deterministic", "action_parser": manifest["action_parser"],
                       "replayed_actions": counts["executed_actions"], "model_calls": counts["model_calls"],
                       "command": sys.argv},
        "summary": {"contexts": len(data["contexts"]), "candidates": len(records),
                    "successes": successes, "labelled": sum(row["u"] is not None for group in data["contexts"]
                                                              for row in group["candidates"])},
        "interpretation": "Within each context, use the weighted empirical distribution of the mean first-success "
                          "primitive action counts among successful candidates; one deterministic attempt gives "
                          "success fractions 0 or 1. Equal costs share the midpoint of their combined interval. "
                          "A candidate without success has u=null; incomplete contexts receive no labels. "
                          "Demonstration offsets k never determine u or duration.",
        "limitation": "One observed successful candidate and one u label are insufficient to train or validate "
                      "scale conditioning; these labels are not used to train the generator.",
    })
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    destination = args.output or run / "policy_calibration_relative_u.json"
    value = build(run)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"output": str(destination), **value["summary"]}))


if __name__ == "__main__":
    main()
