#!/usr/bin/env python3
"""Replay saved Luna actions and independently check their recorded prompts."""

import argparse
import contextlib
import hashlib
import io
import json
import shlex
import sys
from collections import Counter
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import predicate_value
from carl.icl.intent_data import validate_prediction
from carl.icl.luna_components.formatting import board_coordinates
from scripts.prepare_intent_generator_data import write_json

VARIANTS = ("", "structured", "final_response", "expanded")
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def snapshot(path: Path) -> tuple[bytes, str]:
    """Read each input once so the digest identifies exactly what was checked."""
    data = path.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def jsonl(data: bytes) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("A JSONL record is not an object")
    return rows


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_rollout(case: dict[str, Any], row: dict[str, Any], policy_prompt: str) -> dict[str, int]:
    """Verify one saved rollout against the simulator and permitted inputs."""
    identifier = case["record_id"]
    for key in ("record_id", "phase", "family", "layout_id", "context_id"):
        check(row.get(key) == case.get(key), f"{identifier}: rollout {key} mismatch")
    result = row["result"]
    current = case["start_observation"]
    intent = case["intent"]
    check(not validate_prediction(intent, current), f"{identifier}: invalid starting intent")
    actions: list[int] = []
    history: list[dict[str, Any]] = []
    with contextlib.redirect_stdout(io.StringIO()):
        env = environment(from_rows(current))
        steps = result["steps"]
        check(result["model_calls"] == len(steps), f"{identifier}: model call count mismatch")
        for index, step in enumerate(steps):
            check(step["index"] == index and step["before"] == current, f"{identifier}: step {index} starts incorrectly")
            prefix, payload_text = step["prompt"].split("\n\n", 1)
            check(prefix == policy_prompt, f"{identifier}: step {index} policy prompt differs from plan")
            payload = json.loads(payload_text)
            expected = {"state": current, "state_coordinates": board_coordinates(current),
                        "intent": {"effects": intent["effects"], "preserve": intent["preserve"]},
                        "task_goal": case["task_goal"], "attempt_history": history}
            check(payload == expected, f"{identifier}: step {index} prompt contains unexpected or incorrect inputs")
            action = step["action"]
            if action is None:
                check(index == len(steps) - 1 and step["after"] is None,
                      f"{identifier}: non-action step is not final")
                continue
            check(isinstance(action, int) and not isinstance(action, bool) and action in range(4),
                  f"{identifier}: invalid primitive action")
            response = json.loads(step["raw_response"])
            check(response == {"action": DIRECTIONS[action]}, f"{identifier}: action differs from response")
            following = to_rows(env.next_state(from_rows(current), action))
            check(following == step["after"], f"{identifier}: step {index} transition differs from simulator")
            noop = following == current
            holds = all(predicate_value(predicate, following) for predicate in intent["effects"])
            preserved = all(predicate_value(predicate, following) for predicate in intent["preserve"])
            solved = bool(env.is_solved(from_rows(following)))
            check(step["no_op"] is noop and step["intent_holds"] is holds
                  and step["preservation_holds"] is preserved and step["task_solved"] is solved,
                  f"{identifier}: step {index} predicate, no-op or completion flag mismatch")
            actions.append(action)
            history.append({"action": DIRECTIONS[action], "no_op": noop})
            current = following
            if not preserved or holds or solved:
                check(index == len(steps) - 1, f"{identifier}: execution continued after a stopping condition")
                expected_status = "preservation_violated" if not preserved else "achieved" if holds else "task_solved"
                check(result["status"] == expected_status, f"{identifier}: stopping status differs from simulator")
        check(actions == result["actions"] and current == result["endpoint"],
              f"{identifier}: action list or endpoint mismatch")
        if result["status"] == "action_limit":
            check(len(actions) == row["cap"], f"{identifier}: action-limit status differs from cap")
        elif result["status"] in {"invalid_response", "api_error"}:
            check(bool(steps) and steps[-1]["action"] is None, f"{identifier}: error status lacks a failed final call")
        check(bool(env.is_solved(from_rows(current))) == result["task_solved"],
              f"{identifier}: final solved flag mismatch")
        check(result["achieved"] == (result["status"] == "achieved"), f"{identifier}: achieved status mismatch")
        if result["achieved"]:
            check(all(predicate_value(predicate, current) for predicate in intent["effects"]),
                  f"{identifier}: final intent is not satisfied")
    return {"executed_actions": len(actions), "model_calls": len(steps)}


def audit_variant(directory: Path) -> dict[str, Any]:
    case_bytes, case_hash = snapshot(directory / "data/records.jsonl")
    result_bytes, result_hash = snapshot(directory / "results/rollouts.jsonl")
    plan_bytes, plan_hash = snapshot(directory / "experiment_plan.json")
    cases = jsonl(case_bytes)
    results = jsonl(result_bytes)
    plan = json.loads(plan_bytes)
    case_ids = [case["record_id"] for case in cases]
    result_ids = [row["record_id"] for row in results]
    check(len(set(case_ids)) == len(case_ids), f"{directory}: duplicate case IDs")
    check(len(set(result_ids)) == len(result_ids), f"{directory}: duplicate rollout IDs")
    by_id = {case["record_id"]: case for case in cases}
    check(set(result_ids) <= set(plan["record_ids"]) <= set(case_ids), f"{directory}: records not in frozen plan")
    summary: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    for row in results:
        counts = audit_rollout(by_id[row["record_id"]], row, plan["policy_prompt"])
        summary.update(counts)
        statuses[row["result"]["status"]] += 1
        phases[f"{row['phase']}:{row['family']}"] += 1
    return {
        "status": "passed",
        "planned_records": len(plan["record_ids"]),
        "recorded_rollouts": len(results),
        "rollouts_not_saved": len(plan["record_ids"]) - len(results),
        "executed_actions": summary["executed_actions"],
        "model_calls": summary["model_calls"],
        "rollouts_by_status": dict(statuses),
        "rollouts_by_phase_family": dict(phases),
        "input_sha256": {"records_jsonl": case_hash, "rollouts_jsonl": result_hash, "experiment_plan_json": plan_hash},
    }


def audit_campaign(run: Path) -> dict[str, Any]:
    results = {variant or "v1": audit_variant(run / variant) for variant in VARIANTS}
    return {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "variants": results,
        "total_recorded_rollouts": sum(row["recorded_rollouts"] for row in results.values()),
        "total_executed_actions": sum(row["executed_actions"] for row in results.values()),
        "checks": ["record identity and plan membership", "allowed prompt fields, current board and own action history",
                   "saved action versus response", "simulator transition, no-op, predicate and solved flags",
                   "first stopping condition, final action list and endpoint"],
        "limitations": ["Checks saved rollout records, not unsaved attempts or API billing.",
                        "Validates the action actually executed; does not repair earlier parser artifacts.",
                        "Input hashes represent the recorded snapshot, so rerun after continuation."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_campaign(args.run.resolve())
    report["reproduction"] = {
        "working_directory": str(Path(__file__).resolve().parents[1]),
        "command": shlex.join([sys.executable, str(Path(__file__).resolve()), "--run", str(args.run.resolve()),
                               "--output", str(args.output.resolve())]),
    }
    write_json(args.output.resolve(), report)
    print(json.dumps({"status": report["status"], "recorded_rollouts": report["total_recorded_rollouts"],
                      "executed_actions": report["total_executed_actions"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
