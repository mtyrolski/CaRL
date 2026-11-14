#!/usr/bin/env python3
"""Run a frozen, budgeted Luna intent-control experiment on Sokoban boards."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from dotenv import dotenv_values
from openai import OpenAI

from carl.icl.boards import from_rows
from carl.icl.calibration import weighted_percentiles
from carl.icl.data_sources.sokoban import environment
from carl.icl.luna_intent_rollout import POLICY_PROMPT
from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_rollout import rollout
from carl.icl.luna_intent_transport import LunaIntentTransport
from carl.icl.luna_intent_transport import create_luna_budget_config

ROOT = Path(__file__).resolve().parents[1]
ACTION_CAP = 16
MAX_OUTPUT_TOKENS = 64
REQUEST_TIMEOUT_SECONDS = 45


class IntentTransport(Protocol):
    """The transport interface required by the intent controller."""

    def create(self, prompt: str, *, tag: str) -> Any: ...


def digest(path: Path) -> str:
    """Hash an input or code file without relying on its timestamp."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Replace a checkpoint only after its complete contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp, path)


def append_jsonl(path: Path, value: Any) -> None:
    """Preserve every completed event even if a later API request is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def frozen_order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave test families so interruptions can still leave both represented."""
    calibration = [record for record in records if record["phase"] == "calibration"]
    real = [record for record in records if record["phase"] == "test" and record["family"] == "real_12x12x4"]
    synthetic = [record for record in records if record["phase"] == "test" and record["family"] == "proposal_gamma2_12x12x10"]
    if len(calibration) != 12 or len(real) != 8 or len(synthetic) != 8 or len(records) != 28:
        raise ValueError("The frozen selection must contain 12 calibration and 8+8 test records")
    if len({record["record_id"] for record in records}) != len(records):
        raise ValueError("The frozen selection contains duplicate record IDs")
    return calibration + [record for pair in zip(real, synthetic, strict=True) for record in pair]


def create_plan(run_dir: Path) -> dict[str, Any]:
    """Freeze cases, code, controller and API parameters before any paid call."""
    run_dir = run_dir.resolve()
    records_path = run_dir / "data/records.jsonl"
    data_manifest_path = run_dir / "data/manifest.json"
    records = frozen_order(read_jsonl(records_path))
    config_path = create_luna_budget_config(run_dir, "2.00")
    plan = {
        "schema_version": 1,
        "model": "gpt-6-luna",
        "service_tier": "default",
        "reasoning_effort": "none",
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "action_cap": ACTION_CAP,
        "attempts_per_intent": 1,
        "seed": None,
        "pilot": "Separate simple synthetic board, excluded from calibration and test.",
        "record_ids": [record["record_id"] for record in records],
        "record_order": [{key: record[key] for key in ("record_id", "phase", "family", "layout_id", "context_id")}
                         for record in records],
        "records_sha256": digest(records_path),
        "data_manifest_sha256": digest(data_manifest_path),
        "policy_prompt": POLICY_PROMPT,
        "policy_prompt_sha256": hashlib.sha256(POLICY_PROMPT.encode()).hexdigest(),
        "controller_sha256": digest(ROOT / "carl/icl/luna_intent_rollout.py"),
        "transport_sha256": digest(ROOT / "carl/icl/luna_intent_transport.py"),
        "runner_sha256": digest(Path(__file__).resolve()),
        "budget_config_sha256": digest(config_path),
        "budget_ceiling_usd": "2.00",
        "future_data_policy": "Recorded later boards, actions and offsets never enter the controller prompt.",
        "scale_policy": "Within each context, successful action counts are weighted by observed success fractions; failures receive no u.",
        "interpretation": "One attempt per candidate; temperature zero does not guarantee reproducibility.",
    }
    plan_path = run_dir / "experiment_plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved != plan:
            raise ValueError("Frozen experiment inputs, code or configuration have changed")
        return saved
    atomic_json(plan_path, plan)
    return plan


def make_transport(run_dir: Path) -> LunaIntentTransport:
    """Use the explicitly configured project credential without printing it."""
    key = dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("The project .env does not contain an OpenAI API credential")
    client = OpenAI(api_key=key.strip(), timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    return LunaIntentTransport(run_dir / "luna_intent_budget.json", timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                               max_output_tokens=MAX_OUTPUT_TOKENS, client=client)


def run_one(record: dict[str, Any], transport: IntentTransport, run_dir: Path, cap: int) -> dict[str, Any]:
    """Run and durably journal one policy attempt without exposing demonstration data."""
    identifier = record["record_id"]
    started = time.monotonic()

    def event(item: dict[str, Any]) -> None:
        append_jsonl(run_dir / "results/events.jsonl", {"record_id": identifier, **item})

    state = from_rows(record["start_observation"])
    outcome = rollout(environment(state), state, record["intent"],
                      lambda prompt: transport.create(prompt, tag=identifier).output_text,
                      cap=cap, task_goal=record["task_goal"], on_event=event)
    return {
        **{key: record[key] for key in ("record_id", "phase", "family", "layout_id", "context_id")},
        "cap": cap,
        "duration_seconds": time.monotonic() - started,
        "result": asdict(outcome),
    }


def save_calibration(records: list[dict[str, Any]], results: list[dict[str, Any]], path: Path) -> None:
    """Measure u only from successful policy attempts in the same context."""
    by_record = {item["record_id"]: item for item in results}
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["phase"] != "calibration":
            continue
        outcome = by_record.get(record["record_id"])
        # Infrastructure failures are missing measurements, not failed policy attempts.
        measured = (outcome is not None and outcome["result"]["status"] != RolloutStatus.API_ERROR
                    and outcome["result"]["model_calls"] > 0)
        success = bool(measured and outcome is not None and outcome["result"]["achieved"])
        candidate = {
            "record_id": record["record_id"],
            "attempts": int(measured),
            "successes": int(success),
            "success_fraction": float(success) if measured else None,
            "mean_success_actions": len(outcome["result"]["actions"]) if success and outcome is not None else None,
            "u": None,
        }
        contexts[record["context_id"]].append(candidate)
    result = []
    for context_id, candidates in contexts.items():
        complete = all(candidate["attempts"] for candidate in candidates)
        reached = [candidate for candidate in candidates if candidate["successes"]]
        if reached and complete:
            _, percentiles = weighted_percentiles([row["mean_success_actions"] for row in reached],
                                                  [row["successes"] for row in reached])
            for candidate, percentile in zip(reached, percentiles, strict=True):
                candidate["u"] = float(percentile)
        result.append({"context_id": context_id, "complete": complete, "candidates": candidates})
    atomic_json(path, {"schema_version": 1, "contexts": result,
                       "warning": "One attempt per candidate. u is descriptive, conditional on observed success."})


def pilot(run_dir: Path, transport: IntentTransport) -> int:
    path = run_dir / "results/pilot.jsonl"
    if path.exists() and read_jsonl(path):
        raise ValueError("The pilot has already run; inspect its result instead of retrying")
    if any(item.get("record_id") == "synthetic_pilot" for item in read_jsonl(run_dir / "results/events.jsonl")):
        raise ValueError("An interrupted pilot has recorded events; do not repeat its attempt")
    record = {
        "record_id": "synthetic_pilot", "phase": "pilot", "family": "synthetic_pilot",
        "layout_id": "synthetic_pilot", "context_id": "synthetic_pilot",
        "start_observation": ["#####", "#####", "#@$.#", "#####", "#####"],
        "task_goal": "Put every box on a goal.",
        "intent": {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": []},
    }
    result = run_one(record, transport, run_dir, cap=1)
    append_jsonl(path, result)
    print(json.dumps({"phase": "pilot", "status": result["result"]["status"],
                      "duration_seconds": result["duration_seconds"]}), flush=True)
    return int(result["result"]["status"] in {RolloutStatus.API_ERROR, RolloutStatus.INVALID_RESPONSE})


def experiment(run_dir: Path, transport: IntentTransport) -> int:
    records = frozen_order(read_jsonl(run_dir / "data/records.jsonl"))
    pilot_results = read_jsonl(run_dir / "results/pilot.jsonl")
    if (len(pilot_results) != 1
            or pilot_results[0]["result"]["status"] in {RolloutStatus.API_ERROR, RolloutStatus.INVALID_RESPONSE}):
        raise ValueError("A completed pilot with a valid API response is required")
    path = run_dir / "results/rollouts.jsonl"
    results = read_jsonl(path)
    completed = {item["record_id"] for item in results}
    planned = {record["record_id"] for record in records}
    started = {item.get("record_id") for item in read_jsonl(run_dir / "results/events.jsonl")}
    if (started & planned) - completed:
        raise ValueError("An interrupted attempt has recorded events; do not repeat it")
    if any(item["result"]["status"] == RolloutStatus.API_ERROR for item in results):
        raise ValueError("An earlier API failure must be reconciled before continuing")
    for index, record in enumerate(records, start=1):
        if record["record_id"] in completed:
            continue
        item = run_one(record, transport, run_dir, cap=ACTION_CAP)
        append_jsonl(path, item)
        results.append(item)
        save_calibration(records, results, run_dir / "results/calibration.json")
        print(json.dumps({"index": index, "total": len(records), "record_id": item["record_id"],
                          "phase": item["phase"], "family": item["family"],
                          "status": item["result"]["status"], "actions": len(item["result"]["actions"]),
                          "duration_seconds": round(item["duration_seconds"], 2)}), flush=True)
        if item["result"]["status"] == RolloutStatus.API_ERROR:
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "experiment"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    create_plan(run_dir)
    if args.phase == "prepare":
        print(f"Frozen experiment plan: {run_dir / 'experiment_plan.json'}")
        return 0
    transport = make_transport(run_dir)
    return pilot(run_dir, transport) if args.phase == "pilot" else experiment(run_dir, transport)


if __name__ == "__main__":
    sys.exit(main())
