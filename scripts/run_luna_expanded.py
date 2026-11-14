#!/usr/bin/env python3
"""Re-evaluate all previously selected Luna test boards with a larger output cap."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI

from carl.icl.luna_intent_rollout import POLICY_PROMPT
from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_transport import LunaIntentTransport
from carl.icl.luna_response import LunaResponseError
from carl.icl.luna_response import final_assistant_text
from scripts.run_luna_intent_experiment import ACTION_CAP
from scripts.run_luna_intent_experiment import REQUEST_TIMEOUT_SECONDS
from scripts.run_luna_intent_experiment import append_jsonl
from scripts.run_luna_intent_experiment import atomic_json
from scripts.run_luna_intent_experiment import digest
from scripts.run_luna_intent_experiment import pilot
from scripts.run_luna_intent_experiment import read_jsonl
from scripts.run_luna_intent_experiment import run_one
from scripts.run_luna_intent_experiment import save_calibration

ROOT = Path(__file__).resolve().parents[1]
TAG_PREFIX = "expanded:"
MAX_OUTPUT_TOKENS = 256


class FinalMessageTransport:
    """Use the final model message and share the original campaign ledger."""

    def __init__(self, transport: LunaIntentTransport, run_dir: Path) -> None:
        self.transport = transport
        self.run_dir = run_dir

    def create(self, prompt: str, *, tag: str) -> SimpleNamespace:
        response = self.transport.create(prompt, tag=TAG_PREFIX + tag)
        if response.status in {"failed", "cancelled"}:
            raise RuntimeError("The Responses service did not complete the request")
        try:
            text = final_assistant_text(response)
        except LunaResponseError as error:
            append_jsonl(self.run_dir / "results/response_errors.jsonl", {"record_id": tag, "error": str(error)})
            text = ""
        return SimpleNamespace(output_text=text)


def order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calibration = [row for row in records if row["phase"] == "calibration"]
    real = [row for row in records if row["phase"] == "test" and row["family"] == "real_12x12x4"]
    synthetic = [row for row in records if row["phase"] == "test" and row["family"] == "proposal_gamma2_12x12x10"]
    if len(calibration) != 12 or len(real) != 20 or len(synthetic) != 20 or len(records) != 52:
        raise ValueError("Expected 12 calibration records and 20+20 test records")
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("The expanded selection contains duplicate IDs")
    return calibration + [row for pair in zip(real, synthetic, strict=True) for row in pair]


def create_plan(campaign: Path, run_dir: Path) -> dict[str, Any]:
    campaign, run_dir = campaign.resolve(), run_dir.resolve()
    if run_dir != campaign / "expanded":
        raise ValueError("The expanded variant must stay in the existing campaign directory")
    config = campaign / "luna_intent_budget.json"
    earlier_dirs = [campaign, campaign / "structured", campaign / "final_response"]
    earlier_data = [path / "data/records.jsonl" for path in earlier_dirs]
    earlier_plans = [path / "experiment_plan.json" for path in earlier_dirs]
    if not config.is_file() or not all(path.is_file() for path in earlier_data + earlier_plans):
        raise ValueError("The original budget and all previous frozen selections are required")
    records_path = run_dir / "data/records.jsonl"
    manifest_path = run_dir / "data/manifest.json"
    records = order(read_jsonl(records_path))
    earlier = [read_jsonl(path) for path in earlier_data]
    reference_calibration = {row["record_id"]: row for row in earlier[0] if row["phase"] == "calibration"}
    expected_tests = {row["record_id"]: row for group in earlier for row in group if row["phase"] == "test"}
    calibration = {row["record_id"]: row for row in records if row["phase"] == "calibration"}
    tests = {row["record_id"]: row for row in records if row["phase"] == "test"}
    if calibration != reference_calibration or tests != expected_tests or len(expected_tests) != 40:
        raise ValueError("Expanded records must exactly match all previously selected cases")
    if len({row["layout_id"] for row in tests.values()}) != 40:
        raise ValueError("Expanded test layouts must be distinct")
    inputs = {
        "controller": ROOT / "carl/icl/luna_intent_rollout.py",
        "transport": ROOT / "carl/icl/luna_intent_transport.py",
        "action_format": ROOT / "carl/icl/luna_action_format.py",
        "final_response": ROOT / "carl/icl/luna_response.py",
        "shared_runner": ROOT / "scripts/run_luna_intent_experiment.py",
        "variant_runner": Path(__file__).resolve(),
        "boards": ROOT / "carl/icl/boards.py",
        "predicates": ROOT / "carl/icl/intent_data.py",
        "formatting": ROOT / "carl/icl/luna_components/formatting.py",
        "sokoban_data_source": ROOT / "carl/icl/data_sources/sokoban.py",
        "sokoban_environment": ROOT / "carl/environment/sokoban/env.py",
        "calibration": ROOT / "carl/icl/calibration.py",
    }
    plan = {
        "schema_version": 1, "variant": "strict_final_expanded_256", "model": "gpt-6-luna",
        "service_tier": "default", "reasoning_effort": "none", "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "action_cap": ACTION_CAP, "attempts_per_intent": 1, "seed": None,
        "pilot": "Separate simple synthetic board, excluded from calibration and test.",
        "record_ids": [row["record_id"] for row in records],
        "record_order": [{key: row[key] for key in ("record_id", "phase", "family", "layout_id", "context_id")}
                         for row in records],
        "records_sha256": digest(records_path), "data_manifest_sha256": digest(manifest_path),
        "preceding_plan_sha256": [digest(path) for path in earlier_plans],
        "code_sha256": {name: digest(path) for name, path in inputs.items()},
        "policy_prompt": POLICY_PROMPT, "policy_prompt_sha256": hashlib.sha256(POLICY_PROMPT.encode()).hexdigest(),
        "budget_config_sha256": digest(config), "budget_ceiling_usd": "2.00 shared across all variants",
        "audit_run": "..", "audit_tag_prefix": TAG_PREFIX,
        "future_data_policy": "Recorded later boards, actions and offsets never enter the controller prompt.",
        "selection": "All previously selected test boards, 20 per family; no selection by model outcome.",
        "interpretation": "Repeated evaluation of previously tested boards, not a fresh holdout.",
        "adaptation": "Strict schema and final-message parser; output cap raised from 64 to 256 after calibration.",
    }
    plan_path = run_dir / "experiment_plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved != plan:
            raise ValueError("Frozen experiment inputs, code or configuration have changed")
        return saved
    atomic_json(plan_path, plan)
    return plan


def make_transport(campaign: Path, run_dir: Path) -> FinalMessageTransport:
    key = dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("The project .env does not contain an OpenAI API credential")
    client = OpenAI(api_key=key.strip(), timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    transport = LunaIntentTransport(campaign / "luna_intent_budget.json", timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                                    max_output_tokens=MAX_OUTPUT_TOKENS, client=client, structured_actions=True)
    return FinalMessageTransport(transport, run_dir)


def experiment(run_dir: Path, transport: FinalMessageTransport) -> int:
    records = order(read_jsonl(run_dir / "data/records.jsonl"))
    pilot_results = read_jsonl(run_dir / "results/pilot.jsonl")
    if (len(pilot_results) != 1
            or pilot_results[0]["result"]["status"] in {RolloutStatus.API_ERROR, RolloutStatus.INVALID_RESPONSE}):
        raise ValueError("A completed pilot with a valid final API response is required")
    path = run_dir / "results/rollouts.jsonl"
    results = read_jsonl(path)
    completed = {row["record_id"] for row in results}
    planned = {row["record_id"] for row in records}
    started = {row.get("record_id") for row in read_jsonl(run_dir / "results/events.jsonl")}
    if (started & planned) - completed:
        raise ValueError("An interrupted attempt has recorded events; do not repeat it")
    if any(row["result"]["status"] == RolloutStatus.API_ERROR for row in results):
        raise ValueError("An earlier API failure must be reconciled before continuing")
    save_calibration(records, results, run_dir / "results/calibration.json")
    for index, record in enumerate(records, 1):
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
    parser.add_argument("--campaign-run", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_run.resolve()
    run_dir = campaign / "expanded"
    create_plan(campaign, run_dir)
    if args.phase == "prepare":
        print(f"Frozen expanded plan: {run_dir / 'experiment_plan.json'}")
        return 0
    transport = make_transport(campaign, run_dir)
    return pilot(run_dir, transport) if args.phase == "pilot" else experiment(run_dir, transport)


if __name__ == "__main__":
    sys.exit(main())
