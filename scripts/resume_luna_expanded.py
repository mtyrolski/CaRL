#!/usr/bin/env python3
"""Continue an interrupted Luna run while retaining its one unknown reservation."""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI

from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_transport import LunaIntentTransport
from scripts.run_luna_expanded import FinalMessageTransport
from scripts.run_luna_expanded import MAX_OUTPUT_TOKENS
from scripts.run_luna_expanded import order
from scripts.run_luna_intent_experiment import ACTION_CAP
from scripts.run_luna_intent_experiment import REQUEST_TIMEOUT_SECONDS
from scripts.run_luna_intent_experiment import append_jsonl
from scripts.run_luna_intent_experiment import atomic_json
from scripts.run_luna_intent_experiment import digest
from scripts.run_luna_intent_experiment import read_jsonl
from scripts.run_luna_intent_experiment import run_one
from scripts.run_luna_intent_experiment import save_calibration

ROOT = Path(__file__).resolve().parents[1]


def create_recovery_plan(campaign: Path, unknown_id: str) -> dict[str, Any]:
    """Freeze the retained reservation and unchanged first part of the run."""
    campaign = campaign.resolve()
    if re.fullmatch(r"[0-9a-f]{32}", unknown_id) is None:
        raise ValueError("The unknown reservation ID must be a lowercase hexadecimal identifier")
    run = campaign / "expanded"
    plan_path = run / "recovery_plan.json"
    old_plan_path = run / "experiment_plan.json"
    old_plan = json.loads(old_plan_path.read_text(encoding="utf-8"))
    dependencies = {
        "controller": ROOT / "carl/icl/luna_intent_rollout.py",
        "action_format": ROOT / "carl/icl/luna_action_format.py",
        "final_response": ROOT / "carl/icl/luna_response.py",
        "shared_runner": ROOT / "scripts/run_luna_intent_experiment.py",
        "variant_runner": ROOT / "scripts/run_luna_expanded.py",
        "boards": ROOT / "carl/icl/boards.py",
        "predicates": ROOT / "carl/icl/intent_data.py",
        "formatting": ROOT / "carl/icl/luna_components/formatting.py",
        "sokoban_data_source": ROOT / "carl/icl/data_sources/sokoban.py",
        "sokoban_environment": ROOT / "carl/environment/sokoban/env.py",
        "calibration": ROOT / "carl/icl/calibration.py",
    }
    if any(digest(path) != old_plan["code_sha256"][name] for name, path in dependencies.items()):
        raise ValueError("A frozen policy, environment or experiment dependency has changed")
    if digest(campaign / "luna_intent_budget.json") != old_plan["budget_config_sha256"]:
        raise ValueError("The frozen budget configuration changed")
    if digest(campaign / "code_v4/luna_intent_transport.py") != old_plan["code_sha256"]["transport"]:
        raise ValueError("The original transport snapshot does not match the expanded plan")
    data_path = run / "data/records.jsonl"
    if digest(data_path) != old_plan["records_sha256"]:
        raise ValueError("The frozen expanded data changed")
    audit_path = campaign / "luna_requests" / f"{unknown_id}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("reservation_id") != unknown_id or audit.get("status") != "unknown"
            or audit.get("error_diagnostics", {}).get("http_status") != 400
            or not str(audit.get("tag", "")).startswith("expanded:")):
        raise ValueError("Expected the recorded unknown HTTP 400 from this experiment")
    with sqlite3.connect(campaign / "luna_intent_ledger.sqlite3") as db:
        pending = db.execute("SELECT id,status,reserved_nanos,charged_nanos,request_hash "
                             "FROM attempts WHERE status != 'settled'").fetchall()
    if (len(pending) != 1 or pending[0][0] != unknown_id or pending[0][1] != "unknown"
            or pending[0][3] is not None or pending[0][4] != audit.get("request_sha256")):
        raise ValueError("Expected precisely the same outstanding ledger reservation")
    results_path = run / "results/rollouts.jsonl"
    results_bytes = results_path.read_bytes().splitlines(keepends=True)
    results = [json.loads(line) for line in results_bytes]
    existing_plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    initial_count = existing_plan["initial_result_count"] if existing_plan else len(results)
    if len(results) < initial_count:
        raise ValueError("The saved prefix of completed attempts is incomplete")
    prefix = results[:initial_count]
    prefix_ids = [row["record_id"] for row in prefix]
    if prefix_ids != old_plan["record_ids"][:initial_count]:
        raise ValueError("Saved results do not follow the frozen selection order")
    errors = [row for row in prefix if row["result"]["status"] == RolloutStatus.API_ERROR]
    failed_id = str(audit["tag"]).removeprefix("expanded:")
    if len(errors) != 1 or errors[0]["record_id"] != failed_id:
        raise ValueError("The interrupted attempt does not match the unknown API request")
    plan = {
        "schema_version": 1,
        "original_plan_sha256": digest(old_plan_path),
        "data_sha256": digest(data_path),
        "budget_config_sha256": digest(campaign / "luna_intent_budget.json"),
        "unknown_reservation_id": unknown_id,
        "unknown_request_sha256": audit["request_sha256"],
        "unknown_reserved_nanos": pending[0][2],
        "unknown_status": "unknown; full reservation retained, actual cost not known",
        "failed_record_id": failed_id,
        "initial_result_count": initial_count,
        "initial_results_sha256": hashlib.sha256(b"".join(results_bytes[:initial_count])).hexdigest(),
        "controller_sha256": digest(ROOT / "carl/icl/luna_intent_rollout.py"),
        "transport_sha256": digest(ROOT / "carl/icl/luna_intent_transport.py"),
        "response_parser_sha256": digest(ROOT / "carl/icl/luna_response.py"),
        "action_format_sha256": digest(ROOT / "carl/icl/luna_action_format.py"),
        "shared_runner_sha256": digest(ROOT / "scripts/run_luna_intent_experiment.py"),
        "expanded_runner_sha256": digest(ROOT / "scripts/run_luna_expanded.py"),
        "recovery_runner_sha256": digest(Path(__file__).resolve()),
        "policy": "Continue the same frozen selection; do not retry the failed attempt. "
                  "Keep the full unknown reservation against the shared $2 limit. "
                  "Any second uncertain or reserved request stops the campaign.",
    }
    if existing_plan is not None:
        if plan != existing_plan:
            raise ValueError("Recovery inputs, original results or code have changed")
        return existing_plan
    atomic_json(plan_path, plan)
    return plan


def make_transport(campaign: Path, run: Path, unknown_id: str) -> FinalMessageTransport:
    key = dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("The project .env does not contain an OpenAI API credential")
    client = OpenAI(api_key=key.strip(), timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    transport = LunaIntentTransport(campaign / "luna_intent_budget.json", timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                                    max_output_tokens=MAX_OUTPUT_TOKENS, client=client, structured_actions=True,
                                    accepted_unknown_id=unknown_id)
    return FinalMessageTransport(transport, run)


def resume(campaign: Path, plan: dict[str, Any]) -> int:
    run = campaign / "expanded"
    records = order(read_jsonl(run / "data/records.jsonl"))
    path = run / "results/rollouts.jsonl"
    results = read_jsonl(path)
    completed = {row["record_id"] for row in results}
    if len(completed) != len(results):
        raise ValueError("Duplicate saved attempts require manual inspection")
    planned = {row["record_id"] for row in records}
    started = {row.get("record_id") for row in read_jsonl(run / "results/events.jsonl")}
    if completed - planned or (started & planned) - completed:
        raise ValueError("Unexpected or interrupted attempts require manual inspection")
    errors = [row for row in results if row["result"]["status"] == RolloutStatus.API_ERROR]
    if len(errors) != 1 or errors[0]["record_id"] != plan["failed_record_id"]:
        raise ValueError("A new API failure or missing original failure prevents continuation")
    save_calibration(records, results, run / "results/calibration.json")
    transport = make_transport(campaign, run, plan["unknown_reservation_id"])
    for index, record in enumerate(records, 1):
        if record["record_id"] in completed:
            continue
        item = run_one(record, transport, run, cap=ACTION_CAP)
        append_jsonl(path, item)
        results.append(item)
        save_calibration(records, results, run / "results/calibration.json")
        print(json.dumps({"index": index, "total": len(records), "record_id": item["record_id"],
                          "phase": item["phase"], "family": item["family"],
                          "status": item["result"]["status"], "actions": len(item["result"]["actions"]),
                          "duration_seconds": round(item["duration_seconds"], 2)}), flush=True)
        if item["result"]["status"] == RolloutStatus.API_ERROR:
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "resume"))
    parser.add_argument("--campaign-run", type=Path, required=True)
    parser.add_argument("--unknown-id", required=True)
    args = parser.parse_args()
    campaign = args.campaign_run.resolve()
    plan = create_recovery_plan(campaign, args.unknown_id)
    if args.phase == "prepare":
        print(f"Frozen recovery plan: {campaign / 'expanded/recovery_plan.json'}")
        return 0
    return resume(campaign, plan)


if __name__ == "__main__":
    sys.exit(main())
