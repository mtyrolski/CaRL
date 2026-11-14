#!/usr/bin/env python3
"""Run a strict-JSON Luna policy on a separate, preselected Sokoban test set."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from openai import OpenAI

from carl.icl.luna_intent_rollout import POLICY_PROMPT
from carl.icl.luna_intent_transport import LunaIntentTransport
from scripts.run_luna_intent_experiment import ACTION_CAP
from scripts.run_luna_intent_experiment import MAX_OUTPUT_TOKENS
from scripts.run_luna_intent_experiment import REQUEST_TIMEOUT_SECONDS
from scripts.run_luna_intent_experiment import atomic_json
from scripts.run_luna_intent_experiment import digest
from scripts.run_luna_intent_experiment import experiment
from scripts.run_luna_intent_experiment import frozen_order
from scripts.run_luna_intent_experiment import pilot
from scripts.run_luna_intent_experiment import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
TAG_PREFIX = "structured:"


class TaggedTransport:
    """Identify this variant while using the original campaign's budget ledger."""

    def __init__(self, transport: LunaIntentTransport) -> None:
        self.transport = transport

    def create(self, prompt: str, *, tag: str) -> Any:
        return self.transport.create(prompt, tag=TAG_PREFIX + tag)


def create_plan(campaign: Path, run_dir: Path) -> dict[str, Any]:
    """Freeze the new variant and verify that its test boards are independent."""
    campaign = campaign.resolve()
    run_dir = run_dir.resolve()
    if run_dir != campaign / "structured":
        raise ValueError("The structured variant must stay inside the original campaign directory")
    config_path = campaign / "luna_intent_budget.json"
    original_plan_path = campaign / "experiment_plan.json"
    if not config_path.is_file() or not original_plan_path.is_file():
        raise ValueError("The existing campaign budget and frozen plan are required")
    records_path = run_dir / "data/records.jsonl"
    data_manifest_path = run_dir / "data/manifest.json"
    records = frozen_order(read_jsonl(records_path))
    original = read_jsonl(campaign / "data/records.jsonl")
    original_calibration = {row["record_id"] for row in original if row["phase"] == "calibration"}
    calibration = {row["record_id"] for row in records if row["phase"] == "calibration"}
    original_layouts = {row["layout_id"] for row in original}
    new_test_layouts = {row["layout_id"] for row in records if row["phase"] == "test"}
    if calibration != original_calibration or new_test_layouts & original_layouts:
        raise ValueError("Calibration must match and test layouts must be disjoint from the original selection")
    inputs = {
        "controller": ROOT / "carl/icl/luna_intent_rollout.py",
        "transport": ROOT / "carl/icl/luna_intent_transport.py",
        "action_format": ROOT / "carl/icl/luna_action_format.py",
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
        "schema_version": 1,
        "variant": "strict_json_schema",
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
        "record_ids": [row["record_id"] for row in records],
        "record_order": [{key: row[key] for key in ("record_id", "phase", "family", "layout_id", "context_id")}
                         for row in records],
        "records_sha256": digest(records_path),
        "data_manifest_sha256": digest(data_manifest_path),
        "original_plan_sha256": digest(original_plan_path),
        "code_sha256": {name: digest(path) for name, path in inputs.items()},
        "policy_prompt": POLICY_PROMPT,
        "policy_prompt_sha256": hashlib.sha256(POLICY_PROMPT.encode()).hexdigest(),
        "budget_config_sha256": digest(config_path),
        "budget_ceiling_usd": "2.00 shared with the original variant",
        "audit_run": "..",
        "audit_tag_prefix": TAG_PREFIX,
        "future_data_policy": "Recorded later boards, actions and offsets never enter the controller prompt.",
        "selection": "Identical calibration intents; test layouts disjoint from the entire original selection.",
        "adaptation": "Strict JSON schema was selected after observing format failures on calibration examples.",
        "interpretation": "The test cohorts differ between variants; comparisons are exploratory, not paired.",
    }
    plan_path = run_dir / "experiment_plan.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text(encoding="utf-8"))
        if saved != plan:
            raise ValueError("Frozen experiment inputs, code or configuration have changed")
        return saved
    atomic_json(plan_path, plan)
    return plan


def make_transport(campaign: Path) -> TaggedTransport:
    key = dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("The project .env does not contain an OpenAI API credential")
    client = OpenAI(api_key=key.strip(), timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    transport = LunaIntentTransport(campaign / "luna_intent_budget.json", timeout_seconds=REQUEST_TIMEOUT_SECONDS,
                                    max_output_tokens=MAX_OUTPUT_TOKENS, client=client, structured_actions=True)
    return TaggedTransport(transport)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "pilot", "experiment"))
    parser.add_argument("--campaign-run", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_run.resolve()
    run_dir = campaign / "structured"
    create_plan(campaign, run_dir)
    if args.phase == "prepare":
        print(f"Frozen structured experiment plan: {run_dir / 'experiment_plan.json'}")
        return 0
    transport = make_transport(campaign)
    return pilot(run_dir, transport) if args.phase == "pilot" else experiment(run_dir, transport)


if __name__ == "__main__":
    sys.exit(main())
