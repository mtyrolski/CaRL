#!/usr/bin/env python3
"""Prepare or run budgeted Luna intent rollouts on a frozen JSONL selection.

Without ``--execute`` this command only checks and describes the input. The
budget directory can be shared across separately recorded calibration and test
stages. A failed or uncertain API request stops the stage without retrying it.
"""

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
from decimal import InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from dotenv import dotenv_values
from openai import OpenAI

from carl.icl.boards import from_rows
from carl.icl.boards import validate_subgoal
from carl.icl.calibration import weighted_percentiles
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import validate_prediction
from carl.icl.luna_intent_rollout import POLICY_PROMPT
from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_rollout import rollout
from carl.icl.luna_intent_transport import LunaIntentTransport
from carl.icl.luna_intent_transport import LunaRequestUncertain
from carl.icl.luna_intent_transport import create_luna_budget_config
from scripts.run_luna_final_response import FinalMessageTransport
from scripts.run_luna_intent_experiment import append_jsonl
from scripts.run_luna_intent_experiment import atomic_json
from scripts.run_luna_intent_experiment import digest
from scripts.run_luna_intent_experiment import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_RUN = ROOT / "analysis/luna_intent_20261114T060730Z"
HISTORICAL_LEDGER = HISTORICAL_RUN / "luna_intent_ledger.sqlite3"
MAX_OUTPUT_TOKENS = 256
REQUIRED_FIELDS = frozenset({"record_id", "context_id", "split", "phase", "family", "layout_id",
                             "start_observation", "task_goal", "intent"})
MEASURED_STATUSES = frozenset({RolloutStatus.ACHIEVED, RolloutStatus.ACTION_LIMIT, RolloutStatus.TASK_SOLVED,
                               RolloutStatus.PRESERVATION_VIOLATED, RolloutStatus.INVALID_RESPONSE})


class MeasurementStatus(StrEnum):
    """Distinguish missing measurements from unsuccessful policy attempts."""

    NOT_STARTED = "not_started"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    INVALID_INTENT = "invalid_intent"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    OUTPUT_LIMIT = "output_limit"
    UNASSESSED = "unassessed"


def sha256_json(value: Any) -> str:
    """Hash a canonical JSON value for controller and plan identities."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ModelIdentityGuard:
    """Stop if the service-reported model changes within a shared campaign.

    An unchanged alias does not prove the provider's weights are unchanged.
    Saved raw responses remain authoritative if a process is interrupted.
    """

    def __init__(self, transport: LunaIntentTransport) -> None:
        self.transport = transport
        self.ledger = transport.ledger
        self.run_dir = transport.run_dir
        self.audit_dir = transport.audit_dir

    def _recorded_identity(self) -> tuple[str | None, str | None]:
        identities: set[str] = set()
        fingerprints: set[str] = set()
        identity_path = self.run_dir / "response_model.json"

        def add_fingerprint(value: Any) -> None:
            if value is None or value == "":
                return
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Invalid response fingerprint")
            fingerprints.add(value)

        try:
            if identity_path.exists():
                saved = json.loads(identity_path.read_text(encoding="utf-8"))
                model = saved["model"]
                if not isinstance(model, str) or not model:
                    raise ValueError("Invalid saved model identity")
                identities.add(model)
                # Older campaign identity files have no fingerprint field.
                add_fingerprint(saved.get("system_fingerprint"))
            for path in self.audit_dir.glob("*.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") != "settled":
                    continue  # The underlying transport blocks unresolved audits.
                response = record.get("response")
                model = response.get("model") if isinstance(response, dict) else None
                if not isinstance(model, str) or not model:
                    raise ValueError("A settled response has no model identity")
                identities.add(model)
                add_fingerprint(response.get("system_fingerprint"))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise LunaRequestUncertain("Cannot verify recorded response model identities") from error
        if len(identities) > 1:
            raise LunaRequestUncertain("Service-reported model identity changed; campaign stopped")
        if len(fingerprints) > 1:
            raise LunaRequestUncertain("Service-reported model fingerprint changed; campaign stopped")
        return next(iter(identities), None), next(iter(fingerprints), None)

    def create(self, prompt: str, *, tag: str) -> Any:
        # This separate lock covers both identity checks and the physical call;
        # LunaIntentTransport uses its own request/ledger lock internally.
        with (self.run_dir / "response_model.lock").open("a", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                expected, expected_fingerprint = self._recorded_identity()
                response = self.transport.create(prompt, tag=tag)
                model = getattr(response, "model", None)
                fingerprint = getattr(response, "system_fingerprint", None)
                if not isinstance(model, str) or not model or (expected is not None and model != expected):
                    raise LunaRequestUncertain("Service-reported model identity changed; campaign stopped")
                if fingerprint == "":
                    fingerprint = None
                if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint.strip()):
                    raise LunaRequestUncertain("Service-reported model fingerprint is invalid; campaign stopped")
                if expected_fingerprint is not None and fingerprint is not None and fingerprint != expected_fingerprint:
                    raise LunaRequestUncertain("Service-reported model fingerprint changed; campaign stopped")
                # Check the audit too, including the just-settled response.
                recorded, recorded_fingerprint = self._recorded_identity()
                if recorded not in (None, model):
                    raise LunaRequestUncertain("Response model disagrees with saved audits; campaign stopped")
                if fingerprint is not None and recorded_fingerprint != fingerprint:
                    raise LunaRequestUncertain("Response fingerprint disagrees with saved audits; campaign stopped")
                identity_path = self.run_dir / "response_model.json"
                if not identity_path.exists():
                    atomic_json(identity_path, {"model": model, "system_fingerprint": fingerprint,
                                                "note": "An alias or missing fingerprint does not prove frozen weights."})
                return response
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def validate_records(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Freeze exact context membership and validate the current-board contract."""
    if not records:
        raise ValueError("Input must contain at least one record")
    contexts: dict[str, list[str]] = defaultdict(list)
    context_inputs: dict[str, tuple[Any, ...]] = {}
    context_intents: dict[str, set[str]] = defaultdict(set)
    record_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not REQUIRED_FIELDS <= record.keys():
            raise ValueError("Each record must contain all required fields")
        for key in ("record_id", "context_id", "split", "family", "layout_id", "task_goal"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        if record["phase"] not in {"calibration", "test"}:
            raise ValueError("phase must be calibration or test")
        identifier, context = record["record_id"], record["context_id"]
        if identifier in record_ids:
            raise ValueError(f"Duplicate record_id: {identifier}")
        record_ids.add(identifier)
        board = from_rows(record["start_observation"])
        validate_subgoal(board, board)
        if board.shape[0] != board.shape[1]:
            raise ValueError("The Sokoban environment requires a square board")
        signature = (record["phase"], record["split"], record["family"], record["layout_id"],
                     record["task_goal"], tuple(record["start_observation"]))
        if context in context_inputs and signature != context_inputs[context]:
            raise ValueError(f"Context {context} does not have one fixed board, task and split")
        context_inputs[context] = signature
        intent = record["intent"]
        if record["phase"] == "calibration" and not validate_prediction(intent, record["start_observation"]):
            key = sha256_json({name: sorted(json.dumps(item, sort_keys=True) for item in intent[name])
                               for name in ("effects", "preserve")})
            if key in context_intents[context]:
                raise ValueError(f"Context {context} repeats a calibration intent")
            context_intents[context].add(key)
        contexts[context].append(identifier)
    for record in records:
        expected = record.get("expected_candidates")
        if expected is not None and (not isinstance(expected, int) or isinstance(expected, bool)
                                     or expected != len(contexts[record["context_id"]])):
            raise ValueError("expected_candidates must equal the frozen context membership")
    return dict(contexts)


def controller_config(action_cap: int, timeout: float) -> dict[str, Any]:
    """Describe the fixed controller identically across later study stages."""
    paths = {
        "rollout": ROOT / "carl/icl/luna_intent_rollout.py",
        "transport": ROOT / "carl/icl/luna_intent_transport.py",
        "response_parser": ROOT / "carl/icl/luna_response.py",
        "action_format": ROOT / "carl/icl/luna_action_format.py",
        "boards": ROOT / "carl/icl/boards.py",
        "predicates": ROOT / "carl/icl/intent_data.py",
        "formatting": ROOT / "carl/icl/luna_components/formatting.py",
        "data_source": ROOT / "carl/icl/data_sources/sokoban.py",
        "types": ROOT / "carl/icl/types.py",
        "environment": ROOT / "carl/environment/sokoban/env.py",
        "environment_core": ROOT / "carl/environment/sokoban/core.py",
        "environment_tokenizer": ROOT / "carl/environment/sokoban/tokenizer.py",
        "wrapper": ROOT / "scripts/run_luna_final_response.py",
        "runner": Path(__file__).resolve(),
    }
    return {
        "model": "gpt-6-luna", "model_snapshot": None, "service_tier": "default",
        "reasoning_effort": "none", "temperature": 0, "action_cap": action_cap,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "request_timeout_seconds": timeout,
        "structured_actions": True, "policy_prompt": POLICY_PROMPT,
        "incomplete_response_policy": "output_limit is not a completed behavioural attempt",
        "policy_prompt_sha256": hashlib.sha256(POLICY_PROMPT.encode()).hexdigest(),
        "code_sha256": {name: digest(path) for name, path in paths.items()},
    }


def history_summary() -> dict[str, Any]:
    """Read historical reservations without changing or reconciling the ledger."""
    result: dict[str, Any] = {"ledger_path": str(HISTORICAL_LEDGER), "status": "unavailable",
                              "unknown_attempts": None, "unknown_reserved_usd": None,
                              "included_in_new_limit": False}
    if not HISTORICAL_LEDGER.is_file():
        return result
    try:
        with sqlite3.connect(f"{HISTORICAL_LEDGER.resolve().as_uri()}?mode=ro", uri=True) as database:
            count, amount = database.execute(
                "SELECT COUNT(*), COALESCE(SUM(reserved_nanos), 0) FROM attempts WHERE status='unknown'"
            ).fetchone()
    except sqlite3.Error:
        return result
    result.update(status="recorded", unknown_attempts=count, unknown_reserved_usd=amount / 1_000_000_000)
    return result


def reconcile_history(ledger: Path) -> dict[str, Any]:
    """Account for every outstanding historical reservation at its full bound.

    Read and validate the original ledger and audit files without modifying or
    declaring any unknown request free. Subtracting the held amount from the
    new campaign ceiling is deliberately conservative.
    """
    if not ledger.is_file():
        raise ValueError(f"Historical ledger is unavailable: {ledger}")
    config_path = ledger.parent / "luna_intent_budget.json"
    audit_dir = ledger.parent / "luna_requests"
    if not config_path.is_file() or not audit_dir.is_dir():
        raise ValueError("Historical budget configuration or request audits are missing")
    try:
        with sqlite3.connect(f"{ledger.resolve().as_uri()}?mode=ro", uri=True) as database:
            database.execute("BEGIN")
            metadata = dict(database.execute("SELECT key,value FROM metadata").fetchall())
            rows = database.execute(
                "SELECT id,status,reserved_nanos,charged_nanos,request_hash FROM attempts "
                "WHERE status != 'settled' ORDER BY id"
            ).fetchall()
        if metadata.get("config_hash") != digest(config_path):
            raise ValueError("Historical ledger configuration hash differs")
        outstanding = []
        for identifier, status, reserved, charged, request_hash in rows:
            if (not isinstance(identifier, str) or len(identifier) != 32
                    or any(c not in "0123456789abcdef" for c in identifier)
                    or status not in {"unknown", "reserved"} or charged is not None
                    or isinstance(reserved, bool) or not isinstance(reserved, int) or reserved <= 0):
                raise ValueError("Historical outstanding reservation is invalid")
            path = audit_dir / f"{identifier}.json"
            audit = json.loads(path.read_text(encoding="utf-8"))
            payload = audit.get("request")
            request_digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if (not isinstance(payload, dict) or audit.get("reservation_id") != identifier
                    or audit.get("status") != status or audit.get("request_sha256") != request_hash
                    or request_digest != request_hash):
                raise ValueError("Historical reservation and request audit differ")
            outstanding.append({"reservation_id": identifier, "status": status, "reserved_nanos": reserved,
                                "audit_sha256": digest(path)})
    except (OSError, sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot verify historical reservations; paid requests are blocked") from error
    total = sum(item["reserved_nanos"] for item in outstanding)
    return {"status": "held_at_full_reservation", "ledger_path": str(ledger.resolve()),
            "ledger_sha256": digest(ledger), "config_sha256": digest(config_path),
            "outstanding": outstanding, "reserved_nanos": total,
            "reserved_usd": format(Decimal(total) / Decimal(1_000_000_000), "f")}


def make_plan(records: list[dict[str, Any]], input_path: Path, budget_dir: Path, output_dir: Path,
              attempts: int, action_cap: int, timeout: float, budget: str, *,
              migration_manifest: Path | None = None, accepted_unknown_id: str | None = None) -> dict[str, Any]:
    """Describe the exact intended work without making a request or creating files."""
    contexts = validate_records(records)
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("attempts must be a positive integer")
    if not isinstance(action_cap, int) or isinstance(action_cap, bool) or not 1 <= action_cap <= 16:
        raise ValueError("action cap must be between 1 and 16")
    if not 0 < timeout <= 600:
        raise ValueError("timeout must be positive and at most 600 seconds")
    try:
        amount = Decimal(budget)
    except InvalidOperation as error:
        raise ValueError("budget must be a decimal amount") from error
    if not amount.is_finite() or not Decimal(0) < amount <= Decimal(2):
        raise ValueError("The Luna transport supports a budget up to $2")
    for path in (budget_dir.resolve(), output_dir.resolve()):
        if path == HISTORICAL_RUN.resolve() or path.is_relative_to(HISTORICAL_RUN.resolve()):
            raise ValueError("The historical campaign directory cannot be reused")
    if output_dir.resolve() == budget_dir.resolve():
        raise ValueError("Output and shared budget directories must be different")
    history = reconcile_history(HISTORICAL_LEDGER)
    migration = None
    if migration_manifest is not None:
        from scripts.migrate_luna_campaign import verify

        migration = verify(migration_manifest, budget_dir, HISTORICAL_LEDGER)
        if amount != Decimal(2) or accepted_unknown_id != migration["accepted_unknown_id"]:
            raise ValueError("Migrated budget requires its exact retained unknown and $2 campaign ceiling")
        available = amount
    else:
        if accepted_unknown_id is not None:
            raise ValueError("Accepting an unknown requires a verified migration manifest")
        available = amount - Decimal(history["reserved_usd"])
    if available <= 0:
        raise ValueError("Historical outstanding reservations leave no budget for a new campaign")
    config = controller_config(action_cap, timeout)
    return {
        "schema_version": 1,
        "input_path": str(input_path.resolve()), "input_sha256": digest(input_path),
        "budget_dir": str(budget_dir.resolve()), "output_dir": str(output_dir.resolve()),
        "budget_usd": format(available.normalize(), "f"),
        "campaign_ceiling_usd": format(amount.normalize(), "f"), "attempts_per_candidate": attempts,
        "contexts": contexts, "record_ids": [record["record_id"] for record in records],
        "controller": config, "controller_hash": sha256_json(config),
        "runner_sha256": digest(Path(__file__).resolve()),
        "budget_config_sha256": None,
        "historical_reservations": history,
        "migration_manifest": str(migration_manifest.resolve()) if migration_manifest else None,
        "migration_manifest_sha256": digest(migration_manifest) if migration_manifest else None,
        "accepted_unknown_id": accepted_unknown_id,
        "budget_note": ("This $2 ledger includes the complete carried pilot and future requests; "
                        "earlier campaign reservations remain separate." if migration else
                        "Historical outstanding reservations remain untouched and are held at their full bound "
                        "against the campaign ceiling; the new ledger enforces the remainder."),
        "repeats_note": "Each attempt makes new physical requests; temperature zero does not guarantee determinism.",
    }


def calibrated_records(records: list[dict[str, Any]], results: list[dict[str, Any]],
                       plan: dict[str, Any], started_attempts: set[tuple[str, int]] | None = None
                       ) -> list[dict[str, Any]]:
    """Attach measurements; label only contexts with every frozen candidate and attempt."""
    by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    planned_ids = set(plan["record_ids"])
    seen: set[tuple[str, int]] = set()
    for row in results:
        key = (row["record_id"], row["attempt"])
        if row["record_id"] not in planned_ids or key in seen or not 1 <= row["attempt"] <= plan["attempts_per_candidate"]:
            raise ValueError("Results contain an unexpected or duplicate attempt")
        seen.add(key)
        by_record[row["record_id"]].append(row)
    started_attempts = started_attempts or set()
    output = []
    for record in records:
        identifier, context = record["record_id"], record["context_id"]
        rows = by_record[identifier]
        measured = [row for row in rows if row["result"]["status"] in MEASURED_STATUSES
                    and row["result"]["model_calls"] > 0]
        attempts = [{"status": row["result"]["status"],
                     "primitive_actions_attempted": len(row["result"]["actions"])}
                    for row in sorted(measured, key=lambda item: item["attempt"])]
        statuses = {row["result"]["status"] for row in rows}
        if RolloutStatus.API_ERROR in statuses:
            status = MeasurementStatus.INFRASTRUCTURE_ERROR
        elif RolloutStatus.OUTPUT_LIMIT in statuses:
            status = MeasurementStatus.OUTPUT_LIMIT
        elif RolloutStatus.INVALID_INTENT in statuses:
            status = MeasurementStatus.INVALID_INTENT
        elif any(row["result"]["model_calls"] == 0 for row in rows):
            status = MeasurementStatus.UNASSESSED
        elif len(measured) == plan["attempts_per_candidate"]:
            status = MeasurementStatus.COMPLETE
        elif not rows and not any(key[0] == identifier for key in started_attempts):
            status = MeasurementStatus.NOT_STARTED
        else:
            status = MeasurementStatus.INCOMPLETE
        output.append({**record, "policy_measurements": {
            "controller_hash": plan["controller_hash"], "action_cap": plan["controller"]["action_cap"],
            "expected_candidates": len(plan["contexts"][context]), "complete": False,
            "attempts": attempts, "success_fraction": None, "mean_success_actions": None,
            "u": None, "weight": None, "status": status,
        }})
    by_id = {record["record_id"]: record for record in output}
    for context, candidate_ids in plan["contexts"].items():
        actual = {record["record_id"] for record in output if record["context_id"] == context}
        if actual != set(candidate_ids) or len(actual) != len(candidate_ids):
            raise ValueError("Output does not match frozen candidate identities")
        candidates = [by_id[identifier] for identifier in candidate_ids]
        complete = all(row["policy_measurements"]["status"] == MeasurementStatus.COMPLETE for row in candidates)
        if not complete:
            continue
        successes = []
        for row in candidates:
            measurement = row["policy_measurements"]
            measurement["complete"] = True
            durations = [attempt["primitive_actions_attempted"] for attempt in measurement["attempts"]
                         if attempt["status"] == RolloutStatus.ACHIEVED]
            measurement["success_fraction"] = len(durations) / plan["attempts_per_candidate"]
            if durations:
                measurement["mean_success_actions"] = sum(durations) / len(durations)
                successes.append((measurement, len(durations)))
        if successes and candidates[0]["phase"] == "calibration":
            weights, quantiles = weighted_percentiles(
                [item[0]["mean_success_actions"] for item in successes], [item[1] for item in successes]
            )
            for (measurement, _), weight, quantile in zip(successes, weights, quantiles, strict=True):
                measurement["weight"] = float(weight)
                measurement["u"] = float(quantile)
    return output


def write_calibrated(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write records, including incomplete and unstarted cases."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_outputs(records: list[dict[str, Any]], results: list[dict[str, Any]],
                   plan: dict[str, Any], output_dir: Path, state: str) -> None:
    """Persist a complete accounting of completed, missing and invalid cases."""
    starts = read_jsonl(output_dir / "events.jsonl")
    started = {(event["record_id"], event["attempt"]) for event in starts if event["type"] == "attempt_started"}
    finished = {(row["record_id"], row["attempt"]) for row in results}
    calibrated = calibrated_records(records, results, plan, started)
    write_calibrated(output_dir / "calibrated.jsonl", calibrated)
    planned = len(records) * plan["attempts_per_candidate"]
    counts = Counter(row["result"]["status"] for row in results)
    atomic_json(output_dir / "status.json", {
        "state": state, "planned_attempts": planned, "saved_attempts": len(results),
        "started_without_saved_result": len(started - finished), "not_started": planned - len(started | finished),
        "result_statuses": dict(counts),
        "record_statuses": dict(Counter(row["policy_measurements"]["status"] for row in calibrated)),
    })


def make_transport(budget_dir: Path, output_dir: Path, timeout: float, budget: str,
                   accepted_unknown_id: str | None = None) -> FinalMessageTransport:
    """Bind calls to the project credential and one shared immutable ledger."""
    key = dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    if not key or not key.strip():
        raise ValueError("The project .env does not contain an OpenAI API credential")
    config = create_luna_budget_config(budget_dir, budget)
    client = OpenAI(api_key=key.strip(), timeout=timeout, max_retries=0)
    transport = LunaIntentTransport(config, timeout_seconds=timeout, max_output_tokens=MAX_OUTPUT_TOKENS,
                                    client=client, structured_actions=True, accepted_unknown_id=accepted_unknown_id)
    guarded = ModelIdentityGuard(transport)
    return FinalMessageTransport(cast(LunaIntentTransport, guarded), output_dir, classify_incomplete=True)


def execute(records: list[dict[str, Any]], plan: dict[str, Any], transport: Any, output_dir: Path) -> int:
    """Execute distinct attempts, saving every stage boundary and stopping on API errors."""
    results: list[dict[str, Any]] = []
    update_outputs(records, results, plan, output_dir, "running")
    try:
        for record in records:
            for attempt in range(1, plan["attempts_per_candidate"] + 1):
                identifier = record["record_id"]
                tag = f"{identifier}:attempt:{attempt}"
                append_jsonl(output_dir / "events.jsonl", {"type": "attempt_started", "record_id": identifier,
                                                            "attempt": attempt})
                # Publish the started attempt before the first physical request,
                # so a signal cannot turn it into a claimed unstarted case.
                update_outputs(records, results, plan, output_dir, "running")
                started = time.monotonic()

                def event(item: dict[str, Any]) -> None:
                    append_jsonl(output_dir / "events.jsonl", {"record_id": identifier, "attempt": attempt, **item})

                state = from_rows(record["start_observation"])
                outcome = rollout(environment(state), state, record["intent"],
                                  lambda prompt: transport.create(prompt, tag=tag).output_text,
                                  cap=plan["controller"]["action_cap"], task_goal=record["task_goal"],
                                  on_event=event)
                result = {"record_id": identifier, "context_id": record["context_id"], "attempt": attempt,
                          "controller_hash": plan["controller_hash"], "duration_seconds": time.monotonic() - started,
                          "result": asdict(outcome)}
                append_jsonl(output_dir / "results.jsonl", result)
                results.append(result)
                if outcome.status == RolloutStatus.API_ERROR:
                    update_outputs(records, results, plan, output_dir, "stopped_api_error")
                    return 1
                update_outputs(records, results, plan, output_dir, "running")
        update_outputs(records, results, plan, output_dir, "complete")
        return 0
    except BaseException:
        update_outputs(records, results, plan, output_dir, "interrupted")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Frozen calibration or test records as JSONL")
    parser.add_argument("--budget-dir", type=Path, required=True, help="Shared new Luna budget and ledger directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this stage's outputs")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--action-cap", type=int, default=16)
    parser.add_argument("--budget-usd", default="2.00")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--migration-manifest", type=Path, help="Verified carried-pilot budget manifest")
    parser.add_argument("--accepted-unknown-id", help="Retain this one audited unknown at its full reservation")
    parser.add_argument("--execute", action="store_true", help="Permit paid API requests")
    args = parser.parse_args(argv)
    records = read_jsonl(args.input)
    plan = make_plan(records, args.input, args.budget_dir, args.output_dir,
                     args.attempts, args.action_cap, args.timeout, args.budget_usd,
                     migration_manifest=args.migration_manifest, accepted_unknown_id=args.accepted_unknown_id)
    if not args.execute:
        print(json.dumps({"mode": "plan_only", **plan}, ensure_ascii=False, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output_dir / "plan.json", plan)
    update_outputs(records, [], plan, args.output_dir, "prepared")
    try:
        transport = make_transport(args.budget_dir, args.output_dir, args.timeout, plan["budget_usd"],
                                   plan["accepted_unknown_id"])
        plan["budget_config_sha256"] = digest(args.budget_dir / "luna_intent_budget.json")
        atomic_json(args.output_dir / "plan.json", plan)
    except BaseException:
        update_outputs(records, [], plan, args.output_dir, "setup_error")
        raise
    return execute(records, plan, transport, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
