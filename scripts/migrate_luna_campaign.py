#!/usr/bin/env python3
"""Carry a stopped Luna pilot into one new, full-reservation campaign ledger.

The old files are read only. The new $2 ceiling covers all carried pilot
attempts and all subsequent requests; pre-campaign unknowns remain separate.
No model or API is called by this migration.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from decimal import ROUND_CEILING
from decimal import Decimal
from pathlib import Path
from typing import Any

from carl.icl.luna_action_format import action_text_format
from carl.icl.luna_intent_transport import CONFIG_NAME
from carl.icl.luna_intent_transport import LEDGER_NAME
from carl.icl.luna_intent_transport import MODEL
from carl.icl.luna_intent_transport import MODEL_PRICE
from scripts import intent_experiment_supervisor as supervisor
from scripts.run_luna_intent_experiment import digest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "migration_manifest.json"
CLAIM_NAME = "budget_migration_claim.json"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rows(database: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    if table not in {"attempts", "events"}:
        raise ValueError("Unexpected ledger table")
    database.row_factory = sqlite3.Row
    return [dict(row) for row in database.execute(f"SELECT * FROM {table} ORDER BY id")]


def _read_ledger(path: Path) -> tuple[dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as database:
        metadata = dict(database.execute("SELECT key,value FROM metadata").fetchall())
        return metadata, _rows(database, "attempts"), _rows(database, "events")


def _source_config(source: Path) -> dict[str, Any]:
    config = _json(source / CONFIG_NAME)
    if (config.get("campaign_id") != "luna_intent_experiment" or config.get("version") != 1
            or Path(config.get("ledger_path", "")).resolve() != (source / LEDGER_NAME).resolve()
            or config.get("models") != {MODEL: MODEL_PRICE}
            or set(config.get("stage_limits_usd", {})) != {"experiment"}):
        raise ValueError("Source budget configuration is not the expected Luna campaign")
    return config


def _pilot_finished(source: Path, stage_name: str) -> None:
    state = _json(source.parent / "run_manifest.json")
    stages = [stage for stage in state.get("stages", []) if stage.get("name") == stage_name]
    if len(stages) != 1:
        raise ValueError("Pilot stage is missing or ambiguous")
    stage = stages[0]
    command = stage.get("command", [])
    if (stage.get("status") not in {"completed", "failed", "timeout", "deadline_stopped", "lost_or_exited"}
            or not isinstance(stage.get("pid"), int) or "--budget-dir" not in command
            or command.index("--budget-dir") + 1 >= len(command)
            or (ROOT / command[command.index("--budget-dir") + 1]).resolve() != source.resolve()):
        raise ValueError("Pilot stage is not finished or does not own the source budget")
    if state.get("boot_id") == supervisor.boot_id() and supervisor.group_has_live_members(stage["pid"]):
        raise ValueError("The registered pilot process group is still running")


def _sealed_source(config: dict[str, Any], metadata: dict[str, str], rows: list[dict[str, Any]]) -> tuple[str, int]:
    unknown = [row for row in rows if row["status"] != "settled"]
    if len(unknown) != 1 or unknown[0]["status"] != "unknown" or unknown[0]["charged_nanos"] is not None:
        raise ValueError("Exactly one outstanding pilot unknown is required")
    price = config["models"][MODEL]
    input_rate = max(Decimal(str(price[name])) for name in
                     ("input_usd_per_million", "cached_input_usd_per_million",
                      "cache_write_30m_usd_per_million", "cache_write_12h_usd_per_million"))
    # Even the smallest allowed output cap of one token requires this full-context reservation.
    minimum = ((input_rate * 2 * price["max_input_tokens"]
                + Decimal(str(price["output_usd_per_million"])) * Decimal("1.5"))
               * Decimal(str(price["regional_multiplier"])) * Decimal(1000))
    minimum_nanos = int(minimum.to_integral_value(rounding=ROUND_CEILING))
    effective = min(int(metadata["effective_limit_nanos"]),
                    int(Decimal(str(config["limit_usd"])) * 1_000_000_000),
                    int(Decimal(str(config["stage_limits_usd"]["experiment"])) * 1_000_000_000))
    used = sum(row["reserved_nanos"] if row["charged_nanos"] is None else row["charged_nanos"] for row in rows)
    if effective - used >= minimum_nanos:
        raise ValueError("Source ledger could still reserve another full-context request")
    return unknown[0]["id"], used


def _audits(source: Path, attempts: list[dict[str, Any]]) -> dict[str, str]:
    directory = source / "luna_requests"
    paths = {path.stem: path for path in directory.glob("*.json")}
    if set(paths) != {row["id"] for row in attempts}:
        raise ValueError("Source audit files do not match ledger attempts")
    hashes = {}
    for row in attempts:
        path = paths[row["id"]]
        record = _json(path)
        payload_hash = hashlib.sha256(json.dumps(record.get("request"), sort_keys=True,
                                                 ensure_ascii=False).encode()).hexdigest()
        if (record.get("reservation_id") != row["id"] or record.get("status") != row["status"]
                or record.get("request_sha256") != row["request_hash"] or payload_hash != row["request_hash"]):
            raise ValueError("Source request audit does not match its ledger attempt")
        if row["status"] == "unknown":
            request = record["request"]
            if (record.get("error_diagnostics", {}).get("http_status") != 400
                    or request.get("model") != MODEL or request.get("reasoning") != {"effort": "none"}
                    or request.get("temperature") != 0 or request.get("service_tier") != "default"
                    or request.get("tools") != [] or request.get("store") is not False
                    or request.get("text") != action_text_format() or request.get("max_output_tokens") != 256
                    or row["reserved_nanos"] != 577_711_200):
                raise ValueError("Retained pilot unknown is not the expected fully reserved request")
        elif record.get("response", {}).get("model") != MODEL:
            raise ValueError("A settled pilot response reports a different model")
        hashes[row["id"]] = digest(path)
    return hashes


def migrate(source: Path, target: Path, stage_name: str, legacy_ledger: Path) -> Path:
    """Create one verifiable carry-forward destination while holding the source lock."""
    from scripts.run_luna_lora_controller import reconcile_history

    source, target = source.resolve(), target.resolve()
    if target.exists() or target == source or target.is_relative_to(source):
        raise ValueError("Use a new destination outside the source budget directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    claim = source.parent / CLAIM_NAME
    with (source / "luna_intent.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("The source budget is still locked") from error
        _pilot_finished(source, stage_name)
        if claim.exists():
            raise ValueError("A migration claim already exists for this pilot")
        config = _source_config(source)
        source_ledger = source / LEDGER_NAME
        before = digest(source_ledger)
        metadata, attempts, events = _read_ledger(source_ledger)
        if metadata.get("config_hash") != digest(source / CONFIG_NAME):
            raise ValueError("Source ledger configuration hash differs")
        unknown_id, used = _sealed_source(config, metadata, attempts)
        audit_hashes = _audits(source, attempts)
        identity = _json(source / "response_model.json")
        if identity.get("model") != MODEL:
            raise ValueError("Source response identity does not match the Luna alias")
        legacy = reconcile_history(legacy_ledger)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            destination_ledger = staging / LEDGER_NAME
            with sqlite3.connect(f"{source_ledger.as_uri()}?mode=ro", uri=True) as origin:
                with sqlite3.connect(destination_ledger) as destination:
                    origin.backup(destination)
            new_config = {**config, "ledger_path": str(target / LEDGER_NAME), "limit_usd": "2",
                          "stage_limits_usd": {"experiment": "2"}}
            _atomic(staging / CONFIG_NAME, new_config)
            with sqlite3.connect(destination_ledger) as database:
                database.execute("UPDATE metadata SET value=? WHERE key='config_hash'", (digest(staging / CONFIG_NAME),))
                database.execute("UPDATE metadata SET value='2000000000' WHERE key='effective_limit_nanos'")
            copied_metadata, copied_attempts, copied_events = _read_ledger(destination_ledger)
            if copied_attempts != attempts or copied_events != events or digest(source_ledger) != before:
                raise ValueError("Source changed during ledger copy")
            (staging / "luna_requests").mkdir()
            for identifier, expected_hash in audit_hashes.items():
                original = source / "luna_requests" / f"{identifier}.json"
                copied = staging / "luna_requests" / original.name
                shutil.copyfile(original, copied)
                if digest(original) != expected_hash or digest(copied) != expected_hash:
                    raise ValueError("A source request audit changed during migration")
            shutil.copyfile(source / "response_model.json", staging / "response_model.json")
            if digest(staging / "response_model.json") != digest(source / "response_model.json"):
                raise ValueError("Response identity changed during migration")
            manifest = {"schema_version": 1, "source_budget": str(source), "target_budget": str(target),
                        "source_ledger_sha256": before, "source_config_sha256": digest(source / CONFIG_NAME),
                        "target_config_sha256": copied_metadata["config_hash"], "source_attempts": attempts,
                        "source_events": events, "source_audits_sha256": audit_hashes,
                        "source_response_model_sha256": digest(source / "response_model.json"),
                        "accepted_unknown_id": unknown_id, "carried_committed_nanos": used,
                        "previous_campaign_reservations": legacy,
                        "scope": "The new $2 limit includes the complete pilot and future calls; "
                                 "earlier campaign unknowns remain separate."}
            _atomic(staging / MANIFEST_NAME, manifest)
            # An exclusive, durable claim prevents a second $2 ledger from the same pilot.
            with claim.open("x", encoding="utf-8") as output:
                json.dump({"status": "preparing", "target_budget": str(target)}, output)
                output.flush()
                os.fsync(output.fileno())
            target.mkdir(exist_ok=False)
            for item in staging.iterdir():
                os.rename(item, target / item.name)
            _atomic(claim, {"status": "complete", "target_budget": str(target),
                            "manifest_sha256": digest(target / MANIFEST_NAME)})
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return target / MANIFEST_NAME


def verify(manifest_path: Path, target: Path, legacy_ledger: Path) -> dict[str, Any]:
    """Recheck the carried attempts and original files before each later stage."""
    from scripts.run_luna_lora_controller import reconcile_history

    manifest_path, target = manifest_path.resolve(), target.resolve()
    if manifest_path != target / MANIFEST_NAME:
        raise ValueError("Migration manifest must be inside the shared budget directory")
    manifest = _json(manifest_path)
    source = Path(manifest["source_budget"])
    claim = _json(source.parent / CLAIM_NAME)
    if (manifest.get("schema_version") != 1 or manifest.get("target_budget") != str(target)
            or claim.get("status") != "complete" or claim.get("target_budget") != str(target)
            or claim.get("manifest_sha256") != digest(manifest_path)
            or digest(source / LEDGER_NAME) != manifest["source_ledger_sha256"]
            or digest(source / CONFIG_NAME) != manifest["source_config_sha256"]
            or digest(target / CONFIG_NAME) != manifest["target_config_sha256"]
            or reconcile_history(legacy_ledger) != manifest["previous_campaign_reservations"]):
        raise ValueError("Migration claim, source or historical reservations changed")
    config = _source_config(source)
    source_metadata, source_attempts, source_events = _read_ledger(source / LEDGER_NAME)
    unknown_id, used = _sealed_source(config, source_metadata, source_attempts)
    if (unknown_id != manifest["accepted_unknown_id"] or used != manifest["carried_committed_nanos"]
            or source_attempts != manifest["source_attempts"] or source_events != manifest["source_events"]
            or _audits(source, source_attempts) != manifest["source_audits_sha256"]):
        raise ValueError("Original pilot no longer matches its migration manifest")
    target_config = _source_config(target)
    if (Decimal(str(target_config["limit_usd"])) != Decimal(2)
            or Decimal(str(target_config["stage_limits_usd"]["experiment"])) != Decimal(2)):
        raise ValueError("Migrated campaign ceiling must be $2")
    metadata, attempts, events = _read_ledger(target / LEDGER_NAME)
    if (metadata.get("config_hash") != digest(target / CONFIG_NAME)
            or int(metadata["effective_limit_nanos"]) > 2_000_000_000):
        raise ValueError("Migrated ledger configuration hash differs")
    by_id = {row["id"]: row for row in attempts}
    if any(by_id.get(row["id"]) != row for row in manifest["source_attempts"]):
        raise ValueError("A carried pilot attempt changed")
    by_event = {row["id"]: row for row in events}
    if any(by_event.get(row["id"]) != row for row in manifest["source_events"]):
        raise ValueError("A carried pilot event changed")
    for identifier, expected in manifest["source_audits_sha256"].items():
        if (digest(source / "luna_requests" / f"{identifier}.json") != expected
                or digest(target / "luna_requests" / f"{identifier}.json") != expected):
            raise ValueError("A carried request audit changed")
    if (digest(source / "response_model.json") != manifest["source_response_model_sha256"]
            or digest(target / "response_model.json") != manifest["source_response_model_sha256"]):
        raise ValueError("A carried response identity changed")
    return manifest


def main() -> None:
    from scripts.run_luna_lora_controller import HISTORICAL_LEDGER

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--pilot-stage", required=True)
    args = parser.parse_args()
    print(migrate(args.source, args.target, args.pilot_stage, HISTORICAL_LEDGER))


if __name__ == "__main__":
    main()
