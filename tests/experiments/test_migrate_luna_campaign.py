"""Offline checks for carrying a stopped pilot into one $2 ledger."""

import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from carl.icl.campaign_budget import CampaignLedger
from carl.icl.luna_action_format import action_text_format
from carl.icl.luna_intent_transport import LunaIntentTransport
from carl.icl.luna_intent_transport import create_luna_budget_config
from scripts import intent_experiment_supervisor as supervisor
from scripts import migrate_luna_campaign as migration
from scripts import run_luna_lora_controller as controller


@pytest.fixture
def source_campaign(tmp_path, monkeypatch):
    import carl.icl.campaign_budget as budget_module

    monkeypatch.setattr(budget_module, "dotenv_values", lambda _path: {})
    monkeypatch.delenv("CARL_CAMPAIGN_MAX_USD", raising=False)
    monkeypatch.setattr(supervisor, "group_has_live_members", lambda _group: False)
    monkeypatch.setattr(controller, "reconcile_history", lambda path: {
        "ledger_path": str(path), "reserved_nanos": 1_155_422_400, "reserved_usd": "1.1554224"
    })
    run = tmp_path / "run"
    run.mkdir()
    source = run / "budget"
    config = create_luna_budget_config(source, "0.8445776")
    ledger = CampaignLedger(config)
    audits = source / "luna_requests"
    audits.mkdir()
    payload = {"model": "gpt-6-luna", "input": "synthetic board", "max_output_tokens": 256,
               "reasoning": {"effort": "none"}, "temperature": 0, "service_tier": "default",
               "tools": [], "store": False, "text": action_text_format()}
    request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    first = ledger.reserve("experiment", "responses", payload, request_hash, tag="first")
    assert ledger.settle(first, {"input_tokens": 500, "output_tokens": 20}, payload)
    second = ledger.reserve("experiment", "responses", payload, request_hash, tag="second")
    ledger.mark_unknown(second)
    for identifier, status in ((first, "settled"), (second, "unknown")):
        record = {"reservation_id": identifier, "status": status, "request": payload,
                  "request_sha256": request_hash, "response": {"model": "gpt-6-luna"} if status == "settled" else None,
                  "error_diagnostics": {"http_status": 400} if status == "unknown" else None}
        (audits / f"{identifier}.json").write_text(json.dumps(record), encoding="utf-8")
    (source / "response_model.json").write_text(json.dumps({"model": "gpt-6-luna"}), encoding="utf-8")
    state = {"boot_id": supervisor.boot_id(), "stages": [{"name": "pilot", "status": "failed", "pid": 99999999,
              "command": ["python", "--budget-dir", str(source)]}]}
    (run / "run_manifest.json").write_text(json.dumps(state), encoding="utf-8")
    return source, tmp_path / "new-budget", second


def test_migration_preserves_pilot_and_enforces_one_shared_ceiling(source_campaign, tmp_path, monkeypatch):
    source, target, unknown_id = source_campaign
    before = (source / migration.LEDGER_NAME).read_bytes()
    manifest_path = migration.migrate(source, target, "pilot", tmp_path / "legacy")
    manifest = migration.verify(manifest_path, target, tmp_path / "legacy")
    assert manifest["accepted_unknown_id"] == unknown_id
    assert (source / migration.LEDGER_NAME).read_bytes() == before
    assert json.loads((target / migration.CONFIG_NAME).read_text())["limit_usd"] == "2"
    assert migration._read_ledger(source / migration.LEDGER_NAME)[1:] == migration._read_ledger(target / migration.LEDGER_NAME)[1:]
    with pytest.raises(ValueError, match="claim"):
        migration.migrate(source, tmp_path / "another-budget", "pilot", tmp_path / "legacy")

    monkeypatch.setattr(controller, "HISTORICAL_LEDGER", tmp_path / "legacy")
    record = {"record_id": "a", "context_id": "c", "split": "train", "phase": "calibration", "family": "test",
              "layout_id": "layout", "start_observation": ["#######", "#@$-.-#", "#-$-.-#", "#-----#",
                                                              "#-----#", "#-----#", "#######"],
              "task_goal": "Put boxes on goals.", "intent": {
                  "effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": []}}
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(record) + "\n")
    plan = controller.make_plan([record], input_path, target, tmp_path / "output", 1, 16, 45, "2",
                                migration_manifest=manifest_path, accepted_unknown_id=unknown_id)
    assert plan["budget_usd"] == "2" and plan["accepted_unknown_id"] == unknown_id
    with pytest.raises(ValueError, match="exact retained unknown"):
        controller.make_plan([record], input_path, target, tmp_path / "output", 1, 16, 45, "2",
                             migration_manifest=manifest_path, accepted_unknown_id="wrong")
    with pytest.raises(ValueError, match="requires a verified migration"):
        controller.make_plan([record], input_path, target, tmp_path / "output", 1, 16, 45, "2",
                             accepted_unknown_id=unknown_id)

    class FakeClient:
        def with_options(self, **_kwargs):
            return self

        responses = SimpleNamespace(create=lambda **_kwargs: SimpleNamespace(
            output_text='{"action":"UP"}', _request_id="req_local1",
            model_dump=lambda **_kwargs: {"model": "gpt-6-luna", "service_tier": "default",
                                          "usage": {"input_tokens": 500, "output_tokens": 20}}))

    import carl.icl.luna_intent_transport as luna
    monkeypatch.setattr(luna, "_pace_requests", lambda _run: None)
    transport = LunaIntentTransport(target / migration.CONFIG_NAME, client=FakeClient(), structured_actions=True,
                                    max_output_tokens=256, accepted_unknown_id=unknown_id)
    transport.create("new synthetic board")
    after = migration.verify(manifest_path, target, tmp_path / "legacy")
    assert after["accepted_unknown_id"] == unknown_id
    assert (source / migration.LEDGER_NAME).read_bytes() == before
    with sqlite3.connect(target / migration.LEDGER_NAME) as database:
        values = database.execute("SELECT status,COUNT(*) FROM attempts GROUP BY status").fetchall()
    assert dict(values) == {"settled": 2, "unknown": 1}


@pytest.mark.parametrize("kind", ["running", "unsealed", "tampered_audit", "missing_audit"])
def test_migration_fails_closed(source_campaign, tmp_path, kind):
    source, target, _ = source_campaign
    if kind == "running":
        path = source.parent / "run_manifest.json"
        state = json.loads(path.read_text())
        state["stages"][0]["status"] = "running"
        path.write_text(json.dumps(state))
    elif kind == "unsealed":
        with sqlite3.connect(source / migration.LEDGER_NAME) as database:
            database.execute("UPDATE metadata SET value='2000000000' WHERE key='effective_limit_nanos'")
        path = source / migration.CONFIG_NAME
        cfg = json.loads(path.read_text())
        cfg["limit_usd"] = "2"
        cfg["stage_limits_usd"] = {"experiment": "2"}
        path.write_text(json.dumps(cfg))
        with sqlite3.connect(source / migration.LEDGER_NAME) as database:
            database.execute("UPDATE metadata SET value=? WHERE key='config_hash'", (migration.digest(path),))
    else:
        path = next((source / "luna_requests").glob("*.json"))
        if kind == "missing_audit":
            path.unlink()
        else:
            path.write_text("{}")
    with pytest.raises(ValueError):
        migration.migrate(source, target, "pilot", tmp_path / "legacy")
    assert not target.exists()


def test_carried_audit_tampering_blocks_verification(source_campaign, tmp_path):
    source, target, unknown_id = source_campaign
    path = migration.migrate(source, target, "pilot", tmp_path / "legacy")
    (target / "luna_requests" / f"{unknown_id}.json").write_text("{}")
    with pytest.raises(ValueError, match="audit changed"):
        migration.verify(path, target, tmp_path / "legacy")
