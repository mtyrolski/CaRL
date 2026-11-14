"""A single acknowledged unknown remains fully reserved in the same ledger."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from carl.icl.campaign_budget import CampaignLedger
from carl.icl.luna_intent_transport import LunaIntentTransport, LunaRequestUncertain, create_luna_budget_config


class FakeResponse:
    output_text = '{"action":"UP"}'
    _request_id = "req_test123"

    def model_dump(self, *, mode):
        assert mode == "json"
        return {"id": "resp_test", "model": "gpt-6-luna", "service_tier": "default", "status": "completed",
                "usage": {"input_tokens": 300, "output_tokens": 7}, "output": []}


class BadRequest(Exception):
    status_code = 400
    code = "invalid_request_error"
    type = "invalid_request_error"
    request_id = "req_test123"


class FakeClient:
    def __init__(self, callback):
        self.responses = SimpleNamespace(create=callback)

    def with_options(self, **_options):
        return self


@pytest.fixture
def config(tmp_path, monkeypatch):
    import carl.icl.campaign_budget as campaign_budget
    monkeypatch.setattr(campaign_budget, "dotenv_values", lambda _path: {})
    monkeypatch.delenv("CARL_CAMPAIGN_MAX_USD", raising=False)
    return create_luna_budget_config(tmp_path)


def create_unknown(config):
    def fail(**_payload):
        raise BadRequest("private details")

    transport = LunaIntentTransport(config, client=FakeClient(fail), structured_actions=True, max_output_tokens=256)
    with pytest.raises(LunaRequestUncertain):
        transport.create("board", tag="expanded:case")
    path = next((config.parent / "luna_requests").glob("*.json"))
    return path.stem


def attempt_rows(config):
    path = config.parent / "luna_intent_ledger.sqlite3"
    with sqlite3.connect(path) as database:
        return database.execute("SELECT id,status,reserved_nanos,charged_nanos,request_hash FROM attempts ORDER BY id").fetchall()


def sql(config, statement, arguments=()):
    with sqlite3.connect(config.parent / "luna_intent_ledger.sqlite3") as database:
        database.execute(statement, arguments)


def test_exact_unknown_remains_unchanged_and_fully_counted(config):
    identifier = create_unknown(config)
    original = attempt_rows(config)[0]
    assert original[1:4] == ("unknown", 577711200, None)
    calls = []

    def success(**payload):
        calls.append(payload)
        return FakeResponse()

    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=FakeClient(success)).create("default still blocks")
    transport = LunaIntentTransport(config, client=FakeClient(success), structured_actions=True,
                                    max_output_tokens=256, accepted_unknown_id=identifier)
    assert transport.create("new context").output_text == '{"action":"UP"}'
    assert len(calls) == 1
    rows = attempt_rows(config)
    assert original in rows
    assert sorted(row[1] for row in rows) == ["settled", "unknown"]
    total = sum(row[3] if row[3] is not None else row[2] for row in rows)
    assert 577711200 < total < 2_000_000_000
    audit = json.loads((config.parent / "luna_requests" / f"{identifier}.json").read_text())
    assert audit["status"] == "unknown" and audit["error_diagnostics"]["http_status"] == 400


def test_second_unknown_stops_immediately_and_blocks_every_restart(config):
    identifier = create_unknown(config)
    calls = []

    def timeout(**_payload):
        calls.append(True)
        raise TimeoutError

    transport = LunaIntentTransport(config, client=FakeClient(timeout), accepted_unknown_id=identifier)
    with pytest.raises(LunaRequestUncertain):
        transport.create("new context")
    assert [row[1] for row in attempt_rows(config)].count("unknown") == 2
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=FakeClient(timeout), accepted_unknown_id=identifier).create("third context")
    assert len(calls) == 1


@pytest.mark.parametrize("accepted", ["", "not-an-id", "A" * 32, 1])
def test_accepted_identifier_must_be_a_uuid_hex_string(config, accepted):
    with pytest.raises(ValueError):
        LunaIntentTransport(config, client=FakeClient(lambda **_: FakeResponse()), accepted_unknown_id=accepted)


def test_other_identifier_and_absent_unknown_are_rejected(config):
    client = FakeClient(lambda **_: FakeResponse())
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=client, accepted_unknown_id="a" * 32).create("board")
    identifier = create_unknown(config)
    other = "b" * 32 if identifier != "b" * 32 else "c" * 32
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=client, accepted_unknown_id=other).create("board")


@pytest.mark.parametrize("modification", ["payload", "hash", "http_status", "missing_audit"])
def test_tampered_or_missing_unknown_audit_blocks_continuation(config, modification):
    identifier = create_unknown(config)
    path = config.parent / "luna_requests" / f"{identifier}.json"
    record = json.loads(path.read_text())
    if modification == "payload":
        record["request"]["input"] = "other board"
    elif modification == "hash":
        record["request_sha256"] = "0" * 64
    elif modification == "http_status":
        record["error_diagnostics"]["http_status"] = 500
    if modification == "missing_audit":
        path.unlink()
    else:
        path.write_text(json.dumps(record))
    calls = []
    client = FakeClient(lambda **_: calls.append(True))
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=client, accepted_unknown_id=identifier).create("new board")
    assert not calls


@pytest.mark.parametrize("statement", [
    "UPDATE attempts SET reserved_nanos=reserved_nanos-1 WHERE id=?",
    "UPDATE attempts SET charged_nanos=1 WHERE id=?",
    "UPDATE attempts SET status='reserved' WHERE id=?",
    "UPDATE attempts SET request_hash='forged' WHERE id=?",
])
def test_tampered_ledger_entry_blocks_continuation(config, statement):
    identifier = create_unknown(config)
    sql(config, statement, (identifier,))
    calls = []
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=FakeClient(lambda **_: calls.append(True)),
                            accepted_unknown_id=identifier).create("new board")
    assert not calls


def test_other_reserved_attempt_blocks_continuation(config):
    identifier = create_unknown(config)
    ledger = CampaignLedger(config)
    ledger.reserve("experiment", "responses", {"model": "gpt-6-luna", "input": "board", "max_output_tokens": 64,
                                                 "service_tier": "default", "tools": []}, "reserved")
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=FakeClient(lambda **_: FakeResponse()),
                            accepted_unknown_id=identifier).create("new board")


def test_config_change_after_transport_initialization_is_rejected(config):
    identifier = create_unknown(config)
    calls = []
    transport = LunaIntentTransport(config, client=FakeClient(lambda **_: calls.append(True)), accepted_unknown_id=identifier)
    with config.open("a") as output:
        output.write("\n")
    with pytest.raises(LunaRequestUncertain):
        transport.create("new board")
    assert not calls


def test_other_audit_with_mismatched_status_blocks_continuation(config):
    identifier = create_unknown(config)
    transport = LunaIntentTransport(config, client=FakeClient(lambda **_: FakeResponse()), accepted_unknown_id=identifier)
    transport.create("first new board")
    path = next(path for path in (config.parent / "luna_requests").glob("*.json") if path.stem != identifier)
    record = json.loads(path.read_text())
    record["status"] = "unknown"
    path.write_text(json.dumps(record))
    with pytest.raises(LunaRequestUncertain):
        transport.create("second new board")


def test_concurrent_continuations_keep_one_unknown_and_reserve_sequentially(config):
    identifier = create_unknown(config)
    client = FakeClient(lambda **_: FakeResponse())

    def call(_index):
        return LunaIntentTransport(config, client=client, accepted_unknown_id=identifier).create("board")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(call, range(3)))
    rows = attempt_rows(config)
    assert [row[1] for row in rows].count("unknown") == 1
    assert [row[1] for row in rows].count("settled") == 3
    assert sum(row[3] if row[3] is not None else row[2] for row in rows) < 2_000_000_000
