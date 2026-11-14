"""The Luna transport reserves and journals every physical API attempt."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from carl.icl.campaign_budget import CampaignLedger
from carl.icl.campaign_budget import CampaignLimitExceeded
from carl.icl.luna_intent_transport import LunaIntentTransport
from carl.icl.luna_intent_transport import LunaRequestUncertain
from carl.icl.luna_intent_transport import create_luna_budget_config


class FakeResponse:
    def __init__(self, **changes):
        self.raw = {
            "id": "resp_fake", "model": "gpt-6-luna-2026-09-22", "service_tier": "default",
            "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 100, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 80}},
        } | changes
        self.output_text = "ok"
        self._request_id = "req_fake"

    def model_dump(self, *, mode):
        assert mode == "json"
        return self.raw


class FakeClient:
    def __init__(self, callback):
        self.responses = SimpleNamespace(create=callback)
        self.options = None

    def with_options(self, **options):
        self.options = options
        return self


@pytest.fixture
def config(tmp_path, monkeypatch):
    import carl.icl.campaign_budget as campaign_budget
    monkeypatch.setattr(campaign_budget, "dotenv_values", lambda _path: {})
    monkeypatch.delenv("CARL_CAMPAIGN_MAX_USD", raising=False)
    return create_luna_budget_config(tmp_path)


def audit_records(config):
    return [json.loads(path.read_text()) for path in (config.parent / "luna_requests").glob("*.json")]


def test_reserves_before_call_and_records_complete_request_and_response(config):
    observed = []

    def create(**payload):
        observed.append(CampaignLedger(config).summary()["attempts"])
        observed.append(audit_records(config))
        assert payload == {
            "model": "gpt-6-luna", "input": "the full board", "max_output_tokens": 512,
            "reasoning": {"effort": "none"}, "temperature": 0, "service_tier": "default", "tools": [], "store": False,
        }
        return FakeResponse()

    client = FakeClient(create)
    transport = LunaIntentTransport(config, timeout_seconds=42, client=client)
    assert transport.create("the full board", stage="experiment", tag="calibration").output_text == "ok"
    assert client.options == {"timeout": 42, "max_retries": 0}
    assert observed[0][0]["status"] == "reserved" and observed[0][0]["committed_usd"] > 0.57
    assert observed[1][0]["status"] == "reserved"
    record = audit_records(config)[0]
    assert record["status"] == "settled"
    assert record["request"]["input"] == "the full board"
    assert record["response"]["usage"]["input_tokens_details"]["cached_tokens"] == 80
    assert record["response"]["model"] == "gpt-6-luna-2026-09-22"
    assert record["request_id"] == "req_fake"
    assert record["tag"] == "calibration"
    assert record["conservative_charge_usd"] == pytest.approx((100 * .25 + 5 * .5) * 1.1 / 1_000_000)


def test_timeout_keeps_reservation_and_blocks_later_attempts_and_restart(config):
    calls = []

    def fail(**_payload):
        calls.append(1)
        raise TimeoutError("Private error details must not be saved")

    client = FakeClient(fail)
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=client).create("board")
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=client).create("another board")
    assert calls == [1]
    row = CampaignLedger(config).summary()["attempts"][0]
    assert row["status"] == "unknown" and row["committed_usd"] > .57
    record = audit_records(config)[0]
    assert record["status"] == "unknown" and record["error_type"] == "TimeoutError"
    assert "Private error" not in json.dumps(record)


def test_service_error_records_safe_diagnostics_without_raw_message_or_key(config):
    class ServiceError(Exception):
        status_code = 429
        code = "insufficient_quota"
        type = "rate_limit_error"
        request_id = "req_abc123"

    def fail(**_payload):
        raise ServiceError("secret api key and raw server response")

    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=FakeClient(fail)).create("board")
    record = audit_records(config)[0]
    assert record["error_diagnostics"] == {
        "http_status": 429, "service_code": "insufficient_quota", "service_type": "rate_limit_error",
        "request_id": "req_abc123",
    }
    assert "secret api key" not in json.dumps(record)


def test_service_error_rejects_unrecognised_fields(config):
    class ServiceError(Exception):
        status_code = True
        code = "sk_protected_value"
        type = "private value"
        request_id = "sk-private-value"

    def fail(**_payload):
        raise ServiceError()

    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=FakeClient(fail)).create("board")
    record = audit_records(config)[0]
    assert record["error_diagnostics"] == {
        "http_status": None, "service_code": None, "service_type": None, "request_id": None,
    }
    assert "sk_protected_value" not in json.dumps(record)


@pytest.mark.parametrize("changes", [
    {"usage": None},
    {"usage": {"input_tokens": True, "output_tokens": 1}},
    {"usage": {"input_tokens": 1, "output_tokens": 513}},
    {"model": "gpt-6.1-terra"},
    {"service_tier": "priority"},
    {"service_tier": None},
])
def test_unknown_or_invalid_usage_keeps_response_and_blocks_retries(config, changes):
    client = FakeClient(lambda **_payload: FakeResponse(**changes))
    transport = LunaIntentTransport(config, client=client)
    with pytest.raises(LunaRequestUncertain):
        transport.create("board")
    with pytest.raises(LunaRequestUncertain):
        transport.create("board")
    assert CampaignLedger(config).summary()["attempts"][0]["status"] == "unknown"
    assert audit_records(config)[0]["response"] is not None


def test_interruption_and_orphaned_reservation_fail_closed(config):
    def interrupted(**_payload):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        LunaIntentTransport(config, client=FakeClient(interrupted)).create("board")
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=FakeClient(lambda **_payload: FakeResponse())).create("board")


def test_stale_audit_blocks_even_if_ledger_was_settled(config):
    transport = LunaIntentTransport(config, client=FakeClient(lambda **_payload: FakeResponse()))
    transport.create("board")
    path = next((config.parent / "luna_requests").glob("*.json"))
    record = json.loads(path.read_text())
    record["status"] = "received"
    path.write_text(json.dumps(record))
    with pytest.raises(LunaRequestUncertain):
        transport.create("next board")


def test_small_budget_prevents_physical_call(config, tmp_path):
    del config
    limited = create_luna_budget_config(tmp_path / "limited", "0.10")
    calls = []
    transport = LunaIntentTransport(limited, client=FakeClient(lambda **_payload: calls.append(True)))
    with pytest.raises(CampaignLimitExceeded):
        transport.create("board")
    assert calls == []


def test_configuration_is_bounded_and_cannot_be_replaced(config, tmp_path):
    with pytest.raises(ValueError):
        create_luna_budget_config(tmp_path / "bad", "2.01")
    with pytest.raises(ValueError):
        create_luna_budget_config(config.parent, "1.00")
    value = json.loads(config.read_text())
    value["models"]["gpt-6-luna"]["cache_write_12h_usd_per_million"] = "0.01"
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        LunaIntentTransport(config, client=FakeClient(lambda **_payload: FakeResponse()))


def test_concurrent_calls_are_serialized(config):
    lock = threading.Lock()
    active = 0
    peak = 0

    def create(**_payload):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return FakeResponse()

    def call(_index):
        return LunaIntentTransport(config, client=FakeClient(create)).create("board")

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(call, range(3)))
    assert peak == 1
    assert len(audit_records(config)) == 3
    assert sum(row["committed_usd"] for row in CampaignLedger(config).summary()["attempts"]) < 2
