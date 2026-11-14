"""Action schema and shared-ledger integration, without network calls."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from carl.icl.campaign_budget import CampaignLedger
from carl.icl.luna_action_format import action_text_format
from carl.icl.luna_intent_rollout import Action
from carl.icl.luna_intent_transport import LunaIntentTransport, LunaRequestUncertain, create_luna_budget_config


def test_format_is_strict_and_matches_controller_actions():
    assert action_text_format() == {
        "format": {
            "type": "json_schema", "name": "sokoban_action_v1", "strict": True,
            "schema": {
                "type": "object", "required": ["action"], "additionalProperties": False,
                "properties": {"action": {"type": "string", "enum": [action.value for action in Action]}},
            },
        },
    }
    assert [action.value for action in Action] == ["UP", "DOWN", "LEFT", "RIGHT"]


def test_each_format_is_independent_of_previous_mutations():
    old = action_text_format()
    old["format"]["schema"]["properties"]["action"]["enum"].append("WAIT")
    assert "WAIT" not in action_text_format()["format"]["schema"]["properties"]["action"]["enum"]


class FakeResponse:
    output_text = '{"action":"UP"}'
    _request_id = "req_test123"

    def model_dump(self, *, mode):
        assert mode == "json"
        return {"id": "resp_test", "model": "gpt-6-luna", "service_tier": "default", "status": "completed",
                "usage": {"input_tokens": 300, "output_tokens": 7}, "output": []}


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


def test_format_is_hashed_and_audited_and_plain_mode_is_unchanged(config):
    sent = []

    def create(**payload):
        sent.append(payload)
        reservations = CampaignLedger(config).summary()["attempts"]
        assert any(row["status"] == "reserved" and row["committed_usd"] > .57 for row in reservations)
        return FakeResponse()

    client = FakeClient(create)
    plain = LunaIntentTransport(config, client=client)
    structured = LunaIntentTransport(config, client=client, structured_actions=True)
    plain.create("same board", tag="plain")
    structured.create("same board", tag="structured")
    assert "text" not in sent[0]
    assert sent[1]["text"] == action_text_format()
    assert {key: value for key, value in sent[1].items() if key != "text"} == sent[0]
    assert client.options["max_retries"] == 0
    audits = [json.loads(path.read_text()) for path in (config.parent / "luna_requests").glob("*.json")]
    by_tag = {record["tag"]: record for record in audits}
    assert set(by_tag) == {"plain", "structured"}
    for tag in ("plain", "structured"):
        audit = by_tag[tag]
        expected = hashlib.sha256(json.dumps(audit["request"], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert audit["request_sha256"] == expected and audit["status"] == "settled"
        assert audit["conservative_charge_usd"] > 0
    assert by_tag["structured"]["request"]["text"] == action_text_format()
    assert by_tag["plain"]["request_sha256"] != by_tag["structured"]["request_sha256"]
    summary = CampaignLedger(config).summary()
    assert sum(row["count"] for row in summary["attempts"]) == 2
    assert sum(row["committed_usd"] for row in summary["attempts"]) < 2


def test_unknown_structured_attempt_blocks_plain_attempt_in_same_ledger(config):
    calls = []

    def fail(**payload):
        calls.append(payload)
        raise TimeoutError

    client = FakeClient(fail)
    structured = LunaIntentTransport(config, client=client, structured_actions=True)
    with pytest.raises(LunaRequestUncertain):
        structured.create("board")
    with pytest.raises(LunaRequestUncertain):
        LunaIntentTransport(config, client=client).create("board")
    assert len(calls) == 1 and calls[0]["text"] == action_text_format()
    row = CampaignLedger(config).summary()["attempts"][0]
    assert row["status"] == "unknown" and row["committed_usd"] > .57


@pytest.mark.parametrize("bad_value", [None, "yes", 1, 0])
def test_format_flag_rejects_non_boolean_values(config, bad_value):
    with pytest.raises(ValueError):
        LunaIntentTransport(config, client=FakeClient(lambda **_: FakeResponse()), structured_actions=bad_value)
