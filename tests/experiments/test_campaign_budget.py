"""The shared ledger prices provider routes and protects limits across failures and processes."""
import hashlib
import json
import multiprocessing
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from carl.icl import campaign_budget
from carl.icl.api import BudgetExceeded
from carl.icl.api import CachedOpenAI
from carl.icl.api import RequestFailed
from carl.icl.campaign_budget import CampaignLedger
from carl.icl.campaign_budget import CampaignLimitExceeded


@pytest.fixture(autouse=True)
def fixed_campaign_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(campaign_budget, "dotenv_values", lambda _path: {})
    monkeypatch.delenv("CARL_CAMPAIGN_MAX_USD", raising=False)


@pytest.fixture
def ledger(tmp_path: Path) -> CampaignLedger:
    config = {
        "campaign_id": "test",
        "ledger_path": str(tmp_path / "budget.sqlite3"),
        "limit_usd": "40.00",
        "stage_limits_usd": {"policy_validation": "2.00", "unassigned": "0.00"},
        "models": {
            "gpt-6-luna": {
                "input_usd_per_million": "0.10",
                "cached_input_usd_per_million": "0.01",
                "cache_write_30m_usd_per_million": "0.125",
                "cache_write_12h_usd_per_million": "0.25",
                "output_usd_per_million": "0.50",
                "max_input_tokens": 1050000,
                "max_output_tokens": 128000,
                "long_context_threshold": 272000,
                "regional_multiplier": "1.10",
            }
        },
    }
    path = tmp_path / "budget.json"
    path.write_text(json.dumps(config))
    return CampaignLedger(path)


def payload() -> dict:
    return {"model": "gpt-6-luna", "input": "board", "max_output_tokens": 128, "tools": []}


def test_missing_config_does_not_create_a_new_ledger(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CampaignLedger(tmp_path / "missing-budget.json")
    assert not list(tmp_path.glob("*.sqlite3"))


def test_copied_config_retains_unknown_reservations_and_rejects_changes(
    ledger: CampaignLedger, tmp_path: Path
) -> None:
    original = tmp_path / "budget.json"
    request = payload()
    reservation = ledger.reserve("policy_validation", "responses", request, "a")
    assert not ledger.settle(reservation, {}, request)
    row = CampaignLedger(original).summary()["attempts"][0]
    assert row["status"] == "reserved" and row["committed_usd"] > .5
    ledger.mark_unknown(reservation)
    copied = tmp_path / "copied-budget.json"
    copied.write_bytes(original.read_bytes())

    reopened = CampaignLedger(copied)
    assert reopened.path == ledger.path
    assert reopened.summary()["attempts"][0]["status"] == "unknown"
    copied.write_bytes(copied.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="configuration changed"):
        CampaignLedger(copied)
    assert CampaignLedger(original).summary()["attempts"][0]["status"] == "unknown"


def _reserve_in_process(config_path: str, index: int) -> bool:
    try:
        CampaignLedger(Path(config_path)).reserve("policy_validation", "responses", payload(), str(index))
        return True
    except CampaignLimitExceeded:
        return False


def test_known_usage_accounts_cache_write_tiers(ledger: CampaignLedger) -> None:
    request = payload()
    reservation = ledger.reserve("policy_validation", "responses", request, "a")
    usage = {"input_tokens": 1000, "output_tokens": 100,
             "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 500,
                                      "cache_write_tokens_details": {"ttl_30m_tokens": 300, "ttl_12h_tokens": 200}}}
    assert ledger.settle(reservation, usage, request)
    actual = ledger.summary()["attempts"][0]["committed_usd"]
    expected = (300 * .10 + 200 * .01 + 300 * .125 + 200 * .25 + 100 * .5) * 1.1 / 1_000_000
    assert actual == pytest.approx(expected, abs=1e-9)


def test_atomic_reservation_across_independent_processes(ledger: CampaignLedger) -> None:
    config = str(ledger.path.parent / "budget.json")
    with multiprocessing.get_context("fork").Pool(6) as pool:
        accepted = pool.starmap(_reserve_in_process, [(config, index) for index in range(6)])
    assert 0 < sum(accepted) < len(accepted)
    assert sum(row["committed_usd"] for row in ledger.summary()["attempts"]) <= 2.0


def test_lower_account_limit_is_durable_across_processes(ledger: CampaignLedger, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARL_CAMPAIGN_MAX_USD", "0.40")
    lowered = CampaignLedger(ledger.path.parent / "budget.json")
    monkeypatch.delenv("CARL_CAMPAIGN_MAX_USD")
    assert CampaignLedger(ledger.path.parent / "budget.json").summary()["effective_limit_usd"] == 0.4
    with pytest.raises(CampaignLimitExceeded):
        lowered.reserve("policy_validation", "responses", payload(), "limited")



def test_dotenv_limit_is_enforced_without_loading_credentials(ledger: CampaignLedger, monkeypatch: pytest.MonkeyPatch) -> None:
    import carl.icl.campaign_budget as campaign_budget
    monkeypatch.setattr(campaign_budget, "dotenv_values", lambda _path: {"CARL_CAMPAIGN_MAX_USD": "0.40"})
    lowered = CampaignLedger(ledger.path.parent / "budget.json")
    assert lowered.summary()["effective_limit_usd"] == 0.4
    with pytest.raises(CampaignLimitExceeded):
        lowered.reserve("policy_validation", "responses", payload(), "limited-in-dotenv")

def test_transport_is_reserved_before_call_and_replay_costs_zero(tmp_path, monkeypatch, ledger):
    import carl.icl.api as api
    monkeypatch.setattr(api, "_campaign_ledger", lambda: ledger)
    monkeypatch.setenv("CARL_CAMPAIGN_STAGE", "policy_validation")
    observed = []
    def create(**_kwargs):
        observed.append(ledger.summary()["attempts"][0]["status"])
        return SimpleNamespace(status="completed", output_text="ok", output=[], model="gpt-6-luna", id="response",
                               usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 2,
                                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 10}}))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    request = payload()
    record = CachedOpenAI(tmp_path, "openai", client=client).call("responses", request)
    assert observed == ["reserved"]
    assert CachedOpenAI(tmp_path, "replay").call("responses", request) == record
    summary = ledger.summary()
    assert summary["attempts"][0]["status"] == "settled"
    assert {(item["kind"], item["count"]) for item in summary["events"]} == {("logical_request", 2), ("local_replay_hit", 1)}


def test_failed_transport_keeps_reservation_and_stage_must_be_explicit(tmp_path, monkeypatch, ledger):
    import carl.icl.api as api
    monkeypatch.setattr(api, "_campaign_ledger", lambda: ledger)
    def fail(**_kwargs):
        raise TimeoutError("do not save private exception text")
    client = SimpleNamespace(responses=SimpleNamespace(create=fail))
    monkeypatch.delenv("CARL_CAMPAIGN_STAGE", raising=False)
    with pytest.raises(BudgetExceeded):
        CachedOpenAI(tmp_path, "openai", client=client).call("responses", payload())
    assert ledger.summary()["attempts"] == []
    monkeypatch.setenv("CARL_CAMPAIGN_STAGE", "policy_validation")
    with pytest.raises(RequestFailed):
        CachedOpenAI(tmp_path, "openai", client=client).call("responses", payload())
    assert ledger.summary()["attempts"][0]["status"] == "unknown"


@pytest.fixture
def budget_path(tmp_path: Path) -> Path:
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({
        "campaign_id": "test-routes",
        "ledger_path": str(tmp_path / "budget.sqlite3"),
        "limit_usd": "0.001",
        "stage_limits_usd": {"test": "0.001"},
        "models": {
            "*": {
                "input_usd_per_million": "0.5",
                "output_usd_per_million": "1.0",
                "max_input_tokens": 100,
                "max_output_tokens": 20,
            },
        },
        "routes": {
            "mistral/chat_completions/miniset/fixture": {
                "input_usd_per_million": "2.0",
                "output_usd_per_million": "3.0",
                "max_input_tokens": 100,
                "max_output_tokens": 20,
            },
            "openai/embeddings/fixture": {
                "input_usd_per_million": "0.1",
                "max_input_tokens": 100,
            },
        },
    }), encoding="utf-8")
    return path


def chat_payload():
    return {
        "model": "miniset/fixture",
        "messages": [{"role": "user", "content": "board"}],
        "max_tokens": 10,
    }


def test_responses_chat_and_batched_embeddings_use_their_own_prices(budget_path):
    ledger = CampaignLedger(budget_path)
    chat = chat_payload()
    embeddings = {"model": "fixture", "input": ["board", "second board"]}
    responses = {"model": chat["model"], "input": "board", "max_output_tokens": 10}
    ids = (
        ledger.reserve("test", "chat_completions", chat, "one", "chat", provider="mistral"),
        ledger.reserve("test", "embeddings", embeddings, "two", "embeddings"),
        ledger.reserve("test", "responses", responses, "three", "responses"),
    )
    reserved = {row["tag"]: row["committed_usd"] for row in ledger.summary()["attempts"]}
    assert reserved == {"chat": 0.000230, "embeddings": 0.000020, "responses": 0.000115}
    assert ledger.settle(ids[0], {"input_tokens": 15, "output_tokens": 5}, chat)
    assert ledger.settle(ids[1], {"input_tokens": 150, "output_tokens": 0}, embeddings)
    assert ledger.settle(ids[2], {"input_tokens": 15, "output_tokens": 5}, responses)
    charged = {row["tag"]: row["committed_usd"] for row in ledger.summary()["attempts"]}
    assert charged == {"chat": 0.000045, "embeddings": 0.000015, "responses": 0.0000125}


@pytest.mark.parametrize("provider, endpoint, payload, message", [
    ("openai", "chat_completions", chat_payload(), "Unsupported campaign billing route"),
    ("mistral", "responses", chat_payload(), "Unsupported campaign billing route"),
    ("mistral", "chat_completions", {**chat_payload(), "model": "unknown"}, "No configured campaign pricing"),
    ("openai", "embeddings", {"model": "unknown", "input": "board"}, "No configured campaign pricing"),
    ("openai", "embeddings", {"model": "fixture", "input": [[1, 2, 3]]}, "nonempty text"),
    ("mistral", "chat_completions", {**chat_payload(), "max_tokens": 21}, "maximum output token count"),
    ("mistral", "chat_completions", {**chat_payload(), "n": 2}, "single choice"),
    ("mistral", "chat_completions", {**chat_payload(), "tools": [{"type": "function"}]}, "without tools"),
    ("openai", "responses", {"model": "fixture", "input": "board", "max_output_tokens": 10,
                            "previous_response_id": "prior"}, "server-side prompt state"),
])
def test_unpriced_or_unsupported_requests_never_reserve(budget_path, provider, endpoint, payload, message):
    ledger = CampaignLedger(budget_path)
    with pytest.raises(CampaignLimitExceeded, match=message):
        ledger.reserve("test", endpoint, payload, "digest", provider=provider)
    assert ledger.summary()["attempts"] == []


def test_settlement_requires_bound_wire_request_and_valid_provider_usage(budget_path):
    ledger = CampaignLedger(budget_path)
    request = chat_payload()
    identifier = ledger.reserve("test", "chat_completions", request, "digest", provider="mistral")
    valid = {"input_tokens": 20, "output_tokens": 4}
    assert not ledger.settle(identifier, valid, {**request, "max_tokens": 11})
    assert not ledger.settle(identifier, valid, {**request, "model": "other"})
    assert not ledger.settle(identifier, valid, {**request, "messages": [{"role": "user", "content": "other"}]})
    for usage in (
        {},
        {**valid, "input_tokens": True},
        {**valid, "input_tokens": 101},
        {**valid, "output_tokens": 11},
        {**valid, "total_tokens": 20},
        {**valid, "input_tokens_details": {"cached_tokens": 21}},
        {**valid, "input_tokens_details": {"cache_write_tokens": 3, "cache_write_tokens_details": {
            "ttl_30m_tokens": True, "ttl_12h_tokens": 2,
        }}},
    ):
        assert not ledger.settle(identifier, usage, request)
    assert ledger.summary()["attempts"][0]["status"] == "reserved"
    assert ledger.settle(identifier, valid, request)


def test_interrupted_request_retains_budget_and_blocks_same_slot_across_restart(budget_path):
    ledger = CampaignLedger(budget_path)
    request = chat_payload()
    identifier = ledger.reserve(
        "test", "chat_completions", request, "digest", provider="mistral", protect_request=True,
    )
    restarted = CampaignLedger(budget_path)
    with pytest.raises(CampaignLimitExceeded, match="requires reconciliation"):
        restarted.reserve("test", "chat_completions", request, "digest", provider="mistral", protect_request=True)
    restarted.mark_unknown(identifier)
    with pytest.raises(CampaignLimitExceeded, match="requires reconciliation"):
        ledger.reserve("test", "chat_completions", request, "digest", provider="mistral", protect_request=True)
    responses = {"model": request["model"], "input": "board", "max_output_tokens": 10}
    ledger.reserve("test", "responses", responses, "digest", protect_request=True)
    assert {row["status"]: row["committed_usd"] for row in ledger.summary()["attempts"]} == {
        "reserved": 0.000115,
        "unknown": 0.000230,
    }


def test_physical_endpoints_share_one_stage_ceiling_and_replay_events_cost_zero(budget_path):
    config = json.loads(budget_path.read_text(encoding="utf-8"))
    config["stage_limits_usd"]["test"] = "0.000240"
    budget_path.write_text(json.dumps(config), encoding="utf-8")
    ledger = CampaignLedger(budget_path)
    request = chat_payload()
    identifier = ledger.reserve("test", "chat_completions", request, "chat", provider="mistral")
    embeddings = {"model": "fixture", "input": ["board", "second board"]}
    ledger.event("test", "logical_request", "embeddings", "cache-hit")
    ledger.event("test", "local_replay_hit", "embeddings", "cache-hit")
    with pytest.raises(CampaignLimitExceeded, match="budget would be exceeded"):
        ledger.reserve("test", "embeddings", embeddings, "new")
    assert {item["kind"]: item["count"] for item in ledger.summary()["events"]} == {
        "local_replay_hit": 1,
        "logical_request": 1,
    }
    assert ledger.settle(identifier, {"input_tokens": 10, "output_tokens": 4}, request)
    ledger.reserve("test", "embeddings", embeddings, "new")


def test_pre_route_ledger_keeps_pending_attempts_and_the_saved_ceiling(budget_path):
    config = json.loads(budget_path.read_text(encoding="utf-8"))
    path = config["ledger_path"]
    config_hash = hashlib.sha256(budget_path.read_bytes()).hexdigest()
    with sqlite3.connect(path) as old:
        old.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE attempts (
                id TEXT PRIMARY KEY, stage TEXT, tag TEXT, endpoint TEXT, model TEXT,
                request_hash TEXT, reserved_nanos INTEGER, charged_nanos INTEGER,
                status TEXT, usage_json TEXT
            );
            INSERT INTO metadata VALUES ('effective_limit_nanos', '250000');
            INSERT INTO attempts (id,stage,tag,endpoint,model,request_hash,reserved_nanos,status)
                VALUES ('old','test','archive','responses','fixture','digest',200000,'reserved');
        """)
        old.execute("INSERT INTO metadata VALUES ('config_hash', ?)", (config_hash,))
    ledger = CampaignLedger(budget_path)
    assert ledger.summary()["effective_limit_usd"] == 0.000250
    with pytest.raises(CampaignLimitExceeded, match="requires reconciliation"):
        ledger.reserve("test", "responses", {"model": "fixture", "input": "board", "max_output_tokens": 10},
                       "digest", protect_request=True)
    with sqlite3.connect(path) as upgraded:
        assert upgraded.execute("SELECT provider,payload_hash,status FROM attempts WHERE id='old'").fetchone() == (
            "openai", None, "reserved",
        )
