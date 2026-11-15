"""The shared ledger reserves before transport and survives failures and concurrency."""
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from carl.icl.api import BudgetExceeded
from carl.icl.api import CachedOpenAI
from carl.icl.api import RequestFailed
from carl.icl.campaign_budget import CampaignLedger
from carl.icl.campaign_budget import CampaignLimitExceeded


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CampaignLedger:
    import carl.icl.campaign_budget as campaign_budget
    monkeypatch.setattr(campaign_budget, "dotenv_values", lambda _path: {})
    monkeypatch.delenv("CARL_CAMPAIGN_MAX_USD", raising=False)
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
    reservation = ledger.reserve("policy_validation", "responses", payload(), "a")
    ledger.mark_unknown(reservation)
    original = tmp_path / "budget.json"
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


def test_unknown_usage_stays_reserved_across_restart(ledger: CampaignLedger) -> None:
    request = payload()
    reservation = ledger.reserve("policy_validation", "responses", request, "b")
    assert not ledger.settle(reservation, {}, request)
    replacement = CampaignLedger(Path(ledger.path.parent / "budget.json"))
    row = replacement.summary()["attempts"][0]
    assert row["status"] == "reserved" and row["committed_usd"] > .5


def test_concurrent_process_equivalent_reservations_cannot_overspend(ledger: CampaignLedger) -> None:
    def reserve_one(index: int) -> bool:
        try:
            ledger.reserve("policy_validation", "responses", payload(), str(index))
            return True
        except CampaignLimitExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        accepted = list(pool.map(reserve_one, range(8)))
    summary = ledger.summary()["attempts"]
    assert 0 < sum(accepted) < len(accepted)
    assert sum(row["committed_usd"] for row in summary) <= 2.0


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
