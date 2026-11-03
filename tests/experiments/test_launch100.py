"""Offline contracts, paired prompts and interruption-safe response checkpoints."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from carl.icl.api import BudgetExceeded, RequestFailed, _CACHE_ATTEMPT_SLOT, request_cache_digest, write_json
from carl.icl.boards import from_rows
from carl.icl.config import ExperimentConfig
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.types import Instance, JSONObject
from scripts.goal_probe.components import TaskProgressComponents
from scripts.launch100.journal import EventJournal, JournalAPI
from scripts.launch100.plan import Bundle, build_plan, component, configuration, ledger_status, load_assets, prepare
from scripts.launch100.runtime import execute, run_job, status, verify_saved


@pytest.fixture(scope="module")
def assets() -> Bundle:
    return load_assets()


def test_plan_never_constructs_api_and_matches_u_d_demonstrations(assets: Bundle, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Planning constructed an API")

    monkeypatch.setattr(ExperimentAPI, "__init__", forbidden)
    plan = build_plan(assets, ["relative_u", "absolute_d", "fixed_k", "verbal_relative"])
    assert plan["jobs"] == 400 and len(plan["cohort"]) == 100
    assert plan["node_budget"] == 500 and plan["action_budget"] == 4000
    relative, absolute = plan["prompt_previews"][0]["payload"], plan["prompt_previews"][2]["payload"]
    for payload, label in ((relative, "u"), (absolute, "d")):
        users = [json.loads(message["content"][0]["text"]) for message in payload["input"] if message["role"] == "user"]
        assert sum(row.get("demonstration_kind") == "task_progress" for row in users) == 6
        assert sum(label in row for row in users) == 27  # 26 training examples plus query.
        assert f"requested {label}." in payload["input"][14]["content"][0]["text"]
    assistant = lambda payload: [message for message in payload["input"] if message["role"] == "assistant"]
    assert assistant(relative) == assistant(absolute)
    assert relative["input"][1:14] == absolute["input"][1:14]
    assert plan["configs"]["fixed_k"]["scales"] == [8, 4]
    assert plan["configs"]["fixed_k"]["fixed_cap"] == "horizon"


def test_action_cap_is_separate_and_default_progress_payload_is_unchanged(assets: Bundle) -> None:
    config = configuration(assets, "relative_u", 7)
    assert config.max_expansions == 500 and config.action_budget == 7
    assert config.calibration.action_cap == 8
    offline = cast(ExperimentAPI, SimpleNamespace(provider="replay"))
    launched = component(assets, config, offline, assets.instances[0])
    historical = TaskProgressComponents(config, offline, launched.env, assets.training, assets.demonstrations)
    state = from_rows(assets.instances[0]["state"])
    assert launched.generator_payload(state, .85, False) == historical.generator_payload(state, .85, False)


def test_prepare_rejects_protocol_drift_and_status_keeps_missing(tmp_path: Path) -> None:
    plan = {"freeze_id": "a", "methods": ["relative_u", "absolute_d"], "seed": 11,
            "node_budget": 500, "action_budget": 4000, "cohort": [{"id": "first"}, {"id": "second"}]}
    prepare(plan, tmp_path)
    prepare(plan, tmp_path)
    with pytest.raises(ValueError, match="differs"):
        prepare({**plan, "action_budget": 500}, tmp_path)
    result = status(plan, tmp_path, write_csv=True)
    assert result["expected"] == result["counts"]["missing"] == 4
    assert len((tmp_path / "status.csv").read_text().splitlines()) == 5


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), 0, -1, 40.01])
def test_execution_rejects_missing_or_invalid_cap_before_credentials(assets: Bundle, tmp_path: Path, cap: float) -> None:
    with pytest.raises(ValueError, match="usd-cap"):
        execute(assets, {}, tmp_path, cap, resume=False)


def api_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ExperimentConfig, list[str]]:
    monkeypatch.setattr("carl.icl.api._campaign_ledger", lambda: None)
    calls = []

    def call(api: ExperimentAPI, _endpoint: str, payload: JSONObject, _mock: object = None) -> JSONObject:
        calls.append(api.provider)
        api.metrics["logical_requests"] += 1
        api.metrics["logical_input_tokens"] += 7
        return {"output_text": json.dumps({"value": payload["value"]}), "usage": {"output_tokens": 3},
                "status": "completed"}

    monkeypatch.setattr(ExperimentAPI, "call", call)
    return ExperimentConfig(cache_dir=str(tmp_path / "cache")), calls


def test_resume_reuses_saved_responses_and_budget_counters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, calls = api_fixture(tmp_path, monkeypatch)
    directory = tmp_path / "journal"
    first = JournalAPI(config, directory, "test")
    expected = first.call("responses", {"value": 1})
    resumed = JournalAPI(config, directory, "test")
    assert resumed.call("responses", {"value": 1})["output_text"] == expected["output_text"]
    assert calls == ["openai"] and resumed.metrics["logical_input_tokens"] == 7
    resumed.call("responses", {"value": 2})
    assert calls == ["openai", "openai"] and resumed.metrics["logical_input_tokens"] == 14
    assert resumed.replayed_calls == 1
    changed = JournalAPI(config, directory, "test")
    with pytest.raises(ValueError, match="differs"):
        changed.call("responses", {"value": 99})


def test_uncertain_request_requires_cache_and_cannot_repeat_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, calls = api_fixture(tmp_path, monkeypatch)
    api = JournalAPI(config, tmp_path / "journal", "test")
    payload = {"value": 1}
    key = request_cache_digest({"endpoint": "responses", "payload": payload})
    write_json(api.pending, {"index": 0, "request_hash": key})
    with pytest.raises(RequestFailed, match="Interrupted request"):
        api.call("responses", payload)
    assert calls == []
    write_json(api.cache_dir / f"{key}.json", {"status": "completed", "request": {"endpoint": "responses", "payload": payload}})
    api.call("responses", payload)
    assert calls == ["replay"] and not api.pending.exists()
    # A crash after committing a response but before deleting pending is recoverable.
    write_json(api.pending, {"index": 0, "request_hash": key})
    reopened = JournalAPI(config, api.directory, "test")
    assert not reopened.pending.exists()


def test_trace_prefix_must_match_and_be_consumed(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"
    first = EventJournal(path)
    first({"action": 1})
    resumed = EventJournal(path)
    with pytest.raises(ValueError, match="consume"):
        resumed.verify_consumed()
    with pytest.raises(ValueError, match="diverged"):
        resumed({"action": 2})
    resumed({"action": 1})
    resumed({"action": 2})
    resumed.verify_consumed()
    assert len(path.read_text().splitlines()) == 2


def test_pending_recovery_replays_original_http_error_and_completed_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("carl.icl.api._campaign_ledger", lambda: None)
    calls = []

    def metered(api: ExperimentAPI, _endpoint: str, _payload: JSONObject, _mock: object = None) -> JSONObject:
        slot = _CACHE_ATTEMPT_SLOT.get()
        calls.append((api.provider, slot))
        api.metrics["logical_requests"] += 1
        if slot == 0:
            raise RequestFailed("Recorded HTTP429")
        return {"status": "completed", "output_text": '{"value":1}'}

    monkeypatch.setattr(ExperimentAPI, "call", metered)
    config = ExperimentConfig(cache_dir=str(tmp_path / "cache"))
    api = JournalAPI(config, tmp_path / "journal", "test")
    payload = {"value": 1}
    request = {"endpoint": "responses", "payload": payload}
    original_key = request_cache_digest(request)
    write_json(api.pending, {"index": 0, "request_hash": original_key})
    write_json(api.cache_dir / f"{original_key}.json", {"status": "error", "http_status": 429, "request": request})
    token = _CACHE_ATTEMPT_SLOT.set(2)
    try:
        retry_key = request_cache_digest(request)
    finally:
        _CACHE_ATTEMPT_SLOT.reset(token)
    write_json(api.cache_dir / f"{retry_key}.json", {"status": "completed", "request": request})
    assert api.call("responses", payload)["output_text"] == '{"value":1}'
    assert calls == [("replay", 0), ("replay", 2)]
    assert api.metrics["logical_requests"] == 2 and api.reconciled_pending_calls == 1


def test_production_search_records_actions_and_solution_without_model_calls(
    assets: Bundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("carl.icl.api._campaign_ledger", lambda: None)
    config = configuration(assets, "relative_u", 2)
    source = ["#####", "#@$.#", "#####", "#####", "#####"]
    target = from_rows(["#####", "#-@*#", "#####", "#####", "#####"])
    model = SimpleNamespace(diagnostics={}, propose=lambda _state, _scale: [target],
                            probabilities=lambda _state, _target: np.array([0., 0., 0., 1.]), value=lambda _state: 0.5)
    monkeypatch.setattr("scripts.launch100.runtime.component", lambda *_args: model)
    plan = {"freeze_id": "unit", "seed": 11, "node_budget": 500, "action_budget": 2,
            "configs": {"relative_u": config.model_dump(mode="json")}}
    row = run_job(assets, plan, {"id": "unit", "state": source, "level_key": "unit"}, "relative_u", tmp_path)
    assert row["outcome"]["solved"] and row["outcome"]["attempted_actions"] == 1
    assert row["all_actions_replay_verified"] and row["solution_replay_verified"]
    instance: Instance = {"id": "unit", "state": source, "level_key": "unit"}
    assert verify_saved(plan, instance, "relative_u", tmp_path)["complete"]
    (tmp_path / "actions.jsonl").write_text('{"action":0}\n')
    with pytest.raises(ValueError, match="trace differs"):
        verify_saved(plan, instance, "relative_u", tmp_path)


@pytest.mark.parametrize("reason,complete", [("Logical API request budget exhausted", True),
                                            ("Campaign or stage pilot_relative budget would be exceeded", False)])
def test_model_resource_cap_is_observed_but_shared_exhaustion_stops(
    assets: Bundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str, complete: bool,
) -> None:
    monkeypatch.setattr("carl.icl.api._campaign_ledger", lambda: None)

    def value(_state: object) -> float:
        raise BudgetExceeded(reason)

    model = SimpleNamespace(diagnostics={}, value=value)
    monkeypatch.setattr("scripts.launch100.runtime.component", lambda *_args: model)
    config = configuration(assets, "relative_u", 2)
    plan = {"freeze_id": "captest", "seed": 11, "node_budget": 500, "action_budget": 2,
            "configs": {"relative_u": config.model_dump(mode="json")}}
    row = run_job(assets, plan, assets.instances[0], "relative_u", tmp_path)
    assert row["complete"] is complete
    assert row["stop_category"] == ("model_resource_cap" if complete else "shared_budget")


def test_offline_ledger_respects_lower_environment_and_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARL_CAMPAIGN_MAX_USD", "0.25")
    monkeypatch.setattr("scripts.launch100.plan.dotenv_values", lambda _path: {"CARL_CAMPAIGN_MAX_USD": "0.1"})
    assert ledger_status()["effective_ceiling_usd"] <= 0.1


def test_live_launch_cannot_reset_missing_campaign_ledger(assets: Bundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.launch100.runtime.ledger_status", lambda: {"ledger_present": False})
    with pytest.raises(ValueError, match="history"):
        execute(assets, {}, tmp_path, 40, resume=False)
