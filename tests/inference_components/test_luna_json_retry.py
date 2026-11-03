"""Retry malformed model JSON without replaying actions or losing attempts."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import numpy as np

from carl.icl.api import BudgetExceeded, RequestFailed
from carl.icl.boards import from_rows, to_rows
from carl.icl.config import ExperimentConfig, Limits
from carl.icl.datasets import environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.search import rollout
from carl.icl.types import TrainingData
from scripts.campaign_components_validation import CapturingAPI
from scripts.tune_campaign_scales import CappedAPI


class CountingLedger:
    """Capture ledger activity independently of the user's campaign."""

    def __init__(self, *, unknown: bool = False) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.reservations: list[tuple[Any, ...]] = []
        self.unknown: list[str] = []
        self.leave_unknown = unknown

    def event(self, *args: Any) -> None:
        self.events.append(args)

    def reserve(self, *args: Any) -> str:
        self.reservations.append(args)
        return str(len(self.reservations))

    def settle(self, *_args: Any) -> bool:
        return not self.leave_unknown

    def mark_unknown(self, reservation: str) -> None:
        self.unknown.append(reservation)


def fake_response(text: str) -> SimpleNamespace:
    """Produce an SDK-shaped response, including a full raw record."""
    return SimpleNamespace(
        status="completed", output_text=text, output=[], model="test-model", id="test-response",
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 8}),
        model_dump=lambda **_kwargs: {"status": "completed", "raw_text": text},
    )


def component(api: ExperimentAPI, retries: int = 1) -> LunaComponents:
    board = from_rows(["#####", "#---#", "#@$.#", "#---#", "#####"])
    training: TrainingData = {"contexts": [{"id": "train", "state": to_rows(board), "remaining_actions": 1,
                              "candidates": [{"k": 1, "actions": [3],
                                              "subgoal": to_rows(environment(board).next_state(board, 3))}],
                              "calibration_indices": [0]}]}
    return LunaComponents(ExperimentConfig(json_max_retries=retries, few_shot=1), api, environment(board), training)


def payload() -> dict[str, Any]:
    return {"model": "test-model", "input": "unchanged", "max_output_tokens": 64,
            "metadata": {"component": "policy"}}


def test_invalid_then_valid_preserves_both_raw_records_and_replays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = CountingLedger()
    monkeypatch.setattr("carl.icl.api._campaign_ledger", lambda: ledger)
    replies = iter(['{"probabilities":[0,1,0,0}', '{"probabilities":[0,1,0,0]}'])
    requests: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        requests.append(kwargs)
        return fake_response(next(replies))

    api = ExperimentAPI(tmp_path, "openai", Limits(max_requests=4, max_output_tokens=256),
                        SimpleNamespace(responses=SimpleNamespace(create=create), max_retries=0))
    policy = component(api)
    assert policy._json(payload(), dict) == {"probabilities": [0, 1, 0, 0]}
    assert requests == [payload(), payload()]
    assert api.metrics["logical_requests"] == api.metrics["requests"] == 2
    records = [json.loads(path.read_text()) for path in (tmp_path / "openai").glob("*.json")]
    assert len(records) == 2
    assert {record["output_text"] for record in records} == {'{"probabilities":[0,1,0,0}', '{"probabilities":[0,1,0,0]}'}
    assert all(record["request"]["payload"] == payload() for record in records)
    assert len(ledger.reservations) == 2
    assert len({row[3] for row in ledger.reservations}) == 2
    assert sum(row[1] == "json_retry" for row in ledger.events) == 1

    replay = ExperimentAPI(tmp_path, "replay", Limits(max_requests=4, max_output_tokens=256))
    assert component(replay)._json(payload(), dict) == {"probabilities": [0, 1, 0, 0]}
    assert replay.metrics["cache_hits"] == replay.metrics["logical_requests"] == 2
    assert len(requests) == 2


def test_retry_exhaustion_and_logical_limit_are_explicit(tmp_path: Path) -> None:
    calls: list[str] = []

    def create(**_kwargs: Any) -> SimpleNamespace:
        calls.append("call")
        return fake_response("[]")

    client = SimpleNamespace(responses=SimpleNamespace(create=create), max_retries=0)
    api = ExperimentAPI(tmp_path, "openai", Limits(max_requests=2, max_output_tokens=128), client)
    with pytest.raises(RequestFailed, match="Invalid JSON"):
        component(api)._json(payload(), dict)
    assert len(calls) == len(list((tmp_path / "openai").glob("*.json"))) == 2
    limited = ExperimentAPI(tmp_path / "limited", "openai", Limits(max_requests=1, max_output_tokens=128), client)
    with pytest.raises(BudgetExceeded, match="Logical API request budget"):
        component(limited)._json(payload(), dict)
    assert len(calls) == 3


def test_transport_error_with_unknown_billing_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = CountingLedger(unknown=True)
    monkeypatch.setattr("carl.icl.api._campaign_ledger", lambda: ledger)
    calls: list[str] = []

    def create(**_kwargs: Any) -> None:
        calls.append("call")
        raise ConnectionError("unknown provider billing")

    api = ExperimentAPI(tmp_path, "openai", Limits(),
                        SimpleNamespace(responses=SimpleNamespace(create=create), max_retries=0))
    with pytest.raises(RequestFailed, match="responses failed"):
        component(api)._json(payload(), dict)
    assert calls == ["call"] and ledger.unknown == ["1"]
    assert not any(row[1] == "json_retry" for row in ledger.events)
    first_record = next((tmp_path / "openai").glob("*.json")).read_bytes()
    another = ExperimentAPI(tmp_path, "openai", Limits(), api.client)
    with pytest.raises(RequestFailed, match="requires reconciliation"):
        component(another)._json(payload(), dict)
    assert calls == ["call"] and first_record == next((tmp_path / "openai").glob("*.json")).read_bytes()


def test_concurrent_callers_share_initial_and_retry_slots(tmp_path: Path) -> None:
    calls: list[str] = []
    calls_lock = threading.Lock()

    def create(**_kwargs: Any) -> SimpleNamespace:
        with calls_lock:
            calls.append("call")
            index = len(calls)
        return fake_response("{" if index == 1 else '{"ok":true}')

    client = SimpleNamespace(responses=SimpleNamespace(create=create), max_retries=0)
    components = [component(ExperimentAPI(tmp_path, "openai", Limits(), client)) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(lambda item: item._json(payload(), dict), components))
    assert values == [{"ok": True}, {"ok": True}]
    assert len(calls) == 2


def test_retry_setting_changes_policy_fingerprint(tmp_path: Path) -> None:
    api = ExperimentAPI(tmp_path, "mock", Limits())
    original = component(api, retries=0)
    retry = component(api, retries=1)
    assert original.fingerprint() != retry.fingerprint()
    locked = original.config.model_copy(update={"locked_policy_fingerprint": original.fingerprint(), "json_max_retries": 1})
    with pytest.raises(ValueError, match="Frozen policy fingerprint mismatch"):
        LunaComponents(locked, api, original.env, original.training)


def test_policy_retry_keeps_completed_rollout_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replies = iter(['{"probabilities":[1,0,0,0]}', '{"probabilities":[1,0,0,0}',
                    '{"probabilities":[1,0,0,0]}'])
    calls: list[str] = []

    def create(**_kwargs: Any) -> SimpleNamespace:
        calls.append("call")
        return fake_response(next(replies))

    client = SimpleNamespace(responses=SimpleNamespace(create=create), max_retries=0)
    policy = component(ExperimentAPI(tmp_path, "openai", Limits(), client))
    monkeypatch.setattr(policy, "policy_payload", lambda board, _target: {**payload(), "input": to_rows(board)})
    initial = from_rows(["#####", "#---#", "#@$.#", "#---#", "#####"])
    target = from_rows(["#####", "#--@#", "#-$.#", "#---#", "#####"])

    class CountedEnvironment:
        def __init__(self) -> None:
            self.real = environment(initial)
            self.actions = 0

        def is_solved(self, board: Any) -> bool:
            return bool(self.real.is_solved(board))

        def next_state(self, board: Any, action: int) -> Any:
            self.actions += 1
            return self.real.next_state(board, action)

    env = CountedEnvironment()
    result = rollout(env, policy, initial, target, cap=2, rng=np.random.default_rng(11))
    assert result.actions == [0, 0]
    assert env.actions == 2
    assert len(calls) == 3


@pytest.mark.parametrize("api_type", [CapturingAPI, CappedAPI])
def test_retry_uses_subclass_call_and_counters(tmp_path: Path, api_type: type[ExperimentAPI]) -> None:
    responses = iter(["{", '{"ok":true}'])
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: fake_response(next(responses))),
                             max_retries=0)
    kwargs = {"max_generator_calls": 4, "max_total_tokens": 10000} if api_type is CappedAPI else {}
    api = api_type(tmp_path, "openai", Limits(), client, **kwargs)
    assert component(api)._json(payload(), dict) == {"ok": True}
    assert api.metrics["policy_calls"] == 2
    if isinstance(api, CapturingAPI):
        assert len(api.responses) == 2


def test_capturing_api_reports_retry_transport_failure_slot(tmp_path: Path) -> None:
    calls: list[str] = []

    def create(**_kwargs: Any) -> SimpleNamespace:
        calls.append("call")
        if len(calls) == 1:
            return fake_response("{")
        raise ConnectionError("unknown billing")

    client = SimpleNamespace(responses=SimpleNamespace(create=create), max_retries=0)
    api = CapturingAPI(tmp_path, "openai", Limits(), client)
    with pytest.raises(RequestFailed, match="responses failed"):
        component(api)._json(payload(), dict)
    assert [row["status"] for row in api.responses] == ["completed", "error"]
    assert api.responses[1]["cache_attempt_slot"] == 1
