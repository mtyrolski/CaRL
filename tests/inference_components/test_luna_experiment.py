"""Search, rollout and calibration behaviour for ICL components."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from carl.icl.api import BudgetExceeded
from carl.icl.api import RequestFailed
from carl.icl.boards import from_rows
from carl.icl.calibration import calibrate
from carl.icl.calibration import calibration_id
from carl.icl.calibration import weighted_percentiles
from carl.icl.config import Calibration
from carl.icl.config import ExperimentConfig
from carl.icl.config import Limits
from carl.icl.datasets import environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.search import rollout
from carl.icl.search import search
from carl.icl.search_tree import SearchTreeRecorder
from carl.icl.search_tree import expansion_figure
from carl.icl.search_tree import tree_figure
from carl.icl.trace import trace_rollout
from carl.icl.types import BoardEnvironment
from carl.icl.types import RolloutPolicy
from carl.icl.types import SearchComponents
from tests.inference_components._luna_experiment_fixtures import sokoban_board as sokoban_board
from tests.inference_components._luna_experiment_fixtures import training_for


def test_weighted_tie_midpoints_and_zero_hit_rejection():
    weights, percentiles = weighted_percentiles([2, 2, 5], [1, 3, 4])
    np.testing.assert_allclose(weights, [0.125, 0.375, 0.5])
    np.testing.assert_allclose(percentiles, [0.25, 0.25, 0.75])
    with pytest.raises(ValueError):
        weighted_percentiles([1], [0])


def test_terminal_checked_before_target_and_attempts_include_noops(board):
    env = environment(board)
    target = from_rows(["#####", "#@--#", "#--*#", "#---#", "#####"])
    right = cast(RolloutPolicy, SimpleNamespace(probabilities=lambda s, t: np.array([0., 0., 0., 1.])))
    outcome = rollout(env, right, board, target, cap=10, rng=np.random.default_rng(0))
    assert outcome.solved and not outcome.reached and outcome.actions == [3]
    left = cast(RolloutPolicy, SimpleNamespace(probabilities=lambda s, t: np.array([0., 0., 1., 0.])))
    outcome = rollout(env, left, board, target, cap=3, rng=np.random.default_rng(0))
    assert not outcome.solved and not outcome.reached and len(outcome.actions) == 3
    np.testing.assert_array_equal(outcome.state, board)


def test_trace_uses_shared_rollout_with_same_random_actions_and_stopping(board):
    env = environment(board)
    target = from_rows(["#####", "#@--#", "#--*#", "#---#", "#####"])
    policy = cast(RolloutPolicy, SimpleNamespace(probabilities=lambda _state, _target: np.array([0.0, 0.0, 0.5, 0.5])))
    expected = rollout(env, policy, board, target, cap=5, rng=np.random.default_rng(8))
    actual, steps = trace_rollout(env, policy, board, target, cap=5, rng=np.random.default_rng(8))
    assert actual.actions == expected.actions == [step.action for step in steps]
    assert (actual.reached, actual.solved) == (expected.reached, expected.solved)
    np.testing.assert_array_equal(actual.state, expected.state)
    for index, step in enumerate(steps):
        np.testing.assert_array_equal(step.probabilities, [0.0, 0.0, 0.5, 0.5])
        before = board if index == 0 else steps[index - 1].successor
        np.testing.assert_array_equal(step.state, before)
        np.testing.assert_array_equal(step.successor, env.next_state(before, step.action))
        assert step.changed == (not np.array_equal(before, step.successor))


def test_trace_initial_terminal_has_no_policy_call(board):
    def forbidden(*_args):
        pytest.fail("A target already reached must not query the policy")

    result, steps = trace_rollout(
        environment(board), cast(RolloutPolicy, SimpleNamespace(probabilities=forbidden)), board, board,
        cap=4, rng=np.random.default_rng(1),
    )
    assert result.reached and not result.actions and not steps


@pytest.mark.usefixtures("deterministic_token_estimates")
@pytest.mark.parametrize(
    "probabilities",
    [[True, 0, 0, 0], ["1", 0, 0, 0], [0, 0, 0, 0], [1.1, 0, 0, 0], [1, 0, 0]],
)
def test_policy_rejects_invalid_json_probabilities(tmp_path, board, probabilities, monkeypatch):
    config = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    api = ExperimentAPI(tmp_path, "mock", config.limits)
    components = LunaComponents(config, api, environment(board), training_for(board))
    monkeypatch.setattr(components, "_json", lambda _payload, _mock: {"probabilities": probabilities})
    with pytest.raises(RequestFailed, match="Policy must return"):
        components.probabilities(board, board)


def test_search_checks_terminal_and_budget_on_failed_rollout(board):
    env = environment(board)
    target = from_rows(["#####", "#@--#", "#--*#", "#---#", "#####"])
    config = ExperimentConfig(action_budget=2, few_shot=1)
    components = SimpleNamespace(
        value=lambda s: 0.5, propose=lambda s, q: [target], probabilities=lambda s, t: np.array([0., 0., 0., 1.])
    )
    result = search(env, cast(SearchComponents, components), board, config, 1)
    assert result.solved and result.action_path == [3] and result.attempted_actions == 1
    components.probabilities = lambda s, t: np.array([0., 0., 1., 0.])
    result = search(env, cast(SearchComponents, components), board, config, 1)
    assert not result.solved and result.attempted_actions == 2
    assert result.accepted == 0 and result.rejected == 1
    assert result.finish_reason == "action_budget"


def test_primitive_fallback_never_calls_generator_or_policy(board):
    def forbidden(*args):
        pytest.fail("Primitive fallback queried generator or policy")

    components = SimpleNamespace(value=lambda s: 0.5, propose=forbidden, probabilities=forbidden)
    result = search(
        environment(board),
        cast(SearchComponents, components),
        board,
        ExperimentConfig(method="primitive", action_budget=4),
        1,
    )
    assert result.solved and result.attempted_actions == 4


@pytest.mark.parametrize("method", ["fixed", "primitive", "direct"])
def test_search_recorder_preserves_behavior_and_records_real_attempts(board, method):
    env = environment(board)
    target = from_rows(["#####", "#@--#", "#--*#", "#---#", "#####"])
    calls = []
    components = SimpleNamespace(
        value=lambda _state: calls.append("value") or 0.5,
        propose=lambda _state, _scale: calls.append("generator") or [target],
        probabilities=lambda _state, _target: calls.append("policy") or np.array([0., 0., 0., 1.]),
    )
    config = ExperimentConfig(method=method, scales=[2, 1], action_budget=8)
    expected = search(env, cast(SearchComponents, components), board, config, 7)
    expected_calls = calls.copy()
    calls.clear()
    recorder = SearchTreeRecorder()
    actual = search(env, cast(SearchComponents, components), board, config, 7, observer=recorder)
    assert calls == expected_calls
    for key in vars(actual):
        if key != "seconds":
            assert getattr(actual, key) == getattr(expected, key)
    assert actual.solved
    assert len(recorder.expansions) == actual.expansions
    assert sum(len(node.actions) for node in recorder.nodes) == actual.attempted_actions
    assert recorder.solution_node is not None
    assert recorder.solution_nodes() == {0, recorder.solution_node}
    assert recorder.nodes[recorder.solution_node].solved
    assert recorder.summary()["attempted_simulator_actions"] == actual.attempted_actions
    assert tree_figure(recorder).data and expansion_figure(recorder).data


def test_search_recorder_keeps_partial_actions_on_provider_failure(board):
    env = environment(board)
    target = from_rows(["#####", "#@--#", "#--*#", "#---#", "#####"])
    calls = 0

    def policy(_state, _target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RequestFailed("simulated failure")
        return np.array([0., 0., 1., 0.])  # No-op; the second call fails.

    components = SimpleNamespace(value=lambda _s: 0.5, propose=lambda _s, _q: [target],
                                 probabilities=policy)
    recorder = SearchTreeRecorder()
    result = search(env, cast(SearchComponents, components), board, ExperimentConfig(action_budget=8), 1, observer=recorder)
    assert result.finish_reason == "api_error"
    assert result.attempted_actions == 1
    assert recorder.nodes[-1].status == "interrupted"
    assert recorder.nodes[-1].actions == (2,)
    assert sum(len(node.actions) for node in recorder.nodes) == result.attempted_actions


def test_shorter_paths_reopen_states_and_stale_entries_are_skipped():
    states = [np.eye(7, dtype=np.float32)[i].reshape(1, 1, 7) for i in range(4)]

    def identity(s):
        return int(s.argmax(-1)[0, 0])

    edges = {(0, 0): 1, (1, 0): 2, (2, 0): 3, (0, 1): 3}
    env = SimpleNamespace(
        is_solved=lambda s: False,
        next_state=lambda s, a: states[edges.get((identity(s), a), identity(s))],
    )

    def propose(s, scale):
        assert scale > 0
        if identity(s) == 0:
            return [states[2 if scale == 0.9 else 3]]
        return [states[3]] if identity(s) == 2 and scale == 0.9 else []

    components = SimpleNamespace(
        value=lambda s: 0.5,
        propose=propose,
        probabilities=lambda s, t: np.array([0., 1., 0., 0.])
        if identity(s) == 0 and identity(t) == 3
        else np.array([1., 0., 0., 0.]),
    )
    result = search(
        cast(BoardEnvironment, env), cast(SearchComponents, components), states[0], ExperimentConfig(action_budget=64, max_expansions=30), 1
    )
    assert result.reopened == 1
    assert result.finish_reason == "frontier_exhausted"


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_cached_requests_still_consume_logical_budget(tmp_path):
    api = ExperimentAPI(tmp_path, "mock", Limits(max_requests=2))
    payload = dict(max_output_tokens=16, metadata=dict(component="policy"))
    for _ in range(2):
        api.call("responses", payload, mock=lambda: dict(output_text='{"x":1}', usage={}))
    assert api.metrics["requests"] == 1 and api.metrics["cache_hits"] == 1
    assert api.metrics["logical_requests"] == 2
    with pytest.raises(BudgetExceeded):
        api.call("responses", payload)


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_calibration_exact_hits_and_provider_separation(tmp_path, board, monkeypatch):
    config = ExperimentConfig(few_shot=1, calibration=Calibration(trials=2, action_cap=2))
    training = training_for(board)
    mock = LunaComponents(
        config,
        ExperimentAPI(tmp_path, "mock", config.calibration.limits),
        environment(board),
        training,
    )
    monkeypatch.setattr(mock, "probabilities", lambda s, t: np.array([0., 0., 0., 1.]))
    calibrated = calibrate(config, mock, tmp_path / "labels")
    assert calibrated["status"] == "completed"
    assert calibrated["examples"][0]["u"] == 0.5
    assert calibrated["examples"][0]["d"] == 1
    assert calibrated["calibration_actions"] == 2
    live = LunaComponents(
        config,
        ExperimentAPI(tmp_path, "replay", config.limits),
        environment(board),
        training,
        calibrated=calibrated,
    )
    assert calibration_id(config, mock) != calibration_id(config, live)
    with pytest.raises(ValueError, match="Mock calibration"):
        live.generator_examples(board, 0.5, False)
