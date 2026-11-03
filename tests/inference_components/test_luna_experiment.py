import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from carl.icl.api import BudgetExceeded, RequestFailed
from carl.icl.boards import from_rows, level_key, to_rows
from carl.icl.calibration import calibrate, calibration_id, weighted_percentiles
from carl.icl.config import Calibration, ExperimentConfig, Limits, load_config
from carl.icl.datasets import environment, reverse_board, verify_solution
from carl.icl.diagnostics import audit_subgoal
from carl.icl.experiment import credentials, matrix_configs
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.search import rollout, search
from carl.icl.search_tree import SearchTreeRecorder, expansion_figure, tree_figure
from carl.icl.trace import trace_rollout
from carl.icl.types import Board, BoardEnvironment, RolloutPolicy, SearchComponents, TrainingData


@pytest.fixture
def board():
    return from_rows(["#####", "#---#", "#@$.#", "#---#", "#####"])


def training_for(board: Board) -> TrainingData:
    env = environment(board)
    after = env.next_state(board, 3)
    return {"contexts": [{"id": "train", "state": to_rows(board), "remaining_actions": 1,
                          "candidates": [{"k": 1, "actions": [3], "subgoal": to_rows(after)}],
                          "calibration_indices": [0]}]}


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


def test_cached_requests_still_consume_logical_budget(tmp_path):
    api = ExperimentAPI(tmp_path, "mock", Limits(max_requests=2))
    payload = dict(max_output_tokens=16, metadata=dict(component="policy"))
    for _ in range(2):
        api.call("responses", payload, mock=lambda: dict(output_text='{"x":1}', usage={}))
    assert api.metrics["requests"] == 1 and api.metrics["cache_hits"] == 1
    assert api.metrics["logical_requests"] == 2
    with pytest.raises(BudgetExceeded):
        api.call("responses", payload)


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


def test_all_three_components_have_few_shot_and_fixed_k(tmp_path, board):
    config = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    components = LunaComponents(
        config,
        ExperimentAPI(tmp_path, "mock", config.limits),
        environment(board),
        training_for(board),
    )
    payloads = [
        components.policy_payload(board, board),
        components.value_payload(board),
        components.generator_payload(board, 1),
    ]
    for payload in payloads:
        assert payload["model"] == "gpt-6-luna"
        assert len([m for m in payload["input"] if m["role"] == "assistant"]) == 1
        assert payload["text"] == {"format": {"type": "text"}, "verbosity": "low"}
        assert payload["stream"] is True
        assert payload["store"] is False
        assert payload["service_tier"] == "default"
        assert payload["input"][-1]["role"] == "user"
        for message in payload["input"]:
            expected = "output_text" if message["role"] == "assistant" else "input_text"
            assert message["content"][0]["type"] == expected
        assert payload["reasoning"] == {
            "effort": "none",
            "mode": "standard",
            "summary": "auto",
        }
    instructions = payloads[-1]["input"][0]["content"][0]["text"]
    assert "full explicit 5x5 board" in instructions
    assert "not a latent vector" in instructions
    generator = payloads[-1]
    questions = [
        json.loads(m["content"][0]["text"])
        for m in generator["input"]
        if m["role"] == "user"
    ]
    assert all(q["k"] == 1 and "u" not in q for q in questions)
    with pytest.raises(ValueError, match="No eligible"):
        components.generator_payload(board, 2)


def test_generator_rejects_latents_and_wrong_size_boards(tmp_path, board, monkeypatch):
    config = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    api = ExperimentAPI(tmp_path, "mock", config.limits)
    components = LunaComponents(config, api, environment(board), training_for(board))
    monkeypatch.setattr(components, "_json", lambda _payload, _mock: {"latent": [0.1, 0.2]})
    with pytest.raises(RequestFailed, match="invalid subgoals"):
        components.propose(board, 1)
    monkeypatch.setattr(components, "_json", lambda _payload, _mock: {
        "subgoals": [{"latent": [0.1]}, {"rows": ["###", "#@#", "###"]}]
    })
    # The configured maximum count is enforced before individual board validation.
    with pytest.raises(RequestFailed, match="invalid subgoals"):
        components.propose(board, 1)
    monkeypatch.setattr(components, "_json", lambda _payload, _mock: {"subgoals": [{"rows": ["###", "#@#", "###"]}]})
    assert components.propose(board, 1) == []
    assert components.diagnostics["invalid_proposals"] == 1


def test_configurable_prompts_and_calibration_policy_fingerprint(tmp_path, board):
    base = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    env, training = environment(board), training_for(board)

    def make(config):
        return LunaComponents(config, ExperimentAPI(tmp_path, "mock", config.limits), env, training)

    original = make(base)
    config = base.model_dump()
    config["prompts"]["generator"] = "A new generator instruction."
    generator_changed = make(ExperimentConfig.model_validate(config))
    assert original.fingerprint() == generator_changed.fingerprint()
    prompt = generator_changed.generator_payload(board, 1)["input"][0]["content"][0]["text"]
    assert "A new generator instruction." in prompt and "full explicit 5x5 board" in prompt
    config["prompts"]["policy"] = "A new policy instruction."
    policy_changed = make(ExperimentConfig.model_validate(config))
    assert original.fingerprint() != policy_changed.fingerprint()
    assert policy_changed.policy_payload(board, board)["input"][0]["content"][0]["text"].startswith(
        "A new policy instruction."
    )
    structured = make(base.model_copy(update={"policy_response_format": "json_schema"}))
    assert structured.fingerprint() != original.fingerprint()
    assert structured.policy_payload(board, board)["text"]["format"]["type"] == "json_schema"


def test_coordinate_policy_adds_only_board_derived_features_and_changes_fingerprint(tmp_path, board):
    base = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    augmented = base.model_copy(update={"policy_coordinate_features": True})
    env, training = environment(board), training_for(board)
    original = LunaComponents(base, ExperimentAPI(tmp_path, "mock", base.limits), env, training)
    coordinates = LunaComponents(augmented, ExperimentAPI(tmp_path, "mock", base.limits), env, training)
    target = np.asarray(env.next_state(board, 3))
    baseline_payload = original.policy_payload(board, target)
    payload = coordinates.policy_payload(board, target)
    assert original.fingerprint() != coordinates.fingerprint()
    assert original.generator_payload(board, 1) == coordinates.generator_payload(board, 1)
    assert original.value_payload(board) == coordinates.value_payload(board)
    assert "zero-based" in payload["input"][0]["content"][0]["text"]
    for baseline, encoded in zip(baseline_payload["input"], payload["input"], strict=True):
        if encoded["role"] != "user":
            continue
        question = json.loads(encoded["content"][0]["text"])
        old_question = json.loads(baseline["content"][0]["text"])
        assert question["state"] == old_question["state"]
        assert question["subgoal"] == old_question["subgoal"]
        assert question["state_coordinates"]["player_rc"] == [2, 1]
        assert question["subgoal_coordinates"]["player_rc"] == [2, 2]
        assert question["subgoal_coordinates"]["boxes_rc"] == [[2, 3]]
        assert question["subgoal_coordinates"]["goals_rc"] == [[2, 3]]


def test_layout_generator_retries_invalid_board_with_feedback(tmp_path, board, monkeypatch):
    config = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1,
        generator_layout_schema=True, generator_response_format="json_schema", generator_max_attempts=2)
    env = environment(board)
    components = LunaComponents(config, ExperimentAPI(tmp_path, "mock", config.limits), env, training_for(board))
    target = np.asarray(env.next_state(board, 3))
    invalid = to_rows(board)
    invalid[2] = invalid[2].replace("@", "-")

    def encoded(rows):
        return {f"row_{index:02d}": row for index, row in enumerate(rows)}

    responses = iter([{"subgoals": [encoded(invalid)]}, {"subgoals": [encoded(to_rows(target))]}])
    messages = []

    def respond(payload, _mock):
        messages.append(payload["input"].copy())
        return next(responses)

    monkeypatch.setattr(components, "_json", respond)
    proposals = components.propose(board, 1)
    assert len(proposals) == 1
    np.testing.assert_array_equal(proposals[0], target)
    assert components.diagnostics["invalid_proposals"] == 1
    assert len(messages[1]) == len(messages[0]) + 2
    assert "exactly one player" in messages[1][-1]["content"][0]["text"]
    schema = components.generator_payload(board, 1)["text"]["format"]["schema"]
    assert schema["properties"]["subgoals"]["items"]["properties"]["row_00"]["pattern"] == "^#####$"


def test_generator_does_not_probe_simulator_for_proposals(tmp_path, board, monkeypatch):
    config = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    target = from_rows(["#####", "#---#", "#@-*#", "#---#", "#####"])

    def forbidden(*_args):
        pytest.fail("Live generation must not query the simulator")

    env = SimpleNamespace(next_state=forbidden, is_solved=forbidden)
    components = LunaComponents(config, ExperimentAPI(tmp_path, "mock", config.limits), cast(BoardEnvironment, env), training_for(board))
    monkeypatch.setattr(components, "_json", lambda _payload, _mock: {"subgoals": [{"rows": to_rows(target)}]})
    proposals = components.propose(board, 1)
    assert len(proposals) == 1
    np.testing.assert_array_equal(proposals[0], target)
    monkeypatch.setattr(components, "_json", lambda _payload, _mock: {"value": 0.5})
    assert components.value(board) == 0.5


def test_remaining_value_uses_training_trajectory_without_terminal_override(tmp_path, board, monkeypatch):
    config = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1,
        value_examples="trajectory", value_target="remaining_actions",
        value_response_format="json_schema")
    env, training = environment(board), training_for(board)
    target = np.asarray(env.next_state(board, 3))
    training["value_examples"] = [
        {"context": "train", "state": to_rows(board), "remaining_actions": 1},
        {"context": "train", "state": to_rows(target), "remaining_actions": 0},
    ]
    api = ExperimentAPI(tmp_path, "mock", config.limits)
    components = LunaComponents(config, api, env, training)
    payload = components.value_payload(board)
    answers = [json.loads(m["content"][0]["text"]) for m in payload["input"] if m["role"] == "assistant"]
    assert answers == [{"remaining_actions": 1}, {"remaining_actions": 0}]
    assert payload["text"]["format"]["type"] == "json_schema"
    monkeypatch.setattr(components, "_json", lambda _payload, _mock: {"remaining_actions": 5})
    assert components.value(board) == pytest.approx(config.discount ** 5)
    assert components.value(target) == pytest.approx(config.discount ** 5)
    base = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    without_extra = training_for(board)
    normal = LunaComponents(base, api, env, without_extra)
    extra = LunaComponents(base, api, env, training)
    assert normal.fingerprint() == extra.fingerprint()


def test_subgoal_audit_has_witness_and_reports_inconclusive_limits(board):
    env = environment(board)
    target = np.asarray(env.next_state(board, 3))
    reachable = audit_subgoal(env, board, target, cap=1)
    assert reachable.status == "reachable" and reachable.actions == (3,)
    assert reachable.simulator_probes == 4
    assert audit_subgoal(env, board, target, cap=0).status == "unreachable_within_cap"
    assert audit_subgoal(env, board, target, cap=1, max_states=1).status == "node_limit"


def test_verified_32x32x10_reverse_board():
    board, actions, metadata = reverse_board(32, 10, 123, pulls=20)
    states = verify_solution(board, actions)
    assert board.shape == (32, 32, 7)
    assert int(np.isin(board.argmax(-1), [3, 4]).sum()) == 10
    assert len(actions) > 20 and metadata["boxes_off_goal"] >= 5
    assert all(level_key(s) == level_key(board) for s in states)


def test_config_nested_inheritance_and_matched_matrix(tmp_path):
    (tmp_path / "base.json").write_text(
        json.dumps({"calibration": {"limits": {"max_requests": 55, "timeout_seconds": 8}}})
    )
    (tmp_path / "child.json").write_text(
        json.dumps({"extends": "base.json", "calibration": {"limits": {"max_requests": 12}}})
    )
    config = load_config(tmp_path / "child.json")
    assert config.calibration.limits.timeout_seconds == 8
    assert config.calibration.limits.max_requests == 12
    for name in ("matrix_pilot.json", "matrix_smoke.json", "matrix_embeddings.json"):
        matrix_configs(Path("configs/icl/experiments") / name)
    with pytest.raises(ValueError):
        ExperimentConfig(method="relative", scales=[1, 4])


def test_prompt_config_overrides_merge_and_reject_unknown_fields(tmp_path):
    (tmp_path / "base.json").write_text(
        json.dumps({"prompts": {"policy": "First policy", "value": "Shared value"}}),
        encoding="utf-8",
    )
    (tmp_path / "child.json").write_text(
        json.dumps({"extends": "base.json", "prompts": {"policy": "Updated policy"}}),
        encoding="utf-8",
    )
    config = load_config(tmp_path / "child.json")
    assert config.prompts.policy == "Updated policy"
    assert config.prompts.value == "Shared value"
    assert config.prompts.generator
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({"prompts": {"not_a_component": "bad"}})


def test_placeholder_key_fails_before_network(monkeypatch):
    monkeypatch.setattr("carl.icl.experiment.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("OPENAI_API_KEY", "replace-with-your-openai-api-key")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        credentials("openai")
    credentials("mock")


@pytest.mark.parametrize(
    "grid",
    [
        {"relative_scales": []},
        {"relative_scales": [[0.8, 0.2]]},
        {"relative_scales": [[1.0, 0.5], [0.8, 0.2], [0.7, 0.3]]},
        {"fixed_scales": [[8, 4], [8, 4], [16, 8]]},
        {"fixed_scales": [[4, 8], [12, 4], [16, 8]]},
    ],
)
def test_tuning_rejects_unmatched_or_invalid_grids(grid):
    with pytest.raises(ValueError):
        ExperimentConfig(tuning=grid)


def test_tuning_grids_can_be_configured_with_inheritance(tmp_path):
    base = {"tuning": {"relative_scales": [[0.8, 0.3]], "fixed_scales": [[6, 2]]}}
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "experiment.json").write_text(
        json.dumps({"extends": "base.json"}), encoding="utf-8"
    )
    config = load_config(tmp_path / "experiment.json")
    assert config.tuning.relative_scales == [(0.8, 0.3)]
    assert config.tuning.fixed_scales == [(6, 2)]


def test_real_sdk_all_three_roles_and_incomplete_usage_accounting(tmp_path, board):
    httpx = pytest.importorskip("httpx")
    OpenAI = pytest.importorskip("openai").OpenAI
    config = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    training = training_for(board)
    captured = []
    incomplete = False

    def handle(request):
        payload = json.loads(request.content)
        captured.append(payload)
        role = payload["metadata"]["component"]
        answers = dict(
            policy=dict(probabilities=[0, 0, 0, 1]),
            value=dict(value=0.5),
            generator=dict(
                subgoals=[dict(rows=training["contexts"][0]["candidates"][0]["subgoal"])]
            ),
        )
        response = dict(
            id="resp_test",
            object="response",
            created_at=0,
            model="gpt-6-luna",
            status="incomplete" if incomplete else "completed",
            output=[
                dict(
                    type="message",
                    id="msg_test",
                    role="assistant",
                    status="completed",
                    content=[
                        dict(type="output_text", text=json.dumps(answers[role]), annotations=[])
                    ],
                ),
            ],
            usage=dict(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                input_tokens_details=dict(cached_tokens=20),
                output_tokens_details=dict(reasoning_tokens=0),
            ),
        )
        event = {
            "type": "response.incomplete" if incomplete else "response.completed",
            "sequence_number": 0,
            "response": response,
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"event: {event['type']}\ndata: {json.dumps(event)}\n\n",
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = OpenAI(api_key="unit-test-not-a-key", http_client=transport, max_retries=0)
        api = ExperimentAPI(tmp_path, "openai", config.limits, client)
        components = LunaComponents(config, api, environment(board), training)
        np.testing.assert_array_equal(components.probabilities(board, board), [0, 0, 0, 1])
        assert components.value(board) == 0.5
        assert len(components.propose(board, 1)) == 1
        assert {p["metadata"]["component"] for p in captured} == {"policy", "value", "generator"}
        assert all(p["stream"] and p["tools"] == [] for p in captured)
        assert api.metrics["output_tokens"] == 30
        completed = [json.loads(path.read_text()) for path in (tmp_path / "openai").glob("*.json")]
        assert len(completed) == 3
        assert all(record["response"]["output"][0]["content"][0]["text"]
                   == record["output_text"] for record in completed)
        assert all(record["response"]["usage"]["input_tokens"] == 100 for record in completed)
        incomplete = True
        other = np.asarray(environment(board).next_state(board, 0), dtype=np.float32)
        with pytest.raises(RequestFailed):
            components.value(other)
        assert api.metrics["output_tokens"] == 40  # Incomplete responses can still be billed.
        records = [json.loads(path.read_text()) for path in (tmp_path / "openai").glob("*.json")]
        failed = next(record for record in records if record["status"] == "error")
        assert failed["response"]["status"] == "incomplete"


def test_total_input_budget_charges_cached_calls(tmp_path):
    payload = dict(max_output_tokens=16, metadata=dict(component="value"))
    first = ExperimentAPI(tmp_path, "mock", Limits())
    first.call("responses", payload, mock=lambda: dict(output_text='{"value":0.5}', usage={}))
    tokens = int(first.metrics["logical_input_tokens"])
    api = ExperimentAPI(tmp_path, "mock", Limits(max_total_input_tokens=tokens))
    api.call("responses", payload)
    with pytest.raises(BudgetExceeded, match="Total estimated input"):
        api.call("responses", payload)
    assert api.metrics["requests"] == 0 and api.metrics["cache_hits"] == 1


def test_relative_and_absolute_use_same_weighted_examples(tmp_path, board):
    relative = ExperimentConfig(few_shot=1)
    training = training_for(board)
    components = LunaComponents(
        relative, ExperimentAPI(tmp_path, "mock", relative.limits), environment(board), training
    )
    examples = [
        dict(
            training["contexts"][0]["candidates"][0],
            state=to_rows(board),
            context="train",
            weight=w,
            u=u,
            d=d,
        )
        for w, u, d in [(0.2, 0.1, 1), (0.8, 0.6, 3)]
    ]
    components.calibrated = dict(
        status="completed",
        provider="mock",
        policy_fingerprint=components.fingerprint(),
        examples=examples,
    )
    selected_relative = components.generator_examples(board, 0.9, False)
    absolute = ExperimentConfig(method="absolute", scales=[8, 4], few_shot=1)
    absolute_components = LunaComponents(
        absolute, components.api, components.env, training, calibrated=components.calibrated
    )
    assert absolute_components.generator_examples(board, 8, False) == selected_relative


def test_locked_policy_rejects_changed_prompt_and_sampling(tmp_path, board):
    base = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    training, env = training_for(board), environment(board)
    api = ExperimentAPI(tmp_path, "mock", base.limits)
    fingerprint = LunaComponents(base, api, env, training).fingerprint()
    locked = base.model_copy(update={"locked_policy_fingerprint": fingerprint})
    assert LunaComponents(locked, api, env, training).fingerprint() == fingerprint
    with pytest.raises(ValueError, match="Frozen policy fingerprint mismatch"):
        LunaComponents(locked.model_copy(update={"policy_temperature": 0.7}), api, env, training)
    changed_prompts = locked.prompts.model_copy(update={"policy": "Changed policy prompt"})
    with pytest.raises(ValueError, match="Frozen policy fingerprint mismatch"):
        LunaComponents(locked.model_copy(update={"prompts": changed_prompts}), api, env, training)
    with pytest.raises(ValueError, match="Frozen policy fingerprint mismatch"):
        changed_training = training_for(board)
        changed_training["contexts"][0]["remaining_actions"] = 2
        LunaComponents(locked, api, env, changed_training)
