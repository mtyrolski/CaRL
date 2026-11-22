"""Payloads and frozen-policy behaviour of ICL components."""

import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from carl.icl.api import RequestFailed
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.types import BoardEnvironment
from tests.inference_components._luna_experiment_fixtures import sokoban_board as sokoban_board
from tests.inference_components._luna_experiment_fixtures import training_for


@pytest.mark.usefixtures("deterministic_token_estimates")
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


@pytest.mark.usefixtures("deterministic_token_estimates")
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


@pytest.mark.usefixtures("deterministic_token_estimates")
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


@pytest.mark.usefixtures("deterministic_token_estimates")
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


@pytest.mark.usefixtures("deterministic_token_estimates")
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


@pytest.mark.usefixtures("deterministic_token_estimates")
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


@pytest.mark.usefixtures("deterministic_token_estimates")
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


@pytest.mark.usefixtures("deterministic_token_estimates")
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


@pytest.mark.usefixtures("deterministic_token_estimates")
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
