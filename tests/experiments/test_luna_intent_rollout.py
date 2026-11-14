"""Test the intent controller contract against the real Sokoban environment."""
import json
from dataclasses import asdict

import pytest

from carl.icl.boards import from_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.luna_intent_rollout import Action
from carl.icl.luna_intent_rollout import ActionAttempt
from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_rollout import build_prompt
from carl.icl.luna_intent_rollout import rollout

ROWS = ["#######", "#@$-.-#", "#-$-.-#", "#-----#", "#-----#", "#-----#", "#######"]
BOX_INTENT = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": []}


def setup_board():
    state = from_rows(ROWS)
    return environment(state), state


def test_no_op_counts_and_first_satisfaction_stops():
    env, state = setup_board()
    responses = iter(['{"action":"UP"}', '{"action":"RIGHT"}', '{"action":"DOWN"}'])
    events = []
    result = rollout(env, state, BOX_INTENT, lambda _: next(responses), cap=3,
                     task_goal="all boxes on goals", on_event=events.append)
    assert result.status == RolloutStatus.ACHIEVED
    assert result.achieved
    assert result.actions == [0, 3]
    assert result.model_calls == 2
    assert [step["no_op"] for step in result.steps] == [True, False]
    assert result.endpoint[1] == "#-@$.-#"
    first, second = [json.loads(step["prompt"].split("\n\n")[-1]) for step in result.steps]
    assert first["attempt_history"] == []
    assert second["attempt_history"] == [{"action": "UP", "no_op": True}]
    assert second["state"] == result.steps[0]["after"]
    assert [event["type"] for event in events] == ["request", "response", "action"] * 2
    json.dumps(asdict(result))


def test_preservation_violation_fails_even_if_effect_is_simultaneously_true():
    env, state = setup_board()
    intent = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}],
              "preserve": [{"predicate": "box_at", "row": 1, "col": 2}]}
    result = rollout(env, state, intent, lambda _: '{"action":"RIGHT"}', cap=4,
                     task_goal="all boxes on goals")
    assert result.status == RolloutStatus.PRESERVATION_VIOLATED
    assert not result.achieved
    assert result.steps[0]["intent_holds"]
    assert not result.steps[0]["preservation_holds"]
    assert result.actions == [3]


def test_conjunction_requires_all_predicates_at_same_step():
    env, state = setup_board()
    intent = {"effects": [{"predicate": "box_at", "row": 1, "col": 3},
                          {"predicate": "player_at", "row": 2, "col": 1}], "preserve": []}
    responses = iter(['{"action":"RIGHT"}', '{"action":"LEFT"}', '{"action":"DOWN"}'])
    result = rollout(env, state, intent, lambda _: next(responses), cap=4, task_goal="all boxes on goals")
    assert result.status == RolloutStatus.ACHIEVED
    assert result.actions == [3, 2, 1]
    assert [step["intent_holds"] for step in result.steps] == [False, False, True]
    question = json.loads(result.steps[2]["prompt"].split("\n\n")[-1])
    assert question["attempt_history"] == [{"action": "RIGHT", "no_op": False}, {"action": "LEFT", "no_op": False}]
    assert question["state"] == result.steps[1]["after"]


def test_task_completion_stops_without_claiming_a_different_intent():
    state = from_rows(["#####", "#@$.#", "#---#", "#---#", "#####"])
    intent = {"effects": [{"predicate": "player_at", "row": 2, "col": 1}], "preserve": []}
    result = rollout(environment(state), state, intent, lambda _: '{"action":"RIGHT"}', cap=2,
                     task_goal="all boxes on goals")
    assert result.status == RolloutStatus.TASK_SOLVED
    assert result.task_solved
    assert not result.achieved
    assert result.actions == [3]


def test_policy_input_only_contains_current_state_task_and_structured_intent():
    _, state = setup_board()
    intent = {**BOX_INTENT, "instruction": "PRIVATE-FUTURE-STATE-AND-ID"}
    prompt = build_prompt(state, intent, "all boxes on goals")
    question = json.loads(prompt.split("\n\n")[-1])
    assert set(question) == {"state", "state_coordinates", "intent", "task_goal", "attempt_history"}
    assert question["state"] == ROWS
    assert question["intent"] == BOX_INTENT
    assert "PRIVATE-FUTURE-STATE-AND-ID" not in prompt
    assert "later_observation" not in prompt
    assert "fragment_actions" not in prompt


def test_memory_accepts_only_actions_and_flags_without_future_observations():
    _, state = setup_board()
    prompt = build_prompt(state, BOX_INTENT, "all boxes on goals", [ActionAttempt(Action.UP, True)])
    question = json.loads(prompt.split("\n\n")[-1])
    assert question["attempt_history"] == [{"action": "UP", "no_op": True}]
    with pytest.raises(ValueError):
        build_prompt(state, BOX_INTENT, "all boxes on goals", [{"action": "UP", "no_op": True,
                                                                 "later_observation": "SECRET-FUTURE"}])
    with pytest.raises(ValueError):
        build_prompt(state, BOX_INTENT, "all boxes on goals", [ActionAttempt(Action.UP, "SECRET-FUTURE")])


def test_memory_is_reset_for_every_attempt():
    env, state = setup_board()
    first = rollout(env, state, BOX_INTENT, lambda _: '{"action":"UP"}', cap=1, task_goal="all boxes on goals")
    second = rollout(env, state, BOX_INTENT, lambda _: '{"action":"UP"}', cap=1, task_goal="all boxes on goals")
    for result in (first, second):
        question = json.loads(result.steps[0]["prompt"].split("\n\n")[-1])
        assert question["attempt_history"] == []


def test_invalid_response_records_raw_answer_without_adding_action():
    env, state = setup_board()
    result = rollout(env, state, BOX_INTENT, lambda _: '{"action":"LEFT","action":"RIGHT"}', cap=2,
                     task_goal="all boxes on goals")
    assert result.status == RolloutStatus.INVALID_RESPONSE
    assert result.actions == []
    assert result.model_calls == 1
    assert result.steps[0]["raw_response"] == '{"action":"LEFT","action":"RIGHT"}'
    assert result.steps[0]["after"] is None


def test_transport_failure_keeps_prior_steps_and_terminal_status():
    env, state = setup_board()
    count = 0

    def decide(_prompt):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("transport failed")
        return '{"action":"UP"}'

    result = rollout(env, state, BOX_INTENT, decide, cap=3, task_goal="all boxes on goals")
    assert result.status == RolloutStatus.API_ERROR
    assert result.actions == [0]
    assert result.model_calls == 2
    assert result.endpoint == ROWS
    assert "transport failed" in result.error


def test_invalid_intent_never_calls_policy():
    env, state = setup_board()
    intent = {"effects": [{"predicate": "box_at", "row": 1, "col": 2}], "preserve": []}
    result = rollout(env, state, intent, lambda _: (_ for _ in ()).throw(AssertionError()), cap=2,
                     task_goal="all boxes on goals")
    assert result.status == RolloutStatus.INVALID_INTENT
    assert result.model_calls == 0
