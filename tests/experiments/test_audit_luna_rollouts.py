"""Check that the offline rollout audit catches altered actions and inputs."""

import json
from dataclasses import asdict

import pytest

from carl.icl.boards import from_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.luna_intent_rollout import POLICY_PROMPT
from carl.icl.luna_intent_rollout import rollout
from scripts.audit_luna_rollouts import audit_rollout


@pytest.fixture
def example():
    board = ["#######", "#@$-.-#", "#-$-.-#", "#-----#", "#-----#", "#-----#", "#######"]
    intent = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": []}
    case = {"record_id": "id", "phase": "test", "family": "family", "layout_id": "layout",
            "context_id": "context", "start_observation": board, "task_goal": "all boxes on goals",
            "intent": intent}
    state = from_rows(board)
    answers = iter(['{"action":"UP"}', '{"action":"RIGHT"}'])
    result = rollout(environment(state), state, intent, lambda _: next(answers), cap=3, task_goal=case["task_goal"])
    row = {key: case[key] for key in ("record_id", "phase", "family", "layout_id", "context_id")}
    row["result"] = asdict(result)
    row["cap"] = 3
    return case, row


def test_audit_replays_actions_and_counts_noops(example):
    case, row = example
    assert audit_rollout(case, row, POLICY_PROMPT) == {"executed_actions": 2, "model_calls": 2}


def test_audit_rejects_altered_state(example):
    case, row = example
    row["result"]["steps"][1]["after"] = row["result"]["steps"][1]["before"]
    with pytest.raises(ValueError, match="transition differs from simulator"):
        audit_rollout(case, row, POLICY_PROMPT)


def test_audit_rejects_future_data_in_prompt(example):
    case, row = example
    step = row["result"]["steps"][0]
    prefix, body = step["prompt"].split("\n\n", 1)
    question = json.loads(body)
    question["later_observation"] = "unpermitted future state"
    step["prompt"] = prefix + "\n\n" + json.dumps(question)
    with pytest.raises(ValueError, match="unexpected or incorrect inputs"):
        audit_rollout(case, row, POLICY_PROMPT)


def test_audit_rejects_a_misreported_stopping_status(example):
    case, row = example
    row["result"]["status"] = "action_limit"
    row["result"]["achieved"] = False
    with pytest.raises(ValueError, match="stopping status differs from simulator"):
        audit_rollout(case, row, POLICY_PROMPT)
