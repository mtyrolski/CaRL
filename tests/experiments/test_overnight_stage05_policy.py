"""Check the direct policy's full-start, transition and cycle contract."""
from collections import namedtuple

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts.run_overnight_stage05_policy import run_board
from scripts.run_overnight_stage05_policy import validate_board

GeneratedAction = namedtuple("GeneratedAction", "action generation_metadata")


def board(boxes: int, *, final_push: bool = False) -> dict:
    cells = np.ones((12, 12), dtype=int)
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 0
    positions = [(4, col) for col in range(1, boxes + 1)]
    if boxes == 10:
        positions = [(4, col) for col in range(1, 11)]
    goals = [(8, col) for col in range(1, boxes + 1)]
    if final_push:
        positions = [(1, 2), (4, 1), (4, 2), (4, 3)]
        goals = [(1, 3), (4, 1), (4, 2), (4, 3)]
    for position in goals:
        cells[position] = 2
    for position in positions:
        cells[position] = 3 if position in goals else 4
    cells[1, 1] = 5
    state = np.eye(7, dtype=np.float32)[cells]
    return {"family": f"12x12x{boxes}", "source_family": "fixture", "split": "evaluation", "board_rank": 0,
            "layout_id": canonical_layout_id(state), "trajectory_id": "fixture", "initial_state": to_rows(state)}


class StaticPolicy:
    def __init__(self, action: int):
        self.action = action
        self.inputs = []
        self.env = None

    def get_actions(self, state):
        self.inputs.append(state.copy())
        return [GeneratedAction(self.action, {})]


class TwoStatePolicy(StaticPolicy):
    def get_actions(self, state):
        row, col = np.argwhere(np.isin(state.argmax(-1), (5, 6)))[0]
        return [GeneratedAction(3 if (row, col) == (1, 1) else 2, {})]


@pytest.mark.parametrize("boxes", [4, 10])
def test_wall_noop_is_a_real_counted_transition_and_deterministic_cycle(boxes):
    initial = board(boxes)
    policy = StaticPolicy(0)
    result = run_board(initial, policy, budget=10, wall_seconds=10)
    assert result["status"] == "completed"
    assert result["solved"] is False
    assert result["failure_reason"] == "deterministic_cycle"
    assert result["transitions"] == result["policy_calls"] == result["noop_actions"] == 1
    assert result["cycle_first_seen_transition"] == 0
    assert result["cycle_length"] == 1
    assert result["trace"][0]["completed"] is True
    assert result["policy_input_tokens"] == 146
    assert np.array_equal(policy.inputs[0], from_rows(initial["initial_state"]))
    assert policy.env.num_boxes == boxes


def test_cycle_repetition_stops_without_fabricating_future_transitions():
    result = run_board(board(4), TwoStatePolicy(0), budget=20, wall_seconds=10)
    assert result["failure_reason"] == "deterministic_cycle"
    assert result["transitions"] == result["policy_calls"] == 2
    assert result["cycle_first_seen_transition"] == 0
    assert result["cycle_length"] == 2
    assert result["noop_actions"] == 0
    assert [item["action"] for item in result["trace"]] == [3, 2]


def test_one_move_full_puzzle_success_and_solution_cost():
    result = run_board(board(4, final_push=True), StaticPolicy(3), budget=20, wall_seconds=10)
    assert result["solved"] is True
    assert result["solve_at_transition"] == result["solution_length"] == 1
    assert result["solution_verified"] is True
    assert result["verification_kind"] == "direct_simulation_goal_check"
    assert result["trace"][0]["boxes_on_goals"] == 4
    assert result["solve_wall_seconds"] is not None


def test_family_mismatch_cannot_silently_reconfigure_ten_boxes_as_four():
    record = board(10)
    record["family"] = "12x12x4"
    with pytest.raises(ValueError, match="box"):
        validate_board(record)


def test_four_boxes_on_goals_do_not_solve_a_ten_box_board():
    record = board(10)
    cells = from_rows(record["initial_state"]).argmax(-1)
    for col in range(1, 5):
        cells[4, col] = 1
        cells[8, col] = 3
    state = np.eye(7, dtype=np.float32)[cells]
    record["initial_state"] = to_rows(state)
    checked = validate_board(record)
    env = environment(checked)
    assert env.num_boxes == 10
    assert env.num_boxes_on_target(checked) == 4
    assert not env.is_solved(checked)


def test_policy_infrastructure_failure_is_not_a_behavioural_failure():
    class BrokenPolicy(StaticPolicy):
        def get_actions(self, state):
            raise RuntimeError("mock infrastructure failure")

    result = run_board(board(4), BrokenPolicy(0), budget=10, wall_seconds=10)
    assert result["status"] == "infra_error"
    assert result["solved"] is None
    assert result["transitions"] == 0
    assert result["policy_calls"] == 1
    assert "mock infrastructure failure" in result["failure_reason"]
