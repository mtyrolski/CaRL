"""Check primitive search uses the real simulator and charges all attempts."""
import numpy as np

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import to_rows
from carl.planners.greedy import AstarPlanner
from carl.solver.nodes import SearchTreeNode
from scripts.run_overnight_stage05_primitive import search


def board(boxes: int) -> tuple[dict, np.ndarray]:
    cells = np.ones((12, 12), dtype=int)
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 0
    cells[1, 1], cells[1, 2], cells[1, 3] = 5, 4, 2
    for col in range(1, boxes):
        cells[4, col] = 3
    state = np.eye(7, dtype=np.float32)[cells]
    return ({"family": f"12x12x{boxes}", "source_family": "fixture", "split": "evaluation", "board_rank": 0,
             "layout_id": canonical_layout_id(state), "trajectory_id": "fixture", "initial_state": to_rows(state)}, state)


class ConstantValue:
    env = None

    def get_value(self, state):
        return -0.5


def test_success_charges_noops_and_other_attempts_not_only_solution_length():
    record, _ = board(4)
    result = search(record, ConstantValue(), budget=10, wall_seconds=10)
    assert result["status"] == "completed"
    assert result["solved"] is True
    assert result["solution_actions"] == [3]
    assert result["solution_length"] == 1
    assert result["solve_at_transition"] == result["transitions"] == 4
    assert result["noop_actions"] == 2
    assert [event["action"] for event in result["trace"]] == [0, 1, 2, 3]
    assert result["value_calls"] == 1
    assert result["value_input_tokens"] == 146


def test_transition_budget_counts_wall_move_and_is_never_exceeded():
    record, _ = board(10)
    result = search(record, ConstantValue(), budget=1, wall_seconds=10)
    assert result["solved"] is False
    assert result["transitions"] == result["noop_actions"] == 1
    assert result["failure_reason"] == "transition_budget_exhausted"
    assert result["value_calls"] == 0


def test_value_sign_prioritises_smaller_predicted_distance():
    _, root = board(4)
    planner = AstarPlanner(root, value_weight=150.0, depth_weight=1.0)
    parent = planner.get()
    close = SearchTreeNode(root.copy(), -0.2, [1], parent)
    far = SearchTreeNode(root.copy(), -0.8, [1], parent)
    planner.add(far)
    planner.add(close)
    assert planner.get() is close


def test_nonfinite_value_is_recorded_as_infrastructure_failure():
    class BrokenValue(ConstantValue):
        def get_value(self, state):
            return float("nan")

    record, _ = board(4)
    result = search(record, BrokenValue(), budget=10, wall_seconds=10)
    assert result["status"] == "infra_error"
    assert result["solved"] is None
    assert result["transitions"] == 2
    assert result["value_calls"] == 1
    assert "nonfinite" in result["failure_reason"]
