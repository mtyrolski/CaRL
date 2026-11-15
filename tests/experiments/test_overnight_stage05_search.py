from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from scripts import run_overnight_stage05_search as run


def board() -> dict:
    cells = [["#" if row in (0, 11) or col in (0, 11) else "-" for col in range(12)] for row in range(12)]
    for col, tile in {1: "@", 2: "$", 3: ".", 5: "*", 7: "*", 9: "*"}.items():
        cells[1][col] = tile
    rows = ["".join(row) for row in cells]
    return {"initial_state": rows, "layout_id": canonical_layout_id(from_rows(rows)),
            "family": "12x12x4", "source_family": "test", "split": "development", "board_rank": 0}


def config(budget: int = 8) -> dict:
    return {"method": "ksubs", "ks": [8], "budget": budget, "board_wall_seconds": 10.0,
            "max_expansions": 100, "stop_epoch": time.time() + 20, "stop_monotonic": time.monotonic() + 20,
            "generation": {"num_beams": 4, "num_return_sequences": 4, "max_new_tokens": 145}}


class FixedCLLP:
    def __init__(self, action: int):
        self.env = None
        self.action = action

    def get_action(self, current: np.ndarray, target: np.ndarray) -> torch.Tensor:
        result = torch.zeros(4)
        result[self.action] = 1
        return result


class DummyValue:
    env = None

    def get_value(self, state: np.ndarray) -> float:
        return -1.0


@pytest.fixture(autouse=True)
def clean_worker():
    run.WORKER.clear()
    yield
    run.WORKER.clear()


def test_actual_board_count_and_goal_are_checked():
    example = board()
    state, env = run.valid_start(example)
    assert env.num_boxes == 4
    assert not env.is_solved(state)
    example["family"] = "12x12x10"
    with pytest.raises(ValueError, match="incorrect box or goal"):
        run.valid_start(example)


def test_ten_box_goal_check_uses_ten_boxes():
    cells = [["#" if row in (0, 11) or col in (0, 11) else "-" for col in range(12)] for row in range(12)]
    for col in range(1, 10):
        cells[2][col] = "*"
    cells[2][10], cells[3][9], cells[3][8] = ".", "$", "@"
    rows = ["".join(row) for row in cells]
    state = from_rows(rows)
    example = {"initial_state": rows, "layout_id": canonical_layout_id(state), "family": "12x12x10",
               "source_family": "test", "split": "development", "board_rank": 0}
    _, env = run.valid_start(example)
    assert env.num_boxes == 10 and not env.is_solved(state)
    cells[2][10], cells[3][9] = "*", "-"
    solved = from_rows(["".join(row) for row in cells])
    assert env.is_solved(solved)


def test_counts_every_noop_and_stops_exactly_at_budget(monkeypatch):
    example = board()
    state, env = run.valid_start(example)
    target = np.asarray(env.next_state(state, 1))
    run.WORKER.update(generators={8: SimpleNamespace(env=None)}, cllp=FixedCLLP(0), value=DummyValue())
    monkeypatch.setattr(run, "generate", lambda *_: ([(target, [0, 1, 2])], 145))
    result = run.search(example, config(2))
    assert not result["solved"]
    assert result["status"] == "completed"
    assert result["failure_reason"] == "transition_budget"
    assert result["transitions"] == result["policy_calls"] == result["noop_actions"] == 2
    assert result["controller_attempts"] == 1 and result["controller_hits"] == 0
    assert result["controller_cutoffs"] == 1 and result["controller_attempts_completed"] == 0
    assert result["controller_attempts_missed"] == 0 and result["policy_input_tokens"] == 2 * 291
    assert result["trace"][-1]["reason"] == "controller_cutoff_transition_budget"
    assert result["trace"][-1]["actions"] == [0, 0]


def test_terminal_during_controller_attempt_returns_executed_plan(monkeypatch):
    example = board()
    state, env = run.valid_start(example)
    target = np.asarray(env.next_state(state, 1))
    run.WORKER.update(generators={8: SimpleNamespace(env=None)}, cllp=FixedCLLP(3), value=DummyValue())
    monkeypatch.setattr(run, "generate", lambda *_: ([(target, [0, 1, 2])], 145))
    result = run.search(example, config())
    assert result["solved"] and result["solution_verified"]
    assert result["solution_actions"] == [3] and result["solution_length"] == 1
    assert result["solve_at_transition"] == result["transitions"] == 1
    assert result["controller_attempts"] == 1 and result["controller_hits"] == 0
    assert result["controller_task_solved_before_hit"] == 1 and result["controller_attempts_completed"] == 0
    assert result["trace"][-1]["reason"] == "task_solved"


def test_target_with_changed_box_count_is_rejected_without_controller_call(monkeypatch):
    example = board()
    state, _ = run.valid_start(example)
    rows = to_rows(state)
    rows[1] = rows[1].replace("$", "-", 1)
    invalid_target = from_rows(rows)
    cllp = FixedCLLP(3)
    run.WORKER.update(generators={8: SimpleNamespace(env=None)}, cllp=cllp, value=DummyValue())
    monkeypatch.setattr(run, "generate", lambda *_: ([(invalid_target, [0, 1, 2])], 145))
    result = run.search(example, config())
    assert not result["solved"] and result["candidates_valid"] == 0
    assert result["controller_attempts"] == result["transitions"] == result["policy_calls"] == 0
    assert result["target_rejections"] == {"invalid_target": 1}


def test_generated_candidates_are_all_assessed_after_budget_exhaustion(monkeypatch):
    example = board()
    state, env = run.valid_start(example)
    target = np.asarray(env.next_state(state, 1))
    invalid_rows = to_rows(state)
    invalid_rows[1] = invalid_rows[1].replace("$", "-", 1)
    invalid_target = from_rows(invalid_rows)
    run.WORKER.update(generators={8: SimpleNamespace(env=None)}, cllp=FixedCLLP(0), value=DummyValue())
    monkeypatch.setattr(run, "generate", lambda *_: (
        [(target, [0]), (invalid_target, [0]), (target, [0]), (target, [0])], 145))
    result = run.search(example, config(2))
    assert result["transitions"] == 2
    assert result["candidates_proposed"] == result["candidates_assessed"] == 4
    assert result["candidates_valid"] == 3
    assert result["controller_attempts"] == 1
    assert result["candidate_outcomes"] == {"controller_cutoff_transition_budget": 1, "invalid_target": 1,
                                            "not_attempted_transition_budget": 2}


def test_reconstructed_plan_contains_multiple_executed_edges(monkeypatch):
    example = board()
    state, env = run.valid_start(example)
    lower = np.asarray(env.next_state(state, 1))
    solved = np.asarray(env.next_state(state, 3))
    pending = iter([lower, solved])

    class TwoEdgeCLLP:
        env = None

        def get_action(self, current: np.ndarray, target: np.ndarray) -> torch.Tensor:
            if np.array_equal(target, lower):
                action = 1
            elif np.array_equal(current, lower):
                action = 0
            else:
                action = 3
            logits = torch.zeros(4)
            logits[action] = 1
            return logits

    run.WORKER.update(generators={8: SimpleNamespace(env=None)}, cllp=TwoEdgeCLLP(), value=DummyValue())
    monkeypatch.setattr(run, "generate", lambda *_: ([(next(pending), [0, 1, 2])], 145))
    result = run.search(example, config())
    assert result["solved"] and result["solution_verified"]
    assert result["solution_actions"] == [1, 0, 3]
    assert result["transitions"] == result["policy_calls"] == 3
    assert result["value_calls"] == 1 and result["controller_hits"] == 2
    assert result["controller_attempts_completed"] == 2 and result["controller_cutoffs"] == 0
    assert result["policy_input_tokens"] == 3 * 291 and result["value_input_tokens"] == 146


def test_full_horizon_miss_is_not_a_cutoff(monkeypatch):
    example = board()
    state, env = run.valid_start(example)
    target = np.asarray(env.next_state(state, 1))
    run.WORKER.update(generators={1: SimpleNamespace(env=None)}, cllp=FixedCLLP(0), value=DummyValue())
    monkeypatch.setattr(run, "generate", lambda *_: ([(target, [0, 1, 2])], 145))
    options = {**config(1), "ks": [1]}
    result = run.search(example, options)
    assert result["controller_attempts"] == result["controller_attempts_completed"] == 1
    assert result["controller_attempts_missed"] == 1 and result["controller_cutoffs"] == 0
    assert result["trace"][-1]["reason"] == "controller_target_not_reached"


def test_wall_cutoff_attempt_is_not_a_full_miss(monkeypatch):
    example = board()
    state, env = run.valid_start(example)
    target = np.asarray(env.next_state(state, 1))

    class SlowCLLP(FixedCLLP):
        def get_action(self, current: np.ndarray, target: np.ndarray) -> torch.Tensor:
            time.sleep(0.35)
            return super().get_action(current, target)

    run.WORKER.update(generators={8: SimpleNamespace(env=None)}, cllp=SlowCLLP(0), value=DummyValue())
    monkeypatch.setattr(run, "generate", lambda *_: ([(target, [0, 1, 2])], 145))
    result = run.search(example, {**config(), "board_wall_seconds": 0.3})
    assert result["status"] == "timeout" and result["controller_cutoffs"] == 1
    assert result["controller_attempts_missed"] == result["controller_attempts_completed"] == 0
    assert result["trace"][-1]["reason"] == "controller_cutoff_wall_limit"
