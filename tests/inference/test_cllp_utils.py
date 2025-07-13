import os
import numpy as np
import pytest
from typing import List, Any, Tuple, Union

from carl.inference_components.cllp_utils import (
    trajectory_from_actions,
    verify_cllp_reaches_subgoals_from_initial_state,
    CLLPVerificationResult,
)
from carl.environment.env import GameEnv
from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy
from carl.inference_components.validator import BasicValidator
from carl.solver.nodes import SearchTreeNode, ValidationResult
from carl.utils.aliases import State

class DummyEnv(GameEnv):
    def __init__(self):
        self._state = 0
    @property
    def name(self) -> str:
        return "dummy"
    @property
    def tokenizer(self):
        raise NotImplementedError
    def detect_action(self, board_before: State, board_after: State) -> int:
        return int(board_after - board_before)
    @staticmethod
    def distribution_to_action(distribution):
        return 0
    def step(self, action: int) -> Tuple[State, float, bool, dict]:
        self._state += action
        done = False
        return self._state, 0.0, done, {}
    def next_state(self, state: State, action: int) -> State:
        return state + action
    def is_solved(self, board: State) -> bool:
        return False
    def state_to_repr(self, state: State, title=None) -> Any:
        return str(state)
    def many_states_to_repr(self, states: List[State], title=None) -> Any:
        return states
    def set_state(self, state: State) -> None:
        self._state = state

def test_trajectory_from_actions_simple_moves():
    """Test that trajectory replay collects correct states."""
    env = DummyEnv()
    start = 0
    actions = [1, 2, 3]
    traj = trajectory_from_actions(env, start, actions)
    assert traj == [0, 1, 3, 6]


def test_verify_cllp_reaches_subgoals_batch(monkeypatch):
    """Test verifying reachability returns correct summary metrics."""
    # Prepare dummy goals
    goals = [10, 20, 30]
    initial_state = 0
    # stub CLLP
    cllp = ConditionalLowLevelPolicy  # not used in validator stub
    # stub validator to return ValidationResult with is_valid True/False and low_level_nodes_visited counts
    class DummyResult:
        def __init__(self, is_valid, path, visited):
            self.is_valid = is_valid
            self.path = path
            self.low_level_nodes_visited = visited
    dummy_sequence = [DummyResult(True, [1], 2), DummyResult(False, [], 3), DummyResult(True, [2,3], 4)]
    def fake_is_valid(self, init, goal):
        idx = goals.index(goal)
        return dummy_sequence[idx]
    monkeypatch.setattr(BasicValidator, 'is_valid', fake_is_valid)
    # call verify
    def mk_env():
        return DummyEnv()  # reuse dummy env above
    result: CLLPVerificationResult = verify_cllp_reaches_subgoals_from_initial_state(
        cllp=cllp,
        goals=goals,
        initial_state=initial_state,
        env_creation_fn=mk_env,
        max_radius=5,
    )
    # assertions
    assert isinstance(result, CLLPVerificationResult)
    assert result.total_goals == 3
    assert pytest.approx(result.success_rate) == 2/3
    assert list(result.paths) == [[1], [], [2,3]]
    assert result.calls == 3
    # sums
    assert result.cllp_samples_in_calls == 2+3+4
    assert result.node_computations == 2+3+4
