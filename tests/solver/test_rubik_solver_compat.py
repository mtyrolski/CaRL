"""Regression tests for AdaSubS over a mutable native simulator."""

from functools import partial
from unittest.mock import Mock

import numpy as np
import torch

from carl.environment.gym_rubik.rubik_env import RubikEnv
from carl.environment.gym_rubik.tokenizer import RubikCubeTokenizer
from carl.inference_components.validator import BasicValidator
from carl.planners.adasubs import AdasubsPlanner
from carl.solver.subgoal_search import Solver


def test_rubik_diagnostic_replay_preserves_terminal_subgoal() -> None:
    env = RubikEnv(RubikCubeTokenizer(), shuffles=0)
    target = env.reset().copy()
    scrambled = env.next_state(target.copy(), 0)
    assert not env.is_solved(scrambled)
    generator = Mock(generator_k_list=[1], mode="bank")
    generator.get_subgoals.return_value = [(target, {})]
    policy = Mock()
    policy.get_action.return_value = torch.tensor([0.0, 1.0] + [0.0] * 10)
    value = Mock()
    value.get_value.return_value = 0.5
    solver = Solver(
        max_nodes=5,
        planner_class=partial(AdasubsPlanner, generators_k_list=[1]),
        subgoal_generator=generator,
        validator=BasicValidator(env, policy, budget_for_achieving_subgoal=1),
        value_function=value,
    )

    result = solver.solve(scrambled)

    assert result.solution.solved
    assert result.solution.action_path == [1]
    subgoals = result.solution.subgoal_path
    assert subgoals is not None and np.array_equal(subgoals[-1], target)
    assert env.is_solved(env.get_state())
    assert result.search_info.backtracking_ratio == 0.0
