"""Recording and replay through AdaSubS with native environments."""

from functools import partial
from unittest.mock import Mock

import numpy as np
import torch

from carl.environment.env import GameEnv
from carl.environment.gym_rubik.rubik_env import RubikEnv
from carl.environment.gym_rubik.tokenizer import RubikCubeTokenizer
from carl.environment.n_puzzle.env import NPuzzleEnv
from carl.environment.n_puzzle.tokenizer import NPuzzleTokenizer
from carl.inference_components.validator import BasicValidator
from carl.planners.adasubs import AdasubsPlanner
from carl.solver.subgoal_search import Solver


def scripted_solver(env: GameEnv, proposals: list[np.ndarray], action_scores: list[float], k: int) -> Solver:
    generator = Mock(generator_k_list=[k], mode="bank")
    generator.get_subgoals.side_effect = [[(state, {})] for state in proposals]
    policy = Mock()
    policy.get_action.return_value = torch.tensor(action_scores)
    value = Mock()
    value.get_value.return_value = 0.5
    return Solver(
        max_nodes=5,
        planner_class=partial(AdasubsPlanner, generators_k_list=[k]),
        subgoal_generator=generator,
        validator=BasicValidator(env, policy, budget_for_achieving_subgoal=k),
        value_function=value,
    )


def test_adasubs_keeps_intermediate_proposal_reached_after_requeue() -> None:
    env = NPuzzleEnv(NPuzzleTokenizer(size_of_board=(3, 3)))
    initial = np.array([1, 2, 3, 4, 5, 6, 0, 7, 8])
    middle = env.next_state(initial.copy(), 1)
    goal = env.next_state(middle.copy(), 1)
    solver = scripted_solver(env, [middle, goal], [0.0, 1.0, 0.0, 0.0], k=2)

    result = solver.solve(initial)

    assert result.solution.solved and result.solution.action_path == [1, 1]
    assert [event["reached"] for event in result.search_info.proposal_events] == [True, True]
    assert result.search_info.realized_segment_lengths == [1, 1]
    assert result.search_info.detour_ratio == 0.5


def test_rubik_diagnostic_replay_preserves_terminal_subgoal() -> None:
    env = RubikEnv(RubikCubeTokenizer(), shuffles=0)
    target = env.reset().copy()
    scrambled = env.next_state(target.copy(), 0)
    assert not env.is_solved(scrambled)
    solver = scripted_solver(env, [target], [0.0, 1.0] + [0.0] * 10, k=1)

    result = solver.solve(scrambled)

    assert result.solution.solved
    assert result.solution.action_path == [1]
    subgoals = result.solution.subgoal_path
    assert subgoals is not None and np.array_equal(subgoals[-1], target)
    assert env.is_solved(env.get_state())
    assert result.search_info.backtracking_ratio == 0.0
