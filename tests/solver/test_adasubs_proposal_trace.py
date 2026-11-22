"""Regressions for stable and complete winning proposal traces."""

from functools import partial
from unittest.mock import Mock

import numpy as np
import torch

from carl.environment.n_puzzle.env import NPuzzleEnv
from carl.environment.n_puzzle.tokenizer import NPuzzleTokenizer
from carl.inference_components.validator import BasicValidator
from carl.planners.adasubs import AdasubsPlanner
from carl.solver.subgoal_search import Solver


def test_saved_state_identifier_is_stable_and_includes_array_shape() -> None:
    identify = Solver._hash_state  # pylint: disable=protected-access
    assert identify("fixture") == "-4431358843294443104"
    assert identify(1) != identify("1")
    state = np.array([0, 1, 2, 3], dtype=np.int8)
    assert identify(state) != identify(state.reshape(2, 2))


def test_adasubs_keeps_intermediate_proposal_reached_after_requeue() -> None:
    env = NPuzzleEnv(NPuzzleTokenizer(size_of_board=(3, 3)))
    initial = np.array([1, 2, 3, 4, 5, 6, 0, 7, 8])
    middle = env.next_state(initial.copy(), 1)
    goal = env.next_state(middle.copy(), 1)
    generator = Mock(generator_k_list=[2], mode="bank")
    generator.get_subgoals.side_effect = [[(middle, {})], [(goal, {})]]
    policy = Mock()
    policy.get_action.return_value = torch.tensor([0.0, 1.0, 0.0, 0.0])
    value = Mock()
    value.get_value.return_value = 0.5
    solver = Solver(
        max_nodes=5,
        planner_class=partial(AdasubsPlanner, generators_k_list=[2]),
        subgoal_generator=generator,
        validator=BasicValidator(env, policy, budget_for_achieving_subgoal=2),
        value_function=value,
    )

    result = solver.solve(initial)

    assert result.solution.solved and result.solution.action_path == [1, 1]
    assert [event["reached"] for event in result.search_info.proposal_events] == [True, True]
    assert result.search_info.realized_segment_lengths == [1, 1]
    assert result.search_info.detour_ratio == 0.5
