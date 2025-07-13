"""
Utilities for verifying Conditional Low-Level Policy (CLLP) reachability.
Provides state trajectory replay and batch validation functions.
"""

from typing import Callable, List, Union
from dataclasses import dataclass

import numpy as np

from carl.environment.env import GameEnv
from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy
from carl.inference_components.validator import BasicValidator
from carl.solver.nodes import ValidationResult
from carl.utils.aliases import State

@dataclass
class CLLPVerificationResult:
    """Result of CLLP reachability check."""
    success_rate: float  # fraction of goals reached
    total_goals: int
    reached: np.ndarray
    calls: int
    cllp_samples_in_calls: int
    node_computations: int
    paths: List[List[int]]


def trajectory_from_actions(
    env: GameEnv,
    state: State,
    actions: List[int]
) -> List[State]:
    """Replay a sequence of actions from the initial state and return all visited states."""
    # reset to given state
    env.set_state(state)
    states = [state]
    for act in actions:
        state, _, _, _ = env.step(act)
        states.append(state)
    return states


def verify_cllp_reaches_subgoals_from_initial_state(
    cllp: ConditionalLowLevelPolicy,
    goals: Union[State, List[State]],
    initial_state: State,
    env_creation_fn: Callable[[], GameEnv],
    max_radius: int,
    add_first_batch_to_node_computations: bool = False,
) -> CLLPVerificationResult:
    """
    Batch-verify that a conditional low-level policy (CLLP) can reach specified subgoals.

    Args:
        cllp: Initialized CLLP instance.
        goals: Single or list of target states to reach.
        initial_state: Starting state for each reachability check.
        env_creation_fn: Factory to create fresh GameEnv instances.
        max_radius: Maximum steps allowed for each subgoal.
        add_first_batch_to_node_computations: (unused) placeholder for future extensions.

    Returns:
        CLLPVerificationResult containing success metrics and paths.
    """

    # initialize validator with budget limit
    validator = BasicValidator(env_creation_fn(), cllp, budget_for_achieving_subgoal=max_radius)

    # normalize single goal to a list
    if not isinstance(goals, list):
        goals = [goals]

    # validate reachability for each goal
    reach_results: List[ValidationResult] = [validator.is_valid(initial_state, g) for g in goals]

    # extract outcomes and paths
    reached_states = np.array([res.is_valid for res in reach_results])
    paths = [res.path for res in reach_results]

    return CLLPVerificationResult(
        success_rate=float(np.mean(reached_states)),
        total_goals=len(reached_states),
        reached=reached_states,
        calls=len(reach_results),
        cllp_samples_in_calls=sum(res.low_level_nodes_visited for res in reach_results),
        node_computations=sum(res.low_level_nodes_visited for res in reach_results),
        paths=paths,
    )
