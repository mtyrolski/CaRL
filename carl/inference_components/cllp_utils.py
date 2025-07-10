from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from carl.environment.env import GameEnv
from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy
from carl.inference_components.subgoal_generator import SubgoalGenerator
from carl.inference_components.validator import BasicValidator
from carl.solver.nodes import GeneratedSubgoal, ValidationResult
from carl.utils.aliases import State

@dataclass
class CLLPVerificationResult:
    success_rate: float
    total_goals: int
    reached: np.ndarray
    calls: int
    cllp_samples_in_calls: int
    node_computations: int
    paths: list[list[int]]


def trajectory_from_actions(env, state: State, actions: list[int]) -> list[State]:
    env.set_state(state)
    states = [state]
    for action in actions:
        state, _, _, _ = env.step(action)
        states.append(state)
    return states


def verify_cllp_reaches_subgoals_from_initial_state(
    cllp: ConditionalLowLevelPolicy,
    goals: State | list[State],
    initial_state: State,
    env_creation_fn: Callable[[], GameEnv],
    max_radius: int,
    add_first_batch_to_node_computations: bool = False,
) -> CLLPVerificationResult:

    validator = BasicValidator(env_creation_fn(), cllp, budget_for_achieving_subgoal=max_radius)

    if not isinstance(goals, list):
        goals = [goals]

    reachability: list[ValidationResult] = [validator.is_valid(initial_state, goal) for goal in goals]

    reached_states = np.array([result.is_valid for result in reachability])
    paths = [result.path for result in reachability]

    return CLLPVerificationResult(
        success_rate=np.mean(reached_states).item(),
        total_goals=len(reached_states),
        reached=reached_states,
        calls=len(reachability),
        cllp_samples_in_calls=sum(result.low_level_nodes_visited for result in reachability),
        node_computations=sum(result.low_level_nodes_visited for result in reachability),
        paths=paths,
    )
