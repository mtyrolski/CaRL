from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from carl.inference_components.validator import BasicValidator
from carl.environment.env import GameEnv
from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy
from carl.inference_components.subgoal_generator import SubgoalGenerator
from carl.solver.nodes import ValidationResult


@dataclass
class CLLPVerificationResult:
    success_rate: float
    total_goals: int
    reached: np.ndarray
    calls: int
    cllp_samples_in_calls: int
    node_computations: int
    paths: list[list[int]]


def trajectory_from_actions(env, state: np.ndarray, actions: list[int]) -> list[np.ndarray]:
    env.set_state(state)
    states = [state]
    for action in actions:
        state, _, _, _ = env.step(action)
        states.append(state)
    return states


def verify_cllp_reaches_subgoals_from_initial_state(
    cllp: ConditionalLowLevelPolicy,
    goals: np.ndarray | list[np.ndarray],
    initial_state: np.ndarray,
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

    # if isinstance(goals, list):
    #     goals = np.stack(goals, axis=0)

    # bs = goals.shape[0]
    # node_computations = 0

    # current_states = np.stack([initial_state] * bs, axis=0)
    # assert current_states.shape[0] == goals.shape[0]

    # reached_states = np.all(np.equal(current_states, goals), axis=tuple(i for i in range(1, current_states.ndim)))

    # # For each unreached state, we have to initialize it
    # paths = [[] for _ in range(bs)]

    # envs = [env_creation_fn() for _ in range(bs)]
    # for env in envs:
    #     env.set_state(initial_state.copy())

    # calls = 0
    # cllp_samples_in_calls = 0
    # for i in range(max_radius):

    #     active_idxs = np.where(np.equal(reached_states, False))[0]

    #     if len(active_idxs) == 0:
    #         assert len(np.where(np.equal(reached_states, True))[0]) == bs
    #         break

    #     active_states = np.stack([current_states[i, :] for i in active_idxs], axis=0)
    #     active_goals = np.stack([goals[i, :] for i in active_idxs], axis=0)

    #     # TODO: make it work with batched inputs
    #     actions = np.stack(
    #         [
    #             torch.argmax(
    #                 cllp.get_action(state=active_states[k, :], state_after_k=active_goals[k, :]),
    #                 dim=-1,
    #             ).cpu().numpy() for k in range(active_states.shape[0])
    #         ],
    #         axis=0,
    #     )

    #     cllp_samples_in_calls += len(actions)

    #     if i != 0 or not add_first_batch_to_node_computations:
    #         # don't add the first batch to node computations if verifier was used
    #         node_computations += len(actions)

    #     for j, action in zip(active_idxs, actions, strict=True):
    #         paths[j].append(action)    # Update of path
    #         # if isinstance(envs[j], gymnasium.Env):
    #         current_states[j], _, _, _ = envs[j].step(action)    # Update of current state
    #         if np.array_equal(current_states[j], goals[j]):
    #             reached_states[j] = True

    # return CLLPVerificationResult(
    #     success_rate=np.mean(reached_states).item(),
    #     total_goals=reached_states.shape[0],
    #     reached=reached_states,
    #     calls=calls,
    #     cllp_samples_in_calls=cllp_samples_in_calls,
    #     node_computations=node_computations,
    #     paths=paths,
    # )


def verify_cllp_reaches_subgoals_from_generator(
    cllp: ConditionalLowLevelPolicy,
    generator: SubgoalGenerator,
    goals_to_generate: int,
    initial_state: np.ndarray,
    env_creation_fn: Callable[[], GameEnv],
    max_radius: int,
    add_first_batch_to_node_computations: bool = False,
) -> CLLPVerificationResult:

    goals = generator.get_subgoals(
        state=initial_state,
        num_return_sequences=goals_to_generate,
        num_beams=max(generator.subgoal_generation_kwargs['num_beams'], goals_to_generate),
    )
    return verify_cllp_reaches_subgoals_from_initial_state(
        cllp=cllp,
        goals=goals,
        initial_state=initial_state,
        env_creation_fn=env_creation_fn,
        max_radius=max_radius,
        add_first_batch_to_node_computations=add_first_batch_to_node_computations,
    )
