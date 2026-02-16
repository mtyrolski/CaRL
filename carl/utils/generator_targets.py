
from collections.abc import Iterable, Iterator, Sequence
from enum import StrEnum
from typing import Any, cast

from carl.environment.env import GameEnv
from carl.environment.training_goal import TrainingGoal
from carl.planners.base import Experience
from carl.solver.nodes import EnvWithRestore
from carl.solver.nodes import SearchTreeNode
from carl.solver.nodes import get_solving_path_data
from carl.utils.aliases import State

DEFAULT_K_OFFSETS: tuple[int, ...] = (-1, 1, -2, 2, -3, 3)


class SubgoalTargetMode(StrEnum):
    SLIDING_WINDOW = "sliding_window"
    K_OFFSETS = "k_offsets"


def _iter_experiences(
    experiences: Iterable[Experience | Iterable[Experience] | None],
) -> Iterator[Experience]:
    for item in experiences:
        if item is None:
            continue
        if isinstance(item, Experience):
            yield item
            continue
        if isinstance(item, (list, tuple)):
            for exp in item:
                if exp is None:
                    continue
                if not isinstance(exp, Experience):
                    raise TypeError(f"Expected Experience, got {type(exp).__name__}")
                yield exp
            continue
        raise TypeError(f"Expected Experience or sequence, got {type(item).__name__}")


def _state_path_from_experience(exp: Experience, env: EnvWithRestore) -> list[State] | None:
    if not exp.solution.solved:
        return None
    solving_node = exp.search_info.solving_node
    if solving_node is None:
        return None
    _, _, _, _, state_path = get_solving_path_data(
        solving_node,
        include_state_path=True,
        env=env,
    )
    return state_path


def _solving_path_nodes(solving_node: SearchTreeNode) -> list[SearchTreeNode]:
    nodes: list[SearchTreeNode] = []
    current: SearchTreeNode | None = solving_node
    while current is not None:
        nodes.append(current)
        current = current.parent_node
    nodes.reverse()
    return nodes


def _subgoal_positions_with_k(
    solving_node: SearchTreeNode,
    fallback_k_used: Sequence[int] | None,
) -> list[tuple[int, int | None]]:
    nodes = _solving_path_nodes(solving_node)
    if len(nodes) <= 1:
        return []

    fallback: list[int] | None = None
    if fallback_k_used is not None and len(fallback_k_used) == len(nodes) - 1:
        fallback = list(fallback_k_used)

    positions_with_k: list[tuple[int, int | None]] = []
    index = 0
    for idx, node in enumerate(nodes[1:]):
        low_level_path = node.low_level_path or []
        index += len(low_level_path)
        k_used = node.next_expand_with_k_generator
        if k_used is None and fallback is not None:
            k_used = fallback[idx]
        positions_with_k.append((index, k_used))
    return positions_with_k


def iter_state_pairs_sliding_window(
    state_path: Sequence[State],
    distance_range: Sequence[int] | None = None,
) -> Iterator[tuple[State, State]]:
    if distance_range is not None and any(dist <= 0 for dist in distance_range):
        raise ValueError("distance_range must contain positive integers")

    trajectory_length = len(state_path)
    for position in range(trajectory_length - 1):
        if distance_range is None:
            distances: Iterable[int] = range(1, trajectory_length - position)
        else:
            distances = distance_range

        for dist in distances:
            inner_dist = min(dist, trajectory_length - 1 - position)
            yield state_path[position], state_path[position + inner_dist]
            if position + inner_dist >= trajectory_length - 1:
                break


def iter_state_pairs_k_offsets(
    state_path: Sequence[State],
    subgoal_positions: Sequence[tuple[int, int | None]],
    k: int,
    offsets: Sequence[int] = DEFAULT_K_OFFSETS,
) -> Iterator[tuple[State, State]]:
    if k <= 0:
        raise ValueError("k must be positive")

    trajectory_length = len(state_path)
    for position, k_used in subgoal_positions:
        if k_used != k:
            continue
        if position < 0 or position >= trajectory_length:
            continue
        for offset in offsets:
            start_index = position - k + offset
            if 0 <= start_index < position:
                yield state_path[start_index], state_path[position]


def generate_sliding_window_generator_targets(
    experiences: Iterable[Experience | Iterable[Experience] | None],
    env: EnvWithRestore,
    distance_range: Sequence[int] | None = None,
    training_goal: TrainingGoal = TrainingGoal.GENERATOR,
) -> list[tuple[Any, Any]]:
    targets: list[tuple[Any, Any]] = []
    tokenizer = cast(GameEnv, env).tokenizer
    for exp in _iter_experiences(experiences):
        state_path = _state_path_from_experience(exp, env)
        if not state_path:
            continue
        for state, subgoal in iter_state_pairs_sliding_window(state_path, distance_range):
            x, y = tokenizer.x_y_tokenizer(state, subgoal, training_goal)
            targets.append((x, y))
    return targets


def generate_k_offset_generator_targets(
    experiences: Iterable[Experience | Iterable[Experience] | None],
    env: EnvWithRestore,
    k: int,
    offsets: Sequence[int] = DEFAULT_K_OFFSETS,
    training_goal: TrainingGoal = TrainingGoal.GENERATOR,
) -> list[tuple[Any, Any]]:
    if k <= 0:
        raise ValueError("k must be positive")

    targets: list[tuple[Any, Any]] = []
    tokenizer = cast(GameEnv, env).tokenizer
    for exp in _iter_experiences(experiences):
        state_path = _state_path_from_experience(exp, env)
        if not state_path:
            continue
        solving_node = exp.search_info.solving_node
        if solving_node is None:
            continue
        subgoal_positions = _subgoal_positions_with_k(
            solving_node,
            exp.solution.subgoal_distance_path,
        )
        for state, subgoal in iter_state_pairs_k_offsets(
            state_path,
            subgoal_positions,
            k,
            offsets,
        ):
            x, y = tokenizer.x_y_tokenizer(state, subgoal, training_goal)
            targets.append((x, y))
    return targets


def generate_subgoal_generator_targets(
    experiences: Iterable[Experience | Iterable[Experience] | None],
    env: EnvWithRestore,
    mode: SubgoalTargetMode,
    distance_range: Sequence[int] | None = None,
    k: int | None = None,
    offsets: Sequence[int] = DEFAULT_K_OFFSETS,
    training_goal: TrainingGoal = TrainingGoal.GENERATOR,
) -> list[tuple[Any, Any]]:
    if mode == SubgoalTargetMode.SLIDING_WINDOW:
        return generate_sliding_window_generator_targets(
            experiences,
            env,
            distance_range=distance_range,
            training_goal=training_goal,
        )
    if mode == SubgoalTargetMode.K_OFFSETS:
        if k is None:
            raise ValueError("k is required for K_OFFSETS mode")
        return generate_k_offset_generator_targets(
            experiences,
            env,
            k=k,
            offsets=offsets,
            training_goal=training_goal,
        )
    raise ValueError(f"Unsupported mode: {mode}")
