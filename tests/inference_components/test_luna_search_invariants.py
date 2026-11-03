"""Critical scheduler and action-budget invariants for the actual ICL solver."""
from types import SimpleNamespace
from typing import cast

import numpy as np

from carl.icl.config import ExperimentConfig
from carl.icl.search import Frontier, SearchPath, search
from carl.icl.search_tree import SearchTreeRecorder
from carl.icl.types import Board, BoardEnvironment, SearchComponents


def state(index: int) -> Board:
    """Small categorical state, sufficient for the search's exact-identity contract."""
    return np.eye(7, dtype=np.float32)[index].reshape(1, 1, 7)


def test_frontier_orders_scale_before_value_and_discards_reopened_stale_entries() -> None:
    components = cast(SearchComponents, SimpleNamespace(value=lambda s: 0.8 if int(s.argmax(-1)[0, 0]) == 2 else 0.2))
    frontier = Frontier([0.9, 0.5, 0.0], components)
    low, high = SearchPath(state(1), [0, 0], []), SearchPath(state(2), [0], [])
    assert not frontier.push(low) and not frontier.push(high)
    short = SearchPath(state(1), [1], [])
    assert frontier.push(short)
    results = []
    while (entry := frontier.pop()) is not None:
        results.append((-entry.negative_scale, -entry.negative_value, len(entry.path.actions)))
    assert results == [(0.9, 0.8, 1), (0.9, 0.2, 1), (0.5, 0.8, 1),
                       (0.5, 0.2, 1), (0.0, 0.8, 1), (0.0, 0.2, 1)]


def test_zero_scale_expands_real_actions_and_counts_blocked_moves_without_generator() -> None:
    original = state(1)
    actions: list[int] = []
    requested_scales: list[float] = []

    def transition(current: Board, action: int) -> Board:
        actions.append(action)
        return current.copy()

    def propose(_current: Board, scale: float) -> list[Board]:
        requested_scales.append(scale)
        assert scale > 0
        return []

    env = cast(BoardEnvironment, SimpleNamespace(next_state=transition, is_solved=lambda _state: False))
    components = cast(SearchComponents, SimpleNamespace(value=lambda _state: 0.5, propose=propose))
    observer = SearchTreeRecorder()
    result = search(env, components, original, ExperimentConfig(action_budget=3), 11, observer=observer)
    assert requested_scales == [0.9, 0.5]
    assert actions == [0, 1, 2]
    assert result.attempted_actions == 3 and result.finish_reason == 'action_budget'
    assert [node.status for node in observer.nodes[1:]] == ['no_op'] * 3
    assert sum(len(node.actions) for node in observer.nodes) == result.attempted_actions


def test_multiple_subgoals_share_remaining_action_budget() -> None:
    source, first, second = state(0), state(1), state(6)
    actions: list[int] = []

    def transition(_current: Board, action: int) -> Board:
        actions.append(action)
        return first.copy()

    env = cast(BoardEnvironment, SimpleNamespace(next_state=transition, is_solved=lambda _state: False))
    components = cast(SearchComponents, SimpleNamespace(
        value=lambda _state: 0.5,
        propose=lambda _state, _scale: [first, second],
        probabilities=lambda _state, _target: np.array([1.0, 0.0, 0.0, 0.0]),
    ))
    observer = SearchTreeRecorder()
    result = search(env, components, source, ExperimentConfig(action_budget=2), 11, observer=observer)
    assert actions == [0, 0]
    assert result.attempted_actions == 2 and result.finish_reason == 'action_budget'
    assert result.accepted == 1 and result.rejected == 1
    assert [event['attempted_actions'] for event in result.events] == [1, 1]
    assert [event['reached'] for event in result.events] == [True, False]
    assert sum(len(node.actions) for node in observer.nodes) == result.attempted_actions


def test_one_low_level_budget_does_not_charge_high_level_expansions() -> None:
    original = state(1)
    visited_actions: list[int] = []

    def transition(current: Board, action: int) -> Board:
        visited_actions.append(action)
        return current.copy()

    env = cast(BoardEnvironment, SimpleNamespace(next_state=transition, is_solved=lambda _state: False))
    components = cast(SearchComponents, SimpleNamespace(value=lambda _state: 0.5, propose=lambda _state, _scale: []))
    result = search(env, components, original, ExperimentConfig(action_budget=1, max_expansions=None), 11)
    assert result.expansions == 3  # Two scales and primitive fallback; none consumes a low-level node.
    assert visited_actions == [0] and result.attempted_actions == 1
    assert result.finish_reason == 'action_budget'


def test_reached_high_level_state_is_not_charged_twice() -> None:
    source, target = state(0), state(1)
    env = cast(BoardEnvironment, SimpleNamespace(next_state=lambda _state, _action: target.copy(),
                                               is_solved=lambda _state: False))
    components = cast(SearchComponents, SimpleNamespace(
        value=lambda _state: 0.5, propose=lambda _state, _scale: [target],
        probabilities=lambda _state, _target: np.array([1.0, 0.0, 0.0, 0.0]),
    ))
    result = search(env, components, source, ExperimentConfig(action_budget=1, max_expansions=None), 11)
    assert result.accepted == 1 and result.expansions == 1
    assert result.attempted_actions == 1 and result.finish_reason == 'action_budget'


def test_task_solution_before_another_subgoal_stops_immediately() -> None:
    source, solved, requested = state(0), state(1), state(2)
    actions: list[int] = []

    def transition(_current: Board, action: int) -> Board:
        actions.append(action)
        return solved.copy()

    env = cast(BoardEnvironment, SimpleNamespace(
        next_state=transition, is_solved=lambda board: np.array_equal(board, solved),
    ))
    components = cast(SearchComponents, SimpleNamespace(
        value=lambda _state: 0.5, propose=lambda _state, _scale: [requested],
        probabilities=lambda _state, _target: np.array([1.0, 0.0, 0.0, 0.0]),
    ))
    observer = SearchTreeRecorder()
    result = search(env, components, source, ExperimentConfig(action_budget=8), 11, observer=observer)
    assert actions == [0] and result.action_path == [0]
    assert result.solved and result.finish_reason == 'solved' and result.attempted_actions == 1
    assert result.events[0]['solved'] and not result.events[0]['reached']
    assert observer.nodes[-1].solved and not observer.nodes[-1].reached
