"""Bounded, offline checks of explicit subgoals against the exact simulator."""
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from carl.icl.types import Board, BoardEnvironment

ReachabilityStatus = Literal["reachable", "unreachable_within_cap", "node_limit"]


@dataclass(frozen=True)
class ReachabilityAudit:
    """Distinguish a witnessed path from exhaustive and interrupted checks."""

    status: ReachabilityStatus
    actions: tuple[int, ...] | None
    expanded_states: int
    simulator_probes: int


def audit_subgoal(
    env: BoardEnvironment,
    initial: Board,
    target: Board,
    *,
    cap: int,
    max_states: int = 10000,
) -> ReachabilityAudit:
    """Search up to cap actions without expanding states after task completion.

    The simulator probes belong to this diagnostic, not to solver accounting.
    This is a bounded reachability check, not proof of general solvability.
    """
    if cap < 0 or max_states < 1:
        raise ValueError("cap must be nonnegative and max_states must be positive")
    queue: deque[tuple[Board, tuple[int, ...]]] = deque([(initial.copy(), ())])
    visited = {initial.argmax(-1).tobytes()}
    expanded = probes = 0
    while queue:
        state, path = queue.popleft()
        if np.array_equal(state, target):
            return ReachabilityAudit("reachable", path, expanded, probes)
        if len(path) == cap or env.is_solved(state):
            continue
        expanded += 1
        for action in range(4):
            successor = np.asarray(env.next_state(state, action))
            probes += 1
            if np.array_equal(successor, target):
                return ReachabilityAudit("reachable", (*path, action), expanded, probes)
            key = successor.argmax(-1).tobytes()
            if key in visited:
                continue
            if len(visited) >= max_states:
                return ReachabilityAudit("node_limit", None, expanded, probes)
            visited.add(key)
            queue.append((successor, (*path, action)))
    return ReachabilityAudit("unreachable_within_cap", None, expanded, probes)
