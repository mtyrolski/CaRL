"""Bounded Sokoban intent search and a deterministic local baseline."""

import time
from collections import deque
from enum import StrEnum
from typing import Any, cast

from carl.icl.intent_data import SokobanPredicate
from carl.icl.intent_data import predicate_value
from carl.icl.intent_data import validate_prediction

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class SearchStatus(StrEnum):
    WITNESS_FOUND = "witness_found"
    DEPTH_LIMIT = "depth_limit"
    NODE_LIMIT = "node_limit"
    TIME_LIMIT = "time_limit"
    EXHAUSTED = "exhausted"


Coordinate = tuple[int, int]
Positions = frozenset[Coordinate]
State = tuple[Coordinate, Positions]
Record = dict[str, Any]


def board_parts(
    rows: list[str],
) -> tuple[int, int, set[Coordinate], set[Coordinate], Positions, Coordinate]:
    if not rows or not rows[0]:
        raise ValueError("Invalid Sokoban board")
    height, width = len(rows), len(rows[0])
    walls = {
        (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "#"
    }
    goals = {
        (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in ".*+"
    }
    boxes = frozenset(
        (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in "$*"
    )
    players = [
        (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in "@+"
    ]
    if len(players) != 1 or any(len(row) != width for row in rows):
        raise ValueError("Invalid Sokoban board")
    return height, width, walls, goals, boxes, players[0]


# Keep the existing positional transition interface.
# pylint: disable-next=too-many-positional-arguments
def step(
    player: Coordinate,
    boxes: Positions,
    action: int,
    height: int,
    width: int,
    walls: set[Coordinate],
) -> State | None:
    dr, dc = DIRECTIONS[action]
    dest = player[0] + dr, player[1] + dc
    if not (0 <= dest[0] < height and 0 <= dest[1] < width) or dest in walls:
        return None
    if dest in boxes:
        beyond = dest[0] + dr, dest[1] + dc
        if (
            not (0 <= beyond[0] < height and 0 <= beyond[1] < width)
            or beyond in walls
            or beyond in boxes
        ):
            return None
        boxes = frozenset((boxes - {dest}) | {beyond})
    return dest, boxes


def goal_satisfied(
    effects: list[Record], player: Coordinate, boxes: Positions, goals: set[Coordinate]
) -> bool:
    for predicate in effects:
        if predicate["predicate"] == SokobanPredicate.BOX_AT:
            value = (predicate["row"], predicate["col"]) in boxes
        elif predicate["predicate"] == SokobanPredicate.PLAYER_AT:
            value = (predicate["row"], predicate["col"]) == player
        else:
            value = len(boxes & goals) >= predicate["count"]
        if not value:
            return False
    return True


def real_replay(rows: list[str], intent: Record, actions: list[int]) -> list[str]:
    """Verify candidate witness with the same real environment as the data."""
    import numpy as np

    from carl.icl.boards import from_rows
    from carl.icl.boards import to_rows
    from carl.icl.data_sources.sokoban import environment

    state = from_rows(rows)
    env = environment(state)
    if not all(predicate_value(p, rows) for p in intent.get("preserve", [])):
        raise AssertionError("Preservation false initially")
    for action in actions:
        if env.is_solved(state):
            raise AssertionError("Witness continues after task termination")
        new = np.asarray(env.next_state(state, action))
        if np.array_equal(new, state):
            raise AssertionError("Witness contains a failed or no-op action")
        state = new
        current = to_rows(state)
        if not all(predicate_value(p, current) for p in intent.get("preserve", [])):
            raise AssertionError("Witness violates preservation")
    reached = to_rows(state)
    if not all(predicate_value(p, reached) for p in intent["effects"]):
        raise AssertionError("Witness failed real environment check")
    return cast(list[str], reached)


def bounded_witness(
    rows: list[str],
    intent: Record,
    max_depth: int = 24,
    max_nodes: int = 10000,
    seconds: float = 1.0,
) -> Record:
    errors = validate_prediction(intent, rows)
    if errors:
        raise ValueError("; ".join(errors))
    height, width, walls, goals, boxes, player = board_parts(rows)
    initial = player, boxes
    frontier = deque([(initial, 0)])
    parents: dict[State, tuple[State, int] | None] = {initial: None}
    start = time.monotonic()
    expanded = 0
    primitive_edges_examined = 0
    depth_limited = False
    reason = SearchStatus.EXHAUSTED
    end_state: State | None = None
    while frontier:
        if time.monotonic() - start >= seconds:
            reason = SearchStatus.TIME_LIMIT
            break
        if expanded >= max_nodes:
            reason = SearchStatus.NODE_LIMIT
            break
        (player, boxes), depth = frontier.popleft()
        if depth >= max_depth:
            depth_limited = True
            continue
        # The actual environment ends the task when all boxes occupy goals.
        if boxes <= goals:
            continue
        expanded += 1
        for action in range(4):
            primitive_edges_examined += 1
            nxt = step(player, boxes, action, height, width, walls)
            if nxt is None or nxt in parents:
                continue
            next_player, next_boxes = nxt
            if not goal_satisfied(
                intent.get("preserve", []), next_player, next_boxes, goals
            ):
                continue
            parents[nxt] = ((player, boxes), action)
            if goal_satisfied(intent["effects"], next_player, next_boxes, goals):
                end_state = nxt
                reason = SearchStatus.WITNESS_FOUND
                break
            frontier.append((nxt, depth + 1))
        if end_state is not None:
            break
    if reason == SearchStatus.EXHAUSTED and depth_limited:
        reason = SearchStatus.DEPTH_LIMIT
    search_seconds = time.monotonic() - start
    actions = None
    reached = None
    verification_seconds = None
    if end_state is not None:
        actions = []
        node = end_state
        while (parent := parents[node]) is not None:
            node, action = parent
            actions.append(action)
        actions.reverse()
        verification_start = time.monotonic()
        reached = real_replay(rows, intent, actions)
        verification_seconds = time.monotonic() - verification_start
    return {
        "status": reason,
        "witness_found": end_state is not None,
        "depth_limit": max_depth,
        "node_limit": max_nodes,
        "time_limit_seconds": seconds,
        "expanded_nodes": expanded,
        "primitive_edges_examined": primitive_edges_examined,
        "elapsed_seconds": search_seconds,
        "verification_seconds": verification_seconds,
        "witness_depth": len(actions) if actions is not None else None,
        "witness_actions": actions,
        "reached_observation": reached,
    }


def local_rule_with_stats(rows: list[str]) -> tuple[Record | None, Record]:
    """Pick an accessible push and measure the rule's local graph work.

    The input is only the current board. Walking reachability is computed on
    the unchanged current arrangement; no demonstration state is consulted.
    Edges count every tested direction, including blocked and repeated ones.
    """
    started = time.perf_counter()
    height, width, walls, goals, boxes, player = board_parts(rows)
    paths = {player: 0}
    queue = deque([player])
    expanded = edges = push_checks = goal_distance_checks = 0
    while queue:
        here = queue.popleft()
        expanded += 1
        for dr, dc in DIRECTIONS:
            edges += 1
            nxt = here[0] + dr, here[1] + dc
            if (
                0 <= nxt[0] < height
                and 0 <= nxt[1] < width
                and nxt not in walls
                and nxt not in boxes
                and nxt not in paths
            ):
                paths[nxt] = paths[here] + 1
                queue.append(nxt)
    choices = []
    for r, c in sorted(boxes):
        for dr, dc in DIRECTIONS:
            push_checks += 1
            stand, dest = (r - dr, c - dc), (r + dr, c + dc)
            if (
                stand in paths
                and 0 <= dest[0] < height
                and 0 <= dest[1] < width
                and dest not in walls
                and dest not in boxes
            ):
                goal_distance_checks += len(goals)
                nearest = min(
                    (abs(dest[0] - gr) + abs(dest[1] - gc) for gr, gc in goals),
                    default=0,
                )
                choices.append((dest not in goals, nearest, paths[stand], dest))
    if choices:
        _, _, _, (r, c) = min(choices)
        effect = {"predicate": SokobanPredicate.BOX_AT, "row": r, "col": c}
    elif len(paths) > 1:
        r, c = min((p for p in paths if p != player), key=lambda p: (paths[p], p))
        effect = {"predicate": SokobanPredicate.PLAYER_AT, "row": r, "col": c}
    else:
        effect = None
    intent = (
        {"effects": [effect], "preserve": [], "instruction": None}
        if effect is not None
        else None
    )
    stats = {
        "walk_graphs": 1,
        "walk_nodes_expanded": expanded,
        "walk_edges_examined": edges,
        "reachable_tiles": len(paths),
        "push_checks": push_checks,
        "legal_pushes": len(choices),
        "goal_distance_checks": goal_distance_checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return intent, stats


def local_rule(rows: list[str]) -> Record | None:
    """Return the deterministic current-board proposal, preserving the original API."""
    return local_rule_with_stats(rows)[0]
