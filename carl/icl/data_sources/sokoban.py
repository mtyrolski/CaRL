"""Generate Sokoban boards with witnesses that replay in the environment."""
from collections import deque
from typing import Final

import numpy as np

from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.icl.boards import tiles, validate_subgoal
from carl.icl.types import Board, Coordinate, JSONObject

DIRECTIONS: Final[tuple[Coordinate, ...]] = ((-1, 0), (1, 0), (0, -1), (0, 1))
INVERSE: Final = (1, 0, 3, 2)


def environment(state: Board) -> SokobanEnv:
    """Construct the simulator at the board's actual size and box count."""
    if state.shape[0] != state.shape[1]:
        raise ValueError("This experiment expects square boards")
    return SokobanEnv(
        SokobanTokenizer(size_of_board=(state.shape[0], state.shape[1])),
        num_boxes=int(np.isin(tiles(state), [3, 4]).sum()),
    )


def verify_solution(state: Board, actions: list[int]) -> list[Board]:
    """Replay a complete witness, rejecting no-ops and post-terminal actions."""
    env = environment(state)
    states = [state]
    validate_subgoal(state, state)
    if int(np.isin(tiles(state), [2, 3, 6]).sum()) != int(np.isin(tiles(state), [3, 4]).sum()):
        raise ValueError("Goal and box counts must match")
    for action in actions:
        if action not in range(4) or env.is_solved(states[-1]):
            raise ValueError("Invalid action or witness continues past terminal")
        successor = np.asarray(env.next_state(states[-1], action))
        validate_subgoal(state, successor)
        if np.array_equal(successor, states[-1]):
            raise ValueError("Witness contains a no-op")
        states.append(successor)
    if not env.is_solved(states[-1]):
        raise ValueError("Witness does not solve the board")
    return states


def _walks(
    player: tuple[int, int], floor: set[Coordinate], boxes: set[Coordinate]
) -> dict[Coordinate, list[int]]:
    paths: dict[Coordinate, list[int]] = {player: []}
    queue = deque([player])
    while queue:
        position = queue.popleft()
        for action, delta in enumerate(DIRECTIONS):
            neighbor = (position[0] + delta[0], position[1] + delta[1])
            if neighbor in floor and neighbor not in boxes and neighbor not in paths:
                paths[neighbor] = paths[position] + [action]
                queue.append(neighbor)
    return paths


def reverse_board(
    size: int, num_boxes: int, seed: int, pulls: int | None = None
) -> tuple[Board, list[int], JSONObject]:
    """Scramble a solved board with legal inverse pushes; retain a forward witness.

    A connected random-walk floor occupies 65% of the interior. This is a
    synthetic transfer distribution, not the original Sokoban benchmark.
    """
    if size < 6 or not 1 <= num_boxes < (size - 2) ** 2 // 3:
        raise ValueError("Board size or box count unsupported")
    rng = np.random.default_rng(seed)
    pulls = pulls or max(20, 4 * num_boxes)
    for retry in range(100):
        position = (size // 2, size // 2)
        floor = {position}
        while len(floor) < int(0.65 * (size - 2) ** 2):
            dr, dc = DIRECTIONS[int(rng.integers(4))]
            position = (
                max(1, min(size - 2, position[0] + dr)),
                max(1, min(size - 2, position[1] + dc)),
            )
            floor.add(position)
        ordered = sorted(floor)
        indices = rng.choice(len(ordered), num_boxes + 1, replace=False)
        goals = {ordered[int(i)] for i in indices[:-1]}
        boxes = set(goals)
        player = ordered[int(indices[-1])]
        reverse_actions: list[int] = []
        completed = 0
        for _ in range(pulls):
            paths = _walks(player, floor, boxes)
            choices = []
            for box in sorted(boxes):
                for action, (dr, dc) in enumerate(DIRECTIONS):
                    stand, destination = (box[0] + dr, box[1] + dc), (
                        box[0] + 2 * dr,
                        box[1] + 2 * dc,
                    )
                    if stand in paths and destination in floor and destination not in boxes:
                        choices.append((box, stand, destination, action))
            if not choices:
                break
            box, stand, destination, action = choices[int(rng.integers(len(choices)))]
            reverse_actions.extend(paths[stand])
            reverse_actions.append(action)
            boxes.remove(box)
            boxes.add(stand)
            player = destination
            completed += 1
        if completed < pulls or len(boxes - goals) < max(1, num_boxes // 2):
            continue
        board = np.zeros((size, size), dtype=int)
        for cell in floor:
            board[cell] = 1
        for cell in goals:
            board[cell] = 2
        for cell in boxes:
            board[cell] = 3 if cell in goals else 4
        board[player] = 6 if player in goals else 5
        state = np.eye(7, dtype=np.float32)[board]
        env, current = environment(state), state
        witness = []
        for action in (INVERSE[a] for a in reversed(reverse_actions)):
            current = env.next_state(current, action)
            witness.append(action)
            if env.is_solved(current):
                break
        verify_solution(state, witness)
        return (
            state,
            witness,
            {
                "seed": seed,
                "retries": retry,
                "reverse_pulls": completed,
                "floor_cells": len(floor),
                "witness_length": len(witness),
                "boxes_off_goal": len(boxes - goals),
            },
        )
    raise ValueError(f"Could not generate a verified board for seed {seed}")
