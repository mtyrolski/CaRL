"""Exact dihedral transformations for structured Sokoban intent training."""

from __future__ import annotations

import copy
from enum import StrEnum
from typing import Any

import numpy as np

from carl.icl.intent_data import SokobanPredicate
from carl.icl.intent_data import validate_prediction


class BoardSymmetry(StrEnum):
    """Reflect left/right first, then rotate anticlockwise by 0, 90, 180 or 270 degrees."""

    IDENTITY = "identity"
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    REFLECT = "reflect"
    REFLECT_ROTATE_90 = "reflect_rotate_90"
    REFLECT_ROTATE_180 = "reflect_rotate_180"
    REFLECT_ROTATE_270 = "reflect_rotate_270"


SYMMETRIES = tuple(BoardSymmetry)
TRANSFORM_INVARIANT_TASK = "Put every box on a goal."


def symmetry_parameters(symmetry: BoardSymmetry | str) -> tuple[bool, int]:
    """Return reflection and rotation parameters in their declared order."""
    index = SYMMETRIES.index(BoardSymmetry(symmetry))
    return index >= 4, index % 4


def transform_coordinate(row: int, col: int, size: int, symmetry: BoardSymmetry | str) -> tuple[int, int]:
    """Transform a zero-based (row, column) without clipping invalid inputs."""
    if (not isinstance(size, int) or isinstance(size, bool) or size < 1
            or not isinstance(row, int) or isinstance(row, bool)
            or not isinstance(col, int) or isinstance(col, bool)
            or not 0 <= row < size or not 0 <= col < size):
        raise ValueError("coordinate is outside the square board")
    reflect, rotations = symmetry_parameters(symmetry)
    if reflect:
        col = size - 1 - col
    for _ in range(rotations):
        row, col = size - 1 - col, row
    return row, col


def transform_rows(rows: list[str], symmetry: BoardSymmetry | str) -> list[str]:
    """Transform every cell, including walls, goals, player and boxes."""
    if not rows or not all(isinstance(row, str) and len(row) == len(rows) for row in rows):
        raise ValueError("a nonempty square character board is required")
    board = np.asarray([list(row) for row in rows])
    reflect, rotations = symmetry_parameters(symmetry)
    if reflect:
        board = np.fliplr(board)
    board = np.rot90(board, rotations)
    return ["".join(row) for row in board.tolist()]


def transform_actions(actions: list[int], symmetry: BoardSymmetry | str) -> list[int]:
    """Map the environment's primitive U/D/L/R actions under the same symmetry."""
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    origin = transform_coordinate(2, 2, 5, symmetry)
    mapping = []
    for dr, dc in directions:
        target = transform_coordinate(2 + dr, 2 + dc, 5, symmetry)
        mapping.append(directions.index((target[0] - origin[0], target[1] - origin[1])))
    if any(not isinstance(action, int) or isinstance(action, bool) or action not in range(4) for action in actions):
        raise ValueError("primitive actions must be integers in the U/D/L/R convention")
    return [mapping[action] for action in actions]


def transform_intent(intent: dict[str, Any], size: int, symmetry: BoardSymmetry | str) -> dict[str, Any]:
    """Transform coordinates in both conjunctions; arbitrary prose cannot be mapped safely."""
    if not isinstance(intent, dict) or intent.get("instruction") is not None:
        raise ValueError("augmentation requires structured intents with a null instruction")
    result = copy.deepcopy(intent)
    for name in ("effects", "preserve"):
        predicates = result.get(name)
        if not isinstance(predicates, list):
            raise ValueError("both intent conjunctions must be arrays")
        for predicate in predicates:
            kind = predicate.get("predicate") if isinstance(predicate, dict) else None
            if kind in (SokobanPredicate.BOX_AT, SokobanPredicate.PLAYER_AT):
                predicate["row"], predicate["col"] = transform_coordinate(predicate["row"], predicate["col"], size,
                                                                         symmetry)
            elif kind != SokobanPredicate.BOXES_ON_GOALS_AT_LEAST:
                raise ValueError("unsupported intent predicate")
    return result


def transform_training_record(record: dict[str, Any], symmetry: BoardSymmetry | str) -> dict[str, Any]:
    """Return only permitted training fields, with no stale future-state evidence."""
    rows = record["start_observation"]
    if record.get("task_goal") != TRANSFORM_INVARIANT_TASK:
        raise ValueError("augmentation requires the coordinate-free, rotation-invariant Sokoban task")
    if validate_prediction(record["intent"], rows):
        raise ValueError("only checked, initially-unsatisfied intents can be augmented")
    result = {key: record[key] for key in ("record_id", "layout_id", "split", "task_goal")}
    result["start_observation"] = transform_rows(rows, symmetry)
    result["intent"] = transform_intent(record["intent"], len(rows), symmetry)
    if "effect_type" in record:
        result["effect_type"] = record["effect_type"]
    if validate_prediction(result["intent"], result["start_observation"]):
        raise ValueError("transformed intent is not valid on its transformed board")
    return result
