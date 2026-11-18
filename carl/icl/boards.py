"""Lossless ASCII interchange using the environment's seven tile channels."""

import hashlib
from typing import Final

import numpy as np

SYMBOLS: Final[str] = "#-.*$@+"


def tiles(state: np.ndarray | str) -> np.ndarray:
    """Validate the one-hot encoding and return categorical tile indices."""
    state = np.asarray(state)
    if state.ndim != 3 or state.shape[-1] != 7:
        raise ValueError("Expected a Sokoban state with shape (height, width, 7)")
    if not np.all((state == 0) | (state == 1)) or not np.all(state.sum(-1) == 1):
        raise ValueError("Expected a one-hot Sokoban state")
    result: np.ndarray = state.argmax(-1)
    return result


def to_rows(state: np.ndarray | str) -> list[str]:
    """Encode a board losslessly as ASCII rows."""
    return ["".join(SYMBOLS[int(tile)] for tile in row) for row in tiles(state)]


def from_rows(rows: list[str]) -> np.ndarray:
    """Decode rectangular ASCII rows into the seven-channel environment representation."""
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, str) and row for row in rows)
    ):
        raise ValueError("Expected nonempty ASCII board rows")
    if any(len(row) != len(rows[0]) for row in rows) or any(
        c not in SYMBOLS for row in rows for c in row
    ):
        raise ValueError("Invalid board dimensions or symbols")
    indices = np.array([[SYMBOLS.index(c) for c in row] for row in rows])
    result: np.ndarray = np.eye(7, dtype=np.float32)[indices]
    return result


def level_key(state: np.ndarray | str) -> str:
    """Hash the board dimensions, walls and goals independently of movable pieces."""
    board = tiles(state)
    layout = np.stack([board == 0, np.isin(board, [2, 3, 6])])
    return hashlib.sha256(str(board.shape).encode() + layout.tobytes()).hexdigest()


def state_hash(state: np.ndarray | str) -> str:
    """Identify the complete, oriented board, including boxes and player."""
    board = tiles(state).astype(np.uint8)
    data = f"carl-state-v1:{board.shape[0]}x{board.shape[1]}:".encode() + board.tobytes()
    return hashlib.sha256(data).hexdigest()


def canonical_layout_id(state: np.ndarray | str) -> str:
    """Identify walls and goals modulo rotations and reflections, ignoring pieces."""
    board = tiles(state)
    layout = (board == 0).astype(np.uint8) + 2 * np.isin(board, [2, 3, 6]).astype(np.uint8)
    variants = []
    for flipped in (layout, np.fliplr(layout)):
        for rotation in range(4):
            transformed = np.rot90(flipped, rotation)
            header = f"carl-layout-d4-v1:{transformed.shape[0]}x{transformed.shape[1]}:".encode()
            variants.append(header + transformed.tobytes())
    return hashlib.sha256(min(variants)).hexdigest()


def validate_subgoal(state: np.ndarray | str, subgoal: np.ndarray) -> None:
    """Require the same layout, box count and exactly one player."""
    before, after = tiles(state), tiles(subgoal)
    if before.shape != after.shape or level_key(state) != level_key(subgoal):
        raise ValueError("Subgoal must preserve the board shape, walls and goal squares")
    boxes = int(np.isin(before, [3, 4]).sum())
    if boxes == 0 or int(np.isin(after, [3, 4]).sum()) != boxes:
        raise ValueError("Subgoal must preserve the number of boxes")
    if int(np.isin(before, [5, 6]).sum()) != 1 or int(np.isin(after, [5, 6]).sum()) != 1:
        raise ValueError("Each board must contain exactly one player")


def embedding_text(rows: list[str]) -> str:
    """Explicit coordinates supplement ASCII, whose spatial similarity is uncertain."""
    state = from_rows(rows)
    board = tiles(state)
    features = {
        name: np.argwhere(np.isin(board, values)).tolist()
        for name, values in (
            ("walls", [0]),
            ("goals", [2, 3, 6]),
            ("boxes", [3, 4]),
            ("player", [5, 6]),
        )
    }
    return "Sokoban board\n" + "\n".join(rows) + "\n" + str(features)
