"""Shared Sokoban inputs for model and search tests."""

import pytest

from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.datasets import environment
from carl.icl.types import Board
from carl.icl.types import TrainingData


@pytest.fixture(name="board")
def sokoban_board():
    return from_rows(["#####", "#---#", "#@$.#", "#---#", "#####"])


def training_for(board: Board) -> TrainingData:
    env = environment(board)
    after = env.next_state(board, 3)
    return {"contexts": [{"id": "train", "state": to_rows(board), "remaining_actions": 1,
                          "candidates": [{"k": 1, "actions": [3], "subgoal": to_rows(after)}],
                          "calibration_indices": [0]}]}
