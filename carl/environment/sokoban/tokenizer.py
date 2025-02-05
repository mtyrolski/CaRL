from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch import Tensor
from torch import tensor

from carl.environment.tokenizer import GameTokenizer
from carl.environment.training_goal import TrainingGoal
from loguru import logger


class SokobanTokenizer(GameTokenizer):
    def __init__(
            self,
            cut_distance: int | None = None,
            type_of_value_training: str = 'regression',
            size_of_board: tuple[int, int] = (12, 12),
            remove_border: bool = False,
    ) -> None:
        self._size_of_board = size_of_board
        assert self._size_of_board[0] == self._size_of_board[1], 'Board must be square'

        if type_of_value_training == 'classification':
            assert cut_distance is not None, ('Cut distance must be specified for classification.'
                                              ' This is number of class labels.')

        self._special_vocab_to_tokens: dict[str, int] = {
            '<BOS>': 0,
            '<PAD>': 1,
            '<EOS>': 2,
            '<SEP>': 3,
            '<UNK>': 4,
            '<MASK>': 5,
            '<CLS>': 6,
        }
        self._sokoban_vocab_to_tokens: dict[str, int] = {
            'empty': 15,
            'wall': 16,
            'goal': 17,
            'box_on_the_goal': 18,
            'box': 19,
            'agent': 20,
            'agent_on_goal': 21,
        }
        self._tokens_to_sokoban_vocab: dict[int, str] = {
            15: 'empty',
            16: 'wall',
            17: 'goal',
            18: 'box_on_the_goal',
            19: 'box',
            20: 'agent',
            21: 'agent_on_goal',
        }
        self._position_to_symbol: dict[str, int] = {
            'empty': 0,
            'wall': 1,
            'goal': 2,
            'box_on_the_goal': 3,
            'box': 4,
            'agent': 5,
            'agent_on_goal': 6,
        }
        self._symbol_to_position: dict[int, str] = {
            0: 'empty',
            1: 'wall',
            2: 'goal',
            3: 'box_on_the_goal',
            4: 'box',
            5: 'agent',
            6: 'agent_on_goal',
        }
        self._cut_distance = cut_distance
        self.remove_border = remove_border
        self._type_of_value_training = type_of_value_training
        assert self._type_of_value_training in [
            'classification', 'regression'
        ], ("Type of value training must be either 'classification' or 'regression."
            ' Also it should be the same as the type of value training of the value network evaluation.')

    @property
    def size_of_board(self) -> tuple[int, int]:
        return self._size_of_board

    def board_tokenizer(self, board: np.ndarray) -> Tensor:
        width: int
        height: int

        if self.remove_border:
            board = self.cut_border(board)
        width, height, _ = board.shape
        assert width == height, 'Board must be square'
        assert (
            width == self._size_of_board[0]), f'Board must be of size {self._size_of_board[0]}x{self._size_of_board[1]}'
        return tensor([
            self._sokoban_vocab_to_tokens[self._symbol_to_position[int(np.argmax(board[i][j]))]]
            for i in range(width)
            for j in range(height)
        ])

    def board_detokenizer(self, tokens: list[int]) -> Optional[np.ndarray]:
        try:
            special_tokens: set[int] = set(self._special_vocab_to_tokens.values())
            valid_tokens: set[int] = set(self._sokoban_vocab_to_tokens.values())
            filtered_tokens: list[int] = [
                token for token in tokens if token not in special_tokens and token in valid_tokens
            ]

            board_size: int

            if self.remove_border:
                filtered_tokens = self.add_border_tokens(filtered_tokens)
                board_size = self._size_of_board[0] + 2
            else:
                board_size = self._size_of_board[0]

            assert board_size == int(np.sqrt(len(filtered_tokens)))

            board = np.zeros((board_size, board_size, 7))

            for idx, token in enumerate(filtered_tokens):
                i: int
                j: int
                symbol_index: int
                i, j = divmod(idx, board_size)
                symbol_index = self._position_to_symbol[self._tokens_to_sokoban_vocab[token]]
                board[i, j, symbol_index] = 1

            return board
        except AssertionError:
            logger.warning('Board is not valid')
            return None

    @staticmethod
    def cut_border(state: np.ndarray) -> np.ndarray:
        return state[1:-1, 1:-1, :]

    def add_border_tokens(self, tokens: list[int]) -> list[int]:
        width: int
        height: int
        width, height = self._size_of_board
        wall_token: int = self._sokoban_vocab_to_tokens['empty']
        top_bottom_border: list[int] = [wall_token] * (width + 2)
        middle_rows: list[list[int]] = [
            [wall_token] + tokens[i * width:(i + 1) * width] + [wall_token] for i in range(height)
        ]
        return top_bottom_border + sum(middle_rows, []) + top_bottom_border

    def x_y_tokenizer(
        self,
        x: np.ndarray | tuple[np.ndarray, np.ndarray],
        y: np.ndarray | int,
        training_goal: TrainingGoal,
    ) -> tuple[Tensor, Tensor]:
        match training_goal:
            case TrainingGoal.POLICY:
                return torch.cat(
                    (
                        tensor([self._special_vocab_to_tokens['<CLS>']]),
                        self.board_tokenizer(x),
                        tensor([self._special_vocab_to_tokens['<SEP>']]),
                    ),
                    dim=0,
                )[None, :], tensor([y])
            case TrainingGoal.VALUE:
                target: Tensor
                if self._type_of_value_training == 'classification':
                    target = tensor(
                        [min(y, self._cut_distance - 1)],
                        dtype=torch.long,
                    )
                else:
                    target = tensor(
                        [y if self._cut_distance is None else min(y, self._cut_distance) / self._cut_distance],
                        dtype=torch.float32,
                    )
                return (
                    torch.cat(
                        (
                            tensor([self._special_vocab_to_tokens['<CLS>']]),
                            self.board_tokenizer(x),
                            tensor([self._special_vocab_to_tokens['<SEP>']]),
                        ),
                        dim=0,
                    )[None, :],
                    target,
                )
            case TrainingGoal.CLLP:
                x1: np.ndarray
                x2: np.ndarray
                x1, x2 = x
                return torch.cat(
                    (
                        tensor([self._special_vocab_to_tokens['<CLS>']]),
                        self.board_tokenizer(x1),
                        tensor([self._special_vocab_to_tokens['<SEP>']]),
                        self.board_tokenizer(x2),
                        tensor([self._special_vocab_to_tokens['<SEP>']]),
                    ),
                    dim=0,
                )[None, :], tensor([y])
            case TrainingGoal.GENERATOR:
                return (
                    torch.cat(
                        (
                            self.board_tokenizer(x),
                            tensor([self._special_vocab_to_tokens['<SEP>']]),
                        ),
                        dim=0,
                    )[None, :],
                    self.board_tokenizer(y)[None, :],
                )
