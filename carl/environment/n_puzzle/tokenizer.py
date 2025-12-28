import numpy as np
from typing import TypeAlias
import torch
from loguru import logger
from torch import Tensor
from torch import tensor

from carl.environment.tokenizer import GameTokenizer
from carl.environment.training_goal import TrainingGoal
from carl.utils.aliases import State

StatePair: TypeAlias = tuple[State, State]
StateAction: TypeAlias = tuple[State, int]
ActionState: TypeAlias = tuple[int, State]


class NPuzzleTokenizer(GameTokenizer):
    def __init__(
            self,
            cut_distance: int | None = None,
            type_of_value_training: str = 'regression',
            size_of_board: tuple[int, int] = (5, 5),
    ):
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

        self._tokens_to_n_puzzle_vocab: dict[int, int] = {i + 15: i for i in range(self._size_of_board[0]**2)}
        self._n_puzzle_vocab_to_tokens: dict[int, int] = {i: i + 15 for i in range(self._size_of_board[0]**2)}

        self._cut_distance = cut_distance
        self._type_of_value_training = type_of_value_training

        assert self._type_of_value_training in [
            'regression', 'classification'
        ], ('type_of_value_training must be either "regression" or "classification". '
            'Also it should be the same as the type of value training of the value network evaluation.')

    @property
    def size_of_board(self) -> tuple[int, int]:
        return self._size_of_board

    def board_tokenizer(self, board: np.ndarray) -> Tensor:
        side_of_board: int = int(np.sqrt(len(board)))
        assert side_of_board == self._size_of_board[0], (
            f'Board must be square, and of the same size as the tokenizer '
            f'was initialized with, i.e. {self._size_of_board}. Got {side_of_board}')

        return tensor([self._n_puzzle_vocab_to_tokens[i] for i in board])

    def board_detokenizer(self, sequence_of_tokens: list[int]) -> np.ndarray | None:
        try:
            filtered_tokens: list[int] = [
                token for token in sequence_of_tokens
                if token not in self._special_vocab_to_tokens.values() and token in self._tokens_to_n_puzzle_vocab
            ]
            board: np.ndarray = np.array([self._tokens_to_n_puzzle_vocab[token] for token in filtered_tokens])
            if len(board) == self._size_of_board[0]**2:
                return board
        except AssertionError:
            logger.warning('Board is not valid')


    def x_y_tokenizer(
        self,
        x: State | StatePair | StateAction,
        y: State | int | StateAction | ActionState,
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
