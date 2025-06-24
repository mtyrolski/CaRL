import torch
from loguru import logger
from torch import Tensor
from torch import tensor

from carl.environment.tokenizer import GameTokenizer
from carl.environment.training_goal import TrainingGoal


class RubikCubeTokenizer(GameTokenizer):
    def __init__(
        self,
        cut_distance: int | None = None,
        type_of_value_training: str = 'regression',
        num_classes_generation: int | None = None,
    ) -> None:
        if type_of_value_training == 'classification':
            assert cut_distance is not None, ('Cut distance must be specified for classification.'
                                              ' This is number of class labels.')

        if type_of_value_training == 'generation':
            assert (num_classes_generation is not None), 'Number of classes for generation must be specified.'

        self._sequence_length: int = 54
        self._special_tokens: list[str] = [
            '<BOS>',
            '<PAD>',
            '<EOS>',
            '<SEP>',
            '<UNK>',
            '<MASK>',
            '<CLS>',
        ]

        self._vocabulary: list[str] = self._special_tokens + [
            'r',
            'g',
            'b',
            'w',
            'y',
            'o',
        ]

        self._distance_to_tokens: dict[int, int] | None = None
        self._tokens_to_distance: dict[int, int] | None = None

        if type_of_value_training == 'generation':
            self._distance_to_tokens = {i: i + 13 for i in range(num_classes_generation)}
            self._tokens_to_distance = {value: key for key, value in self._distance_to_tokens.items()}

        self._action_space_to_tokens: dict[int, int] = dict(list(enumerate(range(13, 25))))
        self._tokens_to_action_space: dict[int, int] = {
            value: key for key, value in self._action_space_to_tokens.items()
        }
        self._special_tokens_to_str: dict[int, str] = dict(list(enumerate(self._special_tokens)))
        self._tokens_to_str: dict[int, str] = dict(list(enumerate(self._vocabulary)))
        self._str_to_tokens: dict[str, int] = {str_: token for token, str_ in self._tokens_to_str.items()}
        assert len(self._str_to_tokens) == len(self._str_to_tokens), 'There are some duplicated lexemes in vocabulary.'

        self._cut_distance = cut_distance
        self._type_of_value_training = type_of_value_training

        assert self._type_of_value_training in [
            'regression', 'classification', 'generation'
        ], ('type_of_value_training must be either "regression", "classification" or "generation".'
            'Also it should be the same as the type of value training of the value network evaluation.')

    def board_tokenizer(self, board: str) -> Tensor:
        return tensor([self._str_to_tokens[i] for i in board])

    def board_detokenizer(self, sequence_of_tokens: list[int]) -> str | None:
        try:
            filtered_tokens: list[int] = [
                token for token in sequence_of_tokens
                if token in self._tokens_to_str and token not in self._special_tokens_to_str
            ]
            board: str = ''.join([self._tokens_to_str[token] for token in filtered_tokens])
            if len(board) == self._sequence_length:
                return board
        except AssertionError:
            logger.warning('Board is not valid')

    def action_tokenizer(self, action: int) -> Tensor:
        return tensor([self._action_space_to_tokens[action]])

    def action_detokenizer(self, sequence_of_tokens: list[int]) -> int | None:
        try:
            filtered_tokens: list[int] = [
                token for token in sequence_of_tokens if token in self._tokens_to_action_space
            ]
            action: int = self._tokens_to_action_space[filtered_tokens[0]]
            return action
        except AssertionError:
            logger.warning('Action is not valid')

    def distance_tokenizer(self, distance: int) -> Tensor:
        return tensor([self._distance_to_tokens[distance]])

    def distance_detokenizer(self, sequence_of_tokens: list[int]) -> int | None:
        try:
            filtered_tokens: list[int] = [token for token in sequence_of_tokens if token in self._tokens_to_distance]
            distance: int = self._tokens_to_distance[filtered_tokens[0]]
            return distance
        except AssertionError:
            logger.warning('Distance is not valid')

    def x_y_tokenizer(
        self,
        x: str | tuple[str, str] | tuple[str, int],
        y: str | int | tuple[str, int],
        training_goal: TrainingGoal,
    ) -> tuple[Tensor, Tensor]:
        match training_goal:
            case TrainingGoal.POLICY:
                return torch.cat(
                    (
                        tensor([self._str_to_tokens['<CLS>']]),
                        self.board_tokenizer(x),
                        tensor([self._str_to_tokens['<SEP>']]),
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
                            tensor([self._str_to_tokens['<CLS>']]),
                            self.board_tokenizer(x),
                            tensor([self._str_to_tokens['<SEP>']]),
                        ),
                        dim=0,
                    )[None, :],
                    target,
                )
            case TrainingGoal.CLLP:
                x1: str
                x2: str
                x1, x2 = x
                return torch.cat(
                    (
                        tensor([self._str_to_tokens['<CLS>']]),
                        self.board_tokenizer(x1),
                        tensor([self._str_to_tokens['<SEP>']]),
                        self.board_tokenizer(x2),
                        tensor([self._str_to_tokens['<SEP>']]),
                    ),
                    dim=0,
                )[None, :], tensor([y])
            case TrainingGoal.GENERATOR:
                return (
                    torch.cat(
                        (
                            self.board_tokenizer(x),
                            tensor([self._str_to_tokens['<SEP>']]),
                        ),
                        dim=0,
                    )[None, :],
                    self.board_tokenizer(y)[None, :],
                )
            case TrainingGoal.POLICY_GENERATION:
                return (
                    torch.cat(
                        (
                            self.board_tokenizer(x),
                            tensor([self._str_to_tokens['<SEP>']]),
                        ),
                        dim=0,
                    )[None, :],
                    self.action_tokenizer(y)[None, :],
                )
            case TrainingGoal.VALUE_GENERATION:
                return (
                    torch.cat(
                        (
                            self.board_tokenizer(x),
                            tensor([self._str_to_tokens['<SEP>']]),
                        ),
                        dim=0,
                    )[None, :],
                    self.distance_tokenizer(y)[None, :],
                )
            case TrainingGoal.STATE_ACTION_STATE:
                state: str
                action: int

                state, action = x

                return (
                    torch.cat(
                        (
                            self.board_tokenizer(state),
                            tensor([self._str_to_tokens['<SEP>']]),
                            self.action_tokenizer(action),
                            tensor([self._str_to_tokens['<SEP>']]),
                        ),
                        dim=0,
                    )[None, :],
                    self.board_tokenizer(y)[None, :],
                )
            case TrainingGoal.STATE_STATE_ACTION:
                state: str
                action: int
                predicted_state: str

                state = x
                predicted_state, action = y

                return (
                    torch.cat(
                        (self.board_tokenizer(state), tensor([self._str_to_tokens['<SEP>']])),
                        dim=0,
                    )[None, :],
                    torch.cat(
                        (
                            self.board_tokenizer(predicted_state),
                            tensor([self._str_to_tokens['<SEP>']]),
                            self.action_tokenizer(action),
                        ),
                        dim=0,
                    )[None, :],
                )
            case TrainingGoal.STATE_ACTION_STATE_GENERATOR:
                state: str
                action: int
                predicted_state: str

                state = x
                action, predicted_state = y

                return (
                    torch.cat(
                        (self.board_tokenizer(state), tensor([self._str_to_tokens['<SEP>']])),
                        dim=0,
                    )[None, :],
                    torch.cat(
                        (
                            self.action_tokenizer(action),
                            tensor([self._str_to_tokens['<SEP>']]),
                            self.board_tokenizer(predicted_state),
                        ),
                        dim=0,
                    )[None, :],
                )
