from abc import ABC
from abc import abstractmethod
from typing import TypeAlias

import numpy as np
from torch import Tensor

from carl.environment.training_goal import TrainingGoal
from carl.utils.aliases import State

StatePair: TypeAlias = tuple[State, State]
StateAction: TypeAlias = tuple[State, int]
ActionState: TypeAlias = tuple[int, State]


class GameTokenizer(ABC):
    @abstractmethod
    def board_tokenizer(self, board: np.ndarray | str) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def board_detokenizer(self, sequence_of_tokens: list[int]) -> np.ndarray | str:
        raise NotImplementedError

    @abstractmethod
    def x_y_tokenizer(
        self,
        x: State | StatePair | StateAction,
        y: State | int | StateAction | ActionState,
        training_goal: TrainingGoal,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def action_detokenizer(self, sequence_of_tokens: list[int]) -> int | None:
        pass
