from abc import ABC
from abc import abstractmethod

import numpy as np
from torch import Tensor

from carl.environment.training_goal import TrainingGoal


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
        x: np.ndarray | tuple[np.ndarray, np.ndarray] | str | tuple[str, str],
        y: np.ndarray | int | str,
        training_goal: TrainingGoal,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def action_detokenizer(self, sequence_of_tokens: list[int]) -> int | None:
        pass
