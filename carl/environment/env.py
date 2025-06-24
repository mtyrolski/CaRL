from abc import ABC
from abc import abstractmethod
from enum import Enum
from enum import auto
from typing import TypeVar

import numpy as np
# plotly fig
from matplotlib import figure as plt
from plotly import graph_objects as go
from torch import Tensor

from carl.environment.tokenizer import GameTokenizer

ReadableReprT = TypeVar('RepresentationT', str, go.Figure, plt.Figure)


class RepresentationType(Enum):
    PLT_FIGURE = auto()
    GO_FIGURE = auto()
    STR = auto()


class GameEnv(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def tokenizer(self) -> GameTokenizer:
        raise NotImplementedError

    @abstractmethod
    def detect_action(self, board_before: np.ndarray, board_after: np.ndarray) -> int | None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def distribution_to_action(distribution: Tensor) -> int:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError

    @abstractmethod
    def next_state(self, state: np.ndarray, action: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_solved(self, board: np.ndarray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def state_to_repr(
        self,
        state: np.ndarray,
        title: str | None = None,
    ) -> ReadableReprT:
        pass

    @abstractmethod
    def many_states_to_repr(
        self,
        states: list[np.ndarray],
        title: str | None = None,
    ) -> ReadableReprT:
        pass

    @abstractmethod
    def set_state(self, state: np.ndarray) -> None:
        raise NotImplementedError
