"""
Defines the abstract interface for game environments, handling state transitions,
representations, and action mappings for reinforcement learning tasks.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TypeVar

# Visualization types
from matplotlib.figure import Figure as MatplotlibFigure
from plotly.graph_objects import Figure as PlotlyFigure
from torch import Tensor

from carl.environment.tokenizer import GameTokenizer
from carl.utils.aliases import State

# Resulting representation can be a string or a figure
ReadableReprT = TypeVar('ReadableReprT', str, MatplotlibFigure, PlotlyFigure)


class RepresentationType(Enum):
    PLT_FIGURE = auto()
    GO_FIGURE = auto()
    STR = auto()


class GameEnv(ABC):
    """Abstract base class for game environments."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique environment identifier."""
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> GameTokenizer:
        """Tokenizer for converting states and actions."""
        ...

    @abstractmethod
    def detect_action(self, board_before: State, board_after: State) -> Optional[int]:
        """Infer the action taken between two consecutive states."""
        ...

    @staticmethod
    @abstractmethod
    def distribution_to_action(distribution: Tensor) -> int:
        """Map a model output (probabilities/logits) to a discrete action."""
        ...

    @abstractmethod
    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Apply an action and return (state, reward, done, info)."""
        ...

    @abstractmethod
    def next_state(self, state: State, action: int) -> State:
        """Compute the next state without side effects."""
        ...

    @abstractmethod
    def is_solved(self, board: State) -> bool:
        """Check if the given state is a terminal/solved condition."""
        ...

    @abstractmethod
    def state_to_repr(
        self,
        state: State,
        title: Optional[str] = None,
    ) -> ReadableReprT:
        """Render a single state to a human-readable form."""
        ...

    @abstractmethod
    def many_states_to_repr(
        self,
        states: List[State],
        title: Optional[str] = None,
    ) -> ReadableReprT:
        """Render multiple states as a combined representation."""
        ...

    @abstractmethod
    def set_state(self, state: State) -> None:
        """Override the current environment state."""
        ...
