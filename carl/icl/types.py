"""Domain contracts; dynamic dictionaries are confined to JSON/API boundaries."""
from collections.abc import Callable
from typing import Any, Literal, NotRequired, Protocol, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

from carl.utils.aliases import State

Board: TypeAlias = NDArray[np.float32]
Probabilities: TypeAlias = NDArray[np.float64]
Coordinate: TypeAlias = tuple[int, int]
BoardRows: TypeAlias = list[str]
Provider: TypeAlias = Literal["openai", "replay", "mock"]
ComponentRole: TypeAlias = Literal["generator", "policy", "value"]
PathSetting: TypeAlias = Literal[
    "trajectory_source", "data_dir", "cache_dir", "output_dir", "generator_checkpoint"
]
# The SDK accepts heterogeneous JSON schemas, messages and provider metadata.
JSONObject: TypeAlias = dict[str, Any]
MockResponse: TypeAlias = Callable[[], JSONObject]


class Candidate(TypedDict):
    """A verified future board and its trajectory offset/action prefix."""

    subgoal: BoardRows
    k: int
    actions: list[int]


class TrainingContext(TypedDict):
    """One layout contributes equally to the demonstration pool."""

    id: str
    state: BoardRows
    remaining_actions: int
    candidates: list[Candidate]
    calibration_indices: list[int]


class ValueExample(TypedDict):
    """A board and continuation length from a verified training trajectory."""

    context: str
    state: BoardRows
    remaining_actions: int


class TrainingData(TypedDict):
    """Frozen demonstrations with optional export provenance."""

    contexts: list[TrainingContext]
    provenance: NotRequired[JSONObject]
    value_examples: NotRequired[list[ValueExample]]


class Example(Candidate):
    """A source context and target, optionally annotated by policy calibration."""

    state: BoardRows
    context: str
    d: NotRequired[float]
    ratio_d_over_ac: NotRequired[float]
    target_index: NotRequired[int]
    normalized_offset: NotRequired[float]
    u: NotRequired[float]
    weight: NotRequired[float]
    hits: NotRequired[int]
    success_rate: NotRequired[float]


class Instance(TypedDict):
    """Public evaluation input without its private solution witness."""

    id: str
    state: BoardRows
    level_key: str


class TrialData(TypedDict):
    """Resumable hit times and executed-action cost for one candidate."""

    hits: list[int | None]
    attempted_actions: int


class SearchEvent(TypedDict):
    """A proposal and its actual controller endpoint, saved for diagnostics."""

    scale: float
    attempted_actions: int
    reached: bool
    solved: bool
    source: BoardRows
    proposal: BoardRows
    endpoint: BoardRows


class BoardEnvironment(Protocol):
    """The deterministic subset of CaRL's simulator used by ICL."""

    def next_state(self, state: State, action: int, /) -> State:
        """Execute an action after restoring the supplied state."""
        ...

    def is_solved(self, state: State, /) -> bool:
        """Return whether the task's terminal goal is satisfied."""
        ...


class RolloutPolicy(Protocol):
    """The controller contract shared by calibration, search and tracing."""

    def probabilities(self, state: Board, subgoal: Board) -> Probabilities:
        """Return a normalized categorical distribution over four actions."""
        ...


class SearchComponents(RolloutPolicy, Protocol):
    """A controller and heuristic shared by calibration and search."""

    def value(self, state: Board) -> float:
        """Estimate discounted task completion return."""
        ...

    def propose(self, state: Board, scale: float) -> list[Board]:
        """Generate structurally valid subgoals for a positive scale."""
        ...
