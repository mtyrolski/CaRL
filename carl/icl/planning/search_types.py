"""Serializable search outcomes, paths and observer contracts."""
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

from carl.icl.types import Board, BoardRows, SearchEvent

FinishReason: TypeAlias = Literal[
    "frontier_exhausted",
    "solved",
    "action_budget",
    "expansion_budget",
    "api_budget",
    "api_error",
]

TransitionKind: TypeAlias = Literal["primitive", "subgoal", "direct"]
TransitionStatus: TypeAlias = Literal["new", "reopened", "duplicate", "no_op", "rejected", "solved", "interrupted"]


@dataclass(frozen=True)
class Attempt:
    """Actual endpoint and actions, including unsuccessful/no-op attempts."""

    state: Board
    actions: list[int]
    reached: bool
    solved: bool


@dataclass
class SearchResult:
    """Serializable outcome; attempted actions are distinct from solution length."""

    solved: bool = False
    finish_reason: FinishReason = "frontier_exhausted"
    attempted_actions: int = 0
    expansions: int = 0
    accepted: int = 0
    rejected: int = 0
    reopened: int = 0
    primitive_successors: int = 0
    action_path: list[int] = field(default_factory=list)
    subgoal_path: list[BoardRows] = field(default_factory=list)
    events: list[SearchEvent] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    def count_step(self) -> None:
        """Charge every simulator action, even when the state is unchanged."""
        self.attempted_actions += 1


@dataclass(frozen=True)
class SearchPath:
    """A state and the primitive plan that reaches it from the initial board."""

    state: Board
    actions: list[int]
    endpoints: list[Board]

    @property
    def key(self) -> bytes:
        """Use categorical tiles so array dtype does not affect state identity."""
        key: bytes = self.state.argmax(-1).tobytes()
        return key

    def extend(self, state: Board, actions: list[int]) -> "SearchPath":
        """Append an executed segment without modifying the parent plan."""
        return SearchPath(state, self.actions + actions, self.endpoints + [state])


@dataclass(order=True, frozen=True)
class QueueEntry:
    """Only scale, value and insertion order participate in heap comparisons."""

    negative_scale: float
    negative_value: float
    serial: int
    path: SearchPath = field(compare=False)


@dataclass(frozen=True)
class SearchTransition:
    """A completed or interrupted attempt, including rejected frontier candidates."""

    source: SearchPath
    destination: SearchPath
    kind: TransitionKind
    status: TransitionStatus
    scale: float
    actions: tuple[int, ...]
    proposal: Board | None = None
    reached: bool = False
    solved: bool = False


class SearchObserver(Protocol):
    """Optional observation of actual search operations, without extra inference."""

    def on_start(self, root: SearchPath) -> None:
        """Record the initial search path."""
        ...

    def on_expansion(self, entry: QueueEntry, index: int, actions: int, queue_size: int) -> None:
        """Record a popped frontier entry and its priority."""
        ...

    def on_transition(self, transition: SearchTransition) -> None:
        """Record the actual outcome of one high-level attempt or primitive action."""
        ...

    def on_finish(self, result: SearchResult) -> None:
        """Record final accounting and status."""
        ...
