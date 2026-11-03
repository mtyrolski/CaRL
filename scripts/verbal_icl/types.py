"""Typed records for the isolated verbal-goal experiment."""
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypedDict, NotRequired

from carl.icl.types import Board, BoardRows

Action = Literal['UP', 'DOWN', 'LEFT', 'RIGHT']
Finish = Literal['solved', 'action_budget', 'expansion_budget', 'frontier_exhausted', 'api_error', 'api_budget']


class GeneratorResponse(TypedDict):
    """Textual conditions, never future board tensors."""

    subgoals: list[str]


class PolicyResponse(TypedDict):
    """Exactly one requested simulator move."""

    action: Action


class JudgeResponse(TypedDict):
    """Independent observed-condition decision and inspectable evidence."""

    reached: bool
    reason: str
    progress: NotRequired[float]


@dataclass(frozen=True)
class Judgment:
    """The model's judgment; this is not a ground-truth semantic oracle."""

    reached: bool
    progress: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class Step:
    """One executed simulator action and the model's following judgment."""

    before: BoardRows
    action: int
    after: BoardRows
    judgment: Judgment | None
    solved: bool


@dataclass(frozen=True)
class Attempt:
    """One controller rollout; the recorded endpoint is the actual board."""

    endpoint: Board
    actions: list[int]
    steps: list[Step]
    judged_reached: bool
    solved: bool
    interrupted: bool = False


@dataclass
class Result:
    """Keep task success distinct from the judge's subgoal claims."""

    solved: bool = False
    finish_reason: Finish = 'frontier_exhausted'
    attempted_actions: int = 0
    expansions: int = 0
    accepted: int = 0
    rejected: int = 0
    reopened: int = 0
    primitive_actions: int = 0
    action_path: list[int] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    error: str | None = None


class Components(Protocol):
    """A generator, a single-action controller, a judge and a board value."""

    def propose(self, state: Board, scale: float) -> list[str]: ...

    def action(self, state: Board, goal: str, source: Board | None = None) -> int: ...

    def judge(self, state: Board, goal: str, source: Board | None = None) -> Judgment: ...

    def value(self, state: Board) -> float: ...
