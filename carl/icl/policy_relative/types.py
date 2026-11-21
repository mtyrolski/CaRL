"""Types for policy-relative generator experiments."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

MISSING_SCALE = "<missing-scale>"
SCALE_INSTRUCTION = (
    " The requested relative scale is a within-context percentile of successful "
    "controller action durations. A missing-scale marker means no scale is requested."
)


class Arm(StrEnum):
    UNSCALED = "unscaled"
    SCALED = "scaled"
    BASE = "B"
    DEMONSTRATION = "D"
    MEASURED = "N"
    ABSOLUTE = "T"
    RELATIVE = "U"


class Mode(StrEnum):
    INSPECT = "inspect"
    RESOURCE_CHECK = "resource-check"
    BENCHMARK = "benchmark"
    TRAIN = "train"
    EVALUATE = "evaluate"


class AttemptStatus(StrEnum):
    ACHIEVED = "achieved"
    ACTION_LIMIT = "action_limit"
    TASK_SOLVED = "task_solved"
    PRESERVATION_VIOLATED = "preservation_violated"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


class PredictionStatus(StrEnum):
    VALID = "valid_intent"
    INVALID_JSON = "invalid_json"
    INVALID_INTENT = "invalid_intent"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class Candidate:
    record: dict[str, Any]
    success_fraction: float | None
    mean_success_actions: float | None
    u: float | None
    weight: float
    measured: bool = True


@dataclass(frozen=True)
class Context:
    context_id: str
    layout_id: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class FitData:
    train: tuple[Context, ...]
    validation: tuple[Context, ...]
    controller_hash: str | None
    action_cap: int | None
    attempts_per_candidate: int | None
    audit: dict[str, Any]
