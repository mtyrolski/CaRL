"""The observable Sokoban intent contract and deterministic fragment checker."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Literal, TypeAlias, TypedDict, cast

import numpy as np

from carl.icl.boards import SYMBOLS, from_rows, to_rows, validate_subgoal
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_augmentation import TRANSFORM_INVARIANT_TASK
from carl.icl.intent_data import SCHEMA_VERSION, SokobanPredicate, validate_intent
from carl.icl.intents.representation.formats import Representation, parse_representation, render_representation
from carl.icl.types import Board

DOMAIN_VERSION: Final = "sokoban-observation-v1"
TASK_GOAL: Final = TRANSFORM_INVARIANT_TASK
UNKNOWN_TILE: Final = "?"


class Action(IntEnum):
    """Primitive simulator actions."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class AtCell(TypedDict):
    predicate: Literal["box_at", "player_at"]
    row: int
    col: int


class GoalCount(TypedDict):
    predicate: Literal["boxes_on_goals_at_least"]
    count: int


Predicate: TypeAlias = AtCell | GoalCount


class Intent(TypedDict):
    """A conjunction of endpoint effects and a conjunction of invariants."""

    effects: list[Predicate]
    preserve: list[Predicate]
    instruction: None


@dataclass(frozen=True)
class PredicateSpec:
    name: SokobanPredicate
    arguments: tuple[str, ...]
    meaning: str


@dataclass(frozen=True)
class DomainSpec:
    version: str
    intent_schema: str
    task_goal: str
    board_symbols: str
    tile_meanings: tuple[tuple[str, str], ...]
    unknown_tile: str
    coordinates: str
    actions: tuple[str, ...]
    predicates: tuple[PredicateSpec, ...]
    effect_rule: str
    preservation_rule: str


DOMAIN_SPEC: Final = DomainSpec(
    version=DOMAIN_VERSION,
    intent_schema=SCHEMA_VERSION,
    task_goal=TASK_GOAL,
    board_symbols=SYMBOLS,
    tile_meanings=tuple(zip(
        SYMBOLS, ("wall", "floor", "goal", "box on goal", "box", "player", "player on goal"),
        strict=True,
    )),
    unknown_tile=UNKNOWN_TILE,
    coordinates="Zero-based row and column",
    actions=tuple(action.name for action in Action),
    predicates=(
        PredicateSpec(SokobanPredicate.BOX_AT, ("row", "col"), "A box occupies the cell."),
        PredicateSpec(SokobanPredicate.PLAYER_AT, ("row", "col"), "The player occupies the cell."),
        PredicateSpec(SokobanPredicate.BOXES_ON_GOALS_AT_LEAST, ("count",), "At least count boxes occupy goals."),
    ),
    effect_rule="Every effect is false at the start and true in the same final observation.",
    preservation_rule="Every preservation predicate is true at every observation, including both endpoints.",
)


def _rows(rows: Sequence[str], alphabet: str) -> tuple[str, ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("A board must be a sequence of nonempty rows")
    result = tuple(rows)
    if not result or any(not isinstance(row, str) or not row for row in result):
        raise ValueError("A board must be a sequence of nonempty rows")
    allowed = set(alphabet)
    if any(len(row) != len(result[0]) or not set(row) <= allowed for row in result):
        raise ValueError("Invalid observation dimensions or symbols")
    return result


def _full_board(rows: tuple[str, ...]) -> Board:
    board = from_rows(list(rows))
    validate_subgoal(board, board)
    return board


@dataclass(frozen=True)
class SokobanObservation:
    """A model-visible board. A question mark hides the entire cell."""

    rows: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _rows(self.rows, SYMBOLS + UNKNOWN_TILE))
        if not self.partial:
            _full_board(self.rows)
        elif sum(row.count("@") + row.count("+") for row in self.rows) > 1:
            raise ValueError("An observation cannot show more than one player")

    @property
    def partial(self) -> bool:
        return any(UNKNOWN_TILE in row for row in self.rows)


@dataclass(frozen=True)
class SokobanSnapshot:
    """Oriented full board for restoring deterministic simulator state."""

    rows: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", _rows(self.rows, SYMBOLS))
        _full_board(self.rows)

    @property
    def observation(self) -> SokobanObservation:
        return SokobanObservation(self.rows)

    def restore(self) -> Board:
        """Return a fresh simulator state so rollouts cannot mutate the snapshot."""
        return from_rows(list(self.rows))


CheckStatus: TypeAlias = Literal["accepted", "rejected", "unknown"]


@dataclass(frozen=True)
class IntentCheck:
    """Result of checking all clauses over a selected demonstration fragment."""

    status: CheckStatus
    errors: tuple[str, ...] = ()


def decode_intent(sketch: object) -> Intent:
    """Read strict JSON or a legacy dictionary with an omitted instruction."""
    if isinstance(sketch, str):
        return cast(Intent, parse_representation(Representation.JSON, sketch))
    if not isinstance(sketch, dict):
        raise ValueError("An intent sketch must be a JSON object")
    try:
        text = json.dumps({"instruction": None, **sketch}, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("An intent sketch must contain JSON values") from exc
    return cast(Intent, parse_representation(Representation.JSON, text))


def render_instruction(intent: Intent) -> str:
    """Render ℓ(h) from the checked predicates using the existing natural format."""
    return render_representation(Representation.NATURAL, intent)


def predicate_truth(predicate: Predicate, observation: SokobanObservation) -> bool | None:
    """Return known truth, known falsity, or None for a checked predicate."""
    rows = observation.rows
    if predicate["predicate"] == "boxes_on_goals_at_least":
        count = predicate["count"]
        visible = sum(row.count("*") for row in rows)
        if visible >= count:
            return True
        if visible + sum(row.count(UNKNOWN_TILE) for row in rows) < count:
            return False
        return None
    cell = rows[predicate["row"]][predicate["col"]]
    if cell == UNKNOWN_TILE:
        return None
    return cell in ("$*" if predicate["predicate"] == SokobanPredicate.BOX_AT else "@+")


def _partial_layout(observations: Sequence[SokobanObservation]) -> list[list[str]]:
    """Collect wall and goal facts wherever at least one observation sees the cell."""
    first = observations[0].rows
    layout = [[UNKNOWN_TILE] * len(first[0]) for _ in first]
    for observation in observations:
        if len(observation.rows) != len(first) or len(observation.rows[0]) != len(first[0]):
            raise ValueError("Fragment observation shape changed")
        for row, tiles in enumerate(observation.rows):
            for col, tile in enumerate(tiles):
                if tile == UNKNOWN_TILE:
                    continue
                static = "#" if tile == "#" else "." if tile in ".*+" else "-"
                if layout[row][col] not in (UNKNOWN_TILE, static):
                    raise ValueError("Fragment changes a visible wall or goal square")
                layout[row][col] = static
    return layout


def _check_partial_legality(intent: Intent, layout: list[list[str]]) -> tuple[str, ...]:
    errors: list[str] = []
    players, boxes = set(), set()
    maximum_goals = sum(tile in (".", UNKNOWN_TILE) for row in layout for tile in row)
    for group in ("effects", "preserve"):
        seen = set()
        for index, predicate in enumerate(intent[group]):
            where = f"{group}[{index}]"
            key = tuple(sorted(predicate.items()))
            if key in seen:
                errors.append(f"{where}: duplicate predicate")
            seen.add(key)
            if predicate["predicate"] == "boxes_on_goals_at_least":
                if not 1 <= predicate["count"] <= maximum_goals:
                    errors.append(f"{where}: goal count outside valid range")
                continue
            row, col = predicate["row"], predicate["col"]
            if not (0 <= row < len(layout) and 0 <= col < len(layout[0])):
                errors.append(f"{where}: row/column out of range")
            elif layout[row][col] == "#":
                errors.append(f"{where}: position is a wall")
            if predicate["predicate"] == SokobanPredicate.PLAYER_AT:
                players.add((row, col))
            else:
                boxes.add((row, col))
    if len(players) > 1:
        errors.append("conjunction requires the player at multiple positions")
    if players & boxes:
        errors.append("conjunction requires player and box on the same position")
    return tuple(errors)


def _check_partial(intent: Intent, observations: Sequence[SokobanObservation]) -> IntentCheck:
    try:
        problems = _check_partial_legality(intent, _partial_layout(observations))
    except ValueError as exc:
        return IntentCheck("rejected", (str(exc),))
    if problems:
        return IntentCheck("rejected", problems)
    rejected: list[str] = []
    unknown: list[str] = []
    for index, predicate in enumerate(intent["effects"]):
        start = predicate_truth(predicate, observations[0])
        end = predicate_truth(predicate, observations[-1])
        if start is True:
            rejected.append(f"effects[{index}]: effect already true at start")
        elif start is None:
            unknown.append(f"effects[{index}]: unknown at start")
        if end is False:
            rejected.append(f"effects[{index}]: effect false at end")
        elif end is None:
            unknown.append(f"effects[{index}]: unknown at end")
    for index, predicate in enumerate(intent["preserve"]):
        for step, observation in enumerate(observations):
            truth = predicate_truth(predicate, observation)
            if truth is False:
                rejected.append(f"preserve[{index}]: false at fragment state {step}")
                break
            if truth is None:
                unknown.append(f"preserve[{index}]: unknown at fragment state {step}")
    if rejected:
        return IntentCheck("rejected", tuple(rejected))
    if unknown:
        return IntentCheck("unknown", tuple(unknown))
    return IntentCheck("accepted")


def check_intent(intent: Intent, observations: Sequence[SokobanObservation]) -> IntentCheck:
    """Verify an intent against the whole fragment, with unknowns retained explicitly."""
    if len(observations) < 2:
        return IntentCheck("rejected", ("fragment must contain at least two states",))
    if any(observation.partial for observation in observations):
        return _check_partial(intent, observations)
    errors = validate_intent(intent, [list(observation.rows) for observation in observations])
    return IntentCheck("rejected", tuple(errors)) if errors else IntentCheck("accepted")


def replay_fragment(start: SokobanSnapshot, actions: Sequence[int]) -> tuple[SokobanSnapshot, ...]:
    """Restore and replay complete Sokoban states before annotation."""
    state = start.restore()
    simulator = environment(state)
    snapshots = [start]
    for index, action in enumerate(actions):
        if not isinstance(action, int) or isinstance(action, bool) or action not in range(len(Action)):
            raise ValueError(f"Invalid primitive action at index {index}")
        if simulator.is_solved(state):
            raise ValueError(f"Fragment contains an action after terminal state at index {index}")
        successor = simulator.next_state(state, action)
        if not isinstance(successor, np.ndarray):
            raise TypeError("Sokoban simulator must return an array state")
        state = np.asarray(successor, dtype=np.float32)
        snapshots.append(SokobanSnapshot(tuple(to_rows(state))))
    return tuple(snapshots)
