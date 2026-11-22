"""Exact, versioned checks for small Sokoban effect intents.

Coordinates are zero-based (row, column), boxes are indistinguishable, and
effects/preservation are conjunctions.  A training label is accepted only if
*each* effect is false initially and true at the end and every preservation
condition holds at every recorded state, including both endpoints.
"""
from enum import StrEnum
from typing import Any

from carl.icl.boards import from_rows
from carl.icl.boards import validate_subgoal

SCHEMA_VERSION = "sokoban-intent-v1"
DA_VERSION = "carl-da-v1"


class SokobanPredicate(StrEnum):
    """Predicate names used by the versioned intent JSON schema."""

    BOX_AT = "box_at"
    PLAYER_AT = "player_at"
    BOXES_ON_GOALS_AT_LEAST = "boxes_on_goals_at_least"


class EffectType(StrEnum):
    BOX = "box"
    BOX_CONJUNCTION = "box_conjunction"
    GOAL_CONJUNCTION = "goal_conjunction"
    GOAL_PROGRESS = "goal_progress"
    WALK_ONLY = "walk_only"


PREDICATES = {predicate.value for predicate in SokobanPredicate}


def predicate_value(predicate: dict[str, Any], rows: list[str]) -> bool:
    """Return the truth of a syntactically checked predicate on an ASCII board."""
    if predicate["predicate"] == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST:
        count: int = predicate["count"]
        return sum(row.count("*") for row in rows) >= count
    char = rows[predicate["row"]][predicate["col"]]
    return char in ("$*" if predicate["predicate"] == SokobanPredicate.BOX_AT else "@+")


def _predicate_errors(item: Any, rows: list[str], where: str) -> list[str]:
    name = item.get("predicate") if isinstance(item, dict) else None
    if not isinstance(name, str) or name not in PREDICATES:
        return [f"{where}: unsupported predicate"]
    expected = {"predicate", "count"
               } if name == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST else {"predicate", "row", "col"}
    if set(item) != expected:
        return [f"{where}: unexpected or missing predicate fields"]
    if name == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST:
        n = item["count"]
        if not isinstance(n, int) or isinstance(
                n, bool) or not 1 <= n <= sum(row.count(".") + row.count("*") + row.count("+") for row in rows):
            return [f"{where}: goal count outside valid range"]
    else:
        r, c = item["row"], item["col"]
        if (not isinstance(r, int) or isinstance(r, bool) or not isinstance(c, int) or isinstance(c, bool)
                or not (0 <= r < len(rows) and 0 <= c < len(rows[0]))):
            return [f"{where}: row/column out of range"]
        if rows[r][c] == "#":
            return [f"{where}: position is a wall"]
    return []


def _intent_structure_error(intent: Any) -> str | None:
    if not isinstance(intent, dict):
        return "intent is not an object"
    if set(intent) - {"effects", "preserve", "instruction"}:
        return "unexpected intent fields"
    effects = intent.get("effects")
    if not isinstance(effects, list) or not effects:
        return "effects must be a nonempty array"
    if not isinstance(intent.get("preserve"), list):
        return "preserve must be an array"
    if intent.get("instruction") is not None and not isinstance(intent["instruction"], str):
        return "instruction must be a string or null"
    return None


def validate_prediction(intent: Any, start_rows: list[str]) -> list[str]:
    """Check syntax and non-triviality using *only* the current board."""
    try:
        from_rows(start_rows)
    except (TypeError, ValueError) as exc:
        return [f"invalid start board: {exc}"]
    structure_error = _intent_structure_error(intent)
    if structure_error:
        return [structure_error]
    effects = intent["effects"]
    preserve = intent["preserve"]
    errors: list[str] = []
    for group, predicates in (("effects", effects), ("preserve", preserve)):
        seen = set()
        for i, pred in enumerate(predicates):
            where = f"{group}[{i}]"
            found = _predicate_errors(pred, start_rows, where)
            errors.extend(found)
            if found:
                continue
            key = tuple(sorted(pred.items()))
            if key in seen:
                errors.append(f"{where}: duplicate predicate")
            seen.add(key)
            truth = predicate_value(pred, start_rows)
            if group == "effects" and truth:
                errors.append(f"{where}: effect already true at start")
            if group == "preserve" and not truth:
                errors.append(f"{where}: preservation false at start")
    if not errors:
        # Every effect and preservation condition must hold simultaneously at
        # the endpoint. One player cannot occupy two cells or share a box cell.
        combined = effects + preserve
        players = {(p["row"], p["col"]) for p in combined if p["predicate"] == SokobanPredicate.PLAYER_AT}
        boxes = {(p["row"], p["col"]) for p in combined if p["predicate"] == SokobanPredicate.BOX_AT}
        if len(players) > 1:
            errors.append("conjunction requires the player at multiple positions")
        if players & boxes:
            errors.append("conjunction requires player and box on the same position")
    return errors


def validate_intent(intent: Any, fragment_rows: list[list[str]]) -> list[str]:
    """Validate an annotation on its complete recorded fragment.

    The entire fragment is checked for preservation, rather than only its
    endpoints.  This is annotation-time checking, not generator input.
    """
    if not isinstance(fragment_rows, list) or len(fragment_rows) < 2:
        return ["fragment must contain at least two states"]
    errors = validate_prediction(intent, fragment_rows[0])
    if errors:
        return errors
    try:
        initial = from_rows(fragment_rows[0])
        for rows in fragment_rows[1:]:
            validate_subgoal(initial, from_rows(rows))
    except (TypeError, ValueError) as exc:
        return [f"invalid fragment board: {exc}"]
    for i, pred in enumerate(intent["effects"]):
        if not predicate_value(pred, fragment_rows[-1]):
            errors.append(f"effects[{i}]: effect false at end")
    for i, pred in enumerate(intent.get("preserve", [])):
        for j, rows in enumerate(fragment_rows):
            if not predicate_value(pred, rows):
                errors.append(f"preserve[{i}]: false at fragment state {j}")
                break
    return errors
