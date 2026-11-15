"""Strict reversible representations of Sokoban intents."""

import json
import re
from enum import StrEnum
from typing import Any

from carl.icl.intent_data import SokobanPredicate


class Representation(StrEnum):
    JSON = "json"
    FORMAL = "formal"
    NATURAL = "natural"


def _canonical(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "effects",
        "preserve",
        "instruction",
    }:
        raise ValueError("expected effects, preserve and instruction keys")
    if value["instruction"] is not None:
        raise ValueError("this experiment fixes instruction to null")
    result: dict[str, Any] = {"effects": [], "preserve": [], "instruction": None}
    for group in ("effects", "preserve"):
        items = value[group]
        if not isinstance(items, list) or (group == "effects" and not items):
            raise ValueError(
                f"{group} must be an array"
                + (" with at least one item" if group == "effects" else "")
            )
        for predicate in items:
            if not isinstance(predicate, dict):
                raise ValueError("predicate must be an object")
            name = predicate.get("predicate")
            if name == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST:
                fields = ("count",)
            elif name in (SokobanPredicate.BOX_AT, SokobanPredicate.PLAYER_AT):
                fields = ("row", "col")
            else:
                raise ValueError("unknown predicate")
            if set(predicate) != {"predicate", *fields}:
                raise ValueError("unexpected predicate fields")
            if any(
                not isinstance(predicate[key], int)
                or isinstance(predicate[key], bool)
                or predicate[key] < 0
                for key in fields
            ):
                raise ValueError("predicate arguments must be nonnegative integers")
            result[group].append(
                {"predicate": str(name), **{key: predicate[key] for key in fields}}
            )
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _render_predicate(predicate: dict[str, Any], representation: Representation) -> str:
    name = predicate["predicate"]
    if name == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST:
        return (
            f"goals({predicate['count']})"
            if representation == Representation.FORMAL
            else f"at least {predicate['count']} boxes on goals"
        )
    subject = "box" if name == SokobanPredicate.BOX_AT else "player"
    if representation == Representation.FORMAL:
        return f"{subject}({predicate['row']},{predicate['col']})"
    return f"{subject} at row {predicate['row']} column {predicate['col']}"


def render_representation(representation: Representation | str, value: Any) -> str:
    """Serialize all predicates and arrays; reject unsupported instructions."""
    representation = Representation(representation)
    intent = _canonical(value)
    if representation == Representation.JSON:
        return json.dumps(intent, ensure_ascii=False, separators=(",", ":"))
    effects = [_render_predicate(item, representation) for item in intent["effects"]]
    preserve = [_render_predicate(item, representation) for item in intent["preserve"]]
    if representation == Representation.FORMAL:
        return f"E[{';'.join(effects)}] P[{';'.join(preserve)}]"
    return f"Achieve: {' and '.join(effects)}. Preserve: {' and '.join(preserve) if preserve else 'nothing'}."


def _parse_predicate(text: str, representation: Representation) -> dict[str, Any]:
    if representation == Representation.FORMAL:
        position = re.fullmatch(r"(box|player)\((\d+),(\d+)\)", text)
        goals = re.fullmatch(r"goals\((\d+)\)", text)
    else:
        position = re.fullmatch(r"(box|player) at row (\d+) column (\d+)", text)
        goals = re.fullmatch(r"at least (\d+) boxes on goals", text)
    if position:
        return {
            "predicate": "box_at" if position[1] == "box" else "player_at",
            "row": int(position[2]),
            "col": int(position[3]),
        }
    if goals:
        return {"predicate": "boxes_on_goals_at_least", "count": int(goals[1])}
    raise ValueError("predicate does not match the specified representation")


def parse_representation(
    representation: Representation | str, raw: str
) -> dict[str, Any]:
    """Strictly parse one output into the common JSON intent schema."""
    representation = Representation(representation)
    if not isinstance(raw, str):
        raise ValueError("output must be text")
    text = raw.strip()
    if representation == Representation.JSON:

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-JSON constant {value}")

        try:
            return _canonical(
                json.loads(
                    text,
                    object_pairs_hook=_unique_object,
                    parse_constant=reject_constant,
                )
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if representation == Representation.FORMAL:
        match = re.fullmatch(r"E\[(.+?)\] P\[(.*?)\]", text)
        separator, empty = ";", ""
    else:
        match = re.fullmatch(r"Achieve: (.+?)\. Preserve: (.+?)\.", text)
        separator, empty = " and ", "nothing"
    if not match:
        raise ValueError("output does not match the specified representation")
    effects = [
        _parse_predicate(item, representation) for item in match[1].split(separator)
    ]
    preserve = (
        []
        if match[2] == empty
        else [
            _parse_predicate(item, representation) for item in match[2].split(separator)
        ]
    )
    return _canonical({"effects": effects, "preserve": preserve, "instruction": None})
