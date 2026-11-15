"""Prompt construction and checked support selection for intent representations."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_data import SokobanPredicate
from carl.icl.intent_in_context import select_support
from carl.icl.intents.generation.data import DataSplit
from carl.icl.intents.generation.data import file_sha256
from carl.icl.intents.generation.data import prompt_messages
from carl.icl.intents.generation.data import read_records
from carl.icl.intents.representation.formats import Representation
from carl.icl.intents.representation.formats import _canonical
from carl.icl.intents.representation.formats import render_representation


class SupportSet(StrEnum):
    ZERO = "zero"
    LEGACY = "legacy"
    BALANCED0 = "balanced0"
    BALANCED1 = "balanced1"
    BALANCED2 = "balanced2"


COMMON = (
    "You propose one achievable, non-trivial Sokoban intent using only the current board. "
    "Rows and columns are zero-based. # is wall, - is floor, . is goal, $ is box, "
    "* is box on goal, @ is player and + is player on goal. "
    "An intent has required effects and preservation conditions. All effects must hold "
    "together and each must be false on the current board. Every preservation condition "
    "must be true now and remain true. Boxes are indistinguishable. "
    "The only predicates are box at (row,column), player at (row,column), "
    "and at least a given number of boxes on goals. Return one intent, with no other text. "
)

REPRESENTATION_INSTRUCTIONS = {
    Representation.JSON: 'Return exactly a JSON object with keys "effects", "preserve", "instruction". '
    "Both effects and preserve are arrays; effects is nonempty. instruction must be null. "
    'Predicates are {"predicate":"box_at","row":R,"col":C}, '
    '{"predicate":"player_at","row":R,"col":C}, or '
    '{"predicate":"boxes_on_goals_at_least","count":N}.',
    Representation.FORMAL: "Return exactly E[effects] P[preserve]. Separate predicates in each list with semicolons; "
    "the preserve list may be empty. Write predicates only as box(R,C), player(R,C), or goals(N). "
    "For example E[box(2,3);goals(2)] P[] requires both effects and preserves nothing.",
    Representation.NATURAL: "Return exactly 'Achieve: effects. Preserve: conditions.' Join predicates with ' and '. "
    "If there are no preservation conditions write 'nothing'. "
    "Write predicates only as 'box at row R column C', 'player at row R column C', or "
    "'at least N boxes on goals'. Example: Achieve: box at row 2 column 3. Preserve: nothing.",
}


def _whole_task(row: dict[str, Any]) -> bool:
    board = row["start_observation"]
    goals = sum(line.count(".") + line.count("*") + line.count("+") for line in board)
    return any(
        item["predicate"] == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST
        and item["count"] >= goals
        for item in row["intent"]["effects"]
    )


def support_bank(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Prespecified train-only support selection; independent of test inputs."""
    candidates = sorted(
        (
            row
            for row in records
            if row.get("split") == DataSplit.TRAIN
            and row.get("validation", {}).get("accepted") is True
            and row.get("intent", {}).get("instruction") is None
        ),
        key=lambda row: (row["layout_id"], row["record_id"]),
    )
    selected = {
        SupportSet.ZERO.value: [],
        SupportSet.LEGACY.value: select_support(records),
    }
    used_layouts: set[str] = set()
    for name in (SupportSet.BALANCED0, SupportSet.BALANCED1, SupportSet.BALANCED2):
        pair = []
        for effect_type in ("box", "goal_conjunction"):
            match = next(
                (
                    row
                    for row in candidates
                    if row.get("effect_type") == effect_type
                    and row["layout_id"] not in used_layouts
                    and not _whole_task(row)
                ),
                None,
            )
            if match is None:
                raise ValueError(
                    f"not enough distinct training layouts for {name} {effect_type}"
                )
            _canonical(match["intent"])
            pair.append(match)
            used_layouts.add(match["layout_id"])
        selected[name.value] = pair
    return selected


def support_manifest(data: Path) -> dict[str, Any]:
    bank = support_bank(read_records(data))
    return {
        "schema_version": 1,
        "support_data_sha256": file_sha256(data),
        "selection": "legacy = existing deterministic support; balanced sets = sorted checked TRAIN "
        "box and goal_conjunction examples on six distinct layouts, excluding full-goal labels; "
        "zero = no examples. No test records or model outputs were used.",
        "caveat": "Balanced support variants were conceived after historical ICL results and are exploratory.",
        "support_sets": {
            name: [
                {
                    "record_id": row["record_id"],
                    "layout_id": row["layout_id"],
                    "effect_type": row.get("effect_type"),
                    "whole_task": _whole_task(row),
                }
                for row in group
            ]
            for name, group in bank.items()
        },
    }


def load_supports(
    data: Path, manifest_path: Path, name: SupportSet
) -> list[dict[str, Any]]:
    expected = support_manifest(data)
    if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
        raise ValueError(
            "support manifest differs from the prespecified train-only selection"
        )
    return support_bank(read_records(data))[name.value]


def messages_for(
    record: dict[str, Any],
    support: list[dict[str, Any]],
    representation: Representation | str,
) -> list[dict[str, str]]:
    representation = Representation(representation)
    query = prompt_messages(record)
    messages = [
        {
            "role": "system",
            "content": COMMON + REPRESENTATION_INSTRUCTIONS[representation],
        }
    ]
    for example in support:
        if (
            example.get("split") != DataSplit.TRAIN
            or example.get("validation", {}).get("accepted") is not True
            or example["layout_id"] == record["layout_id"]
        ):
            raise ValueError(
                "support must be checked TRAIN data from a distinct layout"
            )
        messages.extend(
            [
                prompt_messages(example)[1],
                {
                    "role": "assistant",
                    "content": render_representation(representation, example["intent"]),
                },
            ]
        )
    messages.append(query[1])
    return messages
