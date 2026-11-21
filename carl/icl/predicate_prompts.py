"""Shared visible-input formatter for prompted and locally trained generator roles."""

import json
import math
from collections.abc import Sequence
from typing import Final

from carl.icl.types import JSONObject

GENERATOR_PROMPT_VERSION: Final = "sokoban-predicate-generator-v1"
COMMON: Final = (
    "You propose one achievable, non-trivial Sokoban intent using only the current board. "
    "Rows and columns are zero-based. # is wall, - is floor, . is goal, $ is box, "
    "* is box on goal, @ is player and + is player on goal. "
    "An intent has required effects and preservation conditions. All effects must hold "
    "together and each must be false on the current board. Every preservation condition "
    "must be true now and remain true. Boxes are indistinguishable. "
    "The only predicates are box at (row,column), player at (row,column), "
    "and at least a given number of boxes on goals. Return one intent, with no other text. "
)
JSON_INSTRUCTION: Final = (
    'Return exactly a JSON object with keys "effects", "preserve", "instruction". '
    "Both effects and preserve are arrays; effects is nonempty. instruction must be null. "
    'Predicates are {"predicate":"box_at","row":R,"col":C}, '
    '{"predicate":"player_at","row":R,"col":C}, or '
    '{"predicate":"boxes_on_goals_at_least","count":N}.'
)
PREDICATE_GENERATOR_PROMPT: Final = (
    COMMON + JSON_INSTRUCTION
    + " ? is an unseen cell. Each effect must be visibly false and each preservation condition visibly true now. "
    "Use the task goal to choose progress."
)
SCALE_INSTRUCTIONS: Final = {
    "unscaled": "The scale is null. Propose useful progress without a length target.",
    "absolute_d": "Scale is the mean primitive-action count of successful controller attempts.",
    "ratio_d_over_ac": "Scale is the mean successful controller action count divided by the context median.",
    "offset_rank": "Scale ranks demonstration action offsets on a zero-to-one scale.",
    "normalized_offset": "Scale is a demonstration action offset divided by a fixed denominator.",
    "relative_u": "Scale is the controller-relative percentile of successful action counts, weighted by success rate.",
}


def generator_instruction(scale_arm: str, prompt: str = PREDICATE_GENERATOR_PROMPT) -> str:
    """Render the generator instruction with the selected scale definition."""
    if scale_arm not in SCALE_INSTRUCTIONS:
        raise ValueError(f"Unknown predicate scale arm: {scale_arm}")
    return f"{prompt}\n{SCALE_INSTRUCTIONS[scale_arm]}"


def generator_query(
    rows: Sequence[str], task_goal: str, scale: float | None, scale_arm: str, *, allow_missing: bool = False,
) -> JSONObject:
    """Whitelist visible data and allow null in mixed training demonstrations."""
    if not isinstance(task_goal, str) or not task_goal.strip():
        raise ValueError("Task goal must be nonempty")
    if scale_arm == "unscaled":
        if scale is not None:
            raise ValueError("Unscaled generation requires a null scale")
    elif scale is not None or not allow_missing:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid requested scale for {scale_arm}")
        if scale_arm in ("relative_u", "offset_rank") and scale >= 1:
            raise ValueError(f"Requested {scale_arm} must be strictly between zero and one")
    return {"observation": list(rows), "task_goal": task_goal, "scale": scale}


def generator_messages(
    rows: Sequence[str], task_goal: str, scale: float | None, scale_arm: str, *, allow_missing: bool = False,
) -> list[dict[str, str]]:
    """Render the same user content for the local Qwen chat template."""
    query = generator_query(rows, task_goal, scale, scale_arm, allow_missing=allow_missing)
    return [
        {"role": "system", "content": generator_instruction(scale_arm)},
        {"role": "user", "content": json.dumps(query, sort_keys=True, ensure_ascii=False,
                                             separators=(",", ":"), allow_nan=False)},
    ]
