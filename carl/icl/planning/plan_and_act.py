"""A bounded Plan-and-Act interface for checked Craftax intents."""

import json
from collections.abc import Mapping
from typing import Final

from carl.icl.planning.craftax_tasks import CraftaxIntent, _unique_object, decode_intent, recipe_objectives
from carl.icl.types import JSONValue


PLAN_AND_ACT_VARIANT: Final = "plan-and-act-on-failure"


def decode_plan(value: JSONValue, *, max_steps: int = 8) -> tuple[CraftaxIntent, ...]:
    """Decode one ordered list of typed intents within the planner's output bound."""
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(value, str):
        value = json.loads(value, object_pairs_hook=_unique_object)
    if not isinstance(value, dict) or set(value) != {"intents"}:
        raise ValueError("A Craftax plan must contain exactly one intents list")
    intents = value["intents"]
    if not isinstance(intents, list) or len(intents) > max_steps:
        raise ValueError(f"A Craftax plan must contain at most {max_steps} intents")
    return tuple(decode_intent(item) for item in intents)


def mock_plan(
    observation: Mapping[str, JSONValue], goal: CraftaxIntent, *, max_steps: int = 8,
) -> tuple[CraftaxIntent, ...]:
    """Generate one complete recipe plan and pass the mock output through the same decoder."""
    objectives = recipe_objectives(observation, goal)
    return decode_plan(
        {"intents": [intent.model_dump(mode="json") for intent in objectives]}, max_steps=max_steps,
    )
