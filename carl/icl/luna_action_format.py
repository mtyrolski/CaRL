"""Responses JSON Schema for one Sokoban primitive action."""

from typing import Any

from carl.icl.luna_intent_rollout import Action


def action_text_format() -> dict[str, Any]:
    """Build a fresh strict action format for the Responses ``text`` field."""
    return {
        "format": {
            "type": "json_schema",
            "name": "sokoban_action_v1",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": [action.value for action in Action]}},
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }
