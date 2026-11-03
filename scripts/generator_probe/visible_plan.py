"""Isolated visible-action experiment; never execute plans or construct boards.

The caller supplies the exact selected, verified training Examples. This changes
both output format and demonstration content, so it is a new experimental factor.
No production generator or controller uses this helper automatically.
"""

import copy
import json
from collections.abc import Sequence

from carl.icl.types import Example, JSONObject

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


def _text(message: JSONObject) -> JSONObject:
    content = message.get("content")
    if (not isinstance(content, list) or len(content) != 1
            or content[0].get("type") not in ("input_text", "output_text")):
        raise ValueError("Expected one text block per message")
    return content[0]


def _rows(candidate: JSONObject) -> list[str]:
    if "rows" in candidate:
        return candidate["rows"]
    keys = [f"row_{index:02d}" for index in range(len(candidate))]
    if set(keys) != set(candidate):
        raise ValueError("Expected a complete board row object")
    return [candidate[key] for key in keys]


def visible_plan_payload(
    payload: JSONObject, selected_training_examples: Sequence[Example], *, action_cap: int,
) -> JSONObject:
    """Copy a one-candidate Responses payload and expose training witness actions.

    Check demonstration identity before adding witnesses; retain every original
    board and scale label verbatim. Witness replay belongs to training-data
    validation, not this transformation. Query inputs gain no action information.
    """
    if not isinstance(action_cap, int) or isinstance(action_cap, bool) or action_cap < 1:
        raise ValueError("action_cap must be a positive integer")
    result = copy.deepcopy(payload)
    messages = result["input"]
    roles = [message.get("role") for message in messages]
    expected_roles = ["developer", *[role for _ in selected_training_examples
                                    for role in ("user", "assistant")], "user"]
    if roles != expected_roles:
        raise ValueError("Payload must contain exactly the selected training pairs and query")
    schema = result["text"]["format"]["schema"]
    if set(schema["properties"]) != {"subgoals"}:
        raise ValueError("Expected the original subgoals-only output schema")
    subgoals = schema["properties"]["subgoals"]
    if subgoals.get("maxItems") != 1:
        raise ValueError("Visible plans support exactly one candidate")
    for index, example in enumerate(selected_training_examples):
        question = json.loads(_text(messages[2 * index + 1])["text"])
        answer_block = _text(messages[2 * index + 2])
        answer = json.loads(answer_block["text"])
        if (question.get("state") != example["state"] or question.get("num_subgoals") != 1
                or len(answer.get("subgoals", [])) != 1
                or _rows(answer["subgoals"][0]) != example["subgoal"]):
            raise ValueError("Selected training example does not match its prompt pair")
        actions = example["actions"]
        if (not 1 <= len(actions) <= action_cap or len(actions) != example["k"]
                or any(not isinstance(action, int) or isinstance(action, bool) or action not in range(4)
                       for action in actions)):
            raise ValueError("Training witness must contain k primitive actions within the cap")
        answer_block["text"] = json.dumps({
            "planned_actions": [ACTIONS[action] for action in actions], **answer,
        })
    query = json.loads(_text(messages[-1])["text"])
    if query.get("num_subgoals") != 1:
        raise ValueError("Query must request one candidate")
    subgoals["minItems"] = 1
    schema["properties"] = {
        "planned_actions": {
            "type": "array", "minItems": 1, "maxItems": action_cap,
            "items": {"type": "string", "enum": list(ACTIONS)},
        },
        "subgoals": subgoals,
    }
    schema["required"] = ["planned_actions", "subgoals"]
    developer = _text(messages[0])
    developer["text"] = developer["text"].replace(
        "keep the internal plan out of the output", "include the short action plan in planned_actions"
    )
    developer["text"] += (
        "\nVisible-plan experiment: first emit planned_actions, a short sequence of "
        f"1 to {action_cap} primitive UP, DOWN, LEFT or RIGHT actions, then emit exactly "
        "one complete endpoint board in subgoals. The demonstrations' planned_actions "
        "are actual verified training witness actions leading to their displayed boards. "
        "Choose the endpoint using the requested scale and the examples; u is not an "
        "action count. Trace your proposed actions from the query state and write the "
        "resulting complete board, updating both the vacated and entered cells at every "
        "move or push. Count exactly one player and the original number of boxes. "
        "The endpoint must differ from the input. The action list is explanatory output; "
        "the separate frozen controller still receives only your complete endpoint board."
    )
    return result


def recorded_plan(output_text: str) -> list[str] | None:
    """Read a model plan for post-hoc records only; never infer or repair an endpoint."""
    try:
        decoded = json.loads(output_text)
        actions = decoded.get("planned_actions")
    except (ValueError, AttributeError):
        return None
    if (not isinstance(actions, list) or not actions
            or any(not isinstance(action, str) or action not in ACTIONS for action in actions)):
        return None
    return list(actions)
