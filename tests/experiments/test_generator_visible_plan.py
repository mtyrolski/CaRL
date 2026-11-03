"""Visible plans preserve full-board output and keep query witnesses absent."""

import copy
import json

import pytest

from carl.icl.types import Example, JSONObject
from scripts.generator_probe.visible_plan import recorded_plan, visible_plan_payload


def fixture_payload(layout: bool = True) -> tuple[JSONObject, list[Example]]:
    source = ["#######", "#@--$.#", "#######"]
    target = ["#######", "#-@-$.#", "#######"]
    query = ["#######", "#@-$-.#", "#######"]
    example: Example = {"state": source, "subgoal": target, "actions": [3],
                        "k": 1, "context": "train_only", "u": 0.25}
    candidate = ({f"row_{index:02d}": row for index, row in enumerate(target)}
                 if layout else {"rows": target})

    def message(role: str, text: str) -> JSONObject:
        return {"role": role, "content": [{"type": "output_text" if role == "assistant"
                                            else "input_text", "text": text}]}

    payload = {
        "model": "unchanged-model", "reasoning": {"effort": "low"},
        "text": {"format": {"type": "json_schema", "strict": True, "schema": {
            "type": "object", "additionalProperties": False, "required": ["subgoals"],
            "properties": {"subgoals": {"type": "array", "maxItems": 1,
                                           "items": {"type": "object"}}},
        }}},
        "input": [
            message("developer", "Keep all walls; keep the internal plan out of the output."),
            message("user", json.dumps({"state": source, "u": 0.25, "num_subgoals": 1})),
            message("assistant", json.dumps({"subgoals": [candidate]})),
            message("user", json.dumps({"state": query, "u": 0.85, "num_subgoals": 1})),
        ],
    }
    return payload, [example]


@pytest.mark.parametrize("layout", [True, False])
def test_preserves_boards_labels_query_and_configuration(layout: bool) -> None:
    payload, examples = fixture_payload(layout)
    original = copy.deepcopy(payload)
    transformed = visible_plan_payload(payload, examples, action_cap=8)
    assert payload == original
    assert transformed["input"][-1] == original["input"][-1]
    assert transformed["input"][1] == original["input"][1]
    assert transformed["model"] == original["model"]
    assert transformed["reasoning"] == original["reasoning"]
    answer = json.loads(transformed["input"][2]["content"][0]["text"])
    old_answer = json.loads(original["input"][2]["content"][0]["text"])
    assert list(answer) == ["planned_actions", "subgoals"]
    assert answer["planned_actions"] == ["RIGHT"]
    assert answer["subgoals"] == old_answer["subgoals"]
    schema = transformed["text"]["format"]["schema"]
    assert list(schema["properties"]) == ["planned_actions", "subgoals"]
    assert schema["required"] == ["planned_actions", "subgoals"]
    assert schema["properties"]["subgoals"]["minItems"] == 1
    assert schema["properties"]["subgoals"]["maxItems"] == 1
    assert schema["properties"]["subgoals"]["items"] == original["text"]["format"]["schema"]["properties"]["subgoals"]["items"]
    assert "keep the internal plan out" not in transformed["input"][0]["content"][0]["text"]


@pytest.mark.parametrize("field,value", [
    ("state", ["wrong source"]), ("subgoal", ["wrong endpoint"]),
    ("actions", [4]), ("actions", [True]), ("actions", []), ("k", 2),
])
def test_rejects_mismatched_example_or_invalid_witness(field: str, value: object) -> None:
    payload, examples = fixture_payload()
    examples[0][field] = value
    with pytest.raises(ValueError):
        visible_plan_payload(payload, examples, action_cap=8)


def test_rejects_multiple_candidates_or_missing_examples() -> None:
    payload, examples = fixture_payload()
    with pytest.raises(ValueError, match="selected training"):
        visible_plan_payload(payload, [], action_cap=8)
    payload["text"]["format"]["schema"]["properties"]["subgoals"]["maxItems"] = 2
    with pytest.raises(ValueError, match="one candidate"):
        visible_plan_payload(payload, examples, action_cap=8)


def test_recording_does_not_construct_or_repair_a_board() -> None:
    raw = json.dumps({"planned_actions": ["RIGHT"], "subgoals": [{"rows": ["invalid"]}]})
    assert recorded_plan(raw) == ["RIGHT"]
    assert json.loads(raw)["subgoals"] == [{"rows": ["invalid"]}]
    for invalid in ("broken", "[]", "null", '{}', '{"planned_actions": []}',
                    '{"planned_actions": ["PUSH"]}', '{"planned_actions": [0]}'):
        assert recorded_plan(invalid) is None
