"""Task-progress examples cannot become query hints or invented temporal labels."""

import copy
import json

import pytest

from carl.icl.responses import text_message
from carl.icl.types import JSONObject
from scripts.goal_probe.components import MechanicsExample, with_progress_examples


@pytest.mark.parametrize("layout_schema", [False, True])
def test_progress_examples_preserve_query_and_calibrated_messages(layout_schema: bool) -> None:
    query = {"state": ["#####", "#@$.#", "#####"], "u": 0.85}
    payload: JSONObject = {"model": "gpt-6-luna", "input": [
        text_message("developer", "Original instructions"),
        text_message("user", json.dumps({"state": query["state"], "u": 0.5})),
        text_message("assistant", '{"subgoals": []}'),
        text_message("user", json.dumps(query)),
    ]}
    before = copy.deepcopy(payload)
    example = MechanicsExample(id="train", state=["#####", "#@$.#", "#####"],
                               subgoal=["#####", "#-@*#", "#####"])
    result = with_progress_examples(payload, [example], layout_schema=layout_schema)
    assert payload == before
    assert result["input"][-3:] == payload["input"][1:]
    demonstration = json.loads(result["input"][2]["content"][0]["text"])
    assert not {"u", "d", "actions", "subgoal"} & demonstration.keys()
    answer = json.loads(result["input"][3]["content"][0]["text"])["subgoals"][0]
    actual = [answer[f"row_{i:02d}"] for i in range(3)] if layout_schema else answer["rows"]
    assert actual == example["subgoal"]
    assert json.loads(result["input"][-1]["content"][0]["text"]) == query
