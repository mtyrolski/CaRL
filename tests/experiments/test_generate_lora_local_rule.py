"""A local-rule baseline sees the current board and no future demonstration."""

import json

from carl.icl.intent_assessment import check_raw_prediction
from carl.icl.intent_data import validate_prediction
from scripts import generate_lora_local_rule as generator


def test_rule_receives_only_current_board_and_records_provenance(tmp_path, monkeypatch):
    board = ["#####", "#@$.#", "#####"]
    record = {"record_id": "a", "layout_id": "puzzle", "split": "test", "start_observation": board,
              "later_observation": ["SECRET"], "fragment_actions": [99], "task_goal": "secret task"}
    data, ids, output = tmp_path / "data.jsonl", tmp_path / "ids.json", tmp_path / "rule.jsonl"
    data.write_text(json.dumps(record) + "\n")
    ids.write_text(json.dumps({"record_ids": ["a"]}))
    original = generator.local_rule
    received = []

    def capture(rows):
        received.append(rows)
        return original(rows)

    monkeypatch.setattr(generator, "local_rule", capture)
    summary = generator.generate(data, ids, output)
    row = json.loads(output.read_text())
    assert received == [board]
    assert row["method"] == "local_rule" and row["generated_tokens"] is None
    assert not validate_prediction(row["parsed"], board)
    check_raw_prediction(row)
    assert summary["input_fields"] == ["start_observation"] and summary["model"] is None
    assert generator.generate(data, ids, output) == summary


def test_no_available_intent_is_not_invented(tmp_path):
    record = {"record_id": "a", "layout_id": "puzzle", "split": "test", "start_observation": ["###", "#@#", "###"]}
    data, ids, output = tmp_path / "data.jsonl", tmp_path / "ids.json", tmp_path / "rule.jsonl"
    data.write_text(json.dumps(record) + "\n")
    ids.write_text(json.dumps({"record_ids": ["a"]}))
    generator.generate(data, ids, output)
    row = json.loads(output.read_text())
    assert row["parsed"] is None and row["raw_output"] == "" and row["json_valid"] is False
    check_raw_prediction(row)
