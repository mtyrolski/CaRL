"""A local-rule baseline sees the current board and no future demonstration."""

import hashlib
import json
from pathlib import Path

import pytest

from carl.icl.intent_assessment import check_raw_prediction
from carl.icl.intent_data import validate_prediction
from carl.icl.intents.assessment import search
from carl.icl.intents.generation import data as generation_data
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
    assert summary["algorithm_components_sha256"]["intents/assessment/search.py"] == hashlib.sha256(
        Path(search.__file__).read_bytes()).hexdigest()
    assert summary["input_pipeline_components_sha256"]["intents/generation/data.py"] == hashlib.sha256(
        Path(generation_data.__file__).read_bytes()).hexdigest()

    legacy = {key: value for key, value in summary.items()
              if key not in {"algorithm_components_sha256", "input_pipeline_components_sha256"}}
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(legacy, ensure_ascii=False, indent=2) + "\n")
    old_output, old_summary = output.read_bytes(), summary_path.read_bytes()
    with pytest.raises(FileExistsError, match="code or provenance may have changed"):
        generator.generate(data, ids, output)
    assert output.read_bytes() == old_output
    assert summary_path.read_bytes() == old_summary


def test_no_available_intent_is_not_invented(tmp_path):
    record = {"record_id": "a", "layout_id": "puzzle", "split": "test", "start_observation": ["###", "#@#", "###"]}
    data, ids, output = tmp_path / "data.jsonl", tmp_path / "ids.json", tmp_path / "rule.jsonl"
    data.write_text(json.dumps(record) + "\n")
    ids.write_text(json.dumps({"record_ids": ["a"]}))
    generator.generate(data, ids, output)
    row = json.loads(output.read_text())
    assert row["parsed"] is None and row["raw_output"] == "" and row["json_valid"] is False
    check_raw_prediction(row)
