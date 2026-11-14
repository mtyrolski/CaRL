"""Frozen transfer IDs must be compatible with the ICL input contract."""

import json
from types import SimpleNamespace

import pytest

from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_in_context import freeze_support
from carl.icl.intent_in_context import prepare as prepare_icl
from scripts.prepare_lora_icl_ids import prepare


def record(identifier, layout, split, effect_type):
    return {"record_id": identifier, "layout_id": layout, "split": split, "effect_type": effect_type,
            "validation": {"accepted": True}, "start_observation": ["#####", "#@$.#", "#####"],
            "task_goal": "Put every box on a goal.",
            "intent": {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": [], "instruction": None},
            "later_observation": ["SECRET_FUTURE_" + identifier]}


class Tokenizer:
    chat_template = "available"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt and enable_thinking is False
        rendered = json.dumps(messages)
        return list(rendered) if tokenize else rendered


def test_prepared_ids_work_with_icl_and_preserve_order(tmp_path, monkeypatch):
    support_data = tmp_path / "support_data.jsonl"
    support_rows = [record("box", "train-a", "train", "box"), record("goal", "train-b", "train", "goal_conjunction")]
    support_data.write_text("".join(json.dumps(row) + "\n" for row in support_rows))
    support = tmp_path / "support.json"
    freeze_support(support_data, support)
    test_data = tmp_path / "test_data.jsonl"
    test_rows = [record("x4", "test-a", "test", "box"), record("x10", "test-b", "test", "box")]
    test_data.write_text("".join(json.dumps(row) + "\n" for row in test_rows))
    original = tmp_path / "original_ids.json"
    original.write_text(json.dumps({"record_ids": ["x10", "x4"]}) + "\n")
    before = original.read_bytes()
    output = tmp_path / "icl_test_ids.json"
    manifest = prepare(test_data, original, output)
    ids = json.loads(output.read_text())
    assert original.read_bytes() == before
    assert ids["dataset_sha256"] == file_sha256(test_data)
    assert ids["record_ids"] == ["x10", "x4"]
    assert manifest["icl_ids_sha256"] == file_sha256(output)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: Tokenizer())
    args = SimpleNamespace(data=str(test_data), support_data=str(support_data), support=str(support),
                           test_ids=str(output), model="not-loaded", max_length=10000, max_new_tokens=128,
                           output=str(tmp_path / "predictions.jsonl"))
    selected, _, prompts, _ = prepare_icl(args)
    assert [row["record_id"] for row in selected] == ["x10", "x4"]
    assert all("SECRET_FUTURE" not in json.dumps(prompt["messages"]) for prompt in prompts)


def test_preparation_rejects_missing_or_duplicate_ids(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(record("x", "a", "test", "box")) + "\n")
    ids = tmp_path / "ids.json"
    output = tmp_path / "icl_ids.json"
    ids.write_text(json.dumps({"record_ids": ["missing"]}))
    with pytest.raises(ValueError, match="absent"):
        prepare(data, ids, output)
    ids.write_text(json.dumps({"record_ids": ["x", "x"]}))
    with pytest.raises(ValueError, match="unique"):
        prepare(data, ids, output)
    assert not output.exists()
