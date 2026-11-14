import json

import pytest

from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_in_context import freeze_support
from scripts.prepare_icl_followup import make_plan
from scripts.prepare_icl_followup import validate_hidden_fields


def record(identifier, layout, split, effect_type="box"):
    return {
        "record_id": identifier, "layout_id": layout, "split": split, "effect_type": effect_type,
        "validation": {"accepted": True}, "start_observation": ["#####", "#@$.#", "#####"],
        "later_observation": ["secret-future"], "fragment_actions": [1], "k": 1,
        "task_goal": "Put every box on a goal.",
        "intent": {"effects": [{"predicate": "box_at", "row": 1, "col": 3}],
                   "preserve": [], "instruction": None},
    }


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "test.jsonl"
    rows = [record("t1", "test1", "test"), record("t2", "test2", "test")]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    canonical = tmp_path / "ids.json"
    canonical.write_text(json.dumps({"record_ids": ["t2", "t1"]}))
    enriched = tmp_path / "icl_ids.json"
    enriched.write_text(json.dumps({"record_ids": ["t2", "t1"], "dataset_sha256": file_sha256(data)}))
    training = tmp_path / "train.jsonl"
    training.write_text("".join(json.dumps(r) + "\n" for r in
                                [record("s1", "train1", "train"),
                                 record("s2", "train2", "train", "goal_conjunction")]))
    support = tmp_path / "support.json"
    freeze_support(training, support)
    model = tmp_path / "model"
    model.mkdir()
    (model / "experiment_revision.json").write_text(json.dumps({"model": "Qwen/Qwen3-0.6B", "revision": "rev"}))
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (model / filename).write_text("contents")
    return data, canonical, enriched, training, support, model, "rev", tmp_path / "plan.json"


def test_plan_retains_order_hashes_and_commands(inputs):
    plan = make_plan(*inputs, expected_contexts=2)
    assert plan["test_record_ids"] == ["t2", "t1"]
    assert plan["test_layout_count"] == 2
    assert plan["inputs"][str(inputs[0].resolve())] == file_sha256(inputs[0])
    assert plan["status"] == "prepared_not_evaluated"
    assert "inspect-prompts" in plan["commands"]["inspect_prompts"]
    assert "evaluate" in plan["commands"]["evaluate"]
    assert "--support-data" in plan["commands"]["evaluate"]


def test_rejects_changed_order_and_dataset_hash(inputs):
    enriched = inputs[2]
    enriched.write_text(json.dumps({"record_ids": ["t1", "t2"], "dataset_sha256": file_sha256(inputs[0])}))
    with pytest.raises(ValueError, match="exact frozen order"):
        make_plan(*inputs, expected_contexts=2)
    enriched.write_text(json.dumps({"record_ids": ["t2", "t1"], "dataset_sha256": "wrong"}))
    with pytest.raises(ValueError, match="dataset hash"):
        make_plan(*inputs, expected_contexts=2)


def test_rejects_support_overlap_and_hidden_input_dependency(inputs, monkeypatch):
    data, canonical, enriched = inputs[:3]
    rows = [record("t1", "train1", "test"), record("t2", "test2", "test")]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    enriched.write_text(json.dumps({"record_ids": ["t2", "t1"], "dataset_sha256": file_sha256(data)}))
    with pytest.raises(ValueError, match="layouts overlap"):
        make_plan(*inputs, expected_contexts=2)
    monkeypatch.setattr("scripts.prepare_icl_followup.in_context_messages", lambda query, support: query["later_observation"])
    with pytest.raises(ValueError, match="hidden fields"):
        validate_hidden_fields(rows[0], [])
