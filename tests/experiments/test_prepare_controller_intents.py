"""Only valid generated intents and current states reach the fixed controller."""

import json

import pytest

from scripts.prepare_controller_intents import prepare
from scripts.prepare_controller_intents import verify_generator_config


def make_record(identifier):
    return {"record_id": identifier, "layout_id": identifier, "split": "test",
            "start_observation": ["#######", "#@$-.-#", "#-$-.-#", "#-----#", "#-----#", "#-----#", "#######"],
            "task_goal": "put boxes on goals", "later_observation": "SECRET", "fragment_actions": [99],
            "intent": {"effects": "DEMONSTRATION LABEL"}}


def prediction(identifier, col):
    intent = {"effects": [{"predicate": "box_at", "row": 1, "col": col}], "preserve": [], "instruction": None}
    return {"record_id": identifier, "layout_id": identifier, "variant": "candidate", "representation": "json",
            "raw_output": json.dumps(intent), "syntax_valid": True, "parsed_intent": intent}


def test_only_valid_generated_intent_reaches_controller():
    records = [make_record(key) for key in ("a", "b", "c", "d")]
    malformed = {**prediction("c", 3), "raw_output": "invalid", "syntax_valid": False, "parsed_intent": None}
    inputs, decisions = prepare(records, ["a", "b", "c", "d"],
                                [prediction("a", 3), prediction("b", 2), malformed], "candidate")
    assert [row["status"] for row in decisions] == ["ready", "invalid_intent", "invalid_representation",
                                                     "missing_generation"]
    assert len(inputs) == 1 and inputs[0]["intent"] == prediction("a", 3)["parsed_intent"]
    assert "SECRET" not in json.dumps(inputs) and "DEMONSTRATION LABEL" not in json.dumps(inputs)
    assert "fragment_actions" not in inputs[0]


def test_mismatched_or_duplicate_predictions_fail_closed():
    one = prediction("a", 3)
    with pytest.raises(ValueError, match="provenance"):
        prepare([make_record("a")], ["a"], [one, one], "candidate")
    with pytest.raises(ValueError, match="independent parsing"):
        prepare([make_record("a")], ["a"], [{**one, "parsed_intent": {}}], "candidate")
    with pytest.raises(ValueError, match="parse status"):
        prepare([make_record("a")], ["a"], [{**one, "raw_output": "invalid"}], "candidate")


def test_generator_configuration_must_match_frozen_matrix_and_prompts(tmp_path):
    import hashlib

    prediction_path = tmp_path / "candidate.jsonl"
    prompts = prediction_path.with_suffix(".prompts.jsonl")
    prompts.write_text('{"record_id":"a"}\n')
    plan = {"run": str(tmp_path), "methods": [{"name": "candidate", "representation": "json",
            "support_set": "zero", "output": str(prediction_path), "adapter": None, "adapter_pending": False}],
            "model": {"path": "/model", "revision": "revision", "sha256":
                      {"model.safetensors": "weights", "config.json": "config", "tokenizer.json": "tokenizer"}},
            "inputs": {name: {"sha256": name} for name in
                       ("data", "test_ids", "support_data", "support_manifest", "protocol", "generator", "support_selector")},
            "settings": {"seed": 1729, "dtype": "bfloat16"}}
    metadata = {"variant": "candidate", "representation": "json", "support_set": "zero", "model": "/model",
                "revision": "revision", "model_weights_sha256": "weights", "model_config_sha256": "config",
                "tokenizer_sha256": "tokenizer", "data_sha256": "data", "test_ids_sha256": "test_ids",
                "support_data_sha256": "support_data", "support_manifest_sha256": "support_manifest",
                "protocol_sha256": "protocol", "generator_sha256": "generator",
                "support_selection_sha256": "support_selector", "adapter": None, "adapter_weights_sha256": None,
                "adapter_config_sha256": None, "seed": 1729, "dtype": "bfloat16",
                "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest()}
    verify_generator_config(metadata, plan, "candidate", prediction_path)
    with pytest.raises(ValueError, match="configuration"):
        verify_generator_config({**metadata, "test_ids_sha256": "other"}, plan, "candidate", prediction_path)
    prompts.write_text("changed")
    with pytest.raises(ValueError, match="prompts"):
        verify_generator_config(metadata, plan, "candidate", prediction_path)
