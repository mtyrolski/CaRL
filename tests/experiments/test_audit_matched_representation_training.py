"""Audit matched training data, masks, samplers and validation checkpoints."""

import json
from itertools import islice

import pytest

from carl.icl.intent_generator_experiment import board_balanced
from scripts.audit_matched_representation_training import COMMON_FIELDS
from scripts.audit_matched_representation_training import DATA
from scripts.audit_matched_representation_training import EXPECTED
from scripts.audit_matched_representation_training import compare_common
from scripts.audit_matched_representation_training import digest_records
from scripts.audit_matched_representation_training import inspect_checkpoint
from scripts.audit_matched_representation_training import training_inputs
from scripts.audit_matched_representation_training import verify_command
from scripts.audit_matched_representation_training import verify_curve
from scripts.audit_matched_representation_training import verify_response_mask


def record(identifier, layout, split):
    return {"record_id": identifier, "layout_id": layout, "split": split,
            "start_observation": ["#####", "#@$.#", "#####", "#####", "#####"],
            "task_goal": "Put boxes on goals.", "intent": {"effects": [
                {"predicate": "box_at", "row": 1, "col": 3}], "preserve": [], "instruction": None},
            "validation": {"accepted": True}, "effect_type": "box"}


def test_saved_records_are_rederived_and_test_layout_cannot_leak():
    train, valid = record("train", "layout-a", "train"), record("valid", "layout-b", "validation")
    test = record("test", "layout-c", "test")
    config = {"train_record_ids": ["train"], "validation_record_ids": ["valid"],
              "train_layouts": ["layout-a"], "validation_layouts": ["layout-b"],
              "training_records_sha256": digest_records([train]), "validation_records_sha256": digest_records([valid]),
              "command": ["train", "--validation-boards", "1", "--validation-contexts-per-board", "4"]}
    assert training_inputs(config, [train, valid, test], [test]) == ([train], [valid])
    with pytest.raises(ValueError, match="overlaps"):
        training_inputs(config, [train, valid, test], [{**test, "layout_id": "layout-a"}])
    with pytest.raises(ValueError, match="digest"):
        training_inputs({**config, "training_records_sha256": "wrong"}, [train, valid, test], [test])
    with pytest.raises(ValueError, match="order differs"):
        training_inputs({**config, "train_record_ids": ["test"]}, [train, valid, test], [test])


def test_reconstructed_board_balanced_order_uses_seed_and_train_only():
    rows = [record(str(index), f"layout-{index % 3}", "train") for index in range(12)]

    def sample(seed):
        return [row["record_id"] for row in islice(board_balanced(rows, seed), 400)]

    assert sample(1729) == sample(1729)
    assert sample(1729) != sample(2718)
    assert set(sample(1729)) <= {row["record_id"] for row in rows}


def test_comparison_detects_one_different_hyperparameter():
    base = {key: None for key in COMMON_FIELDS}
    base.update(EXPECTED)
    configs = {name: {**base, "representation": name} for name in ("json", "formal", "natural")}
    assert compare_common(configs, 1729)["accumulation"] == 4
    configs["formal"]["learning_rate"] = 0.01
    with pytest.raises(ValueError, match="differ"):
        compare_common(configs, 1729)


def test_recorded_launch_must_match_configuration_and_paths(tmp_path):
    config = {**EXPECTED, "model": str(tmp_path / "model"), "revision": "revision", "representation": "json"}
    config["command"] = ["trainer.py", "train", "--data", str(DATA), "--model", config["model"],
                         "--output", str(tmp_path / "adapter"), "--representation", "json", "--revision", "revision",
                         "--dtype", "bfloat16", "--seed", "1729", "--accumulation", "4", "--lr", "0.0002",
                         "--max-length", "512", "--max-new-tokens", "128", "--max-steps", "100", "--eval-every", "20",
                         "--gradient-checkpointing", "--require-cuda"]
    verify_command(config, tmp_path / "adapter")
    changed = {**config, "learning_rate": 0.01}
    with pytest.raises(ValueError, match="--lr"):
        verify_command(changed, tmp_path / "adapter")


def test_response_only_mask_includes_eos_without_future_fields():
    class Tokenizer:
        eos_token = "<EOS>"

        def apply_chat_template(self, messages, **kwargs):
            assert all("hidden-future" not in item["content"] for item in messages)
            return [10, 11, 12]

        def encode(self, text, **kwargs):
            assert text.endswith("<EOS>")
            return [20, 21]

    for representation in ("json", "formal", "natural"):
        result = verify_response_mask(Tokenizer(), record("x", "a", "train"), representation, 64)
        assert result == {"prompt_tokens_masked": 3, "response_tokens_supervised": 2, "total_tokens": 5}


def test_complete_curve_requires_validation_and_true_best():
    curve = [{"step": step, **({"validation_loss": 1 / step} if step % 20 == 0 else {})}
             for step in range(1, 101)]
    status = {"status": "finished", "completed_steps": 100, "last_validation_step": 100}
    verify_curve(status, {"step": 100, "validation_loss": .01}, curve)
    with pytest.raises(ValueError, match="minimum"):
        verify_curve(status, {"step": 80, "validation_loss": .0125}, curve)
    with pytest.raises(ValueError, match="cadence"):
        verify_curve(status, {"step": 100, "validation_loss": .01},
                     [{"step": 40} if row["step"] == 40 else row for row in curve])


def test_checkpoint_hashes_and_lora_settings_are_checked(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_bytes(b"abc")
    config = {"rank": 4, "lora_alpha": 8, "lora_targets": ["q_proj", "v_proj"]}
    settings = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0,
                "bias": "none", "task_type": "CAUSAL_LM"}
    (tmp_path / "adapter_config.json").write_text(json.dumps(settings))
    first = inspect_checkpoint(tmp_path, config)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"abcd")
    assert first["weights_sha256"] != inspect_checkpoint(tmp_path, config)["weights_sha256"]
    (tmp_path / "adapter_config.json").write_text(json.dumps({**settings, "r": 8}))
    with pytest.raises(ValueError, match="LoRA settings"):
        inspect_checkpoint(tmp_path, config)
