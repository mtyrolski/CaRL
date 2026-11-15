"""Checks for independent reconstruction of sampled training updates."""

import json
from pathlib import Path

import pytest

from scripts.audit_overnight_stage04_training import SYMMETRIES
from scripts.audit_overnight_stage04_training import audit_arm
from scripts.audit_overnight_stage04_training import independent_transform
from scripts.audit_overnight_stage04_training import recompute_tokens
from scripts.audit_overnight_stage04_training import sha


class CharTokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, **_kwargs):
        return [ord(char) for char in json.dumps(messages)]

    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]


def record(identifier="a", layout="layout", split="train"):
    return {"record_id": identifier, "layout_id": layout, "split": split, "effect_type": "box",
            "task_goal": "Put every box on a goal.",
            "start_observation": ["#####", "#@--#", "#-$-#", "#.--#", "#####"],
            "intent": {"effects": [{"predicate": "box_at", "row": 3, "col": 1}],
                       "preserve": [], "instruction": None},
            "later_observation": ["HIDDEN-FUTURE"], "fragment_actions": ["HIDDEN-ACTIONS"]}


def test_independent_transforms_map_structured_coordinates_and_reject_prose():
    row = record()
    expected = ((3, 1), (3, 3), (1, 3), (1, 1), (3, 3), (1, 3), (1, 1), (3, 1))
    for symmetry, position in zip(SYMMETRIES, expected, strict=True):
        rows, intent = independent_transform(row, symmetry)
        assert (intent["effects"][0]["row"], intent["effects"][0]["col"]) == position
        assert rows[position[0]][position[1]] == "."
    invalid = record()
    invalid["intent"]["instruction"] = "move left"
    with pytest.raises(ValueError, match="free-text"):
        independent_transform(invalid, "rotate_90")


def test_independent_tokens_ignore_recorded_future_and_enforce_length():
    row = record()
    tokenizer = CharTokenizer()
    length, answer = recompute_tokens(tokenizer, "SYSTEM", row, "identity", 5000)
    changed = dict(row, later_observation=["DIFFERENT"], fragment_actions=[1, 2, 3])
    assert recompute_tokens(tokenizer, "SYSTEM", changed, "identity", 5000) == (length, answer)
    with pytest.raises(ValueError, match="exceeds"):
        recompute_tokens(tokenizer, "SYSTEM", row, "identity", length - 1)


def test_audit_arm_checks_token_counts_and_first_validation_argmin(tmp_path: Path):
    directory = tmp_path / "adapter_wide"
    checkpoint = directory / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "dataset.jsonl"
    train, valid = record(), record("v", "val-layout", "validation")
    data.write_text("\n".join(json.dumps(item) for item in (train, valid)) + "\n")
    freeze = {"seed": 42, "rank": 4, "accumulation": 1, "learning_rate": .0002, "max_length": 5000,
              "padding_length": 1000, "max_steps": 2, "eval_every": 1, "dtype": "bfloat16",
              "gradient_checkpointing": True, "revision": "rev", "validation_boards": 1,
              "validation_contexts_per_board": 1, "training_stop_monotonic": 123.0, "device": "cuda",
              "lora_alpha": 8, "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"],
              "model_tokens_per_step": 1000, "datasets": {"wide": str(data)}, "data_sha256": {"wide": sha(data)}}
    args = {"model": str(model), "revision": "rev", "dtype": "bfloat16", "threads": 4, "seed": 42,
            "rank": 4, "accumulation": 1, "lr": .0002, "max_length": 5000, "max_new_tokens": 128,
            "gradient_checkpointing": True, "pad_to_length": 1000, "max_steps": 2, "eval_every": 1,
            "validation_boards": 1, "validation_contexts_per_board": 1, "stop_monotonic": 123.0,
            "augmentation": "none", "data": str(data)}
    config = {"args": args, "device": "cuda", "enable_thinking": False, "data_sha256": sha(data),
              "validation_record_ids": ["v"], "validation_layouts": ["val-layout"]}
    status = {"status": "finished", "completed_steps": 2, "best_adapter": str(checkpoint)}
    best = {"step": 1, "validation_loss": .2, "adapter": str(checkpoint)}
    (directory / "configuration.json").write_text(json.dumps(config))
    (directory / "status.json").write_text(json.dumps(status))
    (directory / "best.json").write_text(json.dumps(best))
    (checkpoint / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"], "base_model_name_or_path": str(model)}))
    (checkpoint / "adapter_model.safetensors").write_bytes(b"test")
    effective, supervised = recompute_tokens(CharTokenizer(), "SYSTEM", train, "identity", 5000)
    sample = {"record_id": "a", "layout_id": "layout", "symmetry": "identity"}
    curve = [{"step": step, "model_tokens": 1000, "effective_tokens": effective, "supervised_tokens": supervised,
              "samples": [sample], "validation_loss": loss} for step, loss in ((1, .2), (2, .3))]
    curve_path = directory / "training_curve.jsonl"
    curve_path.write_text("\n".join(json.dumps(item) for item in curve) + "\n")
    result = audit_arm("wide", directory, freeze, CharTokenizer(), "SYSTEM")
    assert result["status"] == "pass" and result["samples"] == 2
    assert result["sampled_effect_types"] == {"box": 2}
    assert result["selected_step"] == 1 and result["model_tokens_per_update"]["total"] == 2000
    curve[1]["supervised_tokens"] += 1
    curve_path.write_text("\n".join(json.dumps(item) for item in curve) + "\n")
    result = audit_arm("wide", directory, freeze, CharTokenizer(), "SYSTEM")
    assert any("token counts" in error for error in result["errors"])
    best["validation_loss"] = .3
    (directory / "best.json").write_text(json.dumps(best))
    result = audit_arm("wide", directory, freeze, CharTokenizer(), "SYSTEM")
    assert any("first validation argmin" in error for error in result["errors"])
