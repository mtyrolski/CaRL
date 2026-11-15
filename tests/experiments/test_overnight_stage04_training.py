"""Model-free tests for matched board augmentation and token budgets."""

import json
from argparse import Namespace
from itertools import islice

import numpy as np
import pytest
import torch

from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_augmentation import SYMMETRIES
from carl.icl.intent_augmentation import transform_actions
from carl.icl.intent_augmentation import transform_coordinate
from carl.icl.intent_augmentation import transform_training_record
from carl.icl.intent_data import validate_intent
from carl.icl.intent_generator_experiment import augmented_training_samples
from carl.icl.intent_generator_experiment import encode_example
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import tensor_example
from carl.icl.intent_generator_experiment import training_lengths
from scripts.inspect_overnight_stage04_tokens import inspect


def record(layout="one", identifier="record"):
    return {"record_id": identifier, "layout_id": layout, "split": "train",
            "task_goal": "Put every box on a goal.",
            "start_observation": ["#######", "#*----#", "#-----#", "#-----#", "#-@$.-#", "#-----#", "#######"],
            "intent": {"effects": [{"predicate": "box_at", "row": 4, "col": 4},
                                   {"predicate": "boxes_on_goals_at_least", "count": 2}],
                       "preserve": [{"predicate": "box_at", "row": 1, "col": 1}], "instruction": None},
            "later_observation": ["SECRET FUTURE"], "fragment_actions": ["SECRET ACTION"], "k": 999}


class CharTokenizer:
    eos_token = "!"
    pad_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return [ord(char) for char in json.dumps(messages)]

    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]


def test_all_eight_symmetries_replay_with_exact_conjunction_and_preservation():
    original = record()
    start = from_rows(original["start_observation"])
    later = np.asarray(environment(start).next_state(start, 3))
    assert not validate_intent(original["intent"], [to_rows(start), to_rows(later)])
    for symmetry in SYMMETRIES:
        transformed = transform_training_record(original, symmetry)
        mapped = transform_actions([3], symmetry)[0]
        state = from_rows(transformed["start_observation"])
        endpoint = np.asarray(environment(state).next_state(state, mapped))
        assert not validate_intent(transformed["intent"], [to_rows(state), to_rows(endpoint)])
        goal = transformed["intent"]["effects"][0]
        assert (goal["row"], goal["col"]) == transform_coordinate(4, 4, 7, symmetry)
        preserved = transformed["intent"]["preserve"][0]
        assert (preserved["row"], preserved["col"]) == transform_coordinate(1, 1, 7, symmetry)
        assert "SECRET" not in json.dumps(transformed)
        assert "SECRET" not in json.dumps(prompt_messages(transformed))


def test_b_and_c_sample_same_source_records_with_independent_symmetry_rng():
    rows = [record("one", "a"), record("one", "b"), record("two", "c")]
    unaugmented = list(islice(augmented_training_samples(rows, 1729, "none"), 80))
    augmented = list(islice(augmented_training_samples(rows, 1729, "d4"), 80))
    assert [row[0]["record_id"] for row in unaugmented] == [row[0]["record_id"] for row in augmented]
    assert {symmetry for _, symmetry in augmented} == {symmetry.value for symmetry in SYMMETRIES}
    assert {symmetry for _, symmetry in unaugmented} == {"identity"}


def test_invariant_task_and_null_instruction_are_required():
    invalid = record()
    invalid["task_goal"] = "Move toward (1, 3)."
    with pytest.raises(ValueError, match="coordinate-free"):
        transform_training_record(invalid, "rotate_90")
    invalid = record()
    invalid["intent"]["instruction"] = "Move toward (1, 3)."
    with pytest.raises(ValueError, match="null instruction"):
        transform_training_record(invalid, "rotate_90")


def test_fixed_padding_keeps_response_only_loss_and_never_truncates():
    tokenizer = CharTokenizer()
    original = record()
    encoded = encode_example(tokenizer, original, 5000)
    prefix = tokenizer.apply_chat_template(prompt_messages(original))
    assert encoded["labels"][:len(prefix)] == [-100] * len(prefix)
    target_length = len(encoded["input_ids"]) + 8
    batch = tensor_example(torch, encoded, "cpu", pad_to_length=target_length, pad_token_id=0)
    assert batch["input_ids"].shape == (1, target_length)
    assert batch["attention_mask"][0, -8:].tolist() == [0] * 8
    assert batch["labels"][0, -8:].tolist() == [-100] * 8
    with pytest.raises(ValueError, match="padding limit"):
        tensor_example(torch, encoded, "cpu", pad_to_length=len(encoded["input_ids"]) - 1, pad_token_id=0)
    with pytest.raises(ValueError, match="no truncation"):
        encode_example(tokenizer, original, 10)


def test_preflight_checks_every_train_orientation_and_original_validation():
    train, valid = record(), record("other", "val")
    valid["split"] = "validation"
    summary = training_lengths(CharTokenizer(), [train], [valid],
                               Namespace(max_length=5000, augmentation="d4", pad_to_length=0))
    assert summary["n"] == 9 and summary["training_orientations_per_record"] == 8
    assert summary["truncated"] == 0
    with pytest.raises(ValueError, match="padding limit"):
        training_lengths(CharTokenizer(), [train], [valid],
                         Namespace(max_length=5000, augmentation="d4", pad_to_length=10))


def test_multiarm_preflight_requires_shared_validation_and_selects_common_padding(tmp_path):
    train, valid = record(), record("other", "val")
    valid["split"] = "validation"
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_text("\n".join(json.dumps(row) for row in [train, valid]) + "\n")
    second.write_text("\n".join(json.dumps(row) for row in [dict(train, record_id="wide"), valid]) + "\n")
    result = inspect(CharTokenizer(), [first, second], max_length=5000, multiple=8)
    assert result["common_pad_to_length"] % 8 == 0 and result["truncated_examples"] == 0
    assert all(item["token_lengths_all_d4"]["n"] == 9 for item in result["datasets"])
    different = dict(valid, record_id="different-validation")
    second.write_text("\n".join(json.dumps(row) for row in [train, different]) + "\n")
    with pytest.raises(ValueError, match="same validation"):
        inspect(CharTokenizer(), [first, second], max_length=5000, multiple=8)
