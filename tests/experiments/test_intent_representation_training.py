"""Safety and parity checks for representation-matched generator fitting."""

import json
from argparse import Namespace

import pytest

from carl.icl.intent_generator_experiment import board_balanced
from carl.icl.intent_representation_experiment import Representation
from carl.icl.intent_representation_training import completion_status
from carl.icl.intent_representation_training import encode_example
from carl.icl.intent_representation_training import expired
from carl.icl.intent_representation_training import save_adapter
from carl.icl.intent_representation_training import token_lengths
from carl.icl.intent_representation_training import train
from carl.icl.intent_representation_training import training_splits


class FakeTokenizer:
    eos_token = "<end>"

    def __init__(self):
        self.messages = []
        self.answers = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        return [10, 11, 12]

    def encode(self, answer, **kwargs):
        self.answers.append(answer)
        assert kwargs["add_special_tokens"] is False
        return [20 + n for n in range(len(answer))]


def row(record_id, layout, split):
    return {"record_id": record_id, "layout_id": layout, "split": split, "effect_type": "box",
            "validation": {"accepted": True}, "task_goal": "move the boxes", "start_observation": ["#####", "#@$.#", "#####"],
            "intent": {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": [],
                       "instruction": None}, "later_observation": ["SECRET_FUTURE"], "fragment_actions": [999], "k": 123}


@pytest.mark.parametrize("representation", list(Representation))
def test_response_only_loss_and_no_future_in_messages(representation):
    tokenizer = FakeTokenizer()
    sample = row("train-id", "train-layout", "train")
    encoded = encode_example(tokenizer, sample, representation, 1024)
    assert encoded["labels"][:3] == [-100, -100, -100]
    assert encoded["labels"][3:] == encoded["input_ids"][3:]
    assert tokenizer.answers[0].endswith(tokenizer.eos_token)
    assert "SECRET_FUTURE" not in json.dumps(tokenizer.messages)
    assert "train-id" not in json.dumps(tokenizer.messages)
    assert "999" not in json.dumps(tokenizer.messages)
    with pytest.raises(ValueError, match="no truncation"):
        encode_example(tokenizer, sample, representation, 4)


def test_splits_ignore_test_targets_and_produce_matching_sampling(tmp_path):
    path = tmp_path / "dataset.jsonl"
    train1 = row("a", "train-a", "train")
    train2 = row("b", "train-b", "train")
    validation = row("v", "validation", "validation")
    test = row("t", "test", "test")
    test["intent"] = {"invalid": "MUST_NOT_BE_USED"}
    path.write_text("".join(json.dumps(r) + "\n" for r in (train1, test, train2, validation)))
    training, held_out_validation = training_splits(path, max_boards=12, per_board=4)
    assert [r["record_id"] for r in training] == ["a", "b"]
    assert [r["record_id"] for r in held_out_validation] == ["v"]
    sequences = []
    for representation in Representation:
        tokenizer = FakeTokenizer()
        assert token_lengths(tokenizer, training + held_out_validation, representation, 1024)["n"] == 3
        assert "MUST_NOT_BE_USED" not in json.dumps(tokenizer.messages + tokenizer.answers)
        sample_order = board_balanced(training, seed=1729)
        sequences.append([next(sample_order)["record_id"] for _ in range(10)])
    assert sequences[0] == sequences[1] == sequences[2]


def test_split_overlap_or_unchecked_annotation_is_rejected(tmp_path):
    path = tmp_path / "dataset.jsonl"
    training = row("a", "same", "train")
    validation = row("v", "same", "validation")
    path.write_text(json.dumps(training) + "\n" + json.dumps(validation) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        training_splits(path, max_boards=12, per_board=4)
    validation["layout_id"] = "other"
    training["validation"]["accepted"] = False
    path.write_text(json.dumps(training) + "\n" + json.dumps(validation) + "\n")
    with pytest.raises(ValueError, match="checked"):
        training_splits(path, max_boards=12, per_board=4)


def test_adapter_checkpoint_only_appears_after_successful_write(tmp_path):
    class FakeModel:
        def save_pretrained(self, path, safe_serialization):
            assert safe_serialization
            from pathlib import Path
            Path(path).mkdir()
            (Path(path) / "adapter_model.safetensors").write_text("weights")

    destination = save_adapter(FakeModel(), tmp_path, "checkpoint-20")
    assert (destination / "adapter_model.safetensors").exists()
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError):
        save_adapter(FakeModel(), tmp_path, "checkpoint-20")


def test_deadline_and_initialisation_failure_are_recorded(tmp_path, monkeypatch):
    args = Namespace(stop_monotonic=0.0, stop_epoch=None)
    assert expired(args)
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(row("a", "train", "train")) + "\n" +
                    json.dumps(row("v", "validation", "validation")) + "\n")
    args = Namespace(data=str(path), output=str(tmp_path / "out"), validation_boards=12,
                     validation_contexts_per_board=4, model=str(tmp_path / "missing-model"),
                     revision="wrong", require_cuda=False, stop_monotonic=None, stop_epoch=None)
    with pytest.raises(ValueError, match="snapshot"):
        train(args)
    status = json.loads((tmp_path / "out/status.json").read_text())
    assert status["status"] == "failed" and status["completed_steps"] == 0


def test_missing_final_validation_is_not_reported_as_finished():
    assert completion_status(100, 100, 100) == ("finished", None)
    assert completion_status(100, 100, 80) == ("stopped", "final_validation_missing")
    assert completion_status(99, 100, 80) == ("stopped", "signal_or_deadline")
