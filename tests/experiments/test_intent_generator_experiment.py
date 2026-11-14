import json

import pytest

from carl.icl.intent_generator_experiment import board_balanced
from carl.icl.intent_generator_experiment import encode_example
from carl.icl.intent_generator_experiment import evaluation_records
from carl.icl.intent_generator_experiment import one_per_board
from carl.icl.intent_generator_experiment import parse_json
from carl.icl.intent_generator_experiment import prepare_evaluation_output
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import validation_contexts


def example(layout="board-a"):
    return {
        "layout_id": layout,
        "start_observation": ["#####", "#@$.#", "#####"],
        "task_goal": "Put every box on a goal.",
        "intent": {
            "effects": [{
                "predicate": "box_at",
                "row": 1,
                "col": 3
            }],
            "preserve": [],
            "instruction": None
        },
        "later_observation": ["SECRET-FUTURE"],
        "fragment_actions": ["SECRET-ACTIONS"],
        "record_id": "SECRET-ID",
        "source_path": "SECRET-PATH",
        "k": "SECRET-OFFSET",
        "validation": "SECRET-VALIDATION",
    }


class CharTokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and enable_thinking is False
        return [ord(c) for c in json.dumps(messages)]

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [ord(c) for c in text]


def test_generator_whitelist_excludes_future_actions_and_metadata():
    prompt = json.dumps(prompt_messages(example()))
    assert "#@$.#" in prompt
    assert "Put every box on a goal." in prompt
    assert "SECRET-" not in prompt


def test_response_only_loss_and_no_implicit_truncation():
    tokenizer = CharTokenizer()
    record = example()
    prefix = tokenizer.apply_chat_template(prompt_messages(record),
                                           tokenize=True,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
    encoded = encode_example(tokenizer, record, max_length=5000)
    assert encoded["labels"][:len(prefix)] == [-100] * len(prefix)
    assert encoded["labels"][len(prefix):] == encoded["input_ids"][len(prefix):]
    assert encoded["labels"][-1] == ord("!")
    with pytest.raises(ValueError, match="no truncation"):
        encode_example(tokenizer, record, max_length=20)


def test_round_robin_balances_contexts_and_evaluation_is_per_board():
    records = [example("a"), example("a"), example("a"), example("b")]
    samples = board_balanced(records, seed=42)
    for _ in range(4):
        assert {next(samples)["layout_id"], next(samples)["layout_id"]} == {"a", "b"}
    assert [r["layout_id"] for r in one_per_board(records)] == ["a", "b"]


def test_frozen_evaluation_selection_is_enforced(tmp_path):
    a, b = example("a"), example("b")
    a["record_id"], b["record_id"] = "a0", "b0"
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"record_ids": ["b0", "a0"]}))
    assert [r["record_id"] for r in evaluation_records([a, b], str(frozen), 0)] == ["b0", "a0"]
    frozen.write_text(json.dumps({"record_ids": ["b0", "missing"]}))
    with pytest.raises(ValueError, match="absent"):
        evaluation_records([a, b], str(frozen), 0)
    frozen.write_text(json.dumps({"record_ids": ["b0", "b0"]}))
    with pytest.raises(ValueError, match="unique"):
        evaluation_records([a, b], str(frozen), 0)


def test_validation_contexts_are_fixed_and_diverse():
    records = []
    for layout in ("a", "b"):
        for effect in ("goal", "goal", "box", "box", "walk"):
            row = example(layout)
            row["effect_type"] = effect
            records.append(row)
    selected = validation_contexts(records, max_boards=2, per_board=3)
    assert len(selected) == 6
    for layout in ("a", "b"):
        assert {r["effect_type"] for r in selected if r["layout_id"] == layout} == {"goal", "box", "walk"}


def test_json_parsing_rejects_python_only_nonfinite_constants():
    assert parse_json('{"n":1}') == {"n": 1}
    with pytest.raises(ValueError):
        parse_json('{"n":NaN}')


def test_evaluation_resume_preserves_valid_rows_and_interrupted_tail(tmp_path):
    output = tmp_path / "predictions.jsonl"
    records = [dict(example("a"), record_id="a0"), dict(example("b"), record_id="b0")]
    config = {"model": "fixed-model", "adapter": None, "seed": 42}
    assert prepare_evaluation_output(output, config, records, resume=False) == []
    row = {
        "record_id": "a0",
        "layout_id": "a",
        "split": "test",
        "method": "base",
        "raw_output": '{"x":1}',
        "json_valid": True,
        "parsed": {
            "x": 1
        }
    }
    output.write_text(json.dumps(row) + '\n{"incomplete":', encoding="utf-8")
    assert prepare_evaluation_output(output, config, records, resume=True) == [row]
    assert len(list(tmp_path.glob("predictions.jsonl.partial-*"))) == 1
    with pytest.raises(ValueError, match="configuration changed"):
        prepare_evaluation_output(output, dict(config, seed=43), records, resume=True)
    with pytest.raises(FileExistsError):
        prepare_evaluation_output(output, config, records, resume=False)


def test_evaluation_resume_rejects_unidentified_or_mismatched_outputs(tmp_path):
    output = tmp_path / "predictions.jsonl"
    output.write_text('{}\n', encoding="utf-8")
    config = {"model": "fixed-model", "adapter": None}
    records = [dict(example("a"), record_id="a0")]
    with pytest.raises(ValueError, match="without its original configuration"):
        prepare_evaluation_output(output, config, records, resume=True)
    output.with_suffix(".run.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered frozen"):
        prepare_evaluation_output(output, config, records, resume=True)
