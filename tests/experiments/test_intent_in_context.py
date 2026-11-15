import json
from argparse import Namespace

import pytest

from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import prepare_evaluation_output
from carl.icl.intent_in_context import build_prompts
from carl.icl.intent_in_context import freeze_support
from carl.icl.intent_in_context import in_context_messages
from carl.icl.intent_in_context import load_support
from carl.icl.intent_in_context import mark_partial_superseded
from carl.icl.intent_in_context import prepare
from carl.icl.intent_in_context import run
from carl.icl.intent_in_context import save_prompts
from carl.icl.intent_in_context import select_support
from carl.icl.intent_in_context import stop_if_deadline


def record(record_id, layout, effect_type, split="train"):
    return {
        "record_id": record_id,
        "layout_id": layout,
        "split": split,
        "effect_type": effect_type,
        "validation": {
            "accepted": True
        },
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
        "later_observation": ["SECRET_FUTURE_" + record_id],
        "fragment_actions": ["SECRET_ACTIONS_" + record_id],
        "source_path": "SECRET_PATH_" + record_id,
        "k": "SECRET_OFFSET_" + record_id,
    }


class TestTokenizer:
    chat_template: str

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert add_generation_prompt and enable_thinking is False
        rendered = json.dumps(messages)
        return list(rendered) if tokenize else rendered


def test_support_selection_is_fixed_checked_train_and_board_disjoint():
    rows = [
        record("train-b", "b", "box"),
        record("train-a", "a", "box"),
        record("goal-a", "a", "goal_conjunction"),
        record("goal-c", "c", "goal_conjunction"),
        record("test", "0", "box", "test")
    ]
    assert [row["record_id"] for row in select_support(rows)] == ["train-a", "goal-c"]
    rows[1]["validation"]["accepted"] = False
    assert [row["record_id"] for row in select_support(rows)] == ["train-b", "goal-a"]


def test_prompt_contains_only_demonstration_labels_and_current_query():
    train = record("train", "a", "box")
    query = record("query", "b", "box", "test")
    query["intent"]["instruction"] = "SECRET_TEST_LABEL"
    messages = in_context_messages(query, [train])
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert json.loads(messages[2]["content"]) == train["intent"]
    assert "#@$.#" in messages[-1]["content"]
    assert "SECRET_" not in json.dumps(messages)
    changed = dict(query, later_observation=["OTHER"], fragment_actions=[999], intent={"secret": "changed"}, k=999)
    assert in_context_messages(changed, [train]) == messages
    with pytest.raises(ValueError, match="share the query layout"):
        in_context_messages(query, [dict(train, layout_id="b")])
    with pytest.raises(ValueError, match="checked training"):
        in_context_messages(query, [dict(train, split="test")])


def test_frozen_support_and_complete_prompt_length(tmp_path):
    data = tmp_path / "data.jsonl"
    rows = [record("a", "a", "box"), record("c", "c", "goal_conjunction"), record("test", "t", "box", "test")]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "support.json"
    freeze_support(data, manifest)
    support = load_support(data, manifest, rows)
    prompts = build_prompts(TestTokenizer(), [rows[-1]], support, max_length=10000, max_new_tokens=128)
    assert prompts[0]["messages"][-1]["role"] == "user"
    assert prompts[0]["rendered_prompt"] == json.dumps(prompts[0]["messages"])
    with pytest.raises(ValueError, match="no truncation"):
        build_prompts(TestTokenizer(), [rows[-1]], support, max_length=100, max_new_tokens=128)
    path = tmp_path / "prompts.jsonl"
    save_prompts(path, prompts)
    save_prompts(path, prompts)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        save_prompts(path, [])
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="different dataset"):
        load_support(data, manifest, rows)


def test_support_can_come_from_original_train_dataset_when_test_data_is_separate(tmp_path, monkeypatch):
    original = tmp_path / "original.jsonl"
    train = [record("a", "a", "box"), record("c", "c", "goal_conjunction")]
    original.write_text("".join(json.dumps(row) + "\n" for row in train))
    support_path = tmp_path / "support.json"
    freeze_support(original, support_path)
    extended = tmp_path / "extended.jsonl"
    query = record("test", "t", "box", "test")
    extended.write_text(json.dumps(query) + "\n")
    test_ids = tmp_path / "test_ids.json"
    test_ids.write_text(json.dumps({"dataset_sha256": file_sha256(extended), "record_ids": ["test"]}))
    tokenizer = TestTokenizer()
    tokenizer.chat_template = "available"
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer)
    args = Namespace(data=str(extended),
                           support_data=str(original),
                           support=str(support_path),
                           test_ids=str(test_ids),
                           model="local",
                           max_length=10000,
                           max_new_tokens=128,
                           output=str(tmp_path / "predictions.jsonl"))
    records, support, prompts, _ = prepare(args)
    assert [row["record_id"] for row in records] == ["test"]
    assert [row["record_id"] for row in support] == ["a", "c"]
    assert len(prompts) == 1
    query["layout_id"] = "a"
    extended.write_text(json.dumps(query) + "\n")
    test_ids.write_text(json.dumps({"dataset_sha256": file_sha256(extended), "record_ids": ["test"]}))
    with pytest.raises(ValueError, match="layouts overlap"):
        prepare(args)


def test_in_context_resume_checks_the_method(tmp_path):
    output = tmp_path / "predictions.jsonl"
    query = record("query", "t", "box", "test")
    config = {"model": "fixed", "adapter": None, "method": "in_context"}
    assert prepare_evaluation_output(output, config, [query], resume=False) == []
    row = {
        "record_id": "query",
        "layout_id": "t",
        "split": "test",
        "method": "in_context",
        "raw_output": "{}",
        "json_valid": True,
        "parsed": {}
    }
    output.write_text(json.dumps(row) + "\n")
    assert prepare_evaluation_output(output, config, [query], resume=True) == [row]
    row["method"] = "base"
    output.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="ordered frozen"):
        prepare_evaluation_output(output, config, [query], resume=True)


def test_expired_deadline_fails_stage_before_model_loading_and_records_progress(tmp_path, monkeypatch):
    output = tmp_path / "predictions.jsonl"
    query = record("query", "t", "box", "test")
    support = [{"record_id": "train"}]
    prompts = [{"record_id": "query", "messages": []}]
    monkeypatch.setattr("carl.icl.intent_in_context.prepare", lambda args:
                        ([query], support, prompts, tmp_path / "prompts.jsonl"))
    monkeypatch.setattr("carl.icl.intent_in_context.evaluation_configuration", lambda args: {"adapter": None})
    monkeypatch.setattr("carl.icl.intent_in_context.file_sha256", lambda path: "test-hash")

    def unexpected_initialise(*args, **kwargs):
        raise AssertionError("The model must not be loaded after the deadline")

    monkeypatch.setattr("carl.icl.intent_in_context.initialise", unexpected_initialise)
    args = Namespace(mode="evaluate",
                           data="data.jsonl",
                           support="support.json",
                           support_data=None,
                           output=str(output),
                           resume=False,
                           stop_monotonic=0)
    with pytest.raises(SystemExit, match="deadline"):
        run(args)
    partial = json.loads(output.with_suffix(".partial-summary.json").read_text())
    assert partial["status"] == "stopped"
    assert partial["processed"] == 0 and partial["planned"] == 1
    assert output.with_suffix(".run.json").is_file()
    assert not output.exists()


def test_deadline_preserves_existing_prediction_lines(tmp_path):
    output = tmp_path / "predictions.jsonl"
    output.write_text('{"record_id":"saved"}\n')
    with pytest.raises(SystemExit, match="deadline"):
        stop_if_deadline(output, stop_monotonic=0, processed=1, planned=2, started=0)
    assert output.read_text() == '{"record_id":"saved"}\n'
    partial = json.loads(output.with_suffix(".partial-summary.json").read_text())
    assert partial["processed"] == 1 and partial["planned"] == 2
    output.with_suffix(".summary.json").write_text('{"status":"complete"}\n')
    mark_partial_superseded(output)
    resumed = json.loads(output.with_suffix(".partial-summary.json").read_text())
    assert resumed["processed"] == 1 and resumed["reason"] == "deadline"
    assert resumed["final_status"] == "complete"
    assert resumed["superseded_by"] == "predictions.summary.json"
