"""Exact training-label matches remain distinct from board validity and missing parses."""

import json

import pytest

from scripts.audit_lora_training_intents import describe
from scripts.audit_lora_training_intents import enrich
from scripts.audit_lora_training_intents import label_index
from scripts.audit_lora_training_intents import write_output


def label(identifier, layout="train-board"):
    return {"record_id": identifier, "layout_id": layout, "split": "train", "validation": {"accepted": True},
            "intent": {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": [],
                       "instruction": None}}


def context(identifier, layout="test-board"):
    board = ["############", "#$$$$------#", "#@....-----#"] + ["#----------#"] * 8 + ["############"]
    return {"record_id": identifier, "layout_id": layout, "start_observation": board}


def proposal(identifier, layout, *, valid, signature):
    return {"variant": "rep_lora_json_zero", "mode": "strict", "record_id": identifier, "layout_id": layout,
            "family": "real_12x12x4", "parseable": signature is not None, "semantic_valid": valid,
            "validation_errors": [] if valid else ["position is a wall"] if valid is False else None,
            "canonical_signature": signature,
            "canonical_intent": {"effects": label("x")["intent"]["effects"], "preserve": []} if signature else None}


def test_training_label_index_excludes_nontraining_and_unchecked_records():
    rows = [label("a"), label("b", "other-board")]
    index = label_index(rows)
    assert len(index) == 1 and len(next(iter(index.values()))) == 2
    for modified in ({**rows[0], "split": "test"}, {**rows[0], "validation": {"accepted": False}},
                     {**rows[0], "intent": {**rows[0]["intent"], "instruction": "other"}}):
        with pytest.raises(ValueError, match="Only checked training labels"):
            label_index([modified])


def test_exact_match_and_current_board_validity_have_separate_counts():
    labels = [label("a"), label("b", "other-board")]
    index = label_index(labels)
    signature = next(iter(index))
    first = enrich(proposal("one", "b1", valid=True, signature=signature), context("one", "b1"), index, index)
    second = enrich(proposal("two", "b2", valid=False, signature=signature), context("two", "b2"), index, index)
    malformed = enrich(proposal("three", "b3", valid=None, signature=None), context("three", "b3"), index, index)
    summary, frequencies = describe("model", "strict", "all", "all", [first, second, malformed])
    assert (summary["outputs"], summary["parseable"], summary["invalid_format"]) == (3, 2, 1)
    assert (summary["training_pool_match"], summary["training_pool_match_valid_on_board"]) == (2, 1)
    assert (frequencies[0]["occurrences"], frequencies[0]["valid_on_board"], frequencies[0]["invalid_on_board"]) == (2, 1, 1)
    assert frequencies[0]["matching_train_record_ids"] == ["a", "b"]
    assert malformed["training_pool_match"] is None and malformed["reconstructed_sample_match"] is None


def test_reconstructed_sample_and_training_pool_are_not_conflated():
    labels = [label("a")]
    pool = label_index(labels)
    signature = next(iter(pool))
    row = enrich(proposal("one", "board", valid=True, signature=signature), context("one", "board"), pool, {})
    assert row["training_pool_match"] is True and row["reconstructed_sample_match"] is False
    with pytest.raises(ValueError, match="current board"):
        enrich(proposal("one", "different", valid=True, signature=signature), context("one", "board"), pool, {})


def test_output_is_new_and_rejects_changed_inputs(tmp_path):
    source = tmp_path / "input"
    source.write_text("original")
    from carl.icl.intent_generator_experiment import file_sha256
    summary = {"input_sha256": {str(source.resolve()): file_sha256(source)}, "summaries": [{"outputs": 1}]}
    destination = tmp_path / "result"
    write_output(destination, summary, [{"occurrences": 1}], [{"record_id": "one"}], [source])
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["analysis_kind"] == "posthoc_descriptive"
    assert set(provenance["output_sha256"]) == {"summary.json", "summary.csv", "frequencies.csv", "records.jsonl"}
    with pytest.raises(FileExistsError):
        write_output(destination, summary, [], [], [source])
    source.write_text("changed")
    with pytest.raises(ValueError, match="input changed"):
        write_output(tmp_path / "new", summary, [], [], [source])
