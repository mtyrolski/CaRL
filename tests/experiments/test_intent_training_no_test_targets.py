"""Training resource checks must not inspect held-out target text."""

from types import SimpleNamespace

import pytest

from carl.icl import intent_generator_experiment


class StopAfterLengths(Exception):
    """Stop the local test before creating an optimiser or running a model."""


@pytest.mark.parametrize("operation", ["benchmark", "train"])
def test_length_checks_never_tokenize_test_targets(tmp_path, monkeypatch, operation):
    rows = [
        {"record_id": "training", "layout_id": "training", "split": "train", "intent": {"instruction": "TRAIN"}},
        {"record_id": "validation", "layout_id": "validation", "split": "validation", "intent": {"instruction": "VAL"}},
        {"record_id": "held_out", "layout_id": "held_out", "split": "test", "intent": {"instruction": "SECRET TEST TARGET"}},
    ]
    module = intent_generator_experiment
    monkeypatch.setattr(module, "read_records", lambda _path: rows)
    monkeypatch.setattr(module, "initialise", lambda _args, train: (object(), object(), object(), "cpu"))
    observed = []

    def encode(_tokenizer, record, _max_length):
        observed.append(record["split"])
        assert record["split"] != "test", "held-out intent was sent to the tokenizer"
        return {"input_ids": [1, 2]}

    real_lengths = module.lengths

    def checked_lengths(tokenizer, records, max_length):
        real_lengths(tokenizer, records, max_length)
        raise StopAfterLengths

    monkeypatch.setattr(module, "encode_example", encode)
    monkeypatch.setattr(module, "lengths", checked_lengths)
    args = SimpleNamespace(data="unused", output=str(tmp_path / "output"), accumulation=1, max_length=512,
                           validation_boards=1, validation_contexts_per_board=1)
    with pytest.raises(StopAfterLengths):
        getattr(module, operation)(args)
    assert set(observed) == {"train", "validation"}
