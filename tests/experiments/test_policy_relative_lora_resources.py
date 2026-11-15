"""Synthetic contract tests for policy-relative generators."""

import json
from argparse import Namespace
from collections import Counter

import pytest

from carl.icl.policy_relative_lora import resource_check
from carl.icl.policy_relative_lora import resource_encodings
from carl.icl.policy_relative_lora import resource_records
from tests.experiments._policy_relative_lora_fixtures import FakeTokenizer
from tests.experiments._policy_relative_lora_fixtures import make_dataset
from tests.experiments._policy_relative_lora_fixtures import write_rows


@pytest.fixture(name="dataset")
def calibration_dataset(tmp_path):
    return make_dataset(tmp_path)


def test_resource_data_excludes_test_and_uses_scale_only_as_diagnostic_input(dataset):
    path, rows = dataset
    for row in rows:
        row["policy_measurements"] = None
    rows.append({"split": "test", "record_id": "SECRET_TEST_WITHOUT_A_BOARD"})
    write_rows(path, rows)
    records = resource_records(path)
    assert len(records) == 12
    tokenizer = FakeTokenizer()
    prepared = resource_encodings(tokenizer, records, 1024, 32)
    assert len(prepared) == 12
    assert all(
        "Requested relative scale: 0.5" in json.dumps(messages)
        for messages in tokenizer.messages
    )
    assert all(
        "SECRET_FUTURE" not in json.dumps(messages) for messages in tokenizer.messages
    )
    assert all(
        "SECRET_TEST_WITHOUT_A_BOARD" not in json.dumps(item[2]) for item in prepared
    )
    assert all(
        item[1]["labels"][:3] == [-100] * 3 and item[2]["truncated"] is False
        for item in prepared
    )
    assert all(
        item[2]["complete_example_tokens"]
        == item[2]["prompt_tokens"] + item[2]["response_tokens"]
        for item in prepared
    )
    with pytest.raises(ValueError, match="no truncation"):
        resource_encodings(FakeTokenizer(), records, 4, 32)


def test_resource_check_full_flow_with_stubs_only(dataset, tmp_path, monkeypatch):
    import carl.icl.policy_relative.resource as module

    path, rows = dataset
    for row in rows:
        row["policy_measurements"] = None
    write_rows(path, rows)
    model = tmp_path / "snapshot"
    model.mkdir()
    revision = "a" * 40
    (model / "experiment_revision.json").write_text(json.dumps({"revision": revision}))
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    calls = Counter()

    class Scalar:
        def __truediv__(self, _number):
            return self

        def detach(self):
            return self

        def cpu(self):
            return 0.5

        def backward(self):
            calls["backward"] += 1

    class StubModel:
        def parameters(self):
            return [Namespace(requires_grad=True)]

        def train(self):
            pass

        def __call__(self, **_inputs):
            return Namespace(loss=Scalar())

    class Optimiser:
        def __init__(self, parameters, lr):
            assert len(list(parameters)) == 1 and lr == 0.0002

        def step(self):
            calls["optimizer_step"] += 1

        def zero_grad(self, set_to_none):
            assert set_to_none is True

    fake_torch = Namespace(optim=Namespace(AdamW=Optimiser))
    monkeypatch.setattr(
        module,
        "initialise",
        lambda *_args, **_kwargs: (fake_torch, FakeTokenizer(), StubModel(), "stub"),
    )
    monkeypatch.setattr(
        module, "tensor_example", lambda _torch, example, _device: example
    )
    monkeypatch.setattr(module, "sync_gpu", lambda _torch: None)
    monkeypatch.setattr(
        module, "memory", lambda _torch: {"gpu_peak_reserved_bytes": None}
    )

    def fake_generate(_torch, _model, _tokenizer, record, _args, _device, messages):
        calls["generate"] += 1
        assert record["split"] in ("train", "validation")
        assert "Requested relative scale: 0.5" in json.dumps(messages)
        return "STUB_OUTPUT", 4, 0.5

    monkeypatch.setattr(module, "generate", fake_generate)
    output = tmp_path / "resource.json"
    args = Namespace(
        data=str(path),
        output=str(output),
        model=str(model),
        revision=revision,
        dtype="bfloat16",
        rank=4,
        seed=1729,
        threads=8,
        accumulation=4,
        max_length=1024,
        max_new_tokens=32,
        gradient_checkpointing=True,
        lr=0.0002,
        require_cuda=False,
        stop_monotonic=None,
        stop_epoch=None,
    )
    resource_check(args)
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    assert report["measurement_kind"] == "resource_only_no_luna_labels"
    assert (
        report["synthetic_requested_scale"] == 0.5
        and len(report["token_lengths"]) == 12
    )
    assert calls == {"backward": 4, "optimizer_step": 1, "generate": 1}
    assert not list(tmp_path.rglob("adapter_model.safetensors"))
