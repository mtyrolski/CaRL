"""Synthetic contract tests for policy-relative generators."""

import hashlib
import json
from argparse import Namespace

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.policy_relative_lora import Arm
from carl.icl.policy_relative_lora import PredictionStatus
from carl.icl.policy_relative_lora import _configuration
from carl.icl.policy_relative_lora import assess_output
from carl.icl.policy_relative_lora import evaluate
from carl.icl.policy_relative_lora import generator_messages
from carl.icl.policy_relative_lora import load_fit_data
from carl.icl.policy_relative_lora import test_contexts as load_test_contexts
from carl.icl.policy_relative_lora import verify_adapter
from tests.experiments._policy_relative_lora_fixtures import BOARDS
from tests.experiments._policy_relative_lora_fixtures import FakeTokenizer
from tests.experiments._policy_relative_lora_fixtures import make_dataset
from tests.experiments._policy_relative_lora_fixtures import write_rows


@pytest.fixture(name="dataset")
def calibration_dataset(tmp_path):
    return make_dataset(tmp_path)


def test_test_contexts_and_prediction_statuses_are_explicit(tmp_path):
    board = BOARDS[0]
    context = {
        "record_id": "test-id",
        "context_id": "test-context",
        "layout_id": canonical_layout_id(from_rows(board)),
        "split": "test",
        "family": "test-family",
        "start_observation": board,
        "task_goal": "Solve.",
    }
    path = tmp_path / "test.jsonl"
    write_rows(path, [context])
    assert load_test_contexts(path) == [context]
    valid = '{"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[],"instruction":null}'
    assert assess_output(valid, board)["status"] == PredictionStatus.VALID
    assert (
        assess_output("```json\n" + valid + "\n```", board)["status"]
        == PredictionStatus.INVALID_JSON
    )
    assert (
        assess_output('{"x":1,"x":2}', board)["status"] == PredictionStatus.INVALID_JSON
    )
    assert (
        assess_output('{"effects":[],"preserve":[]}', board)["status"]
        == PredictionStatus.INVALID_INTENT
    )
    write_rows(path, [context, context])
    with pytest.raises(ValueError, match="unique"):
        load_test_contexts(path)


def test_adapter_is_validation_selected_and_never_tested_on_fitting_layouts(
    dataset, tmp_path
):
    path, _ = dataset
    data = load_fit_data(path)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    args = Namespace(
        model=str(model),
        revision="a" * 40,
        data=str(path),
        seed=1729,
        dtype="bfloat16",
        rank=4,
        accumulation=4,
        lr=0.0002,
        max_steps=100,
        eval_every=20,
        max_length=512,
        gradient_checkpointing=True,
        arm=Arm.SCALED,
    )
    training = _configuration(args, data, FakeTokenizer(), "cuda")
    run = tmp_path / "training"
    adapter = run / "checkpoint-20"
    adapter.mkdir(parents=True)
    (run / "configuration.json").write_text(json.dumps(training))
    (run / "best.json").write_text(json.dumps({"adapter": str(adapter)}))
    unseen = {"layout_id": "new-held-out-layout"}
    assert (
        verify_adapter(adapter, model, args.revision, Arm.SCALED, [unseen]) == training
    )
    legacy = dict(training)
    legacy.pop("code_sha256", None)
    legacy.pop("code_files_sha256", None)
    (run / "configuration.json").write_text(json.dumps(legacy))
    assert verify_adapter(adapter, model, args.revision, Arm.SCALED, [unseen]) == legacy
    (run / "configuration.json").write_text(json.dumps(training))
    with pytest.raises(ValueError, match="arm, model revision"):
        verify_adapter(adapter, model, args.revision, Arm.UNSCALED, [unseen])
    with pytest.raises(ValueError, match="overlap"):
        verify_adapter(
            adapter,
            model,
            args.revision,
            Arm.SCALED,
            [{"layout_id": data.train[0].layout_id}],
        )
    (run / "best.json").write_text(json.dumps({"adapter": str(run / "checkpoint-40")}))
    with pytest.raises(ValueError, match="validation-selected checkpoint"):
        verify_adapter(adapter, model, args.revision, Arm.SCALED, [unseen])


def test_evaluation_uses_only_synthetic_stubs_and_retains_every_status(
    dataset, tmp_path, monkeypatch
):
    import carl.icl.policy_relative.evaluation as module

    path, _ = dataset
    data = load_fit_data(path)
    model = tmp_path / "model"
    model.mkdir()
    revision = "a" * 40
    (model / "experiment_revision.json").write_text(json.dumps({"revision": revision}))
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    common = dict(
        model=str(model),
        revision=revision,
        data=str(path),
        seed=1729,
        dtype="bfloat16",
        rank=4,
        accumulation=4,
        lr=0.0002,
        max_steps=100,
        eval_every=20,
        max_length=512,
        max_new_tokens=128,
        gradient_checkpointing=True,
        arm=Arm.SCALED,
    )
    training = _configuration(Namespace(**common), data, FakeTokenizer(), "cpu")
    run = tmp_path / "training"
    adapter = run / "checkpoint-20"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_text("fake weights")
    (adapter / "adapter_config.json").write_text("{}")
    (run / "configuration.json").write_text(json.dumps(training))
    (run / "best.json").write_text(json.dumps({"adapter": str(adapter)}))
    board = ["#######", "#@-$--#", "#-.---#", "#######"]
    layout = canonical_layout_id(from_rows(board))
    contexts = [
        {
            "record_id": name,
            "context_id": name,
            "layout_id": layout,
            "split": "test",
            "family": "fixture",
            "start_observation": board,
            "task_goal": "Solve.",
        }
        for name in ("valid", "invalid", "failure", "later")
    ]
    test_file = tmp_path / "test.jsonl"
    write_rows(test_file, contexts)
    monkeypatch.setattr(
        module,
        "initialise",
        lambda *_args, **_kwargs: (object(), FakeTokenizer(), object(), "cpu"),
    )
    valid = '{"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[],"instruction":null}'

    def fake_generate(_torch, _model, _tokenizer, record, _args, _device, messages):
        assert "Requested relative scale: 0.25" in json.dumps(messages)
        if record["record_id"] == "failure":
            raise RuntimeError("synthetic inference infrastructure error")
        return (valid if record["record_id"] == "valid" else "not-json"), 3, 0.1

    monkeypatch.setattr(module, "generate", fake_generate)
    output = tmp_path / "predictions.jsonl"
    args = Namespace(
        **{**common, "data": str(test_file)},
        output=str(output),
        requested_scale=0.25,
        adapter=str(adapter),
        require_cuda=False,
        stop_monotonic=None,
        stop_epoch=None
    )
    with pytest.raises(RuntimeError, match="synthetic inference infrastructure error"):
        evaluate(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["status"] for row in rows] == [
        "valid_intent",
        "invalid_json",
        "infrastructure_error",
        "not_attempted",
    ]
    assert rows[0]["parsed"]["effects"][0]["col"] == 4
    assert rows[1]["raw_output"] == "not-json" and rows[1]["parsed"] is None
    assert rows[2]["error"].endswith("synthetic inference infrastructure error")
    assert all("SECRET_FUTURE" not in json.dumps(row["prompt"]) for row in rows)


def test_symmetric_single_fence_normalization_retains_strict_validity():
    valid = '{"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[],"instruction":null}'
    wrapped = "```json\n" + valid + "\n```"
    result = assess_output(wrapped, BOARDS[0], normalize=True)
    assert result["status"] == PredictionStatus.VALID
    assert (
        result["strict_json_valid"] is False and result["normalized_json_valid"] is True
    )
    assert result["fence_removed"] is True
    assert assess_output(wrapped, BOARDS[0])["status"] == PredictionStatus.INVALID_JSON
    nested = "```json\n" + wrapped + "\n```"
    assert (
        assess_output(nested, BOARDS[0], normalize=True)["status"]
        == PredictionStatus.INVALID_JSON
    )
    duplicate = '```json\n{"effects":[],"effects":[],"preserve":[]}\n```'
    assert (
        assess_output(duplicate, BOARDS[0], normalize=True)["normalized_json_valid"]
        is False
    )


def test_base_evaluation_uses_d_prompt_without_loading_an_adapter(
    tmp_path, monkeypatch
):
    import carl.icl.policy_relative.evaluation as module

    model = tmp_path / "model"
    model.mkdir()
    revision = "a" * 40
    (model / "experiment_revision.json").write_text(json.dumps({"revision": revision}))
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    board = ["#######", "#@-$--#", "#-.---#", "#######"]
    record = {
        "record_id": "one",
        "context_id": "one",
        "layout_id": canonical_layout_id(from_rows(board)),
        "split": "test",
        "family": "fixture",
        "start_observation": board,
        "task_goal": "Solve.",
        "later_observation": ["SECRET_FUTURE"],
        "fragment_actions": [123456],
    }
    contexts = tmp_path / "contexts.jsonl"
    write_rows(contexts, [record])
    calls = []

    def fake_initialise(_args, adapter=None, train=False):
        calls.append((adapter, train))
        return object(), FakeTokenizer(), object(), "cpu"

    def fake_generate(_torch, _model, _tokenizer, example, _args, _device, messages):
        assert messages == generator_messages(example, Arm.DEMONSTRATION)
        assert "SECRET_FUTURE" not in json.dumps(messages)
        return (
            '```json\n{"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[]}\n```',
            8,
            0.1,
        )

    monkeypatch.setattr(module, "initialise", fake_initialise)
    monkeypatch.setattr(module, "generate", fake_generate)
    output = tmp_path / "base.jsonl"
    args = Namespace(
        arm=Arm.BASE,
        model=str(model),
        revision=revision,
        data=str(contexts),
        adapter=None,
        output=str(output),
        requested_scale=None,
        dtype="bfloat16",
        seed=1729,
        max_length=512,
        max_new_tokens=128,
        require_cuda=False,
        stop_monotonic=None,
        stop_epoch=None,
    )
    evaluate(args)
    assert calls == [(None, False)]
    row = json.loads(output.read_text())
    metadata = json.loads(output.with_suffix(".run.json").read_text())
    assert row["status"] == "valid_intent" and row["fence_removed"] is True
    assert row["prompt"] == generator_messages(record, Arm.DEMONSTRATION)
    assert (
        metadata["adapter"]
        is metadata["adapter_weights_sha256"]
        is metadata["adapter_config_sha256"]
        is None
    )
    assert metadata["training_configuration_sha256"] is None
    assert (
        metadata["model_weights_sha256"]
        == hashlib.sha256((model / "model.safetensors").read_bytes()).hexdigest()
    )
    assert (
        metadata["model_config_sha256"]
        == hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    )
    assert (
        metadata["tokenizer_sha256"]
        == hashlib.sha256((model / "tokenizer.json").read_bytes()).hexdigest()
    )
    assert metadata["dtype"] == "bfloat16" and metadata["decoding"] == "greedy"
    args.adapter = "unexpected"
    with pytest.raises(ValueError, match="must not load an adapter"):
        evaluate(args)
