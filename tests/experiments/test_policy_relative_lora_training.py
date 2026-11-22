"""Synthetic contract tests for policy-relative generators."""

import contextlib
import json
from argparse import Namespace

import pytest

from carl.icl.policy_relative_lora import Arm
from carl.icl.policy_relative_lora import _configuration
from carl.icl.policy_relative_lora import encode_example
from carl.icl.policy_relative_lora import load_fit_data
from carl.icl.policy_relative_lora import load_study_data
from carl.icl.policy_relative_lora import training_prompts
from carl.icl.policy_relative_lora import validation_loss
from tests.experiments._policy_relative_lora_fixtures import CONTROLLER
from tests.experiments._policy_relative_lora_fixtures import FakeTokenizer
from tests.experiments._policy_relative_lora_fixtures import make_dataset
from tests.experiments._policy_relative_lora_fixtures import write_rows


@pytest.fixture(name="dataset")
def calibration_dataset(tmp_path):
    return make_dataset(tmp_path)


def test_validation_averages_contexts_and_uses_relative_weights(dataset, monkeypatch):
    import carl.icl.policy_relative.training as module

    path, _ = dataset
    data = load_fit_data(path)

    class Scalar:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self.value

    class Torch:
        @staticmethod
        def no_grad():
            return contextlib.nullcontext()

    class Model:
        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    losses = {"fast": 1, "tie-heavy": 2, "tie-light": 4}
    monkeypatch.setattr(
        module,
        "_loss",
        lambda _t, _m, _z, item, _a, _d: Scalar(
            next(
                value
                for suffix, value in losses.items()
                if item.record["record_id"].endswith(suffix)
            )
        ),
    )
    model = Model()
    result = validation_loss(
        Torch(), model, None, data.validation, Namespace(arm=Arm.SCALED), "cpu"
    )
    assert result == pytest.approx(0.25 * 1 + 0.5 * 2 + 0.25 * 4)
    assert model.training is True


def test_paired_arm_metadata_and_response_target_match(dataset, tmp_path):
    path, _ = dataset
    data = load_fit_data(path)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    common = dict(
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
    )
    configs = [
        _configuration(Namespace(**common, arm=arm), data, FakeTokenizer(), "cuda")
        for arm in (Arm.UNSCALED, Arm.SCALED)
    ]
    assert configs[0]["comparability_sha256"] == configs[1]["comparability_sha256"]
    assert configs[0]["data_sha256"] == configs[1]["data_sha256"]
    assert configs[0]["arm"] != configs[1]["arm"]
    smaller = _configuration(
        Namespace(**{**common, "rank": 2}, arm=Arm.SCALED),
        data,
        FakeTokenizer(),
        "cuda",
    )
    assert smaller["rank"] == 2
    assert smaller["comparability_sha256"] != configs[1]["comparability_sha256"]
    candidate = data.train[0].candidates[0]
    tok1, tok2 = FakeTokenizer(), FakeTokenizer()
    encode_example(tok1, candidate, Arm.UNSCALED, 1024)
    encode_example(tok2, candidate, Arm.SCALED, 1024)
    assert tok1.answers == tok2.answers
    assert tok1.messages != tok2.messages
    unscaled = list(training_prompts(data, Arm.UNSCALED))
    scaled = list(training_prompts(data, Arm.SCALED))
    assert [row["record_id"] for row in unscaled] == [
        row["record_id"] for row in scaled
    ]
    assert [row["response_target"] for row in unscaled] == [
        row["response_target"] for row in scaled
    ]
    assert all("SECRET_FUTURE" not in json.dumps(row["messages"]) for row in scaled)


def test_study_config_hashes_match_across_measured_arms(dataset, tmp_path):
    path, rows = dataset
    demo_path = tmp_path / "demo.jsonl"
    write_rows(demo_path, [{**row, "policy_measurements": None} for row in rows])
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    common = {
        "model": str(model),
        "revision": "a" * 40,
        "data": str(path),
        "demo_data": str(demo_path),
        "seed": 123,
        "dtype": "bfloat16",
        "rank": 4,
        "accumulation": 4,
        "lr": 0.0002,
        "max_steps": 100,
        "eval_every": 20,
        "max_length": 1024,
        "gradient_checkpointing": True,
    }
    configs = []
    for arm in (Arm.MEASURED, Arm.ABSOLUTE, Arm.RELATIVE):
        args = Namespace(**common, arm=arm)
        data, demo = load_study_data(args)
        configs.append(_configuration(args, data, FakeTokenizer(), "cpu", demo))
    assert len({config["comparability_sha256"] for config in configs}) == 1
    assert len({config["demonstration_sha256"] for config in configs}) == 1
    assert all(config["demonstration_fraction"] == 0.5 for config in configs)
    assert all(config["controller_hash"] == CONTROLLER for config in configs)
    args = Namespace(**{**common, "data": None}, arm=Arm.DEMONSTRATION)
    data, demo = load_study_data(args)
    baseline = _configuration(args, data, FakeTokenizer(), "cpu", demo)
    assert (
        baseline["controller_hash"] is None
        and baseline["data_sha256"] == baseline["demonstration_sha256"]
    )
    assert baseline["train_layout_ids"] == configs[0]["train_layout_ids"]


def test_study_training_pipeline_matches_updates_and_samples(
    dataset, tmp_path, monkeypatch
):
    import carl.icl.policy_relative.training as module

    path, rows = dataset
    demo_path = tmp_path / "demo.jsonl"
    write_rows(demo_path, [{**row, "policy_measurements": None} for row in rows])
    model_path = tmp_path / "model"
    model_path.mkdir()
    revision = "a" * 40
    (model_path / "experiment_revision.json").write_text(
        json.dumps({"revision": revision})
    )
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model_path / name).write_text(name)
    samples = []
    validation_samples = []

    class Loss:
        def __truediv__(self, _divisor):
            return self

        def detach(self):
            return self

        def cpu(self):
            return 1.0

        def backward(self):
            pass

    class Model:
        training = True

        def parameters(self):
            return [Namespace(requires_grad=True)]

        def train(self):
            self.training = True

        def eval(self):
            self.training = False

    class Optimizer:
        def __init__(self, parameters, lr):
            assert len(list(parameters)) == 1 and lr == 0.0002

        def step(self):
            pass

        def zero_grad(self, set_to_none):
            assert set_to_none

    class Tokenizer(FakeTokenizer):
        def save_pretrained(self, path):
            path.mkdir()

    fake_torch = Namespace(
        optim=Namespace(AdamW=Optimizer), no_grad=contextlib.nullcontext
    )
    monkeypatch.setattr(
        module,
        "initialise",
        lambda *_args, **_kwargs: (fake_torch, Tokenizer(), Model(), "cpu"),
    )
    monkeypatch.setattr(module, "memory", lambda _torch: {})

    def loss(_torch, model, _tokenizer, candidate, _args, _device):
        if model.training:
            samples.append((candidate.record["record_id"], candidate.measured))
        else:
            validation_samples.append(
                (candidate.record["record_id"], candidate.measured)
            )
        return Loss()

    monkeypatch.setattr(module, "_loss", loss)

    def checkpoint(_model, output, name):
        location = output / name
        location.mkdir()
        return location

    monkeypatch.setattr(module, "save_adapter", checkpoint)
    schedules = {}
    validation_schedules = {}
    for arm in (Arm.DEMONSTRATION, Arm.MEASURED, Arm.ABSOLUTE, Arm.RELATIVE):
        samples.clear()
        validation_samples.clear()
        output = tmp_path / arm
        args = Namespace(
            arm=arm,
            data=None if arm == Arm.DEMONSTRATION else str(path),
            demo_data=str(demo_path),
            output=str(output),
            model=str(model_path),
            revision=revision,
            seed=123,
            threads=8,
            dtype="bfloat16",
            rank=4,
            accumulation=2,
            lr=0.0002,
            max_steps=2,
            eval_every=1,
            max_length=1024,
            gradient_checkpointing=True,
            require_cuda=False,
            stop_monotonic=None,
            stop_epoch=None,
            min_informative_train_contexts=1,
            min_informative_validation_contexts=1,
        )
        module.train(args)
        schedules[arm] = list(samples)
        validation_schedules[arm] = list(validation_samples)
        status = json.loads((output / "status.json").read_text())
        assert status["status"] == "finished" and status["completed_steps"] == 2
        assert status["last_validation_step"] == 2
        assert (output / "best.json").is_file()
    assert schedules[Arm.MEASURED] == schedules[Arm.ABSOLUTE] == schedules[Arm.RELATIVE]
    assert [measured for _, measured in schedules[Arm.MEASURED]] == [
        False,
        True,
        False,
        True,
    ]
    assert all(not measured for _, measured in schedules[Arm.DEMONSTRATION])
    assert (
        validation_schedules[Arm.MEASURED]
        == validation_schedules[Arm.ABSOLUTE]
        == validation_schedules[Arm.RELATIVE]
    )
    assert all(len(schedule) == 4 for schedule in validation_schedules.values())
