"""Synthetic contract tests; these do not load a model or access a GPU."""

import contextlib
import copy
import hashlib
import json
from argparse import Namespace
from collections import Counter

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.policy_relative_lora import MISSING_SCALE
from carl.icl.policy_relative_lora import Arm
from carl.icl.policy_relative_lora import PredictionStatus
from carl.icl.policy_relative_lora import _configuration
from carl.icl.policy_relative_lora import assess_output
from carl.icl.policy_relative_lora import balanced_samples
from carl.icl.policy_relative_lora import candidate_scale
from carl.icl.policy_relative_lora import encode_example
from carl.icl.policy_relative_lora import evaluate
from carl.icl.policy_relative_lora import generator_messages
from carl.icl.policy_relative_lora import load_fit_data
from carl.icl.policy_relative_lora import load_study_data
from carl.icl.policy_relative_lora import require_scale_information
from carl.icl.policy_relative_lora import resource_check
from carl.icl.policy_relative_lora import resource_encodings
from carl.icl.policy_relative_lora import resource_records
from carl.icl.policy_relative_lora import study_samples
from carl.icl.policy_relative_lora import test_contexts as load_test_contexts
from carl.icl.policy_relative_lora import training_prompts
from carl.icl.policy_relative_lora import validation_loss
from carl.icl.policy_relative_lora import verify_adapter

BOARDS = [
    ["######", "#@-$-#", "#---.#", "######"],
    ["#######", "#@-$--#", "#----.#", "#######"],
    ["######", "#@-$-#", "#-.--#", "######"],
]
CONTROLLER = "a" * 64


class FakeTokenizer:
    eos_token = "<end>"

    def __init__(self):
        self.messages = []
        self.answers = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        return [10, 11, 12]

    def encode(self, answer, **kwargs):
        self.answers.append(answer)
        assert kwargs["add_special_tokens"] is False
        return [20 + index for index in range(len(answer))]


def attempt(status, count):
    return {"status": status, "primitive_actions_attempted": count}


def measured(identifier, context, split, board, predicate, attempts, fraction, duration, u, weight, size):
    return {
        "record_id": identifier,
        "context_id": context,
        "layout_id": canonical_layout_id(from_rows(board)),
        "split": split,
        "family": "synthetic_unit_test",
        "start_observation": board,
        "task_goal": "Put boxes on goals.",
        "intent": {
            "effects": [predicate],
            "preserve": [],
            "instruction": None
        },
        "validation": {
            "accepted": True
        },
        "later_observation": ["SECRET_FUTURE"],
        "k": 987654321,
        "fragment_actions": [87654321],
        "source_path": "SECRET_PATH",
        "trajectory_id": "SECRET_TRAJECTORY",
        "policy_measurements": {
            "controller_hash": CONTROLLER,
            "action_cap": 8,
            "expected_candidates": size,
            "complete": True,
            "attempts": attempts,
            "success_fraction": fraction,
            "mean_success_actions": duration,
            "u": u,
            "weight": weight
        }
    }


def group(prefix, split, board):
    context = prefix + "-context"
    return [
        measured(prefix + "-fast", context, split, board, {
            "predicate": "box_at",
            "row": 1,
            "col": 4
        }, [attempt("achieved", 1), attempt("action_limit", 8)], .5, 1, .125, .25, 4),
        measured(prefix + "-tie-heavy", context, split, board, {
            "predicate": "box_at",
            "row": 2,
            "col": 1
        }, [attempt("achieved", 2), attempt("achieved", 2)], 1, 2, .625, .5, 4),
        measured(prefix + "-tie-light", context, split, board, {
            "predicate": "player_at",
            "row": 2,
            "col": 2
        }, [attempt("achieved", 2), attempt("invalid_response", 3)], .5, 2, .625, .25, 4),
        measured(prefix + "-failed", context, split, board, {
            "predicate": "box_at",
            "row": 2,
            "col": 3
        }, [attempt("action_limit", 8), attempt("action_limit", 8)], 0, None, None, None, 4),
    ]


@pytest.fixture
def dataset(tmp_path):
    rows = group("train-a", "train", BOARDS[0]) + group("train-b", "train", BOARDS[1])
    rows += group("validation", "validation", BOARDS[2])
    path = tmp_path / "calibration.jsonl"
    write_rows(path, rows)
    return path, rows


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_measured_weights_ties_and_failures_are_recomputed(dataset):
    path, _ = dataset
    data = load_fit_data(path)
    assert data.controller_hash == CONTROLLER
    assert data.action_cap == 8 and data.attempts_per_candidate == 2
    assert (len(data.train), len(data.validation)) == (2, 1)
    candidates = {item.record["record_id"]: item for item in data.train[0].candidates}
    assert set(candidates) == {"train-a-fast", "train-a-tie-heavy", "train-a-tie-light"}
    assert (candidates["train-a-fast"].weight, candidates["train-a-fast"].u) == (.25, .125)
    assert (candidates["train-a-tie-heavy"].weight, candidates["train-a-tie-heavy"].u) == (.5, .625)
    assert (candidates["train-a-tie-light"].weight, candidates["train-a-tie-light"].u) == (.25, .625)
    assert data.audit["splits"]["train"]["contexts_with_multiple_successes"] == 2


def test_prompt_is_whitelisted_and_supervised_loss_only_on_response(dataset):
    path, _ = dataset
    candidate = load_fit_data(path).train[0].candidates[0]
    first = generator_messages(candidate.record, Arm.SCALED, candidate.u)
    poisoned = copy.deepcopy(candidate.record)
    poisoned.update(record_id="OTHER_ID",
                    context_id="OTHER_CONTEXT",
                    layout_id="OTHER_LAYOUT",
                    k=42,
                    later_observation=["OTHER_FUTURE"],
                    fragment_actions=[123],
                    policy_measurements={"u": 0.999})
    assert generator_messages(poisoned, Arm.SCALED, candidate.u) == first
    prompt = json.dumps(first)
    for forbidden in ("SECRET_FUTURE", "SECRET_PATH", "SECRET_TRAJECTORY", "987654321", "87654321",
                      candidate.record["record_id"], candidate.record["context_id"], candidate.record["layout_id"]):
        assert forbidden not in prompt
    assert MISSING_SCALE in json.dumps(generator_messages(candidate.record, Arm.UNSCALED))
    assert "Requested relative scale: 0.125" in prompt
    with pytest.raises(ValueError, match="unscaled"):
        generator_messages(candidate.record, Arm.UNSCALED, .5)
    for invalid in (-.1, 1.1, float("nan"), True, None):
        with pytest.raises(ValueError):
            generator_messages(candidate.record, Arm.SCALED, invalid)

    for arm in (Arm.UNSCALED, Arm.SCALED):
        tokenizer = FakeTokenizer()
        encoded = encode_example(tokenizer, candidate, arm, 1024)
        assert encoded["labels"][:3] == [-100, -100, -100]
        assert encoded["labels"][3:] == encoded["input_ids"][3:]
        assert tokenizer.answers[0].endswith(tokenizer.eos_token)
        with pytest.raises(ValueError, match="no truncation"):
            encode_example(tokenizer, candidate, arm, 3)


def test_both_arms_use_identical_sample_order_and_context_balanced_weights(dataset):
    path, _ = dataset
    contexts = load_fit_data(path).train
    first = balanced_samples(contexts, 1729)
    second = balanced_samples(contexts, 1729)
    sampled = [next(first).record["record_id"] for _ in range(10000)]
    assert sampled == [next(second).record["record_id"] for _ in range(10000)]
    assert sum(identifier.startswith("train-a") for identifier in sampled) == 5000
    frequencies = Counter(sampled)
    assert abs(frequencies["train-a-fast"] / 5000 - .25) < .025
    assert abs(frequencies["train-a-tie-heavy"] / 5000 - .5) < .025
    assert abs(frequencies["train-a-tie-light"] / 5000 - .25) < .025


@pytest.mark.parametrize("change, message", [
    (lambda rows: rows[0]["policy_measurements"].update(u=.5), "u disagrees"),
    (lambda rows: rows[0]["policy_measurements"].update(weight=.5), "weight disagrees"),
    (lambda rows: rows[0]["policy_measurements"].update(success_fraction=1), "success_fraction disagrees"),
    (lambda rows: rows[0]["policy_measurements"].update(mean_success_actions=7), "mean_success_actions disagrees"),
    (lambda rows: rows[0]["policy_measurements"].update(expected_candidates=5), "expected candidate"),
    (lambda rows: rows[0]["policy_measurements"]["attempts"][0].update(status="api_error"), "infrastructure"),
    (lambda rows: rows[0]["policy_measurements"]["attempts"][1].update(primitive_actions_attempted=7), "action-limit"),
    (lambda rows: rows[3]["policy_measurements"].update(u=0), "without successes"),
    (lambda rows: rows[1].update(intent=copy.deepcopy(rows[0]["intent"])), "duplicate candidate"),
    (lambda rows: rows[0].update(layout_id="not-canonical"), "canonical"),
    (lambda rows: rows[0].update(split="test"), "only train and validation"),
    (lambda rows: rows[0]["validation"].update(accepted=False), "checked"),
])
def test_bad_or_incomplete_calibration_is_rejected(dataset, change, message):
    path, rows = dataset
    change(rows)
    write_rows(path, rows)
    with pytest.raises(ValueError, match=message):
        load_fit_data(path)


def test_incomplete_context_is_excluded_as_a_whole_and_audited(dataset):
    path, rows = dataset
    rows[0]["policy_measurements"].update(complete=False, attempts=None, success_fraction=None, u=None, weight=None)
    write_rows(path, rows)
    data = load_fit_data(path)
    assert [item.context_id for item in data.train] == ["train-b-context"]
    assert data.audit["rejected_context_count"] == 1
    assert data.audit["rejected_contexts"][0]["context_id"] == "train-a-context"
    assert data.audit["rejected_contexts"][0]["reason"] == "incomplete_measurements"


def test_missing_candidate_excludes_context_without_inferring_its_label(dataset):
    path, rows = dataset
    write_rows(path, rows[1:])
    data = load_fit_data(path)
    assert [item.context_id for item in data.train] == ["train-b-context"]
    assert data.audit["rejected_contexts"][0]["reason"] == "missing_candidates"


def test_no_complete_training_context_stops_with_rejected_ids(dataset):
    path, rows = dataset
    for row in rows:
        if row["split"] == "train":
            row["policy_measurements"] = None
    write_rows(path, rows)
    with pytest.raises(ValueError, match="complete context.*train-a-context.*train-b-context"):
        load_fit_data(path)


def test_all_tied_training_durations_do_not_masquerade_as_scale_supervision(dataset):
    path, rows = dataset
    for row in rows:
        if row["split"] != "train" or row["policy_measurements"]["success_fraction"] == 0:
            continue
        measurement = row["policy_measurements"]
        for attempt_row in measurement["attempts"]:
            if attempt_row["status"] == "achieved":
                attempt_row["primitive_actions_attempted"] = 2
        measurement.update(mean_success_actions=2, u=.5)
    write_rows(path, rows)
    data = load_fit_data(path)
    assert data.audit["splits"]["train"]["contexts_with_distinct_success_durations"] == 0
    require_scale_information(data, Arm.UNSCALED)
    with pytest.raises(ValueError, match="two distinct measured successful durations"):
        require_scale_information(data, Arm.SCALED)


def test_layout_and_visible_context_leakage_are_rejected(dataset):
    path, rows = dataset
    for row in rows[-4:]:
        row["start_observation"] = BOARDS[0]
        row["layout_id"] = canonical_layout_id(from_rows(BOARDS[0]))
    write_rows(path, rows)
    with pytest.raises(ValueError, match="visible context|layouts overlap"):
        load_fit_data(path)


def test_validation_averages_contexts_and_uses_relative_weights(dataset, monkeypatch):
    import carl.icl.policy_relative_lora as module

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
        module, "_loss", lambda _t, _m, _z, item, _a, _d: Scalar(
            next(value for suffix, value in losses.items() if item.record["record_id"].endswith(suffix))))
    model = Model()
    result = validation_loss(Torch(), model, None, data.validation, Namespace(arm=Arm.SCALED), "cpu")
    assert result == pytest.approx(.25 * 1 + .5 * 2 + .25 * 4)
    assert model.training is True


def test_paired_arm_metadata_and_response_target_match(dataset, tmp_path):
    path, _ = dataset
    data = load_fit_data(path)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    common = dict(model=str(model),
                  revision="a" * 40,
                  data=str(path),
                  seed=1729,
                  dtype="bfloat16",
                  rank=4,
                  accumulation=4,
                  lr=.0002,
                  max_steps=100,
                  eval_every=20,
                  max_length=512,
                  gradient_checkpointing=True)
    configs = [_configuration(Namespace(**common, arm=arm), data, FakeTokenizer(), "cuda")
               for arm in (Arm.UNSCALED, Arm.SCALED)]
    assert configs[0]["comparability_sha256"] == configs[1]["comparability_sha256"]
    assert configs[0]["data_sha256"] == configs[1]["data_sha256"]
    assert configs[0]["arm"] != configs[1]["arm"]
    smaller = _configuration(Namespace(**{**common, "rank": 2}, arm=Arm.SCALED), data, FakeTokenizer(), "cuda")
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
    assert [row["record_id"] for row in unscaled] == [row["record_id"] for row in scaled]
    assert [row["response_target"] for row in unscaled] == [row["response_target"] for row in scaled]
    assert all("SECRET_FUTURE" not in json.dumps(row["messages"]) for row in scaled)


def test_test_contexts_and_prediction_statuses_are_explicit(tmp_path):
    board = BOARDS[0]
    context = {
        "record_id": "test-id",
        "context_id": "test-context",
        "layout_id": canonical_layout_id(from_rows(board)),
        "split": "test",
        "family": "test-family",
        "start_observation": board,
        "task_goal": "Solve."
    }
    path = tmp_path / "test.jsonl"
    write_rows(path, [context])
    assert load_test_contexts(path) == [context]
    valid = '{"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[],"instruction":null}'
    assert assess_output(valid, board)["status"] == PredictionStatus.VALID
    assert assess_output("```json\n" + valid + "\n```", board)["status"] == PredictionStatus.INVALID_JSON
    assert assess_output('{"x":1,"x":2}', board)["status"] == PredictionStatus.INVALID_JSON
    assert assess_output('{"effects":[],"preserve":[]}', board)["status"] == PredictionStatus.INVALID_INTENT
    write_rows(path, [context, context])
    with pytest.raises(ValueError, match="unique"):
        load_test_contexts(path)


def test_adapter_is_validation_selected_and_never_tested_on_fitting_layouts(dataset, tmp_path):
    path, _ = dataset
    data = load_fit_data(path)
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    args = Namespace(model=str(model),
                     revision="a" * 40,
                     data=str(path),
                     seed=1729,
                     dtype="bfloat16",
                     rank=4,
                     accumulation=4,
                     lr=.0002,
                     max_steps=100,
                     eval_every=20,
                     max_length=512,
                     gradient_checkpointing=True,
                     arm=Arm.SCALED)
    training = _configuration(args, data, FakeTokenizer(), "cuda")
    run = tmp_path / "training"
    adapter = run / "checkpoint-20"
    adapter.mkdir(parents=True)
    (run / "configuration.json").write_text(json.dumps(training))
    (run / "best.json").write_text(json.dumps({"adapter": str(adapter)}))
    unseen = {"layout_id": "new-held-out-layout"}
    assert verify_adapter(adapter, model, args.revision, Arm.SCALED, [unseen]) == training
    with pytest.raises(ValueError, match="arm, model revision"):
        verify_adapter(adapter, model, args.revision, Arm.UNSCALED, [unseen])
    with pytest.raises(ValueError, match="overlap"):
        verify_adapter(adapter, model, args.revision, Arm.SCALED, [{"layout_id": data.train[0].layout_id}])
    (run / "best.json").write_text(json.dumps({"adapter": str(run / "checkpoint-40")}))
    with pytest.raises(ValueError, match="validation-selected checkpoint"):
        verify_adapter(adapter, model, args.revision, Arm.SCALED, [unseen])


def test_evaluation_uses_only_synthetic_stubs_and_retains_every_status(dataset, tmp_path, monkeypatch):
    import carl.icl.policy_relative_lora as module

    path, _ = dataset
    data = load_fit_data(path)
    model = tmp_path / "model"
    model.mkdir()
    revision = "a" * 40
    (model / "experiment_revision.json").write_text(json.dumps({"revision": revision}))
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    common = dict(model=str(model),
                  revision=revision,
                  data=str(path),
                  seed=1729,
                  dtype="bfloat16",
                  rank=4,
                  accumulation=4,
                  lr=.0002,
                  max_steps=100,
                  eval_every=20,
                  max_length=512,
                  max_new_tokens=128,
                  gradient_checkpointing=True,
                  arm=Arm.SCALED)
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
    contexts = [{
        "record_id": name,
        "context_id": name,
        "layout_id": layout,
        "split": "test",
        "family": "fixture",
        "start_observation": board,
        "task_goal": "Solve."
    } for name in ("valid", "invalid", "failure", "later")]
    test_file = tmp_path / "test.jsonl"
    write_rows(test_file, contexts)
    monkeypatch.setattr(module, "initialise", lambda *_args, **_kwargs: (object(), FakeTokenizer(), object(), "cpu"))
    valid = '{"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[],"instruction":null}'

    def fake_generate(_torch, _model, _tokenizer, record, _args, _device, messages):
        assert "Requested relative scale: 0.25" in json.dumps(messages)
        if record["record_id"] == "failure":
            raise RuntimeError("synthetic inference infrastructure error")
        return (valid if record["record_id"] == "valid" else "not-json"), 3, .1

    monkeypatch.setattr(module, "generate", fake_generate)
    output = tmp_path / "predictions.jsonl"
    args = Namespace(**{
        **common, "data": str(test_file)
    },
                     output=str(output),
                     requested_scale=.25,
                     adapter=str(adapter),
                     require_cuda=False,
                     stop_monotonic=None,
                     stop_epoch=None)
    with pytest.raises(RuntimeError, match="synthetic inference infrastructure error"):
        evaluate(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["valid_intent", "invalid_json", "infrastructure_error", "not_attempted"]
    assert rows[0]["parsed"]["effects"][0]["col"] == 4
    assert rows[1]["raw_output"] == "not-json" and rows[1]["parsed"] is None
    assert rows[2]["error"].endswith("synthetic inference infrastructure error")
    assert all("SECRET_FUTURE" not in json.dumps(row["prompt"]) for row in rows)


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
    assert all("Requested relative scale: 0.5" in json.dumps(messages) for messages in tokenizer.messages)
    assert all("SECRET_FUTURE" not in json.dumps(messages) for messages in tokenizer.messages)
    assert all("SECRET_TEST_WITHOUT_A_BOARD" not in json.dumps(item[2]) for item in prepared)
    assert all(item[1]["labels"][:3] == [-100] * 3 and item[2]["truncated"] is False for item in prepared)
    assert all(item[2]["complete_example_tokens"] == item[2]["prompt_tokens"] + item[2]["response_tokens"]
               for item in prepared)
    with pytest.raises(ValueError, match="no truncation"):
        resource_encodings(FakeTokenizer(), records, 4, 32)


def test_resource_check_full_flow_with_stubs_only(dataset, tmp_path, monkeypatch):
    import carl.icl.policy_relative_lora as module

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
            return .5

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
            assert len(list(parameters)) == 1 and lr == .0002

        def step(self):
            calls["optimizer_step"] += 1

        def zero_grad(self, set_to_none):
            assert set_to_none is True

    fake_torch = Namespace(optim=Namespace(AdamW=Optimiser))
    monkeypatch.setattr(module, "initialise", lambda *_args, **_kwargs:
                        (fake_torch, FakeTokenizer(), StubModel(), "stub"))
    monkeypatch.setattr(module, "tensor_example", lambda _torch, example, _device: example)
    monkeypatch.setattr(module, "sync_gpu", lambda _torch: None)
    monkeypatch.setattr(module, "memory", lambda _torch: {"gpu_peak_reserved_bytes": None})

    def fake_generate(_torch, _model, _tokenizer, record, _args, _device, messages):
        calls["generate"] += 1
        assert record["split"] in ("train", "validation")
        assert "Requested relative scale: 0.5" in json.dumps(messages)
        return "STUB_OUTPUT", 4, .5

    monkeypatch.setattr(module, "generate", fake_generate)
    output = tmp_path / "resource.json"
    args = Namespace(data=str(path),
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
                     lr=.0002,
                     require_cuda=False,
                     stop_monotonic=None,
                     stop_epoch=None)
    resource_check(args)
    report = json.loads(output.read_text())
    assert report["status"] == "complete"
    assert report["measurement_kind"] == "resource_only_no_luna_labels"
    assert report["synthetic_requested_scale"] == .5 and len(report["token_lengths"]) == 12
    assert calls == {"backward": 4, "optimizer_step": 1, "generate": 1}
    assert not list(tmp_path.rglob("adapter_model.safetensors"))


def test_study_arms_share_demonstrations_and_paired_weighted_samples(dataset, tmp_path):
    path, rows = dataset
    demonstrations = copy.deepcopy(rows)
    for row in demonstrations:
        row["policy_measurements"] = None
    demo_path = tmp_path / "demonstrations.jsonl"
    write_rows(demo_path, demonstrations)
    arms = (Arm.DEMONSTRATION, Arm.MEASURED, Arm.ABSOLUTE, Arm.RELATIVE)
    loaded = {arm: load_study_data(Namespace(arm=arm, data=str(path), demo_data=str(demo_path))) for arm in arms}
    schedules = {}
    for arm, (data, demo) in loaded.items():
        sampler = study_samples(data, demo, arm, 123)
        samples = [next(sampler) for _ in range(100)]
        schedules[arm] = samples
        assert all(not sample.measured for sample in samples[::2])
        assert all(sample.measured == (arm != Arm.DEMONSTRATION) for sample in samples[1::2])
    for arm in (Arm.ABSOLUTE, Arm.RELATIVE):
        assert [x.record["record_id"] for x in schedules[arm]] == [
            x.record["record_id"] for x in schedules[Arm.MEASURED]
        ]
    for arm in arms:
        assert [x.record["record_id"] for x in schedules[arm][::2]] == [
            x.record["record_id"] for x in schedules[Arm.DEMONSTRATION][::2]
        ]
    measured = schedules[Arm.MEASURED][1]
    assert candidate_scale(measured, Arm.MEASURED) is None
    assert candidate_scale(measured, Arm.ABSOLUTE) == measured.mean_success_actions
    assert candidate_scale(measured, Arm.RELATIVE) == measured.u
    assert candidate_scale(schedules[Arm.RELATIVE][0], Arm.RELATIVE) is None
    for arm in (Arm.ABSOLUTE, Arm.RELATIVE):
        assert MISSING_SCALE in json.dumps(generator_messages(schedules[arm][0].record, arm, None))
    absolute = generator_messages(measured.record, Arm.ABSOLUTE, measured.mean_success_actions)
    relative = generator_messages(measured.record, Arm.RELATIVE, measured.u)
    assert f"Requested duration in actions: {measured.mean_success_actions!r}" in absolute[1]["content"]
    assert f"Requested relative scale: {measured.u!r}" in relative[1]["content"]
    assert all(secret not in json.dumps(absolute + relative) for secret in (
        "SECRET_FUTURE", "SECRET_PATH", "SECRET_TRAJECTORY", "987654321", "87654321"
    ))


def test_study_rejects_incomplete_or_modified_common_pool(dataset, tmp_path):
    path, rows = dataset
    demo = copy.deepcopy(rows)
    for row in demo:
        row["policy_measurements"] = None
    demo_path = tmp_path / "demo.jsonl"
    write_rows(demo_path, demo[:-1])
    with pytest.raises(ValueError, match="every calibrated candidate"):
        load_study_data(Namespace(arm=Arm.MEASURED, data=str(path), demo_data=str(demo_path)))
    demo[-1]["intent"]["effects"][0]["col"] = 2
    write_rows(demo_path, demo)
    with pytest.raises(ValueError):
        load_study_data(Namespace(arm=Arm.MEASURED, data=str(path), demo_data=str(demo_path)))


def test_study_config_hashes_match_across_measured_arms(dataset, tmp_path):
    path, rows = dataset
    demo_path = tmp_path / "demo.jsonl"
    write_rows(demo_path, [{**row, "policy_measurements": None} for row in rows])
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    common = {"model": str(model), "revision": "a" * 40, "data": str(path), "demo_data": str(demo_path),
              "seed": 123, "dtype": "bfloat16", "rank": 4, "accumulation": 4, "lr": .0002,
              "max_steps": 100, "eval_every": 20, "max_length": 1024, "gradient_checkpointing": True}
    configs = []
    for arm in (Arm.MEASURED, Arm.ABSOLUTE, Arm.RELATIVE):
        args = Namespace(**common, arm=arm)
        data, demo = load_study_data(args)
        configs.append(_configuration(args, data, FakeTokenizer(), "cpu", demo))
    assert len({config["comparability_sha256"] for config in configs}) == 1
    assert len({config["demonstration_sha256"] for config in configs}) == 1
    assert all(config["demonstration_fraction"] == .5 for config in configs)
    assert all(config["controller_hash"] == CONTROLLER for config in configs)
    args = Namespace(**{**common, "data": None}, arm=Arm.DEMONSTRATION)
    data, demo = load_study_data(args)
    baseline = _configuration(args, data, FakeTokenizer(), "cpu", demo)
    assert baseline["controller_hash"] is None and baseline["data_sha256"] == baseline["demonstration_sha256"]
    assert baseline["train_layout_ids"] == configs[0]["train_layout_ids"]


def test_study_training_pipeline_matches_updates_and_samples(dataset, tmp_path, monkeypatch):
    import carl.icl.policy_relative_lora as module

    path, rows = dataset
    demo_path = tmp_path / "demo.jsonl"
    write_rows(demo_path, [{**row, "policy_measurements": None} for row in rows])
    model_path = tmp_path / "model"
    model_path.mkdir()
    revision = "a" * 40
    (model_path / "experiment_revision.json").write_text(json.dumps({"revision": revision}))
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
            assert len(list(parameters)) == 1 and lr == .0002

        def step(self):
            pass

        def zero_grad(self, set_to_none):
            assert set_to_none

    class Tokenizer(FakeTokenizer):
        def save_pretrained(self, path):
            path.mkdir()

    fake_torch = Namespace(optim=Namespace(AdamW=Optimizer), no_grad=contextlib.nullcontext)
    monkeypatch.setattr(module, "initialise", lambda *_args, **_kwargs:
                        (fake_torch, Tokenizer(), Model(), "cpu"))
    monkeypatch.setattr(module, "memory", lambda _torch: {})
    def loss(_torch, model, _tokenizer, candidate, _args, _device):
        if model.training:
            samples.append((candidate.record["record_id"], candidate.measured))
        else:
            validation_samples.append((candidate.record["record_id"], candidate.measured))
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
        args = Namespace(arm=arm, data=None if arm == Arm.DEMONSTRATION else str(path), demo_data=str(demo_path),
                         output=str(output), model=str(model_path), revision=revision, seed=123, threads=8,
                         dtype="bfloat16", rank=4, accumulation=2, lr=.0002, max_steps=2, eval_every=1,
                         max_length=1024, gradient_checkpointing=True, require_cuda=False, stop_monotonic=None,
                         stop_epoch=None, min_informative_train_contexts=1, min_informative_validation_contexts=1)
        module.train(args)
        schedules[arm] = list(samples)
        validation_schedules[arm] = list(validation_samples)
        status = json.loads((output / "status.json").read_text())
        assert status["status"] == "finished" and status["completed_steps"] == 2
        assert status["last_validation_step"] == 2
        assert (output / "best.json").is_file()
    assert schedules[Arm.MEASURED] == schedules[Arm.ABSOLUTE] == schedules[Arm.RELATIVE]
    assert [measured for _, measured in schedules[Arm.MEASURED]] == [False, True, False, True]
    assert all(not measured for _, measured in schedules[Arm.DEMONSTRATION])
    assert validation_schedules[Arm.MEASURED] == validation_schedules[Arm.ABSOLUTE] == validation_schedules[Arm.RELATIVE]
    assert all(len(schedule) == 4 for schedule in validation_schedules.values())


def test_symmetric_single_fence_normalization_retains_strict_validity():
    valid = '{"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[],"instruction":null}'
    wrapped = "```json\n" + valid + "\n```"
    result = assess_output(wrapped, BOARDS[0], normalize=True)
    assert result["status"] == PredictionStatus.VALID
    assert result["strict_json_valid"] is False and result["normalized_json_valid"] is True
    assert result["fence_removed"] is True
    assert assess_output(wrapped, BOARDS[0])["status"] == PredictionStatus.INVALID_JSON
    nested = "```json\n" + wrapped + "\n```"
    assert assess_output(nested, BOARDS[0], normalize=True)["status"] == PredictionStatus.INVALID_JSON
    duplicate = '```json\n{"effects":[],"effects":[],"preserve":[]}\n```'
    assert assess_output(duplicate, BOARDS[0], normalize=True)["normalized_json_valid"] is False


def test_base_evaluation_uses_d_prompt_without_loading_an_adapter(tmp_path, monkeypatch):
    import carl.icl.policy_relative_lora as module

    model = tmp_path / "model"
    model.mkdir()
    revision = "a" * 40
    (model / "experiment_revision.json").write_text(json.dumps({"revision": revision}))
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text(name)
    board = ["#######", "#@-$--#", "#-.---#", "#######"]
    record = {"record_id": "one", "context_id": "one", "layout_id": canonical_layout_id(from_rows(board)),
              "split": "test", "family": "fixture", "start_observation": board, "task_goal": "Solve.",
              "later_observation": ["SECRET_FUTURE"], "fragment_actions": [123456]}
    contexts = tmp_path / "contexts.jsonl"
    write_rows(contexts, [record])
    calls = []

    def fake_initialise(_args, adapter=None, train=False):
        calls.append((adapter, train))
        return object(), FakeTokenizer(), object(), "cpu"

    def fake_generate(_torch, _model, _tokenizer, example, _args, _device, messages):
        assert messages == generator_messages(example, Arm.DEMONSTRATION)
        assert "SECRET_FUTURE" not in json.dumps(messages)
        return '```json\n{"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[]}\n```', 8, .1

    monkeypatch.setattr(module, "initialise", fake_initialise)
    monkeypatch.setattr(module, "generate", fake_generate)
    output = tmp_path / "base.jsonl"
    args = Namespace(arm=Arm.BASE, model=str(model), revision=revision, data=str(contexts), adapter=None,
                     output=str(output), requested_scale=None, dtype="bfloat16", seed=1729, max_length=512,
                     max_new_tokens=128, require_cuda=False, stop_monotonic=None, stop_epoch=None)
    evaluate(args)
    assert calls == [(None, False)]
    row = json.loads(output.read_text())
    metadata = json.loads(output.with_suffix(".run.json").read_text())
    assert row["status"] == "valid_intent" and row["fence_removed"] is True
    assert row["prompt"] == generator_messages(record, Arm.DEMONSTRATION)
    assert metadata["adapter"] is metadata["adapter_weights_sha256"] is metadata["adapter_config_sha256"] is None
    assert metadata["training_configuration_sha256"] is None
    assert metadata["model_weights_sha256"] == hashlib.sha256((model / "model.safetensors").read_bytes()).hexdigest()
    assert metadata["model_config_sha256"] == hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    assert metadata["tokenizer_sha256"] == hashlib.sha256((model / "tokenizer.json").read_bytes()).hexdigest()
    assert metadata["dtype"] == "bfloat16" and metadata["decoding"] == "greedy"
    args.adapter = "unexpected"
    with pytest.raises(ValueError, match="must not load an adapter"):
        evaluate(args)
