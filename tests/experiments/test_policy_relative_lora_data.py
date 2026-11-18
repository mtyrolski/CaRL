"""Synthetic contract tests for policy-relative generators."""

import copy
import json
from argparse import Namespace
from collections import Counter

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.policy_relative_lora import MISSING_SCALE
from carl.icl.policy_relative_lora import Arm
from carl.icl.policy_relative_lora import balanced_samples
from carl.icl.policy_relative_lora import candidate_scale
from carl.icl.policy_relative_lora import encode_example
from carl.icl.policy_relative_lora import generator_messages
from carl.icl.policy_relative_lora import load_fit_data
from carl.icl.policy_relative_lora import load_study_data
from carl.icl.policy_relative_lora import require_scale_information
from carl.icl.policy_relative_lora import study_samples
from tests.experiments._policy_relative_lora_fixtures import BOARDS
from tests.experiments._policy_relative_lora_fixtures import CONTROLLER
from tests.experiments._policy_relative_lora_fixtures import FakeTokenizer
from tests.experiments._policy_relative_lora_fixtures import make_dataset
from tests.experiments._policy_relative_lora_fixtures import write_rows


@pytest.fixture(name="dataset")
def calibration_dataset(tmp_path):
    return make_dataset(tmp_path)


def test_measured_weights_ties_and_failures_are_recomputed(dataset):
    path, _ = dataset
    data = load_fit_data(path)
    assert data.controller_hash == CONTROLLER
    assert data.action_cap == 8 and data.attempts_per_candidate == 2
    assert (len(data.train), len(data.validation)) == (2, 1)
    candidates = {item.record["record_id"]: item for item in data.train[0].candidates}
    assert set(candidates) == {"train-a-fast", "train-a-tie-heavy", "train-a-tie-light"}
    assert (candidates["train-a-fast"].weight, candidates["train-a-fast"].u) == (
        0.25,
        0.125,
    )
    assert (
        candidates["train-a-tie-heavy"].weight,
        candidates["train-a-tie-heavy"].u,
    ) == (0.5, 0.625)
    assert (
        candidates["train-a-tie-light"].weight,
        candidates["train-a-tie-light"].u,
    ) == (0.25, 0.625)
    assert data.audit["splits"]["train"]["contexts_with_multiple_successes"] == 2


def test_prompt_is_whitelisted_and_supervised_loss_only_on_response(dataset):
    path, _ = dataset
    candidate = load_fit_data(path).train[0].candidates[0]
    first = generator_messages(candidate.record, Arm.SCALED, candidate.u)
    poisoned = copy.deepcopy(candidate.record)
    poisoned.update(
        record_id="OTHER_ID",
        context_id="OTHER_CONTEXT",
        layout_id="OTHER_LAYOUT",
        k=42,
        later_observation=["OTHER_FUTURE"],
        fragment_actions=[123],
        policy_measurements={"u": 0.999},
    )
    assert generator_messages(poisoned, Arm.SCALED, candidate.u) == first
    prompt = json.dumps(first)
    for forbidden in (
        "SECRET_FUTURE",
        "SECRET_PATH",
        "SECRET_TRAJECTORY",
        "987654321",
        "87654321",
        candidate.record["record_id"],
        candidate.record["context_id"],
        candidate.record["layout_id"],
    ):
        assert forbidden not in prompt
    assert MISSING_SCALE in json.dumps(
        generator_messages(candidate.record, Arm.UNSCALED)
    )
    assert "Requested relative scale: 0.125" in prompt
    with pytest.raises(ValueError, match="unscaled"):
        generator_messages(candidate.record, Arm.UNSCALED, 0.5)
    for invalid in (-0.1, 1.1, float("nan"), True, None):
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
    assert abs(frequencies["train-a-fast"] / 5000 - 0.25) < 0.025
    assert abs(frequencies["train-a-tie-heavy"] / 5000 - 0.5) < 0.025
    assert abs(frequencies["train-a-tie-light"] / 5000 - 0.25) < 0.025


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda rows: rows[0]["policy_measurements"].update(u=0.5), "u disagrees"),
        (
            lambda rows: rows[0]["policy_measurements"].update(weight=0.5),
            "weight disagrees",
        ),
        (
            lambda rows: rows[0]["policy_measurements"].update(success_fraction=1),
            "success_fraction disagrees",
        ),
        (
            lambda rows: rows[0]["policy_measurements"].update(mean_success_actions=7),
            "mean_success_actions disagrees",
        ),
        (
            lambda rows: rows[0]["policy_measurements"].update(expected_candidates=5),
            "expected candidate",
        ),
        (
            lambda rows: rows[0]["policy_measurements"]["attempts"][0].update(
                status="api_error"
            ),
            "infrastructure",
        ),
        (
            lambda rows: rows[0]["policy_measurements"]["attempts"][1].update(
                primitive_actions_attempted=7
            ),
            "action-limit",
        ),
        (lambda rows: rows[3]["policy_measurements"].update(u=0), "without successes"),
        (
            lambda rows: rows[1].update(intent=copy.deepcopy(rows[0]["intent"])),
            "duplicate candidate",
        ),
        (lambda rows: rows[0].update(layout_id="not-canonical"), "canonical"),
        (lambda rows: rows[0].update(split="test"), "only train and validation"),
        (lambda rows: rows[0]["validation"].update(accepted=False), "checked"),
    ],
)
def test_bad_or_incomplete_calibration_is_rejected(dataset, change, message):
    path, rows = dataset
    change(rows)
    write_rows(path, rows)
    with pytest.raises(ValueError, match=message):
        load_fit_data(path)


def test_incomplete_context_is_excluded_as_a_whole_and_audited(dataset):
    path, rows = dataset
    rows[0]["policy_measurements"].update(
        complete=False, attempts=None, success_fraction=None, u=None, weight=None
    )
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
    with pytest.raises(
        ValueError, match="complete context.*train-a-context.*train-b-context"
    ):
        load_fit_data(path)


def test_all_tied_training_durations_do_not_masquerade_as_scale_supervision(dataset):
    path, rows = dataset
    for row in rows:
        if (
            row["split"] != "train"
            or row["policy_measurements"]["success_fraction"] == 0
        ):
            continue
        measurement = row["policy_measurements"]
        for attempt_row in measurement["attempts"]:
            if attempt_row["status"] == "achieved":
                attempt_row["primitive_actions_attempted"] = 2
        measurement.update(mean_success_actions=2, u=0.5)
    write_rows(path, rows)
    data = load_fit_data(path)
    assert (
        data.audit["splits"]["train"]["contexts_with_distinct_success_durations"] == 0
    )
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


def test_study_arms_share_demonstrations_and_paired_weighted_samples(dataset, tmp_path):
    path, rows = dataset
    demonstrations = copy.deepcopy(rows)
    for row in demonstrations:
        row["policy_measurements"] = None
    demo_path = tmp_path / "demonstrations.jsonl"
    write_rows(demo_path, demonstrations)
    arms = (Arm.DEMONSTRATION, Arm.MEASURED, Arm.ABSOLUTE, Arm.RELATIVE)
    loaded = {
        arm: load_study_data(
            Namespace(arm=arm, data=str(path), demo_data=str(demo_path))
        )
        for arm in arms
    }
    schedules = {}
    for arm, (data, demo) in loaded.items():
        sampler = study_samples(data, demo, arm, 123)
        samples = [next(sampler) for _ in range(100)]
        schedules[arm] = samples
        assert all(not sample.measured for sample in samples[::2])
        assert all(
            sample.measured == (arm != Arm.DEMONSTRATION) for sample in samples[1::2]
        )
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
        assert MISSING_SCALE in json.dumps(
            generator_messages(schedules[arm][0].record, arm, None)
        )
    absolute = generator_messages(
        measured.record, Arm.ABSOLUTE, measured.mean_success_actions
    )
    relative = generator_messages(measured.record, Arm.RELATIVE, measured.u)
    assert (
        f"Requested duration in actions: {measured.mean_success_actions!r}"
        in absolute[1]["content"]
    )
    assert f"Requested relative scale: {measured.u!r}" in relative[1]["content"]
    assert all(
        secret not in json.dumps(absolute + relative)
        for secret in (
            "SECRET_FUTURE",
            "SECRET_PATH",
            "SECRET_TRAJECTORY",
            "987654321",
            "87654321",
        )
    )


def test_study_rejects_incomplete_or_modified_common_pool(dataset, tmp_path):
    path, rows = dataset
    demo = copy.deepcopy(rows)
    for row in demo:
        row["policy_measurements"] = None
    demo_path = tmp_path / "demo.jsonl"
    write_rows(demo_path, demo[:-1])
    with pytest.raises(ValueError, match="every calibrated candidate"):
        load_study_data(
            Namespace(arm=Arm.MEASURED, data=str(path), demo_data=str(demo_path))
        )
    demo[-1]["intent"]["effects"][0]["col"] = 2
    write_rows(demo_path, demo)
    with pytest.raises(ValueError):
        load_study_data(
            Namespace(arm=Arm.MEASURED, data=str(path), demo_data=str(demo_path))
        )
