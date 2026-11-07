"""Check the sampling and statistics behind empirical Figure 1 profiles."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from carl.icl.boards import from_rows, level_key
from carl.icl.config import load_config
from carl.icl.datasets import dataset_signature, verify_solution

from scripts.figure1_reachability import (
    Dataset, PolicyModelOptions, Study, job_path, job_rows, prepare, run, sample_targets, saved_rows,
)
from scripts.figure1_report import label_targets, summarize_trials, write_report


def test_sample_targets_is_seeded_distinct_and_excludes_source() -> None:
    # The witness visits the source and another target twice. Those occurrences
    # must not receive extra sampling mass in the candidate population.
    states = [np.eye(7, dtype=np.float32)[value].reshape(1, 1, 7) for value in [0, 1, 0, 2, 1, 3, 4]]
    chosen = sample_targets(states, 0, window=5, count=3, seed=23)
    assert chosen == sample_targets(states, 0, window=5, count=3, seed=23)
    assert len(chosen) == 3
    assert all(1 <= offset <= 5 for offset in chosen)
    assert {int(states[offset].argmax()) for offset in chosen} == {1, 2, 3}


def test_sample_targets_uses_the_selected_position_and_clips_to_trajectory() -> None:
    states = [np.eye(7, dtype=np.float32)[value].reshape(1, 1, 7) for value in [0, 1, 2, 3]]
    assert sample_targets(states, 2, window=9, count=1, seed=17) == [1]


def test_summarize_trials_uses_only_hits_for_first_hit_mean() -> None:
    records = [
        {"exact_hit": True, "first_hit": 2, "solved_before_target": False},
        {"exact_hit": False, "first_hit": None, "solved_before_target": True},
        {"exact_hit": True, "first_hit": 6, "solved_before_target": False},
        {"exact_hit": False, "first_hit": None, "solved_before_target": False},
    ]
    result = summarize_trials(records, expected_trials=4)
    assert result["status"] == "complete"
    assert result["hits"] == 2
    assert result["p_hat"] == pytest.approx(0.5)
    assert result["d"] == pytest.approx(4.0)
    assert result["solved_before_target"] == 1


def test_completed_zero_hit_target_has_no_mean_hit_time() -> None:
    result = summarize_trials([
        {"exact_hit": False, "first_hit": None, "solved_before_target": False},
        {"exact_hit": False, "first_hit": None, "solved_before_target": False},
    ], expected_trials=2)
    assert result["status"] == "complete"
    assert result["hits"] == 0 and result["p_hat"] == 0
    assert result["d"] is None


@pytest.mark.parametrize("record", [
    {"exact_hit": True, "first_hit": 1, "solved_before_target": True, "actions_attempted": 1},
    {"exact_hit": True, "first_hit": 3, "solved_before_target": False, "actions_attempted": 1},
    {"exact_hit": True, "first_hit": 1, "solved_before_target": False, "actions_attempted": 2},
    {"exact_hit": False, "first_hit": None, "solved_before_target": False, "actions_attempted": -1},
])
def test_inconsistent_trial_cannot_enter_success_denominator(record: dict) -> None:
    result = summarize_trials([record], expected_trials=1)
    assert result["status"] == "incomplete"
    assert result["p_hat"] is None and result["d"] is None


@pytest.mark.parametrize("records", [
    [],
    [{"exact_hit": False, "first_hit": None, "solved_before_target": False}],
    [{"error": "request failed"}, {"exact_hit": True, "first_hit": 1, "solved_before_target": False}],
])
def test_missing_or_failed_trial_is_incomplete_not_a_policy_failure(records: list[dict]) -> None:
    result = summarize_trials(records, expected_trials=2)
    assert result["status"] == "incomplete"
    assert result["p_hat"] is None
    assert result["d"] is None


def test_label_targets_weights_hits_and_gives_ties_the_same_midpoint() -> None:
    rows = [
        {"status": "complete", "hits": 1, "p_hat": 0.25, "d": 2.0},
        {"status": "complete", "hits": 2, "p_hat": 0.50, "d": 4.0},
        {"status": "complete", "hits": 1, "p_hat": 0.25, "d": 4.0},
        {"status": "complete", "hits": 0, "p_hat": 0.00, "d": None},
    ]
    result = label_targets(rows)
    assert [row["weight"] for row in result[:3]] == pytest.approx([0.25, 0.5, 0.25])
    assert [row["u"] for row in result[:3]] == pytest.approx([0.125, 0.625, 0.625])
    assert result[3]["weight"] is None and result[3]["u"] is None


def test_incomplete_context_never_yields_partial_percentile_labels() -> None:
    rows = [
        {"status": "complete", "hits": 1, "p_hat": 0.5, "d": 3.0},
        {"status": "incomplete", "hits": 0, "p_hat": None, "d": None},
    ]
    result = label_targets(rows)
    assert all(row["weight"] is None and row["u"] is None for row in result)


def test_incomplete_context_does_not_prevent_another_model_being_labelled() -> None:
    rows = [
        {"model": "A", "context_id": "same", "status": "complete", "hits": 1, "p_hat": 0.5, "d": 3.0},
        {"model": "A", "context_id": "same", "status": "incomplete", "hits": 0, "p_hat": None, "d": None},
        {"model": "B", "context_id": "same", "status": "complete", "hits": 2, "p_hat": 1.0, "d": 4.0},
    ]
    result = label_targets(rows)
    assert result[0]["u"] is None and result[1]["u"] is None
    assert result[2]["weight"] == pytest.approx(1.0)
    assert result[2]["u"] == pytest.approx(0.5)


def test_report_keeps_unfinished_targets_visible_and_does_not_label_context(tmp_path) -> None:
    common = {"model": "model-a", "context_id": "c1"}
    records = [
        {**common, "target_hash": "a", "k": 1, "trial": 0, "status": "complete",
         "exact_hit": True, "first_hit": 1, "solved_before_target": False},
        {**common, "target_hash": "b", "k": 2, "trial": 0, "status": "missing"},
    ]
    targets = write_report(records, tmp_path, expected_trials=1)
    assert len(targets) == 2
    assert targets[0]["status"] == "complete" and targets[1]["status"] == "incomplete"
    assert all(row["u"] is None for row in targets)
    assert (tmp_path / "targets.csv").exists()
    assert (tmp_path / "reachability.pdf").exists()
    assert (tmp_path / "percentile.svg").exists()


def test_report_labels_mock_provenance_in_exported_tables_and_figure(tmp_path) -> None:
    import csv

    records = [{"model": "gpt-6-luna", "context_id": "c1", "target_hash": "a", "k": 1,
                "trial": 0, "status": "complete", "exact_hit": True, "first_hit": 1,
                "actions_attempted": 1, "solved_before_target": False}]
    write_report(records, tmp_path, expected_trials=1, provider="mock")
    with (tmp_path / "results.csv").open(newline="") as stream:
        assert next(csv.DictReader(stream))["provider"] == "mock"
    with (tmp_path / "targets.csv").open(newline="") as stream:
        assert next(csv.DictReader(stream))["provider"] == "mock"
    assert "[MOCK]" in (tmp_path / "reachability.svg").read_text()
    with pytest.raises(ValueError, match="provider"):
        write_report([{**records[0], "provider": "openai"}], tmp_path, 1, provider="mock")


@pytest.fixture
def synthetic_dataset(tmp_path):
    """One replayable validation puzzle and separate public training examples."""
    assets = Path("configs/icl/checkpoints/launch100/assets/data")
    root = tmp_path / "data"
    root.mkdir()
    training = json.loads((assets / "training.json").read_text())
    training["provenance"]["signature"] = dataset_signature(load_config("configs/icl/experiments/tuned_fixed.json"))
    training_path = root / "training.json"
    training_path.write_text(json.dumps(training))
    values = json.loads((assets / "value_examples.json").read_text())
    values["training_sha256"] = hashlib.sha256(training_path.read_bytes()).hexdigest()
    (root / "value_examples.json").write_text(json.dumps(values))

    rows = ["#######", "#-----#", "#-@---#", "#--$.-#", "#-----#", "#-----#", "#######"]
    actions = [0, 1, 1, 0, 1, 3]
    verify_solution(from_rows(rows), actions)
    public = root / "fixture"
    public.mkdir()
    (public / "validation.json").write_text(json.dumps({"instances": [
        {"id": "puzzle", "state": rows, "level_key": level_key(from_rows(rows))}
    ]}))
    (public / "test.json").write_text(json.dumps({"instances": []}))
    private = root / "private_witnesses" / "fixture"
    private.mkdir(parents=True)
    (private / "validation.json").write_text(json.dumps({"witnesses": [
        {"id": "puzzle", "actions": actions}
    ]}))
    return root


def test_frozen_study_runs_offline_resumes_and_rejects_changed_config(tmp_path, monkeypatch, synthetic_dataset) -> None:
    from carl.icl.api import CachedOpenAI
    from carl.icl.experiment_api import tiktoken

    class DummyEncoding:
        # The test never estimates real costs; it exercises only wiring.
        def encode(self, _text, **_kwargs):
            return [0]

    def no_network(*_args, **_kwargs):
        pytest.fail("The mock experiment must not create an API client")

    monkeypatch.setattr(tiktoken, "get_encoding", lambda _name: DummyEncoding())
    monkeypatch.setattr(CachedOpenAI, "_client", no_network)
    study = Study(
        experiment_config="configs/icl/experiments/tuned_fixed.json",
        output_dir=str(tmp_path / "study"), cache_dir=str(tmp_path / "cache"),
        models=["mock-a", "mock-b"],
        model_overrides={"mock-b": PolicyModelOptions(policy_reasoning_effort="medium",
                                                    policy_response_format="text", policy_tokens=64)},
        datasets=[Dataset(name="real", data_dir=str(synthetic_dataset), family="fixture",
                          split="validation", trajectories=1, contexts_per_trajectory=1)],
        window=4, targets_per_context=2, trials=2, action_cap=2, seed=23,
    )
    manifest = prepare(study, "mock")
    assert manifest["resolved_config"]["real"]["mock-b"]["policy_reasoning_effort"] == "medium"
    assert manifest["resolved_config"]["real"]["mock-b"]["policy_response_format"] == "text"
    assert manifest["resolved_config"]["real"]["mock-b"]["policy_tokens"] == 64
    assert manifest == prepare(study, "mock")
    rows = job_rows(manifest)
    assert len(rows) == 8
    assert len({row["context_id"] for row in rows}) == 1
    assert len({row["target_hash"] for row in rows}) == 2
    assert all(row["status"] == "missing" for row in saved_rows(manifest, tmp_path / "study"))
    run(study, "mock")
    completed = saved_rows(manifest, tmp_path / "study")
    assert len(completed) == 8 and all(row["status"] == "complete" for row in completed)
    assert all(row["actions_attempted"] <= 2 for row in completed)
    assert {row["model"] for row in completed} == {"mock-a", "mock-b"}
    assert len({row["policy_fingerprint"] for row in completed}) == 2
    pairs = {(row["target_id"], row["trial"], row["seed"]) for row in completed if row["model"] == "mock-a"}
    assert pairs == {(row["target_id"], row["trial"], row["seed"]) for row in completed if row["model"] == "mock-b"}
    recorded = {job_path(tmp_path / "study", row): job_path(tmp_path / "study", row).read_bytes()
                for row in completed}
    with pytest.raises(ValueError, match="resume"):
        run(study, "mock")
    run(study, "mock", resume=True)
    assert all(path.read_bytes() == content for path, content in recorded.items())
    missing = next(iter(recorded))
    missing.unlink()
    run(study, "mock", resume=True)
    assert missing.exists()
    assert all(path.read_bytes() == content for path, content in recorded.items() if path != missing)
    assert all(row["status"] == "complete" for row in saved_rows(manifest, tmp_path / "study"))
    study.seed += 1
    with pytest.raises(ValueError, match="Manifest differs"):
        run(study, "mock", resume=True)
    with pytest.raises(ValueError, match="Freeze differs"):
        prepare(study, "mock")


def test_saved_trial_cannot_change_its_frozen_identity(tmp_path, synthetic_dataset) -> None:
    from carl.icl.api import write_json

    study = Study(
        experiment_config="configs/icl/experiments/tuned_fixed.json",
        output_dir=str(tmp_path / "study"), cache_dir=str(tmp_path / "cache"), models=["mock-a"],
        datasets=[Dataset(name="real", data_dir=str(synthetic_dataset), family="fixture",
                          split="validation", trajectories=1)],
        window=2, targets_per_context=1, trials=1, action_cap=2, seed=11,
    )
    manifest = prepare(study, "mock")
    original = job_rows(manifest)[0]
    write_json(job_path(tmp_path / "study", original), {**original, "k": original["k"] + 1})
    with pytest.raises(ValueError, match="Trial does not match freeze"):
        saved_rows(manifest, tmp_path / "study")


def test_resume_after_error_preserves_history_and_counts_completed_trial(tmp_path, monkeypatch, synthetic_dataset) -> None:
    from carl.icl.api import CachedOpenAI, RequestFailed
    from scripts import figure1_reachability as runner

    def no_network(*_args, **_kwargs):
        pytest.fail("The mock experiment must not create an API client")

    def fail_rollout(*_args, **_kwargs):
        raise RequestFailed("synthetic interrupted request")

    monkeypatch.setattr(CachedOpenAI, "_client", no_network)
    study = Study(
        experiment_config="configs/icl/experiments/tuned_fixed.json",
        output_dir=str(tmp_path / "study"), cache_dir=str(tmp_path / "cache"), models=["mock-a"],
        datasets=[Dataset(name="real", data_dir=str(synthetic_dataset), family="fixture",
                          split="validation", trajectories=1)],
        window=2, targets_per_context=1, trials=1, action_cap=2, seed=11,
    )
    manifest = prepare(study, "mock")
    with monkeypatch.context() as patch:
        patch.setattr(runner, "rollout", fail_rollout)
        with pytest.raises(RequestFailed, match="synthetic interrupted request"):
            run(study, "mock")
    failed = saved_rows(manifest, tmp_path / "study")[0]
    assert failed["status"] == "error"
    assert failed["error"] == "synthetic interrupted request"
    run(study, "mock", resume=True)
    completed = saved_rows(manifest, tmp_path / "study")[0]
    assert completed["status"] == "complete"
    assert "error" not in completed and "error_type" not in completed
    assert completed["previous_errors"][0]["error"] == "synthetic interrupted request"
    assert completed["previous_errors"][0]["error_type"] == "RequestFailed"
    assert summarize_trials([completed], expected_trials=1)["status"] == "complete"
    assert summarize_trials([completed], expected_trials=1)["p_hat"] in (0.0, 1.0)
