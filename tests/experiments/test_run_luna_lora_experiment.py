"""The prepared launcher must not run a model unless explicitly requested."""

import json
import sys

import pytest

from carl.icl.intent_generator_experiment import file_sha256
from scripts import run_luna_lora_experiment as launcher


def test_default_mode_does_not_start_an_experiment(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["run_luna_lora_experiment"])
    monkeypatch.setattr(launcher, "read_json", lambda _: {})
    monkeypatch.setattr(launcher, "check_preparation", lambda *_: {"status": "prepared_not_started"})

    def unexpected(*_args) -> None:
        raise AssertionError("Default mode must not start a run or watchdog")

    monkeypatch.setattr(launcher, "run", unexpected)
    launcher.main()
    assert json.loads(capsys.readouterr().out)["status"] == "prepared_not_started"


def test_execute_requires_an_explicit_new_run_directory(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_luna_lora_experiment", "--execute"])
    monkeypatch.setattr(launcher, "read_json", lambda _: {})
    monkeypatch.setattr(launcher, "check_preparation", lambda *_: {"status": "prepared_not_started"})
    with pytest.raises(SystemExit):
        launcher.main()


def test_preflight_rejects_modified_calibration_data(tmp_path) -> None:
    data = tmp_path / "data"
    evaluation = tmp_path / "evaluation"
    data.mkdir()
    evaluation.mkdir()
    (data / "calibration_candidates.jsonl").write_text("{}\n")
    test_contexts = evaluation / "test_contexts.jsonl"
    test_contexts.write_text("{}\n")
    (evaluation / "manifest.json").write_text(json.dumps({"calibration_candidates_sha256": "wrong",
                                                           "test_contexts_sha256": file_sha256(test_contexts)}))
    with pytest.raises(ValueError, match="frozen evaluation manifest"):
        launcher.check_preparation({}, tmp_path)


def test_failed_resource_checks_stop_before_any_paid_stage(tmp_path, monkeypatch) -> None:
    config = launcher.read_json(launcher.CONFIG)
    run_dir = tmp_path / "run"
    stages = []

    def initialise(directory, _clock, *, report_mode) -> None:
        assert report_mode == "status_only"
        directory.mkdir()
        (directory / "logs").mkdir()

    def stage(_directory, name, *_args) -> bool:
        stages.append(name)
        return False

    monkeypatch.setattr(launcher.supervisor, "initialise", initialise)
    monkeypatch.setattr(launcher.supervisor, "finish", lambda *_: None)
    monkeypatch.setattr(launcher, "_snapshot", lambda *_: None)
    monkeypatch.setattr(launcher, "_gpu_free_bytes", lambda: 1_000_000)
    monkeypatch.setattr(launcher, "_stage", stage)
    launcher.run(config, tmp_path, run_dir, start_epoch=1.0, start_monotonic=1.0)
    status = json.loads((run_dir / "study_status.json").read_text())
    assert status["status"] == "partial"
    assert status["reason"] == "resource_check_failed_before_api"
    assert stages == ["resource_check", "resource_check_smaller", "render_report"]


def test_unverifiable_watchdog_blocks_stage_before_start(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(launcher, "_watchdog_live", lambda _: False)
    called = []
    monkeypatch.setattr(launcher.supervisor, "run_stage", lambda *_: called.append(True))
    result = launcher._stage(tmp_path, "calibration", launcher.supervisor.StageKind.EXPENSIVE, 30, ["unused"])
    assert not result
    assert not called
    assert "watchdog" in (tmp_path / "decisions.log").read_text()


def test_successful_exit_does_not_hide_incomplete_training() -> None:
    completed = {"status": "finished", "completed_steps": 100, "last_validation_step": 100,
                 "best_adapter": "checkpoint-80"}
    assert launcher._training_finished(completed, 100)
    for changes in ({"status": "stopped"}, {"completed_steps": 80}, {"last_validation_step": 80}):
        assert not launcher._training_finished({**completed, **changes}, 100)


def test_adapter_rank_is_passed_to_all_model_stages() -> None:
    config = launcher.read_json(launcher.CONFIG)
    config["generator"]["rank"] = 2
    options = launcher._options(config, launcher.ROOT / config["generator"]["model"])
    assert options[options.index("--rank") + 1] == "2"
