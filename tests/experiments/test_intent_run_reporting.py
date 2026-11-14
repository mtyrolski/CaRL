import json

import pytest

from scripts.intent_experiment_supervisor import StageKind
from scripts.report_intent_generator_experiment import write_report
from scripts.run_intent_generator_experiment import evaluation_needs_resume
from scripts.run_intent_generator_experiment import expensive_work_allowed
from scripts.run_intent_generator_experiment import is_current_driver
from scripts.run_intent_generator_experiment import make_clock
from scripts.run_intent_generator_experiment import recovery_stage_name
from scripts.run_intent_generator_experiment import run_once
from scripts.run_intent_generator_experiment import training_steps
from scripts.run_intent_generator_experiment import wait_for_existing_stages


def test_resume_deadline_uses_original_clock_and_wall_time():
    clock = make_clock(1000.0, 2000.0, 16200)
    assert expensive_work_allowed(clock, 2000 + 13499, 1000 + 13499)
    assert not expensive_work_allowed(clock, 2000 + 13499, 1000 + 13500)
    assert not expensive_work_allowed(clock, 2000 + 13500, 1000 + 13499)
    with pytest.raises(ValueError):
        make_clock(1000.0, 2000.0, 16201)


def test_training_budget_uses_measured_speed_and_the_original_deadline():
    clock = make_clock(1000.0, 2000.0, 16200)
    assert training_steps(clock, {"complete_step_seconds": 2}, 300, 3000, 2000) == 300
    assert training_steps(clock, {"complete_step_seconds": 2}, 300, 16300, 15300) == 0
    assert training_steps(clock, {}, 300, 3000, 2000) == 0


def test_empty_or_missing_jsonl_with_run_config_requires_resume(tmp_path):
    output = tmp_path / "adapted.jsonl"
    assert not evaluation_needs_resume(output)
    output.with_suffix(".run.json").write_text("{}")
    assert evaluation_needs_resume(output)
    output.touch()
    assert evaluation_needs_resume(output)
    output.with_suffix(".run.json").unlink()
    # An older result without metadata still requests safe resume; the
    # generator will reject it rather than overwrite unverifiable output.
    assert evaluation_needs_resume(output)


def test_completed_stage_is_not_run_again_and_running_recovery_is_reused(tmp_path):
    import time
    state = {
        "status":
            "active",
        "clock":
            make_clock(time.time(), time.monotonic(), 16200),
        "stages": [{
            "name": "assessment",
            "status": "completed"
        }, {
            "name": "adapted_evaluation_resume_1",
            "status": "running"
        }]
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(state))
    assert run_once(tmp_path, "assessment", StageKind.EVALUATION, 10, ["false"])
    assert recovery_stage_name(tmp_path, "adapted_evaluation") == "adapted_evaluation_resume_1"
    assert recovery_stage_name(tmp_path, "assessment") == "assessment_resume_1"


def test_only_the_registered_current_group_is_skipped(monkeypatch):
    from scripts import run_intent_generator_experiment as runner

    monkeypatch.setattr(runner.os, "getpgrp", lambda: 123)
    monkeypatch.setattr(runner.supervisor, "owned_process", lambda stage: stage.get("proc_start") == 456)
    assert is_current_driver({"pid": 123, "proc_start": 456})
    assert not is_current_driver({"pid": 124, "proc_start": 456})
    assert not is_current_driver({"pid": 123, "proc_start": 999})


def test_resume_waits_for_group_members_after_the_leader_disappears(tmp_path, monkeypatch):
    import time

    from scripts import run_intent_generator_experiment as runner

    state = {
        "status": "active",
        "clock": make_clock(time.time(), time.monotonic(), 16200),
        "stages": [{
            "name": "training",
            "status": "running",
            "pid": 98765,
            "proc_start": 1
        }]
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(state))
    monkeypatch.setattr(runner.supervisor, "owned_process", lambda _stage: False)
    monkeypatch.setattr(runner.supervisor, "group_has_live_members", lambda group: group == 98765)
    monkeypatch.setattr(runner, "write_report", lambda _path: None)

    def stop_waiting(_seconds):
        raise TimeoutError("still active")

    monkeypatch.setattr(runner.time, "sleep", stop_waiting)
    with pytest.raises(TimeoutError, match="still active"):
        wait_for_existing_stages(tmp_path)
    assert json.loads((tmp_path / "run_manifest.json").read_text())["stages"][0]["status"] == "running"


def test_partial_report_preserves_watchdog_status_without_inventing_results(tmp_path):
    state = {"status": "active", "clock": make_clock(1000.0, 2000.0, 16200), "stages": []}
    (tmp_path / "run_manifest.json").write_text(json.dumps(state))
    (tmp_path / "REPORT.md"
    ).write_text("old\n<!-- supervisor status begin -->\nwatchdog snapshot\n<!-- supervisor status end -->\n")
    write_report(tmp_path)
    report = (tmp_path / "REPORT.md").read_text()
    assert "No completed training status is available" in report
    assert "The held-out assessment has not been completed" in report
    assert "watchdog snapshot" in report
    assert not list(tmp_path.glob(".REPORT.md.*.tmp"))
