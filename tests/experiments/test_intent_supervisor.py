import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts import intent_experiment_supervisor as supervisor


def state():
    return {
        "status": "active",
        "clock": {"deadline_monotonic": 270.0},
        "expensive_start_cutoff_monotonic": 225.0,
        "training_stop_monotonic": 240.0,
    }


def test_cutoffs_do_not_restart_the_clock():
    assert supervisor.compute_stop(state(), "training", 100, 220) == 240
    assert supervisor.compute_stop(state(), "evaluation", 100, 250) == 270
    with pytest.raises(ValueError, match="Too late"):
        supervisor.compute_stop(state(), "training", 10, 225)
    with pytest.raises(ValueError, match="deadline"):
        supervisor.compute_stop(state(), "evaluation", 10, 270)


def test_absolute_deadline_still_applies_after_system_suspend():
    item = state()
    item["clock"].update({"start_epoch": 1000, "deadline_epoch": 17200})
    with pytest.raises(ValueError, match="deadline"):
        supervisor.compute_stop(item, "evaluation", 10, 200, epoch=17200)
    with pytest.raises(ValueError, match="Too late"):
        supervisor.compute_stop(item, "training", 10, 200, epoch=14500)
    assert supervisor.compute_stop(item, "evaluation", 100, 200, epoch=17150) == 250


def test_custom_stage_cutoffs_are_saved_and_enforced(tmp_path, monkeypatch):
    clock = tmp_path / "clock.json"
    clock.write_text(
        json.dumps(
            {
                "start_monotonic": 2000,
                "start_epoch": 1000,
                "deadline_monotonic": 2050,
                "deadline_epoch": 1050,
                "deadline_utc": "test",
            }
        )
    )
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: SimpleNamespace(pid=12345),
    )
    monkeypatch.setattr(supervisor, "proc_start", lambda _pid: 789)
    run = tmp_path / "run"

    supervisor.initialise(run, clock, expensive_start_after=12, training_stop_after=20)

    recorded = supervisor.read_json(run / "run_manifest.json")
    assert recorded["expensive_start_cutoff_monotonic"] == 2012
    assert recorded["expensive_start_cutoff_epoch"] == 1012
    assert recorded["training_stop_monotonic"] == 2020
    assert recorded["training_stop_epoch"] == 1020
    assert supervisor.compute_stop(recorded, "training", 40, 2005, epoch=1005) == 2020
    with pytest.raises(ValueError, match="Too late"):
        supervisor.compute_stop(recorded, "training", 40, 2005, epoch=1013)


@pytest.mark.parametrize("registry", ["workers.jsonl", "terra_workers.jsonl"])
def test_worker_registry_tracks_only_unfinished_own_groups(tmp_path, registry):
    rows = [
        {
            "phase": "started",
            "pid": 123,
            "pgid": 123,
            "proc_start": 456,
            "stop_monotonic": 789,
        },
        {"phase": "finished", "pid": 123, "proc_start": 456},
        {
            "phase": "started",
            "pid": 124,
            "pgid": 9,
            "proc_start": 457,
            "stop_monotonic": 789,
        },
        {
            "phase": "started",
            "pid": 125,
            "pgid": 125,
            "proc_start": 458,
            "stop_monotonic": 790,
        },
    ]
    (tmp_path / registry).write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert [row["pid"] for row in supervisor.registered_workers(tmp_path)] == [125]


def test_a_finished_legacy_record_cannot_hide_an_active_registered_worker(tmp_path):
    start = {
        "phase": "started",
        "pid": 125,
        "pgid": 125,
        "proc_start": 458,
        "stop_monotonic": 790,
    }
    finish = {"phase": "finished", "pid": 125, "proc_start": 458}
    (tmp_path / "workers.jsonl").write_text(json.dumps(start) + "\n")
    (tmp_path / "terra_workers.jsonl").write_text(json.dumps(finish) + "\n")

    assert supervisor.registered_workers(tmp_path) == [start]


def test_status_only_never_calls_an_explicit_hook(tmp_path):
    current = {
        "status": "active",
        "clock": {"deadline_utc": "test"},
        "stages": [],
        "report_mode": "status_only",
        "report_hook": "missing:report",
    }
    supervisor.update_report(tmp_path, current)
    before = (tmp_path / "REPORT.md").read_text()
    supervisor.refresh_results_report(tmp_path, current)
    assert not (tmp_path / "report_hook.log").exists()
    assert (tmp_path / "REPORT.md").read_text() == before
    assert "automatic checkpoint" in before


def test_manifests_without_a_hook_never_load_an_implicit_reporter(tmp_path):
    current = {"status": "active"}
    supervisor.refresh_results_report(tmp_path, current)
    supervisor.refresh_results_report(tmp_path, {**current, "report_mode": "legacy"})
    assert not (tmp_path / "report_hook.log").exists()


def test_unknown_report_mode_fails_closed(tmp_path):
    supervisor.refresh_results_report(
        tmp_path, {"report_mode": "unknown", "report_hook": "missing:report"}
    )
    assert not (tmp_path / "report_hook.log").exists()
    assert "Unknown report mode" in (tmp_path / "decisions.log").read_text()


def test_explicit_hook_receives_run_and_state(tmp_path, monkeypatch):
    (tmp_path / "reporter.py").write_text(
        "import json\n"
        "def write(run_dir, state):\n"
        "    (run_dir / 'result.json').write_text(json.dumps(state))\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    current = {
        "status": "active",
        "report_mode": "hook",
        "report_hook": "reporter:write",
    }
    supervisor.refresh_results_report(tmp_path, current)
    assert json.loads((tmp_path / "result.json").read_text()) == current


def test_explicit_hook_timeout_preserves_unrelated_processes(tmp_path, monkeypatch):
    (tmp_path / "slow_reporter.py").write_text(
        "import time\ndef write(run_dir, state):\n    time.sleep(20)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True
    )
    try:
        start = time.monotonic()
        supervisor.refresh_results_report(
            tmp_path,
            {
                "report_mode": "hook",
                "report_hook": "slow_reporter:write",
                "report_timeout_seconds": 0.2,
            },
        )
        assert time.monotonic() - start < 3
        assert unrelated.poll() is None
        assert "Report hook timed out" in (tmp_path / "decisions.log").read_text()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


@pytest.mark.parametrize(
    "mode, hook",
    [(None, None), ("legacy", None), ("status_only", None), ("hook", "reporter:write")],
)
def test_initialise_records_report_mode_before_watchdog_start(
    tmp_path, monkeypatch, mode, hook
):
    clock = tmp_path / "clock.json"
    clock.write_text(
        json.dumps({"start_monotonic": 1, "start_epoch": 2, "deadline_utc": "test"})
    )
    run = tmp_path / "run"

    def fake_start(*_args, **_kwargs):
        state = json.loads((run / "run_manifest.json").read_text())
        assert state["report_mode"] == (mode or "legacy")
        assert state["report_hook"] == hook
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_start)
    monkeypatch.setattr(supervisor, "proc_start", lambda _pid: 789)
    if mode is None:
        supervisor.initialise(run, clock)
    else:
        supervisor.initialise(run, clock, report_mode=mode, report_hook=hook)
    state = json.loads((run / "run_manifest.json").read_text())
    assert state["report_mode"] == (mode or "legacy")
    assert state["watchdog_pid"] == 12345
    assert "automatic checkpoint" in (run / "REPORT.md").read_text()


def test_invalid_report_mode_is_rejected_before_creating_run(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError):
        supervisor.initialise(
            run, tmp_path / "missing-clock.json", report_mode="invalid"
        )
    assert not run.exists()


@pytest.mark.parametrize(
    "mode, hook",
    [("hook", None), ("hook", "not-an-entrypoint"), ("hook", "x:bad:name")],
)
def test_invalid_report_hook_is_rejected_before_creating_run(tmp_path, mode, hook):
    run = tmp_path / "run"
    with pytest.raises(ValueError):
        supervisor.initialise(
            run, tmp_path / "missing-clock.json", report_mode=mode, report_hook=hook
        )
    assert not run.exists()


@pytest.mark.parametrize("status", ["deadline_reached", "boot_mismatch", "complete"])
def test_finish_preserves_terminal_run_status(tmp_path, status):
    now = time.monotonic()
    current = {
        "status": status,
        "boot_id": supervisor.boot_id(),
        "stages": [],
        "report_mode": "status_only",
        "clock": {"start_monotonic": now - 5, "deadline_utc": "test"},
    }
    supervisor.write_json(tmp_path / "run_manifest.json", current)
    supervisor.finish(tmp_path)
    recorded = supervisor.read_json(tmp_path / "run_manifest.json")
    assert recorded["status"] == status
    assert recorded["finished_utc"]
    assert recorded["elapsed_seconds"] >= 5


def test_finish_marks_active_supervisor_complete_not_a_study_success(tmp_path):
    now = time.monotonic()
    current = {
        "status": "active",
        "boot_id": supervisor.boot_id(),
        "stages": [],
        "report_mode": "status_only",
        "clock": {"start_monotonic": now, "deadline_utc": "test"},
    }
    supervisor.write_json(tmp_path / "run_manifest.json", current)
    supervisor.finish(tmp_path)
    assert supervisor.read_json(tmp_path / "run_manifest.json")["status"] == "complete"


def test_finish_refuses_live_descendant_after_stage_leader_exits(tmp_path, monkeypatch):
    now = time.monotonic()
    supervisor.write_json(
        tmp_path / "run_manifest.json",
        {
            "status": "active",
            "boot_id": supervisor.boot_id(),
            "stages": [{"name": "train", "pid": 123456, "status": "completed"}],
            "report_mode": "status_only",
            "clock": {"start_monotonic": now, "deadline_utc": "test"},
        },
    )
    monkeypatch.setattr(
        supervisor, "group_has_live_members", lambda group: group == 123456
    )
    with pytest.raises(SystemExit, match="registered stages run"):
        supervisor.finish(tmp_path)
    assert supervisor.read_json(tmp_path / "run_manifest.json")["status"] == "active"
