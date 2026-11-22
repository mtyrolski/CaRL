"""Behavioural checks for the read-only local experiment overview."""

import json
from pathlib import Path

from scripts import show_active_experiments as overview


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_active_run_reports_only_observed_results(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "run_manifest.json", {
        "status": "active", "boot_id": "test-boot", "stages": [
            {"name": "evaluation", "status": "running", "pid": 123, "proc_start": 456, "output": "run.log"},
        ],
    })
    write_json(run / "status.json", {"completed": 12, "planned": 100, "reason": "working"})
    (run / "summary.csv").write_text("family,k,completed,hits\n12x12x10,1,12,9\n12x12x10,2,0,0\n")
    (run / "RESULTS.md").write_text("report")
    monkeypatch.setattr(overview, "owned_process", lambda stage: stage["pid"] == 123)
    result = overview.inspect_run(run / "run_manifest.json", "test-boot")
    assert result["state"] == "running"
    assert result["completed"] == 12 and result["planned"] == 100
    assert len(result["results"]) == 1
    overview.display([result])
    rendered = capsys.readouterr().out
    assert "9/12" in rendered and "0/0" not in rendered
    assert "RESULTS.md" in rendered and "run.log" in rendered


def test_stale_or_reused_pid_is_not_reported_active(tmp_path, monkeypatch):
    run = tmp_path / "old"
    run.mkdir()
    write_json(run / "run_manifest.json", {"status": "active", "boot_id": "old-boot", "stages": [
        {"name": "evaluation", "status": "running", "pid": 123, "proc_start": 456},
    ]})
    monkeypatch.setattr(overview, "boot_id", lambda: "current-boot")
    monkeypatch.setattr(overview, "owned_process", lambda _: True)
    assert overview.discover(tmp_path) == []
    assert overview.discover(tmp_path, include_all=True)[0]["state"] == "not verified"


def test_watchdog_with_invisible_stage_is_not_claimed_running(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "run_manifest.json", {"status": "active", "boot_id": "boot", "watchdog_pid": 12,
        "watchdog_proc_start": 34, "stages": [{"name": "eval", "status": "running", "pid": 9}]})
    monkeypatch.setattr(overview, "owned_process", lambda _: False)
    monkeypatch.setattr(overview, "proc_start", lambda pid: 34 if pid == 12 else None)
    assert overview.inspect_run(run / "run_manifest.json", "boot")["state"] == "stage not visible"
