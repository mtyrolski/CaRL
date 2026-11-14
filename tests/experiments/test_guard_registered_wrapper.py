"""A lost supervisor wrapper never implies an invented successful exit."""

import json
import os
import subprocess
import sys

import pytest

from scripts.guard_registered_wrapper import Presence
from scripts.guard_registered_wrapper import Reconcile
from scripts.guard_registered_wrapper import Target
from scripts.guard_registered_wrapper import capture_target
from scripts.guard_registered_wrapper import group_presence
from scripts.guard_registered_wrapper import monitor
from scripts.guard_registered_wrapper import parse_stat
from scripts.guard_registered_wrapper import pidfd_alive
from scripts.guard_registered_wrapper import reconcile
from scripts.guard_registered_wrapper import registered_guardian
from scripts.guard_registered_wrapper import watchdog_visible
from scripts.guard_registered_wrapper import wrapper_presence


def stat(pid: int, group: int, start: int) -> str:
    fields = ["S", "1", str(group), str(group)] + ["0"] * 15 + [str(start)] + ["0"] * 5
    return f"{pid} (name with spaces) " + " ".join(fields)


def process(proc, pid: int, group: int, start: int):
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    (directory / "stat").write_text(stat(pid, group, start))


def manifest():
    return {"status": "active", "boot_id": "boot", "watchdog_pid": 404, "watchdog_proc_start": 40004,
            "clock": {"deadline_utc": "2026-11-14T12:23:24Z"},
            "stages": [{"name": "representation_core", "pid": 101, "proc_start": 10001,
                        "kind": "expensive", "status": "running", "exit_code": None, "output": "core.log"},
                       {"name": "guardian", "pid": 303, "proc_start": 30003,
                        "kind": "normal", "status": "running", "exit_code": None, "output": "guardian.log"}]}


def target():
    return Target("representation_core", 101, 10001, 202, 20002, "boot", 404, 40004)


def test_proc_identity_reuse_and_unavailable_information(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    process(proc, 202, 202, 20002)
    assert parse_stat(stat(202, 202, 20002)) == (202, 20002)
    assert wrapper_presence(202, 20002, proc) == Presence.PRESENT
    assert wrapper_presence(202, 1, proc) == Presence.ABSENT
    assert wrapper_presence(999, 1, proc) == Presence.ABSENT
    (proc / "202/stat").write_text("malformed")
    assert wrapper_presence(202, 20002, proc) == Presence.UNKNOWN
    assert group_presence(101, proc) == Presence.UNKNOWN
    assert wrapper_presence(1, 1, tmp_path / "missing") == Presence.UNKNOWN
    assert group_presence(1, tmp_path / "missing") == Presence.UNKNOWN


def test_group_scan_treats_members_as_present_and_other_groups_as_absent(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    process(proc, 111, 101, 10011)
    process(proc, 222, 202, 20022)
    assert group_presence(101, proc) == Presence.PRESENT
    assert group_presence(303, proc) == Presence.ABSENT


def test_unknown_wrapper_start_requires_pid_absence(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    process(proc, 202, 202, 20002)
    assert wrapper_presence(202, None, proc) == Presence.PRESENT
    (proc / "202/stat").unlink()
    (proc / "202").rmdir()
    assert wrapper_presence(202, None, proc) == Presence.ABSENT

    unknown = capture_target(manifest(), "representation_core", 202, None, "boot")
    assert unknown is not None
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest()))
    monkeypatch.setattr("scripts.guard_registered_wrapper.boot_id", lambda: "boot")
    monkeypatch.setattr("scripts.guard_registered_wrapper.watchdog_visible", lambda _manifest: True)
    assert reconcile(tmp_path, unknown, proc) == Reconcile.MARKED
    saved = json.loads((tmp_path / "run_manifest.json").read_text())["stages"][0]
    assert saved["guardian_wrapper_start"] is None
    assert "start time was not recorded" in saved["guardian_reason"]
    assert saved["status"] == "lost_or_exited" and saved["exit_code"] is None


def test_group_scan_with_missing_stat_and_existing_pid_fails_closed(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    (proc / "123").mkdir(parents=True)
    assert group_presence(101, proc) == Presence.UNKNOWN
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest()))
    monkeypatch.setattr("scripts.guard_registered_wrapper.boot_id", lambda: "boot")
    monkeypatch.setattr("scripts.guard_registered_wrapper.watchdog_visible", lambda _manifest: True)
    assert reconcile(tmp_path, target(), proc) == Reconcile.DEFER
    assert json.loads((tmp_path / "run_manifest.json").read_text())["stages"][0]["status"] == "running"


def test_capture_and_guardian_registration_require_frozen_identities(monkeypatch):
    current = manifest()
    monkeypatch.setattr("scripts.guard_registered_wrapper.proc_metadata", lambda _pid: (30003, 303, 303))
    assert capture_target(current, "representation_core", 202, 20002, "boot") == target()
    assert registered_guardian(current, 303, 303)
    assert not registered_guardian(current, 303, 999)
    monkeypatch.setattr("scripts.guard_registered_wrapper.proc_metadata", lambda _pid: (1, 303, 303))
    assert not registered_guardian(current, 303, 303)  # A reused PID is not the registered guardian.
    monkeypatch.setattr("scripts.guard_registered_wrapper.proc_metadata", lambda _pid: None)
    assert not registered_guardian(current, 303, 303)  # Inaccessible process metadata is inconclusive.
    with pytest.raises(ValueError, match="boot"):
        capture_target(current, "representation_core", 202, 20002, "different")
    with pytest.raises(ValueError, match="distinct"):
        capture_target(current, "representation_core", 101, 20002, "boot")
    current["stages"][0]["status"] = "completed"
    assert capture_target(current, "representation_core", 202, 20002, "boot") is None


def test_watchdog_must_be_the_recorded_process_in_the_same_pid_namespace(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    own_namespace = proc / "self/ns/pid"
    watchdog_namespace = proc / "404/ns/pid"
    own_namespace.parent.mkdir(parents=True)
    watchdog_namespace.parent.mkdir(parents=True)
    own_namespace.write_text("namespace")
    os.link(own_namespace, watchdog_namespace)
    current = {"watchdog_pid": 404, "watchdog_proc_start": 40004}
    monkeypatch.setattr("scripts.guard_registered_wrapper.proc_metadata", lambda _pid: (40004, 404, 404))
    monkeypatch.setattr("scripts.guard_registered_wrapper.pidfd_alive", lambda _pid: True)
    assert watchdog_visible(current, proc)
    monkeypatch.setattr("scripts.guard_registered_wrapper.proc_metadata", lambda _pid: (1, 404, 404))
    assert not watchdog_visible(current, proc)
    monkeypatch.setattr("scripts.guard_registered_wrapper.proc_metadata", lambda _pid: (40004, 404, 404))
    watchdog_namespace.unlink()
    watchdog_namespace.write_text("other namespace")
    assert not watchdog_visible(current, proc)
    watchdog_namespace.unlink()
    assert not watchdog_visible(current, proc)


def test_pidfd_rejects_a_real_zombie_without_signalling_it():
    assert pidfd_alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        assert not pidfd_alive(child.pid)
    finally:
        child.wait()


def test_reconcile_marks_only_lost_and_never_signals(tmp_path, monkeypatch):
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest()))
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr("scripts.guard_registered_wrapper.boot_id", lambda: "boot")
    monkeypatch.setattr("scripts.guard_registered_wrapper.watchdog_visible", lambda _manifest: True)
    monkeypatch.setattr(os, "kill", lambda *_args: pytest.fail("guardian must never signal a process"))
    assert reconcile(tmp_path, target(), proc) == Reconcile.MARKED
    saved = json.loads((tmp_path / "run_manifest.json").read_text())
    stage = saved["stages"][0]
    assert stage["status"] == "lost_or_exited" and stage["exit_code"] is None
    assert "unknown" in stage["guardian_reason"]
    assert "guardian sent no signals" in (tmp_path / "decisions.log").read_text()
    assert reconcile(tmp_path, target(), proc) == Reconcile.TERMINAL


def test_final_locked_check_defers_when_wrapper_or_group_reappears(tmp_path, monkeypatch):
    original = manifest()
    (tmp_path / "run_manifest.json").write_text(json.dumps(original))
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr("scripts.guard_registered_wrapper.boot_id", lambda: "boot")
    monkeypatch.setattr("scripts.guard_registered_wrapper.watchdog_visible", lambda _manifest: True)
    process(proc, 202, 202, 20002)
    assert reconcile(tmp_path, target(), proc) == Reconcile.DEFER
    (proc / "202/stat").unlink()
    (proc / "202").rmdir()
    process(proc, 111, 101, 10011)
    assert reconcile(tmp_path, target(), proc) == Reconcile.DEFER
    assert json.loads((tmp_path / "run_manifest.json").read_text()) == original


def test_final_locked_check_defers_if_watchdog_disappeared_or_changed(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    current = manifest()
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(current))
    monkeypatch.setattr("scripts.guard_registered_wrapper.boot_id", lambda: "boot")
    monkeypatch.setattr("scripts.guard_registered_wrapper.watchdog_visible", lambda _manifest: False)
    assert reconcile(tmp_path, target(), proc) == Reconcile.DEFER
    observations = iter((True, False))
    monkeypatch.setattr("scripts.guard_registered_wrapper.watchdog_visible", lambda _manifest: next(observations))
    assert reconcile(tmp_path, target(), proc) == Reconcile.DEFER
    monkeypatch.setattr("scripts.guard_registered_wrapper.watchdog_visible", lambda _manifest: True)
    for key in ("watchdog_pid", "watchdog_proc_start"):
        changed = {**current, key: current[key] + 1}
        path.write_text(json.dumps(changed))
        assert reconcile(tmp_path, target(), proc) == Reconcile.DEFER
        assert json.loads(path.read_text())["stages"][0]["status"] == "running"


def test_three_independent_absence_polls_and_unknown_resets_count(tmp_path, monkeypatch):
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest()))
    monkeypatch.setattr("scripts.guard_registered_wrapper.boot_id", lambda: "boot")
    wrapper = iter([Presence.ABSENT, Presence.ABSENT, Presence.UNKNOWN,
                    Presence.ABSENT, Presence.ABSENT, Presence.ABSENT])
    calls = []
    monkeypatch.setattr("scripts.guard_registered_wrapper.wrapper_presence", lambda *_args: next(wrapper))
    monkeypatch.setattr("scripts.guard_registered_wrapper.group_presence", lambda *_args: Presence.ABSENT)
    monkeypatch.setattr("scripts.guard_registered_wrapper.reconcile",
                        lambda *_args: calls.append("locked-final-check") or Reconcile.MARKED)
    monkeypatch.setattr("scripts.guard_registered_wrapper.time.sleep", lambda _seconds: None)
    assert monitor(tmp_path, target(), interval=.2) == Reconcile.MARKED
    assert calls == ["locked-final-check"]
    with pytest.raises(ValueError, match="interval"):
        monitor(tmp_path, target(), interval=0)


def test_changed_stage_or_boot_fails_closed(tmp_path, monkeypatch):
    current = manifest()
    current["stages"][0]["proc_start"] += 1
    (tmp_path / "run_manifest.json").write_text(json.dumps(current))
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr("scripts.guard_registered_wrapper.boot_id", lambda: "boot")
    monkeypatch.setattr("scripts.guard_registered_wrapper.watchdog_visible", lambda _manifest: True)
    with pytest.raises(ValueError, match="identity"):
        reconcile(tmp_path, target(), proc)
    assert json.loads((tmp_path / "run_manifest.json").read_text()) == current
    current["boot_id"] = "different"
    (tmp_path / "run_manifest.json").write_text(json.dumps(current))
    with pytest.raises(ValueError, match="boot"):
        reconcile(tmp_path, target(), proc)
