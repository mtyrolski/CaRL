import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "intent_supervisor",
    Path(__file__).resolve().parents[2] / "scripts/intent_experiment_supervisor.py")
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def state():
    return {
        "status": "active",
        "clock": {
            "deadline_monotonic": 270.0
        },
        "expensive_start_cutoff_monotonic": 225.0,
        "training_stop_monotonic": 240.0
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


def test_refuses_to_signal_unregistered_or_reused_pid():
    assert not supervisor.stop_owned({"pid": 1, "proc_start": -1})


def test_refuses_to_signal_a_pid_from_another_boot():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    try:
        stage = {"pid": child.pid, "proc_start": supervisor.proc_start(child.pid), "boot_id": "different-boot"}
        assert not supervisor.stop_owned(stage)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait()


def test_legacy_boot_requires_explicit_live_watchdog_adoption(tmp_path):
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    old = subprocess.Popen([
        sys.executable, "-c", "import time; time.sleep(20)",
        str(Path(supervisor.__file__).resolve()), "watchdog",
        str(run_dir)
    ],
                           start_new_session=True)
    new_fd = None
    try:
        now, epoch = time.monotonic(), time.time()
        current = {
            "status": "active",
            "clock": {
                "start_monotonic": now,
                "start_epoch": epoch,
                "deadline_monotonic": now + 30,
                "deadline_epoch": epoch + 30,
                "deadline_utc": "test"
            },
            "stages": [],
            "watchdog_pid": old.pid,
            "watchdog_proc_start": supervisor.proc_start(old.pid)
        }
        supervisor.write_json(run_dir / "run_manifest.json", current)
        with pytest.raises(SystemExit, match="explicitly adopt"):
            supervisor.refresh_watchdog(run_dir)
        assert old.poll() is None
        supervisor.refresh_watchdog(run_dir, adopt_current_boot=True)
        recorded = supervisor.read_json(run_dir / "run_manifest.json")
        assert recorded["boot_id"] == supervisor.boot_id()
        assert old.wait(timeout=2) != 0
        new_fd = os.pidfd_open(recorded["watchdog_pid"])
        supervisor.finish(run_dir)
        deadline = time.monotonic() + 4
        while not supervisor._pidfd_exited(new_fd) and time.monotonic() < deadline:
            time.sleep(.02)
        assert supervisor._pidfd_exited(new_fd)
    finally:
        if old.poll() is None:
            old.terminate()
            old.wait()
        if new_fd is not None:
            if not supervisor._pidfd_exited(new_fd):
                signal.pidfd_send_signal(new_fd, signal.SIGTERM)
            os.close(new_fd)


def test_owned_process_group_can_be_stopped():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True)
    try:
        token = supervisor.proc_start(child.pid)
        assert token is not None
        assert supervisor.stop_owned({"pid": child.pid, "proc_start": token})
        assert child.wait(timeout=1) != 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_stage_leader_does_not_start_command_without_registration(tmp_path):
    marker = tmp_path / "started"
    read_fd, write_fd = os.pipe()
    proc = subprocess.Popen([
        sys.executable,
        str(Path(supervisor.__file__).resolve()), "stage-leader", "--gate-fd",
        str(read_fd), "--", sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
        str(marker)
    ],
                            start_new_session=True,
                            pass_fds=(read_fd,))
    os.close(read_fd)
    os.close(write_fd)
    assert proc.wait(timeout=3) == 125
    assert not marker.exists()


def test_stage_command_sees_registered_process_group_before_start(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    now, epoch = time.monotonic(), time.time()
    current = {
        "status": "active",
        "boot_id": supervisor.boot_id(),
        "clock": {
            "start_monotonic": now,
            "start_epoch": epoch,
            "deadline_monotonic": now + 5,
            "deadline_epoch": epoch + 5,
            "deadline_utc": "test"
        },
        "expensive_start_cutoff_monotonic": now + 4,
        "training_stop_monotonic": now + 4,
        "stages": []
    }
    supervisor.write_json(run_dir / "run_manifest.json", current)
    code = ("import json,os,pathlib,sys; s=json.loads(pathlib.Path(sys.argv[1]).read_text()); "
            "assert s['stages'][-1]['pid']==os.getpgrp()")
    result = supervisor.run_stage(
        run_dir, "probe", "normal", 3,
        [sys.executable, "-c", code, str(run_dir / "run_manifest.json")])
    assert result == 0
    recorded = supervisor.read_json(run_dir / "run_manifest.json")
    assert recorded["stages"][0]["status"] == "completed"


def test_stage_leader_waits_for_orphaned_descendant_before_timeout(tmp_path):
    ready = tmp_path / "ready"
    main_done = tmp_path / "main_done"
    child_code = ("import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                  "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(20)")
    main_code = ("import pathlib,subprocess,sys,time\n"
                 "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]])\n"
                 "while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)\n"
                 "pathlib.Path(sys.argv[3]).touch()\n")
    read_fd, write_fd = os.pipe()
    leader = subprocess.Popen([
        sys.executable,
        str(Path(supervisor.__file__).resolve()), "stage-leader", "--gate-fd",
        str(read_fd), "--", sys.executable, "-c", main_code, child_code,
        str(ready),
        str(main_done)
    ],
                              start_new_session=True,
                              pass_fds=(read_fd,))
    os.close(read_fd)
    os.write(write_fd, b"1")
    os.close(write_fd)
    child_fd = None
    stage = None
    try:
        end = time.monotonic() + 5
        while not main_done.exists() and time.monotonic() < end:
            time.sleep(.01)
        assert main_done.exists() and ready.exists()
        child_fd = os.pidfd_open(int(ready.read_text()))
        time.sleep(.1)
        assert leader.poll() is None
        stage = {"pid": leader.pid, "proc_start": supervisor.proc_start(leader.pid), "boot_id": supervisor.boot_id()}
        assert supervisor.stop_owned(stage, grace_seconds=.1)
        assert leader.wait(timeout=2) != 0
        assert supervisor._pidfd_exited(child_fd)
    finally:
        if leader.poll() is None:
            if stage is not None:
                supervisor.stop_owned(stage, grace_seconds=0)
            leader.kill()
            leader.wait()
        if child_fd is not None:
            if not supervisor._pidfd_exited(child_fd):
                signal.pidfd_send_signal(child_fd, signal.SIGKILL)
            os.close(child_fd)


def test_watchdog_stops_ignoring_descendant_after_nonchild_leader_exits(tmp_path):
    ready = tmp_path / "ready"
    child_code = ("import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                  "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(20)")
    leader_code = ("import subprocess,sys,time; "
                   "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(20)")
    proxy_code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]],start_new_session=True); "
        "print(p.pid,flush=True); p.wait()")
    proxy = subprocess.Popen([sys.executable, "-c", proxy_code, leader_code, child_code,
                              str(ready)],
                             stdout=subprocess.PIPE,
                             text=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
    child_fd = None
    leader_pid = None
    stage = None
    try:
        assert proxy.stdout is not None
        leader_pid = int(proxy.stdout.readline())
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        child_fd = os.pidfd_open(int(ready.read_text()))
        start = supervisor.proc_start(leader_pid)
        assert start is not None
        stage = {"pid": leader_pid, "proc_start": start, "boot_id": supervisor.boot_id()}
        assert supervisor.stop_owned(stage, grace_seconds=0.1)
        assert supervisor._pidfd_exited(child_fd)
        assert proxy.wait(timeout=2) == 0
        assert unrelated.poll() is None
    finally:
        if stage is not None and supervisor.owned_process(stage):
            supervisor.stop_owned(stage, grace_seconds=0)
        if child_fd is not None:
            if not supervisor._pidfd_exited(child_fd):
                signal.pidfd_send_signal(child_fd, signal.SIGKILL)
            os.close(child_fd)
        if proxy.poll() is None:
            proxy.terminate()
            proxy.wait()
        unrelated.terminate()
        unrelated.wait()


def test_finish_request_waits_for_driver_and_other_registered_groups(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = run_dir / "gate"
    code = ("from pathlib import Path\nimport sys,time\n"
            "from scripts.intent_experiment_supervisor import request_finish\n"
            "run=Path(sys.argv[1]); gate=Path(sys.argv[2])\n"
            "while not gate.exists(): time.sleep(.01)\n"
            "request_finish(run)\ntime.sleep(.5)\n")
    driver = subprocess.Popen([sys.executable, "-c", code, str(run_dir), str(gate)], start_new_session=True)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True)
    try:
        now, epoch = time.monotonic(), time.time()
        current = {
            "status":
                "active",
            "boot_id":
                supervisor.boot_id(),
            "clock": {
                "start_monotonic": now,
                "start_epoch": epoch,
                "deadline_monotonic": now + 10,
                "deadline_epoch": epoch + 10,
                "deadline_utc": "test"
            },
            "stages": [{
                "name": "driver",
                "status": "running",
                "pid": driver.pid,
                "proc_start": supervisor.proc_start(driver.pid),
                "exit_code": None,
                "output": "driver.log"
            }, {
                "name": "other",
                "status": "running",
                "pid": other.pid,
                "proc_start": supervisor.proc_start(other.pid),
                "exit_code": None,
                "output": "other.log"
            }]
        }
        supervisor.write_json(run_dir / "run_manifest.json", current)
        gate.touch()
        deadline = time.monotonic() + 5
        while not (run_dir / "finish_request.json").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert (run_dir / "finish_request.json").exists()
        assert not supervisor.complete_requested(run_dir, current)
        assert driver.wait(timeout=2) == 0
        assert not supervisor.complete_requested(run_dir, current)
        other.terminate()
        other.wait()
        assert supervisor.complete_requested(run_dir, current)
        assert current["status"] == "complete"
        assert current["stages"][0]["status"] == "lost_or_exited"
        assert current["stages"][0]["exit_code"] is None
    finally:
        if driver.poll() is None:
            driver.terminate()
            driver.wait()
        if other.poll() is None:
            other.terminate()
            other.wait()


def test_watchdog_stops_tracked_orphan_when_leader_exits_before_timeout(tmp_path):
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    ready = run_dir / "ready"
    child_code = ("import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                  "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(20)")
    leader_code = ("import subprocess,sys,time; "
                   "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(20)")
    leader = subprocess.Popen([sys.executable, "-c", leader_code, child_code, str(ready)], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
    watcher = None
    child_fd = None
    try:
        end = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < end:
            time.sleep(.01)
        assert ready.exists()
        child_fd = os.pidfd_open(int(ready.read_text()))
        now, epoch = time.monotonic(), time.time()
        current = {
            "status":
                "active",
            "boot_id":
                supervisor.boot_id(),
            "clock": {
                "start_monotonic": now,
                "start_epoch": epoch,
                "deadline_utc": "test",
                "deadline_monotonic": now + 8,
                "deadline_epoch": epoch + 8
            },
            "stages": [{
                "name": "orphan",
                "kind": "normal",
                "status": "running",
                "pid": leader.pid,
                "proc_start": supervisor.proc_start(leader.pid),
                "stop_monotonic": now + 1.5,
                "stop_epoch": epoch + 1.5,
                "exit_code": None,
                "output": "orphan.log"
            }]
        }
        supervisor.write_json(run_dir / "run_manifest.json", current)
        watcher = subprocess.Popen(
            [sys.executable, str(Path(supervisor.__file__).resolve()), "watchdog",
             str(run_dir)],
            start_new_session=True)
        log = run_dir / "decisions.log"
        end = time.monotonic() + 1
        while (not log.exists() or "Tracking orphan" not in log.read_text()) and time.monotonic() < end:
            time.sleep(.01)
        assert log.exists() and "Tracking orphan" in log.read_text()
        leader.kill()
        leader.wait()
        end = time.monotonic() + 4
        while not supervisor._pidfd_exited(child_fd) and time.monotonic() < end:
            time.sleep(.02)
        assert supervisor._pidfd_exited(child_fd)
        assert unrelated.poll() is None
        end = time.monotonic() + 1
        while "tracked handles stopped=True" not in log.read_text() and time.monotonic() < end:
            time.sleep(.01)
        assert "tracked handles stopped=True" in log.read_text()
    finally:
        if watcher is not None and watcher.poll() is None:
            watcher.terminate()
            watcher.wait()
        if leader.poll() is None:
            leader.kill()
            leader.wait()
        if child_fd is not None:
            if not supervisor._pidfd_exited(child_fd):
                signal.pidfd_send_signal(child_fd, signal.SIGKILL)
            os.close(child_fd)
        unrelated.terminate()
        unrelated.wait()


def test_annotation_registry_tracks_only_unfinished_own_groups(tmp_path):
    rows = [
        {
            "phase": "started",
            "pid": 123,
            "pgid": 123,
            "proc_start": 456,
            "stop_monotonic": 789
        },
        {
            "phase": "finished",
            "pid": 123,
            "proc_start": 456
        },
        {
            "phase": "started",
            "pid": 124,
            "pgid": 9,
            "proc_start": 457,
            "stop_monotonic": 789
        },
        {
            "phase": "started",
            "pid": 125,
            "pgid": 125,
            "proc_start": 458,
            "stop_monotonic": 790
        },
    ]
    (tmp_path / "terra_workers.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert [row["pid"] for row in supervisor.registered_workers(tmp_path)] == [125]


def test_status_only_never_calls_the_legacy_results_report(tmp_path, monkeypatch):
    current = {"status": "active", "clock": {"deadline_utc": "test"}, "stages": [], "report_mode": "status_only"}
    supervisor.update_report(tmp_path, current)
    before = (tmp_path / "REPORT.md").read_text()
    calls = []
    monkeypatch.setattr(supervisor.importlib.util, "spec_from_file_location", lambda *args: calls.append(args))
    supervisor.refresh_results_report(tmp_path, current)
    assert calls == []
    assert (tmp_path / "REPORT.md").read_text() == before
    assert "automatic checkpoint" in before


def test_legacy_report_remains_default_for_old_manifests(tmp_path, monkeypatch):
    calls = []
    module = SimpleNamespace(refresh_report_unlocked=lambda run, current: calls.append((run, current)))
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _module: None))
    monkeypatch.setattr(supervisor.importlib.util, "spec_from_file_location", lambda *_args: spec)
    monkeypatch.setattr(supervisor.importlib.util, "module_from_spec", lambda _spec: module)
    current = {"status": "active"}    # Saved before report_mode was introduced.
    supervisor.refresh_results_report(tmp_path, current)
    assert calls == [(tmp_path, current)]
    supervisor.refresh_results_report(tmp_path, {**current, "report_mode": "legacy"})
    assert len(calls) == 2


def test_unknown_report_mode_fails_closed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(supervisor.importlib.util, "spec_from_file_location", lambda *args: calls.append(args))
    supervisor.refresh_results_report(tmp_path, {"report_mode": "unknown"})
    assert calls == []
    assert "Unknown report mode" in (tmp_path / "decisions.log").read_text()


@pytest.mark.parametrize("mode", [None, "legacy", "status_only"])
def test_initialise_records_report_mode_before_watchdog_start(tmp_path, monkeypatch, mode):
    clock = tmp_path / "clock.json"
    clock.write_text(json.dumps({"start_monotonic": 1, "start_epoch": 2, "deadline_utc": "test"}))
    run = tmp_path / "run"

    def fake_start(*_args, **_kwargs):
        state = json.loads((run / "run_manifest.json").read_text())
        assert state["report_mode"] == (mode or "legacy")
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_start)
    monkeypatch.setattr(supervisor, "proc_start", lambda _pid: 789)
    if mode is None:
        supervisor.initialise(run, clock)
    else:
        supervisor.initialise(run, clock, report_mode=mode)
    state = json.loads((run / "run_manifest.json").read_text())
    assert state["report_mode"] == (mode or "legacy")
    assert state["watchdog_pid"] == 12345
    assert "automatic checkpoint" in (run / "REPORT.md").read_text()


def test_invalid_report_mode_is_rejected_before_creating_run(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(ValueError):
        supervisor.initialise(run, tmp_path / "missing-clock.json", report_mode="invalid")
    assert not run.exists()


@pytest.mark.parametrize("status", ["deadline_reached", "boot_mismatch", "complete"])
def test_finish_preserves_terminal_run_status(tmp_path, status):
    now = time.monotonic()
    current = {
        "status": status,
        "boot_id": supervisor.boot_id(),
        "stages": [],
        "report_mode": "status_only",
        "clock": {
            "start_monotonic": now - 5,
            "deadline_utc": "test"
        }
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
        "clock": {
            "start_monotonic": now,
            "deadline_utc": "test"
        }
    }
    supervisor.write_json(tmp_path / "run_manifest.json", current)
    supervisor.finish(tmp_path)
    assert supervisor.read_json(tmp_path / "run_manifest.json")["status"] == "complete"


def test_finish_refuses_live_descendant_after_stage_leader_exits(tmp_path, monkeypatch):
    now = time.monotonic()
    supervisor.write_json(tmp_path / "run_manifest.json", {
        "status": "active",
        "boot_id": supervisor.boot_id(),
        "stages": [{"name": "train", "pid": 123456, "status": "completed"}],
        "report_mode": "status_only",
        "clock": {"start_monotonic": now, "deadline_utc": "test"},
    })
    monkeypatch.setattr(supervisor, "group_has_live_members", lambda group: group == 123456)
    with pytest.raises(SystemExit, match="registered stages run"):
        supervisor.finish(tmp_path)
    assert supervisor.read_json(tmp_path / "run_manifest.json")["status"] == "active"
