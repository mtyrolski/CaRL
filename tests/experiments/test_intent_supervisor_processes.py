import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import intent_experiment_supervisor as supervisor


def test_refuses_to_signal_unregistered_or_reused_pid():
    assert not supervisor.stop_owned({"pid": 1, "proc_start": -1})


def test_refuses_to_signal_a_pid_from_another_boot():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True
    )
    try:
        stage = {
            "pid": child.pid,
            "proc_start": supervisor.proc_start(child.pid),
            "boot_id": "different-boot",
        }
        assert not supervisor.stop_owned(stage)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait()


def test_legacy_boot_requires_explicit_live_watchdog_adoption(tmp_path):
    run_dir = (tmp_path / "run").resolve()
    run_dir.mkdir()
    old = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(20)",
            str(Path(supervisor.__file__).resolve()),
            "watchdog",
            str(run_dir),
        ],
        start_new_session=True,
    )
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
                "deadline_utc": "test",
            },
            "stages": [],
            "watchdog_pid": old.pid,
            "watchdog_proc_start": supervisor.proc_start(old.pid),
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
            time.sleep(0.02)
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
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"], start_new_session=True
    )
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
    proc = subprocess.Popen(
        [
            sys.executable,
            str(Path(supervisor.__file__).resolve()),
            "stage-leader",
            "--gate-fd",
            str(read_fd),
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
            str(marker),
        ],
        start_new_session=True,
        pass_fds=(read_fd,),
    )
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
            "deadline_utc": "test",
        },
        "expensive_start_cutoff_monotonic": now + 4,
        "training_stop_monotonic": now + 4,
        "stages": [],
    }
    supervisor.write_json(run_dir / "run_manifest.json", current)
    code = (
        "import json,os,pathlib,sys; s=json.loads(pathlib.Path(sys.argv[1]).read_text()); "
        "assert s['stages'][-1]['pid']==os.getpgrp()"
    )
    result = supervisor.run_stage(
        run_dir,
        "probe",
        "normal",
        3,
        [sys.executable, "-c", code, str(run_dir / "run_manifest.json")],
    )
    assert result == 0
    recorded = supervisor.read_json(run_dir / "run_manifest.json")
    assert recorded["stages"][0]["status"] == "completed"


def test_stage_leader_waits_for_orphaned_descendant_before_timeout(tmp_path):
    ready = tmp_path / "ready"
    main_done = tmp_path / "main_done"
    child_code = (
        "import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(20)"
    )
    main_code = (
        "import pathlib,subprocess,sys,time\n"
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]])\n"
        "while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)\n"
        "pathlib.Path(sys.argv[3]).touch()\n"
    )
    read_fd, write_fd = os.pipe()
    leader = subprocess.Popen(
        [
            sys.executable,
            str(Path(supervisor.__file__).resolve()),
            "stage-leader",
            "--gate-fd",
            str(read_fd),
            "--",
            sys.executable,
            "-c",
            main_code,
            child_code,
            str(ready),
            str(main_done),
        ],
        start_new_session=True,
        pass_fds=(read_fd,),
    )
    os.close(read_fd)
    os.write(write_fd, b"1")
    os.close(write_fd)
    child_fd = None
    stage = None
    try:
        end = time.monotonic() + 5
        while not main_done.exists() and time.monotonic() < end:
            time.sleep(0.01)
        assert main_done.exists() and ready.exists()
        child_fd = os.pidfd_open(int(ready.read_text()))
        time.sleep(0.1)
        assert leader.poll() is None
        stage = {
            "pid": leader.pid,
            "proc_start": supervisor.proc_start(leader.pid),
            "boot_id": supervisor.boot_id(),
        }
        assert supervisor.stop_owned(stage, grace_seconds=0.1)
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
    child_code = (
        "import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(20)"
    )
    leader_code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(20)"
    )
    proxy_code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]],start_new_session=True); "
        "print(p.pid,flush=True); p.wait()"
    )
    proxy = subprocess.Popen(
        [sys.executable, "-c", proxy_code, leader_code, child_code, str(ready)],
        stdout=subprocess.PIPE,
        text=True,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True
    )
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
        stage = {
            "pid": leader_pid,
            "proc_start": start,
            "boot_id": supervisor.boot_id(),
        }
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
