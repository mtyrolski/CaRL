import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from scripts import intent_experiment_supervisor as supervisor
from scripts.experiment_support.supervisor.processes import pidfd_open, pidfd_send_signal


def test_finish_request_waits_for_driver_and_other_registered_groups(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gate = run_dir / "gate"
    code = (
        "from pathlib import Path\nimport sys,time\n"
        "from scripts.intent_experiment_supervisor import request_finish\n"
        "run=Path(sys.argv[1]); gate=Path(sys.argv[2])\n"
        "while not gate.exists(): time.sleep(.01)\n"
        "request_finish(run)\ntime.sleep(.5)\n"
    )
    driver = subprocess.Popen(
        [sys.executable, "-c", code, str(run_dir), str(gate)], start_new_session=True
    )
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(10)"], start_new_session=True
    )
    try:
        now, epoch = time.monotonic(), time.time()
        current = {
            "status": "active",
            "boot_id": supervisor.boot_id(),
            "clock": {
                "start_monotonic": now,
                "start_epoch": epoch,
                "deadline_monotonic": now + 10,
                "deadline_epoch": epoch + 10,
                "deadline_utc": "test",
            },
            "stages": [
                {
                    "name": "driver",
                    "status": "running",
                    "pid": driver.pid,
                    "proc_start": supervisor.proc_start(driver.pid),
                    "exit_code": None,
                    "output": "driver.log",
                },
                {
                    "name": "other",
                    "status": "running",
                    "pid": other.pid,
                    "proc_start": supervisor.proc_start(other.pid),
                    "exit_code": None,
                    "output": "other.log",
                },
            ],
        }
        supervisor.write_json(run_dir / "run_manifest.json", current)
        gate.touch()
        deadline = time.monotonic() + 5
        while (
            not (run_dir / "finish_request.json").exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
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
    child_code = (
        "import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(20)"
    )
    leader_code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(20)"
    )
    leader = subprocess.Popen(
        [sys.executable, "-c", leader_code, child_code, str(ready)],
        start_new_session=True,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True
    )
    watcher = None
    child_fd = None
    try:
        end = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < end:
            time.sleep(0.01)
        assert ready.exists()
        child_fd = pidfd_open(int(ready.read_text()))
        now, epoch = time.monotonic(), time.time()
        current = {
            "status": "active",
            "boot_id": supervisor.boot_id(),
            "clock": {
                "start_monotonic": now,
                "start_epoch": epoch,
                "deadline_utc": "test",
                "deadline_monotonic": now + 8,
                "deadline_epoch": epoch + 8,
            },
            "stages": [
                {
                    "name": "orphan",
                    "kind": "normal",
                    "status": "running",
                    "pid": leader.pid,
                    "proc_start": supervisor.proc_start(leader.pid),
                    "stop_monotonic": now + 1.5,
                    "stop_epoch": epoch + 1.5,
                    "exit_code": None,
                    "output": "orphan.log",
                }
            ],
        }
        supervisor.write_json(run_dir / "run_manifest.json", current)
        watcher = subprocess.Popen(
            [
                sys.executable,
                str(Path(supervisor.__file__).resolve()),
                "watchdog",
                str(run_dir),
            ],
            start_new_session=True,
        )
        log = run_dir / "decisions.log"
        end = time.monotonic() + 1
        while (
            not log.exists() or "Tracking orphan" not in log.read_text()
        ) and time.monotonic() < end:
            time.sleep(0.01)
        assert log.exists() and "Tracking orphan" in log.read_text()
        leader.kill()
        leader.wait()
        end = time.monotonic() + 4
        while not supervisor._pidfd_exited(child_fd) and time.monotonic() < end:
            time.sleep(0.02)
        assert supervisor._pidfd_exited(child_fd)
        assert unrelated.poll() is None
        end = time.monotonic() + 1
        while (
            "tracked handles stopped=True" not in log.read_text()
            and time.monotonic() < end
        ):
            time.sleep(0.01)
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
                pidfd_send_signal(child_fd, signal.SIGKILL)
            os.close(child_fd)
        unrelated.terminate()
        unrelated.wait()
