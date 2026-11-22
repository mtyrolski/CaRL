"""Register and supervise individual commands."""

import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.experiment_support.supervisor.processes import boot_id
from scripts.experiment_support.supervisor.processes import proc_start
from scripts.experiment_support.supervisor.processes import stop_owned
from scripts.experiment_support.supervisor.state import StageStatus
from scripts.experiment_support.supervisor.state import append_log
from scripts.experiment_support.supervisor.state import compute_stop
from scripts.experiment_support.supervisor.state import locked
from scripts.experiment_support.supervisor.state import read_json
from scripts.experiment_support.supervisor.state import refresh_results_report
from scripts.experiment_support.supervisor.state import update_report
from scripts.experiment_support.supervisor.state import utc_now
from scripts.experiment_support.supervisor.state import write_json

ENTRYPOINT = Path(__file__).resolve().parents[2] / "intent_experiment_supervisor.py"


def run_stage(
    run_dir: Path, name: str, kind: str, timeout: float, command: list[str]
) -> int:
    if not command:
        raise SystemExit("A stage command is required after --")
    if command[0] == "--":
        command = command[1:]
    if not command or timeout <= 0:
        raise SystemExit("A non-empty command and positive timeout are required")
    with locked(run_dir):
        state = read_json(run_dir / "run_manifest.json")
        if state.get("boot_id") != boot_id():
            raise SystemExit(
                "Run boot ID is missing or differs; do not reuse its registered PIDs"
            )
        now = time.monotonic()
        epoch = time.time()
        try:
            stop = compute_stop(state, kind, timeout, now, epoch)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if any(x["name"] == name for x in state["stages"]):
            raise SystemExit(f"Stage name already exists: {name}")
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        output = run_dir / "logs" / f"{safe_name}.log"
        read_fd, write_fd = os.pipe()
        try:
            with output.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(ENTRYPOINT),
                        "stage-leader",
                        "--gate-fd",
                        str(read_fd),
                        "--",
                        *command,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    pass_fds=(read_fd,),
                )
            os.close(read_fd)
            read_fd = -1
            stop_epoch = epoch + (stop - now)
            stage = {
                "name": name,
                "kind": kind,
                "command": command,
                "timeout_seconds": timeout,
                "started_utc": utc_now(),
                "started_monotonic": now,
                "stop_monotonic": stop,
                "stop_epoch": stop_epoch,
                "pid": child.pid,
                "proc_start": proc_start(child.pid),
                "boot_id": state["boot_id"],
                "status": StageStatus.RUNNING,
                "output": str(output),
                "exit_code": None,
            }
            state["stages"].append(stage)
            write_json(run_dir / "run_manifest.json", state)
            update_report(run_dir, state)
            append_log(
                run_dir,
                f"Started {name}, pid={child.pid}, timeout={timeout:.0f}s, kind={kind}.",
            )
            os.write(write_fd, b"1")
        finally:
            if read_fd >= 0:
                os.close(read_fd)
            os.close(write_fd)
    while child.poll() is None:
        if time.monotonic() >= stop or time.time() >= stop_epoch:
            remaining = min(
                state["clock"]["deadline_monotonic"] - time.monotonic(),
                state["clock"]["deadline_epoch"] - time.time(),
            )
            stop_owned(stage, grace_seconds=max(0.0, min(3.0, remaining)))
            break
        time.sleep(0.5)
    code = child.wait()
    with locked(run_dir):
        state = read_json(run_dir / "run_manifest.json")
        recorded = next(s for s in state["stages"] if s["name"] == name)
        if recorded["status"] == StageStatus.RUNNING:
            timed_out = time.monotonic() >= stop or time.time() >= stop_epoch
            recorded["status"] = (
                StageStatus.COMPLETED
                if code == 0
                else (StageStatus.TIMEOUT if timed_out else StageStatus.FAILED)
            )
        recorded["exit_code"] = code
        recorded["finished_utc"] = utc_now()
        write_json(run_dir / "run_manifest.json", state)
        update_report(run_dir, state)
        if code != 0:
            refresh_results_report(run_dir, state)
        append_log(
            run_dir, f"Finished {name}, status={recorded['status']}, exit={code}."
        )
    print(f"{name}: {recorded['status']} exit={code} log={output}", flush=True)
    return code if code >= 0 else 128 - code
