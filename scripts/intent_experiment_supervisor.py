#!/usr/bin/env python3
"""Run commands under an independently supervised deadline."""

import argparse
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.experiment_support.supervisor.processes import _group_members
from scripts.experiment_support.supervisor.processes import _open_member
from scripts.experiment_support.supervisor.processes import _pidfd_exited
from scripts.experiment_support.supervisor.processes import boot_id
from scripts.experiment_support.supervisor.processes import close_tracked
from scripts.experiment_support.supervisor.processes import group_has_live_members
from scripts.experiment_support.supervisor.processes import owned_process
from scripts.experiment_support.supervisor.processes import pidfd_send_signal
from scripts.experiment_support.supervisor.processes import proc_metadata
from scripts.experiment_support.supervisor.processes import proc_start
from scripts.experiment_support.supervisor.processes import stage_leader
from scripts.experiment_support.supervisor.processes import stop_owned
from scripts.experiment_support.supervisor.processes import stop_tracked
from scripts.experiment_support.supervisor.processes import track_group
from scripts.experiment_support.supervisor.stages import run_stage
from scripts.experiment_support.supervisor.state import DEFAULT_EXPENSIVE_START_AFTER
from scripts.experiment_support.supervisor.state import DEFAULT_TRAINING_STOP_AFTER
from scripts.experiment_support.supervisor.state import ReportMode
from scripts.experiment_support.supervisor.state import RunStatus
from scripts.experiment_support.supervisor.state import StageKind
from scripts.experiment_support.supervisor.state import StageStatus
from scripts.experiment_support.supervisor.state import append_log
from scripts.experiment_support.supervisor.state import atomic_write
from scripts.experiment_support.supervisor.state import compute_stop
from scripts.experiment_support.supervisor.state import locked
from scripts.experiment_support.supervisor.state import read_json
from scripts.experiment_support.supervisor.state import refresh_results_report
from scripts.experiment_support.supervisor.state import registered_workers
from scripts.experiment_support.supervisor.state import update_report
from scripts.experiment_support.supervisor.state import utc_now
from scripts.experiment_support.supervisor.state import validate_report_hook
from scripts.experiment_support.supervisor.state import write_json
from scripts.experiment_support.supervisor.watchdog import complete_requested
from scripts.experiment_support.supervisor.watchdog import watchdog

__all__ = [
    "ReportMode",
    "RunStatus",
    "StageKind",
    "StageStatus",
    "_group_members",
    "_open_member",
    "_pidfd_exited",
    "append_log",
    "atomic_write",
    "boot_id",
    "close_tracked",
    "complete_requested",
    "compute_stop",
    "finish",
    "group_has_live_members",
    "initialise",
    "locked",
    "main",
    "owned_process",
    "proc_metadata",
    "proc_start",
    "read_json",
    "refresh_results_report",
    "refresh_watchdog",
    "registered_workers",
    "request_finish",
    "run_stage",
    "stage_leader",
    "stop_owned",
    "stop_tracked",
    "track_group",
    "update_report",
    "utc_now",
    "watchdog",
    "write_json",
]


def request_finish(run_dir: Path) -> None:
    """Ask the watchdog to finish once this registered group has exited."""
    run_dir = run_dir.resolve()
    with locked(run_dir):
        state = read_json(run_dir / "run_manifest.json")
        if state.get("status") != RunStatus.ACTIVE or state.get("boot_id") != boot_id():
            raise ValueError("Run is inactive or its boot identity cannot be verified")
        group = os.getpgrp()
        matches = [
            stage
            for stage in state["stages"]
            if stage.get("status") == StageStatus.RUNNING
            and stage.get("pid") == group
            and owned_process({**stage, "boot_id": state["boot_id"]})
        ]
        if len(matches) != 1:
            raise ValueError("Caller is not in exactly one registered live stage group")
        stage = matches[0]
        request = {
            "stage": stage["name"],
            "pid": group,
            "proc_start": stage["proc_start"],
            "boot_id": state["boot_id"],
            "requested_utc": utc_now(),
        }
        write_json(run_dir / "finish_request.json", request)
        append_log(
            run_dir,
            f"Finish requested by stage {stage['name']}; awaiting all groups to exit.",
        )
        update_report(run_dir, state)
        refresh_results_report(run_dir, state)


def initialise(
    run_dir: Path,
    clock_path: Path,
    proposal: Path | None = None,
    *,
    report_mode: ReportMode | str = ReportMode.LEGACY,
    report_hook: str | None = None,
    expensive_start_after: float = DEFAULT_EXPENSIVE_START_AFTER,
    training_stop_after: float = DEFAULT_TRAINING_STOP_AFTER,
) -> None:
    mode = ReportMode(report_mode)
    validate_report_hook(report_hook)
    if mode == ReportMode.HOOK and report_hook is None:
        raise ValueError("Report hook mode requires an explicit module:function")
    if any(
        not math.isfinite(value) or value < 0
        for value in (expensive_start_after, training_stop_after)
    ):
        raise ValueError("Stage cutoffs must be finite and nonnegative")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    clock = read_json(clock_path)
    state = {
        "schema_version": 1,
        "status": RunStatus.ACTIVE,
        "clock": clock,
        "boot_id": boot_id(),
        "created_utc": utc_now(),
        "stages": [],
        "report_mode": mode.value,
        "report_hook": report_hook,
        "proposal": str(proposal.resolve()) if proposal else None,
        "repository": str(Path.cwd()),
        "expensive_start_cutoff_monotonic": clock["start_monotonic"]
        + expensive_start_after,
        "training_stop_monotonic": clock["start_monotonic"] + training_stop_after,
        "expensive_start_cutoff_epoch": clock["start_epoch"] + expensive_start_after,
        "training_stop_epoch": clock["start_epoch"] + training_stop_after,
    }
    write_json(run_dir / "run_manifest.json", state)
    update_report(run_dir, state)
    append_log(
        run_dir,
        f"Started; deadline {clock['deadline_utc']}; preliminary report created.",
    )
    with (run_dir / "watchdog.log").open("a", encoding="utf-8") as log:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "watchdog", str(run_dir)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    state["watchdog_pid"] = worker.pid
    state["watchdog_proc_start"] = proc_start(worker.pid)
    write_json(run_dir / "run_manifest.json", state)
    append_log(run_dir, f"Separate watchdog pid={worker.pid} started.")
    print(f"Initialised {run_dir}; watchdog pid={worker.pid}", flush=True)


def refresh_watchdog(run_dir: Path, adopt_current_boot: bool = False) -> None:
    """Restart only this run's recorded watchdog after updating its code."""
    with locked(run_dir):
        state = read_json(run_dir / "run_manifest.json")
        old = state.get("watchdog_pid")
        old_start = state.get("watchdog_proc_start")
        old_fd = (
            _open_member(old, (old_start, old, old))
            if isinstance(old, int) and isinstance(old_start, int)
            else None
        )
        recorded_boot = state.get("boot_id")
        if recorded_boot is None:
            if not adopt_current_boot or old_fd is None:
                if old_fd is not None:
                    os.close(old_fd)
                raise SystemExit(
                    "Legacy manifest has no boot ID; verify its live watchdog and explicitly adopt the current boot"
                )
            command = Path(f"/proc/{old}/cmdline").read_bytes().split(b"\0")
            if (
                str(Path(__file__).resolve()).encode() not in command
                or b"watchdog" not in command
                or str(run_dir).encode() not in command
            ):
                os.close(old_fd)
                raise SystemExit("Recorded watchdog identity does not match this run")
            state["boot_id"] = boot_id()
            append_log(
                run_dir,
                "Verified live legacy watchdog; current boot explicitly adopted.",
            )
        elif recorded_boot != boot_id():
            if old_fd is not None:
                os.close(old_fd)
            raise SystemExit("Refusing to resume a run from a different system boot")
        # Start and record the replacement before stopping the old watchdog.
        # If this command is interrupted, at least one watchdog remains alive.
        write_json(run_dir / "run_manifest.json", state)
        with (run_dir / "watchdog.log").open("a", encoding="utf-8") as log:
            worker = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "watchdog",
                    str(run_dir),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        state["watchdog_pid"] = worker.pid
        state["watchdog_proc_start"] = proc_start(worker.pid)
        write_json(run_dir / "run_manifest.json", state)
        append_log(run_dir, f"Watchdog refreshed: old={old}, new={worker.pid}.")
        if old_fd is not None:
            try:
                pidfd_send_signal(old_fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
            finally:
                os.close(old_fd)
    print(f"Watchdog refreshed: pid={worker.pid}")


def finish(run_dir: Path) -> None:
    """Close supervision without changing an already terminal run status."""
    with locked(run_dir):
        state = read_json(run_dir / "run_manifest.json")
        if state.get("boot_id") != boot_id():
            raise SystemExit(
                "Run boot ID is missing or differs; cannot verify registered processes"
            )
        # A stage leader can exit while a descendant remains in its session.
        # Checking every registered group fails closed without signalling any
        # process whose ownership is no longer verifiable.
        running = [
            s["name"] for s in state["stages"] if group_has_live_members(s["pid"])
        ]
        running += [
            f"registered worker {s['pid']}"
            for s in registered_workers(run_dir)
            if group_has_live_members(s["pid"])
        ]
        if running:
            raise SystemExit(f"Cannot finish while registered stages run: {running}")
        current_status = RunStatus(state["status"])
        if current_status == RunStatus.ACTIVE:
            state["status"] = RunStatus.COMPLETE
        state["finished_utc"] = utc_now()
        state["elapsed_seconds"] = time.monotonic() - state["clock"]["start_monotonic"]
        write_json(run_dir / "run_manifest.json", state)
        update_report(run_dir, state)
        refresh_results_report(run_dir, state)
        append_log(run_dir, "Experiment finished; no registered stage remains active.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    init = subs.add_parser("init")
    init.add_argument("run_dir", type=Path)
    init.add_argument("--clock", required=True, type=Path)
    init.add_argument("--proposal", type=Path)
    init.add_argument(
        "--report-mode", choices=list(ReportMode), default=ReportMode.LEGACY
    )
    init.add_argument(
        "--report-hook", help="Optional report callback as module:function"
    )
    init.add_argument(
        "--expensive-start-after", type=float, default=DEFAULT_EXPENSIVE_START_AFTER
    )
    init.add_argument(
        "--training-stop-after", type=float, default=DEFAULT_TRAINING_STOP_AFTER
    )
    watch = subs.add_parser("watchdog")
    watch.add_argument("run_dir", type=Path)
    leader = subs.add_parser("stage-leader")
    leader.add_argument("--gate-fd", required=True, type=int)
    leader.add_argument("command", nargs=argparse.REMAINDER)
    refresh = subs.add_parser("refresh-watchdog")
    refresh.add_argument("run_dir", type=Path)
    refresh.add_argument("--adopt-current-boot", action="store_true")
    run = subs.add_parser("run")
    run.add_argument("run_dir", type=Path)
    run.add_argument("--name", required=True)
    run.add_argument("--kind", choices=list(StageKind), default=StageKind.NORMAL)
    run.add_argument("--timeout", type=float, required=True)
    run.add_argument("command", nargs=argparse.REMAINDER)
    done = subs.add_parser("finish")
    done.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.action == "init":
        initialise(
            args.run_dir.resolve(),
            args.clock,
            args.proposal,
            report_mode=args.report_mode,
            report_hook=args.report_hook,
            expensive_start_after=args.expensive_start_after,
            training_stop_after=args.training_stop_after,
        )
    elif args.action == "watchdog":
        watchdog(args.run_dir.resolve())
    elif args.action == "stage-leader":
        raise SystemExit(stage_leader(args.gate_fd, args.command))
    elif args.action == "refresh-watchdog":
        refresh_watchdog(args.run_dir.resolve(), args.adopt_current_boot)
    elif args.action == "run":
        raise SystemExit(
            run_stage(
                args.run_dir.resolve(), args.name, args.kind, args.timeout, args.command
            )
        )
    else:
        finish(args.run_dir.resolve())


if __name__ == "__main__":
    main()
