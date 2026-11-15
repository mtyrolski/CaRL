"""Watchdog loop for registered process groups and deadline enforcement."""

import time
from pathlib import Path

from scripts.experiment_support.supervisor.processes import _pidfd_exited
from scripts.experiment_support.supervisor.processes import boot_id
from scripts.experiment_support.supervisor.processes import close_tracked
from scripts.experiment_support.supervisor.processes import group_has_live_members
from scripts.experiment_support.supervisor.processes import stop_owned
from scripts.experiment_support.supervisor.processes import stop_tracked
from scripts.experiment_support.supervisor.processes import track_group
from scripts.experiment_support.supervisor.state import DEFAULT_TRAINING_STOP_AFTER
from scripts.experiment_support.supervisor.state import RunStatus
from scripts.experiment_support.supervisor.state import StageKind
from scripts.experiment_support.supervisor.state import StageStatus
from scripts.experiment_support.supervisor.state import append_log
from scripts.experiment_support.supervisor.state import locked
from scripts.experiment_support.supervisor.state import read_json
from scripts.experiment_support.supervisor.state import refresh_results_report
from scripts.experiment_support.supervisor.state import registered_workers
from scripts.experiment_support.supervisor.state import update_report
from scripts.experiment_support.supervisor.state import utc_now
from scripts.experiment_support.supervisor.state import write_json


def complete_requested(run_dir: Path, state: dict) -> bool:
    """Complete only after the requesting group and other live stages exit."""
    request_path = run_dir / "finish_request.json"
    if not request_path.exists():
        return False
    try:
        request = read_json(request_path)
    except (OSError, ValueError):
        append_log(run_dir, "Finish request could not be read; run remains active.")
        return False
    matches = [
        stage
        for stage in state["stages"]
        if stage.get("name") == request.get("stage")
        and stage.get("pid") == request.get("pid")
        and stage.get("proc_start") == request.get("proc_start")
    ]
    if len(matches) != 1 or request.get("boot_id") != state.get("boot_id"):
        append_log(run_dir, "Finish request ignored: stage or boot identity mismatch.")
        return False
    if group_has_live_members(request["pid"]):
        return False
    for stage in state["stages"]:
        if group_has_live_members(stage["pid"]):
            return False
    for worker in registered_workers(run_dir):
        if group_has_live_members(worker["pid"]):
            return False
    stage = matches[0]
    if stage.get("status") == StageStatus.RUNNING:
        # The outer caller may have disappeared, so no exit status is known.
        stage["status"] = StageStatus.LOST_OR_EXITED
        stage["finished_utc"] = utc_now()
    state["status"] = RunStatus.COMPLETE
    state["finished_utc"] = utc_now()
    state["elapsed_seconds"] = max(
        time.monotonic() - state["clock"]["start_monotonic"],
        time.time() - state["clock"]["start_epoch"],
    )
    append_log(run_dir, "Finish request completed after all registered groups exited.")
    return True


def watchdog(run_dir: Path) -> None:
    heartbeat = 0.0
    trackers: dict[str, dict[int, tuple[int, int]]] = {}
    try:
        while True:
            now = time.monotonic()
            epoch = time.time()
            delay = 2.0
            with locked(run_dir):
                state = read_json(run_dir / "run_manifest.json")
                if state["status"] != RunStatus.ACTIVE:
                    append_log(
                        run_dir, f"Watchdog exits: run status {state['status']}."
                    )
                    return
                if state.get("boot_id") != boot_id():
                    state["status"] = RunStatus.BOOT_MISMATCH
                    append_log(
                        run_dir,
                        "Watchdog stopped without signalling: system boot is missing or differs from the manifest.",
                    )
                    write_json(run_dir / "run_manifest.json", state)
                    update_report(run_dir, state)
                    return
                changed = False
                active_keys = set()
                for stage in state["stages"]:
                    key = f"stage:{stage['name']}:{stage['pid']}:{stage['proc_start']}"
                    if stage["status"] != StageStatus.RUNNING:
                        continue
                    active_keys.add(key)
                    handles = trackers.setdefault(key, {})
                    verified = {**stage, "boot_id": state["boot_id"]}
                    added = track_group(verified, handles)
                    if added:
                        append_log(
                            run_dir,
                            f"Tracking {stage['name']}: retained {len(handles)} process handles.",
                        )
                    stage_expired = (
                        now >= stage["stop_monotonic"]
                        or epoch >= stage.get("stop_epoch", float("inf"))
                        or epoch >= state["clock"]["deadline_epoch"]
                    )
                    training_expired = stage[
                        "kind"
                    ] == StageKind.TRAINING and epoch >= state.get(
                        "training_stop_epoch",
                        state["clock"]["start_epoch"] + DEFAULT_TRAINING_STOP_AFTER,
                    )
                    if stage_expired or training_expired:
                        stopped = stop_owned(
                            verified,
                            grace_seconds=max(
                                0.0,
                                min(
                                    3.0, state["clock"]["deadline_epoch"] - time.time()
                                ),
                            ),
                        )
                        captured_stopped = stop_tracked(handles)
                        stage["status"] = (
                            StageStatus.DEADLINE_STOPPED
                            if stopped and captured_stopped
                            else StageStatus.LOST_OR_EXITED
                        )
                        stage["finished_utc"] = utc_now()
                        append_log(
                            run_dir,
                            f"Watchdog: {stage['name']} {stage['status']}; tracked handles stopped={captured_stopped}.",
                        )
                        changed = True
                    else:
                        delay = min(
                            delay,
                            stage["stop_monotonic"] - now,
                            stage.get("stop_epoch", float("inf")) - epoch,
                        )
                for worker in registered_workers(run_dir):
                    key = f"worker:{worker['pid']}:{worker['proc_start']}"
                    active_keys.add(key)
                    handles = trackers.setdefault(key, {})
                    verified = {**worker, "boot_id": state["boot_id"]}
                    track_group(verified, handles)
                    if (
                        now
                        >= min(
                            worker["stop_monotonic"],
                            state["clock"]["deadline_monotonic"],
                        )
                        or epoch >= state["clock"]["deadline_epoch"]
                    ):
                        stopped = stop_owned(
                            verified,
                            grace_seconds=max(
                                0.0,
                                min(
                                    3.0, state["clock"]["deadline_epoch"] - time.time()
                                ),
                            ),
                        )
                        captured_stopped = stop_tracked(handles)
                        append_log(
                            run_dir,
                            f"Watchdog registered worker pid={worker['pid']}: group stopped={stopped}, tracked stopped={captured_stopped}.",
                        )
                    else:
                        delay = min(delay, worker["stop_monotonic"] - now)
                for key in list(trackers):
                    if key not in active_keys:
                        handles = trackers.pop(key)
                        if any(not _pidfd_exited(fd) for fd, _ in handles.values()):
                            stopped = stop_tracked(handles)
                            append_log(
                                run_dir,
                                f"Stopped residual tracked members of {key}: {stopped}.",
                            )
                        close_tracked(handles)
                if (
                    now >= state["clock"]["deadline_monotonic"]
                    or epoch >= state["clock"]["deadline_epoch"]
                ):
                    state["status"] = RunStatus.DEADLINE_REACHED
                    append_log(
                        run_dir,
                        "Hard deadline reached; registered process groups were checked for termination; see stage statuses.",
                    )
                    changed = True
                elif complete_requested(run_dir, state):
                    changed = True
                if now >= heartbeat:
                    remaining = max(
                        0,
                        min(
                            state["clock"]["deadline_monotonic"] - now,
                            state["clock"]["deadline_epoch"] - epoch,
                        ),
                    )
                    append_log(
                        run_dir,
                        f"Watchdog heartbeat: {remaining / 60:.1f} minutes remain.",
                    )
                    heartbeat = now + 600
                if changed:
                    write_json(run_dir / "run_manifest.json", state)
                    update_report(run_dir, state)
                    refresh_results_report(run_dir, state)
                if state["status"] != RunStatus.ACTIVE:
                    return
                delay = min(
                    delay,
                    state["clock"]["deadline_monotonic"] - time.monotonic(),
                    state["clock"]["deadline_epoch"] - time.time(),
                )
            time.sleep(max(0.02, delay))
    finally:
        for handles in trackers.values():
            close_tracked(handles)
