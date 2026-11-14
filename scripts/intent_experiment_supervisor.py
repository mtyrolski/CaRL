#!/usr/bin/env python3
"""Deadline supervisor for the local intent experiment.

All stage commands are started in new, registered process groups.  A separate
watchdog can stop those groups even if the invoking session disappears.
"""

import argparse
import contextlib
import ctypes
import datetime as dt
import fcntl
import importlib.util
import json
import os
import select
import signal
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path


class RunStatus(StrEnum):
    ACTIVE = "active"
    COMPLETE = "complete"
    DEADLINE_REACHED = "deadline_reached"
    BOOT_MISMATCH = "boot_mismatch"


class StageKind(StrEnum):
    NORMAL = "normal"
    EXPENSIVE = "expensive"
    TRAINING = "training"
    EVALUATION = "evaluation"


class StageStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    DEADLINE_STOPPED = "deadline_stopped"
    LOST_OR_EXITED = "lost_or_exited"


class ReportMode(StrEnum):
    LEGACY = "legacy"
    STATUS_ONLY = "status_only"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_write(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


@contextlib.contextmanager
def locked(run_dir: Path):
    with (run_dir / ".supervisor.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def append_log(run_dir: Path, message: str) -> None:
    with (run_dir / "decisions.log").open("a", encoding="utf-8") as out:
        out.write(f"{utc_now()} {message}\n")
        out.flush()


def boot_id() -> str:
    """Linux boot identifier; PID identities do not survive a reboot."""
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def proc_metadata(pid: int) -> tuple[int, int, int] | None:
    """Return process start ticks, process group and session, if accessible."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        return int(fields[19]), int(fields[2]), int(fields[3])
    except (OSError, IndexError, ValueError):
        return None


def proc_start(pid: int) -> int | None:
    metadata = proc_metadata(pid)
    return metadata[0] if metadata else None


def owned_process(stage: dict) -> bool:
    pid = stage.get("pid")
    start = stage.get("proc_start")
    if not isinstance(pid, int) or not isinstance(start, int):
        return False
    expected_boot = stage.get("boot_id")
    if expected_boot is not None and expected_boot != boot_id():
        return False
    return proc_metadata(pid) == (start, pid, pid)


def _pidfd_exited(fd: int) -> bool:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(0))


def _open_member(pid: int, metadata: tuple[int, int, int]) -> int | None:
    """Open a process handle only if its identity is stable across the open."""
    if proc_metadata(pid) != metadata:
        return None
    try:
        fd = os.pidfd_open(pid)
    except OSError:
        return None
    if proc_metadata(pid) != metadata or _pidfd_exited(fd):
        os.close(fd)
        return None
    return fd


def _group_members(group: int) -> list[tuple[int, tuple[int, int, int]]]:
    members = []
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            metadata = proc_metadata(pid)
            if metadata is not None and metadata[1:] == (group, group):
                members.append((pid, metadata))
    return members


def group_has_live_members(group: int) -> bool:
    """Fail closed when a registered process group still has a live member."""
    for pid, metadata in _group_members(group):
        try:
            fd = os.pidfd_open(pid)
        except OSError:
            if proc_metadata(pid) == metadata:
                return True
            continue
        try:
            if not _pidfd_exited(fd) and proc_metadata(pid) == metadata:
                return True
        finally:
            os.close(fd)
    return False


def track_group(stage: dict, handles: dict[int, tuple[int, int]]) -> int:
    """Retain pidfds observed while the registered group leader is alive."""
    if not owned_process(stage):
        return 0
    leader_pid = stage["pid"]
    start = stage["proc_start"]
    leader_entry = handles.get(leader_pid)
    if leader_entry is None or leader_entry[1] != start or _pidfd_exited(leader_entry[0]):
        if leader_entry is not None:
            os.close(leader_entry[0])
        leader_fd = _open_member(leader_pid, (start, leader_pid, leader_pid))
        if leader_fd is None:
            handles.pop(leader_pid, None)
            return 0
        handles[leader_pid] = leader_fd, start
    else:
        leader_fd = leader_entry[0]
    added = 0
    for pid, metadata in _group_members(leader_pid):
        if _pidfd_exited(leader_fd):
            break
        previous = handles.get(pid)
        if previous is not None and previous[1] == metadata[0] and not _pidfd_exited(previous[0]):
            continue
        fd = _open_member(pid, metadata)
        if fd is None:
            continue
        if _pidfd_exited(leader_fd):
            os.close(fd)
            break
        if previous is not None:
            os.close(previous[0])
        handles[pid] = fd, metadata[0]
        added += 1
    return added


def stop_tracked(handles: dict[int, tuple[int, int]]) -> bool:
    """Kill only previously verified processes, even if their leader has gone."""
    successful = True
    for fd, _ in handles.values():
        if not _pidfd_exited(fd):
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                successful = False
    confirm_at = time.monotonic() + 0.25
    while time.monotonic() < confirm_at and any(not _pidfd_exited(fd) for fd, _ in handles.values()):
        time.sleep(0.01)
    return successful and all(_pidfd_exited(fd) for fd, _ in handles.values())


def close_tracked(handles: dict[int, tuple[int, int]]) -> None:
    for fd, _ in handles.values():
        os.close(fd)
    handles.clear()


def stop_owned(stage: dict, grace_seconds: float = 3.0) -> bool:
    """Stop a verified session using pidfds; never signal a recycled PID.

    A live leader anchors the scan. Group members are frozen before TERM so a
    disappearing leader cannot orphan an ignoring child. If the leader was
    already gone, ownership cannot safely be reconstructed from a bare PGID.
    """
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal") or not owned_process(stage):
        return False
    pid = stage["pid"]
    leader = _open_member(pid, (stage["proc_start"], pid, pid))
    if leader is None:
        return False
    handles = {pid: leader}
    frozen: set[int] = set()
    complete_scan = False
    inaccessible = False
    try:
        try:
            signal.pidfd_send_signal(leader, signal.SIGSTOP)
        except OSError:
            return False
        frozen.add(leader)
        # A scan after the last newly frozen member is necessary: that member
        # could have forked between enumeration and SIGSTOP.
        for _ in range(32):
            if _pidfd_exited(leader):
                break
            added = 0
            for member_pid, metadata in _group_members(pid):
                if member_pid in handles or _pidfd_exited(leader):
                    continue
                fd = _open_member(member_pid, metadata)
                if fd is None:
                    inaccessible |= proc_metadata(member_pid) == metadata
                    continue
                if _pidfd_exited(leader):
                    os.close(fd)
                    break
                try:
                    signal.pidfd_send_signal(fd, signal.SIGSTOP)
                except OSError as exc:
                    inaccessible |= not isinstance(exc, ProcessLookupError)
                    os.close(fd)
                    continue
                handles[member_pid] = fd
                frozen.add(fd)
                added += 1
            if not added:
                complete_scan = not _pidfd_exited(leader) and not inaccessible
                break
        for fd in frozen:
            try:
                signal.pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                complete_scan = False
        for fd in frozen:
            try:
                signal.pidfd_send_signal(fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
            except OSError:
                complete_scan = False
        frozen.clear()
        grace = max(0.0, grace_seconds)
        stop_at = time.monotonic() + grace
        stop_epoch = time.time() + grace
        while (time.monotonic() < stop_at and time.time() < stop_epoch
               and any(not _pidfd_exited(fd) for fd in handles.values())):
            time.sleep(min(0.05, max(0.0, stop_at - time.monotonic())))
        for fd in handles.values():
            if not _pidfd_exited(fd):
                try:
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    complete_scan = False
        # Allow the kernel a short interval to deliver SIGKILL. Do not claim a
        # complete stop for an uninterruptible process or an incomplete scan.
        confirm_at = time.monotonic() + 0.25
        while time.monotonic() < confirm_at and any(not _pidfd_exited(fd) for fd in handles.values()):
            time.sleep(0.01)
        return complete_scan and all(_pidfd_exited(fd) for fd in handles.values())
    finally:
        # Never leave our group suspended after an unexpected exception.
        for fd in frozen:
            try:
                signal.pidfd_send_signal(fd, signal.SIGCONT)
            except OSError:
                pass
        for fd in handles.values():
            os.close(fd)


def registered_workers(run_dir: Path) -> list[dict]:
    """Read child groups explicitly registered by isolated annotation workers."""
    registry = run_dir / "terra_workers.jsonl"
    if not registry.exists():
        return []
    active: dict[tuple[int, int], dict] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            key = (int(item["pid"]), int(item["proc_start"]))
            if item.get("phase") == "started" and item.get("pgid") == item["pid"]:
                active[key] = item
            elif item.get("phase") == "finished":
                active.pop(key, None)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            # An append may be in progress; the next poll reads it again.
            continue
    return list(active.values())


def update_report(run_dir: Path, state: dict) -> None:
    report = run_dir / "REPORT.md"
    before = report.read_text(encoding="utf-8") if report.exists() else "# Intent experiment: preliminary report\n"
    start = "<!-- supervisor status begin -->"
    end = "<!-- supervisor status end -->"
    if start in before and end in before:
        a, rest = before.split(start, 1)
        _, b = rest.split(end, 1)
    else:
        a, b = before.rstrip() + "\n\n", "\n"
    lines = [
        start, "## Automatic supervision status", "", f"Updated: {utc_now()}.", f"Run state: **{state['status']}**.",
        f"Hard deadline: {state['clock']['deadline_utc']}.", ""
    ]
    if state["stages"]:
        lines += ["| Stage | Status | Exit code | Output |", "|---|---|---:|---|"]
        for stage in state["stages"]:
            code = stage.get("exit_code")
            lines.append(
                f"| {stage['name']} | {stage['status']} | {code if code is not None else '—'} | {stage['output']} |")
    else:
        lines.append("No experiment stages have run yet.")
    lines += ["", "This is an automatic checkpoint, not a claim of experimental results.", end]
    atomic_write(report, a + "\n".join(lines) + b)


def refresh_results_report(run_dir: Path, state: dict) -> None:
    """Update report content under the existing lock without taking it again."""
    try:
        mode = ReportMode(state.get("report_mode", ReportMode.LEGACY))
    except (TypeError, ValueError):
        append_log(run_dir, "Unknown report mode; refusing to replace the generic supervision status.")
        return
    if mode == ReportMode.STATUS_ONLY:
        return
    source = Path(__file__).with_name("report_intent_generator_experiment.py")
    if not source.is_file():
        return
    try:
        spec = importlib.util.spec_from_file_location("intent_results_report", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load report generator: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.refresh_report_unlocked(run_dir, state)
    except Exception as exc:
        append_log(run_dir, f"Could not refresh report content: {type(exc).__name__}: {exc}")


def request_finish(run_dir: Path) -> None:
    """Ask the watchdog to finish once this registered group has exited."""
    run_dir = run_dir.resolve()
    with locked(run_dir):
        state = read_json(run_dir / "run_manifest.json")
        if state.get("status") != RunStatus.ACTIVE or state.get("boot_id") != boot_id():
            raise ValueError("Run is inactive or its boot identity cannot be verified")
        group = os.getpgrp()
        matches = [
            stage for stage in state["stages"]
            if stage.get("status") == StageStatus.RUNNING and stage.get("pid") == group and owned_process({
                **stage, "boot_id": state["boot_id"]
            })
        ]
        if len(matches) != 1:
            raise ValueError("Caller is not in exactly one registered live stage group")
        stage = matches[0]
        request = {
            "stage": stage["name"],
            "pid": group,
            "proc_start": stage["proc_start"],
            "boot_id": state["boot_id"],
            "requested_utc": utc_now()
        }
        write_json(run_dir / "finish_request.json", request)
        append_log(run_dir, f"Finish requested by stage {stage['name']}; awaiting all groups to exit.")
        update_report(run_dir, state)
        refresh_results_report(run_dir, state)


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
        stage for stage in state["stages"] if stage.get("name") == request.get("stage")
        and stage.get("pid") == request.get("pid") and stage.get("proc_start") == request.get("proc_start")
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
    state["elapsed_seconds"] = max(time.monotonic() - state["clock"]["start_monotonic"],
                                   time.time() - state["clock"]["start_epoch"])
    append_log(run_dir, "Finish request completed after all registered groups exited.")
    return True


def initialise(run_dir: Path,
               clock_path: Path,
               proposal: Path | None = None,
               *,
               report_mode: ReportMode | str = ReportMode.LEGACY) -> None:
    mode = ReportMode(report_mode)
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
        "proposal": str(proposal.resolve()) if proposal else None,
        "repository": str(Path.cwd()),
        "expensive_start_cutoff_monotonic": clock["start_monotonic"] + 13500,
        "training_stop_monotonic": clock["start_monotonic"] + 14400,
        "expensive_start_cutoff_epoch": clock["start_epoch"] + 13500,
        "training_stop_epoch": clock["start_epoch"] + 14400
    }
    write_json(run_dir / "run_manifest.json", state)
    update_report(run_dir, state)
    append_log(run_dir, f"Started; deadline {clock['deadline_utc']}; preliminary report created.")
    with (run_dir / "watchdog.log").open("a", encoding="utf-8") as log:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "watchdog",
             str(run_dir)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True)
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
        old_fd = _open_member(old,
                              (old_start, old, old)) if isinstance(old, int) and isinstance(old_start, int) else None
        recorded_boot = state.get("boot_id")
        if recorded_boot is None:
            if not adopt_current_boot or old_fd is None:
                if old_fd is not None:
                    os.close(old_fd)
                raise SystemExit(
                    "Legacy manifest has no boot ID; verify its live watchdog and explicitly adopt the current boot")
            command = Path(f"/proc/{old}/cmdline").read_bytes().split(b"\0")
            if str(Path(__file__).resolve()).encode() not in command or b"watchdog" not in command or str(
                    run_dir).encode() not in command:
                os.close(old_fd)
                raise SystemExit("Recorded watchdog identity does not match this run")
            state["boot_id"] = boot_id()
            append_log(run_dir, "Verified live legacy watchdog; current boot explicitly adopted.")
        elif recorded_boot != boot_id():
            if old_fd is not None:
                os.close(old_fd)
            raise SystemExit("Refusing to resume a run from a different system boot")
        # Start and record the replacement before stopping the old watchdog.
        # If this command is interrupted, at least one watchdog remains alive.
        write_json(run_dir / "run_manifest.json", state)
        with (run_dir / "watchdog.log").open("a", encoding="utf-8") as log:
            worker = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "watchdog",
                 str(run_dir)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True)
        state["watchdog_pid"] = worker.pid
        state["watchdog_proc_start"] = proc_start(worker.pid)
        write_json(run_dir / "run_manifest.json", state)
        append_log(run_dir, f"Watchdog refreshed: old={old}, new={worker.pid}.")
        if old_fd is not None:
            try:
                signal.pidfd_send_signal(old_fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
            finally:
                os.close(old_fd)
    print(f"Watchdog refreshed: pid={worker.pid}")


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
                    append_log(run_dir, f"Watchdog exits: run status {state['status']}.")
                    return
                if state.get("boot_id") != boot_id():
                    state["status"] = RunStatus.BOOT_MISMATCH
                    append_log(
                        run_dir,
                        "Watchdog stopped without signalling: system boot is missing or differs from the manifest.")
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
                        append_log(run_dir, f"Tracking {stage['name']}: retained {len(handles)} process handles.")
                    stage_expired = (now >= stage["stop_monotonic"] or epoch >= stage.get("stop_epoch", float("inf"))
                                     or epoch >= state["clock"]["deadline_epoch"])
                    training_expired = stage[
                        "kind"] == StageKind.TRAINING and epoch >= state["clock"]["start_epoch"] + 14400
                    if stage_expired or training_expired:
                        stopped = stop_owned(verified,
                                             grace_seconds=max(0.0,
                                                               min(3.0,
                                                                   state["clock"]["deadline_epoch"] - time.time())))
                        captured_stopped = stop_tracked(handles)
                        stage[
                            "status"] = StageStatus.DEADLINE_STOPPED if stopped and captured_stopped else StageStatus.LOST_OR_EXITED
                        stage["finished_utc"] = utc_now()
                        append_log(
                            run_dir,
                            f"Watchdog: {stage['name']} {stage['status']}; tracked handles stopped={captured_stopped}.")
                        changed = True
                    else:
                        delay = min(delay, stage["stop_monotonic"] - now, stage.get("stop_epoch", float("inf")) - epoch)
                for worker in registered_workers(run_dir):
                    key = f"worker:{worker['pid']}:{worker['proc_start']}"
                    active_keys.add(key)
                    handles = trackers.setdefault(key, {})
                    verified = {**worker, "boot_id": state["boot_id"]}
                    track_group(verified, handles)
                    if (now >= min(worker["stop_monotonic"], state["clock"]["deadline_monotonic"])
                            or epoch >= state["clock"]["deadline_epoch"]):
                        stopped = stop_owned(verified,
                                             grace_seconds=max(0.0,
                                                               min(3.0,
                                                                   state["clock"]["deadline_epoch"] - time.time())))
                        captured_stopped = stop_tracked(handles)
                        append_log(
                            run_dir,
                            f"Watchdog annotation worker pid={worker['pid']}: group stopped={stopped}, tracked stopped={captured_stopped}."
                        )
                    else:
                        delay = min(delay, worker["stop_monotonic"] - now)
                for key in list(trackers):
                    if key not in active_keys:
                        handles = trackers.pop(key)
                        if any(not _pidfd_exited(fd) for fd, _ in handles.values()):
                            stopped = stop_tracked(handles)
                            append_log(run_dir, f"Stopped residual tracked members of {key}: {stopped}.")
                        close_tracked(handles)
                if now >= state["clock"]["deadline_monotonic"] or epoch >= state["clock"]["deadline_epoch"]:
                    state["status"] = RunStatus.DEADLINE_REACHED
                    append_log(
                        run_dir,
                        "Hard deadline reached; registered process groups were checked for termination; see stage statuses."
                    )
                    changed = True
                elif complete_requested(run_dir, state):
                    changed = True
                if now >= heartbeat:
                    remaining = max(
                        0, min(state["clock"]["deadline_monotonic"] - now, state["clock"]["deadline_epoch"] - epoch))
                    append_log(run_dir, f"Watchdog heartbeat: {remaining / 60:.1f} minutes remain.")
                    heartbeat = now + 600
                if changed:
                    write_json(run_dir / "run_manifest.json", state)
                    update_report(run_dir, state)
                    refresh_results_report(run_dir, state)
                if state["status"] != RunStatus.ACTIVE:
                    return
                delay = min(delay, state["clock"]["deadline_monotonic"] - time.monotonic(),
                            state["clock"]["deadline_epoch"] - time.time())
            time.sleep(max(0.02, delay))
    finally:
        for handles in trackers.values():
            close_tracked(handles)


def compute_stop(state: dict, kind: str, timeout: float, now: float, epoch: float | None = None) -> float:
    epoch = time.time() if epoch is None else epoch
    clock = state["clock"]
    if (state["status"] != RunStatus.ACTIVE or now >= clock["deadline_monotonic"]
            or epoch >= clock.get("deadline_epoch", float("inf"))):
        raise ValueError("Run is not active or has reached its deadline")
    expensive_epoch = clock.get("start_epoch", float("inf")) + 13500
    if (kind in {StageKind.EXPENSIVE, StageKind.TRAINING}
            and (now >= state["expensive_start_cutoff_monotonic"] or epoch >= expensive_epoch)):
        raise ValueError("Too late to start an expensive stage")
    stop = min(now + timeout, clock["deadline_monotonic"], now + clock.get("deadline_epoch", float("inf")) - epoch)
    if kind == StageKind.TRAINING:
        stop = min(stop, state["training_stop_monotonic"], now + clock.get("start_epoch", float("inf")) + 14400 - epoch)
    return float(stop)


def stage_leader(gate_fd: int, command: list[str]) -> int:
    """Keep the registered session alive until its adopted descendants exit."""
    try:
        permitted = os.read(gate_fd, 1) == b"1"
    finally:
        os.close(gate_fd)
    if not permitted:
        return 125
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A stage command is required")
    # Linux subreapers adopt orphaned descendants, including double-forks.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:    # PR_SET_CHILD_SUBREAPER
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))

    def keep_leader_alive(_signum: int, _frame: object) -> None:
        # The watchdog signals every captured member; this process remains to
        # retain ownership and adopt children while they shut down.
        return None

    signal.signal(signal.SIGTERM, keep_leader_alive)
    child = subprocess.Popen(command, close_fds=True)
    code = child.wait()
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break
        except InterruptedError:
            continue
    return code if code >= 0 else 128 - code


def run_stage(run_dir: Path, name: str, kind: str, timeout: float, command: list[str]) -> int:
    if not command:
        raise SystemExit("A stage command is required after --")
    if command[0] == "--":
        command = command[1:]
    if not command or timeout <= 0:
        raise SystemExit("A non-empty command and positive timeout are required")
    with locked(run_dir):
        state = read_json(run_dir / "run_manifest.json")
        if state.get("boot_id") != boot_id():
            raise SystemExit("Run boot ID is missing or differs; do not reuse its registered PIDs")
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
                child = subprocess.Popen([
                    sys.executable,
                    str(Path(__file__).resolve()), "stage-leader", "--gate-fd",
                    str(read_fd), "--", *command
                ],
                                         stdin=subprocess.DEVNULL,
                                         stdout=log,
                                         stderr=subprocess.STDOUT,
                                         start_new_session=True,
                                         close_fds=True,
                                         pass_fds=(read_fd,))
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
                "exit_code": None
            }
            state["stages"].append(stage)
            write_json(run_dir / "run_manifest.json", state)
            update_report(run_dir, state)
            append_log(run_dir, f"Started {name}, pid={child.pid}, timeout={timeout:.0f}s, kind={kind}.")
            os.write(write_fd, b"1")
        finally:
            if read_fd >= 0:
                os.close(read_fd)
            os.close(write_fd)
    while child.poll() is None:
        if time.monotonic() >= stop or time.time() >= stop_epoch:
            remaining = min(state["clock"]["deadline_monotonic"] - time.monotonic(),
                            state["clock"]["deadline_epoch"] - time.time())
            stop_owned(stage, grace_seconds=max(0.0, min(3.0, remaining)))
            break
        time.sleep(0.5)
    code = child.wait()
    with locked(run_dir):
        state = read_json(run_dir / "run_manifest.json")
        recorded = next(s for s in state["stages"] if s["name"] == name)
        if recorded["status"] == StageStatus.RUNNING:
            timed_out = time.monotonic() >= stop or time.time() >= stop_epoch
            recorded["status"] = StageStatus.COMPLETED if code == 0 else (
                StageStatus.TIMEOUT if timed_out else StageStatus.FAILED)
        recorded["exit_code"] = code
        recorded["finished_utc"] = utc_now()
        write_json(run_dir / "run_manifest.json", state)
        update_report(run_dir, state)
        if code != 0:
            refresh_results_report(run_dir, state)
        append_log(run_dir, f"Finished {name}, status={recorded['status']}, exit={code}.")
    print(f"{name}: {recorded['status']} exit={code} log={output}", flush=True)
    return code if code >= 0 else 128 - code


def finish(run_dir: Path) -> None:
    """Close supervision without changing an already terminal run status."""
    with locked(run_dir):
        state = read_json(run_dir / "run_manifest.json")
        if state.get("boot_id") != boot_id():
            raise SystemExit("Run boot ID is missing or differs; cannot verify registered processes")
        # A stage leader can exit while a descendant remains in its session.
        # Checking every registered group fails closed without signalling any
        # process whose ownership is no longer verifiable.
        running = [s["name"] for s in state["stages"] if group_has_live_members(s["pid"])]
        running += [
            f"annotation worker {s['pid']}" for s in registered_workers(run_dir) if group_has_live_members(s["pid"])
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
    init.add_argument("--report-mode", choices=list(ReportMode), default=ReportMode.LEGACY)
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
        initialise(args.run_dir.resolve(), args.clock, args.proposal, report_mode=args.report_mode)
    elif args.action == "watchdog":
        watchdog(args.run_dir.resolve())
    elif args.action == "stage-leader":
        raise SystemExit(stage_leader(args.gate_fd, args.command))
    elif args.action == "refresh-watchdog":
        refresh_watchdog(args.run_dir.resolve(), args.adopt_current_boot)
    elif args.action == "run":
        raise SystemExit(run_stage(args.run_dir.resolve(), args.name, args.kind, args.timeout, args.command))
    else:
        finish(args.run_dir.resolve())


if __name__ == "__main__":
    main()
