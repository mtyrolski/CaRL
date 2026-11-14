#!/usr/bin/env python3
"""Reconcile one vanished stage wrapper without signalling any process.

The guardian must itself be a registered normal supervisor stage. It changes
only a target stage whose recorded process group and wrapper are both absent
across three polls and a final check under the supervisor lock. Loss of /proc
metadata is treated as inconclusive. It must share the watchdog's PID
namespace, though this alone cannot establish complete /proc visibility. The
guardian does not signal processes; the watchdog may separately stop tracked
members owned by a stage once it leaves the running state.
"""

import argparse
import json
import os
import select
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from scripts.intent_experiment_supervisor import RunStatus
from scripts.intent_experiment_supervisor import StageStatus
from scripts.intent_experiment_supervisor import append_log
from scripts.intent_experiment_supervisor import boot_id
from scripts.intent_experiment_supervisor import locked
from scripts.intent_experiment_supervisor import proc_metadata
from scripts.intent_experiment_supervisor import read_json
from scripts.intent_experiment_supervisor import update_report
from scripts.intent_experiment_supervisor import utc_now
from scripts.intent_experiment_supervisor import write_json


class Presence(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class Reconcile(StrEnum):
    MARKED = "marked"
    DEFER = "defer"
    TERMINAL = "terminal"
    INACTIVE = "inactive"


@dataclass(frozen=True)
class Target:
    name: str
    leader_pid: int
    leader_start: int
    wrapper_pid: int
    wrapper_start: int | None
    boot: str
    watchdog_pid: int
    watchdog_start: int


def parse_stat(content: str) -> tuple[int, int]:
    """Extract process group and start ticks from one Linux /proc stat file."""
    if ")" not in content:
        raise ValueError("malformed process stat")
    fields = content.rsplit(")", 1)[1].split()
    if len(fields) <= 19:
        raise ValueError("incomplete process stat")
    return int(fields[2]), int(fields[19])


def wrapper_presence(pid: int, start: int | None, proc: Path = Path("/proc")) -> Presence:
    """Check a wrapper's identity, or require its PID to be absent when unknown."""
    if not proc.is_dir():
        return Presence.UNKNOWN
    try:
        _, observed = parse_stat((proc / str(pid) / "stat").read_text(encoding="ascii"))
    except FileNotFoundError:
        return Presence.ABSENT if not (proc / str(pid)).exists() else Presence.UNKNOWN
    except (OSError, ValueError, UnicodeError):
        return Presence.UNKNOWN
    if start is None:
        return Presence.PRESENT
    return Presence.PRESENT if observed == start else Presence.ABSENT


def group_presence(group: int, proc: Path = Path("/proc")) -> Presence:
    """Scan process groups conservatively; inaccessible entries are unknown."""
    unknown = False
    try:
        entries = list(proc.iterdir())
    except OSError:
        return Presence.UNKNOWN
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            observed_group, _ = parse_stat((entry / "stat").read_text(encoding="ascii"))
        except FileNotFoundError:
            # A vanished stat is inconclusive while its PID still exists.
            try:
                entry.stat()
            except FileNotFoundError:
                pass  # The next independent scan must also find no group.
            except OSError:
                unknown = True
            else:
                unknown = True
            continue
        except (OSError, ValueError, UnicodeError):
            unknown = True
            continue
        if observed_group == group:
            # Treat even zombies as present; the guardian never infers exit
            # status from them and waits for normal kernel reaping.
            return Presence.PRESENT
    return Presence.UNKNOWN if unknown else Presence.ABSENT


def target_stage(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [stage for stage in manifest.get("stages", []) if stage.get("name") == name]
    if len(matches) != 1:
        raise ValueError("target stage is missing or ambiguous")
    return matches[0]


def capture_target(manifest: dict[str, Any], name: str, wrapper_pid: int, wrapper_start: int | None,
                   current_boot: str) -> Target | None:
    if manifest.get("status") != RunStatus.ACTIVE:
        return None
    if manifest.get("boot_id") != current_boot:
        raise ValueError("supervisor boot identity differs")
    stage = target_stage(manifest, name)
    if stage.get("status") != StageStatus.RUNNING:
        return None
    leader, start = stage.get("pid"), stage.get("proc_start")
    watchdog_pid, watchdog_start = manifest.get("watchdog_pid"), manifest.get("watchdog_proc_start")
    if (not isinstance(leader, int) or isinstance(leader, bool) or leader <= 0
            or not isinstance(start, int) or isinstance(start, bool) or start <= 0
            or not isinstance(watchdog_pid, int) or isinstance(watchdog_pid, bool) or watchdog_pid <= 0
            or not isinstance(watchdog_start, int) or isinstance(watchdog_start, bool) or watchdog_start <= 0
            or wrapper_pid <= 0 or (wrapper_start is not None and wrapper_start <= 0) or wrapper_pid == leader):
        raise ValueError("stage and wrapper identities must be distinct positive integers")
    return Target(name, leader, start, wrapper_pid, wrapper_start, current_boot, watchdog_pid, watchdog_start)


def registered_guardian(manifest: dict[str, Any], group: int, session: int) -> bool:
    if group != session:
        return False
    matches = [stage for stage in manifest.get("stages", []) if stage.get("pid") == group
               and stage.get("kind") == "normal" and stage.get("status") == StageStatus.RUNNING]
    if len(matches) != 1:
        return False
    start = matches[0].get("proc_start")
    return isinstance(start, int) and proc_metadata(group) == (start, group, session)


def pidfd_alive(pid: int) -> bool:
    """Reject exited processes, including zombies, without sending a signal."""
    try:
        descriptor = os.pidfd_open(pid)
    except (AttributeError, OSError):
        return False
    try:
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        return not poller.poll(0)
    finally:
        os.close(descriptor)


def watchdog_visible(manifest: dict[str, Any], proc: Path = Path("/proc")) -> bool:
    """Require the recorded watchdog identity in this same PID namespace."""
    pid, start = manifest.get("watchdog_pid"), manifest.get("watchdog_proc_start")
    if (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            or not isinstance(start, int) or isinstance(start, bool) or start <= 0):
        return False
    metadata = proc_metadata(pid)
    if metadata is None or metadata[0] != start:
        return False
    try:
        same_namespace = os.path.samefile(proc / "self/ns/pid", proc / str(pid) / "ns/pid")
    except OSError:
        return False
    return same_namespace and pidfd_alive(pid) and proc_metadata(pid) == metadata


def reconcile(run: Path, target: Target, proc: Path = Path("/proc")) -> Reconcile:
    """Recheck identities under lock, retaining unknown exit code honestly."""
    with locked(run):
        manifest = read_json(run / "run_manifest.json")
        if manifest.get("status") != RunStatus.ACTIVE:
            return Reconcile.INACTIVE
        if manifest.get("boot_id") != target.boot or boot_id() != target.boot:
            raise ValueError("boot identity changed during guardian run")
        if (manifest.get("watchdog_pid") != target.watchdog_pid
                or manifest.get("watchdog_proc_start") != target.watchdog_start
                or not watchdog_visible(manifest)):
            return Reconcile.DEFER
        stage = target_stage(manifest, target.name)
        if stage.get("status") != StageStatus.RUNNING:
            return Reconcile.TERMINAL
        if stage.get("pid") != target.leader_pid or stage.get("proc_start") != target.leader_start:
            raise ValueError("target stage identity changed during guardian run")
        if (wrapper_presence(target.wrapper_pid, target.wrapper_start, proc) != Presence.ABSENT
                or group_presence(target.leader_pid, proc) != Presence.ABSENT):
            return Reconcile.DEFER
        if not watchdog_visible(manifest):
            return Reconcile.DEFER
        stage["status"] = StageStatus.LOST_OR_EXITED
        stage["exit_code"] = None
        stage["finished_utc"] = utc_now()
        identity = ("The original supervisor wrapper disappeared" if target.wrapper_start is not None
                    else "The wrapper PID was absent; its original start time was not recorded")
        stage["guardian_reason"] = (f"{identity} and its registered process group was absent across three polls "
                                    "and a final locked check; exit code is unknown.")
        stage["guardian_wrapper_pid"] = target.wrapper_pid
        stage["guardian_wrapper_start"] = target.wrapper_start
        write_json(run / "run_manifest.json", manifest)
        append_log(run, f"Guardian marked {target.name} lost_or_exited after wrapper and group disappeared; "
                        "exit code is unknown. The guardian sent no signals; the watchdog may stop owned tracked members.")
        update_report(run, manifest)
        return Reconcile.MARKED


def monitor(run: Path, target: Target, interval: float = 2.0, proc: Path = Path("/proc")) -> Reconcile:
    """Require three consecutive independent absence observations."""
    if interval < .2:
        raise ValueError("poll interval must be at least 0.2 seconds")
    consecutive = 0
    last_status: tuple[Presence, Presence] | None = None
    while True:
        manifest = read_json(run / "run_manifest.json")
        if manifest.get("status") != RunStatus.ACTIVE:
            return Reconcile.INACTIVE
        if manifest.get("boot_id") != target.boot or boot_id() != target.boot:
            raise ValueError("boot identity changed while monitoring")
        stage = target_stage(manifest, target.name)
        if stage.get("status") != StageStatus.RUNNING:
            return Reconcile.TERMINAL
        if stage.get("pid") != target.leader_pid or stage.get("proc_start") != target.leader_start:
            raise ValueError("target stage identity changed while monitoring")
        observed = (wrapper_presence(target.wrapper_pid, target.wrapper_start, proc),
                    group_presence(target.leader_pid, proc))
        if observed != last_status:
            print(json.dumps({"wrapper": observed[0], "group": observed[1], "target": target.name}), flush=True)
            last_status = observed
        consecutive = consecutive + 1 if observed == (Presence.ABSENT, Presence.ABSENT) else 0
        if consecutive >= 3:
            result = reconcile(run, target, proc)
            if result != Reconcile.DEFER:
                return result
            consecutive = 0
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--wrapper-pid", required=True, type=int)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--wrapper-start", type=int)
    identity.add_argument("--wrapper-start-unknown", action="store_true",
                          help="Require the entire wrapper PID to be absent; never infer its start time")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    run = args.run.resolve()
    with locked(run):
        manifest = read_json(run / "run_manifest.json")
        if not registered_guardian(manifest, os.getpgrp(), os.getsid(0)):
            raise ValueError("guardian must be a registered normal supervisor stage")
        if not watchdog_visible(manifest):
            raise ValueError("recorded watchdog must be visible in the guardian's PID namespace")
        target = capture_target(manifest, args.stage, args.wrapper_pid, args.wrapper_start, boot_id())
    if target is None:
        print("Target stage is already terminal or the run is inactive.")
        return 0
    result = monitor(run, target, args.interval)
    print(json.dumps({"status": result, "stage": target.name}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
