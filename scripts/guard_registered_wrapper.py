#!/usr/bin/env python3
"""Reconcile one vanished stage wrapper without signalling any process.

The guardian must itself be a registered normal supervisor stage. It changes
only a target stage whose exact recorded process group and original wrapper are
both absent across three polls and a final check under the supervisor lock.
Loss of /proc visibility is inconclusive and can never mark a stage finished.
"""

import argparse
import json
import os
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
    wrapper_start: int
    boot: str


def parse_stat(content: str) -> tuple[int, int]:
    """Extract process group and start ticks from one Linux /proc stat file."""
    if ")" not in content:
        raise ValueError("malformed process stat")
    fields = content.rsplit(")", 1)[1].split()
    if len(fields) <= 19:
        raise ValueError("incomplete process stat")
    return int(fields[2]), int(fields[19])


def wrapper_presence(pid: int, start: int, proc: Path = Path("/proc")) -> Presence:
    """Distinguish an absent/reused wrapper from unavailable process metadata."""
    if not proc.is_dir():
        return Presence.UNKNOWN
    try:
        _, observed = parse_stat((proc / str(pid) / "stat").read_text(encoding="ascii"))
    except FileNotFoundError:
        return Presence.ABSENT if not (proc / str(pid)).exists() else Presence.UNKNOWN
    except (OSError, ValueError, UnicodeError):
        return Presence.UNKNOWN
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
            # A process may disappear during the scan.  The next independent
            # scan must also find no live group before we can reconcile.
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


def capture_target(manifest: dict[str, Any], name: str, wrapper_pid: int, wrapper_start: int,
                   current_boot: str) -> Target | None:
    if manifest.get("status") != RunStatus.ACTIVE:
        return None
    if manifest.get("boot_id") != current_boot:
        raise ValueError("supervisor boot identity differs")
    stage = target_stage(manifest, name)
    if stage.get("status") != StageStatus.RUNNING:
        return None
    leader, start = stage.get("pid"), stage.get("proc_start")
    if (not isinstance(leader, int) or isinstance(leader, bool) or leader <= 0
            or not isinstance(start, int) or isinstance(start, bool) or start <= 0
            or wrapper_pid <= 0 or wrapper_start <= 0 or wrapper_pid == leader):
        raise ValueError("stage and wrapper identities must be distinct positive integers")
    return Target(name, leader, start, wrapper_pid, wrapper_start, current_boot)


def registered_guardian(manifest: dict[str, Any], group: int, session: int) -> bool:
    if group != session:
        return False
    matches = [stage for stage in manifest.get("stages", []) if stage.get("pid") == group
               and stage.get("kind") == "normal" and stage.get("status") == StageStatus.RUNNING]
    return len(matches) == 1


def reconcile(run: Path, target: Target, proc: Path = Path("/proc")) -> Reconcile:
    """Recheck identities under lock, retaining unknown exit code honestly."""
    with locked(run):
        manifest = read_json(run / "run_manifest.json")
        if manifest.get("status") != RunStatus.ACTIVE:
            return Reconcile.INACTIVE
        if manifest.get("boot_id") != target.boot or boot_id() != target.boot:
            raise ValueError("boot identity changed during guardian run")
        stage = target_stage(manifest, target.name)
        if stage.get("status") != StageStatus.RUNNING:
            return Reconcile.TERMINAL
        if stage.get("pid") != target.leader_pid or stage.get("proc_start") != target.leader_start:
            raise ValueError("target stage identity changed during guardian run")
        if (wrapper_presence(target.wrapper_pid, target.wrapper_start, proc) != Presence.ABSENT
                or group_presence(target.leader_pid, proc) != Presence.ABSENT):
            return Reconcile.DEFER
        stage["status"] = StageStatus.LOST_OR_EXITED
        stage["exit_code"] = None
        stage["finished_utc"] = utc_now()
        stage["guardian_reason"] = ("The original supervisor wrapper disappeared and its registered process group "
                                    "was absent across three polls and a final locked check; exit code is unknown.")
        stage["guardian_wrapper_pid"] = target.wrapper_pid
        stage["guardian_wrapper_start"] = target.wrapper_start
        write_json(run / "run_manifest.json", manifest)
        append_log(run, f"Guardian marked {target.name} lost_or_exited after wrapper and group disappeared; "
                        "exit code is unknown. No process was signalled.")
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
    parser.add_argument("--wrapper-start", required=True, type=int)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    run = args.run.resolve()
    with locked(run):
        manifest = read_json(run / "run_manifest.json")
        if not registered_guardian(manifest, os.getpgrp(), os.getsid(0)):
            raise ValueError("guardian must be a registered normal supervisor stage")
        target = capture_target(manifest, args.stage, args.wrapper_pid, args.wrapper_start, boot_id())
    if target is None:
        print("Target stage is already terminal or the run is inactive.")
        return 0
    result = monitor(run, target, args.interval)
    print(json.dumps({"status": result, "stage": target.name}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
