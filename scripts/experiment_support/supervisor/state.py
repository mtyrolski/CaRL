"""Supervisor state, deadlines and report checkpoints."""

import contextlib
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path

from scripts.experiment_support.supervisor.processes import boot_id
from scripts.experiment_support.supervisor.processes import proc_start
from scripts.experiment_support.supervisor.processes import stop_owned

DEFAULT_EXPENSIVE_START_AFTER = 13500
DEFAULT_TRAINING_STOP_AFTER = 14400


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
    HOOK = "hook"


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


def registered_workers(run_dir: Path) -> list[dict]:
    """Read explicitly registered child groups, including older registry files."""
    active: dict[tuple[int, int], dict] = {}
    for name in ("workers.jsonl", "terra_workers.jsonl"):
        registry = run_dir / name
        if not registry.exists():
            continue
        registered: dict[tuple[int, int], dict] = {}
        for line in registry.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                key = (int(item["pid"]), int(item["proc_start"]))
                if item.get("phase") == "started" and item.get("pgid") == item["pid"]:
                    registered[key] = item
                elif item.get("phase") == "finished":
                    registered.pop(key, None)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                # An append may be in progress; the next poll reads it again.
                continue
        active.update(registered)
    return list(active.values())


def update_report(run_dir: Path, state: dict) -> None:
    report = run_dir / "REPORT.md"
    before = (
        report.read_text(encoding="utf-8")
        if report.exists()
        else "# Intent experiment: preliminary report\n"
    )
    start = "<!-- supervisor status begin -->"
    end = "<!-- supervisor status end -->"
    if start in before and end in before:
        a, rest = before.split(start, 1)
        _, b = rest.split(end, 1)
    else:
        a, b = before.rstrip() + "\n\n", "\n"
    lines = [
        start,
        "## Automatic supervision status",
        "",
        f"Updated: {utc_now()}.",
        f"Run state: **{state['status']}**.",
        f"Hard deadline: {state['clock']['deadline_utc']}.",
        "",
    ]
    if state["stages"]:
        lines += ["| Stage | Status | Exit code | Output |", "|---|---|---:|---|"]
        for stage in state["stages"]:
            code = stage.get("exit_code")
            lines.append(
                f"| {stage['name']} | {stage['status']} | {code if code is not None else '—'} | {stage['output']} |"
            )
    else:
        lines.append("No experiment stages have run yet.")
    lines += [
        "",
        "This is an automatic checkpoint, not a claim of experimental results.",
        end,
    ]
    atomic_write(report, a + "\n".join(lines) + b)


def validate_report_hook(hook: str | None) -> None:
    """Accept only an explicit Python module and callable name."""
    if hook is None:
        return
    if not isinstance(hook, str):
        raise ValueError("Report hook must be specified as module:function")
    module, separator, function = hook.partition(":")
    if (
        not separator
        or not function.isidentifier()
        or not all(part.isidentifier() for part in module.split("."))
    ):
        raise ValueError("Report hook must be specified as module:function")


def refresh_results_report(run_dir: Path, state: dict) -> None:
    """Run an explicit optional hook in a bounded, separately owned process."""
    try:
        mode = ReportMode(state.get("report_mode", ReportMode.LEGACY))
    except (TypeError, ValueError):
        append_log(
            run_dir,
            "Unknown report mode; refusing to replace the generic supervision status.",
        )
        return
    hook = state.get("report_hook")
    if mode == ReportMode.STATUS_ONLY or hook is None:
        return
    try:
        validate_report_hook(hook)
        timeout = min(2.0, max(0.0, float(state.get("report_timeout_seconds", 2.0))))
        clock = state.get("clock", {})
        if isinstance(clock, dict):
            if isinstance(clock.get("deadline_epoch"), (int, float)):
                timeout = min(timeout, max(0.0, clock["deadline_epoch"] - time.time()))
            if isinstance(clock.get("deadline_monotonic"), (int, float)):
                timeout = min(
                    timeout, max(0.0, clock["deadline_monotonic"] - time.monotonic())
                )
        if timeout <= 0:
            append_log(
                run_dir, "Report hook skipped: no time remains before the deadline."
            )
            return
        command = (
            "import importlib,json,pathlib,sys; "
            "module,name=sys.argv[1].split(':',1); "
            "getattr(importlib.import_module(module),name)(pathlib.Path(sys.argv[2]),json.load(sys.stdin))"
        )
        env = os.environ.copy()
        root = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (root, env.get("PYTHONPATH", "")))
        )
        with (run_dir / "report_hook.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                [sys.executable, "-c", command, hook, str(run_dir)],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=env,
            )
            start = proc_start(child.pid)
            try:
                child.communicate(json.dumps(state), timeout=timeout)
            except subprocess.TimeoutExpired:
                if start is not None:
                    try:
                        stop_owned(
                            {
                                "pid": child.pid,
                                "proc_start": start,
                                "boot_id": boot_id(),
                            },
                            grace_seconds=0,
                        )
                    except OSError:
                        pass
                if child.poll() is None:
                    child.kill()
                try:
                    child.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    append_log(
                        run_dir,
                        "Report hook timed out and did not exit after termination.",
                    )
                else:
                    append_log(run_dir, "Report hook timed out; supervision continues.")
                return
            if child.returncode:
                append_log(
                    run_dir,
                    f"Report hook exited with status {child.returncode}; supervision continues.",
                )
    except (OSError, TypeError, ValueError) as exc:
        append_log(
            run_dir, f"Could not refresh report content: {type(exc).__name__}: {exc}"
        )


def compute_stop(
    state: dict, kind: str, timeout: float, now: float, epoch: float | None = None
) -> float:
    epoch = time.time() if epoch is None else epoch
    clock = state["clock"]
    if (
        state["status"] != RunStatus.ACTIVE
        or now >= clock["deadline_monotonic"]
        or epoch >= clock.get("deadline_epoch", float("inf"))
    ):
        raise ValueError("Run is not active or has reached its deadline")
    expensive_epoch = state.get(
        "expensive_start_cutoff_epoch",
        clock.get("start_epoch", float("inf")) + DEFAULT_EXPENSIVE_START_AFTER,
    )
    if kind in {StageKind.EXPENSIVE, StageKind.TRAINING} and (
        now >= state["expensive_start_cutoff_monotonic"] or epoch >= expensive_epoch
    ):
        raise ValueError("Too late to start an expensive stage")
    stop = min(
        now + timeout,
        clock["deadline_monotonic"],
        now + clock.get("deadline_epoch", float("inf")) - epoch,
    )
    if kind == StageKind.TRAINING:
        stop = min(
            stop,
            state["training_stop_monotonic"],
            now
            + state.get(
                "training_stop_epoch",
                clock.get("start_epoch", float("inf")) + DEFAULT_TRAINING_STOP_AFTER,
            )
            - epoch,
        )
    return float(stop)
