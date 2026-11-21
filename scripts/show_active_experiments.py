"""Inspect locally supervised experiments without starting or changing them."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.experiment_support.supervisor.processes import boot_id, owned_process, proc_start

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    """Read an optional JSON object that may be updated by another process."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def read_results(run: Path) -> list[dict[str, str]]:
    """Read completed groups; a missing denominator is not a zero result."""
    try:
        with (run / "summary.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return [row for row in rows if int(row.get("completed") or 0) > 0]
    except (OSError, ValueError, csv.Error):
        return []


def inspect_run(path: Path, current_boot: str | None = None) -> dict[str, Any]:
    """Describe a run and verify process identities against its saved boot and start ticks."""
    run = path.parent
    manifest = read_json(path)
    status = read_json(run / "status.json") or read_json(run / "progress.json")
    current_boot = current_boot or boot_id()
    same_boot = manifest.get("boot_id") == current_boot
    raw_stages = manifest.get("stages")
    stages = raw_stages if isinstance(raw_stages, list) else []
    verified = [s for s in stages if isinstance(s, dict) and s.get("status") == "running"
                and same_boot and owned_process({**s, "boot_id": current_boot})]
    watchdog_pid = manifest.get("watchdog_pid")
    watchdog_start = manifest.get("watchdog_proc_start")
    watchdog_alive = (same_boot and isinstance(watchdog_pid, int) and isinstance(watchdog_start, int)
                      and proc_start(watchdog_pid) == watchdog_start)
    if manifest.get("status") != "active":
        state = str(manifest.get("status", "unknown"))
    elif verified:
        state = "running"
    elif watchdog_alive:
        state = "waiting" if not any(s.get("status") == "running" for s in stages if isinstance(s, dict)) else "stage not visible"
    else:
        state = "not verified"
    stage = verified[-1] if verified else {}
    report = next((str(p) for name in ("RESULTS.md", "REPORT.md", "report.md") if (p := run / name).is_file()), None)
    updated = status.get("updated_epoch")
    if not isinstance(updated, (int, float)):
        updated = (run / "status.json").stat().st_mtime if (run / "status.json").exists() else None
    raw_clock = manifest.get("clock")
    clock = raw_clock if isinstance(raw_clock, dict) else {}
    return {
        "name": run.name, "directory": str(run), "state": state,
        "manifest_status": manifest.get("status"), "stage": stage.get("name"), "pid": stage.get("pid"),
        "watchdog_pid": watchdog_pid if watchdog_alive else None,
        "completed": status.get("completed", status.get("recorded")), "planned": status.get("planned"),
        "detail": status.get("reason", status.get("message")), "updated_epoch": updated,
        "deadline_epoch": clock.get("deadline_epoch"), "report": report, "log": stage.get("output"),
        "results": read_results(run),
    }


def discover(analysis: Path, include_all: bool = False) -> list[dict[str, Any]]:
    """Inspect manifest-based runs up to two levels below the analysis directory."""
    if not analysis.is_dir():
        return []
    paths = sorted(set(analysis.glob("*/run_manifest.json")) | set(analysis.glob("*/*/run_manifest.json")))
    current_boot = boot_id()
    runs = [inspect_run(path, current_boot) for path in paths]
    return runs if include_all else [run for run in runs if run["state"] in {"running", "waiting", "stage not visible"}]


def display(runs: list[dict[str, Any]]) -> None:
    if not runs:
        print("No active supervised experiments found under analysis/.")
        print("Only runs with a CaRL supervisor manifest are included; use --all to inspect other recorded states.")
        return
    print(f"Supervised experiments: {len(runs)}\n")
    for run in runs:
        print(f"{run['name']}  [{run['state']}]")
        if run["stage"]:
            print(f"  Stage: {run['stage']}  |  PID: {run['pid']}")
        completed, planned = run["completed"], run["planned"]
        if completed is not None:
            progress = f"{completed}/{planned}" if planned is not None else str(completed)
            print(f"  Progress: {progress}")
        if run["detail"]:
            print(f"  Status: {run['detail']}")
        if isinstance(run["updated_epoch"], (int, float)):
            print(f"  Updated: {max(0, int(time.time() - run['updated_epoch']))} s ago")
        if isinstance(run["deadline_epoch"], (int, float)):
            deadline = datetime.fromtimestamp(run["deadline_epoch"]).astimezone().strftime("%Y-%m-%d %H:%M %Z")
            print(f"  Deadline: {deadline}")
        for row in run["results"]:
            group = " / ".join(str(row[key]) for key in ("family", "method") if row.get(key))
            offset = f"k={row['k']}" if row.get("k") else ""
            print(f"  {group} {offset}: {row.get('hits', '?')}/{row['completed']}")
        if run["report"]:
            print(f"  Report: {run['report']}")
        if run["log"]:
            print(f"  Log: {run['log']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis")
    parser.add_argument("--all", action="store_true", help="Include stopped and unverified runs")
    parser.add_argument("--json", action="store_true", help="Write machine-readable output")
    args = parser.parse_args()
    runs = discover(args.analysis_dir.resolve(), include_all=args.all)
    if args.json:
        print(json.dumps(runs, indent=2))
    else:
        display(runs)


if __name__ == "__main__":
    main()
