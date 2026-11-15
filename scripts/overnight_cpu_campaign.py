"""Inspect the persistent deadline and stage status of the local CPU campaign.

The active pointer and the original clock remain authoritative across later
stage prompts. This command never starts work or changes an existing deadline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTER = ROOT / "analysis/overnight_cpu_active.json"


def read_json(path: Path) -> dict[str, Any]:
    """Read an object and fail closed on malformed campaign files."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def remaining_seconds(clock: dict[str, Any], *, now_epoch: float, now_monotonic: float,
                      same_boot: bool) -> float:
    """Never let a changed wall clock extend a same-boot campaign deadline."""
    remaining = float(clock["deadline_epoch"]) - now_epoch
    if same_boot:
        remaining = min(remaining, float(clock["deadline_monotonic"]) - now_monotonic)
    return max(0.0, remaining)


def status(pointer_path: Path = DEFAULT_POINTER) -> dict[str, Any]:
    """Return campaign status and authoritative stage-supervisor statuses."""
    pointer = read_json(pointer_path)
    manifest_path = Path(pointer["manifest"]).resolve()
    manifest = read_json(manifest_path)
    if pointer["campaign_id"] != manifest["campaign_id"]:
        raise ValueError("Active pointer does not match the campaign manifest")
    clock = manifest["clock"]
    if pointer["global_deadline_utc"] != clock["deadline_utc"]:
        raise ValueError("Active pointer and campaign deadline disagree")
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    current_boot = boot_path.read_text(encoding="ascii").strip() if boot_path.is_file() else None
    recorded_boot = manifest.get("boot_id")
    same_boot = bool(current_boot and recorded_boot and current_boot == recorded_boot)
    left = remaining_seconds(clock, now_epoch=time.time(), now_monotonic=time.monotonic(), same_boot=same_boot)
    stages = {}
    for name, item in sorted(manifest["stages"].items()):
        entry = {"status": item["status"], "report": item.get("report")}
        supervisor = item.get("supervisor_manifest")
        if supervisor:
            path = Path(supervisor)
            entry["supervisor_status"] = read_json(path).get("status") if path.is_file() else "missing"
        stages[name] = entry
    return {"campaign_id": manifest["campaign_id"], "manifest": str(manifest_path),
            "deadline_utc": clock["deadline_utc"], "remaining_seconds": left,
            "monotonic_clock_valid": same_boot, "stages": stages,
            "note": ("After a reboot, only the original absolute UTC deadline can be checked."
                     if not same_boot else "The earlier of monotonic and UTC deadlines is enforced.")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--require-time", action="store_true", help="Exit nonzero after the global deadline")
    args = parser.parse_args()
    result = status(args.pointer)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if args.require_time and result["remaining_seconds"] <= 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
