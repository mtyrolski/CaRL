"""Summarize observed generator requests without promoting missing observations."""

import csv
import sqlite3
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.campaign_budget import CampaignLedger
from carl.icl.types import JSONObject
from scripts.generator_probe.cases import TAG


def costs() -> JSONObject:
    """Read charges for this probe; local cache replays incur no provider charge."""
    ledger = CampaignLedger()
    with sqlite3.connect(ledger.path) as database:
        rows = database.execute(
            "SELECT status, COUNT(*), SUM(COALESCE(charged_nanos,reserved_nanos))/1e9 "
            "FROM attempts WHERE tag LIKE ? GROUP BY status", (TAG + ":%",)
        ).fetchall()
    return {status: {"attempts": count, "usd": amount} for status, count, amount in rows}


def summarize(rows: list[JSONObject], expected: int, output: Path) -> JSONObject:
    """Write CSV, raw counts, and a compact human report from completed cases."""
    groups: dict[str, JSONObject] = {}
    flattened: list[JSONObject] = []
    for row in rows:
        candidates = [item for call in row["classification"] for item in call["candidates"]]
        flat = {"board_id": row["board_id"], "variant": row["variant"], "u": row["scale"],
                "reached": row["reached_any"], "accepted": len(row["proposals"]),
                "invalid": sum(not item["static_valid"] for item in candidates),
                "copies": sum(bool(item["source_copy"]) for item in candidates),
                "actions": sum(item["rollout"]["actions"] for item in row["proposals"]),
                "error": row["error"] is not None,
                "physical_requests": row["metrics"]["requests"],
                "cache_hits": row["metrics"]["cache_hits"]}
        flattened.append(flat)
        group = groups.setdefault(row["variant"], {key: 0 for key in (
            "requests", "reached", "accepted", "invalid", "copies", "actions", "errors")})
        group["requests"] += 1
        for key in ("reached", "accepted", "invalid", "copies", "actions"):
            group[key] += flat[key]
        group["errors"] += flat["error"]
    summary = {"expected": expected, "observed": len(rows), "variants": groups,
               "cost": costs(), "denominator": "all scheduled generator queries; invalid/empty/copies are failures"}
    write_json(output / "summary.json", summary)
    if flattened:
        with (output / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
    lines = ["# Generator validation probe", "", f"Observed {len(rows)}/{expected} queries.", "",
             "| Variant | Reached distinct target / queries | Accepted | Invalid | Copies | Actions |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, group in groups.items():
        lines.append(f"| {name} | {group['reached']}/{group['requests']} | {group['accepted']} | "
                     f"{group['invalid']} | {group['copies']} | {group['actions']} |")
    lines += ["", "These are generator/CLLP tests on validation layouts, not full-board solves.",
              "Saved run configs identify prompts, demonstration counts and the shared frozen controller.",
              "Every recorded action is replayed. No simulator transitions are supplied to the generator.",
              "Multiple scales on one board are dependent observations, not independent layouts."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
