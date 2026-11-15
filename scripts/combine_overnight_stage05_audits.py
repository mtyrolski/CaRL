#!/usr/bin/env python3
"""Combine disjoint offline stage-five replay audits without replaying jobs twice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl


def combine(boards_path: Path, audit_paths: list[Path], interruption: Path | None = None) -> dict[str, Any]:
    boards = [row for row in read_jsonl(boards_path) if row["split"] == "evaluation"]
    board_ids = {row["layout_id"] for row in boards}
    if not boards or len(board_ids) != len(boards):
        raise ValueError("empty or duplicated evaluation boards")
    board_sha = file_sha(boards_path)
    expected_methods = {"direct_policy", "primitive_astar", "adasubs", "ksubs"}
    methods: dict[str, dict[str, Any]] = {}
    offline_transitions = 0
    issues = []
    auditor_hashes: set[str] = set()
    for path in audit_paths:
        audit = json.loads(path.read_text())
        if audit["boards_sha256"] != board_sha or audit["frozen_boards"] != len(boards):
            raise ValueError("replay audits do not cover the same frozen boards")
        audit_methods = set(audit["methods"])
        if not audit_methods <= expected_methods or audit_methods & methods.keys():
            raise ValueError("unexpected or repeated audit method")
        auditor_hashes.add(audit["auditor_sha256"])
        planned = len(boards) * len(audit_methods)
        if audit["expected_jobs"] != planned or audit["present_jobs"] != planned:
            issues.append({"audit": str(path), "planned_jobs": planned,
                           "expected_jobs": audit["expected_jobs"], "present_jobs": audit["present_jobs"]})
        if audit["state"] != "pass" or audit["missing_jobs"] or audit["errors"] or audit.get("incomplete_traces"):
            issues.append({"audit": str(path), "state": audit["state"], "missing": len(audit["missing_jobs"]),
                           "incomplete_traces": len(audit.get("incomplete_traces", [])),
                           "errors": len(audit["errors"])})
        offline_transitions += audit["offline_audit_transitions"]
        for method in audit["methods"]:
            metrics = audit["by_method"].get(method, {})
            if metrics.get("jobs_audited", 0) != len(boards):
                issues.append({"audit": str(path), "method": method,
                               "audited_jobs": metrics.get("jobs_audited", 0), "planned_jobs": len(boards)})
            methods[method] = {"planned_boards": len(boards), **metrics, "audit_file": str(path.resolve()),
                               "audit_sha256": file_sha(path)}
    for method in sorted(expected_methods - methods.keys()):
        issues.append({"method": method, "reason": "no audit file"})
    if len(auditor_hashes) > 1:
        issues.append({"reason": "replay audits used different auditor versions", "auditor_sha256": sorted(auditor_hashes)})
    previous = None
    if interruption is not None:
        previous = json.loads(interruption.read_text())
    return {"state": "pass" if not issues else "partial", "board_sha256": board_sha,
            "evaluation_boards": len(boards), "expected_methods": sorted(expected_methods), "methods": methods,
            "issues": issues, "auditor_sha256": sorted(auditor_hashes), "offline_audit_transitions": offline_transitions,
            "interrupted_prior_audit": ({"path": str(interruption.resolve()), "sha256": file_sha(interruption),
                "offline_transitions": previous["partial_offline_transition_count"],
                "note": previous["partial_offline_transition_count_reason"]} if previous is not None else None),
            "scope": "Disjoint offline replay audits, not method execution; earlier interrupted work is not known."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", required=True, type=Path)
    parser.add_argument("--audits", required=True, nargs="+", type=Path)
    parser.add_argument("--interruption", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = combine(args.boards, args.audits, args.interruption)
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)
    if result["state"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
