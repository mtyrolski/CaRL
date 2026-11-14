#!/usr/bin/env python3
"""Combine all previously frozen intent test records without inspecting responses."""

import argparse
import contextlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.campaign_data import sha256_file
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import validate_intent
from scripts.prepare_luna_intent_data import ROOT
from scripts.prepare_luna_intent_data import Family
from scripts.prepare_luna_intent_data import Phase
from scripts.prepare_luna_intent_data import read_reference_cohort
from scripts.prepare_luna_intent_data import write_frozen


def merge_records(sources: list[tuple[str, list[dict[str, Any]]]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Retain one identical calibration set and every distinct test record."""
    if len(sources) != 3:
        raise ValueError("Exactly three previously frozen cohorts are required")
    calibration: list[dict[str, Any]] | None = None
    tests: list[dict[str, Any]] = []
    origins: dict[str, str] = {}
    layouts: set[str] = set()
    for name, records in sources:
        source_calibration = [row for row in records if row["phase"] == Phase.CALIBRATION]
        if calibration is None:
            calibration = source_calibration
        elif calibration != source_calibration:
            raise ValueError("Calibration records differ across sources")
        for row in records:
            phase = Phase(row["phase"])
            if phase != Phase.TEST:
                continue
            Family(row["family"])
            if row["layout_id"] in layouts or row["record_id"] in origins:
                raise ValueError("Repeated test layout or record")
            layouts.add(row["layout_id"])
            origins[row["record_id"]] = name
            tests.append(row)
    assert calibration is not None
    if len(calibration) != 12 or len({row["layout_id"] for row in calibration}) != 4:
        raise ValueError("Expected 12 calibration records on four layouts")
    context_counts = Counter(row["context_id"] for row in calibration)
    if (len(context_counts) != 4 or any(count != 3 for count in context_counts.values())
            or len({row["record_id"] for row in calibration}) != 12
            or any(row["family"] != Family.REAL for row in calibration)):
        raise ValueError("Calibration must contain three distinct records per real-board context")
    if layouts & {row["layout_id"] for row in calibration}:
        raise ValueError("Test layouts intersect calibration")
    if set(origins) & {row["record_id"] for row in calibration}:
        raise ValueError("Test and calibration records share an ID")
    if Counter(row["family"] for row in tests) != {Family.REAL: 20, Family.SYNTHETIC: 20}:
        raise ValueError("Expected 20 previously frozen test records per family")
    tests.sort(key=lambda row: (0 if row["family"] == Family.REAL else 1, row["layout_id"], row["record_id"]))
    return calibration + tests, origins


def verify_fragment(record: dict[str, Any]) -> int:
    """Recompute a recorded fragment with the real environment and exact checker."""
    states = record["fragment_observations"]
    actions = record["fragment_actions"]
    if len(states) != len(actions) + 1 or len(actions) != record["k"]:
        raise ValueError("Fragment/action length mismatch")
    if states[0] != record["start_observation"] or states[-1] != record["later_observation"]:
        raise ValueError("Fragment endpoints differ")
    if len(states[0]) != 12 or any(len(row) != 12 for row in states[0]):
        raise ValueError("Expected a 12x12 board")
    expected_boxes = 4 if record["family"] == Family.REAL else 10
    if sum(row.count("$") + row.count("*") for row in states[0]) != expected_boxes:
        raise ValueError("Wrong box count for family")
    initial = from_rows(states[0])
    if canonical_layout_id(initial) != record["layout_id"]:
        raise ValueError("Wrong canonical layout ID")
    errors = validate_intent(record["intent"], states)
    if errors:
        raise ValueError(f"Invalid intent: {errors}")
    with contextlib.redirect_stdout(io.StringIO()):
        env = environment(initial)
        for before, after, action in zip(states, states[1:], actions):
            if action not in range(4) or before == after or to_rows(env.next_state(from_rows(before), action)) != after:
                raise ValueError("Recorded action does not reproduce its recorded next state")
    return len(actions)


def prepare(output: Path, paths: list[Path]) -> dict[str, Any]:
    if len(set(paths)) != 3:
        raise ValueError("Pass three distinct frozen source manifests")
    sources = []
    source_manifests = []
    for path in paths:
        read_reference_cohort(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        records_path = path.parent / "records.jsonl"
        records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for family, key in ((Family.REAL, "test_layouts_x4"), (Family.SYNTHETIC, "test_layouts_x10")):
            actual = {row["layout_id"] for row in records if row["phase"] == Phase.TEST and row["family"] == family}
            if actual != set(manifest[key]):
                raise ValueError(f"Frozen test records disagree with source manifest: {path}")
        name = str(path.relative_to(ROOT))
        sources.append((name, records))
        source_manifests.append({"path": name, "sha256": sha256_file(path), "records_sha256": sha256_file(records_path)})
    records, origins = merge_records(sources)
    transitions = sum(verify_fragment(record) for record in records)
    tests = [row for row in records if row["phase"] == Phase.TEST]
    manifest = {
        "schema_version": 1,
        "interpretation": "Re-evaluation of every previously frozen test case, some previously exposed to Luna; not a fresh holdout.",
        "selection_rule": "Include all prior frozen test records, without consulting model responses; sort by family then canonical layout and record ID.",
        "source_manifests": source_manifests,
        "test_record_source_manifest": origins,
        "record_count": len(records),
        "calibration_records": 12,
        "calibration_layouts": sorted({row["layout_id"] for row in records if row["phase"] == Phase.CALIBRATION}),
        "test_records_per_family": {family: sum(row["family"] == family for row in tests) for family in Family},
        "test_layouts_x4": [row["layout_id"] for row in tests if row["family"] == Family.REAL],
        "test_layouts_x10": [row["layout_id"] for row in tests if row["family"] == Family.SYNTHETIC],
        "replayed_fragment_actions": transitions,
        "all_fragments_validated_in_environment": True,
        "constraints": ["No additional independent holdout has been created.",
                        "Prior response failures and model exposure do not determine test inclusion.",
                        "Demonstration offsets are not policy action costs or relative scale labels.",
                        "Later observations and recorded actions are audit data and must not enter the controller prompt."],
    }
    output.mkdir(parents=True, exist_ok=True)
    write_frozen(output, records, [], manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.output.resolve(), [path.resolve() for path in args.source_manifest])
    print(json.dumps({"records": manifest["record_count"], "test_records_per_family": manifest["test_records_per_family"],
                      "replayed_fragment_actions": manifest["replayed_fragment_actions"]}))


if __name__ == "__main__":
    main()
