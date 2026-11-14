#!/usr/bin/env python3
"""Independently replay frozen controller-intent fragments and compare layouts."""

import argparse
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.campaign_data import sha256_file
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import validate_intent
from scripts.prepare_intent_generator_data import write_json
from scripts.prepare_luna_intent_data import Family
from scripts.prepare_luna_intent_data import Phase


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_reference_cohorts(calibration: list[dict[str, Any]], test_layouts: set[str],
                               references: list[list[dict[str, Any]]]) -> None:
    """Ensure every earlier cohort is isolated and has the same calibration."""
    for reference in references:
        previous_layouts = {record["layout_id"] for record in reference}
        if previous_layouts & test_layouts:
            raise ValueError("A new test layout overlaps a reference cohort")
        old_calibration = [row for row in reference if row["phase"] == Phase.CALIBRATION]
        if calibration != old_calibration:
            raise ValueError("Calibration records differ from a reference cohort")


def audit(records: list[dict[str, Any]], reference: list[dict[str, Any]], *,
          additional_references: tuple[list[dict[str, Any]], ...] = (), test_per_family: int = 8) -> dict[str, Any]:
    """Check actual simulator transitions, exact intentions and layout isolation."""
    counts: Counter[tuple[str, str]] = Counter()
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    transitions = 0
    for record in records:
        phase, family = Phase(record["phase"]), Family(record["family"])
        states, actions = record["fragment_observations"], record["fragment_actions"]
        if len(states) != len(actions) + 1 or len(actions) != record["k"]:
            raise ValueError("Fragment length differs from action count and offset")
        if states[0] != record["start_observation"] or states[-1] != record["later_observation"]:
            raise ValueError("Fragment endpoints disagree with the stored observations")
        if len(states[0]) != 12 or any(len(row) != 12 for row in states[0]):
            raise ValueError("Expected a 12x12 board")
        expected_boxes = 4 if family == Family.REAL else 10
        if sum(row.count("$") + row.count("*") for row in states[0]) != expected_boxes:
            raise ValueError("Board box count differs from its family")
        initial = from_rows(states[0])
        if canonical_layout_id(initial) != record["layout_id"]:
            raise ValueError("Canonical layout ID does not match the board")
        errors = validate_intent(record["intent"], states)
        if errors:
            raise ValueError(f"Intent does not validate on its complete fragment: {errors}")
        if record["policy_measurements"] is not None:
            raise ValueError("Frozen dataset unexpectedly contains policy measurements")
        env = environment(initial)
        for before, after, action in zip(states, states[1:], actions):
            if action not in range(4) or before == after or to_rows(env.next_state(from_rows(before), action)) != after:
                raise ValueError("Recorded action does not produce its recorded next state")
            transitions += 1
        counts[(phase, family)] += 1
        groups[(phase, family)].add(record["layout_id"])
        contexts[record["context_id"]].append(record)
    expected_counts = {(Phase.CALIBRATION, Family.REAL): 12, (Phase.TEST, Family.REAL): test_per_family,
                       (Phase.TEST, Family.SYNTHETIC): test_per_family}
    if counts != expected_counts or any(len(groups[key]) != count for key, count in (
            ((Phase.CALIBRATION, Family.REAL), 4), ((Phase.TEST, Family.REAL), test_per_family),
            ((Phase.TEST, Family.SYNTHETIC), test_per_family))):
        raise ValueError("Unexpected numbers of records or independent layouts")
    for name, layouts in groups.items():
        if any(layouts & other for other_name, other in groups.items() if name != other_name):
            raise ValueError("Calibration or test groups share a canonical layout")
    for examples in contexts.values():
        if any(row["start_observation"] != examples[0]["start_observation"] for row in examples):
            raise ValueError("One context ID describes multiple start observations")
        if examples[0]["phase"] == Phase.CALIBRATION:
            intents = {json.dumps(row["intent"], sort_keys=True) for row in examples}
            if len(examples) != 3 or len(intents) != 3:
                raise ValueError("Each calibration context must have three distinct intents")
        elif len(examples) != 1:
            raise ValueError("Each test context must have one intent")
    test_layouts = set().union(*(value for (phase, _), value in groups.items() if phase == Phase.TEST))
    calibration = [row for row in records if row["phase"] == Phase.CALIBRATION]
    validate_reference_cohorts(calibration, test_layouts, [reference, *additional_references])
    return {
        "records": len(records),
        "records_by_group": {f"{phase}:{family}": count for (phase, family), count in counts.items()},
        "layouts_by_group": {f"{phase}:{family}": len(value) for (phase, family), value in groups.items()},
        "replayed_primitive_transitions": transitions,
        "all_actions_match_recorded_states": True,
        "all_intents_pass_exact_fragment_validation": True,
        "calibration_records_identical": True,
        "test_layout_intersection_with_reference": 0,
        "scope": "Replay of stored fragments; full trajectories were replayed separately during dataset preparation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--reference-records", type=Path, action="append", required=True)
    parser.add_argument("--test-per-family", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    references = [read_records(path) for path in args.reference_records]
    summary = audit(read_records(args.records), references[0], additional_references=tuple(references[1:]),
                    test_per_family=args.test_per_family)
    summary["records_sha256"] = sha256_file(args.records)
    if len(args.reference_records) == 1:
        summary["reference_records_sha256"] = sha256_file(args.reference_records[0])
    else:
        summary["reference_records_sha256s"] = [sha256_file(path) for path in args.reference_records]
        summary["reference_cohort_count"] = len(args.reference_records)
    write_json(args.output, summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
