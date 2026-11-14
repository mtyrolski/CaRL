#!/usr/bin/env python3
"""Independently check D4 separation and replay previously frozen fragments."""

import argparse
import contextlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import validate_intent
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import read_records

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "data/icl/campaign_dataset_12x12_v1/real_12x12x4/pilot100.json"
CAMPAIGN_ACTIONS = ROOT / "data/icl/campaign_dataset_12x12_v1_private/pilot100/real_12x12x4.json"
ACTION_IDS = {"U": 0, "D": 1, "L": 2, "R": 3}


def independent_d4(rows: list[str]) -> tuple[str, ...]:
    """Represent walls/goals modulo D4 independently of canonical_layout_id."""
    initial = tuple("".join("#" if cell == "#" else "." if cell in ".*+" else "-" for cell in row)
                    for row in rows)
    if not initial or any(len(row) != len(initial) for row in initial):
        raise ValueError("Expected a nonempty square board")

    def rotate(grid: tuple[str, ...]) -> tuple[str, ...]:
        return tuple("".join(grid[len(grid) - 1 - col][row] for col in range(len(grid)))
                     for row in range(len(grid)))

    variants = []
    for grid in (initial, tuple(row[::-1] for row in initial)):
        for _ in range(4):
            variants.append(grid)
            grid = rotate(grid)
    return min(variants)


def audit_splits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, set[tuple[str, ...]]] = {}
    id_to_layout: dict[str, tuple[str, ...]] = {}
    layout_to_id: dict[tuple[str, ...], str] = {}
    for row in rows:
        value = independent_d4(row["start_observation"])
        identifier = row["layout_id"]
        if identifier in id_to_layout and id_to_layout[identifier] != value:
            raise ValueError("One layout ID identifies different independent D4 classes")
        if value in layout_to_id and layout_to_id[value] != identifier:
            raise ValueError("One independent D4 class has different layout IDs")
        id_to_layout[identifier] = value
        layout_to_id[value] = identifier
        groups.setdefault(row["split"], set()).add(value)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[left] & groups[right]:
            raise ValueError(f"Independent D4 classes overlap: {left}/{right}")
    tests = [row for row in rows if row["split"] == "test"]
    if len(tests) != len({independent_d4(row["start_observation"]) for row in tests}):
        raise ValueError("Duplicate test layouts modulo an independent D4 implementation")
    return {"independent_d4_classes_per_split": {split: len(values) for split, values in groups.items()},
            "train_validation_test_pairwise_overlaps": 0,
            "duplicate_test_layouts": 0, "layout_id_independent_d4_conflicts": 0}


def replay_prior(record: dict[str, Any], board: list[str], actions: list[int]) -> dict[str, int]:
    """Replay the actual source prefix, reject no-ops and check all fragment states."""
    start, end = record["i"], record["i"] + record["k"]
    if start < 0 or end > len(actions) or actions[start:end] != record["fragment_actions"]:
        raise ValueError("Saved fragment actions do not match the full source trajectory")
    current = from_rows(board)
    with contextlib.redirect_stdout(io.StringIO()):
        env = environment(current)
    fragment = []
    for index, action in enumerate(actions[:end]):
        if index == start:
            if to_rows(current) != record["start_observation"]:
                raise ValueError("Source state differs at the recorded start index")
            fragment.append(to_rows(current))
        if not isinstance(action, int) or isinstance(action, bool) or action not in range(4) or env.is_solved(current):
            raise ValueError("Invalid action or action after terminal state")
        after = np.asarray(env.next_state(current, action))
        if np.array_equal(current, after):
            raise ValueError("Source prefix contains a no-op")
        current = after
        if index >= start:
            fragment.append(to_rows(current))
    if not fragment or fragment[-1] != record["later_observation"]:
        raise ValueError("Source replay differs from the recorded end state")
    stored = record.get("fragment_observations")
    if stored is not None and stored != fragment:
        raise ValueError("A stored intermediate state differs from its real replay")
    errors = validate_intent(record["intent"], fragment)
    if errors:
        raise ValueError(f"Recorded intent or preservation failed: {errors}")
    return {"source_prefix_actions": end, "fragment_actions": record["k"],
            "stored_intermediate_states_checked": max(0, len(fragment) - 2) if stored is not None else 0}


def audit(data: Path, ids: Path) -> dict[str, Any]:
    all_rows = read_records(data)
    manifest = json.loads(ids.read_text())
    if manifest.get("dataset_sha256") != file_sha256(data):
        raise ValueError("Frozen ID list does not match dataset")
    by_id = {row["record_id"]: row for row in all_rows}
    ordered = [by_id[key] for key in manifest["record_ids"]]
    previous = ordered[:40]
    if len(previous) != 40 or any(not row["previously_evaluated_in_lora_matrix"] for row in previous):
        raise ValueError("Previous 40 contexts are not the first 40")
    summary = audit_splits(all_rows)
    boards = {row["layout_id"]: row for row in json.loads(CAMPAIGN.read_text())["instances"]}
    solutions = {row["board_id"]: row["actions"] for row in json.loads(CAMPAIGN_ACTIONS.read_text())["witnesses"]}
    counts: Counter[str] = Counter()
    for record in previous:
        family = record["evaluation_family"]
        if family == "real_12x12x4":
            board = boards[record["layout_id"]]
            initial, actions = board["state"], solutions[board["board_id"]]
        elif family == "proposal_gamma2_12x12x10":
            raw_path = ROOT / record["source_path"]
            if file_sha256(raw_path) != record["source_sha256"]:
                raise ValueError("Proposal source hash differs")
            source = json.loads(raw_path.read_text())
            if source["id"] != record["trajectory_id"]:
                raise ValueError("Proposal source trajectory ID differs")
            initial = [line.replace(" ", "-") for line in source["ascii"].splitlines()]
            actions = [ACTION_IDS[action] for action in source["actions"]]
        else:
            raise ValueError("Unexpected family in previous 40")
        result = replay_prior(record, initial, actions)
        counts.update(result)
        counts[family] += 1
    for row in ordered:
        altered = {**row, "record_id": "secret", "source_path": "secret", "i": 999, "k": 999,
                   "fragment_actions": [999], "later_observation": ["secret"],
                   "intent": {"secret": "target"}, "validation": {"secret": "target"}}
        if prompt_messages(row) != prompt_messages(altered):
            raise ValueError("Generator prompt depends on hidden future fields")
    return {"schema_version": 1, "status": "passed", "dataset_sha256": file_sha256(data),
            "test_ids_sha256": file_sha256(ids), "symmetry": summary,
            "previous_40_source_replay": dict(counts), "prompt_leakage_checks": len(ordered),
            "note": "The x4 records include stored intermediate observations, all compared. "
                    "The x10 records store only endpoints; each intervening action and state was "
                    "reconstructed from its hashed original source and checked, but no separately "
                    "recorded intermediate observations exist to compare."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.data, args.ids)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
