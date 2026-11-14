#!/usr/bin/env python3
"""Independently audit LoRA transfer counts and replay bounded-search results."""

import argparse
import collections
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment

VARIANTS = ("r2_100", "r4_100", "r8_100")
FAMILIES = ("real_12x12x4", "proposal_gamma2_12x12x10")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_strict(text: str) -> tuple[bool, Any]:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")

    try:
        return True, json.loads(text, parse_constant=invalid_constant, object_pairs_hook=no_duplicate_keys)
    except ValueError:
        return False, None


def predicate_true(predicate: dict[str, Any], board: list[str]) -> bool:
    name = predicate["predicate"]
    if name == "box_at":
        return board[predicate["row"]][predicate["col"]] in "$*"
    if name == "player_at":
        return board[predicate["row"]][predicate["col"]] in "@+"
    return sum(row.count("*") for row in board) >= predicate["count"]


def independently_valid(intent: Any, board: list[str]) -> bool:
    """Check the documented predicate schema without using the assessment checker."""
    if (not isinstance(intent, dict) or set(intent) - {"effects", "preserve", "instruction"}
            or not isinstance(intent.get("effects"), list) or not intent["effects"]
            or not isinstance(intent.get("preserve"), list)
            or (intent.get("instruction") is not None and not isinstance(intent["instruction"], str))):
        return False
    occupied: dict[str, set[tuple[int, int]]] = {"box_at": set(), "player_at": set()}
    for name in ("effects", "preserve"):
        seen = set()
        for predicate in intent[name]:
            if not isinstance(predicate, dict):
                return False
            kind = predicate.get("predicate")
            if kind == "boxes_on_goals_at_least":
                if (set(predicate) != {"predicate", "count"} or not isinstance(predicate["count"], int)
                        or isinstance(predicate["count"], bool)
                        or not 1 <= predicate["count"] <= sum(sum(c in ".*+" for c in row) for row in board)):
                    return False
            elif isinstance(kind, str) and kind in occupied:
                if (set(predicate) != {"predicate", "row", "col"} or not isinstance(predicate["row"], int)
                        or isinstance(predicate["row"], bool) or not isinstance(predicate["col"], int)
                        or isinstance(predicate["col"], bool)):
                    return False
                row, col = predicate["row"], predicate["col"]
                if not (0 <= row < len(board) and 0 <= col < len(board[0])) or board[row][col] == "#":
                    return False
                occupied[kind].add((row, col))
            else:
                return False
            key = json.dumps(predicate, sort_keys=True, separators=(",", ":"))
            if key in seen:
                return False
            seen.add(key)
            if predicate_true(predicate, board) != (name == "preserve"):
                return False
    return len(occupied["player_at"]) <= 1 and not (occupied["player_at"] & occupied["box_at"])


def replay_found(board: list[str], intent: dict[str, Any], solver: dict[str, Any]) -> None:
    """Replay actions in CaRL, checking every preservation condition after every step."""
    actions = solver["witness_actions"]
    if (not isinstance(actions, list) or len(actions) != solver["witness_depth"]
            or len(actions) > solver["depth_limit"] or not all(isinstance(action, int) and not isinstance(action, bool)
                                                              and action in range(4) for action in actions)):
        raise ValueError("Invalid bounded-search action sequence")
    state = from_rows(board)
    with contextlib.redirect_stdout(io.StringIO()):
        env = environment(state)
    if not all(predicate_true(predicate, board) for predicate in intent["preserve"]):
        raise ValueError("Preservation fails initially")
    for action in actions:
        if env.is_solved(state):
            raise ValueError("Action after task termination")
        next_state = np.asarray(env.next_state(state, action))
        if np.array_equal(state, next_state):
            raise ValueError("Bounded search contains a no-op")
        state = next_state
        if not all(predicate_true(predicate, to_rows(state)) for predicate in intent["preserve"]):
            raise ValueError("Preservation fails during the action sequence")
    reached = to_rows(state)
    if reached != solver["reached_observation"] or not all(predicate_true(predicate, reached) for predicate in intent["effects"]):
        raise ValueError("Recorded result or intent satisfaction disagrees with replay")


def audit(run: Path) -> dict[str, Any]:
    """Audit all 120 predictions and verify the generated search sequences."""
    data = run / "eval_transfer/dataset.jsonl"
    ids_path = run / "eval_transfer/test_ids.json"
    assessment = run / "assessment_transfer_core/assessment.jsonl"
    summary_path = run / "assessment_transfer_core/summary.csv"
    config_path = run / "assessment_transfer_core/config.json"
    records = json_lines(data)
    record_map = {row["record_id"]: row for row in records}
    identifiers = json.loads(ids_path.read_text(encoding="utf-8"))["record_ids"]
    if len(records) != 40 or len(record_map) != 40 or len(identifiers) != 40 or set(identifiers) != set(record_map):
        raise ValueError("Frozen evaluation selection is incomplete or duplicated")
    for family in FAMILIES:
        group = [row for row in records if row["evaluation_family"] == family]
        if len(group) != 20 or len({row["layout_id"] for row in group}) != 20:
            raise ValueError("Expected 20 independent test layouts per family")
        for row in group:
            board = row["start_observation"]
            boxes = 4 if family == FAMILIES[0] else 10
            if (row["split"] != "test" or len(board) != 12 or any(len(line) != 12 for line in board)
                    or sum(line.count("$") + line.count("*") for line in board) != boxes
                    or canonical_layout_id(from_rows(board)) != row["layout_id"]):
                raise ValueError("Board family, split or canonical layout disagrees")
    assessment_rows = [row for row in json_lines(assessment) if row["variant"] in VARIANTS]
    predictions = {}
    for variant in VARIANTS:
        rows = json_lines(run / f"eval_transfer/{variant}.jsonl")
        indexed = {row["record_id"]: row for row in rows}
        if len(rows) != 40 or set(indexed) != set(record_map):
            raise ValueError("Prediction file is missing cases or contains duplicate IDs")
        predictions[variant] = indexed
    totals: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(collections.Counter)
    observed: set[tuple[str, str]] = set()
    preserve_nonempty = 0
    for row in assessment_rows:
        variant, identifier = row["variant"], row["record_id"]
        key = (variant, identifier)
        if key in observed or identifier not in record_map or identifier not in predictions[variant]:
            raise ValueError("Missing or duplicate prediction")
        observed.add(key)
        source, prediction = record_map[identifier], predictions[variant][identifier]
        family, board = source["evaluation_family"], source["start_observation"]
        if (row["status"] != "evaluated" or row["family"] != family or row["layout_id"] != source["layout_id"]
                or row["start_observation"] != board or row["raw_output"] != prediction["raw_output"]):
            raise ValueError("Assessment differs from frozen input or raw prediction")
        valid_json, intent = parse_strict(row["raw_output"])
        valid = valid_json and independently_valid(intent, board)
        if valid_json != row["strict_json"] or valid != row["strict_valid"]:
            raise ValueError("Independent JSON or intent check differs")
        total = totals[variant, family]
        total["outputs"] += 1
        total["strict_json"] += int(valid_json)
        total["strict_valid"] += int(valid)
        solver = row["solver"]
        if not valid:
            if solver is not None:
                raise ValueError("Search should not be run on invalid intents")
            continue
        if not isinstance(solver, dict) or solver.get("status") == "verification_error":
            raise ValueError("Missing or unsuccessful search verification")
        if ((solver.get("depth_limit"), solver.get("node_limit"), solver.get("time_limit_seconds")) != (24, 5000, 0.2)
                or not 0 <= solver.get("expanded_nodes", -1) <= 5000
                or not 0 <= solver.get("primitive_edges_examined", -1) <= 4 * solver["expanded_nodes"]
                or bool(solver.get("witness_found")) != (solver.get("status") == "witness_found")):
            raise ValueError("Search budgets, status or work counters disagree")
        if solver.get("witness_found"):
            replay_found(board, intent, solver)
            total["bfs_found"] += 1
            preserve_nonempty += bool(intent["preserve"])
        else:
            if solver.get("witness_actions") is not None or solver.get("reached_observation") is not None:
                raise ValueError("An unresolved search must not have a fabricated path")
            total["bfs_unresolved"] += 1
    if len(observed) != len(VARIANTS) * len(identifiers):
        raise ValueError("A variant does not cover the frozen selection")
    with summary_path.open(encoding="utf-8", newline="") as source:
        csv_summary = {(row["variant"], row["family"]): row for row in csv.DictReader(source)}
    counts = []
    for (variant, family), values in sorted(totals.items()):
        reported = csv_summary[variant, family]
        if (any(values[field] != int(reported[field]) for field in ("outputs", "strict_json", "strict_valid", "bfs_found", "bfs_unresolved"))
                or int(reported["boards_observed"]) != 20 or int(reported["boards_planned"]) != 20):
            raise ValueError("Independent totals disagree with summary CSV")
        counts.append({"variant": variant, "family": family, "boards": 20, **dict(values)})
    paths = [data, ids_path, assessment, summary_path, config_path] + [run / f"eval_transfer/{name}.jsonl" for name in VARIANTS]
    return {
        "schema_version": 1, "status": "passed", "predictions_checked": len(observed),
        "search_sequences_replayed": sum(row.get("bfs_found", 0) for row in counts),
        "replayed_with_nonempty_preservation": preserve_nonempty,
        "criteria": "Independent strict JSON and predicate checks; every found sequence replayed with CaRL environment, no-ops, terminal states and preservation checked.",
        "limits": "Depth 24, 5000 expanded nodes and 0.2 seconds; an unresolved search does not imply impossibility.",
        "counts": counts, "input_sha256": {str(path): digest(path) for path in paths},
        "command": f"python scripts/audit_lora_transfer_results.py --run {run}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    output = args.run / "independent_transfer_audit.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": result["status"], "predictions": result["predictions_checked"],
                      "replayed": result["search_sequences_replayed"], "output": str(output)}))


if __name__ == "__main__":
    main()
