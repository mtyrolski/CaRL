#!/usr/bin/env python3
"""Freeze full Sokoban starts from the campaign's existing layout split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.data_sources.sokoban import environment
from scripts.intent_experiment_supervisor import atomic_write
from scripts.intent_experiment_supervisor import write_json
from scripts.run_overnight_cpu_policy import read_jsonl

FAMILIES = {"12x12x4": 4, "12x12x10": 10}
FIELDS = ("family", "source_family", "split", "board_rank", "layout_id", "trajectory_id", "initial_state",
          "source_path", "source_sha256", "exposure_status", "original_split")


def select(rows: list[dict[str, Any]], evaluation_per_family: int, development_per_family: int) -> list[dict[str, Any]]:
    """Select an outcome-blind rank prefix, preserving complete original starts."""
    if evaluation_per_family < 1 or development_per_family < 1:
        raise ValueError("Both split counts must be positive")
    result: list[dict[str, Any]] = []
    for split, count in (("development", development_per_family), ("evaluation", evaluation_per_family)):
        for family, expected_boxes in FAMILIES.items():
            candidates = sorted((row for row in rows if row["split"] == split and row["family"] == family),
                                key=lambda row: (row["board_rank"], row["source_family"], row["layout_id"]))
            if len(candidates) < count:
                raise ValueError(f"Not enough {split}/{family} original boards")
            for row in candidates[:count]:
                selected = {key: row[key] for key in FIELDS}
                state = from_rows(selected["initial_state"])
                if state.shape != (12, 12, 7):
                    raise ValueError("Expected 12x12 Sokoban boards")
                tiles = state.argmax(-1)
                if (int(np.isin(tiles, [3, 4]).sum()) != expected_boxes
                        or int(np.isin(tiles, [2, 3, 6]).sum()) != expected_boxes
                        or int(np.isin(tiles, [5, 6]).sum()) != 1):
                    raise ValueError("Incorrect box, goal or player count")
                if canonical_layout_id(state) != selected["layout_id"]:
                    raise ValueError("Canonical layout differs from the frozen campaign data")
                env = environment(state)
                if env.num_boxes != expected_boxes:
                    raise ValueError("Environment uses the wrong number of boxes")
                selected["initial_state_sha256"] = state_hash(state)
                selected["initially_solved"] = bool(env.is_solved(state))
                selected["initial_boxes_on_goals"] = int(env.num_boxes_on_target(state))
                result.append(selected)
    ids = [row["layout_id"] for row in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate or cross-split canonical layouts")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluation-per-family", type=int, default=12)
    parser.add_argument("--development-per-family", type=int, default=10)
    args = parser.parse_args()
    rows = select(read_jsonl(args.source), args.evaluation_per_family, args.development_per_family)
    args.output.mkdir(parents=True, exist_ok=True)
    board_path = args.output / "boards.jsonl"
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write(board_path, content)
    counts = Counter((row["split"], row["family"], row["source_family"]) for row in rows)
    summary = {"source": str(args.source.resolve()), "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
               "boards": str(board_path.resolve()), "boards_sha256": hashlib.sha256(content.encode()).hexdigest(),
               "selection": "Fixed board-rank prefix of full initial states, before development or evaluation outcomes.",
               "split_and_source_counts": [{"split": split, "family": family, "source_family": source, "boards": count}
                                           for (split, family, source), count in sorted(counts.items())],
               "initially_solved": sum(row["initially_solved"] for row in rows),
               "test_cohort": "previously inspected and exploratory; training exposure of checkpoints unknown",
               "excluded_from_inputs": ["future states", "demonstration actions", "offsets"]}
    write_json(args.output / "board_manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
