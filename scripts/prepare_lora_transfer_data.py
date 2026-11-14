#!/usr/bin/env python3
"""Freeze balanced 12×12×4 and ×10 evaluation contexts without model outputs."""

import argparse
import contextlib
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
from carl.icl.intent_data import validate_intent
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import read_records

ROOT = Path(__file__).resolve().parents[1]
DATA_10 = ROOT / "analysis/intent_generator_20261114T032232Z/dataset.jsonl"
DATA_4 = ROOT / "analysis/luna_intent_20261114T060730Z/expanded/data/records.jsonl"


def select_contexts(rows_10: list[dict[str, Any]], rows_4: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Take the lowest record ID per layout; selection never examines predictions."""
    occupied = {row["layout_id"] for row in rows_10 if row["split"] in {"train", "validation"}}
    selected = []
    for family, candidates in (
        ("real_12x12x4", [row for row in rows_4 if row.get("phase") == "test" and row.get("family") == "real_12x12x4"]),
        ("proposal_gamma2_12x12x10", [row for row in rows_10 if row["split"] == "test"]),
    ):
        by_layout = {}
        for row in sorted(candidates, key=lambda item: (item["layout_id"], item["record_id"])):
            by_layout.setdefault(row["layout_id"], row)
        if len(by_layout) != 20 or set(by_layout) & occupied:
            raise ValueError(f"Expected 20 disjoint test layouts for {family}")
        occupied.update(by_layout)
        for row in by_layout.values():
            result = dict(row)
            result["split"] = "test"
            result["evaluation_family"] = family
            selected.append(result)
    if len({row["record_id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate evaluation record ID")
    return selected


def replay(record: dict[str, Any]) -> None:
    """Check dimensions, canonical layout, label and every primitive transition."""
    rows = record["start_observation"]
    expected_boxes = 4 if record["evaluation_family"] == "real_12x12x4" else 10
    if (len(rows) != 12 or any(len(row) != 12 for row in rows)
            or sum(row.count("$") + row.count("*") for row in rows) != expected_boxes):
        raise ValueError("Unexpected Sokoban dimensions or box count")
    state = from_rows(rows)
    if canonical_layout_id(state) != record["layout_id"]:
        raise ValueError("Canonical layout does not match the frozen record")
    with contextlib.redirect_stdout(io.StringIO()):
        env = environment(state)
    fragment = [rows]
    for action in record["fragment_actions"]:
        if not isinstance(action, int) or isinstance(action, bool) or action not in range(4):
            raise ValueError("Invalid primitive action")
        state = np.asarray(env.next_state(state, action))
        fragment.append(to_rows(state))
    if fragment[-1] != record["later_observation"]:
        raise ValueError("Recorded next state does not match replay")
    if record.get("fragment_observations") is not None and record["fragment_observations"] != fragment:
        raise ValueError("Recorded fragment does not match replay")
    errors = validate_intent(record["intent"], fragment)
    if errors:
        raise ValueError(f"Invalid recorded intent: {errors}")
    # The existing serialiser is intentionally whitelisted to board and task.
    prompt_messages(record)


def alternating_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Keep interrupted evaluation prefixes balanced across the two families."""
    real = [row["record_id"] for row in rows if row["evaluation_family"] == "real_12x12x4"]
    synthetic = [row["record_id"] for row in rows if row["evaluation_family"] == "proposal_gamma2_12x12x10"]
    if len(real) != 20 or len(synthetic) != 20:
        raise ValueError("The evaluation order requires 20 cases from each family")
    return [identifier for pair in zip(real, synthetic, strict=True) for identifier in pair]


def prepare(output: Path, source_10: Path = DATA_10, source_4: Path = DATA_4) -> dict[str, Any]:
    rows = select_contexts(read_records(source_10), read_records(source_4))
    for row in rows:
        replay(row)
    encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    ids = alternating_ids(rows)
    ids_text = json.dumps({"record_ids": ids}, indent=2) + "\n"
    real_ids = [row["record_id"] for row in rows if row["evaluation_family"] == "real_12x12x4"]
    real_ids_text = json.dumps({"record_ids": real_ids}, indent=2) + "\n"
    manifest = {
        "schema_version": 1,
        "selection": "Lowest record ID per canonical layout; no model outputs or success outcomes consulted.",
        "classification": "Exploratory transfer: x4 source real, x10 synthetic; prior exposure and difficulty may differ.",
        "training_policy": "Evaluation only; no test records may enter training or validation.",
        "serializer": "carl.icl.intent_generator_experiment.prompt_messages: current board and task only",
        "source_x4": str(source_4.resolve()), "source_x4_sha256": file_sha256(source_4),
        "source_x10": str(source_10.resolve()), "source_x10_sha256": file_sha256(source_10),
        "data_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "test_ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
        "test_ids_order": "Alternating x4 and x10, one per canonical layout in each family.",
        "test_ids_x4_sha256": hashlib.sha256(real_ids_text.encode()).hexdigest(),
        "record_count": len(rows), "layouts_per_family": {"real_12x12x4": 20, "proposal_gamma2_12x12x10": 20},
        "record_ids": [row["record_id"] for row in rows],
        "replay": "All primitive actions replayed in SokobanEnv; every intent and preservation condition checked.",
    }
    output.mkdir(parents=True, exist_ok=True)
    files = {output / "dataset.jsonl": encoded, output / "test_ids.json": ids_text,
             output / "test_ids_x4.json": real_ids_text,
             output / "manifest.json": json.dumps(manifest, indent=2) + "\n"}
    for path, text in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Refusing to overwrite different frozen data: {path}")
    for path, text in files.items():
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-x10", type=Path, default=DATA_10)
    parser.add_argument("--source-x4", type=Path, default=DATA_4)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.source_x10, args.source_x4), indent=2))


if __name__ == "__main__":
    main()
