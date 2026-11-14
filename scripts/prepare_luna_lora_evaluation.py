#!/usr/bin/env python3
"""Freeze exploratory evaluation contexts without invoking any model.

The input cohort was examined in earlier experiments. These contexts are
held out from the new training, but are not an untouched benchmark.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records


class EvaluationFamily(StrEnum):
    REAL = "real_12x12x4"
    PROPOSAL = "proposal_gamma2_12x12x10"
    GENERATED = "generated_v1_12x12x10"


DEFAULT_COUNTS = {
    EvaluationFamily.REAL: 12,
    EvaluationFamily.PROPOSAL: 8,
    EvaluationFamily.GENERATED: 4,
}


def _selection_key(seed: int, row: dict[str, Any]) -> str:
    payload = f"luna-lora-evaluation-v1|{seed}|{row['layout_id']}|{row['record_id']}"
    return hashlib.sha256(payload.encode()).hexdigest()


def freeze_contexts(records: list[dict[str, Any]], ids: list[str], excluded_layouts: set[str],
                    counts: dict[EvaluationFamily, int], seed: int) -> list[dict[str, Any]]:
    """Select one source observation per layout without looking at predictions."""
    if (len(ids) != len(set(ids)) or not ids
            or set(counts) != set(EvaluationFamily)
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in counts.values())):
        raise ValueError("Evaluation IDs and positive per-family counts are required")
    by_id = {row["record_id"]: row for row in records}
    if len(by_id) != len(records) or any(identifier not in by_id for identifier in ids):
        raise ValueError("Dataset IDs are duplicated or the frozen selection is missing")
    candidates: dict[EvaluationFamily, list[dict[str, Any]]] = {family: [] for family in EvaluationFamily}
    seen_layouts: set[str] = set()
    for identifier in ids:
        row = by_id[identifier]
        if row.get("split") != "test":
            raise ValueError("A frozen evaluation context is not in the test split")
        layout = row["layout_id"]
        if layout in seen_layouts or layout in excluded_layouts:
            raise ValueError("Test layouts must be unique and disjoint from train and validation")
        seen_layouts.add(layout)
        try:
            family = EvaluationFamily(row.get("evaluation_family", row.get("family")))
        except ValueError:
            raise ValueError("A test context has an unknown family") from None
        board = row["start_observation"]
        if (not isinstance(board, list) or len(board) != 12 or not all(isinstance(line, str) and len(line) == 12
                                                                      for line in board)):
            raise ValueError("Evaluation boards must be complete 12x12 ASCII boards")
        boxes = sum(line.count("$") + line.count("*") for line in board)
        if boxes != (4 if family == EvaluationFamily.REAL else 10):
            raise ValueError("Board family and box count disagree")
        if canonical_layout_id(from_rows(board)) != layout:
            raise ValueError("Saved layout ID does not match the board")
        goal = row.get("task_goal")
        if not isinstance(goal, str) or not goal:
            raise ValueError("Task goal is missing")
        candidates[family].append(row)

    selected = []
    for family, count in counts.items():
        choices = sorted(candidates[family], key=lambda row: _selection_key(seed, row))
        if len(choices) < count:
            raise ValueError(f"Not enough disjoint contexts in {family}")
        for row in choices[:count]:
            selected.append({
                "record_id": row["record_id"],
                "context_id": row.get("context_id", row["record_id"]),
                "layout_id": row["layout_id"],
                "split": "test",
                "family": family.value,
                "start_observation": row["start_observation"],
                "task_goal": row["task_goal"],
                "previously_evaluated_in_lora_matrix": row.get("previously_evaluated_in_lora_matrix"),
            })
    return selected


def _write_new(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != contents:
            raise FileExistsError(f"Refusing to overwrite a different frozen selection: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(contents, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(data: Path, ids_path: Path, calibration: Path, output: Path, seed: int) -> dict[str, Any]:
    frozen = json.loads(ids_path.read_text(encoding="utf-8"))
    if frozen.get("dataset_sha256") != file_sha256(data):
        raise ValueError("Frozen test IDs refer to a different dataset")
    calibration_rows = read_records(calibration)
    if not calibration_rows or any(row.get("split") not in {"train", "validation"} for row in calibration_rows):
        raise ValueError("Calibration candidates must contain train or validation records")
    excluded = {row["layout_id"] for row in calibration_rows}
    selected = freeze_contexts(read_records(data), frozen["record_ids"], excluded, DEFAULT_COUNTS, seed)
    encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
    contexts_path = output / "test_contexts.jsonl"
    _write_new(contexts_path, encoded)
    test_ids = {"schema_version": 1, "dataset_sha256": file_sha256(data),
                "record_ids": [row["record_id"] for row in selected]}
    _write_new(output / "test_ids.json", json.dumps(test_ids, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "selection": "sha256(seed, canonical layout ID, record ID), independent of saved model predictions",
        "seed": seed,
        "source_dataset_sha256": file_sha256(data),
        "source_ids_sha256": file_sha256(ids_path),
        "calibration_candidates_sha256": file_sha256(calibration),
        "test_contexts_sha256": file_sha256(contexts_path),
        "test_ids_sha256": file_sha256(output / "test_ids.json"),
        "counts_by_family": dict(Counter(row["family"] for row in selected)),
        "test_record_ids": [row["record_id"] for row in selected],
        "test_layout_ids": [row["layout_id"] for row in selected],
        "prior_exposure": "Exploratory: source cohort and some contexts were previously evaluated.",
        "input_contract": "Only current observation and task are generator inputs; IDs are for audit.",
    }
    _write_new(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    result = prepare(args.data, args.test_ids, args.calibration, args.output, args.seed)
    print(json.dumps({"contexts": result["counts_by_family"], "manifest": str(args.output / "manifest.json")}))


if __name__ == "__main__":
    main()
