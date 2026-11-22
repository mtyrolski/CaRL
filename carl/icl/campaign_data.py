"""Contracts and integrity summaries for leakage-resistant campaign datasets."""
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.boards import canonical_layout_id, from_rows, level_key, state_hash

SPLITS = ("demonstration", "component_validation", "scale_validation", "pilot100", "holdout1000")
FAMILIES = ("real_12x12x4", "generated_32x32x10")


def sha256_file(path: Path) -> str:
    """Hash an artifact without exposing its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_board(*, rows: list[str], family: str, split: str, trajectory_id: str,
                 source_path: str, source_sha256: str, source_index: int | None,
                 generator_version: str | None, seed: int | None) -> dict[str, Any]:
    """Create a public initial-state record with no solution or future state."""
    board = from_rows(rows)
    digest = state_hash(board)
    identifier = f"{family}-{digest[:24]}"
    return {"id": identifier, "board_id": identifier, "state": rows, "family": family,
            "split": split, "trajectory_id": trajectory_id, "source_path": source_path,
            "source_sha256": source_sha256, "source_index": source_index,
            "generator_version": generator_version, "seed": seed, "state_hash": digest,
            "layout_id": canonical_layout_id(board), "level_key": level_key(board)}


def verify_record(record: Mapping[str, Any], family: str, split: str) -> None:
    """Check shape, box count, and all identities without accessing a witness."""
    rows = record["state"]
    size, boxes = (12, 4) if family == "real_12x12x4" else (32, 10)
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError(f"Invalid {family} board dimensions")
    if sum(row.count("$") + row.count("*") for row in rows) != boxes:
        raise ValueError(f"Invalid {family} box count")
    board = from_rows(rows)
    if (record["family"] != family or record["split"] != split
            or record["layout_id"] != canonical_layout_id(board)
            or record["state_hash"] != state_hash(board) or record["level_key"] != level_key(board)):
        raise ValueError("Campaign record identity mismatch")
    if record["id"] != record["board_id"] or record["id"] != f"{family}-{record['state_hash'][:24]}":
        raise ValueError("Campaign board identifier mismatch")


def summarize(groups: Mapping[str, list[dict[str, Any]]],
              lengths: Mapping[str, list[int]]) -> dict[str, Any]:
    """Summarize counts, pairwise canonical intersections and known witness lengths."""
    identities = {group: {row["layout_id"] for row in records} for group, records in groups.items()}
    pairs = {f"{a} | {b}": len(identities[a] & identities[b])
             for index, a in enumerate(groups) for b in list(groups)[index + 1:]}
    summary: dict[str, Any] = {"groups": {}, "intersections": pairs,
                               "length_note": "Known witness length; not a minimal solution length."}
    for group, rows in groups.items():
        values = np.asarray(lengths[group], dtype=int)
        if len(values) != len(rows):
            raise ValueError(f"Missing witness lengths for {group}")
        summary["groups"][group] = {
            "count": len(rows), "unique_layouts": len(identities[group]),
            "known_witness_actions": {"min": int(values.min()), "q25": float(np.percentile(values, 25)),
                "median": float(np.median(values)), "mean": float(np.mean(values)),
                "q75": float(np.percentile(values, 75)), "max": int(values.max())},
        }
    return summary


def canonical_json_hash(value: Any) -> str:
    """Hash canonical JSON for reproducibility checks independent of file formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
