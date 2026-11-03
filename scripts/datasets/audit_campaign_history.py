"""Freeze canonical layout exclusions from historical manifests and saved queries."""
import ast
import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/icl"
OUTPUT = ROOT / "analysis/luna_icl/campaign_100/history_snapshot.json"
INDEX = DATA / ".private_campaign_shards.sqlite"
EXCLUDED_PARTS = frozenset({"campaign_100", "campaign_100_pilot", "campaign_100_pilot_staging",
                            "campaign_dataset_v2", "campaign_dataset_v2_private",
                            ".private_campaign_holdout_staging", ".private_campaign_scale_staging",
                            ".private_campaign_real_staging", ".private_campaign_real_selection.json",
                            "campaign_data"})
RESERVED_PARTS = frozenset({"component_study_100", "component_study_100_clean"})
ID_PATTERN = re.compile(r"trajectory_(\d+)(?!\d)")


def extract(value: Any, layouts: set[str], indices: set[int], depth: int = 0) -> None:
    """Collect hashes and old trajectory indices without retaining any board or solution."""
    if depth > 25:
        return
    if isinstance(value, str):
        indices.update(int(match) for match in ID_PATTERN.findall(value))
        text = value.strip()
        lines = text.splitlines()
        if len(lines) in (12, 32) and all(len(row) == len(lines) and set(row) <= set("#-.$*@+") for row in lines):
            extract(lines, layouts, indices, depth + 1)
        if text.startswith(("{", "[")):
            try:
                parsed = json.loads(text)
            except (ValueError, RecursionError):
                return
            extract(parsed, layouts, indices, depth + 1)
    elif isinstance(value, list):
        if len(value) in (12, 32) and all(isinstance(row, str) and len(row) == len(value)
                                             and set(row) <= set("#-.$*@+") for row in value):
            try:
                layouts.add(canonical_layout_id(from_rows(value)))
            except ValueError:
                pass
        elif len(value) in (12, 32) and all(isinstance(row, list) and len(row) == len(value) for row in value):
            try:
                array = np.asarray(value, dtype=np.float32)
                if array.shape == (len(value), len(value), 7):
                    layouts.add(canonical_layout_id(array))
                else:
                    for item in value:
                        extract(item, layouts, indices, depth + 1)
            except (ValueError, TypeError):
                for item in value:
                    extract(item, layouts, indices, depth + 1)
        else:
            for item in value:
                extract(item, layouts, indices, depth + 1)
    elif isinstance(value, dict):
        for item in value.values():
            extract(item, layouts, indices, depth + 1)


def notebook_literals(document: dict[str, Any], layouts: set[str], indices: set[int]) -> None:
    """Inspect literal board lists in notebook source without executing cells."""
    for cell in document.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) in (12, 32):
                try:
                    extract(list(ast.literal_eval(node)), layouts, indices)
                except (TypeError, ValueError):
                    continue


def history_files() -> Iterator[Path]:
    """Enumerate past manifests, notebooks, logs as JSON and locally cached queries."""
    for base in (ROOT / "analysis", ROOT / "notebooks", ROOT / "configs/icl", DATA):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".ipynb"} and not EXCLUDED_PARTS.intersection(path.parts):
                yield path
    yield from sorted(ROOT.parent.glob("*.ipynb"))


def main() -> int:
    if not INDEX.exists():
        raise FileNotFoundError("Run scripts/datasets/index_campaign_shards.py first")
    observed: set[str] = set()
    reserved: set[str] = set()
    observed_indices = {199, 434, 732}
    reserved_indices: set[int] = set()
    files = []
    failures = []
    for path in history_files():
        is_reserved = bool(RESERVED_PARTS.intersection(path.parts))
        layouts = reserved if is_reserved else observed
        indices = reserved_indices if is_reserved else observed_indices
        try:
            raw = path.read_bytes()
            document = json.loads(raw)
            extract(document, layouts, indices)
            if path.suffix == ".ipynb" and isinstance(document, dict):
                notebook_literals(document, layouts, indices)
            files.append({"path": str(path.relative_to(ROOT.parent)), "sha256": hashlib.sha256(raw).hexdigest(),
                          "classification": "reserved" if is_reserved else "historical"})
        except (OSError, ValueError, RecursionError) as error:
            failures.append({"path": str(path), "error_type": type(error).__name__})
    if failures:
        raise ValueError(f"Cannot audit {len(failures)} historical files; first: {failures[0]}")
    with sqlite3.connect(INDEX) as db:
        for indices, layouts in ((observed_indices, observed), (reserved_indices, reserved)):
            for index in sorted(indices):
                rows = db.execute("SELECT layout_id FROM boards WHERE path LIKE ? AND trajectory_index=?",
                                  ("%/sokoban_12_12_4_trajectories_part_1.pkl", index)).fetchall()
                if rows:
                    layouts.add(rows[0][0])
        known = {row[0] for row in db.execute("SELECT trajectory_index FROM boards WHERE path LIKE ?",
                                             ("%/sokoban_12_12_4_trajectories_part_1.pkl",))}
        missing_required = sorted({199, 434, 732} - known)
        if missing_required:
            raise ValueError(f"Cannot resolve explicitly excluded trajectory IDs: {missing_required}")
    snapshot = {"version": 1, "canonicalization": "carl-layout-d4-v1", "files": files,
                "observed_layout_ids": sorted(observed), "reserved_layout_ids": sorted(reserved),
                "observed_legacy_trajectory_indices": sorted(observed_indices),
                "reserved_legacy_trajectory_indices": sorted(reserved_indices),
                "explicitly_excluded_indices_in_part_1": [199, 434, 732]}
    if OUTPUT.exists():
        frozen = json.loads(OUTPUT.read_text(encoding="utf-8"))
        exclusion_keys = ("observed_layout_ids", "reserved_layout_ids", "observed_legacy_trajectory_indices",
                          "reserved_legacy_trajectory_indices", "explicitly_excluded_indices_in_part_1")
        if any(frozen[key] != snapshot[key] for key in exclusion_keys):
            raise ValueError(f"History exclusions changed; inspect and choose a new campaign: {OUTPUT}")
        print("Frozen historical layout exclusions are unchanged; preserving original source-file snapshot")
    else:
        write_json(OUTPUT, snapshot)
    print(f"Audited {len(files)} history files; {len(observed)} observed and {len(reserved)} reserved layout IDs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
