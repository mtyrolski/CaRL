#!/usr/bin/env python3
"""Atomically collect per-method stage-five results without hiding missing jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.intent_experiment_supervisor import read_json
from scripts.intent_experiment_supervisor import write_json
from scripts.run_overnight_cpu_policy import read_jsonl

SOURCE_DIRS = {"adasubs": "adasubs_evaluation", "ksubs": "ksubs_evaluation",
               "direct_policy": "policy_evaluation", "primitive_astar": "primitive_evaluation"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def merge(stage: Path) -> dict[str, Any]:
    freeze = read_json(stage / "evaluation_freeze.json")
    boards_path = Path(freeze["input_boards"])
    if digest(boards_path.read_bytes()) != freeze["input_boards_sha256"]:
        raise ValueError("Frozen input boards do not match their recorded hash")
    boards = [row for row in read_jsonl(boards_path) if row["split"] == "evaluation"]
    by_layout = {row["layout_id"]: row for row in boards}
    if not boards or len(by_layout) != len(boards):
        raise ValueError("Expected distinct evaluation layouts")
    methods = freeze["methods"]
    if len(methods) != len(set(methods)) or set(methods) - SOURCE_DIRS.keys():
        raise ValueError("Unknown or duplicated method in frozen configuration")
    dest = stage / "evaluation/shards"
    dest.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    sources: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for method in methods:
        source_root = stage / SOURCE_DIRS[method] / "shards"
        for source in sorted(source_root.rglob("*.json")):
            data = source.read_bytes()
            row = json.loads(data)
            layout = row.get("layout_id")
            if layout not in by_layout or row.get("method") != method:
                raise ValueError(f"Unexpected board or method in {source}")
            board = by_layout[layout]
            if any(row.get(key) != board[key] for key in ("family", "source_family", "split")):
                raise ValueError(f"Shard metadata does not match frozen board: {source}")
            relative = f"{method}/{layout}.json"
            if relative in hashes:
                raise ValueError(f"Duplicate board/method shard: {relative}")
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"Previously merged shard changed: {relative}")
            if not target.exists():
                temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
                try:
                    with temporary.open("wb") as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            hashes[relative] = digest(data)
            sources[relative] = str(source.resolve())
            counts[method] += 1
    existing = {str(path.relative_to(dest)) for path in dest.rglob("*.json")}
    if existing != set(hashes):
        raise ValueError("Merged directory has stale or unlisted shards")
    planned = len(boards) * len(methods)
    result = {"schema_version": 1, "state": "complete" if len(hashes) == planned else "partial",
              "planned_jobs": planned, "merged_jobs": len(hashes), "methods": dict(counts),
              "shards_sha256": hashes, "source_paths": sources,
              "boards_sha256": freeze["input_boards_sha256"], "freeze_sha256": digest(
                  (stage / "evaluation_freeze.json").read_bytes())}
    write_json(stage / "evaluation/merge_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    args = parser.parse_args()
    result = merge(args.stage.resolve())
    print(json.dumps({key: result[key] for key in ("state", "planned_jobs", "merged_jobs", "methods")}))


if __name__ == "__main__":
    main()
