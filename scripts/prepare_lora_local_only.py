#!/usr/bin/env python3
"""Prepare a local-label training ablation without touching evaluation data."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import file_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "analysis/intent_generator_20261114T032232Z/dataset.jsonl"


def filter_records(source: str) -> tuple[str, list[str], dict[str, int]]:
    """Remove historical Terra annotations from train; retain all other lines byte-for-byte."""
    retained, removed = [], []
    counts: Counter[str] = Counter()
    for line in source.splitlines(keepends=True):
        if not line.strip():
            retained.append(line)
            continue
        row: dict[str, Any] = json.loads(line)
        provenance = row.get("annotation_provenance", {})
        model, kind = provenance.get("model"), provenance.get("kind")
        if row["split"] == "train" and model == "gpt-6.1-terra":
            if kind != "historical_shared_context_codex_annotation":
                raise ValueError("Ambiguous Terra training annotation provenance")
            removed.append(row["record_id"])
            continue
        retained.append(line)
        counts[row["split"]] += 1
    return "".join(retained), removed, dict(counts)


def prepare(source: Path, output: Path) -> dict[str, Any]:
    text = source.read_text(encoding="utf-8")
    filtered, removed, counts = filter_records(text)
    if len(removed) != 15 or counts != {"train": 343, "validation": 66, "test": 100}:
        raise ValueError("Source counts differ from the frozen dataset")
    manifest = {
        "schema_version": 1,
        "source_path": str(source.resolve()), "source_sha256": file_sha256(source),
        "data_sha256": hashlib.sha256(filtered.encode()).hexdigest(),
        "selection": "Remove only historical_shared_context_codex_annotation by gpt-6.1-terra from train.",
        "retained_counts": counts, "removed_train_count": len(removed), "removed_train_record_ids": removed,
        "invariant": "Every validation/test line remains unchanged and in original order; no model outputs were used.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_name(output.stem + ".manifest.json")
    files = {output: filtered, manifest_path: json.dumps(manifest, indent=2) + "\n"}
    for path, content in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite different frozen data: {path}")
    for path, content in files.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
