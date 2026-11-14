#!/usr/bin/env python3
"""Bind an existing frozen evaluation order to its exact dataset for ICL."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import evaluation_records
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records


def prepare(data: Path, original_ids: Path, output: Path) -> dict[str, Any]:
    """Create a new immutable ID file; never overwrite the original selection."""
    if output.resolve() in {data.resolve(), original_ids.resolve()}:
        raise ValueError("The ICL ID file must be separate from its sources")
    records = read_records(data, DataSplit.TEST)
    selected = evaluation_records(records, str(original_ids), 0)
    source = json.loads(original_ids.read_text(encoding="utf-8"))
    identifiers = source["record_ids"]
    if not identifiers or identifiers != [row["record_id"] for row in selected]:
        raise ValueError("Frozen test order is empty or does not match its dataset")
    encoded = json.dumps({"schema_version": 1, "dataset_sha256": file_sha256(data), "record_ids": identifiers},
                         indent=2, ensure_ascii=False) + "\n"
    manifest = {
        "schema_version": 1,
        "dataset_path": str(data.resolve()), "dataset_sha256": file_sha256(data),
        "original_ids_path": str(original_ids.resolve()), "original_ids_sha256": file_sha256(original_ids),
        "icl_ids_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "record_count": len(identifiers),
        "selection": "Exact original order, bound to the dataset hash; no results or model outputs consulted.",
    }
    manifest_path = output.with_name(output.stem + ".manifest.json")
    files = {output: encoded, manifest_path: json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"}
    for path, content in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite a different frozen file: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--original-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.original_ids, args.output), indent=2))


if __name__ == "__main__":
    main()
