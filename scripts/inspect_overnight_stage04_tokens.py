#!/usr/bin/env python3
"""Check complete token lengths for every possible training orientation."""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_generator_experiment import training_lengths


def file_sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inspect(tokenizer: Any, paths: list[Path], max_length: int, multiple: int) -> dict[str, Any]:
    if not paths or max_length < 1 or multiple < 1:
        raise ValueError("dataset paths and positive token limits are required")
    validation = None
    results = []
    maximum = 0
    for path in paths:
        rows = read_records(path)
        if any(row.get("split") not in ("train", "validation") for row in rows):
            raise ValueError("training inputs may contain train and validation records only")
        train = [row for row in rows if row["split"] == "train"]
        valid = [row for row in rows if row["split"] == "validation"]
        if not train or not valid:
            raise ValueError("each dataset needs training and validation examples")
        train_layouts = {row["layout_id"] for row in train}
        validation_layouts = {row["layout_id"] for row in valid}
        if train_layouts & validation_layouts:
            raise ValueError("training and validation layouts overlap")
        # Entire checked validation records must be identical, not just their IDs.
        canonical = json.dumps(valid, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if validation is not None and validation != canonical:
            raise ValueError("matched arms must share exactly the same validation records")
        validation = canonical
        stats = training_lengths(tokenizer, train, valid, Namespace(max_length=max_length,
                                                                   augmentation="d4", pad_to_length=0))
        maximum = max(maximum, stats["max"])
        results.append({"path": str(path.resolve()), "sha256": file_sha(path), "train_records": len(train),
                        "train_layouts": len(train_layouts), "validation_records": len(valid),
                        "validation_layouts": len(validation_layouts), "token_lengths_all_d4": stats})
    padded = ((maximum + multiple - 1) // multiple) * multiple
    if padded > max_length:
        raise ValueError("complete examples exceed the common padded token budget")
    return {"schema_version": 1, "datasets": results, "model_max_length": max_length,
            "common_pad_to_length": padded, "length_rounding_multiple": multiple,
            "selection": "all eight training orientations; validation in its original orientation",
            "validation_sha256": hashlib.sha256(validation.encode("utf-8")).hexdigest(),
            "truncated_examples": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--multiple", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    result = inspect(tokenizer, args.data, args.max_length, args.multiple)
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
