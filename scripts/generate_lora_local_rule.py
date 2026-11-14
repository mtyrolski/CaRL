#!/usr/bin/env python3
"""Evaluate the current-board local rule on a frozen Sokoban context selection."""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from carl.icl.intent_assessment import local_rule
from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import evaluation_records
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records


def generate(data: Path, ids: Path, output: Path) -> dict[str, Any]:
    """Run the deterministic algorithm with only each current board as input."""
    records = evaluation_records(read_records(data, DataSplit.TEST), str(ids), 0)
    if not records:
        raise ValueError("The frozen test selection is empty")
    rows = []
    for record in records:
        intent = local_rule(record["start_observation"])
        text = json.dumps(intent, ensure_ascii=False, separators=(",", ":")) if intent is not None else ""
        rows.append({"record_id": record["record_id"], "layout_id": record["layout_id"], "split": "test",
                     "method": "local_rule", "raw_output": text, "json_valid": intent is not None,
                     "parsed": intent, "generated_tokens": None, "generation_seconds": None})
    encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    summary = {"status": "complete", "n": len(rows), "method": "local_rule",
               "algorithm": "carl.icl.intent_assessment.local_rule", "model": None,
               "input_fields": ["start_observation"], "data_sha256": file_sha256(data),
               "test_ids_sha256": file_sha256(ids),
               "algorithm_sha256": file_sha256(Path(__file__).resolve().parents[1] / "carl/icl/intent_assessment.py"),
               "generated_tokens": None,
               "description": "Local graph search for an accessible push, not a language model."}
    summary_path = output.with_suffix(".summary.json")
    files = {output: encoded, summary_path: json.dumps(summary, ensure_ascii=False, indent=2) + "\n"}
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in files.items():
        if path.exists() and path.read_text(encoding="utf-8") != contents:
            raise FileExistsError(f"Refusing to overwrite a different result: {path}")
    for path, contents in files.items():
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(contents, encoding="utf-8")
        os.replace(temporary, path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--test-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.data, args.test_ids, args.output), indent=2))


if __name__ == "__main__":
    main()
