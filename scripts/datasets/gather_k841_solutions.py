"""Gather trusted local Experience batches; report corrupt files and unknown records."""
import argparse
from glob import glob
from pathlib import Path

import joblib

from carl.icl.api import write_json
from carl.planners.base import Experience


def load_records(pattern: str) -> tuple[list[tuple[str, int, Experience]], dict]:
    paths = sorted(glob(pattern))
    records, failures = [], []
    loaded, ignored = 0, 0
    for path in paths:
        try:
            payload = joblib.load(path)
        except Exception as error:
            failures.append({"path": path, "error_type": type(error).__name__})
            continue
        loaded += 1
        items = payload if isinstance(payload, (list, tuple)) else [payload]
        for index, item in enumerate(items):
            if isinstance(item, Experience):
                records.append((path, index, item))
            else:
                ignored += 1
    solved = sum(bool(record.solution.solved) for _, _, record in records)
    summary = {
        "matched_files_count": len(paths),
        "loaded_files_count": loaded,
        "failed_files_count": len(failures),
        "failures": failures,
        "counts": {
            "all_records": len(records),
            "solved": solved,
            "unsolved": len(records) - solved,
            "unknown_items_ignored": ignored,
        },
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records, summary = load_records(args.input_glob)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = [record for _, _, record in records]
    joblib.dump(all_records, args.output_dir / "merged_all.joblib")
    joblib.dump(
        [record for record in all_records if record.solution.solved],
        args.output_dir / "merged_solved.joblib",
    )
    write_json(args.output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
