"""Export a deterministic sample of trusted Experience records for inspection."""
import argparse
import csv
import random
from pathlib import Path

from carl.icl.api import write_json
from scripts.gather_k841_solutions import load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    records, summary = load_records(args.input_glob)
    selected = random.Random(args.seed).sample(records, min(args.n, len(records)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manual_inspection.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sample_rank",
                "source_path",
                "source_index",
                "solved",
                "actions",
                "finish_reason",
            ],
        )
        writer.writeheader()
        for rank, (source, index, record) in enumerate(selected):
            writer.writerow(
                {
                    "sample_rank": rank,
                    "source_path": source,
                    "source_index": index,
                    "solved": record.solution.solved,
                    "actions": record.solution.action_path,
                    "finish_reason": record.search_info.finished_reason,
                }
            )
    write_json(
        args.output_dir / "summary.json",
        {"seed": args.seed, "requested": args.n, "sampled": len(selected), "loading": summary},
    )


if __name__ == "__main__":
    main()
