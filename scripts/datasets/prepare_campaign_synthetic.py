"""Prepare verified synthetic witnesses in a private, resumable local staging area."""
import argparse
import contextlib
import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, state_hash, to_rows
from carl.icl.datasets import reverse_board


def prepare_one(seed: int, destination: Path) -> None:
    """Generate and atomically save one board; do not print its witness."""
    with contextlib.redirect_stdout(io.StringIO()):
        board, actions, metadata = reverse_board(32, 10, seed)
    write_json(destination, {"seed": seed, "state": to_rows(board), "actions": actions,
                             "state_hash": state_hash(board), "layout_id": canonical_layout_id(board),
                             "generator_metadata": metadata})


def main() -> int:
    """Build independent seeds in parallel, preserving prior completed records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1 or args.seed_base < 0:
        parser.error("seed-base must be nonnegative; count and workers must be positive")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(args.output, 0o700)
    jobs = [(seed, args.output / f"{seed}.json") for seed in range(args.seed_base, args.seed_base + args.count)]
    remaining = [(seed, path) for seed, path in jobs if not path.exists()]
    print(f"Private synthetic staging: {len(jobs) - len(remaining)}/{len(jobs)} already prepared", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(prepare_one, seed, path) for seed, path in remaining]
        for index, future in enumerate(futures, start=1):
            future.result()
            if index % 20 == 0 or index == len(futures):
                print(f"Prepared {len(jobs) - len(remaining) + index}/{len(jobs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
