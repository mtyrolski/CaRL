"""Build a resumable, cross-shard index of full initial Sokoban layouts."""
import argparse
import hashlib
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np

from carl.icl.boards import canonical_layout_id, state_hash, tiles


def index_shard(path: Path) -> tuple[str, str, list[tuple[int, str, str, int, int]], int]:
    """Record initial-state identities only; do not export solutions or later states."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = joblib.load(path)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    rows = []
    invalid = 0
    for index, trajectory in enumerate(trajectories):
        try:
            if len(trajectory) < 2:
                invalid += 1
                continue
            state = np.asarray(trajectory[0])
            board = tiles(state)
            if board.shape != (12, 12) or int(np.isin(board, [3, 4]).sum()) != 4:
                invalid += 1
                continue
            rows.append((index, canonical_layout_id(state), state_hash(state), len(trajectory),
                         int(np.any(board == 4))))
        except (TypeError, ValueError, IndexError):
            invalid += 1
    return str(path), digest, rows, invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("rl-data/validation/sokoban/offline/12-12-4"))
    parser.add_argument("--index", type=Path, default=Path("data/icl/.private_campaign_shards.sqlite"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    files = sorted(args.source.glob("sokoban_12_12_4_trajectories_part_*.pkl"),
                   key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not files:
        raise FileNotFoundError(f"No trajectory shards at {args.source}")
    args.index.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.index) as db:
        db.execute("CREATE TABLE IF NOT EXISTS shards (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, records INTEGER NOT NULL, invalid INTEGER NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS boards (path TEXT NOT NULL, trajectory_index INTEGER NOT NULL, "
                   "layout_id TEXT NOT NULL, state_hash TEXT NOT NULL, trajectory_states INTEGER NOT NULL, "
                   "unsolved INTEGER NOT NULL, PRIMARY KEY(path, trajectory_index))")
        db.execute("CREATE INDEX IF NOT EXISTS boards_layout ON boards(layout_id)")
        completed = {row[0]: row[1] for row in db.execute("SELECT path, sha256 FROM shards")}
        todo = []
        for path in files:
            if str(path) in completed:
                if hashlib.sha256(path.read_bytes()).hexdigest() != completed[str(path)]:
                    raise ValueError(f"Source shard changed since indexing: {path}")
            else:
                todo.append(path)
        print(f"Indexing real shards: {len(files) - len(todo)}/{len(files)} already indexed", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for done, (path, digest, rows, invalid) in enumerate(executor.map(index_shard, todo), start=1):
                with db:
                    db.executemany("INSERT INTO boards VALUES (?, ?, ?, ?, ?, ?)", [(path, *row) for row in rows])
                    db.execute("INSERT INTO shards VALUES (?, ?, ?, ?)", (path, digest, len(rows), invalid))
                print(f"Indexed shards: {len(files) - len(todo) + done}/{len(files)}", flush=True)
        total = db.execute("SELECT COUNT(*) FROM boards").fetchone()[0]
        unique = db.execute("SELECT COUNT(DISTINCT layout_id) FROM boards").fetchone()[0]
        cross = db.execute("SELECT COUNT(*) FROM (SELECT layout_id FROM boards GROUP BY layout_id HAVING COUNT(DISTINCT path)>1)").fetchone()[0]
        print(f"Indexed {total} initial states, {unique} canonical layouts, {cross} layouts repeated across shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
