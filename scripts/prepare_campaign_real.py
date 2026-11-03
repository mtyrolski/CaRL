"""Select and replay real full-board trajectories from the cross-shard index."""
import contextlib
import hashlib
import io
import json
import os
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, state_hash, to_rows
from scripts.prepare_fresh_trajectory_holdout import verified_trajectory

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/icl/.private_campaign_shards.sqlite"
HISTORY = ROOT / "analysis/luna_icl/campaign_100/history_snapshot.json"
PILOT = ROOT / "data/icl/campaign_100_pilot/real_12x12x4/test.json"
STAGING = ROOT / "data/icl/.private_campaign_real_staging"
SELECTION = ROOT / "data/icl/.private_campaign_real_selection.json"
SEED = 20261112
SCALE_COUNT = 4
HOLDOUT_COUNT = 500


def verify_batch(path: str, candidates: list[dict[str, Any]]) -> list[tuple[str, bool]]:
    """Replay candidate source trajectories once and stage their private witnesses."""
    raw = joblib.load(ROOT / path)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    outcomes = []
    for candidate in candidates:
        output = STAGING / f"{candidate['source_sha256']}_{candidate['source_index']}.json"
        if output.exists():
            row = json.loads(output.read_text(encoding="utf-8"))
            valid = (row["layout_id"] == candidate["layout_id"]
                     and row["state_hash"] == candidate["state_hash"])
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    states, actions = verified_trajectory(np.asarray(trajectories[candidate["source_index"]]))
                initial = states[0]
                valid = (canonical_layout_id(initial) == candidate["layout_id"]
                         and state_hash(initial) == candidate["state_hash"] and len(actions) >= 16)
                if valid:
                    write_json(output, {**candidate, "state": to_rows(initial), "actions": actions})
            except (ValueError, IndexError, TypeError):
                valid = False
        outcomes.append((candidate["candidate_id"], valid))
    return outcomes


def main() -> int:
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    from carl.icl.boards import from_rows
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))["instances"]
    excluded = set(history["observed_layout_ids"]) | set(history["reserved_layout_ids"])
    excluded.update(canonical_layout_id(from_rows(item["state"])) for item in pilot)
    with sqlite3.connect(INDEX) as db:
        indexed = dict(db.execute("SELECT path, sha256 FROM shards"))
        directory = ROOT / "rl-data/validation/sokoban/offline/12-12-4"
        available = {str(path.relative_to(ROOT)): path for path in directory.glob("sokoban_12_12_4_trajectories_part_*.pkl")}
        if not available or set(indexed) != set(available):
            raise ValueError("Full cross-shard index is incomplete; run index_campaign_shards.py")
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != indexed[name] for name, path in available.items()):
            raise ValueError("A source shard changed after indexing")
        rows = db.execute("SELECT boards.path, shards.sha256, trajectory_index, layout_id, state_hash "
                          "FROM boards JOIN shards ON boards.path=shards.path "
                          "WHERE trajectory_states > 16 AND unsolved=1").fetchall()
    candidates = []
    for path, digest, index, layout, state in rows:
        if layout in excluded:
            continue
        identifier = f"{digest}:{index}"
        rank = hashlib.sha256(f"carl-real-selection-v1:{SEED}:{identifier}".encode()).hexdigest()
        candidates.append({"source_path": path, "source_sha256": digest, "source_index": index,
                           "layout_id": layout, "state_hash": state, "candidate_id": identifier, "rank": rank})
    candidates.sort(key=lambda item: (item["rank"], item["candidate_id"]))
    unique = []
    seen = set(excluded)
    for candidate in candidates:
        if candidate["layout_id"] not in seen:
            seen.add(candidate["layout_id"])
            unique.append(candidate)
    required = SCALE_COUNT + HOLDOUT_COUNT
    print(f"Available distinct real candidate layouts after exclusions: {len(unique)}; required: {required}", flush=True)
    if len(unique) < required:
        raise ValueError(f"Actual candidate shortage: found {len(unique)} distinct layouts, need {required}")
    STAGING.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(STAGING, 0o700)
    accepted = []
    cursor = 0
    with ProcessPoolExecutor(max_workers=6) as executor:
        while len(accepted) < required and cursor < len(unique):
            batch = unique[cursor:cursor + max(128, required - len(accepted) + 48)]
            cursor += len(batch)
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in batch:
                grouped[item["source_path"]].append(item)
            results = executor.map(verify_batch, grouped.keys(), grouped.values())
            valid = {identifier for group in results for identifier, ok in group if ok}
            accepted.extend(item for item in batch if item["candidate_id"] in valid)
            print(f"Verified distinct real candidates: {min(len(accepted), required)}/{required}", flush=True)
    if len(accepted) < required:
        raise ValueError(f"Verified witness shortage: only {len(accepted)} of {required} required real layouts")
    selected = accepted[:required]
    selection = {"version": 1, "selection_seed": SEED, "required": required,
                 "history_sha256": hashlib.sha256(HISTORY.read_bytes()).hexdigest(),
                 "scale_validation": selected[:SCALE_COUNT], "holdout1000": selected[SCALE_COUNT:]}
    if SELECTION.exists() and json.loads(SELECTION.read_text(encoding="utf-8")) != selection:
        raise ValueError("Existing real selection differs; retain it and investigate provenance")
    write_json(SELECTION, selection)
    print(f"Saved private selection: {SCALE_COUNT} scale-validation + {HOLDOUT_COUNT} real holdout layouts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
