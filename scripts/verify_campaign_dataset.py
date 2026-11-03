"""Replay all private campaign witnesses and validate public identities offline."""
import argparse
import contextlib
import io
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.campaign_data import FAMILIES, sha256_file, verify_record
from carl.icl.datasets import _public_instances, environment, verify_solution

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data/icl/campaign_dataset_v2"
PRIVATE = ROOT / "data/icl/campaign_dataset_v2_private"
SPLITS = ("component_validation", "scale_validation", "pilot100", "holdout1000")


def replay(job: tuple[dict[str, Any], list[int]]) -> int:
    """Replay a witness without writing or printing its actions or states."""
    record, actions = job
    with contextlib.redirect_stdout(io.StringIO()):
        initial = from_rows(record["state"])
        if len(actions) < 2 or environment(initial).is_solved(initial):
            raise ValueError("Solved or one-action fixture cannot be a full-board campaign instance")
        verify_solution(initial, actions)
    return len(actions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--identity-only", action="store_true", help="Check manifests and full-state provenance without replay")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest = json.loads((PUBLIC / "campaign_dataset_manifest.json").read_text(encoding="utf-8"))
    groups = [("demonstration", "real_12x12x4")] + [(split, family) for split in SPLITS for family in FAMILIES]
    jobs = []
    identities: dict[str, set[str]] = {}
    witness_hashes: dict[str, str] = {}
    db = sqlite3.connect(ROOT / "data/icl/.private_campaign_shards.sqlite")
    for split, family in groups:
        public_path = PUBLIC / ("demonstrations.json" if split == "demonstration" else f"{family}/{split}.json")
        if split != "demonstration" and sha256_file(public_path) != manifest["manifest_sha256"][f"{family}/{split}.json"]:
            raise ValueError("Public manifest checksum mismatch")
        records = json.loads(public_path.read_text(encoding="utf-8"))["instances"]
        if split != "demonstration":
            _public_instances(public_path)
        witness_path = PRIVATE / split / f"{family}.json"
        witness_hashes[f"{split}/{family}"] = sha256_file(witness_path)
        witnesses = json.loads(witness_path.read_text(encoding="utf-8"))["witnesses"]
        by_id = {row["board_id"]: row["actions"] for row in witnesses}
        if len(by_id) != len(records) or {row["board_id"] for row in records} != set(by_id):
            raise ValueError("Public/private board IDs do not match one-to-one")
        key = f"{split}/{family}"
        identities[key] = set()
        for record in records:
            verify_record(record, family, split)
            if family == "real_12x12x4":
                source = db.execute(
                    "SELECT boards.state_hash, boards.layout_id, shards.sha256 "
                    "FROM boards JOIN shards ON boards.path=shards.path "
                    "WHERE boards.path=? AND boards.trajectory_index=?",
                    (record["source_path"], record["source_index"]),
                ).fetchone()
                if source != (record["state_hash"], record["layout_id"], record["source_sha256"]):
                    raise ValueError("Real board is not the indexed source trajectory's full initial state")
            if record["layout_id"] in identities[key]:
                raise ValueError(f"Duplicate canonical layout in {key}")
            identities[key].add(record["layout_id"])
            jobs.append((record, by_id[record["board_id"]]))
    names = list(identities)
    if any(identities[a] & identities[b] for index, a in enumerate(names) for b in names[index + 1:]):
        raise ValueError("Campaign splits contain overlapping layouts")
    db.close()
    if args.identity_only:
        print(f"Verified {len(jobs)} initial-state identities and split memberships without exposing boards")
        return 0
    print(f"Replaying {len(jobs)} private witnesses without displaying solutions", flush=True)
    total_actions = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, length in enumerate(executor.map(replay, jobs, chunksize=4), start=1):
            total_actions += length
            if index % 100 == 0 or index == len(jobs):
                print(f"Witnesses verified: {index}/{len(jobs)}", flush=True)
    write_json(ROOT / "analysis/luna_icl/campaign_100/witness_verification.json", {
        "public_manifest_sha256": sha256_file(PUBLIC / "campaign_dataset_manifest.json"),
        "private_witness_sha256": witness_hashes, "witnesses_verified": len(jobs),
        "primitive_actions_verified": total_actions, "all_transitions_valid": True,
        "all_final_states_solved": True, "real_initial_states_match_indexed_sources": True,
        "no_solved_or_one_action_fixtures": True,
    })
    print("All transitions valid; every witness ends in is_solved=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
