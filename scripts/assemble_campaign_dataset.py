"""Assemble frozen public manifests and physically separate private witnesses."""
import argparse
import contextlib
import hashlib
import inspect
import io
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import to_rows
from carl.icl.campaign_data import FAMILIES, canonical_json_hash, public_board, sha256_file, summarize, verify_record
from carl.icl.datasets import _walks, reverse_board
from scripts.prepare_fresh_trajectory_holdout import verified_trajectory

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/icl"
PUBLIC = DATA / "campaign_dataset_v2"
PRIVATE = DATA / "campaign_dataset_v2_private"
HISTORY = ROOT / "analysis/luna_icl/campaign_100/history_snapshot.json"
REAL_SELECTION = DATA / ".private_campaign_real_selection.json"
REAL_STAGE = DATA / ".private_campaign_real_staging"
SYNTH_STAGE = DATA / ".private_campaign_holdout_staging"
SCALE_STAGE = DATA / ".private_campaign_scale_staging"
OLD_PILOT = DATA / "campaign_100_pilot"
INDEX = DATA / ".private_campaign_shards.sqlite"
REAL, SYNTH = FAMILIES
GENERATOR = "carl.icl.datasets.reverse_board-v1"
GENERATOR_SHA = hashlib.sha256((inspect.getsource(reverse_board) + inspect.getsource(_walks)).encode()).hexdigest()
PART1 = "rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_1.pkl"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def from_legacy_real(row: dict[str, Any], split: str, source_sha: str) -> dict[str, Any]:
    old_id = row["id"]
    if not old_id.startswith("trajectory_") or not old_id[11:].isdigit():
        raise ValueError("Legacy real ID is not a source trajectory ID")
    index = int(old_id[11:])
    return public_board(rows=row["state"], family=REAL, split=split, trajectory_id=f"part_1:{index}",
                        source_path=PART1, source_sha256=source_sha, source_index=index,
                        generator_version=None, seed=None)


def from_legacy_synthetic(row: dict[str, Any], split: str, seed: int) -> dict[str, Any]:
    return public_board(rows=row["state"], family=SYNTH, split=split, trajectory_id=f"generated:{seed}",
                        source_path="carl/icl/datasets.py:reverse_board", source_sha256=GENERATOR_SHA,
                        source_index=None, generator_version=GENERATOR, seed=seed)


def build() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build deterministic artifact objects without returning test solutions to inference."""
    history = read(HISTORY)
    real = read(REAL_SELECTION)
    if real["history_sha256"] != sha256_file(HISTORY):
        raise ValueError("Real selection does not match frozen history snapshot")
    base = DATA / "training.json"
    training = read(base)
    source_sha = sha256_file(ROOT / PART1)
    if training["provenance"]["source_sha256"] != source_sha:
        raise ValueError("Training data and part-1 source hashes differ")
    raw = joblib.load(ROOT / PART1)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    groups: dict[str, list[dict[str, Any]]] = {}
    witnesses: dict[str, list[dict[str, Any]]] = {}

    def add(split: str, family: str, public: dict[str, Any], actions: list[int]) -> None:
        verify_record(public, family, split)
        key = f"{split}/{family}"
        groups.setdefault(key, []).append(public)
        witnesses.setdefault(key, []).append({"board_id": public["board_id"], "actions": actions})

    for context in training["contexts"]:
        row = {"id": context["id"], "state": context["state"]}
        public = from_legacy_real(row, "demonstration", source_sha)
        with contextlib.redirect_stdout(io.StringIO()):
            states, actions = verified_trajectory(np.asarray(trajectories[public["source_index"]]))
        if to_rows(states[0]) != row["state"]:
            raise ValueError("Training context does not match source initial state")
        add("demonstration", REAL, public, actions)

    old_manifest = read(DATA / "manifest.json")
    old_seeds = {row["id"]: row["seed"] for row in old_manifest["generation"]}
    pilot_metadata = {row["id"]: row for row in read(OLD_PILOT / "campaign_manifest.json")["synthetic"]}
    for split, root, filename in (("component_validation", DATA, "validation"), ("pilot100", OLD_PILOT, "test")):
        for family in FAMILIES:
            records = read(root / family / f"{filename}.json")["instances"]
            private = {item["id"]: item["actions"] for item in read(root / "private_witnesses" / family / f"{filename}.json")}
            for row in records:
                if family == REAL:
                    public = from_legacy_real(row, split, source_sha)
                else:
                    seed = old_seeds[row["id"]] if split == "component_validation" else pilot_metadata[row["id"]]["seed"]
                    public = from_legacy_synthetic(row, split, seed)
                add(split, family, public, private[row["id"]])

    for split in ("scale_validation", "holdout1000"):
        for candidate in real[split]:
            file = REAL_STAGE / f"{candidate['source_sha256']}_{candidate['source_index']}.json"
            row = read(file)
            if any(row[key] != candidate[key] for key in ("source_path", "source_sha256", "source_index", "layout_id", "state_hash")):
                raise ValueError("Staged real candidate differs from frozen selection")
            part = Path(row["source_path"]).stem.rsplit("_", 1)[1]
            public = public_board(rows=row["state"], family=REAL, split=split,
                trajectory_id=f"part_{part}:{row['source_index']}", source_path=row["source_path"],
                source_sha256=row["source_sha256"], source_index=row["source_index"], generator_version=None, seed=None)
            add(split, REAL, public, row["actions"])
        staging = SCALE_STAGE if split == "scale_validation" else SYNTH_STAGE
        start, count = (2026111500, 4) if split == "scale_validation" else (2026120200, 500)
        for seed in range(start, start + count):
            row = read(staging / f"{seed}.json")
            if row["seed"] != seed:
                raise ValueError("Synthetic staging seed mismatch")
            public = from_legacy_synthetic(row, split, seed)
            if public["layout_id"] != row["layout_id"] or public["state_hash"] != row["state_hash"]:
                raise ValueError("Synthetic staged identity mismatch")
            add(split, SYNTH, public, row["actions"])

    expected = {"demonstration/real_12x12x4": 8, "component_validation/real_12x12x4": 4,
        "component_validation/generated_32x32x10": 4, "scale_validation/real_12x12x4": 4,
        "scale_validation/generated_32x32x10": 4, "pilot100/real_12x12x4": 50,
        "pilot100/generated_32x32x10": 50, "holdout1000/real_12x12x4": 500,
        "holdout1000/generated_32x32x10": 500}
    if {name: len(rows) for name, rows in groups.items()} != expected:
        raise ValueError("Campaign group counts do not match the frozen 50+50/500+500 design")
    stats = summarize(groups, {name: [len(item["actions"]) for item in rows] for name, rows in witnesses.items()})
    if any(value for value in stats["intersections"].values()) or any(
        group["count"] != group["unique_layouts"] for group in stats["groups"].values()
    ):
        raise ValueError("Campaign splits overlap or have duplicate/symmetric layouts")
    observed = set(history["observed_layout_ids"])
    reserved = set(history["reserved_layout_ids"])
    for name, rows in groups.items():
        split = name.split("/")[0]
        if split in {"scale_validation", "pilot100", "holdout1000"}:
            forbidden = observed | (reserved if split != "pilot100" else set())
            if any(row["layout_id"] in forbidden for row in rows):
                raise ValueError(f"{name} overlaps earlier observed/reserved data")
    with sqlite3.connect(INDEX) as db:
        shards = db.execute("SELECT COUNT(*) FROM shards").fetchone()[0]
        cross = db.execute("SELECT COUNT(*) FROM (SELECT layout_id FROM boards GROUP BY layout_id HAVING COUNT(DISTINCT path)>1)").fetchone()[0]
    provenance = {"version": 2, "layout_id_algorithm": "carl-layout-d4-v1",
                  "state_hash_algorithm": "carl-state-v1", "historical_snapshot_sha256": sha256_file(HISTORY),
                  "historical_files": len(history["files"]), "shards_indexed": shards,
                  "cross_shard_duplicate_layouts": cross, "source_part1_sha256": source_sha,
                  "real_selection_seed": real["selection_seed"], "synthetic_scale_seed_base": 2026111500,
                  "synthetic_holdout_seed_base": 2026120200, "generator_version": GENERATOR,
                  "generator_source_sha256": GENERATOR_SHA,
                  "pilot_manifest_sha256": sha256_file(OLD_PILOT / "campaign_manifest.json"),
                  "history_note": "Pilot may match the reserved source manifest; neither test cohort matches observed history.",
                  "counts_and_witness_statistics": stats}
    return groups, witnesses, provenance


def artifacts(groups: dict[str, list[dict[str, Any]]], witnesses: dict[str, list[dict[str, Any]]],
              provenance: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    public: dict[str, Any] = {"demonstrations.json": {"provenance": provenance, "instances": groups["demonstration/real_12x12x4"]}}
    private: dict[str, Any] = {}
    for group, rows in groups.items():
        split, family = group.split("/")
        if split != "demonstration":
            public[f"{family}/{split}.json"] = {"provenance": {"version": 2, "family": family, "split": split}, "instances": rows}
        private[f"{split}/{family}.json"] = {"version": 2, "witnesses": witnesses[group]}
    return public, private


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Rebuild in memory and compare all existing artifact identities")
    args = parser.parse_args()
    groups, witnesses, provenance = build()
    public, private = artifacts(groups, witnesses, provenance)
    if args.check:
        if not PUBLIC.exists() or not PRIVATE.exists():
            raise FileNotFoundError("Prepare the campaign dataset before --check")
        for root, items in ((PUBLIC, public), (PRIVATE, private)):
            for name, value in items.items():
                if canonical_json_hash(read(root / name)) != canonical_json_hash(value):
                    raise ValueError(f"Repeated preparation differs: {root / name}")
        hashes = {name: sha256_file(PUBLIC / name) for name in public if name != "demonstrations.json"}
        if read(PUBLIC / "campaign_dataset_manifest.json") != {**provenance, "manifest_sha256": hashes}:
            raise ValueError("Repeated campaign summary or public checksums differ")
        for name in ("training.json", "value_examples.json"):
            if sha256_file(PUBLIC / name) != sha256_file(DATA / name):
                raise ValueError(f"Repeated training artifact differs: {name}")
        if read(ROOT / "analysis/luna_icl/campaign_100/data_integrity.json") != provenance:
            raise ValueError("Repeated integrity report differs")
        print(f"Repeated preparation matches {len(public)} public and {len(private)} private artifacts")
        return 0
    if PUBLIC.exists() or PRIVATE.exists():
        raise FileExistsError("Campaign destination already exists; use --check, never overwrite")
    PUBLIC.mkdir(parents=True)
    PRIVATE.mkdir(parents=True, mode=0o700)
    for root, items in ((PUBLIC, public), (PRIVATE, private)):
        for name, value in items.items():
            write_json(root / name, value)
    shutil.copy2(DATA / "training.json", PUBLIC / "training.json")
    shutil.copy2(DATA / "value_examples.json", PUBLIC / "value_examples.json")
    manifest_hashes = {name: sha256_file(PUBLIC / name) for name in public if name != "demonstrations.json"}
    write_json(PUBLIC / "campaign_dataset_manifest.json", {**provenance, "manifest_sha256": manifest_hashes})
    write_json(ROOT / "analysis/luna_icl/campaign_100/data_integrity.json", provenance)
    print("Prepared public manifests and separate private witnesses; no solutions printed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
