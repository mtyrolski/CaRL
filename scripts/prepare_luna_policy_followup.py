#!/usr/bin/env python3
"""Freeze a small disjoint Luna calibration and previously unused test cohort.

This command reads existing local data only. It performs no model or API calls
and refuses to replace an existing selection.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import validate_intent

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "analysis/luna_lora_preparation_20261114/data/calibration_candidates.jsonl"
HOLDOUT = ROOT / "data/icl/campaign_dataset_v2/real_12x12x4/holdout1000.json"
HISTORY = ROOT / "analysis/luna_icl/campaign_100/history_snapshot.json"
FROZEN_SPLIT = ROOT / "analysis/intent_generator_20261114T032232Z/frozen_split.json"
PRIOR_DATASETS = (
    "analysis/intent_generator_20261114T032232Z/dataset.jsonl",
    "analysis/lora_matrix_20261114T065804Z/eval_transfer/dataset.jsonl",
    "analysis/intent_representations_20261114T075324Z/data/dataset.jsonl",
    "analysis/luna_intent_20261114T060730Z/data/records.jsonl",
    "analysis/luna_intent_20261114T060730Z/structured/data/records.jsonl",
    "analysis/luna_intent_20261114T060730Z/final_response/data/records.jsonl",
    "analysis/luna_intent_20261114T060730Z/expanded/data/records.jsonl",
)
TASK_GOAL = "Put every box on a goal."


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def selection_key(seed: int, category: str, value: str) -> str:
    key = f"luna-policy-relative-followup-v1|{seed}|{category}|{value}"
    return hashlib.sha256(key.encode()).hexdigest()


def semantic_intent_key(record: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Canonicalise conjunctions and discard an implied goal-count predicate.

    A conjunction requiring boxes on N explicit goal tiles necessarily implies
    ``boxes_on_goals_at_least(k)`` whenever k <= N. This catches a real
    duplicate missed by sorting the JSON alone. Other, domain-specific logical
    equivalences are not claimed to have been exhaustively decided.
    """
    board = record["start_observation"]
    effects = record["intent"]["effects"]
    forced_goals = {(p["row"], p["col"]) for p in effects if p["predicate"] == "box_at"
                    and board[p["row"]][p["col"]] in ".*+"}
    retained = [p for p in effects if not (p["predicate"] == "boxes_on_goals_at_least"
                                            and p["count"] <= len(forced_goals))]

    def normalized(predicates: list[dict[str, Any]]) -> tuple[str, ...]:
        return tuple(sorted(json.dumps(p, sort_keys=True, separators=(",", ":")) for p in predicates))

    return normalized(retained), normalized(record["intent"]["preserve"])


def select_calibration(rows: list[dict[str, Any]], seed: int, train_count: int,
                       validation_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject whole duplicate contexts, then choose one context per board."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_ids = set()
    for row in rows:
        if row["record_id"] in record_ids or row.get("split") not in {"train", "validation"}:
            raise ValueError("Calibration records must be unique and train/validation only")
        record_ids.add(row["record_id"])
        groups[row["context_id"]].append(row)

    eligible: dict[str, dict[str, list[dict[str, Any]]]] = {"train": defaultdict(list),
                                                            "validation": defaultdict(list)}
    rejected = []
    for context, group in groups.items():
        first = group[0]
        if len(group) != 4 or any((r["layout_id"], r["split"], r["start_observation"], r["task_goal"]) !=
                                  (first["layout_id"], first["split"], first["start_observation"], first["task_goal"])
                                  for r in group):
            raise ValueError(f"Context {context} must have four candidates from the same state")
        keys = [semantic_intent_key(r) for r in group]
        if len(set(keys)) != len(keys):
            rejected.append({"context_id": context, "layout_id": first["layout_id"], "split": first["split"],
                             "record_ids": [r["record_id"] for r in group],
                             "reason": "semantically equivalent conjunctions within context"})
            continue
        eligible[first["split"]][first["layout_id"]].append(group)

    selected = []
    for split, count in (("train", train_count), ("validation", validation_count)):
        boards = eligible[split]
        ordered = sorted(boards, key=lambda layout: selection_key(seed, f"{split}-board", layout))
        if len(ordered) < count:
            raise ValueError(f"Not enough distinct {split} boards after semantic deduplication")
        for layout in ordered[:count]:
            group = min(boards[layout], key=lambda item: selection_key(seed, f"{split}-context", item[0]["context_id"]))
            selected.extend(sorted(group, key=lambda item: selection_key(seed, "candidate", item["record_id"])))
    if {r["layout_id"] for r in selected if r["split"] == "train"} & {
            r["layout_id"] for r in selected if r["split"] == "validation"}:
        raise ValueError("Train and validation layouts overlap")
    return selected, rejected


def audit_fragments(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Check each target and replay every stored primitive action."""
    if not rows:
        raise ValueError("No calibration records")
    env = environment(from_rows(rows[0]["start_observation"]))
    count = 0
    for row in rows:
        source = ROOT / row["source_path"]
        if digest(source) != row["source_sha256"]:
            raise ValueError(f"Source hash changed: {source}")
        states, actions = row["fragment_observations"], row["fragment_actions"]
        if (len(states) != len(actions) + 1 or states[0] != row["start_observation"]
                or states[-1] != row["later_observation"] or validate_intent(row["intent"], states)):
            raise ValueError(f"Invalid annotation or fragment: {row['record_id']}")
        state = from_rows(states[0])
        if canonical_layout_id(state) != row["layout_id"]:
            raise ValueError("Calibration layout ID differs from board")
        for action, expected in zip(actions, states[1:], strict=True):
            if isinstance(action, bool) or not isinstance(action, int) or action not in range(4):
                raise ValueError("Invalid primitive action")
            state = env.next_state(state, action)
            if to_rows(state) != expected:
                raise ValueError(f"Recorded transition mismatch in {row['record_id']}")
            count += 1
    return {"checked_records": len(rows), "replayed_primitive_actions": count}


def known_layouts() -> tuple[set[str], dict[str, str]]:
    """Exclude known prior data, snapshot observations and reserved layouts."""
    history = json.loads(HISTORY.read_text(encoding="utf-8"))
    if history.get("canonicalization") != "carl-layout-d4-v1":
        raise ValueError("Unknown history canonicalization")
    blocked = set(history["observed_layout_ids"]) | set(history["reserved_layout_ids"])
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in (HISTORY, FROZEN_SPLIT)}
    frozen = json.loads(FROZEN_SPLIT.read_text(encoding="utf-8"))
    blocked.update(item["layout_id"] for item in frozen["sources"])
    for relative in PRIOR_DATASETS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Prior dataset required for exposure audit: {path}")
        hashes[relative] = digest(path)
        blocked.update(row["layout_id"] for row in read_jsonl(path) if row.get("layout_id"))
    return blocked, hashes


def select_test(instances: list[dict[str, Any]], blocked: set[str], seed: int, count: int) -> list[dict[str, Any]]:
    """Select one real ×4 start state per distinct canonical layout."""
    candidates: dict[str, dict[str, Any]] = {}
    for item in instances:
        board, layout = item["state"], item["layout_id"]
        if (item.get("family") != "real_12x12x4" or len(board) != 12
                or any(len(row) != 12 for row in board)
                or sum(row.count("$") + row.count("*") for row in board) != 4
                or canonical_layout_id(from_rows(board)) != layout):
            raise ValueError("Invalid board or canonical layout in holdout")
        if layout in candidates:
            raise ValueError("Duplicate layout within holdout")
        if layout not in blocked:
            candidates[layout] = item
    ordered = sorted(candidates, key=lambda layout: selection_key(seed, "test-board", layout))
    if len(ordered) < count:
        raise ValueError("Not enough history-disjoint test boards")
    selected = []
    for layout in ordered[:count]:
        item = candidates[layout]
        selected.append({"record_id": item["id"], "context_id": item["id"], "layout_id": layout,
                         "split": "test", "family": item["family"], "start_observation": item["state"],
                         "task_goal": TASK_GOAL, "source_path": item["source_path"],
                         "source_sha256": item["source_sha256"], "source_index": item["source_index"]})
    return selected


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new frozen-data directory")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--train-contexts", type=int, default=8)
    parser.add_argument("--validation-contexts", type=int, default=3)
    parser.add_argument("--test-boards", type=int, default=12)
    args = parser.parse_args()
    if min(args.train_contexts, args.validation_contexts, args.test_boards) < 1:
        parser.error("All cohort sizes must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Will not replace frozen data: {output}")
    rows = read_jsonl(CALIBRATION)
    calibration, rejected = select_calibration(rows, args.seed, args.train_contexts, args.validation_contexts)
    excluded_contexts = {item["context_id"] for item in rejected}
    demonstrations = [row for row in rows if row["context_id"] not in excluded_contexts]
    if not {row["record_id"] for row in calibration} <= {row["record_id"] for row in demonstrations}:
        raise ValueError("Every calibrated candidate must be in the common demonstration pool")
    if {r["layout_id"] for r in demonstrations if r["split"] == "train"} & {
            r["layout_id"] for r in demonstrations if r["split"] == "validation"}:
        raise ValueError("Demonstration train and validation layouts overlap")
    audit = audit_fragments(rows)
    blocked, history_hashes = known_layouts()
    blocked.update(r["layout_id"] for r in rows)
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    test = select_test(holdout["instances"], blocked, args.seed, args.test_boards)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".freeze-followup-", dir=output.parent))
    try:
        save_jsonl(staging / "calibration_candidates.jsonl", calibration)
        save_jsonl(staging / "demo_annotations.jsonl", demonstrations)
        save_jsonl(staging / "test_contexts.jsonl", test)
        save_jsonl(staging / "rejected_contexts.jsonl", rejected)
        manifest = {
            "schema_version": 1,
            "seed": args.seed,
            "selection": "SHA-256 order on seed and canonical layout/context/record IDs; one context per board",
            "input_sha256": {str(CALIBRATION.relative_to(ROOT)): digest(CALIBRATION),
                             str(HOLDOUT.relative_to(ROOT)): digest(HOLDOUT), **history_hashes},
            "output_sha256": {p.name: digest(p) for p in sorted(staging.glob("*.jsonl"))},
            "calibration": {"records": len(calibration), "contexts": len({r['context_id'] for r in calibration}),
                            "boards_by_split": {split: len({r['layout_id'] for r in calibration if r['split'] == split})
                                                for split in ("train", "validation")},
                            "records_by_split": dict(Counter(r["split"] for r in calibration)),
                            "effect_types": dict(Counter(r["effect_type"] for r in calibration)),
                            "candidate_counts_per_context": 4},
            "demonstrations": {"records": len(demonstrations),
                               "contexts": len({r["context_id"] for r in demonstrations}),
                               "records_by_split": dict(Counter(r["split"] for r in demonstrations)),
                               "rule": "Full checked candidate pool minus whole semantically duplicated contexts."},
            "input_fragment_audit": audit,
            "excluded_semantic_duplicate_contexts": rejected,
            "test": {"boards": len(test), "family": "real_12x12x4", "layout_ids": [r["layout_id"] for r in test],
                     "excluded_known_layouts": len(blocked),
                     "provenance": "Public holdout1000, recorded as unused for inference/search in prior manifest."},
            "limitations": ["Known-source and snapshot non-overlap do not prove absence of all historical exposure.",
                            "The real ×4 test and synthetic ×10 calibration differ in source and box count.",
                            "Semantic deduplication covers predicate order and goal-count predicates implied by explicit boxes."],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"output": str(output), "calibration": manifest["calibration"], "test": manifest["test"],
                      "rejected_contexts": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
