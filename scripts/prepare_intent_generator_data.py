#!/usr/bin/env python3
"""Freeze and audit the existing proposal Sokoban witnesses for intent learning.

Only local, documented JSON records are read; no pickle files or new levels are
created. The historical 12/50 split is preserved by canonical
wall/goal layout, including rotations and reflections. See the output manifest.
"""
import argparse
import collections
import hashlib
import json
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import SCHEMA_VERSION
from carl.icl.intent_data import EffectType
from carl.icl.intent_data import SokobanPredicate
from carl.icl.intent_data import validate_intent
from carl.icl.reachability_history import scan_history

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/icl/intent_generator_data.json"
OFFSETS = (1, 2, 4, 8, 16, 24)
ACTIONS = {"U": 0, "D": 1, "L": 2, "R": 3}
SEED = 20261114


@dataclass(frozen=True)
class DataSources:
    source_dir: Path
    prior_bundle: Path
    prior_pairs: Path
    prior_states: Path
    historical_labels: Path
    craftax_manifest: Path
    training_archive_member: str
    witness_archive_member: str
    annotated_trajectory_id: str


def load_data_sources(config_path: Path = DEFAULT_CONFIG, *, root: Path = ROOT) -> DataSources:
    """Resolve source paths relative to the repository, independent of cwd."""
    config = json.loads(config_path.read_text(encoding="utf-8"))

    def source_path(name: str) -> Path:
        value: str = config[name]
        return (root / value).resolve()

    return DataSources(
        source_dir=source_path("source_dir"),
        prior_bundle=source_path("prior_bundle"),
        prior_pairs=source_path("prior_pairs"),
        prior_states=source_path("prior_states"),
        historical_labels=source_path("historical_labels"),
        craftax_manifest=source_path("craftax_manifest"),
        training_archive_member=config["training_archive_member"],
        witness_archive_member=config["witness_archive_member"],
        annotated_trajectory_id=config["annotated_trajectory_id"],
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def prior_data(
    sources: DataSources | None = None
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    sources = sources or load_data_sources()
    with tarfile.open(sources.prior_bundle, "r:gz") as archive:
        entry = archive.extractfile(sources.training_archive_member)
        if entry is None:
            raise ValueError("Historical training bundle has no training.json")
        training = json.load(entry)["contexts"]
    train: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in training:
        train[item["id"].rsplit("-", 1)[0]].append(item)
    pairs: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in json.loads(sources.prior_pairs.read_text())["pairs"]:
        pairs[item["trajectory_id"]].append(item)
    states = json.loads(sources.prior_states.read_text())["states"]
    return train, pairs, states


def freeze_split(
        output: Path,
        sources: DataSources | None = None) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Fix independent layouts before inspecting fragments or annotations."""
    sources = sources or load_data_sources()
    train_prior, test_prior, _ = prior_data(sources)
    if set(train_prior) & set(test_prior):
        raise ValueError("Historical train/test trajectory IDs overlap")
    history = scan_history(ROOT)
    if history["errors"] or history["missing"]:
        raise ValueError(f"History audit incomplete: {history['errors']} {history['missing']}")
    protected = set()
    for item in history["sources"].values():
        path, kind = item["path"].lower(), item["kind"]
        if kind == "reserved_layout_ids" or any(
                part in path for part in ("holdout", "pilot", "/test.json", "cohort_checks")):
            protected.update(item["layout_ids"])
    available: dict[str, dict[str, Any]] = {}
    for path in sorted(sources.source_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        board = from_rows([line.replace(" ", "-") for line in raw["ascii"].splitlines()])
        layout = canonical_layout_id(board)
        if layout != raw["canonical_layout_id"]:
            raise ValueError(f"Source canonical layout mismatch: {path}")
        if board.shape != (12, 12, 7):
            raise ValueError(f"Wrong source shape: {path}")
        rows = to_rows(board)
        if sum(row.count("$") + row.count("*") for row in rows) != 10:
            raise ValueError(f"Wrong source box count: {path}")
        if sum(row.count(".") + row.count("*") + row.count("+") for row in rows) != 10:
            raise ValueError(f"Wrong source goal count: {path}")
        available[raw["id"]] = {
            "id":
                raw["id"],
            "path":
                str(path.relative_to(ROOT)),
            "sha256":
                digest(path),
            "layout_id":
                layout,
            "historical_cohort":
                "train" if raw["id"] in train_prior else "test" if raw["id"] in test_prior else "unknown"
        }
    if len(available) != len({x["layout_id"] for x in available.values()}):
        raise ValueError("Source contains duplicate or symmetry-related layouts")
    old_train = sorted(k for k in train_prior if k in available and available[k]["layout_id"] not in protected)
    old_test = sorted(k for k in test_prior if k in available)
    if len(old_train) < 12 or len(old_test) < 20 or sources.annotated_trajectory_id not in old_train:
        raise ValueError(f"Insufficient eligible historical splits: train={len(old_train)} test={len(old_test)}")
    rng = random.Random(SEED)
    old_train.remove(sources.annotated_trajectory_id)
    rng.shuffle(old_train)
    rng.shuffle(old_test)
    # Previously annotated trajectory is fixed in train, never validation/test.
    chosen = {sources.annotated_trajectory_id: "train"}
    chosen.update({identifier: "train" for identifier in old_train[:8]})
    chosen.update({identifier: "validation" for identifier in old_train[8:11]})
    chosen.update({identifier: "test" for identifier in old_test[:20]})
    layouts = {
        split: {available[k]["layout_id"] for k, v in chosen.items() if v == split}
        for split in ("train", "validation", "test")
    }
    if any(layouts[a] & layouts[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Canonical layout or symmetry overlap")
    if (layouts["train"] | layouts["validation"]) & protected:
        raise ValueError("Training or validation consumes a previously protected cohort")
    split = {
        "seed":
            SEED,
        "split_unit":
            "canonical wall/goal layout under D4 symmetries",
        "reason":
            "Historical proposal set contains 12 training and 50 held-out layouts; protect all previous tests, then subdivide prior train 9/3 and sample 20 of prior test.",
        "sources": [dict(available[k], split=chosen[k]) for k in sorted(chosen)],
        "history_audit": {
            "canonicalization": history["canonicalization"],
            "sources_scanned": len(history["sources"]),
            "missing": history["missing"],
            "errors": history["errors"],
            "protected_layouts": len(protected),
            "limitations": history["limitations"]
        },
        "previous_test_pairs_sha256":
            digest(sources.prior_pairs),
        "previous_training_bundle_sha256":
            digest(sources.prior_bundle)
    }
    write_json(output / "frozen_split.json", split)
    return chosen, available, split


def replay(path: Path,
           training: list[dict[str, Any]],
           pairs: list[dict[str, Any]],
           old_states: dict[str, list[str]],
           expected_witness: dict[str, Any] | None = None) -> tuple[list[list[str]], list[int], dict[str, Any]]:
    raw = json.loads(path.read_text())
    state = from_rows([line.replace(" ", "-") for line in raw["ascii"].splitlines()])
    env = environment(state)
    source_actions = [ACTIONS[c] for c in raw["actions"]]
    actions = source_actions
    if expected_witness is not None:
        actions = expected_witness["actions"]
        if (expected_witness["id"] != raw["id"] or expected_witness["initial_state"] != to_rows(state)
                or source_actions[:len(actions)] != actions):
            raise ValueError("historically documented witness is not an exact source prefix")
    states = [to_rows(state)]
    pushes = 0
    for i, action in enumerate(actions):
        if env.is_solved(state):
            raise ValueError(f"action {i} after solved state")
        nxt = np.asarray(env.next_state(state, action))
        validate_subgoal(state, nxt)
        if np.array_equal(state, nxt):
            raise ValueError(f"action {i} is a no-op")
        pushes += int(
            not np.array_equal(np.isin(np.argmax(state, axis=-1), [3, 4]), np.isin(np.argmax(nxt, axis=-1), [3, 4])))
        state = nxt
        states.append(to_rows(state))
    if not env.is_solved(state):
        raise ValueError("source witness does not end solved")
    checked = 0
    for context in training:
        i = int(context["id"].rsplit("-", 1)[1])
        if states[i] != context["state"]:
            raise ValueError(f"historically recorded training state mismatch at {i}")
        checked += 1
        for target in context["candidates"]:
            k = target["k"]
            if states[i + k] != target["subgoal"] or actions[i:i + k] != target["actions"]:
                raise ValueError(f"historically recorded training target/actions mismatch at {i}+{k}")
            checked += 1
    for pair in pairs:
        i, k = pair["i"], pair["k"]
        if states[i] != old_states[pair["source_hash"]] or states[i + k] != old_states[pair["target_hash"]]:
            raise ValueError(f"historically recorded test endpoints mismatch at {i}+{k}")
        checked += 2
    return states, actions, {
        "trajectory_id":
            raw["id"],
        "format":
            "JSON ASCII initial board plus UDLR action string",
        "board_dimensions": [len(states[0]), len(states[0][0])],
        "box_count":
            sum(row.count("$") + row.count("*") for row in states[0]),
        "goal_count":
            sum(row.count(".") + row.count("*") + row.count("+") for row in states[0]),
        "primitive_actions":
            len(actions),
        "raw_source_actions":
            len(source_actions),
        "discarded_post_terminal_actions":
            len(source_actions) - len(actions),
        "discarded_suffix_reason":
            "after first solved state; excluded, as in documented witness"
            if len(source_actions) != len(actions) else None,
        "push_actions":
            pushes,
        "walking_actions":
            len(actions) - pushes,
        "no_ops":
            0,
        "solved":
            True,
        "historically_recorded_states_checked":
            checked
    }


def box_cells(rows: list[str]) -> set[tuple[int, int]]:
    return {(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in "$*"}


def candidate(start: list[str], end: list[str], chooser: int) -> tuple[str, dict[str, Any]] | None:
    new = sorted(box_cells(end) - box_cells(start))
    before, after = sum(r.count("*") for r in start), sum(r.count("*") for r in end)
    if len(new) >= 2 and chooser % 3 == 0:
        effects = [{"predicate": SokobanPredicate.BOX_AT.value, "row": r, "col": c} for r, c in new[:2]]
        kind = EffectType.BOX_CONJUNCTION.value
    elif after > before and new and chooser % 2 == 0:
        goals = [(r, c) for r, c in new if end[r][c] == "*"]
        r, c = (goals or new)[0]
        effects = [{
            "predicate": SokobanPredicate.BOX_AT.value,
            "row": r,
            "col": c
        }, {
            "predicate": SokobanPredicate.BOXES_ON_GOALS_AT_LEAST.value,
            "count": after
        }]
        kind = EffectType.GOAL_CONJUNCTION.value
    elif new:
        r, c = new[chooser % len(new)]
        effects = [{"predicate": SokobanPredicate.BOX_AT.value, "row": r, "col": c}]
        kind = EffectType.BOX.value
    elif start != end:
        r, c = next((r, c) for r, row in enumerate(end) for c, ch in enumerate(row) if ch in "@+")
        effects = [{"predicate": SokobanPredicate.PLAYER_AT.value, "row": r, "col": c}]
        kind = EffectType.WALK_ONLY.value
    else:
        return None
    return kind, {"effects": effects, "preserve": [], "instruction": None}


def sample(states: list[list[str]], cap: int, seed: int) -> list[tuple[int, int, str, dict[str, Any]]]:
    rng = random.Random(seed)
    groups: dict[str, dict[int,
                           list[tuple[int, int, str,
                                      dict[str,
                                           Any]]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for k in OFFSETS:
        for i in range(len(states) - k):
            result = candidate(states[i], states[i + k], i + k)
            if result:
                kind, intent = result
                groups[kind][k].append((i, k, kind, intent))
    for g in groups.values():
        for items in g.values():
            rng.shuffle(items)
    limits = {
        EffectType.BOX.value: int(cap * 0.55),
        EffectType.GOAL_CONJUNCTION.value: int(cap * 0.20),
        EffectType.BOX_CONJUNCTION.value: int(cap * 0.15),
        EffectType.WALK_ONLY.value: max(1, int(cap * 0.10))
    }
    chosen: list[tuple[int, int, str, dict[str, Any]]] = []
    seen = set()
    # Rotate across categories and boards, not from k=1 at every category.
    # This decision depends only on the fixed data seed, never model outcomes.
    offset_cursor = seed % len(OFFSETS)
    # Balanced by effect type and offsets; walking remains at most 10%.
    for kind in (EffectType.GOAL_CONJUNCTION.value, EffectType.BOX_CONJUNCTION.value, EffectType.BOX.value,
                 EffectType.WALK_ONLY.value):
        for _ in range(limits[kind]):
            for delta in range(len(OFFSETS)):
                k = OFFSETS[(offset_cursor + delta) % len(OFFSETS)]
                bucket = groups[kind][k]
                while bucket:
                    item = bucket.pop()
                    key = (tuple(states[item[0]]), json.dumps(item[3], sort_keys=True))
                    if key not in seen:
                        seen.add(key)
                        chosen.append(item)
                        offset_cursor = (offset_cursor + delta + 1) % len(OFFSETS)
                        break
                else:
                    continue
                break
    rng.shuffle(chosen)
    return chosen


def record(available: dict[str, Any], split: str, states: list[list[str]], actions: list[int], i: int, k: int,
           kind: str, intent: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    errors = validate_intent(intent, states[i:i + k + 1])
    if errors:
        raise ValueError("; ".join(errors))
    identity = f"{available['sha256']}:{i}:{k}:{json.dumps(intent, sort_keys=True)}"
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
        "split": split,
        "source_path": available["path"],
        "source_sha256": available["sha256"],
        "layout_id": available["layout_id"],
        "trajectory_id": available["id"],
        "i": i,
        "k": k,
        "start_observation": states[i],
        "later_observation": states[i + k],
        "fragment_actions": actions[i:i + k],
        "task_goal": "Put every box on a goal.",
        "intent": intent,
        "effect_type": kind,
        "annotation_provenance": provenance,
        "validation": {
            "accepted": True,
            "each_effect_false_initially_true_finally": True,
            "preservation_checked_at_every_recorded_state": True,
            "rejection_reasons": []
        },
        "policy_measurements": None
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = load_data_sources(args.config)
    chosen, available, split_meta = freeze_split(out, sources)
    train_prior, pairs_prior, old_states = prior_data(sources)
    with tarfile.open(sources.prior_bundle, "r:gz") as archive:
        entry = archive.extractfile(sources.witness_archive_member)
        if entry is None:
            raise ValueError("Historical bundle has no documented witnesses")
        witnesses = {x["id"]: x for x in json.load(entry)}
    previous = [json.loads(line) for line in sources.historical_labels.read_text().splitlines() if line.strip()]
    history_pool = [
        x for x in previous if any(
            p.get("predicate") in {SokobanPredicate.BOX_AT, SokobanPredicate.BOXES_ON_GOALS_AT_LEAST}
            for p in x.get("intent", {}).get("effects", []))
    ]
    rng = random.Random(SEED + 1)
    history_selection = rng.sample(history_pool, min(15, len(history_pool)))
    write_json(
        out / "historical_annotation_selection.json", {
            "seed":
                SEED + 1,
            "ids": [x["id"] for x in history_selection],
            "source":
                str(sources.historical_labels),
            "sha256":
                digest(sources.historical_labels),
            "selected_before_validation":
                True,
            "sampling_frame":
                "historical labels with at least one box/goal effect; no selection by this experiment's validation or model outcome",
            "eligible_records":
                len(history_pool)
        })
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for n, (identifier, split) in enumerate(sorted(chosen.items())):
        info = available[identifier]
        try:
            states, actions, audit = replay(ROOT / info["path"], train_prior.get(identifier, []),
                                            pairs_prior.get(identifier, []), old_states, witnesses[identifier])
        except (OSError, ValueError, IndexError, KeyError) as exc:
            rejected.append({
                "trajectory_id": identifier,
                "split": split,
                "source_path": info["path"],
                "reason": str(exc)
            })
            continue
        audit.update({
            "split": split,
            "layout_id": info["layout_id"],
            "source_path": info["path"],
            "source_sha256": info["sha256"]
        })
        audits.append(audit)
        if audit["discarded_post_terminal_actions"]:
            rejected.append({
                "trajectory_id": identifier,
                "split": split,
                "source_path": info["path"],
                "type": "discarded_post_terminal_suffix",
                "discarded_actions": audit["discarded_post_terminal_actions"],
                "reason": audit["discarded_suffix_reason"]
            })
        cap = 25 if identifier == sources.annotated_trajectory_id else 40 if split == "train" else 24 if split == "validation" else 6
        for i, k, kind, intent in sample(states, cap, SEED + n):
            try:
                records.append(
                    record(info, split, states, actions, i, k, kind, intent, {
                        "kind": "local_exact_difference",
                        "version": "local-difference-v1",
                        "model": None
                    }))
            except ValueError as exc:
                rejected.append({"trajectory_id": identifier, "i": i, "k": k, "reason": str(exc)})
        if identifier == sources.annotated_trajectory_id:
            for old in history_selection:
                i, k = old["i"], old["k"]
                intent = {
                    "effects": old["intent"]["effects"],
                    "preserve": old["intent"]["preserve"],
                    "instruction": None
                }
                try:
                    if states[i] != old["start_state"]["ascii"] or states[i + k] != old["end_state"]["ascii"]:
                        raise ValueError("historical annotation endpoints disagree with replay")
                    predicates = [p.get("predicate") for p in intent["effects"]]
                    kind = (EffectType.GOAL_CONJUNCTION.value if SokobanPredicate.BOXES_ON_GOALS_AT_LEAST in predicates
                            else EffectType.BOX_CONJUNCTION.value if len(predicates) > 1 else EffectType.BOX.value)
                    records.append(
                        record(
                            info, split, states, actions, i, k, kind, intent, {
                                "kind": "historical_shared_context_codex_annotation",
                                "model": "gpt-6.1-terra",
                                "fresh_isolated_query": False,
                                "source": str(sources.historical_labels),
                                "source_sha256": digest(sources.historical_labels),
                                "record_id": old["id"],
                                "original_instruction": old["intent"]["text"],
                                "instruction_not_used_for_training": True
                            }))
                except (ValueError, IndexError, KeyError) as exc:
                    rejected.append({"trajectory_id": identifier, "historical_id": old["id"], "reason": str(exc)})
        print(json.dumps({
            "status": "replayed",
            "trajectory": identifier,
            "split": split,
            "records": len(records)
        }),
              flush=True)
    write_jsonl(out / "dataset.jsonl", records)
    write_jsonl(out / "rejected.jsonl", rejected)
    test_ids = [r["record_id"] for r in records if r["split"] == "test"]
    write_json(out / "frozen_test_ids.json", {
        "record_ids": test_ids,
        "freeze": "before model training",
        "count": len(test_ids)
    })
    counts = {
        s: {
            "records": sum(r["split"] == s for r in records),
            "boards": len({r["layout_id"] for r in records if r["split"] == s}),
            "effect_types": dict(collections.Counter(r["effect_type"] for r in records if r["split"] == s))
        } for s in ("train", "validation", "test")
    }
    manifest = {
        "schema_version":
            SCHEMA_VERSION,
        "seed":
            SEED,
        "family":
            "proposal_gamma2_12x12x10",
        "classification":
            "existing synthetic generated proposal trajectories; structurally selected, exploratory, not an independent natural benchmark",
        "coordinates":
            "zero-based row, column; origin top left",
        "symbols":
            "# wall, - floor, . goal, * box on goal, $ box, @ player, + player on goal",
        "action_convention":
            "0 UP, 1 DOWN, 2 LEFT, 3 RIGHT; one primitive action; pushes separately counted",
        "fragment_offsets":
            list(OFFSETS),
        "historical_annotation_offsets_may_also_include": [12, 32],
        "conjunction":
            "all effects false initially and all true finally; preserve holds at every recorded state",
        "counts":
            counts,
        "rejected_records":
            len(rejected),
        "trajectories":
            audits,
        "suffix_handling":
            "Documented historical witness must match the raw source action prefix and first terminal; "
            "any remaining source actions are excluded and counted separately in rejected.jsonl.",
        "history":
            split_meta["history_audit"],
        "frozen_split_sha256":
            digest(out / "frozen_split.json"),
        "frozen_test_ids_sha256":
            digest(out / "frozen_test_ids.json"),
        "dataset_sha256":
            digest(out / "dataset.jsonl"),
        "previous_sokoban_annotations": {
            "source": str(sources.historical_labels),
            "sha256": digest(sources.historical_labels),
            "source_records": len(previous),
            "trajectory_id": sources.annotated_trajectory_id,
            "independent_layouts": 1,
            "fresh_isolated_zero_shot": False,
            "note": "Previously authored in a shared conversation; preselected subset checked again, only in train; "
                    "original English text retained in provenance but not trained."
        },
        "previous_craftax_annotations": {
            "manifest": str(sources.craftax_manifest),
            "sha256": digest(sources.craftax_manifest),
            "use": "excluded: different domain and observation/predicate semantics"
        },
        "limitation":
            "Prior model exposure cannot be ruled out; test boards were previously selected/evaluated for an exploratory policy study. "
            "The split is held out from this adaptation, not a virgin benchmark."
    }
    write_json(out / "dataset_manifest.json", manifest)
    if len(audits) != len(chosen):
        raise SystemExit("Some selected trajectories were rejected; see rejected.jsonl and dataset_manifest.json")
    print(json.dumps({"status": "complete", "counts": counts, "rejected": len(rejected)}), flush=True)


if __name__ == "__main__":
    main()
