#!/usr/bin/env python3
"""Prepare checked, multi-intent contexts for a future Luna/LoRA study.

Only the previously frozen 12x12x10 training and validation layouts are
used. This command replays local trajectories and does not call a model.
"""

import argparse
import json
import random
import tarfile
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.intent_data import EffectType
from carl.icl.intent_data import validate_intent
from carl.icl.intent_generator_experiment import DataSplit
from scripts.prepare_intent_generator_data import OFFSETS
from scripts.prepare_intent_generator_data import ROOT
from scripts.prepare_intent_generator_data import candidate
from scripts.prepare_intent_generator_data import digest
from scripts.prepare_intent_generator_data import load_data_sources
from scripts.prepare_intent_generator_data import record
from scripts.prepare_intent_generator_data import replay
from scripts.prepare_intent_generator_data import write_json
from scripts.prepare_intent_generator_data import write_jsonl
from scripts.prepare_luna_intent_data import Family
from scripts.prepare_luna_intent_data import Phase
from scripts.prepare_luna_intent_data import identity

FROZEN_DIR = ROOT / "analysis/intent_generator_20261114T032232Z"
FROZEN_SPLIT = FROZEN_DIR / "frozen_split.json"
PREVIOUS_MANIFEST = FROZEN_DIR / "dataset_manifest.json"
TASK_GOAL = "Put every box on a goal."
OUTPUT_FILES = ("calibration_candidates.jsonl", "rejected.jsonl", "manifest.json")


class PreparationStatus(StrEnum):
    """Whether every planned context has all of its checked candidates."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class Candidate:
    """One distinct intent backed by a recorded trajectory fragment."""

    offset: int
    kind: str
    intent: dict[str, Any]


def validate_split(sources: list[dict[str, Any]], expected: tuple[int, int] = (9, 3)) -> None:
    """Reject duplicates and any use of a historical test layout for calibration."""
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_layouts: set[str] = set()
    for source in sources:
        split = DataSplit(source["split"])
        if source["id"] in seen_ids or source["layout_id"] in seen_layouts:
            raise ValueError("Frozen split contains a repeated trajectory or canonical layout")
        seen_ids.add(source["id"])
        seen_layouts.add(source["layout_id"])
        counts[split] += 1
        if split != DataSplit.TEST and source["historical_cohort"] != DataSplit.TRAIN:
            raise ValueError("Calibration source is not from the historical training cohort")
    if (counts[DataSplit.TRAIN], counts[DataSplit.VALIDATION]) != expected:
        raise ValueError("Frozen split does not contain the expected training and validation boards")


def candidate_pool(states: list[list[str]], index: int) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Derive distinct non-walking intents, checking each complete fragment."""
    pool: list[Candidate] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for offset in OFFSETS:
        if index + offset >= len(states):
            continue
        # The existing local annotator deterministically exposes different box
        # and conjunction choices for the same pair of recorded states.
        for chooser in range(12):
            proposed = candidate(states[index], states[index + offset], chooser)
            if proposed is None or proposed[0] == EffectType.WALK_ONLY:
                continue
            kind, intent = proposed
            signature = json.dumps(intent["effects"], sort_keys=True, separators=(",", ":"))
            if signature in seen:
                continue
            seen.add(signature)
            errors = validate_intent(intent, states[index:index + offset + 1])
            if errors:
                rejected.append({"i": index, "k": offset, "intent": intent, "reasons": errors})
            else:
                pool.append(Candidate(offset, kind, intent))
    return pool, rejected


def select_candidates(pool: list[Candidate], count: int, rng: random.Random) -> list[Candidate]:
    """Prefer a mix of goal, box-conjunction and single-box intents."""
    order = (EffectType.GOAL_CONJUNCTION, EffectType.BOX_CONJUNCTION, EffectType.BOX)
    buckets = {kind: [item for item in pool if item.kind == kind] for kind in order}
    for items in buckets.values():
        rng.shuffle(items)
    selected: list[Candidate] = []
    while len(selected) < count:
        old_length = len(selected)
        for kind in order:
            if buckets[kind] and len(selected) < count:
                selected.append(buckets[kind].pop())
        if len(selected) == old_length:
            break
    return selected


def prepare_board(source: dict[str, Any], states: list[list[str]], actions: list[int], contexts_per_board: int,
                  candidates_per_context: int,
                  seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Choose contexts without seeing a policy's responses or success rates."""
    if len(states) != len(actions) + 1:
        raise ValueError("A replay must contain one more state than actions")
    rng = random.Random(f"{seed}:{source['layout_id']}")
    indices = list(range(len(actions)))
    rng.shuffle(indices)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    selected_states: set[tuple[str, ...]] = set()
    stats = Counter({"examined": 0, "insufficient_candidates": 0, "duplicate_start": 0})
    selected_contexts = 0
    for index in indices:
        if selected_contexts == contexts_per_board:
            break
        stats["examined"] += 1
        start = tuple(states[index])
        if start in selected_states:
            stats["duplicate_start"] += 1
            rejected.append({
                "trajectory_id": source["id"],
                "split": source["split"],
                "i": index,
                "reason": "duplicate selected start state"
            })
            continue
        pool, invalid = candidate_pool(states, index)
        for item in invalid:
            rejected.append({"trajectory_id": source["id"], "split": source["split"], **item})
        options = select_candidates(pool, candidates_per_context, rng)
        if len(options) < candidates_per_context:
            stats["insufficient_candidates"] += 1
            rejected.append({
                "trajectory_id": source["id"],
                "split": source["split"],
                "i": index,
                "reason": "too few distinct non-walking intents",
                "available": len(options),
                "required": candidates_per_context
            })
            continue
        selected_states.add(start)
        selected_contexts += 1
        context_id = identity([Family.SYNTHETIC.value, source["layout_id"], index, states[index]])
        for option in options:
            item = record(
                source, source["split"], states, actions, index, option.offset, option.kind, option.intent, {
                    "kind": "local_exact_difference",
                    "version": "local-difference-v1",
                    "model": None,
                    "uses_recorded_future": True
                })
            item.update({
                "record_id": identity([context_id, option.offset, option.intent]),
                "context_id": context_id,
                "phase": Phase.CALIBRATION.value,
                "family": Family.SYNTHETIC.value,
                "task_goal": TASK_GOAL,
                "fragment_observations": states[index:index + option.offset + 1]
            })
            item["validation"]["environment_replay"] = True
            records.append(item)
    stats["selected"] = selected_contexts
    if selected_contexts < contexts_per_board:
        rejected.append({
            "trajectory_id": source["id"],
            "split": source["split"],
            "reason": "insufficient qualifying contexts",
            "available": selected_contexts,
            "required": contexts_per_board
        })
    return records, rejected, dict(stats)


def write_frozen(output: Path, records: list[dict[str, Any]], rejected: list[dict[str, Any]],
                 manifest: dict[str, Any]) -> None:
    """Write a new cohort, allowing only byte-identical reproduction."""
    contents = (
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records),
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rejected),
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, content in zip(OUTPUT_FILES, contents, strict=True):
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace different frozen data: {path}")
    if not (output / OUTPUT_FILES[0]).exists():
        write_jsonl(output / OUTPUT_FILES[0], records)
    if not (output / OUTPUT_FILES[1]).exists():
        write_jsonl(output / OUTPUT_FILES[1], rejected)
    if not (output / OUTPUT_FILES[2]).exists():
        write_json(output / OUTPUT_FILES[2], manifest)


def prepare(output: Path,
            contexts_per_board: int = 2,
            candidates_per_context: int = 4,
            seed: int = 20261114) -> dict[str, Any]:
    if contexts_per_board < 1 or candidates_per_context < 2:
        raise ValueError("Request at least one context and two candidates")
    frozen = json.loads(FROZEN_SPLIT.read_text(encoding="utf-8"))
    previous = json.loads(PREVIOUS_MANIFEST.read_text(encoding="utf-8"))
    if digest(FROZEN_SPLIT) != previous["frozen_split_sha256"]:
        raise ValueError("Frozen split does not match the previous dataset manifest")
    validate_split(frozen["sources"])
    sources = sorted((row for row in frozen["sources"] if row["split"] in {DataSplit.TRAIN, DataSplit.VALIDATION}),
                     key=lambda row: row["id"])
    data_sources = load_data_sources()
    if digest(data_sources.prior_bundle) != frozen["previous_training_bundle_sha256"]:
        raise ValueError("Historical trajectory bundle hash differs from the frozen split")
    ids = {row["id"] for row in sources}
    with tarfile.open(data_sources.prior_bundle, "r:gz") as archive:
        entry = archive.extractfile(data_sources.witness_archive_member)
        if entry is None:
            raise ValueError("Historical bundle has no trajectory records")
        selected = [row for row in json.load(entry) if row["id"] in ids]
    witnesses = {row["id"]: row for row in selected}
    if len(selected) != len(ids) or set(witnesses) != ids:
        raise ValueError("Historical bundle is missing or duplicates a selected trajectory")

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    audits = []
    for source in sources:
        try:
            path = (ROOT / source["path"]).resolve()
            if not path.is_relative_to(ROOT.resolve()) or digest(path) != source["sha256"]:
                raise ValueError("Source path or hash differs from the frozen split")
            witness = witnesses[source["id"]]
            if witness["layout_id"] != source["layout_id"] or witness["split"] != "training":
                raise ValueError("Historical trajectory provenance disagrees with the frozen split")
            states, actions, audit = replay(path, [], [], {}, witness)
            initial = from_rows(states[0])
            if (initial.shape != (12, 12, 7) or canonical_layout_id(initial) != source["layout_id"]
                    or audit["box_count"] != 10 or audit["goal_count"] != 10):
                raise ValueError("Source board does not match its frozen 12x12x10 layout")
            chosen, failures, stats = prepare_board(source, states, actions, contexts_per_board, candidates_per_context,
                                                    seed)
            records.extend(chosen)
            rejected.extend(failures)
            audits.append({
                "trajectory_id": source["id"],
                "split": source["split"],
                "layout_id": source["layout_id"],
                "source_path": source["path"],
                "source_sha256": source["sha256"],
                "replay": audit,
                "contexts": stats
            })
        except (OSError, ValueError, IndexError, KeyError) as exc:
            rejected.append({
                "trajectory_id": source["id"],
                "split": source["split"],
                "source_path": source["path"],
                "reason": str(exc),
                "type": "source_error"
            })

    counts = {}
    for split in (DataSplit.TRAIN, DataSplit.VALIDATION):
        rows = [row for row in records if row["split"] == split]
        counts[split] = {
            "boards": len({row["layout_id"] for row in rows}),
            "contexts": len({row["context_id"] for row in rows}),
            "records": len(rows),
            "effect_types": dict(Counter(row["effect_type"] for row in rows))
        }
    expected_records = len(sources) * contexts_per_board * candidates_per_context
    manifest = {
        "schema_version":
            1,
        "status":
            PreparationStatus.COMPLETE if len(records) == expected_records else PreparationStatus.INCOMPLETE,
        "seed":
            seed,
        "requested_contexts_per_board":
            contexts_per_board,
        "requested_candidates_per_context":
            candidates_per_context,
        "family":
            Family.SYNTHETIC.value,
        "frozen_split_path":
            str(FROZEN_SPLIT.relative_to(ROOT)),
        "frozen_split_sha256":
            digest(FROZEN_SPLIT),
        "historical_bundle_sha256":
            digest(data_sources.prior_bundle),
        "offsets":
            list(OFFSETS),
        "selection":
            "Seeded shuffled start indices; require distinct starts and non-walking intents; "
            "prefer different intent types. No model outputs or policy results are used.",
        "validation":
            "Exact replay; each intent is false initially and true at the fragment end; "
            "preservation is checked on every state. Coordinates are zero-based row, column.",
        "counts":
            counts,
        "boards":
            audits,
        "rejected_records":
            len(rejected),
        "contexts_with_too_few_candidates":
            sum(row["contexts"]["insufficient_candidates"] for row in audits),
        "policy_measurements":
            None,
        "limitations": [
            "Recorded future states and actions are for labelling and audit, never generator input.",
            "No policy trials or relative-scale labels have been collected.",
            "Layouts are synthetic and previously exposed, not a fresh held-out benchmark."
        ],
    }
    write_frozen(output, records, rejected, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("prepare", help="Prepare an immutable calibration-candidate cohort")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--contexts-per-board", type=int, default=2)
    command.add_argument("--candidates-per-context", type=int, default=4)
    command.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    manifest = prepare(args.output, args.contexts_per_board, args.candidates_per_context, args.seed)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "counts": manifest["counts"],
                "contexts_with_too_few_candidates": manifest["contexts_with_too_few_candidates"]
            },
            indent=2))
    if manifest["status"] != PreparationStatus.COMPLETE:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
