#!/usr/bin/env python3
"""Freeze additional exploratory contexts on the existing Sokoban test boards.

Selection uses only the previously prepared data, never model predictions. It
adds no layouts and removes repeated generator inputs. Every selected fragment
is rechecked against a replay of its original trajectory.
"""
import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.intent_data import SCHEMA_VERSION
from carl.icl.intent_data import validate_intent
from scripts.prepare_intent_generator_data import ROOT
from scripts.prepare_intent_generator_data import digest
from scripts.prepare_intent_generator_data import load_data_sources
from scripts.prepare_intent_generator_data import prior_data
from scripts.prepare_intent_generator_data import replay
from scripts.prepare_intent_generator_data import write_json
from scripts.prepare_intent_generator_data import write_jsonl

Record = dict[str, Any]


def read_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def context_key(row: Record) -> tuple[str, tuple[str, ...], str]:
    """A generator context depends on its current board and task, not its label."""
    return row["layout_id"], tuple(row["start_observation"]), row["task_goal"]


def select_contexts(original: list[Record], candidates: list[Record]) -> list[Record]:
    """Use all distinct candidate contexts from the existing test layouts."""
    original_test = [row for row in original if row["split"] == "test"]
    allowed_layouts = {row["layout_id"] for row in original_test}
    if not allowed_layouts:
        raise ValueError("Original dataset has no test layouts")
    seen = {context_key(row) for row in original_test}
    selected = []
    for row in sorted((item for item in candidates if item["split"] == "test"), key=lambda item: item["record_id"]):
        if row["layout_id"] not in allowed_layouts:
            raise ValueError("Candidate test layout is not in the original evaluation")
        key = context_key(row)
        if key not in seen:
            seen.add(key)
            selected.append(row)
    return selected


def validate_fragment(row: Record, states: list[list[str]], actions: list[int]) -> list[str]:
    """Compare recorded observations, actions and predicates with replay."""
    i, k = row["i"], row["k"]
    if not isinstance(i, int) or isinstance(i, bool) or not isinstance(k, int) or isinstance(k, bool):
        return ["fragment index or offset is not an integer"]
    if i < 0 or k < 1 or i + k >= len(states):
        return ["fragment indices exceed the replayed trajectory"]
    errors = []
    if row["start_observation"] != states[i] or row["later_observation"] != states[i + k]:
        errors.append("recorded observation differs from replay")
    if row["fragment_actions"] != actions[i:i + k]:
        errors.append("recorded actions differ from replay")
    errors.extend(validate_intent(row["intent"], states[i:i + k + 1]))
    return errors


def validate_metadata(row: Record, source: Record) -> None:
    if row["schema_version"] != SCHEMA_VERSION or row["split"] != "test":
        raise ValueError("Unexpected schema or split")
    if row["annotation_provenance"].get("kind") != "local_exact_difference" or row["policy_measurements"] is not None:
        raise ValueError("Candidate provenance is not an unmeasured local annotation")
    for field, source_field in (("trajectory_id", "id"), ("layout_id", "layout_id"), ("source_path", "path"),
                                ("source_sha256", "sha256")):
        if row[field] != source[source_field]:
            raise ValueError(f"Candidate {field} differs from the frozen split")
    identity = f"{row['source_sha256']}:{row['i']}:{row['k']}:{json.dumps(row['intent'], sort_keys=True)}"
    if row["record_id"] != hashlib.sha256(identity.encode()).hexdigest()[:24]:
        raise ValueError("Candidate record ID does not match its content")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dataset", type=Path, required=True)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--frozen-contexts", type=Path, required=True)
    parser.add_argument("--frozen-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; use a new directory")

    original = read_jsonl(args.original_dataset)
    candidates = read_jsonl(args.candidate_dataset)
    frozen = json.loads(args.frozen_contexts.read_text(encoding="utf-8"))
    if frozen["dataset_sha256"] != digest(args.original_dataset):
        raise ValueError("Original dataset differs from the frozen evaluation")
    original_test = [row for row in original if row["split"] == "test"]
    original_ids = [row["record_id"] for row in original_test]
    frozen_ids = frozen["record_ids"]
    if (len(original_ids) != len(set(original_ids)) or len(frozen_ids) != len(set(frozen_ids))
            or set(original_ids) != set(frozen_ids)):
        raise ValueError("Original test records differ from the frozen evaluation")
    split = json.loads(args.frozen_split.read_text(encoding="utf-8"))
    test_sources = {row["id"]: row for row in split["sources"] if row["split"] == "test"}
    if {row["layout_id"] for row in original_test} != {row["layout_id"] for row in test_sources.values()}:
        raise ValueError("Original test layouts differ from the frozen split")
    selected = select_contexts(original, candidates)
    for row in selected:
        if row["trajectory_id"] not in test_sources:
            raise ValueError("Candidate trajectory is absent from the frozen test split")
        validate_metadata(row, test_sources[row["trajectory_id"]])

    sources = load_data_sources()
    prior_train, prior_pairs, prior_states = prior_data(sources)
    with tarfile.open(sources.prior_bundle, "r:gz") as archive:
        entry = archive.extractfile(sources.witness_archive_member)
        if entry is None:
            raise ValueError("Historical bundle has no documented trajectory actions")
        references = {row["id"]: row for row in json.load(entry)}
    replayed = {}
    for trajectory_id in sorted({row["trajectory_id"] for row in selected}):
        source = test_sources[trajectory_id]
        source_path = (ROOT / source["path"]).resolve()
        if not source_path.is_relative_to(ROOT) or digest(source_path) != source["sha256"]:
            raise ValueError("Source path or hash differs from the frozen split")
        if trajectory_id not in references:
            raise ValueError("Historical trajectory reference is missing")
        states, actions, _ = replay(source_path, prior_train.get(trajectory_id, []),
                                    prior_pairs.get(trajectory_id, []), prior_states, references[trajectory_id])
        if canonical_layout_id(from_rows(states[0])) != source["layout_id"]:
            raise ValueError("Replayed layout differs from the frozen split")
        replayed[trajectory_id] = states, actions

    accepted = []
    rejected = []
    for row in selected:
        errors = validate_fragment(row, *replayed[row["trajectory_id"]])
        if errors:
            rejected.append({"record_id": row["record_id"], "reasons": errors})
        else:
            accepted.append(row)

    args.output.mkdir(parents=True)
    dataset_path = args.output / "dataset.jsonl"
    write_jsonl(dataset_path, accepted)
    write_jsonl(args.output / "rejected.jsonl", rejected)
    write_json(args.output / "frozen_contexts.json", {
        "selection": "additional distinct contexts on the original test layouts; exploratory and fixed before inference",
        "dataset_sha256": digest(dataset_path),
        "record_ids": [row["record_id"] for row in accepted],
    })
    write_json(args.output / "manifest.json", {
        "selection_version": "additional-recorded-contexts-v1",
        "classification": "exploratory post-hoc extension; no new independent layouts",
        "original_dataset_sha256": digest(args.original_dataset),
        "candidate_dataset_sha256": digest(args.candidate_dataset),
        "frozen_contexts_sha256": digest(args.frozen_contexts),
        "frozen_split_sha256": digest(args.frozen_split),
        "dataset_sha256": digest(dataset_path),
        "selection_rule": "all distinct test inputs absent from original test; sort by record ID; no model outputs inspected",
        "replay": "real environment; source hashes, recorded endpoints, actions, effects and preservation rechecked",
        "original_test_records": len(original_test),
        "selected_candidates": len(selected),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "replayed_trajectories": len(replayed),
        "test_layouts": len({row["layout_id"] for row in accepted}),
        "records_per_layout": dict(sorted(Counter(row["layout_id"] for row in accepted).items())),
        "offsets": dict(sorted(Counter(row["k"] for row in accepted).items())),
        "effect_types": dict(sorted(Counter(row["effect_type"] for row in accepted).items())),
    })
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "boards": len(replayed)}))


if __name__ == "__main__":
    main()
