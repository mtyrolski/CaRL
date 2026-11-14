#!/usr/bin/env python3
"""Freeze independently grouped Sokoban evaluation contexts for representation studies.

Only pre-existing, trusted local puzzle and solution files are read. Test
boards are never assigned to the training or validation split. The previous
20+20 paired cases keep their exact IDs and precede additional boards.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import sqlite3
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.campaign_data import verify_record
from carl.icl.data_sources.sokoban import verify_solution
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records
from scripts.prepare_lora_transfer_data import replay
from scripts.prepare_luna_intent_data import TEST_TARGET_OFFSETS
from scripts.prepare_luna_intent_data import Family
from scripts.prepare_luna_intent_data import Phase
from scripts.prepare_luna_intent_data import make_records

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "analysis/intent_generator_20261114T032232Z/dataset.jsonl"
PREVIOUS = ROOT / "analysis/lora_matrix_20261114T065804Z/eval_transfer"
CAMPAIGN = ROOT / "data/icl/campaign_dataset_12x12_v1"
PRIVATE = ROOT / "data/icl/campaign_dataset_12x12_v1_private/pilot100/real_12x12x4.json"
GENERATED = ROOT / "data/icl/generated_12x12x10/test.json"
GENERATED_ACTIONS = ROOT / "data/icl/private_witnesses/generated_12x12x10/test.json"
HISTORY = ROOT / "analysis/luna_icl/campaign_100/history_snapshot.json"
INDEX = ROOT / "data/icl/.private_campaign_shards.sqlite"


class EvaluationFamily(StrEnum):
    """Do not conflate the distinct ×10 generators."""

    REAL = "real_12x12x4"
    PROPOSAL = "proposal_gamma2_12x12x10"
    GENERATED = "generated_v1_12x12x10"


def layout(row: dict[str, Any]) -> str:
    """Canonical ID based on dimensions, walls and goal tiles."""
    return canonical_layout_id(from_rows(row["state"]))


def validate_groups(rows: list[dict[str, Any]]) -> None:
    """Reject duplicate IDs and any train/validation versus test layout overlap."""
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate record ID")
    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row["split"], set()).add(row["layout_id"])
    splits = sorted(by_split)
    for index, first in enumerate(splits):
        for second in splits[index + 1:]:
            if by_split[first] & by_split[second]:
                raise ValueError(f"Overlapping layouts between {first} and {second}")
    test = [row for row in rows if row["split"] == "test"]
    if len({row["layout_id"] for row in test}) != len(test):
        raise ValueError("Expected one test context per layout")


def frozen_order(test_rows: list[dict[str, Any]], previous_ids: list[str]) -> list[str]:
    """Keep the previous paired 40 first; interleave newly added families next."""
    by_id = {row["record_id"]: row for row in test_rows}
    if len(previous_ids) != len(set(previous_ids)) or any(i not in by_id for i in previous_ids):
        raise ValueError("Previous IDs must be unique and present")
    extra = [row for row in test_rows if row["record_id"] not in set(previous_ids)]
    queues = {family: sorted((row for row in extra if row["evaluation_family"] == family),
                             key=lambda row: (row["layout_id"], row["record_id"]))
              for family in EvaluationFamily}
    if sum(map(len, queues.values())) != len(extra):
        raise ValueError("Unexpected evaluation family")
    result = list(previous_ids)
    while any(queues.values()):
        for queue in queues.values():
            if queue:
                result.append(queue.pop(0)["record_id"])
    return result


def checked_solution(board: list[str], actions: list[int]) -> list[list[str]]:
    """Replay all primitive actions in the actual environment through terminal."""
    if any(not isinstance(action, int) or isinstance(action, bool) or action not in range(4) for action in actions):
        raise ValueError("Solution contains an invalid primitive action")
    with contextlib.redirect_stdout(io.StringIO()):
        return [to_rows(state) for state in verify_solution(from_rows(board), actions)]


def campaign_inputs() -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, Any]]:
    public = CAMPAIGN / "real_12x12x4/pilot100.json"
    manifest = CAMPAIGN / "campaign_dataset_manifest.json"
    expected = json.loads(manifest.read_text())["manifest_sha256"]["real_12x12x4/pilot100.json"]
    if file_sha256(public) != expected:
        raise ValueError("Campaign public data differs from its manifest")
    boards = json.loads(public.read_text())["instances"]
    solutions = json.loads(PRIVATE.read_text())["witnesses"]
    actions = {row["board_id"]: row["actions"] for row in solutions}
    if len(boards) != 100 or len(actions) != len(boards) or set(actions) != {r["board_id"] for r in boards}:
        raise ValueError("Expected 100 campaign boards with exactly one solution each")
    hashes: dict[str, str] = {}
    with sqlite3.connect(f"file:{INDEX}?mode=ro", uri=True) as db:
        for row in boards:
            verify_record(row, EvaluationFamily.REAL, "pilot100")
            actual = db.execute(
                "SELECT boards.state_hash, boards.layout_id, shards.sha256 FROM boards "
                "JOIN shards ON boards.path=shards.path WHERE boards.path=? AND boards.trajectory_index=?",
                (row["source_path"], row["source_index"]),
            ).fetchone()
            if actual != (row["state_hash"], row["layout_id"], row["source_sha256"]):
                raise ValueError("Campaign board does not match the source index")
            if row["source_path"] not in hashes:
                hashes[row["source_path"]] = file_sha256(ROOT / row["source_path"])
            if hashes[row["source_path"]] != row["source_sha256"]:
                raise ValueError("Campaign source shard hash mismatch")
    return boards, actions, {"public_path": str(public.relative_to(ROOT)), "public_sha256": expected,
                             "solution_path": str(PRIVATE.relative_to(ROOT)),
                             "solution_sha256": file_sha256(PRIVATE), "source_shard_sha256": hashes}


def prepare(output: Path) -> dict[str, Any]:
    original = read_records(ORIGINAL)
    prior = read_records(PREVIOUS / "dataset.jsonl")
    previous_ids = json.loads((PREVIOUS / "test_ids.json").read_text())["record_ids"]
    if len(previous_ids) != 40:
        raise ValueError("Expected the previous paired 40 contexts")
    history = json.loads(HISTORY.read_text())
    observed, reserved = set(history["observed_layout_ids"]), set(history["reserved_layout_ids"])
    campaign, actions_by_id, provenance = campaign_inputs()
    prior_real = {row["layout_id"]: row for row in prior if row["evaluation_family"] == EvaluationFamily.REAL}
    if len(prior_real) != 20 or not set(prior_real) <= {row["layout_id"] for row in campaign}:
        raise ValueError("Previous real contexts are not a subset of the campaign")
    test_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, board in enumerate(sorted(campaign, key=lambda row: row["layout_id"])):
        actions = actions_by_id[board["board_id"]]
        try:
            states = checked_solution(board["state"], actions)
            old = prior_real.get(board["layout_id"])
            if old is not None:
                row = dict(old)
                start, end = row["i"], row["i"] + row["k"]
                if (states[start] != row["start_observation"] or states[end] != row["later_observation"]
                        or actions[start:end] != row["fragment_actions"]):
                    raise ValueError("Previous fragment differs from the complete source replay")
            else:
                exposure = {"snapshot_observed": board["layout_id"] in observed,
                            "snapshot_reserved": board["layout_id"] in reserved,
                            "source_cohort": "campaign_dataset_12x12_v1/pilot100", **provenance}
                row = make_records(board, Phase.TEST, Family.REAL, states, actions, 1, exposure,
                                   TEST_TARGET_OFFSETS[index % len(TEST_TARGET_OFFSETS)])[0]
            row.update(split="test", family=EvaluationFamily.REAL, evaluation_family=EvaluationFamily.REAL,
                       previously_evaluated_in_lora_matrix=old is not None)
            replay(row)
            test_rows.append(row)
        except (ValueError, IndexError, KeyError) as exc:
            rejected.append({"family": EvaluationFamily.REAL, "layout_id": board["layout_id"], "reason": str(exc)})

    proposal = [dict(row) for row in prior if row["evaluation_family"] == EvaluationFamily.PROPOSAL]
    if len(proposal) != 20:
        raise ValueError("Expected previous 20 proposal contexts")
    for row in proposal:
        row.update(family=EvaluationFamily.PROPOSAL, previously_evaluated_in_lora_matrix=True)
        replay(row)
    test_rows.extend(proposal)

    generated = json.loads(GENERATED.read_text())["instances"]
    generated_actions = {row["id"]: row["actions"] for row in json.loads(GENERATED_ACTIONS.read_text())}
    if len(generated) != len(generated_actions) or {row["id"] for row in generated} != set(generated_actions):
        raise ValueError("Generated boards and saved actions do not match")
    for index, board in enumerate(sorted(generated, key=lambda row: layout(row))):
        board_layout = layout(board)
        source = {"layout_id": board_layout, "trajectory_id": board["id"],
                  "source_path": str(GENERATED.relative_to(ROOT)), "source_sha256": file_sha256(GENERATED)}
        try:
            actions = generated_actions[board["id"]]
            states = checked_solution(board["state"], actions)
            exposure = {"snapshot_observed": board_layout in observed, "snapshot_reserved": board_layout in reserved,
                        "source_cohort": "generated_12x12x10/test", "synthetic": True,
                        "actions_path": str(GENERATED_ACTIONS.relative_to(ROOT)),
                        "actions_sha256": file_sha256(GENERATED_ACTIONS)}
            row = make_records(source, Phase.TEST, Family.SYNTHETIC, states, actions, 1, exposure,
                               TEST_TARGET_OFFSETS[index % len(TEST_TARGET_OFFSETS)])[0]
            row.update(split="test", family=EvaluationFamily.GENERATED,
                       evaluation_family=EvaluationFamily.GENERATED, previously_evaluated_in_lora_matrix=False)
            replay(row)
            test_rows.append(row)
        except (ValueError, IndexError, KeyError) as exc:
            rejected.append({"family": EvaluationFamily.GENERATED, "layout_id": board_layout, "reason": str(exc)})

    rows = [row for row in original if row["split"] in {"train", "validation"}] + test_rows
    validate_groups(rows)
    order = frozen_order(test_rows, previous_ids)
    encoded = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    data_hash = hashlib.sha256(encoded.encode()).hexdigest()
    ids = {"schema_version": 1, "dataset_sha256": data_hash, "record_ids": order}
    inputs = (ORIGINAL, PREVIOUS / "dataset.jsonl", PREVIOUS / "test_ids.json", GENERATED, GENERATED_ACTIONS,
              HISTORY, CAMPAIGN / "campaign_dataset_manifest.json")
    summary = {
        "schema_version": 1, "dataset_sha256": data_hash,
        "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in inputs},
        "campaign_source": provenance,
        "selection": "Previous 40 frozen contexts first; then one earliest eligible non-walking context per layout, "
                     "cycling offsets 1,2,4,8,16,24,8,16; selection independent of model results.",
        "evaluation_order": "Previous x4 and proposal x10 contexts retain exact order; remaining families interleaved.",
        "counts": {str(key): value for key, value in Counter(row["split"] for row in rows).items()},
        "test_family_counts": {str(key): value for key, value in Counter(row["evaluation_family"] for row in test_rows).items()},
        "test_previously_evaluated": sum(row["previously_evaluated_in_lora_matrix"] for row in test_rows),
        "test_not_in_preceding_lora_matrix": sum(not row["previously_evaluated_in_lora_matrix"] for row in test_rows),
        "test_snapshot_observed": sum(row["layout_id"] in observed for row in test_rows),
        "rejected": rejected,
        "replay": "Every retained campaign and generated-v1 full solution replayed in SokobanEnv; "
                  "all selected fragments and exact annotations checked. Prior proposal fragments replayed.",
        "limitations": ["No test records are used as train or validation.",
                        "The real pilot cohort is reserved for evaluation; exposure after the saved history is unknown.",
                        "Absence from the history snapshot does not establish model unexposure.",
                        "Generated-v1 ×10 is a separate source and is known observed in the history snapshot.",
                        "Previous 40 test contexts were already assessed; all results must be labelled exploratory."],
    }
    output.mkdir(parents=True, exist_ok=True)
    files = {"dataset.jsonl": encoded, "test_ids.json": json.dumps(ids, indent=2) + "\n",
             "manifest.json": json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
             "rejected.jsonl": "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected)}
    for name, content in files.items():
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to replace frozen data: {path}")
    for name, content in files.items():
        path = output / name
        temporary = path.with_name(f".{name}.{os.getpid()}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({key: result[key] for key in ("counts", "test_family_counts", "rejected")}, indent=2))


if __name__ == "__main__":
    main()
