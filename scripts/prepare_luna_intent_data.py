#!/usr/bin/env python3
"""Freeze and replay short, locally annotated Sokoban controller intents.

The calibration boards are campaign scale-validation boards. Test boards are
from the campaign pilot and the prior proposal's frozen synthetic test split.
Targets and complete demonstration fragments are stored for audit only; none
of the latter belong in controller prompts. No model is invoked by this script.
"""

import argparse
import contextlib
import hashlib
import io
import json
import sqlite3
import tarfile
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.campaign_data import sha256_file
from carl.icl.campaign_data import verify_record
from carl.icl.data_sources.sokoban import verify_solution
from carl.icl.intent_data import SCHEMA_VERSION
from carl.icl.intent_data import EffectType
from carl.icl.intent_data import validate_intent
from scripts.prepare_intent_generator_data import candidate
from scripts.prepare_intent_generator_data import load_data_sources
from scripts.prepare_intent_generator_data import prior_data
from scripts.prepare_intent_generator_data import replay
from scripts.prepare_intent_generator_data import write_json
from scripts.prepare_intent_generator_data import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data/icl/campaign_dataset_v2"
PRIVATE = ROOT / "data/icl/campaign_dataset_v2_private"
HISTORY = ROOT / "analysis/luna_icl/campaign_100/history_snapshot.json"
PREVIOUS_CALIBRATION = ROOT / "data/icl/reachability_exploratory_v2/calibration_pairs.json"
PREVIOUS_SPLIT = ROOT / "analysis/intent_generator_20261114T032232Z/frozen_split.json"
OFFSETS = (1, 2, 4, 8, 16, 24)
TEST_TARGET_OFFSETS = (1, 2, 4, 8, 16, 24, 8, 16)
TASK_GOAL = "Put every box on a goal."


class Family(StrEnum):
    """Dataset families with different board-generation provenance."""

    REAL = "real_12x12x4"
    SYNTHETIC = "proposal_gamma2_12x12x10"


class Phase(StrEnum):
    """Disjoint uses of the frozen datasets."""

    CALIBRATION = "calibration"
    TEST = "test"


def identity(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def write_frozen(output: Path, records: list[dict[str, Any]], rejected: list[dict[str, Any]],
                 manifest: dict[str, Any]) -> None:
    """Allow exact reproduction, but never replace an existing frozen cohort."""
    def lines(values: list[dict[str, Any]]) -> str:
        return "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in values)
    expected = {
        output / "records.jsonl": lines(records),
        output / "rejected.jsonl": lines(rejected),
        output / "manifest.json": json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    }
    for path, content in expected.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to replace different frozen data: {path}")
    if not (output / "records.jsonl").exists():
        write_jsonl(output / "records.jsonl", records)
    if not (output / "rejected.jsonl").exists():
        write_jsonl(output / "rejected.jsonl", rejected)
    if not (output / "manifest.json").exists():
        write_json(output / "manifest.json", manifest)


def candidates_at_start(states: list[list[str]], index: int) -> list[tuple[int, str, dict[str, Any]]]:
    """Build valid non-walking local labels in the fixed offset order."""
    found = []
    for offset in OFFSETS:
        if index + offset >= len(states):
            continue
        proposed = candidate(states[index], states[index + offset], index + offset)
        if proposed is None or proposed[0] == EffectType.WALK_ONLY:
            continue
        kind, intent = proposed
        errors = validate_intent(intent, states[index:index + offset + 1])
        if errors:
            raise ValueError(f"Local candidate failed exact validation: {errors}")
        found.append((offset, kind, intent))
    return found


def choose_context(states: list[list[str]], count: int, target_offset: int | None = None
                   ) -> tuple[int, list[tuple[int, str, dict[str, Any]]]]:
    """Use the earliest state with enough distinct, demonstration-derived targets."""
    for index in range(len(states) - 1):
        options = candidates_at_start(states, index)
        if target_offset is not None and options:
            return index, [min(options, key=lambda item: (abs(item[0] - target_offset), item[0]))]
        # Calibration needs distinct intentions, not repeated offsets to the
        # same condition on the same start board.
        unique = {}
        for option in options:
            unique.setdefault(json.dumps(option[2]["effects"], sort_keys=True), option)
        options = list(unique.values())
        if len(options) >= count:
            return index, options[:count]
    raise ValueError(f"No context with {count} distinct non-walking intents")


def make_records(source: dict[str, Any], phase: Phase, family: Family, states: list[list[str]],
                 actions: list[int], count: int, provenance: dict[str, Any],
                 target_offset: int | None = None) -> list[dict[str, Any]]:
    index, options = choose_context(states, count, target_offset)
    context_id = identity([family, source["layout_id"], index, states[index]])
    records = []
    for offset, kind, intent in options:
        record_id = identity([context_id, offset, intent])
        records.append({
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "context_id": context_id,
            "phase": phase,
            "split": phase,
            "family": family,
            "layout_id": source["layout_id"],
            "trajectory_id": source["trajectory_id"],
            "source_path": source["source_path"],
            "source_sha256": source["source_sha256"],
            "i": index,
            "k": offset,
            "selection_target_offset": target_offset,
            "start_observation": states[index],
            "later_observation": states[index + offset],
            "fragment_observations": states[index:index + offset + 1],
            "fragment_actions": actions[index:index + offset],
            "task_goal": TASK_GOAL,
            "intent": intent,
            "effect_type": kind,
            "annotation_provenance": {"kind": "local_exact_difference", "model": None,
                                      "uses_recorded_future": True},
            "validation": {"accepted": True, "environment_replay": True,
                           "each_effect_false_initially_true_finally": True,
                           "preservation_checked_at_every_recorded_state": True,
                           "rejection_reasons": []},
            "policy_measurements": None,
            "source_provenance": provenance,
        })
    return records


def campaign_group(split: str) -> tuple[list[dict[str, Any]], dict[str, list[int]], dict[str, Any]]:
    public = PUBLIC / Family.REAL / f"{split}.json"
    private = PRIVATE / split / f"{Family.REAL}.json"
    manifest = json.loads((PUBLIC / "campaign_dataset_manifest.json").read_text())
    if sha256_file(public) != manifest["manifest_sha256"][f"{Family.REAL}/{split}.json"]:
        raise ValueError(f"Campaign public checksum differs: {split}")
    boards = json.loads(public.read_text())["instances"]
    witnesses = json.loads(private.read_text())["witnesses"]
    by_id = {row["board_id"]: row["actions"] for row in witnesses}
    if len(by_id) != len(boards) or set(by_id) != {board["board_id"] for board in boards}:
        raise ValueError(f"Public and private board IDs differ: {split}")
    db_path = ROOT / "data/icl/.private_campaign_shards.sqlite"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        for board in boards:
            verify_record(board, Family.REAL, split)
            indexed = db.execute(
                "SELECT boards.state_hash, boards.layout_id, shards.sha256 FROM boards "
                "JOIN shards ON boards.path=shards.path "
                "WHERE boards.path=? AND boards.trajectory_index=?",
                (board["source_path"], board["source_index"]),
            ).fetchone()
            if indexed != (board["state_hash"], board["layout_id"], board["source_sha256"]):
                raise ValueError(f"Campaign source index differs: {board['board_id']}")
    return boards, by_id, {"public_path": str(public.relative_to(ROOT)), "public_sha256": sha256_file(public),
                           "solution_path": str(private.relative_to(ROOT)), "solution_sha256": sha256_file(private)}


def read_reference_cohort(manifest_path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    """Read a frozen dataset only; neither model predictions nor responses are read."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_path = manifest_path.parent / "records.jsonl"
    rows = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    layouts = set(manifest["calibration_layouts"]) | set(manifest["test_layouts_x4"]) | set(manifest["test_layouts_x10"])
    if layouts != {row["layout_id"] for row in rows} or len(rows) != manifest["record_count"]:
        raise ValueError("Reference manifest and frozen records disagree")
    return layouts, [row for row in rows if row["phase"] == Phase.CALIBRATION]


def prepare(output: Path, test_count: int = 8, calibration_count: int = 4,
            reference_manifests: tuple[Path, ...] = ()) -> dict[str, Any]:
    if test_count < 1 or not 1 <= calibration_count <= 4:
        raise ValueError("Counts must be positive; at most four calibration boards exist")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(HISTORY.read_text())
    observed = set(snapshot["observed_layout_ids"])
    reserved = set(snapshot["reserved_layout_ids"])
    previous_calibration = {row["layout_id"] for row in json.loads(PREVIOUS_CALIBRATION.read_text())["pairs"]}
    calibration, calibration_actions, calibration_provenance = campaign_group("scale_validation")
    pilot, pilot_actions, pilot_provenance = campaign_group("pilot100")
    selected_calibration = sorted(calibration, key=lambda row: row["layout_id"])[:calibration_count]
    calibration_ids = {row["layout_id"] for row in selected_calibration}
    if calibration_ids & (observed | reserved | previous_calibration):
        raise ValueError("Calibration intersects a known observed, reserved or previous calibration layout")
    excluded_layouts: set[str] = set()
    references = []
    for reference_manifest in reference_manifests:
        layouts, calibration_rows = read_reference_cohort(reference_manifest)
        excluded_layouts.update(layouts)
        references.append((reference_manifest, layouts, calibration_rows))
    pilot_candidates = sorted((row for row in pilot if row["layout_id"] not in observed | previous_calibration
                               | calibration_ids | excluded_layouts), key=lambda row: row["layout_id"])
    old_split = json.loads(PREVIOUS_SPLIT.read_text())
    proposal_candidates = sorted((row for row in old_split["sources"] if row["split"] == "test"
                                  and row["layout_id"] not in observed | previous_calibration
                                  | calibration_ids | excluded_layouts),
                                 key=lambda row: row["layout_id"])
    sources = load_data_sources()
    old_train, old_pairs, old_states = prior_data(sources)
    with tarfile.open(sources.prior_bundle, "r:gz") as archive:
        entry = archive.extractfile(sources.witness_archive_member)
        if entry is None:
            raise ValueError("Historical proposal bundle lacks documented trajectories")
        prior_witnesses = {row["id"]: row for row in json.load(entry)}
    rejected = []
    records: list[dict[str, Any]] = []

    def add_campaign(board: dict[str, Any], actions: list[int], phase: Phase, count: int,
                     provenance: dict[str, Any], target_offset: int | None = None) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            arrays = verify_solution(from_rows(board["state"]), actions)
        rows = [to_rows(state) for state in arrays]
        exposure = {"snapshot_observed": board["layout_id"] in observed,
                    "snapshot_reserved": board["layout_id"] in reserved,
                    "previous_exploratory_calibration": board["layout_id"] in previous_calibration,
                    "source_cohort": board["split"], **provenance}
        records.extend(make_records(board, phase, Family.REAL, rows, actions, count, exposure, target_offset))

    for board in selected_calibration:
        add_campaign(board, calibration_actions[board["board_id"]], Phase.CALIBRATION, 3, calibration_provenance)
    if any(records != calibration_rows for _, _, calibration_rows in references):
        raise ValueError("Replayed calibration records differ from a reference dataset")
    selected_pilot = []
    for board in pilot_candidates:
        try:
            target = TEST_TARGET_OFFSETS[len(selected_pilot) % len(TEST_TARGET_OFFSETS)]
            add_campaign(board, pilot_actions[board["board_id"]], Phase.TEST, 1, pilot_provenance, target)
            selected_pilot.append(board["layout_id"])
        except ValueError as exc:
            rejected.append({"family": Family.REAL, "layout_id": board["layout_id"], "reason": str(exc)})
        if len(selected_pilot) == test_count:
            break
    selected_proposal = []
    for source in proposal_candidates:
        try:
            path = ROOT / source["path"]
            if sha256_file(path) != source["sha256"]:
                raise ValueError("Source checksum differs from frozen split")
            with contextlib.redirect_stdout(io.StringIO()):
                states, actions, replay_audit = replay(path, old_train.get(source["id"], []),
                                                       old_pairs.get(source["id"], []), old_states,
                                                       prior_witnesses[source["id"]])
            if canonical_layout_id(from_rows(states[0])) != source["layout_id"]:
                raise ValueError("Source layout differs from frozen split")
            if len(states[0]) != 12 or any(len(row) != 12 for row in states[0]) or sum(
                    row.count("$") + row.count("*") for row in states[0]) != 10:
                raise ValueError("Expected an actual 12x12 board with 10 boxes")
            info = {"source_path": source["path"], "source_sha256": source["sha256"],
                    "layout_id": source["layout_id"], "trajectory_id": source["id"]}
            provenance = {"snapshot_observed": source["layout_id"] in observed,
                          "snapshot_reserved": source["layout_id"] in reserved,
                          "previous_exploratory_calibration": source["layout_id"] in previous_calibration,
                          "source_cohort": "previously frozen proposal test",
                          "synthetic": True, "replay_audit": replay_audit}
            target = TEST_TARGET_OFFSETS[len(selected_proposal) % len(TEST_TARGET_OFFSETS)]
            records.extend(make_records(info, Phase.TEST, Family.SYNTHETIC, states, actions, 1, provenance, target))
            selected_proposal.append(source["layout_id"])
        except (ValueError, KeyError) as exc:
            rejected.append({"family": Family.SYNTHETIC, "layout_id": source["layout_id"], "reason": str(exc)})
        if len(selected_proposal) == test_count:
            break
    if len(selected_pilot) != test_count or len(selected_proposal) != test_count:
        raise ValueError("Insufficient valid independent test boards; see selection algorithm and source data")
    if calibration_ids & (set(selected_pilot) | set(selected_proposal)) or set(selected_pilot) & set(selected_proposal):
        raise ValueError("Canonical layout overlap between groups")
    if excluded_layouts & (set(selected_pilot) | set(selected_proposal)):
        raise ValueError("Test layout overlaps the reference dataset")
    counts = Counter((r["phase"], r["family"], r["context_id"]) for r in records)
    manifest = {
        "schema_version": 1,
        "selection_rule": (
            "Sort eligible canonical layout IDs, take first valid boards; earliest trajectory state with enough "
            "distinct non-walking labels; offsets 1,2,4,8,16,24; test targets cycle through 1,2,4,8,16,24,8,16, "
            "choosing the nearest valid offset with ties towards smaller values; no model results used."
        ),
        "calibration_source": "real 12x12x4 scale_validation, not heldout/pilot",
        "test_sources": ["real 12x12x4 campaign pilot100", "synthetic 12x12x10 proposal frozen test"],
        "history_snapshot_path": str(HISTORY.relative_to(ROOT)),
        "history_snapshot_sha256": sha256_file(HISTORY),
        "earlier_calibration_path": str(PREVIOUS_CALIBRATION.relative_to(ROOT)),
        "earlier_calibration_sha256": sha256_file(PREVIOUS_CALIBRATION),
        "previous_proposal_split_path": str(PREVIOUS_SPLIT.relative_to(ROOT)),
        "previous_proposal_split_sha256": sha256_file(PREVIOUS_SPLIT),
        "calibration_layouts": sorted(calibration_ids), "test_layouts_x4": selected_pilot,
        "test_layouts_x10": selected_proposal,
        "intents_per_context": {key[2]: value for key, value in counts.items()},
        "test_offsets": {family: [{"context_id": r["context_id"], "target": r["selection_target_offset"],
                                  "actual": r["k"]} for r in records if r["phase"] == Phase.TEST
                                and r["family"] == family] for family in Family},
        "record_count": len(records), "rejected": rejected,
        "constraints": ["The pilot set is reserved for evaluation; it is not necessarily previously unseen.",
                        "History and exposure after the saved snapshot, or outside the scanned records, are unknown.",
                        "Offsets and teacher actions are demonstrations, never policy duration or scale u.",
                        "The targets were derived using a recorded future; never include it in controller input.",
                        "Scale labels require separately measured policy attempts and are not present here."],
    }
    if references:
        summaries = [{
            "manifest_path": str(path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(path),
            "records_sha256": sha256_file(path.parent / "records.jsonl"),
            "excluded_layouts": len(layouts),
            "calibration_records_identical": True,
            "new_test_layout_intersection": 0,
            "interpretation": "Test boards differ across versions; results are descriptive and not paired by board.",
        } for path, layouts, _ in references]
        if len(summaries) == 1:
            # Preserve the original single-reference manifest for reproducibility.
            manifest["reference_cohort"] = summaries[0]
        else:
            manifest["reference_cohorts"] = summaries
            manifest["unique_reference_layouts_excluded"] = len(excluded_layouts)
            manifest["eligible_test_candidates_after_exclusions"] = {
                Family.REAL: len(pilot_candidates), Family.SYNTHETIC: len(proposal_candidates)}
    write_frozen(output, records, rejected, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-per-family", type=int, default=8)
    parser.add_argument("--calibration-boards", type=int, default=4)
    parser.add_argument("--exclude-test-layouts-from", type=Path, action="append", default=[],
                        help="Repeat for each frozen manifest; retain its calibration and exclude its layouts from tests")
    args = parser.parse_args()
    references = tuple(path.resolve() for path in args.exclude_test_layouts_from)
    manifest = prepare(args.output.resolve(), args.test_per_family, args.calibration_boards, references)
    print(json.dumps({"records": manifest["record_count"], "intents_per_context": manifest["intents_per_context"]}))


if __name__ == "__main__":
    main()
