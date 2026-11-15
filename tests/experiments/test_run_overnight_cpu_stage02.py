"""Checks for trace reuse and atomic full-state stage-two rollouts."""
from __future__ import annotations

import json

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts import run_overnight_cpu_policy as stage1
from scripts import run_overnight_cpu_stage02 as runner
from scripts.figure1_cllp_matched import rollout_pairs


def candidate(origin="bfs", *, candidate_id="a" * 64, fixed=True):
    cells = np.ones((12, 12), dtype=int)
    cells[[0, -1], :] = 0
    cells[:, [0, -1]] = 0
    cells[2, 2:5] = 3
    cells[5, 3], cells[5, 4], cells[5, 5] = 5, 4, 2
    source = np.eye(7, dtype=np.float32)[cells]
    target = np.asarray(environment(source).next_state(source, 2))
    return {"candidate_id": candidate_id, "context_id": "ctx", "layout_id": canonical_layout_id(source),
            "family": "12x12x4", "source_family": "trusted", "split": "development", "board_rank": 0,
            "context_rank": 0, "source": to_rows(source), "source_hash": state_hash(source),
            "target": to_rows(target), "target_hash": state_hash(target), "candidate_origin": origin,
            "demo_pair_ids": ["original"] if origin == "demonstration" else [],
            "demo_offsets": [1] if origin == "demonstration" else [], "in_fixed_pool": fixed}


def old_reference(tmp_path, row):
    old = {**row, "pair_id": "original", "trajectory_id": "traj", "i": 0, "k": 1}
    group = {"family": row["family"], "source_family": row["source_family"], "split": row["split"],
             "layout_id": row["layout_id"], "board_rank": row["board_rank"], "trajectory_id": "traj",
             "records": [old]}
    pair = stage1.materialise_pairs(group)[0]
    raw = rollout_pairs([pair], lambda states, targets: [2] * len(states), "cllp8")[0]
    trial = stage1.decorate_trial(pair, raw, group, stage1.ModelId.CLLP_8)
    shard = {"schema_version": 1, "status": "complete", "model_id": "cllp8", "family": row["family"],
             "source_family": row["source_family"], "split": row["split"], "layout_id": row["layout_id"],
             "contract_sha256": "old-contract", "planned_pair_ids": ["original"], "trials": [trial],
             "memo_stats": {"requests": 1, "unique_decisions": 1, "previous_cache_hits": 0,
                            "same_batch_hits": 0}}
    path = stage1.shard_path(tmp_path, group, stage1.ModelId.CLLP_8)
    stage1.atomic_json(path, shard)
    return {"directory": tmp_path, "contract_sha256": "old-contract", "pairs": {"original": old},
            "groups": {(row["family"], row["layout_id"]): group}, "shards": {}}, path


def test_all_demonstrations_and_generator_outputs_but_only_selected_bfs():
    # Distinct synthetic contexts avoid constructing unneeded duplicate boards.
    rows = [candidate("demonstration", candidate_id="a" * 64, fixed=False),
            candidate("generator", candidate_id="b" * 64, fixed=False),
            candidate("bfs", candidate_id="c" * 64, fixed=False),
            candidate("bfs", candidate_id="d" * 64, fixed=True)]
    for i, row in enumerate(rows):
        row["context_id"] = f"ctx{i}"
    selected = runner.selected_candidates(rows, "development", None)
    assert {row["candidate_id"] for row in selected} == {"a" * 64, "b" * 64, "d" * 64}


def test_generated_alias_of_unselected_bfs_is_measured_once():
    row = candidate("bfs", fixed=False)
    row.update(also_generated=True, generator_ranks=[0, 1])
    selected = runner.selected_candidates([row], "development", None)
    assert len(selected) == 1
    assert selected[0]["candidate_origin"] == "bfs" and not selected[0]["in_fixed_pool"]
    assert selected[0]["generator_ranks"] == [0, 1]
    row["generator_ranks"] = []
    with pytest.raises(ValueError, match="generated-target alias"):
        runner.selected_candidates([row], "development", None)


def test_duplicate_target_and_split_overlap_are_rejected():
    one, duplicate = candidate(candidate_id="a" * 64), candidate(candidate_id="b" * 64)
    with pytest.raises(ValueError, match="Duplicate full target"):
        runner.selected_candidates([one, duplicate], "development", None)
    one["split"] = "train"
    duplicate.update(split="evaluation", context_id="other")
    with pytest.raises(ValueError, match="multiple splits"):
        runner.selected_candidates([one, duplicate], "train", None)


def test_stage_one_trace_is_replayed_and_reused_without_controller(tmp_path):
    row = candidate("demonstration")
    reference, old_shard = old_reference(tmp_path, row)
    trace = runner.existing_attempt(row, stage1.ModelId.CLLP_8, reference)
    assert trace["exact_hit"] is True and trace["first_hit"] == 1
    assert trace["reuse_ref"]["path"] == str(old_shard)
    assert trace["reuse_ref"]["pair_ids"] == ["original"]
    saved = runner.attempt_record(row, stage1.ModelId.CLLP_8, trace, "new-contract")
    runner.validate_attempt(row, stage1.ModelId.CLLP_8, saved, "new-contract")
    shard = json.loads(old_shard.read_text())
    shard["trials"][0]["actions"] = [0]
    stage1.atomic_json(old_shard, shard)
    with pytest.raises(ValueError, match="shard changed"):
        runner.validate_attempt(row, stage1.ModelId.CLLP_8, saved, "new-contract")


def test_missing_demonstration_is_missing_not_a_policy_failure(tmp_path):
    row = candidate("demonstration")
    reference, path = old_reference(tmp_path, row)
    path.unlink()
    assert runner.existing_attempt(row, stage1.ModelId.CLLP_8, reference) is None


def test_new_training_demonstration_runs_without_a_reuse_reference(tmp_path, monkeypatch):
    row = candidate("demonstration")
    row["split"] = "train"
    monkeypatch.setitem(stage1.WORKER, "models", {stage1.ModelId.CLLP_8: object()})
    monkeypatch.setattr(runner, "model_predictor", lambda *_args: lambda states, targets: [2] * len(states))
    runner.run_context([row], "cllp8", str(tmp_path), "contract")
    saved = json.loads(runner.attempt_path(tmp_path, stage1.ModelId.CLLP_8, row["candidate_id"]).read_text())
    assert saved["reuse_ref"] is None and saved["measurement_origin"] == "new_train_demonstration"
    runner.validate_attempt(row, stage1.ModelId.CLLP_8, saved, "contract")
    with pytest.raises(ValueError, match="cannot reuse"):
        runner.existing_attempt(row, stage1.ModelId.CLLP_8, {})


def test_development_demonstration_cannot_be_labelled_new():
    row = candidate("demonstration")
    with pytest.raises(ValueError, match="reuse provenance"):
        runner.validate_attempt(row, stage1.ModelId.CLLP_8,
                                {"schema_version": 1, "status": "complete", "controller": "cllp8",
                                 "contract_sha256": "contract", **{key: row[key] for key in (
                                     "candidate_id", "context_id", "layout_id", "family", "source_family", "split",
                                     "board_rank", "context_rank", "source_hash", "target_hash")}}, "contract")


def train_exposure_fixture(tmp_path):
    old_data = tmp_path / "stage_01" / "data_v2"
    old_data.mkdir(parents=True)
    pairs = old_data / "pairs.jsonl"
    pairs.write_text("")
    trajectories = old_data / "trajectories.jsonl"
    trajectories.write_text(json.dumps({"layout_id": "b" * 64}) + "\n")
    previous = {"canonicalization": "carl-layout-d4-v1", "development_layout_ids": ["b" * 64],
                "campaign_evaluation_layout_ids": ["c" * 64], "training_prohibited_layout_ids": ["c" * 64]}
    previous_path = old_data / "exposure_register.json"
    stage1.atomic_json(previous_path, previous)
    campaign_path = tmp_path / "campaign_manifest.json"
    stage1.atomic_json(campaign_path, {"stages": {"01": {"data_freeze": {
        "exposure_sha256": stage1.file_sha(previous_path)}}}})
    history = {"canonicalization": "carl-layout-d4-v1", "observed_layout_ids": ["d" * 64],
               "previously_protected_layout_ids": ["e" * 64]}
    history_path = tmp_path / "history.json"
    stage1.atomic_json(history_path, history)
    source = tmp_path / "source.json"
    source.write_text("trusted source")
    manifest = {"schema_version": 1, "role": "campaign_train", "canonicalization": "carl-layout-d4-v1",
                "old_checkpoint_exposure": "unknown", "train_layout_ids": [],
                "historical_observed_layout_ids": history["observed_layout_ids"],
                "historical_protected_layout_ids": history["previously_protected_layout_ids"],
                "stage01_development_layout_ids": previous["development_layout_ids"],
                "stage01_evaluation_layout_ids": previous["campaign_evaluation_layout_ids"],
                "stage01_training_prohibited_layout_ids": previous["training_prohibited_layout_ids"],
                "stage01_exposure_path": str(previous_path), "stage01_exposure_sha256": stage1.file_sha(previous_path),
                "stage01_trajectories_sha256": stage1.file_sha(trajectories),
                "fresh_history_path": str(history_path), "fresh_history_sha256": stage1.file_sha(history_path)}
    for prefix in ("source", "indexed_source", "alias_source"):
        manifest[f"{prefix}_path"] = str(source)
        manifest[f"{prefix}_sha256"] = stage1.file_sha(source)
    return manifest, {"contract": {"input_pairs_path": str(pairs)}}, source


def test_training_exposure_verifies_sources_and_rejects_stage1_or_history_overlap(tmp_path):
    manifest, reference, source = train_exposure_fixture(tmp_path)
    row = candidate("demonstration")
    row.update(split="train", source_sha256=stage1.file_sha(source))
    manifest["train_layout_ids"] = [row["layout_id"]]
    path = tmp_path / "train_exposure.json"
    stage1.atomic_json(path, manifest)
    proof = runner.verify_train_exposure([row], path, reference)
    assert proof["old_checkpoint_exposure"] == "unknown"
    manifest["train_layout_ids"].append("b" * 64)
    stage1.atomic_json(path, manifest)
    with pytest.raises(ValueError, match="overlap protected"):
        runner.verify_train_exposure([row], path, reference)
    manifest["train_layout_ids"] = [row["layout_id"]]
    manifest["historical_observed_layout_ids"] = []
    stage1.atomic_json(path, manifest)
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner.verify_train_exposure([row], path, reference)


def test_training_rejects_exposure_changed_after_campaign_freeze(tmp_path):
    manifest, reference, source = train_exposure_fixture(tmp_path)
    row = candidate("demonstration")
    row.update(split="train", source_sha256=stage1.file_sha(source))
    manifest["train_layout_ids"] = [row["layout_id"]]
    path = tmp_path / "train_exposure.json"
    stage1.atomic_json(path, manifest)
    stage1.atomic_json(tmp_path / "campaign_manifest.json", {"stages": {"01": {"data_freeze": {
        "exposure_sha256": "different"}}}})
    with pytest.raises(ValueError, match="frozen stage-one campaign manifest"):
        runner.verify_train_exposure([row], path, reference)


def test_new_target_uses_full_target_and_persists_each_finished_trace(tmp_path, monkeypatch):
    row = candidate("generator", fixed=False)
    seen = []

    def predictor(_model, _tokenizer, _device, _batch_size):
        def predict(states, targets):
            seen.extend((state_hash(state), state_hash(target)) for state, target in zip(states, targets))
            return [2] * len(states)
        return predict

    monkeypatch.setitem(stage1.WORKER, "models", {stage1.ModelId.CLLP_8: object()})
    monkeypatch.setattr(runner, "model_predictor", predictor)
    result = runner.run_context([row], "cllp8", str(tmp_path), "contract")
    saved = json.loads(runner.attempt_path(tmp_path, stage1.ModelId.CLLP_8, row["candidate_id"]).read_text())
    assert result["trials"] == 1 and seen == [(row["source_hash"], row["target_hash"])]
    assert saved["exact_hit"] and saved["first_hit"] == 1 and saved["reuse_ref"] is None
    assert "k" not in saved
    runner.validate_attempt(row, stage1.ModelId.CLLP_8, saved, "contract")


def test_merge_keeps_missing_attempts_distinct_from_failures(tmp_path, monkeypatch):
    row = candidate()
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(row) + "\n")
    plan = {"planned_attempts": [{"candidate_id": row["candidate_id"], "controller": "cllp8"},
                                  {"candidate_id": row["candidate_id"], "controller": "cllp32"}]}
    contract = {"plan_sha256": stage1.digest(plan), "candidates_path": str(candidates),
                "candidates_sha256": stage1.file_sha(candidates)}
    stage1.atomic_json(tmp_path / "plan.json", plan)
    stage1.atomic_json(tmp_path / "contract.json", contract)
    monkeypatch.setitem(stage1.WORKER, "models", {stage1.ModelId.CLLP_8: object()})
    monkeypatch.setattr(runner, "model_predictor", lambda *_args: lambda states, targets: [2] * len(states))
    runner.run_context([row], "cllp8", str(tmp_path), stage1.digest(contract))
    assert runner.merge(tmp_path) == {"planned": 2, "completed": 1, "missing": 1}
    assert len((tmp_path / "rollouts.jsonl").read_text().splitlines()) == 1


def test_interrupted_plan_write_is_recovered_only_for_identical_contract(tmp_path):
    plan = {"planned_attempts": [{"candidate_id": "candidate", "controller": "cllp8"}]}
    contract = {"plan_sha256": stage1.digest(plan), "candidate_file": "frozen"}
    stage1.atomic_json(tmp_path / "contract.json", contract)
    runner.freeze_plan(tmp_path, contract, plan, resume=True)
    assert json.loads((tmp_path / "plan.json").read_text()) == plan

    (tmp_path / "plan.json").unlink()
    changed_plan = {"planned_attempts": [{"candidate_id": "different", "controller": "cllp8"}]}
    with pytest.raises(ValueError, match="does not match its contract"):
        runner.freeze_plan(tmp_path, contract, changed_plan, resume=True)
    with pytest.raises(ValueError, match="saved contract changed"):
        runner.freeze_plan(tmp_path, {**contract, "candidate_file": "different"}, plan, resume=True)
    assert not (tmp_path / "plan.json").exists()
