"""Focused offline checks for frozen CPU policy rollout contracts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from carl.environment.training_goal import TrainingGoal
from carl.icl.boards import canonical_layout_id
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts import run_overnight_cpu_policy as runner
from scripts.figure1_cllp_matched import Pair
from scripts.figure1_cllp_matched import rollout_pairs


def make_board(boxes):
    board = np.ones((12, 12), dtype=int)
    board[[0, -1], :] = 0
    board[:, [0, -1]] = 0
    # The final box can be pushed right onto its goal in a single move.
    for col in range(2, boxes + 1):
        board[2 if col < 10 else 3, col if col < 10 else 2] = 3
    board[5, 3] = 5
    board[5, 4] = 4
    board[5, 5] = 2
    return np.eye(7, dtype=np.float32)[board]


def make_row(boxes, *, split="evaluation", board_rank=0, context_rank=0, k=1):
    source = make_board(boxes)
    target = np.asarray(environment(source).next_state(source, 3))
    layout = canonical_layout_id(source)
    return {"pair_id": f"{boxes}-{split}-{board_rank}-{context_rank}-{k}", "layout_id": layout,
            "family": f"12x12x{boxes}", "source_family": f"source-{boxes}", "split": split,
            "trajectory_id": f"t-{boxes}-{board_rank}", "board_rank": board_rank,
            "context_rank": context_rank, "context_id": f"ctx-{boxes}-{context_rank}", "i": context_rank,
            "k": k, "source": to_rows(source), "target": to_rows(target),
            "source_hash": state_hash(source), "target_hash": state_hash(target),
            "demonstration_actions": [3], "private_teacher_detail": "not a generator input"}


def test_selection_filters_frozen_ranks_and_strips_demonstration_actions():
    rows = [make_row(4, context_rank=1), make_row(10, context_rank=1),
            make_row(4, context_rank=0), make_row(10, context_rank=0)]
    groups = runner.select_groups(rows, "evaluation", 1, max_contexts=1)
    assert {group["family"] for group in groups} == {"12x12x4", "12x12x10"}
    assert all(len(group["records"]) == 1 and group["records"][0]["context_rank"] == 0 for group in groups)
    assert all("demonstration_actions" not in group["records"][0] for group in groups)
    both = runner.select_groups(rows, "evaluation", 1, max_contexts=2)
    assert all([row["context_rank"] for row in group["records"]] == [0, 1] for group in both)


def test_split_overlap_and_duplicate_pair_fail_closed():
    row4, row10 = make_row(4), make_row(10)
    other = copy.deepcopy(row4)
    other.update(split="development", pair_id="different")
    with pytest.raises(ValueError, match="more than one split"):
        runner.select_groups([row4, row10, other], "evaluation", 1)
    with pytest.raises(ValueError, match="Duplicate pair_id"):
        runner.select_groups([row4, row10, row4], "evaluation", 1)


def test_box_counts_hashes_and_complete_state_targets_are_validated():
    rows = [make_row(4), make_row(10)]
    groups = runner.select_groups(rows, "evaluation", 1)
    for group in groups:
        pairs = runner.materialise_pairs(group)
        assert len(pairs) == 1 and pairs[0].target.shape == (12, 12, 7)
    bad = copy.deepcopy(groups[0])
    bad["records"][0]["target_hash"] = "not-the-state"
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.materialise_pairs(bad)
    bad = copy.deepcopy(groups[0])
    bad["family"] = "12x12x10"
    with pytest.raises(ValueError, match="box or goal count"):
        runner.materialise_pairs(bad)


def test_ordinary_policy_receives_current_state_only():
    seen = []

    class Tokenizer:
        def x_y_tokenizer(self, state, label, training_goal):
            assert label == 0 and training_goal is TrainingGoal.POLICY
            seen.append(state)
            return torch.ones((1, 146), dtype=torch.long), None

    class Model:
        config = SimpleNamespace(max_position_embeddings=512)

        def __call__(self, tokens):
            return SimpleNamespace(logits=torch.tensor([[0.1, 0.2, 0.3, 0.9]] * len(tokens)))

    source, secret_target = make_board(4), make_board(10)
    action = runner.policy_predictor(Model(), Tokenizer(), 4)([source], [secret_target])
    assert action == [3] and len(seen) == 1 and seen[0] is source


def test_simultaneous_target_and_task_goal_and_noops():
    source = make_board(4)
    env = environment(source)
    target = np.asarray(env.next_state(source, 3))
    assert env.is_solved(target)
    row = make_row(4)
    group = runner.select_groups([row, make_row(10)], "evaluation", 1)[0]
    pair = runner.materialise_pairs(group)[0]
    result = rollout_pairs([pair], lambda _states, _targets: [3], str(runner.ModelId.CLLP_8))[0]
    saved = runner.decorate_trial(pair, result, group, runner.ModelId.CLLP_8)
    assert saved["exact_hit"] and saved["task_solved"] and saved["stop_reason"] == "exact_target"
    assert saved["first_hit"] == 1 and saved["final_player"] == [5, 4]


def test_shared_board_environment_matches_separate_environments():
    source = make_board(4)
    env = environment(source)
    targets = [np.asarray(env.next_state(source, action)) for action in (2, 3)]
    layout = canonical_layout_id(source)
    pairs = [Pair(str(i), layout, "t", 0, 0, 1, state_hash(source), state_hash(target), source, target)
             for i, target in enumerate(targets)]

    def predictor(states, goals):
        return [2 if state_hash(goal) == state_hash(targets[0]) else 3 for goal in goals]

    separate = rollout_pairs(pairs, predictor, "cllp8")
    shared = rollout_pairs(pairs, predictor, "cllp8", make_environment=lambda _state: env)
    assert shared == separate


def test_cycle_detection_distinguishes_noop_and_larger_loops():
    assert runner.cycle_info(["a", "a", "a"]) == (False, None, None)
    assert runner.cycle_info(["a", "b", "a"]) == (True, 2, 2)
    assert runner.cycle_info(["a", "a", "b", "a"]) == (True, 3, 3)
    assert runner.cycle_info(["a", "b", "c"]) == (False, None, None)


def test_existing_shard_is_replayed_before_reuse(tmp_path):
    group = runner.select_groups([make_row(4), make_row(10)], "evaluation", 1)[0]
    pair = runner.materialise_pairs(group)[0]
    raw = rollout_pairs([pair], lambda _states, _targets: [3], str(runner.ModelId.CLLP_8))[0]
    trial = runner.decorate_trial(pair, raw, group, runner.ModelId.CLLP_8)
    shard = {"schema_version": 1, "status": "complete", "model_id": "cllp8", "family": group["family"],
             "source_family": group["source_family"], "split": group["split"], "layout_id": group["layout_id"],
             "contract_sha256": "abc", "planned_pair_ids": [pair.pair_id], "trials": [trial]}
    path = tmp_path / "shard.json"
    runner.atomic_json(path, shard)
    assert runner.validate_existing_shard(path, group, runner.ModelId.CLLP_8, "abc")["trials"] == 1
    shard["trials"][0]["task_solved"] = False
    runner.atomic_json(path, shard)
    with pytest.raises(ValueError, match="independent replay"):
        runner.validate_existing_shard(path, group, runner.ModelId.CLLP_8, "abc")


def test_requested_models_and_benchmark_configuration_parser():
    parser = runner.create_parser()
    args = parser.parse_args(["benchmark", "--pairs", "x", "--output", "y", "--max-boards", "10",
                              "--configs", "1x8", "2x4", "4x2", "8x1"])
    assert args.split == "development" and args.configs == ["1x8", "2x4", "4x2", "8x1"]
    assert set(runner.ModelId) == set(runner.CHECKPOINTS)
    assert runner.DEFAULT_MODELS[:2] == (runner.ModelId.CLLP_8, runner.ModelId.CLLP_32)


def test_deadline_stops_before_dispatch_and_records_missing_shards(tmp_path, monkeypatch):
    groups = runner.select_groups([make_row(4), make_row(10)], "evaluation", 1)
    plan = {"planned_shards": [{"pair_ids": [group["records"][0]["pair_id"]]} for group in groups],
            "planned_trials": 2}
    monkeypatch.setattr(runner, "controller_contract", lambda *_args: (plan, {"source_sha256": {}}))
    snapshot = {"host_total_bytes": 100, "host_available_bytes": 90, "host_used_fraction": 0.1,
                "swap_used_bytes": 0, "swap_total_bytes": 100, "children": [], "children_rss_bytes": 0}
    monkeypatch.setattr(runner, "resource_snapshot", lambda: snapshot)
    args = SimpleNamespace(workers=1, threads=1, batch_size=4, pairs=tmp_path / "pairs.jsonl",
                           output=tmp_path / "run", resume=False, stop_epoch=1.0, stop_monotonic=None,
                           use_memo=False, memo_max_entries=8192)
    result = runner.execute(args, groups, [runner.ModelId.CLLP_8])
    assert result["state"] == runner.RunState.PARTIAL
    assert result["stop_reason"] == "stage_epoch_cutoff"
    assert result["completed_shards"] == 0 and result["missing_shards"] == 2
    assert not list((args.output / "shards").glob("**/*.json"))


def test_memory_guard_stops_before_dispatch(tmp_path, monkeypatch):
    groups = runner.select_groups([make_row(4), make_row(10)], "evaluation", 1)
    plan = {"planned_shards": [{"pair_ids": [group["records"][0]["pair_id"]]} for group in groups],
            "planned_trials": 2}
    monkeypatch.setattr(runner, "controller_contract", lambda *_args: (plan, {"source_sha256": {}}))
    snapshot = {"host_total_bytes": 100, "host_available_bytes": 19, "host_used_fraction": 0.81,
                "swap_used_bytes": 0, "swap_total_bytes": 100, "children": [], "children_rss_bytes": 0}
    monkeypatch.setattr(runner, "resource_snapshot", lambda: snapshot)
    args = SimpleNamespace(workers=1, threads=1, batch_size=4, pairs=tmp_path / "pairs.jsonl",
                           output=tmp_path / "run", resume=False, stop_epoch=None, stop_monotonic=None,
                           use_memo=False, memo_max_entries=8192)
    result = runner.execute(args, groups, [runner.ModelId.CLLP_8])
    assert result["state"] == runner.RunState.PARTIAL
    assert result["stop_reason"] == "host_memory_limit" and result["missing_shards"] == 2


def test_memo_shard_preserves_full_rollout_and_records_decisions(tmp_path, monkeypatch):
    group = runner.select_groups([make_row(4), make_row(10)], "evaluation", 1)[0]
    monkeypatch.setitem(runner.WORKER, "models", {runner.ModelId.CLLP_8: object()})
    monkeypatch.setattr(runner, "model_predictor", lambda *_args: lambda states, _targets: [0] * len(states))
    baseline = runner.evaluate_shard(group, "cllp8", str(tmp_path / "base"), "base-hash", 4, False, 8192)
    memo = runner.evaluate_shard(group, "cllp8", str(tmp_path / "memo"), "memo-hash", 4, True, 8192)
    baseline_shard = json.loads(Path(baseline["shard"]).read_text())
    memo_shard = json.loads(Path(memo["shard"]).read_text())
    assert baseline_shard["trials"] == memo_shard["trials"]
    stats = memo_shard["memo_stats"]
    assert stats["requests"] == memo_shard["trials"][0]["attempted_actions"] == 32
    assert stats["unique_decisions"] < stats["requests"] and stats["previous_cache_hits"] > 0
    assert stats["requests"] == stats["unique_decisions"] + stats["previous_cache_hits"] + stats["same_batch_hits"]
    assert runner.validate_existing_shard(Path(memo["shard"]), group, runner.ModelId.CLLP_8,
                                          "memo-hash", use_memo=True)["memo_stats"] == stats
    stats["requests"] -= 1
    runner.atomic_json(Path(memo["shard"]), memo_shard)
    with pytest.raises(ValueError, match="memo statistics"):
        runner.validate_existing_shard(Path(memo["shard"]), group, runner.ModelId.CLLP_8,
                                       "memo-hash", use_memo=True)


def test_memo_module_hash_and_scope_are_frozen_in_contract(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text("{}\n")
    groups = runner.select_groups([make_row(4), make_row(10)], "evaluation", 1)
    args = SimpleNamespace(split="evaluation", max_boards=1, max_contexts=1, seed=7, workers=1, threads=1,
                           batch_size=4, use_memo=True, memo_max_entries=8192)
    plan, contract = runner.controller_contract(pairs_path, groups, [runner.ModelId.CLLP_8], args)
    memo_path = str(runner.ROOT / "scripts/overnight_cpu_memo.py")
    assert contract["memo"]["enabled"] and contract["memo"]["simulator_steps_skipped"] == 0
    assert contract["memo"]["scope"] == "one board and one fixed model"
    assert memo_path in contract["source_sha256"] and contract["plan_sha256"] == runner.digest(plan)


def test_dispatch_prefix_interleaves_families_by_frozen_board_rank():
    rows = []
    for boxes, rank in ((4, 2), (4, 0), (10, 1), (4, 1), (10, 0)):
        row = make_row(boxes, board_rank=rank)
        # Selection ordering is independent of board replay; other tests
        # validate that a real layout hash matches its board.
        row["layout_id"] = f"layout-{boxes}-{rank}"
        rows.append(row)
    groups = runner.select_groups(rows, "evaluation", None, max_contexts=1)
    assert [(group["family"], group["board_rank"]) for group in groups] == [
        ("12x12x4", 0), ("12x12x10", 0), ("12x12x4", 1), ("12x12x10", 1), ("12x12x4", 2)]
