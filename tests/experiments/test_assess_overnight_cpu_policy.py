"""Focused checks for board-balanced frozen-policy assessment."""

import csv
import json

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts.assess_overnight_cpu_policy import analyse
from scripts.assess_overnight_cpu_policy import load_plan
from scripts.assess_overnight_cpu_policy import load_unavailable
from scripts.assess_overnight_cpu_policy import make_plot
from scripts.assess_overnight_cpu_policy import read_shards
from scripts.assess_overnight_cpu_policy import replay_trial
from scripts.assess_overnight_cpu_policy import validate_frozen_plan
from scripts.assess_overnight_cpu_policy import write_csv


@pytest.fixture
def boards():
    source = from_rows(["#####", "#@--#", "#-$-#", "#-.-#", "#####"])
    env = environment(source)
    right = np.asarray(env.next_state(source, 3))
    far = np.asarray(env.next_state(right, 3))
    solved = np.asarray(env.next_state(right, 1))
    return source, far, solved


def make_pair(source, target, *, pair_id="p1", layout=None, context_rank=0, board_rank=0):
    layout = layout or canonical_layout_id(source)
    return {"pair_id": pair_id, "layout_id": layout, "family": "12x12x4", "source_family": "real",
            "split": "evaluation", "trajectory_id": "trajectory", "context_id": f"context-{context_rank}", "i": 0,
            "context_rank": context_rank, "board_rank": board_rank, "k": 1,
            "source": to_rows(source), "target": to_rows(target),
            "source_hash": state_hash(source), "target_hash": state_hash(target)}


def make_trial(pair, actions, *, model="cllp8"):
    source, target = from_rows(pair["source"]), from_rows(pair["target"])
    env = environment(source)
    current = source
    hashes = [state_hash(current)]
    noops = 0
    for action in actions:
        before = current
        current = np.asarray(env.next_state(before, action))
        hashes.append(state_hash(current))
        noops += int(np.array_equal(before, current))
    hit = bool(np.array_equal(current, target))
    solved = bool(env.done)
    stop = "exact_target" if hit else "task_terminal" if solved else "action_limit"
    player = np.argwhere(np.isin(current.argmax(-1), (5, 6)))[0].tolist()
    fields = ("pair_id", "layout_id", "family", "source_family", "split", "k", "source_hash", "target_hash")
    return {**{key: pair[key] for key in fields}, "status": "complete", "model_id": model,
            "actions": actions, "state_hashes": hashes, "no_op_count": noops, "action_cap": 32,
            "attempted_actions": len(actions), "first_hit": len(actions) if hit else None,
            "exact_hit": hit, "stop_reason": stop, "task_solved": solved,
            "final_state": to_rows(current), "final_player": player}


def test_plan_prefix_is_per_family_and_excludes_unselected_contexts(boards, tmp_path):
    source, target, _ = boards
    pairs = [make_pair(source, target, pair_id="a"),
             make_pair(source, target, pair_id="b", context_rank=1),
             make_pair(source, target, pair_id="c", layout="other", board_rank=1)]
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in pairs))
    assert [row["pair_id"] for row in load_plan(path, "evaluation", 1, 1)] == ["a"]
    assert [row["pair_id"] for row in load_plan(path, "evaluation", 2, 1)] == ["a", "b"]
    assert len(load_plan(path, "evaluation", 2, 2)) == 3


def test_plan_rejects_one_layout_in_development_and_evaluation(boards, tmp_path):
    source, target, _ = boards
    training = {**make_pair(source, target, pair_id="a"), "split": "development"}
    evaluation = make_pair(source, target, pair_id="b")
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in (training, evaluation)))
    with pytest.raises(ValueError, match="more than one split"):
        load_plan(path, "evaluation", 1, 1)


def test_replay_distinguishes_noop_from_nontrivial_cycle(boards):
    source, target, _ = boards
    pair = make_pair(source, target)
    noop = replay_trial(pair, make_trial(pair, [0, 3, 3]))
    assert noop["no_op_count"] == 1 and not noop["cycle_detected"]
    assert noop["first_hit"] == 3 and not noop["task_solved"]
    cycle = replay_trial(pair, make_trial(pair, [3, 2, 3, 3]))
    assert cycle["no_op_count"] == 0 and cycle["cycle_detected"]
    assert cycle["first_cycle_length"] == 2 and cycle["first_hit"] == 4


def test_terminal_and_target_are_recorded_separately(boards):
    source, far, solved = boards
    terminal_target = make_pair(source, solved)
    outcome = replay_trial(terminal_target, make_trial(terminal_target, [3, 1]))
    assert outcome["task_solved"] and outcome["exact_hit"] and outcome["stop_reason"] == "exact_target"
    different_target = make_pair(source, far)
    outcome = replay_trial(different_target, make_trial(different_target, [3, 1]))
    assert outcome["task_solved"] and not outcome["exact_hit"] and outcome["stop_reason"] == "task_terminal"


def test_replay_rejects_tampered_trace_and_action_after_hit(boards):
    source, target, _ = boards
    pair = make_pair(source, target)
    trial = make_trial(pair, [0, 3, 3])
    trial["no_op_count"] = 0
    with pytest.raises(ValueError, match="no_op_count"):
        replay_trial(pair, trial)
    later = make_trial(pair, [3, 3, 2])
    with pytest.raises(ValueError, match="after a stopping"):
        replay_trial(pair, later)


def test_corrupt_shard_is_missing_not_a_policy_failure(boards, tmp_path):
    source, target, _ = boards
    pair = make_pair(source, target)
    path = tmp_path / "evaluation/12x12x4/cllp8" / f"{pair['layout_id']}.json"
    path.parent.mkdir(parents=True)
    trial = make_trial(pair, [3, 3])
    shard = {"schema_version": 1, "status": "complete", "model_id": "cllp8", "family": "12x12x4",
             "source_family": "real", "split": "evaluation", "layout_id": pair["layout_id"],
             "planned_pair_ids": ["p1"], "contract_sha256": "known", "trials": [trial]}
    path.write_text(json.dumps(shard))
    observed, errors = read_shards([pair], tmp_path, ["cllp8", "cllp32"], "evaluation")
    assert len(observed) == 1 and errors == []
    shard["trials"][0]["no_op_count"] = 100
    path.write_text(json.dumps(shard))
    observed, errors = read_shards([pair], tmp_path, ["cllp8", "cllp32"], "evaluation")
    assert observed == {} and len(errors) == 1 and "no_op_count" in errors[0]["reason"]


def stub(hit):
    return {"exact_hit": hit, "first_hit": 1 if hit else None, "attempted_actions": 1 if hit else 32,
            "no_op_count": 0, "stop_reason": "exact_target" if hit else "action_limit", "task_solved": False,
            "final_state": ["state"], "final_player": [1, 1], "source_player": [1, 1], "target_player": [1, 2],
            "cycle_detected": False, "first_cycle_length": None, "cycles": [], "trace_file": "trace.json"}


def test_board_weighting_missing_bounds_and_paired_denominators(boards):
    source, target, _ = boards
    plan = [make_pair(source, target, pair_id="a"), make_pair(source, target, pair_id="b"),
            make_pair(source, target, pair_id="c", layout="layout2", board_rank=1)]
    observed = {("cllp8", "a"): stub(True), ("cllp8", "b"): stub(True), ("cllp8", "c"): stub(False),
                ("cllp32", "a"): stub(True), ("cllp32", "b"): stub(True)}
    _, summary, _, paired = analyse(plan, observed, ["cllp8", "cllp32"], replicates=100)
    def select(model):
        return next(row for row in summary if row["source_family"] == "all" and row["model_id"] == model
                    and row["k"] == 1 and row["budget"] == 1)
    assert select("cllp8")["p_hat"] == .5  # (2/2 + 0/1) / 2, not 2/3.
    assert select("cllp32")["p_hat"] is None
    assert select("cllp32")["bound_low"] == .5 and select("cllp32")["bound_high"] == 1
    assert select("cllp32")["available_board_mean"] == 1 and select("cllp32")["observed_layouts"] == 1
    diff = next(row for row in paired if row["source_family"] == "all" and row["k"] == row["budget"] == 1)
    assert diff["difference"] is None and diff["missing_pairs"] == 1 and diff["paired_pairs"] == 2


def test_csv_handles_missing_and_complete_rows(tmp_path):
    path = tmp_path / "cases.csv"
    write_csv(path, [{"pair_id": "missing", "status": "missing"},
                     {"pair_id": "done", "status": "complete", "final_state": ["#@#"]}])
    rows = list(csv.DictReader(path.open()))
    assert rows[0]["final_state"] == "" and json.loads(rows[1]["final_state"]) == ["#@#"]


def test_unavailable_offsets_and_missing_context_are_not_policy_failures(boards, tmp_path):
    source, target, _ = boards
    plan = [make_pair(source, target)]
    unavailable = [{"family": "12x12x4", "source_family": "real", "split": "evaluation",
                    "layout_id": plan[0]["layout_id"], "board_rank": 0, "context_rank": 0,
                    "k": 2, "reason": "beyond_trajectory_end"},
                   {"family": "12x12x4", "source_family": "real", "split": "evaluation",
                    "layout_id": plan[0]["layout_id"], "board_rank": 0, "context_rank": 1,
                    "k": None, "reason": "no_additional_distinct_context"}]
    path = tmp_path / "unavailable_offsets.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in unavailable))
    selected = load_unavailable(path, "evaluation", 2, 1)
    observed = {("cllp8", "p1"): stub(True)}
    _, summary, _, _ = analyse(plan, observed, ["cllp8"], unavailable=selected, replicates=100)
    row = next(item for item in summary if item["source_family"] == "all" and item["k"] == 2
               and item["budget"] == 1)
    assert row["frozen_layouts"] == 1 and row["eligible_layouts"] == 0
    assert row["unavailable_target_slots"] == 2 and row["missing_contexts"] == 1
    assert row["planned"] == row["missing"] == 0 and row["p_hat"] is None


def test_runner_plan_requires_same_pair_prefix_and_model(boards):
    source, target, _ = boards
    plan = [make_pair(source, target)]
    frozen = {"split": "evaluation", "layout_limit_per_family": 1, "contexts_per_layout": 1,
              "model_ids": ["cllp8"], "planned_shards": [{"family": "12x12x4", "layout_id": plan[0]["layout_id"],
                                                          "model_id": "cllp8", "pair_ids": ["p1"]}]}
    validate_frozen_plan(frozen, plan, ["cllp8"], "evaluation", 1, 1)
    with pytest.raises(ValueError, match="prefix"):
        validate_frozen_plan(frozen, plan, ["cllp8"], "evaluation", 2, 1)
    frozen["planned_shards"][0]["pair_ids"] = ["different"]
    with pytest.raises(ValueError, match="exactly match"):
        validate_frozen_plan(frozen, plan, ["cllp8"], "evaluation", 1, 1)


def test_plot_handles_completely_missing_family(boards, tmp_path):
    source, target, _ = boards
    first = make_pair(source, target)
    second = {**make_pair(source, target, pair_id="p2", layout="another"),
              "family": "12x12x10", "source_family": "synthetic"}
    _, summary, _, _ = analyse([first, second], {("cllp8", "p1"): stub(True)}, ["cllp8"], replicates=100)
    make_plot(summary, tmp_path, ["cllp8"])
    assert all((tmp_path / f"frozen_policy_reachability.{ext}").is_file() for ext in ("pdf", "svg", "png"))
