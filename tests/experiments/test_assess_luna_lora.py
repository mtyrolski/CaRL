"""Offline checks of missing cases, paired estimates and exact Luna replay."""

import csv
import json
import sys
from dataclasses import asdict

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.luna_intent_rollout import rollout
from scripts.assess_luna_lora import main
from scripts.assess_luna_lora import prepare_cases
from scripts.assess_luna_lora import read_rollouts
from scripts.assess_luna_lora import summarise
from scripts.assess_luna_lora import validate_plan
from scripts.assess_luna_lora import write_results

CONTROLLER_HASH = "a" * 64
INTENT = {"effects": [{"predicate": "box_at", "row": 1, "col": 4}], "preserve": [], "instruction": None}


def board(number=0):
    rows = ["#######", "#@-$-.#", "#-----#", "#-----#", "#-----#", "#-----#", "#######"]
    if number:
        for bit, (row, col) in enumerate(((4, 1), (4, 2), (4, 3))):
            if number & (1 << bit):
                chars = list(rows[row])
                chars[col] = "#"
                rows[row] = "".join(chars)
    return rows


def case(number, arm, *, status="valid_intent", scale=0.25, intent=INTENT, rows=None):
    rows = board(number) if rows is None else rows
    status_data = {
        "valid_intent": (json.dumps(intent), intent, True, []),
        "invalid_json": ("not json", None, False, ["bad JSON"]),
        "invalid_intent": ("null", None, True, ["intent is not an object"]),
        "not_attempted": (None, None, None, []),
        "infrastructure_error": (None, None, None, []),
    }
    raw, parsed, valid, errors = status_data[status]
    return {
        "record_id": f"{number}-{arm}",
        "source_record_id": f"source-{number}",
        "context_id": f"context-{number}",
        "layout_id": canonical_layout_id(from_rows(rows)),
        "family": "toy",
        "arm": arm,
        "requested_scale": scale,
        "prediction_status": status,
        "raw_output": raw,
        "parsed": parsed,
        "json_valid": valid,
        "validation_errors": errors,
        "start_observation": rows
    }


def saved_rollout(plan, answers, *, attempt=1, cap=3):
    state = from_rows(plan["start_observation"])
    iterator = iter(answers)

    def decide(_prompt):
        answer = next(iterator)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps({"action": answer})

    result = rollout(environment(state), state, plan["parsed"], decide, cap=cap, task_goal="Solve the board")
    return {
        "record_id": plan["record_id"],
        "context_id": plan["context_id"],
        "attempt": attempt,
        "controller_hash": CONTROLLER_HASH,
        "duration_seconds": 1.0,
        "result": asdict(result)
    }


def test_invalid_missing_and_api_errors_have_explicit_denominators():
    plan = [
        case(0, "unscaled"),
        case(0, "scaled", status="invalid_json"),
        case(1, "unscaled"),
        case(1, "scaled"),
        case(2, "unscaled", status="not_attempted"),
        case(2, "scaled", status="invalid_intent")
    ]
    results = [saved_rollout(plan[0], ["RIGHT", "RIGHT"]), saved_rollout(plan[2], ["LEFT", OSError("offline")])]
    cases = prepare_cases(plan, results, controller_hash=CONTROLLER_HASH, action_cap=3, expected_attempts=1)
    summary, curves, paired = summarise(cases, action_cap=3, bootstrap_samples=50)
    by_arm = {row["arm"]: row for row in summary}
    assert by_arm["scaled"]["planned_cases"] == 3
    assert by_arm["scaled"]["assessed_cases"] == 2    # Invalid predictions are measured failures.
    assert by_arm["scaled"]["not_attempted_cases"] == 1
    assert by_arm["scaled"]["board_mean"] == 0
    assert by_arm["scaled"]["assessed_fraction_of_planned"] == 2 / 3
    assert by_arm["scaled"]["planned_fraction_lower_bound"] == 0
    assert by_arm["scaled"]["planned_fraction_upper_bound"] == 1 / 3
    assert by_arm["unscaled"]["planned_cases"] == 3
    assert by_arm["unscaled"]["assessed_cases"] == 1
    assert by_arm["unscaled"]["infrastructure_cases"] == 1
    assert by_arm["unscaled"]["not_attempted_cases"] == 1
    assert by_arm["unscaled"]["planned_fraction_lower_bound"] == 1 / 3
    assert by_arm["unscaled"]["planned_fraction_upper_bound"] == 1
    assert by_arm["unscaled"]["all_saved_primitive_actions"] == 3
    assert by_arm["unscaled"]["all_saved_no_ops"] == 1
    assert by_arm["unscaled"]["all_saved_model_calls"] == 4
    curve = {row["budget"]: row for row in curves if row["arm"] == "unscaled"}
    assert [curve[i]["board_mean"] for i in range(4)] == [0, 0, 1, 1]
    assert paired[0]["paired_contexts"] == paired[0]["paired_boards"] == 1
    assert paired[0]["bootstrap_low"] is None


def test_paired_bootstrap_resamples_boards_and_preserves_zeros(tmp_path):
    plan = [case(i, arm) for i in range(6) for arm in ("unscaled", "scaled")]
    assert len({row["layout_id"] for row in plan}) == 6
    results = []
    for row in plan:
        answers = ["RIGHT", "RIGHT"] if row["arm"] == "scaled" else ["LEFT", "LEFT", "LEFT"]
        results.append(saved_rollout(row, answers))
    cases = prepare_cases(plan, results, controller_hash=CONTROLLER_HASH, action_cap=3, expected_attempts=1)
    summary, curves, paired = summarise(cases, action_cap=3, bootstrap_samples=100)
    assert {row["arm"]: row["board_mean"] for row in summary} == {"scaled": 1, "unscaled": 0}
    assert paired[0]["paired_boards"] == 6
    assert paired[0]["right_minus_left_board_mean"] == 1
    assert paired[0]["bootstrap_low"] == paired[0]["bootstrap_high"] == 1
    assert paired[0]["right_minus_left_actions_board_mean"] == -1
    assert paired[0]["actions_bootstrap_low"] == paired[0]["actions_bootstrap_high"] == -1
    assert paired[0]["right_minus_left_calls_board_mean"] == -1
    write_results(tmp_path / "complete", cases, summary, curves, paired, {"test": True})
    assert (tmp_path / "complete" / "intent_achievement.svg").is_file()


def test_intent_distributions_and_goal_progress_use_replayed_endpoints(tmp_path):
    goal_intent = {
        "effects": [{
            "predicate": "box_at",
            "row": 1,
            "col": 5
        }, {
            "predicate": "boxes_on_goals_at_least",
            "count": 1
        }],
        "preserve": [],
        "instruction": None
    }
    plan = [case(0, "unscaled"), case(0, "scaled", intent=goal_intent)]
    results = [saved_rollout(plan[0], ["RIGHT", "RIGHT"]), saved_rollout(plan[1], ["RIGHT", "RIGHT", "RIGHT"])]
    cases = prepare_cases(plan, results, controller_hash=CONTROLLER_HASH, action_cap=3, expected_attempts=1)
    summary, curves, paired = summarise(cases, action_cap=3, bootstrap_samples=5)
    by_arm = {row["arm"]: row for row in summary}
    scaled = by_arm["scaled"]
    assert scaled["intent_type_counts"] == {"box_at & boxes_on_goals_at_least": 1}
    assert scaled["effect_predicate_count_distribution"] == {"2": 1}
    assert scaled["preserve_predicate_count_distribution"] == {"0": 1}
    assert scaled["observed_endpoint_goal_increases"] == 1
    assert scaled["observed_endpoint_goal_delta_mean"] == 1
    assert scaled["successful_endpoint_goal_delta_mean"] == 1
    assert scaled["complete_task_solved_attempts"] == 1
    assert by_arm["unscaled"]["successful_endpoint_goal_delta_mean"] == 0
    assert paired[0]["goal_delta_paired_boards"] == 1
    assert paired[0]["right_minus_left_goal_delta_board_mean"] == 1
    output = tmp_path / "progress"
    write_results(output, cases, summary, curves, paired, {"test": True})
    with (output / "intent_distribution.csv").open(newline="") as stream:
        distributions = list(csv.DictReader(stream))
    assert any(row["value"] == "2" and row["count"] == "1" and row["valid_intents"] == "1"
               for row in distributions
               if row["distribution"] == "effect_predicate_count_distribution")
    with (output / "summary.csv").open(newline="") as stream:
        csv_summary = next(row for row in csv.DictReader(stream) if row["arm"] == "scaled")
    assert json.loads(csv_summary["effect_predicate_count_distribution"]) == {"2": 1}


def test_goal_decreases_and_infrastructure_progress_remain_separate():
    negative_board = ["#######", "#@-*--#", "#--$-.#", "#-----#", "#-----#", "#-----#", "#######"]
    plan = [case(0, "scaled", rows=negative_board), case(0, "unscaled", status="invalid_json", rows=negative_board)]
    result = saved_rollout(plan[0], ["RIGHT", "RIGHT"])
    cases = prepare_cases(plan, [result], controller_hash=CONTROLLER_HASH, action_cap=3, expected_attempts=1)
    summary, _, paired = summarise(cases, action_cap=3, bootstrap_samples=5)
    scaled = next(row for row in summary if row["arm"] == "scaled")
    assert scaled["observed_endpoint_goal_decreases"] == 1
    assert scaled["successful_endpoint_goal_delta_mean"] == -1
    assert paired[0]["goal_delta_paired_boards"] == 0
    assert paired[0]["right_minus_left_goal_delta_board_mean"] is None

    partial_board = ["#######", "#@-$-.#", "#--$-.#", "#-----#", "#-----#", "#-----#", "#######"]
    walk_intent = {"effects": [{"predicate": "player_at", "row": 5, "col": 1}], "preserve": [], "instruction": None}
    partial = [
        case(0, "scaled", rows=partial_board, intent=walk_intent),
        case(0, "unscaled", rows=partial_board, status="not_attempted")
    ]
    error_result = saved_rollout(partial[0], ["RIGHT", "RIGHT", "RIGHT", OSError("offline")], cap=4)
    cases = prepare_cases(partial, [error_result], controller_hash=CONTROLLER_HASH, action_cap=4, expected_attempts=1)
    summary, _, _ = summarise(cases, action_cap=4, bootstrap_samples=5)
    scaled = next(row for row in summary if row["arm"] == "scaled")
    assert scaled["infrastructure_attempts"] == 1
    assert scaled["infrastructure_endpoint_goal_delta_mean"] == 1
    assert scaled["infrastructure_model_calls"] == 4
    assert scaled["observed_endpoint_attempts"] == 0
    assert scaled["observed_endpoint_goal_delta_mean"] is None


def test_partial_repeats_do_not_become_failures():
    plan = [case(0, "scaled"), case(0, "unscaled", status="not_attempted")]
    result = saved_rollout(plan[0], ["RIGHT", "RIGHT"])
    cases = prepare_cases(plan, [result], controller_hash=CONTROLLER_HASH, action_cap=3, expected_attempts=2)
    summary, _, _ = summarise(cases, action_cap=3, bootstrap_samples=5)
    assert cases[0]["case_status"] == "incomplete_attempts"
    scaled = next(row for row in summary if row["arm"] == "scaled")
    assert scaled["assessed_cases"] == 0
    assert scaled["board_mean"] is None
    assert scaled["all_saved_model_calls"] == 2


def test_plan_and_replay_fail_closed():
    plan = [case(0, "scaled"), case(0, "unscaled", status="not_attempted")]
    result = saved_rollout(plan[0], ["RIGHT", "RIGHT"])
    with pytest.raises(ValueError, match="Duplicate planned"):
        validate_plan(plan + [plan[0]])
    with pytest.raises(ValueError, match="controller hash"):
        prepare_cases(plan, [result], controller_hash="b" * 64, action_cap=3, expected_attempts=1)
    with pytest.raises(ValueError, match="Duplicate physical attempt"):
        prepare_cases(plan, [result, result], controller_hash=CONTROLLER_HASH, action_cap=3, expected_attempts=1)
    result["result"]["actions"][0] = 2
    with pytest.raises(ValueError, match="differs from the simulator"):
        prepare_cases(plan, [result], controller_hash=CONTROLLER_HASH, action_cap=3, expected_attempts=1)
    false_claim = {**plan[0], "parsed": {"effects": [], "preserve": []}}
    with pytest.raises(ValueError, match="status or parse"):
        validate_plan([false_claim, plan[1]])
    with pytest.raises(ValueError, match="both arms"):
        validate_plan([plan[0]])


def test_plot_and_tables_keep_unassessed_values_empty(tmp_path):
    plan = [case(0, "scaled", status="not_attempted"), case(0, "unscaled", status="not_attempted")]
    cases = prepare_cases(plan, [], controller_hash=CONTROLLER_HASH, action_cap=3, expected_attempts=1)
    summary, curves, paired = summarise(cases, action_cap=3, bootstrap_samples=5)
    output = tmp_path / "assessment"
    write_results(output, cases, summary, curves, paired, {"test": True})
    assert all((output / f"intent_achievement.{extension}").is_file() for extension in ("pdf", "svg", "png"))
    assert (output / "summary.csv").is_file() and (output / "paired.csv").is_file()
    data = json.loads((output / "summary.json").read_text())
    assert data["summary"][0]["planned_cases"] == 1
    assert data["summary"][0]["board_mean"] is None
    with pytest.raises(FileExistsError):
        write_results(output, cases, summary, curves, paired, {"test": True})


def test_missing_and_interrupted_rollout_file_are_reported(tmp_path):
    path = tmp_path / "results.jsonl"
    rows, audit = read_rollouts(path)
    assert rows == [] and audit["present"] is False
    path.write_bytes(b'{"record_id":"one"}\n{"record_id":')
    rows, audit = read_rollouts(path)
    assert rows == [{"record_id": "one"}]
    assert audit["interrupted_tail_bytes"] > 0 and audit["interrupted_tail_sha256"] is not None
    path.write_bytes(b'broken\n{"record_id":"one"}\n')
    with pytest.raises(ValueError, match="Invalid saved rollout"):
        read_rollouts(path)


def test_cli_saves_full_plan_if_controller_never_started(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.jsonl"
    plan = [case(0, "scaled"), case(0, "unscaled", status="not_attempted")]
    plan_path.write_text("".join(json.dumps(row) + "\n" for row in plan))
    output = tmp_path / "assessment"
    monkeypatch.setattr(sys, "argv", [
        "assess_luna_lora", "--plan",
        str(plan_path), "--rollouts",
        str(tmp_path / "not-created.jsonl"), "--output",
        str(output), "--controller-hash", CONTROLLER_HASH, "--action-cap", "3"
    ])
    main()
    result = json.loads((output / "summary.json").read_text())
    assert result["metadata"]["rollouts_file"]["present"] is False
    assert all(row["planned_cases"] == 1 and row["assessed_cases"] == 0 for row in result["summary"])
    assert all(row["board_mean"] is None for row in result["summary"])
