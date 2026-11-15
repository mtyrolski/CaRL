"""Planning never opens predictions and handles unavailable calibration honestly."""

import copy
import json
from pathlib import Path

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.policy_relative_lora import load_fit_data
from scripts import plan_luna_policy_followup_evaluation as planner
from tests.experiments.test_policy_relative_lora import BOARDS
from tests.experiments.test_policy_relative_lora import group


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def inputs(tmp_path):
    rows = group("train-a", "train", BOARDS[0]) + group("train-b", "train", BOARDS[1])
    rows += group("validation", "validation", BOARDS[2])
    calibration = tmp_path / "calibrated.jsonl"
    demo = tmp_path / "demo.jsonl"
    write_rows(calibration, rows)
    write_rows(demo, [{**row, "policy_measurements": None} for row in rows])
    board = ["######", "#@---#", "#-$--#", "#---.#", "#----#", "######"]
    context = {"record_id": "case", "context_id": "case", "layout_id": canonical_layout_id(from_rows(board)),
               "family": "test", "split": "test", "start_observation": board, "task_goal": "Put boxes on goals."}
    contexts = tmp_path / "test.jsonl"
    write_rows(contexts, [context])
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"state": "complete", "saved_attempts": 24, "planned_attempts": 24,
                                  "not_started": 0, "started_without_saved_result": 0}))
    return calibration, status, demo, contexts, rows


def test_full_scale_decision_uses_verified_train_validation_durations(tmp_path):
    calibration, status, demo, contexts, _ = inputs(tmp_path)
    output, predictions = tmp_path / "frozen", tmp_path / "predictions"
    decision = planner.freeze(calibration, status, demo, contexts, predictions, output, 2, 1, 1, True)
    assert decision["arms"] == ["D", "N", "T", "U", "R"]
    assert decision["absolute_levels"] == [1.0, 2.0]
    assert decision["relative_levels"] == [.25, .75]
    matrix = json.loads((output / "matrix.json").read_text())
    assert matrix["tiers"] == [.25, .75]
    for arm in ("D", "N", "R"):
        assert [slot["requested_input"] for slot in matrix["arms"][arm]] == [None, None]
    assert [slot["requested_input"] for slot in matrix["arms"]["T"]] == [1.0, 2.0]
    assert [slot["requested_input"] for slot in matrix["arms"]["U"]] == [.25, .75]
    assert all(not Path(slot["path"]).exists() for slots in matrix["arms"].values() for slot in slots)


def test_insufficient_variation_keeps_only_unscaled_measured_comparison(tmp_path):
    calibration, _, _, _, _ = inputs(tmp_path)
    data = load_fit_data(calibration)
    arms, levels, reason = planner.arm_decision(data, 3, 1, False)
    assert arms == ["D", "N"] and levels is None
    assert reason["scale_information"] is False


def test_missing_calibration_records_d_b_and_r_without_fabricating_measurements(tmp_path):
    calibration, status, demo, contexts, _ = inputs(tmp_path)
    calibration.unlink()
    status.unlink()
    output, predictions = tmp_path / "missing-plan", tmp_path / "predictions"
    decision = planner.freeze(calibration, status, demo, contexts, predictions, output, 4, 1, 1, True,
                              include_base=True)
    assert decision["arms"] == ["B", "D", "R"]
    assert decision["calibration_sha256"] is decision["calibration_status_sha256"] is None
    assert decision["calibration_inventory"] is decision["absolute_levels"] is None
    assert "missing" in decision["no_success_diagnostic"]
    matrix = json.loads((output / "matrix.json").read_text())
    assert set(matrix["arms"]) == {"B", "D", "R"}
    assert all(slot["requested_input"] is None for slots in matrix["arms"].values() for slot in slots)


def test_zero_success_is_explicit_and_retains_calibration_inventory(tmp_path):
    calibration, status, demo, contexts, rows = inputs(tmp_path)
    failures = copy.deepcopy(rows)
    for row in failures:
        row["policy_measurements"].update(attempts=[{"status": "action_limit", "primitive_actions_attempted": 8}] * 2,
                                          success_fraction=0, mean_success_actions=None, u=None, weight=None)
    write_rows(calibration, failures)
    decision = planner.freeze(calibration, status, demo, contexts, tmp_path / "predictions", tmp_path / "plan",
                              2, 1, 1, True)
    assert decision["arms"] == ["D", "R"] and decision["absolute_levels"] is None
    assert decision["calibration_inventory"]["contexts"] == 3
    assert "measured success" in decision["no_success_diagnostic"]


def test_preexisting_prediction_or_nonterminal_calibration_blocks_freezing(tmp_path):
    calibration, status, demo, contexts, _ = inputs(tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "U_slot0.jsonl").write_text("do not read")
    with pytest.raises(FileExistsError, match="already started"):
        planner.freeze(calibration, status, demo, contexts, predictions, tmp_path / "blocked", 2, 1, 1, True)
    (predictions / "U_slot0.jsonl").unlink()
    status.write_text(json.dumps({"state": "running"}))
    with pytest.raises(ValueError, match="terminal"):
        planner.freeze(calibration, status, demo, contexts, predictions, tmp_path / "blocked", 2, 1, 1, True)
