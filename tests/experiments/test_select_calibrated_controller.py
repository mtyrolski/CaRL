"""Controller selection is determined by calibration boards and fixed ties."""

import pytest

from scripts.select_calibrated_controller import choose
from scripts.select_calibrated_controller import score


def test_success_is_averaged_per_board_and_infrastructure_is_excluded():
    records = [{"record_id": "a", "layout_id": "first"}, {"record_id": "b", "layout_id": "first"},
               {"record_id": "c", "layout_id": "second"}, {"record_id": "d", "layout_id": "second"}]

    def result(identifier, status, achieved):
        return {"record_id": identifier,
                "result": {"status": status, "achieved": achieved, "steps": [{"no_op": True}],
                           "actions": [0], "model_calls": 1},
                "raw_policy_responses": [{"fence_removed": False}]}

    rows = [result("a", "achieved", True), result("b", "action_limit", False),
            result("c", "api_error", False), result("d", "achieved", True)]
    measured = score(records, rows)
    assert measured["board_mean_success"] == .75
    assert measured["completed_for_success"] == 3
    assert measured["statuses"]["api_error"] == 1
    assert measured["no_ops"] == measured["primitive_actions"] == 4


def test_ties_prefer_fewer_noops_then_smaller_model():
    scores = {"05b": {"board_mean_success": .25, "no_op_fraction": .5},
              "15b": {"board_mean_success": .25, "no_op_fraction": .25}}
    assert choose(scores) == "15b"
    scores["15b"]["no_op_fraction"] = .5
    assert choose(scores) == "05b"


def test_order_or_missing_cases_cannot_choose_controller():
    with pytest.raises(ValueError, match="same ordered"):
        score([{"record_id": "a", "layout_id": "x"}], [])
