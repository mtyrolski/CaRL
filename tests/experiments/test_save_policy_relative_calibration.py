"""Relative u derives solely from measured controller actions within a context."""

import pytest

from scripts.run_fixed_local_controller import evaluate_record
from scripts.save_policy_relative_calibration import calculate
from scripts.save_policy_relative_calibration import check_result
from scripts.save_policy_relative_calibration import validate_identities


def record(identifier, context, target):
    return {"record_id": identifier, "context_id": context, "phase": "calibration", "layout_id": context,
            "start_observation": ["#####", "#####", "#@$.#", "#####", "#####"], "task_goal": "Solve Sokoban.",
            "intent": {"effects": [{"predicate": "box_at", "row": 2, "col": target}], "preserve": [],
                       "instruction": None}, "k": 9999}


def outcome(identifier, status, actions, *, calls=1):
    return {"record_id": identifier, "result": {"status": status, "achieved": status == "achieved",
                                                  "model_calls": calls, "actions": [0] * actions}}


def test_one_success_gets_midpoint_and_never_uses_demonstration_offset(tmp_path):
    records = [record(f"{i}-{j}", f"context-{i}", j) for i in range(4) for j in range(3)]
    results = [outcome(row["record_id"], "achieved" if index == 4 else "action_limit", 1 if index == 4 else 16)
               for index, row in enumerate(records)]
    result = calculate(records, results, tmp_path)
    candidates = {row["record_id"]: row for ctx in result["contexts"] for row in ctx["candidates"]}
    assert candidates["1-1"]["u"] == .5
    assert candidates["1-1"]["mean_success_actions"] == 1
    assert (candidates["1-1"]["weight"], candidates["1-1"]["cdf_before"], candidates["1-1"]["cdf_at"]) == (1, 0, 1)
    assert sum(row["u"] is not None for row in candidates.values()) == 1
    assert all(row["attempts"] == 1 for row in candidates.values())


def test_ties_use_midpoints_and_failure_has_no_scale(tmp_path):
    records = [record(name, "same", index) for index, name in enumerate(("fast", "tie1", "tie2", "slow", "failed"))]
    results = [outcome("fast", "achieved", 1), outcome("tie1", "achieved", 2),
               outcome("tie2", "achieved", 2), outcome("slow", "achieved", 4),
               outcome("failed", "action_limit", 16)]
    candidates = {row["record_id"]: row for row in calculate(records, results, tmp_path)["contexts"][0]["candidates"]}
    assert {name: candidates[name]["u"] for name in candidates} == {
        "fast": .125, "tie1": .5, "tie2": .5, "slow": .875, "failed": None}
    assert (candidates["tie1"]["weight"], candidates["tie1"]["cdf_before"], candidates["tie1"]["cdf_at"]) == (.25, .25, .75)
    assert candidates["failed"]["weight"] is None


def test_missing_or_duplicate_attempts_are_not_invented():
    records = [record("a", "same", 0), record("b", "same", 1)]
    with pytest.raises(ValueError, match="exact frozen"):
        validate_identities(records, [outcome("a", "achieved", 1)])
    with pytest.raises(ValueError, match="distinct"):
        validate_identities(records, [outcome("a", "achieved", 1), outcome("a", "achieved", 1)])


def test_infrastructure_failure_does_not_produce_a_scale_label(tmp_path):
    records = [record("yes", "same", 0), record("unmeasured", "same", 1)]
    results = [outcome("yes", "achieved", 1), outcome("unmeasured", "api_error", 0)]
    context = calculate(records, results, tmp_path)["contexts"][0]
    assert context["complete"] is False
    assert all(row["u"] is None for row in context["candidates"])


def test_saved_rollout_is_checked_against_real_environment_and_raw_output():
    case = {**record("a", "same", 3), "family": "pilot"}

    def policy(prompt):
        return '{"action":"RIGHT"}'

    actual = evaluate_record(case, policy, 16)
    counts = check_result(case, actual)
    assert counts == {"executed_actions": 1, "model_calls": 1}
    actual["raw_policy_responses"][0]["normalised"] = '{"action":"UP"}'
    with pytest.raises(ValueError, match="normalisation"):
        check_result(case, actual)
