"""Verify the synthetic transfer adapter's shared planner semantics offline."""

from dataclasses import asdict

import pytest

from carl.icl.transfer.discobench import (
    Intent, Snapshot, SyntheticQueryExecutor, public_task_pin, run_synthetic_smoke,
)


@pytest.mark.parametrize("mode", ["direct", "one-step", "full-search"])
def test_synthetic_pilot_replays_through_the_shared_planner(mode: str) -> None:
    evidence = run_synthetic_smoke(mode)
    result = evidence["result"]
    assert evidence["provenance"] == "synthetic_contract_smoke"
    assert evidence["benchmark_score"] is None
    assert result["solved"] and result["action_path"] == [0, 1, 2]
    assert result["accounting"]["physical_requests"] == 0
    assert result["accounting"]["attempted_simulator_actions"] >= 3
    state = Snapshot(2)
    for point in result["action_path"]:
        state = state.step(point)
    assert result["endpoint"] == SyntheticQueryExecutor.snapshot_record(state)
    assert result["endpoint"]["samples"] == [[0, 4], [1, 1], [2, 0]]


def test_snapshot_forks_keep_the_objective_private() -> None:
    ancestor = Snapshot(2).step(0)
    fork = ancestor.step(1)
    assert fork == ancestor.step(1)
    assert ancestor.step(0) is ancestor
    assert ancestor.step(2) != fork
    state = SyntheticQueryExecutor.initial_state(ancestor, ())
    observation = SyntheticQueryExecutor.observation(state)
    assert asdict(observation) == {"samples": ({"point": 0, "score": 4},)}
    assert "optimum" not in SyntheticQueryExecutor.snapshot_record(ancestor)
    assert SyntheticQueryExecutor.validate_intent(observation, Intent(2, 4)) == Intent(2, 4)
    with pytest.raises(ValueError, match="preservation threshold"):
        SyntheticQueryExecutor.validate_intent(observation, Intent(2, 3))


def test_synthetic_budget_retains_a_replayable_partial_plan() -> None:
    result = run_synthetic_smoke("direct", action_budget=2)["result"]
    assert not result["task_solved"] and result["finish_reason"] == "action_budget"
    assert result["action_path"] == [0, 1]
    assert result["accounting"]["attempted_simulator_actions"] == 2
    result = run_synthetic_smoke("direct", request_budget=2)["result"]
    assert not result["task_solved"] and result["finish_reason"] == "api_budget"
    assert result["action_path"] == [0, 1]
    assert result["accounting"]["logical_requests"] == 2


def test_public_pin_identifies_disjoint_instance_ids_and_exact_source() -> None:
    pin = public_task_pin()
    assert pin["requires_python"] == ">=3.12,<4.0"
    assert len(pin["revision"]) == len(pin["config_git_blob"]) == 40
    assert f"/blob/{pin['revision']}/" in pin["public_config_url"]
    assert pin["small_meta_train_instance"] == "Ackley1d"
    assert pin["small_meta_test_instance"] == "Ackley2d"
