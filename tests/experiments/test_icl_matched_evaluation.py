"""Offline matched evaluation keeps budgets, replicate groups and censors intact."""
import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from carl.icl.evaluation.data import EvaluationConfig
from carl.icl.evaluation.matched import MatchedSweep, evaluate_matched, matched_tables
from carl.icl.types import JSONObject


def _budget(name: str, actions: int) -> JSONObject:
    return {
        "id": name,
        "limits": {"action_budget": actions, "max_expansions": 12, "wall_seconds": 10, "api": {"max_requests": 40}},
    }


def _instance(group: str, seed: int) -> JSONObject:
    return {
        "id": f"{group}-{seed}", "group": group, "seed": seed,
        "policy_fingerprint": "frozen-policy", "initial_fingerprint": f"snapshot-{group}",
    }


def _trial(
    budget: JSONObject, instance: JSONObject, mode: str, *, solved: bool, elapsed: float | None,
    success_seconds: float | None = None,
) -> JSONObject:
    return {
        "budget_id": budget["id"], "instance_id": instance["id"], "mode": mode,
        "status": "solved" if solved else "action_budget", "solved": solved, "elapsed_seconds": elapsed,
        "limits": budget["limits"], "artifact": f"{budget['id']}/{instance['id']}/{mode}.json",
        "accounting": {"attempted_simulator_actions": 3, "logical_requests": 4, "physical_requests": 0},
        "policy_fingerprint": instance["policy_fingerprint"], "initial_fingerprint": instance["initial_fingerprint"],
        "seed": instance["seed"], "cost_usd": 0, "success_seconds": success_seconds,
    }


@pytest.fixture(name="sweep_payload")
def planned_sweep_payload() -> JSONObject:
    """One board repeats three times while the other succeeds under the opposite method."""
    small, large = _budget("small", 4), _budget("large", 10)
    a1, a2, a3, b, c, d, e = [_instance(group, seed) for group, seed in (
        ("a", 1), ("a", 2), ("a", 3), ("b", 1), ("c", 1), ("d", 1), ("e", 1),
    )]
    trials = [
        _trial(large, a1, "direct", solved=True, elapsed=5, success_seconds=2),
        _trial(large, a2, "direct", solved=True, elapsed=3),
        _trial(large, a3, "direct", solved=True, elapsed=4),
        _trial(large, a1, "full-search", solved=False, elapsed=0.1),
        _trial(large, a2, "full-search", solved=False, elapsed=2),
        _trial(large, a3, "full-search", solved=False, elapsed=5),
        _trial(large, b, "direct", solved=False, elapsed=0.5),
        _trial(large, b, "full-search", solved=True, elapsed=4),
        _trial(large, c, "direct", solved=True, elapsed=None),
        _trial(large, d, "full-search", solved=True, elapsed=11),
        _trial(small, a1, "direct", solved=False, elapsed=3),
        _trial(small, a1, "full-search", solved=False, elapsed=2),
    ]
    return {
        "schema_version": 1, "domain": "sokoban", "execution_mode": "mock", "split_fingerprint": "fixture",
        "versions": {"carl": "test"}, "config": {"fixture": "sweep"}, "offline_compute_seconds": None,
        "adaptation_seconds": None, "modes": ["direct", "full-search"], "budgets": [small, large],
        "instances": [a1, a2, a3, b, c, d, e], "trials": trials,
    }


def _config() -> EvaluationConfig:
    return EvaluationConfig(run_dirs=["."], bootstrap_replicates=200, bootstrap_seed=73)


def test_balanced_pairs_and_right_censor_rule(sweep_payload: JSONObject) -> None:
    tables = matched_tables(MatchedSweep.model_validate(sweep_payload), _config())
    small = tables["comparisons"][0]
    large = tables["comparisons"][1]
    assert small["paired_trials"] == 1
    assert small["success_delta"] == 0
    assert small["success_delta_ci95_low"] is None
    assert large["paired_trials"] == 4 and large["paired_groups"] == 2
    assert large["success_delta"] == 0
    assert large["success_delta_ci95_low"] == -1 and large["success_delta_ci95_high"] == 1
    assert large["restricted_time_delta_seconds"] == -0.5
    assert large["missing_left_only"] == ["d-1"]
    assert large["missing_right_only"] == ["c-1"]
    assert large["missing_both"] == ["e-1"]
    lookup = {(row["budget_id"], row["instance_id"], row["mode"]): row for row in tables["trials"]}
    failure = lookup[("large", "a-1", "full-search")]
    assert failure["event_within_horizon"] is False and failure["elapsed_seconds"] == 0.1
    assert failure["restricted_time_to_success_seconds"] == 10 and failure["observed_censor_seconds"] == 0.1
    late = lookup[("large", "d-1", "full-search")]
    assert late["time_to_success_seconds"] == 11 and late["event_within_horizon"] is False
    assert late["observed_censor_seconds"] == 10
    unknown = lookup[("large", "c-1", "direct")]
    assert unknown["event_within_horizon"] is None and unknown["restricted_time_to_success_seconds"] is None


def test_missing_arm_and_time_keeps_paired_coverage(sweep_payload: JSONObject) -> None:
    missing_arm = {**sweep_payload, "trials": [row for row in sweep_payload["trials"] if row["mode"] == "direct"]}
    first = matched_tables(MatchedSweep.model_validate(missing_arm), _config())["comparisons"][0]
    assert first["paired_trials"] == 0 and first["success_delta"] is None
    assert first["missing_right_only"] == ["a-1"]

    sweep_payload["trials"][0].update(elapsed_seconds=None, success_seconds=None)
    large = matched_tables(MatchedSweep.model_validate(sweep_payload), _config())["comparisons"][1]
    assert large["paired_trials"] == 4 and large["timed_pairs"] == 3
    assert large["success_delta"] == 0

    sweep_payload["trials"] = []
    tables = matched_tables(MatchedSweep.model_validate(sweep_payload), _config())
    assert all(row["coverage"] == 0 and row["total_physical_requests"] is None for row in tables["coverage"])
    assert all(row["observed_censor_seconds"] is None for row in tables["trials"])


@pytest.mark.parametrize(("key", "value", "message"), [
    ("limits", {"action_budget": 10, "wall_seconds": 10, "max_expansions": 12, "api": {"max_requests": 20}}, "Actual limits differ"),
    ("seed", 13, "Mismatched trial seed"),
    ("policy_fingerprint", "different", "Mismatched trial policy_fingerprint"),
    ("initial_fingerprint", "different", "Mismatched trial initial_fingerprint"),
    ("accounting", {"attempted_simulator_actions": 11}, "Action budget exceeded"),
    ("finish_reason", "action_budget", "solved and finish_reason disagree"),
    ("status", "missing", "by omitting the trial"),
    ("success_seconds", 9, "cannot exceed elapsed_seconds"),
])
def test_reject_contradictory_records(sweep_payload: JSONObject, key: str, value: object, message: str) -> None:
    sweep_payload["trials"][0][key] = value
    with pytest.raises(ValidationError, match=message):
        MatchedSweep.model_validate(sweep_payload)


def test_reject_duplicate_trial_and_split_group(sweep_payload: JSONObject) -> None:
    sweep_payload["trials"].append(sweep_payload["trials"][0])
    with pytest.raises(ValidationError, match="Duplicate matched trial"):
        MatchedSweep.model_validate(sweep_payload)
    sweep_payload["trials"].pop()
    sweep_payload["instances"][1]["group"] = "other-family"
    with pytest.raises(ValidationError, match="share their bootstrap group"):
        MatchedSweep.model_validate(sweep_payload)


def test_export_repeatable_reports(tmp_path: Path, sweep_payload: JSONObject) -> None:
    source = tmp_path / "sweep.json"
    source.write_text(json.dumps(sweep_payload), encoding="utf-8")
    output = tmp_path / "offline"
    first = evaluate_matched(source, output, _config())
    data = first.read_bytes()
    assert evaluate_matched(source, output, _config()).read_bytes() == data
    result = json.loads(data)
    assert result["design"]["offline_compute_seconds"] is None
    assert result["design"]["adaptation_seconds"] is None
    assert result["source"]["sha256"] and result["bootstrap"] == {"seed": 73, "replicates": 200}
    with (output / "trials.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 28 and sum(row["status"] == "missing" for row in rows) == 16
    assert (output / "REPORT.md").is_file() and (output / "coverage.csv").is_file()
