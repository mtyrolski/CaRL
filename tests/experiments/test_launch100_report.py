"""Coverage and paired statistics must never turn missing jobs into failed observations."""

import json
from pathlib import Path

import pytest

from scripts.launch100.report import paired, run_rows, summaries, wilson
from scripts.launch100.runtime import job_key


def test_missing_jobs_stay_null_and_pairs_keep_their_denominator(tmp_path: Path) -> None:
    plan = {"freeze_id": "fixture", "node_budget": 500, "action_budget": 4000, "seed": 11,
            "methods": ["relative_u", "absolute_d"],
            "cohort": [{"id": name, "state_hash": name} for name in ("first", "second")]}
    for index, (board, method, solved) in enumerate([
        (plan["cohort"][0], "relative_u", True),
        (plan["cohort"][0], "absolute_d", False),
        (plan["cohort"][1], "relative_u", False),
    ]):
        key = job_key(plan, board, method)
        directory = tmp_path / "jobs" / key
        directory.mkdir(parents=True)
        (directory / "case.json").write_text(json.dumps({
            "job_key": key, "freeze_id": "fixture", "board_id": board["id"], "method": method,
            "complete": True, "solution_replay_verified": solved, "all_actions_replay_verified": True,
            "outcome": {"solved": solved, "action_path": [1] if solved else [], "attempted_actions": index + 1,
                        "expansions": 1, "accepted": 1, "rejected": 0}, "metrics": {},
        }))
    rows = run_rows(plan, tmp_path, {})
    assert len(rows) == 4 and rows[-1]["solved"] is None
    assert rows[-1]["attempted_actions"] is None and rows[-1]["settled_usd"] is None
    relative, absolute = summaries(rows)
    assert relative["solve_rate"] == 0.5 and relative["denominator"] == 2
    assert absolute["solve_rate"] == 0 and absolute["denominator"] == 1
    assert absolute["expected"] == 2 and absolute["full_cohort_high"] == 0.5
    assert paired(rows) == {"complete_pairs": 1, "both": 0, "relative_only": 1,
                            "absolute_only": 0, "neither": 0, "delta_pp": 100.0}


def test_solved_record_requires_verified_replay(tmp_path: Path) -> None:
    board = {"id": "first", "state_hash": "first"}
    plan = {"freeze_id": "fixture", "node_budget": 500, "action_budget": 4000, "seed": 11,
            "methods": ["relative_u"], "cohort": [board]}
    key = job_key(plan, board, "relative_u")
    directory = tmp_path / "jobs" / key
    directory.mkdir(parents=True)
    (directory / "case.json").write_text(json.dumps({
        "job_key": key, "freeze_id": "fixture", "board_id": "first", "method": "relative_u",
        "complete": True, "solution_replay_verified": False,
        "outcome": {"solved": True}, "metrics": {},
    }))
    with pytest.raises(ValueError, match="verified replay"):
        run_rows(plan, tmp_path, {})
    lower, upper = wilson(0, 100)
    assert lower == 0 and upper is not None and 0.036 < upper < 0.038
    assert wilson(0, 0) == (None, None)
