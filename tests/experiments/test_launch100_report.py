"""Coverage and paired statistics must never turn missing jobs into failed observations."""

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.launch100.report import compatible, costs, figure, paired, run_rows, summaries, wilson
from scripts.launch100.runtime import job_key


def test_missing_jobs_stay_null_and_pairs_keep_their_denominator(tmp_path: Path) -> None:
    plan = {"freeze_id": "fixture", "node_budget": 500, "seed": 11,
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
    assert rows[0]["low_level_nodes"] == 1 and rows[0]["high_level_expansions"] == 1
    assert rows[-1]["attempted_actions"] is None and rows[-1]["settled_usd"] is None
    summary = {row['method']: row for row in summaries(rows)}
    relative, absolute = summary['relative_u'], summary['absolute_d']
    assert relative["solve_rate"] == 0.5 and relative["denominator"] == 2
    assert absolute["solve_rate"] == 0 and absolute["denominator"] == 1
    assert absolute["expected"] == 2 and absolute["full_cohort_high"] == 0.5
    assert paired(rows) == {"complete_pairs": 1, "both": 0, "relative_only": 1,
                            "absolute_only": 0, "neither": 0, "delta_pp": 100.0}
    other_budget = [{**row, 'node_budget': 250} for row in rows]
    by_budget = summaries(rows + other_budget)
    assert len(by_budget) == 4
    plot = figure(by_budget, plan)
    assert {tuple(trace['x']) for trace in plot.to_dict()['data']} == {(250, 500)}
    missing_budget = [{**row, 'node_budget': 375, 'solved': None} for row in rows]
    plot = figure(summaries(rows + other_budget + missing_budget), plan)
    assert all(trace['y'][1] is None for trace in plot.to_dict()['data'])


def test_budget_reports_cannot_combine_different_protocols() -> None:
    plan = {'provider': 'openai', 'methods': ['relative_u'], 'seed': 11, 'cohort': [{'id': 'x'}],
            'asset_manifest_sha256': 'a', 'policy_fingerprint': 'p', 'source_hashes': {'search': 'v1'},
            'configs': {'relative_u': {'action_budget': 500, 'model': 'gpt-6-luna'}}}
    compatible(plan, {**plan, 'configs': {'relative_u': {'action_budget': 128, 'model': 'gpt-6-luna'}}})
    with pytest.raises(ValueError, match='configuration'):
        compatible(plan, {**plan, 'configs': {'relative_u': {'action_budget': 128, 'model': 'other'}}})


def test_solved_record_requires_verified_replay(tmp_path: Path) -> None:
    board = {"id": "first", "state_hash": "first"}
    plan = {"freeze_id": "fixture", "node_budget": 500, "seed": 11,
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


def test_costs_include_settled_usage_and_keep_unknown_reservations_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scripts.launch100.report.ROOT", tmp_path)
    config = tmp_path / "configs/icl/campaign_budget.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"ledger_path": "cost.sqlite3"}))
    with sqlite3.connect(tmp_path / "cost.sqlite3") as database:
        database.execute("CREATE TABLE attempts(tag TEXT, charged_nanos INTEGER, reserved_nanos INTEGER, usage_json TEXT)")
        database.executemany("INSERT INTO attempts VALUES(?,?,?,?)", [
            ("launch100:frozen:job", 1000000, 500000000, json.dumps({"input_tokens": 100, "output_tokens": 20})),
            ("launch100:frozen:job", None, 500000000, None),
            ("launch100:other:job", 9000000, 500000000, None),
        ])
    row = costs("frozen")["job"]
    assert row["paid_attempts"] == 2 and row["paid_usage_unknown_attempts"] == 1
    assert row["settled_usd"] == 0.001 and row["unsettled_reserved_usd"] == 0.5
    assert row["paid_input_tokens"] == 100 and row["paid_output_tokens"] == 20
