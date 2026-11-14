"""Keep incomplete API work distinct from measured policy failures."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import nbformat

from scripts.build_luna_intent_notebook import audited_calibration
from scripts.build_luna_intent_notebook import board_interval
from scripts.build_luna_intent_notebook import figures
from scripts.build_luna_intent_notebook import load_report
from scripts.build_luna_intent_notebook import read_ledger


def test_denominators_exclude_infrastructure_and_missing_cases(tmp_path):
    data = tmp_path / "data"
    results = tmp_path / "results"
    data.mkdir()
    results.mkdir()
    cases = [{"record_id": str(i), "context_id": str(i), "layout_id": str(i),
              "phase": "test", "family": "real_12x12x4"} for i in range(4)]
    (data / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cases))
    (tmp_path / "experiment_plan.json").write_text(json.dumps({"record_ids": [r["record_id"] for r in cases]}))
    statuses = ("achieved", "invalid_response", "api_error")
    attempts = [{**case, "cap": 4, "result": {"status": status, "model_calls": 1, "actions": [3] if i != 1 else []}}
                for i, (case, status) in enumerate(zip(cases, statuses))]
    (results / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in attempts))
    report = load_report(tmp_path)
    summary = report["summaries"][0]
    assert summary["planned"] == 4
    assert summary["completed"] == 2
    assert summary["successes"] == 1
    assert summary["infrastructure_failures"] == 1
    assert summary["invalid_responses"] == 1
    assert summary["not_started_or_unsaved"] == 1
    assert all(row["denominator"] == 2 and row["successes"] == 1 for row in report["curve"])
    figures(tmp_path, report)
    assert (tmp_path / "figures/success_by_action_budget.svg").exists()
    assert (tmp_path / "figures/successful_durations.csv").exists()


def test_duplicate_rollouts_are_reported_and_not_double_counted(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    case = {"record_id": "a", "context_id": "a", "layout_id": "a", "phase": "test", "family": "real_12x12x4"}
    (tmp_path / "data/records.jsonl").write_text(json.dumps(case) + "\n")
    row = {**case, "cap": 1, "result": {"status": "achieved", "model_calls": 1, "actions": [3]}}
    (tmp_path / "results/rollouts.jsonl").write_text((json.dumps(row) + "\n") * 2)
    report = load_report(tmp_path)
    assert not report["results"]
    assert "Duplicate rollout IDs" in report["warnings"][0]
    assert report["summaries"][0]["not_started_or_unsaved"] == 1


def test_zero_model_calls_are_not_measured_as_model_failures(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    case = {"record_id": "a", "context_id": "a", "layout_id": "a", "phase": "test", "family": "real_12x12x4"}
    row = {**case, "cap": 1, "result": {"status": "task_solved", "model_calls": 0, "actions": []}}
    (tmp_path / "data/records.jsonl").write_text(json.dumps(case) + "\n")
    (tmp_path / "results/rollouts.jsonl").write_text(json.dumps(row) + "\n")
    summary = load_report(tmp_path)["summaries"][0]
    assert summary["completed"] == 0
    assert summary["other_unassessed"] == 1
    assert summary["success_fraction"] is None


def test_board_bootstrap_requires_five_independent_layouts():
    data = [{"layout_id": str(i), "success": i % 2} for i in range(5)]
    assert board_interval(data[:4]) == (None, None)
    low, high = board_interval(data)
    assert low is not None and high is not None
    assert 0 <= low <= high <= 1


def test_shared_audit_is_split_by_variant_and_campaign_cost(tmp_path):
    audit_dir = tmp_path / "luna_requests"
    audit_dir.mkdir()
    structured = tmp_path / "structured"
    structured.mkdir()
    (structured / "experiment_plan.json").write_text(json.dumps({
        "audit_run": "..", "audit_tag_prefix": "structured:", "record_ids": []}))
    final = tmp_path / "final_response"
    final.mkdir()
    (final / "experiment_plan.json").write_text(json.dumps({
        "audit_run": "..", "audit_tag_prefix": "final:", "record_ids": []}))
    expanded = tmp_path / "expanded"
    expanded.mkdir()
    (expanded / "experiment_plan.json").write_text(json.dumps({
        "audit_run": "..", "audit_tag_prefix": "expanded:", "record_ids": []}))
    for i, (tag, charge) in enumerate((("calibration:1", "0.1"), ("structured:test:2", "0.2"),
                                       ("final:3", "0.4"), ("expanded:4", "0.8"))):
        (audit_dir / f"{i}.json").write_text(json.dumps({
            "tag": tag, "status": "settled", "conservative_charge_usd": charge}))
    original = load_report(tmp_path)
    second = load_report(structured)
    third = load_report(final)
    fourth = load_report(expanded)
    assert original["charge"] == "0.1"
    assert second["charge"] == "0.2"
    assert third["charge"] == "0.4"
    assert fourth["charge"] == "0.8"
    assert original["campaign_charge"] == second["campaign_charge"] == third["campaign_charge"] == fourth["campaign_charge"] == "1.5"
    assert [row["tag"] for row in second["audits"]] == ["structured:test:2"]


def test_ledger_preserves_uncertain_reservation_separate_from_charged_amount(tmp_path):
    path = tmp_path / "luna_intent_ledger.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE attempts (status TEXT, charged_nanos INTEGER, reserved_nanos INTEGER)")
        db.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
        db.execute("INSERT INTO attempts VALUES ('settled', 10000000, 600000000)")
        db.execute("INSERT INTO attempts VALUES ('unknown', NULL, 580000000)")
        db.execute("INSERT INTO metadata VALUES ('effective_limit_nanos', '2000000000')")
    ledger = read_ledger(tmp_path)
    assert ledger["attempts"] == 2
    assert ledger["known_charge_usd"] == "0.01"
    assert ledger["unknown_reserved_usd"] == "0.58"
    assert ledger["conservative_total_with_reservations_usd"] == "0.59"
    assert ledger["effective_limit_usd"] == "2"


def test_proven_concatenation_error_is_unassessed_without_replaying_final_action(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "luna_requests").mkdir()
    cases = [{"record_id": str(i), "context_id": str(i), "layout_id": str(i),
              "phase": "test", "family": "real_12x12x4"} for i in range(2)]
    (tmp_path / "data/records.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    raw = 'thinking {"action":"RIGHT"}'
    output = [{"type": "message", "phase": "commentary", "status": "completed",
               "content": [{"type": "output_text", "text": "thinking "}]},
              {"type": "message", "phase": "final_answer", "status": "completed",
               "content": [{"type": "output_text", "text": '{"action":"RIGHT"}'}]}]
    audit = {"tag": "0", "reservation_id": "audit1", "response": {"status": "completed", "output": output}}
    (tmp_path / "luna_requests/1.json").write_text(json.dumps(audit))
    rows = [{**case, "cap": 4, "result": {"status": "invalid_response", "model_calls": 1,
            "actions": [], "steps": [{"raw_response": raw if i == 0 else "not JSON"}]}}
            for i, case in enumerate(cases)]
    (tmp_path / "results/rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = load_report(tmp_path)
    assert list(report["parser_artifacts"]) == ["0"]
    assert report["parser_artifacts"]["0"]["audit_matches"][0]["final_action"] == "RIGHT"
    assert report["summaries"][0]["parser_artifacts_unassessed"] == 1
    assert report["summaries"][0]["completed"] == 1
    assert report["summaries"][0]["invalid_responses"] == 1
    assert report["results"]["0"]["result"]["actions"] == []


def test_parser_error_masks_the_entire_context_scale():
    cases = [{"record_id": str(i), "context_id": "same", "phase": "calibration"} for i in range(2)]
    results = {"0": {"result": {"status": "achieved", "model_calls": 1}},
               "1": {"result": {"status": "invalid_response", "model_calls": 1}}}
    calibration = {"contexts": [{"context_id": "same", "complete": True, "candidates": [
        {"record_id": "0", "successes": 1, "success_fraction": 1.0, "mean_success_actions": 1, "u": 0.5},
        {"record_id": "1", "successes": 0, "success_fraction": 0.0, "mean_success_actions": None, "u": None},
    ]}]}
    audited = audited_calibration(cases, results, {"1": {}}, calibration)
    assert all(not row["context_complete_after_audit"] and row["u"] is None for row in audited)
    assert audited[0]["successes"] == 1
    assert audited[1]["successes"] is None
    assert calibration["contexts"][0]["candidates"][0]["u"] == 0.5


def test_notebook_executes_offline_with_a_realistic_partial_run(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "luna_requests").mkdir()
    board = ["#####", "#@$.#", "#---#", "#---#", "#####"]
    final = ["#####", "#-@*#", "#---#", "#---#", "#####"]
    case = {"record_id": "a", "context_id": "a", "layout_id": "a", "phase": "test",
            "family": "real_12x12x4", "source_path": "example", "source_sha256": "abc",
            "annotation_provenance": {"kind": "local_exact_difference"},
            "start_observation": board, "later_observation": final,
            "intent": {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": []}}
    result = {"status": "achieved", "model_calls": 1, "actions": [3], "endpoint": final, "steps": [{
        "action": 3, "no_op": False, "intent_holds": True, "preservation_holds": True,
        "raw_response": '{"action":"RIGHT"}', "after": final}]}
    (tmp_path / "data/records.jsonl").write_text(json.dumps(case) + "\n")
    (tmp_path / "experiment_plan.json").write_text(json.dumps({"record_ids": ["a"]}))
    (tmp_path / "results/rollouts.jsonl").write_text(json.dumps({**case, "cap": 4, "result": result}) + "\n")
    audit = {"status": "settled", "conservative_charge_usd": "0.0001", "reservation_id": "1",
             "request": {"input": "a complete prompt", "max_output_tokens": 32}}
    (tmp_path / "luna_requests/1.json").write_text(json.dumps(audit))
    builder = Path(__file__).resolve().parents[2] / "scripts/build_luna_intent_notebook.py"
    subprocess.run([sys.executable, str(builder), "--run", str(tmp_path)], check=True, capture_output=True, timeout=60)
    document = nbformat.read(tmp_path / "luna_intent_report.ipynb", as_version=4)
    nbformat.validate(document)
    output = json.dumps([cell.get("outputs", []) for cell in document.cells])
    assert "a complete prompt" in output
    assert "0.0001" in output
    assert (tmp_path / "figures/successful_durations.pdf").exists()
