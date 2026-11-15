"""Verify disjoint replay audits cover exactly the same evaluation boards."""
import hashlib
import json

import pytest

from scripts.combine_overnight_stage05_audits import combine


def write_audit(path, boards, methods, *, state="pass", audited=2):
    value = {"state": state, "boards_sha256": hashlib.sha256(boards.read_bytes()).hexdigest(),
             "frozen_boards": 2, "methods": methods, "expected_jobs": len(methods) * 2,
             "present_jobs": len(methods) * 2, "missing_jobs": [], "errors": [], "incomplete_traces": [],
             "offline_audit_transitions": 8, "auditor_sha256": "same_auditor",
             "by_method": {method: {"jobs_audited": audited, "solutions_verified": 0} for method in methods}}
    path.write_text(json.dumps(value))


def test_combine_checks_two_disjoint_method_sets_and_preserves_unknown_cost(tmp_path):
    boards = tmp_path / "boards.jsonl"
    boards.write_text('\n'.join(json.dumps({"split": "evaluation", "layout_id": str(i)}) for i in range(2)))
    baseline, search, interrupted = (tmp_path / f"{name}.json" for name in ("baseline", "search", "interrupted"))
    write_audit(baseline, boards, ["direct_policy", "primitive_astar"])
    write_audit(search, boards, ["adasubs", "ksubs"])
    interrupted.write_text(json.dumps({"partial_offline_transition_count": None,
                                       "partial_offline_transition_count_reason": "not recorded"}))
    result = combine(boards, [baseline, search], interrupted)
    assert result["state"] == "pass"
    assert result["offline_audit_transitions"] == 16
    assert result["interrupted_prior_audit"]["offline_transitions"] is None
    assert len(result["methods"]) == 4


def test_combine_rejects_different_boards_or_duplicate_methods(tmp_path):
    boards = tmp_path / "boards.jsonl"
    boards.write_text('\n'.join(json.dumps({"split": "evaluation", "layout_id": str(i)}) for i in range(2)))
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_audit(first, boards, ["direct_policy"])
    write_audit(second, boards, ["direct_policy"])
    with pytest.raises(ValueError, match="repeated"):
        combine(boards, [first, second])
    raw = json.loads(second.read_text())
    raw["boards_sha256"] = "other"
    second.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="same frozen boards"):
        combine(boards, [first, second])


def test_combine_reports_incomplete_method_coverage(tmp_path):
    boards = tmp_path / "boards.jsonl"
    boards.write_text('\n'.join(json.dumps({"split": "evaluation", "layout_id": str(i)}) for i in range(2)))
    first = tmp_path / "first.json"
    write_audit(first, boards, ["direct_policy", "primitive_astar"], audited=1)
    result = combine(boards, [first])
    assert result["state"] == "partial"
    assert {issue.get("method") for issue in result["issues"]} >= {"direct_policy", "adasubs", "ksubs"}
