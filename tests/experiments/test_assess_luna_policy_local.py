"""Check the local fallback keeps paired slots and independent-board counts."""

import pytest

from scripts import assess_luna_policy_local as assessment


def test_local_assessment_keeps_all_requested_slots(monkeypatch):
    def search(rows, intent, *, max_depth, max_nodes, seconds):
        assert (max_depth, max_nodes, seconds) == (4, 50, 1.)
        return {"status": "witness_found", "witness_found": True, "reached_observation": rows,
                "expanded_nodes": 3, "elapsed_seconds": .1}

    monkeypatch.setattr(assessment, "bounded_witness", search)
    rows = []
    for board in ("board-a", "board-b"):
        for slot in (.25, .75):
            for method in assessment.METHODS:
                valid = not (board == "board-a" and slot == .25 and method == "B")
                rows.append({"record_id": f"{board}-{slot}-{method}", "layout_id": board, "arm": method,
                             "requested_scale": slot, "start_observation": ["#####", "#@$.#", "#####"],
                             "prediction_status": "valid_intent" if valid else "invalid_json", "json_valid": valid,
                             "parsed": {"effects": [{"predicate": "box_at", "row": 1, "col": 3}],
                                        "preserve": [], "instruction": None} if valid else None})
    results, summary, paired = assessment.analyse(rows, max_depth=4, max_nodes=50, seconds=1.)
    base = next(item for item in summary if item["method"] == "B")
    assert len(results) == 12
    assert (base["boards"], base["requested_outputs"], base["valid_intents"], base["paths_found"]) == (2, 4, 3, 3)
    assert paired["valid_intent"]["boards"] == 2
    assert paired["valid_intent"]["difference_D_minus_B"] == .25
    assert assessment.summarise(results) == (summary, paired)


def test_local_assessment_rejects_missing_paired_slots():
    with pytest.raises(ValueError, match="two predeclared slots"):
        assessment.analyse([{"record_id": "only", "layout_id": "board", "arm": "B",
                             "prediction_status": "invalid_json", "start_observation": ["#@"]}],
                           max_depth=4, max_nodes=50, seconds=1.)


def test_solver_error_is_unresolved_infrastructure_not_a_failed_intent(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("replay unavailable")

    monkeypatch.setattr(assessment, "bounded_witness", fail)
    rows = []
    for slot in (.25, .75):
        for method in assessment.METHODS:
            rows.append({"record_id": f"{slot}-{method}", "layout_id": "board", "arm": method,
                         "requested_scale": slot, "start_observation": ["#####", "#@$.#", "#####"],
                         "prediction_status": "valid_intent", "json_valid": True,
                         "parsed": {"effects": [{"predicate": "box_at", "row": 1, "col": 3}],
                                    "preserve": [], "instruction": None}})
    results, summary, paired = assessment.analyse(rows, max_depth=4, max_nodes=50, seconds=1.)
    assert all(row["path_found"] is None and row["solver_error"] for row in results)
    assert all(item["solver_infrastructure_errors"] == 2 and item["no_path_found_among_valid"] == 0
               for item in summary)
    assert paired["valid_intent"]["boards"] == 1
    assert paired["path_found"]["boards"] == 0


def test_unknown_rule_work_is_not_reported_as_zero():
    result = assessment._local_rule_work([{"local_rule_work": {"walk_graphs": 1}}, {}])
    assert result["metrics"]["walk_graphs"] == {"total": None, "known_sum": 1, "known_outputs": 1}
    assert result["metrics"]["walk_nodes_expanded"]["total"] is None
