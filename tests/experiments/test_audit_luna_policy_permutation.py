"""Check the exploratory within-family permutation diagnostic."""

import json

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.policy_relative_lora import PredictionStatus
from scripts import audit_luna_policy_permutation as audit

BOARD_A = ["#####", "#@$.#", "#####"]
BOARD_B = ["#####", "#@.$#", "#####"]
LAYOUT_A = canonical_layout_id(from_rows(BOARD_A))
LAYOUT_B = canonical_layout_id(from_rows(BOARD_B))


def _cases(*, shared_intent=False):
    boards = {LAYOUT_A: BOARD_A, LAYOUT_B: BOARD_B}
    rows = []
    for layout, board in boards.items():
        for method in audit.METHODS:
            for slot in (.25, .75):
                col = 1 if shared_intent else 3 if layout == LAYOUT_A else 2
                intent = {"effects": [{"predicate": "box_at", "row": 1, "col": col}],
                          "preserve": [], "instruction": None}
                search = {"status": "witness_found", "witness_found": True, "expanded_nodes": 1,
                          "primitive_edges_examined": 4, "elapsed_seconds": .1, "verification_seconds": .01,
                          "depth_limit": 24, "node_limit": 10000, "time_limit_seconds": 1.}
                rows.append({"record_id": f"{layout}-{method}-{slot}", "layout_id": layout, "family": "family",
                             "arm": method, "requested_scale": slot, "start_observation": board,
                             "prediction_status": PredictionStatus.VALID, "raw_output": json.dumps(intent),
                             "search": search, "solver_error": None, "path_found": True})
    return rows


def test_derangements_are_deterministic_and_never_fixed():
    mappings = audit.derangements("family", 6, 20261115, 1000)
    assert mappings == audit.derangements("family", 6, 20261115, 1000)
    assert len(mappings) == 1000
    assert all(sorted(row) == list(range(6)) and all(i != value for i, value in enumerate(row)) for row in mappings)
    with pytest.raises(ValueError):
        audit.derangements("family", 1, 20261115, 1)


def test_permutation_preserves_raw_frequencies_and_detects_board_mismatch(monkeypatch):
    monkeypatch.setattr(audit, "bounded_witness", lambda *a, **k: pytest.fail("Invalid intents need no search"))
    rows = _cases()
    cases, distribution, result = audit.analyse(rows, permutations=5)
    assert len(cases) == 6
    assert len(distribution) == 15
    assert all(item["observed_valid"] == 2 and item["permuted_valid_fraction_requested_mean"] == 0
               for item in result["methods"])
    assert all(item["prediction_status"] == PredictionStatus.INVALID_INTENT for item in cases)
    for method in audit.METHODS:
        source = sorted(row["raw_output"] for row in rows if row["arm"] == method and row["requested_scale"] == .25)
        permuted = sorted(row["raw_output"] for row in cases if row["arm"] == method)
        assert source == permuted
    assert {(row["target_layout_id"], row["source_layout_id"]) for row in cases} == {
        (LAYOUT_A, LAYOUT_B), (LAYOUT_B, LAYOUT_A)}


def test_infrastructure_missing_remains_unknown_and_in_requested_denominator(monkeypatch):
    monkeypatch.setattr(audit, "bounded_witness", lambda *a, **k: pytest.fail("Invalid intents need no search"))
    rows = _cases()
    row = next(row for row in rows if row["layout_id"] == LAYOUT_A and row["arm"] == "B"
               and row["requested_scale"] == .25)
    row.update(prediction_status=PredictionStatus.INFRASTRUCTURE_ERROR, raw_output=None,
               search=None, path_found=None)
    cases, _, result = audit.analyse(rows, permutations=3)
    base = next(item for item in result["methods"] if item["arm"] == "B")
    assert (base["requested"], base["completed"], base["generator_infrastructure_missing"]) == (2, 1, 1)
    assert base["observed_valid_fraction_requested"] == .5
    assert base["observed_valid_fraction_completed"] == 1
    assert base["permutation_fraction_at_least_observed"] is None
    missing = next(item for item in cases if item["arm"] == "B" and item["source_layout_id"] == LAYOUT_A)
    assert missing["path_found"] is None
    assert missing["prediction_status"] == PredictionStatus.INFRASTRUCTURE_ERROR


def test_rule_cost_keeps_known_work_and_does_not_invent_missing_costs(monkeypatch):
    monkeypatch.setattr(audit, "bounded_witness", lambda *a, **k: pytest.fail("Invalid intents need no search"))
    rows = _cases()
    first = next(row for row in rows if row["layout_id"] == LAYOUT_A and row["arm"] == "R"
                 and row["requested_scale"] == .25)
    first["local_rule_work"] = {"walk_nodes_expanded": 5, "elapsed_seconds": .2}
    _, _, result = audit.analyse(rows, permutations=2)
    rule = next(item for item in result["methods"] if item["arm"] == "R")
    work = rule["source_generator_work"]
    assert work["new_calls_for_permutation"] == 0
    assert work["local_rule_work"]["walk_nodes_expanded"] == {
        "requested_outputs": 2, "known_outputs": 1, "known_sum": 5, "total": None}
    assert work["model_generation_seconds"]["known_sum"] is None


def test_frozen_contexts_keep_missing_rows_outputs_and_solver_results(monkeypatch):
    monkeypatch.setattr(audit, "bounded_witness", lambda *a, **k: pytest.fail("Invalid intents need no search"))
    rows = _cases()
    contexts = [{"family": "family", "layout_id": layout,
                 "start_observation": next(row["start_observation"] for row in rows if row["layout_id"] == layout)}
                for layout in (LAYOUT_A, LAYOUT_B)]
    rows = [row for row in rows if not (row["layout_id"] == LAYOUT_A and row["arm"] == "B"
                                       and row["requested_scale"] == .25)]
    bad_raw = next(row for row in rows if row["layout_id"] == LAYOUT_A and row["arm"] == "D"
                   and row["requested_scale"] == .25)
    bad_raw["raw_output"] = None
    missing_status = next(row for row in rows if row["layout_id"] == LAYOUT_A and row["arm"] == "R"
                          and row["requested_scale"] == .25)
    del missing_status["prediction_status"]
    missing_solver = next(row for row in rows if row["layout_id"] == LAYOUT_B and row["arm"] == "R"
                          and row["requested_scale"] == .25)
    missing_solver.update(search=None, path_found=None)
    cases, _, result = audit.analyse(rows, contexts=contexts, permutations=2)
    for method in audit.METHODS:
        summary = next(item for item in result["methods"] if item["arm"] == method)
        assert summary["requested"] == 2 and summary["generator_infrastructure_missing"] == 1
    rule = next(item for item in result["methods"] if item["arm"] == "R")
    assert rule["original_solver"]["solver_results_missing"] == 1
    assert rule["original_solver"]["fraction_per_completed"] is None
    assert sum(row["path_found"] is None for row in cases) == 3


def test_frozen_contexts_include_fully_missing_board(monkeypatch):
    monkeypatch.setattr(audit, "bounded_witness", lambda *a, **k: pytest.fail("Invalid intents need no search"))
    rows = _cases()
    contexts = [{"family": "family", "layout_id": layout,
                 "start_observation": next(row["start_observation"] for row in rows if row["layout_id"] == layout)}
                for layout in (LAYOUT_A, LAYOUT_B)]
    rows = [row for row in rows if row["layout_id"] == LAYOUT_B]
    cases, _, result = audit.analyse(rows, contexts=contexts, permutations=2)
    assert len(cases) == 6
    assert all(item["requested"] == 2 and item["generator_infrastructure_missing"] == 1
               for item in result["methods"])


def test_only_first_permutation_runs_solver_and_errors_are_separate(monkeypatch):
    calls = []
    validity_saved = []

    def search(board, intent, **limits):
        assert validity_saved
        calls.append((board, intent, limits))
        if board[1] == "#@.$#":
            raise RuntimeError("synthetic replay error")
        return {"status": "witness_found", "witness_found": True, "expanded_nodes": 4,
                "primitive_edges_examined": 16, "elapsed_seconds": .2, "verification_seconds": .01,
                "depth_limit": 24, "node_limit": 10000, "time_limit_seconds": 1.}

    monkeypatch.setattr(audit, "bounded_witness", search)
    cases, _, result = audit.analyse(_cases(shared_intent=True), permutations=10,
                                     on_validity=lambda distribution, partial, seconds: validity_saved.append(True))
    assert len(calls) == 6
    assert all(call[2] == {"max_depth": 24, "max_nodes": 10000, "seconds": 1.} for call in calls)
    assert sum(row["solver_error"] is not None for row in cases) == 3
    for item in result["methods"]:
        search_summary = item["first_permutation_solver"]
        assert (search_summary["requested"], search_summary["paths_found"],
                search_summary["solver_infrastructure_errors"]) == (2, 1, 1)
        assert search_summary["fraction_per_requested_lower_bound_if_missing"] == .5
        assert search_summary["fraction_per_completed"] == 1


def test_rejects_duplicate_selected_slot():
    rows = _cases()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicated"):
        audit.analyse(rows, permutations=2)


def test_rejects_noncanonical_layout_id():
    rows = _cases()
    for row in rows:
        if row["layout_id"] == LAYOUT_A:
            row["layout_id"] = "another-id-for-the-same-board"
    with pytest.raises(ValueError, match="canonical D4"):
        audit.analyse(rows, permutations=2)


def test_original_solver_budget_mismatch_is_excluded_separately(monkeypatch):
    monkeypatch.setattr(audit, "bounded_witness", lambda *a, **k: pytest.fail("Invalid intents need no search"))
    rows = _cases()
    row = next(row for row in rows if row["layout_id"] == LAYOUT_A and row["arm"] == "B"
               and row["requested_scale"] == .25)
    row["search"]["node_limit"] = 99
    _, _, result = audit.analyse(rows, permutations=2)
    base = next(item for item in result["methods"] if item["arm"] == "B")
    solver = base["original_solver"]
    assert solver["solver_budget_mismatches"] == 1
    assert solver["solver_infrastructure_errors"] == 0
    assert solver["paths_found"] == 1 and solver["fraction_per_completed"] == 1


def test_plot_has_vector_and_raster_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "bounded_witness", lambda *a, **k: pytest.fail("Invalid intents need no search"))
    _, distribution, result = audit.analyse(_cases(), permutations=3)
    paths = audit.plot_distributions(tmp_path, distribution, result["methods"])
    assert {path.rsplit(".", 1)[-1] for path in paths} == {"pdf", "svg", "png"}
    assert all((tmp_path / path).stat().st_size > 0 for path in paths)
    svg = next((tmp_path / path).read_text() for path in paths if path.endswith(".svg"))
    assert all(label in svg for label in ("Base Qwen", "Qwen + LoRA", "Local rule"))
    assert "2 boards · 3 permutations" in svg


def test_missing_result_file_is_audited_against_frozen_contexts(tmp_path, monkeypatch):
    rows = _cases()
    contexts = [{"family": "family", "layout_id": layout,
                 "start_observation": next(row["start_observation"] for row in rows if row["layout_id"] == layout)}
                for layout in (LAYOUT_A, LAYOUT_B)]
    context_path = tmp_path / "contexts.jsonl"
    context_path.write_text("".join(json.dumps(context) + "\n" for context in contexts))
    out = tmp_path / "audit"
    monkeypatch.setattr(audit, "plot_distributions", lambda *args: [])
    monkeypatch.setattr("sys.argv", ["audit", "--results", str(tmp_path / "missing.jsonl"),
                                     "--contexts", str(context_path), "--allow-missing-results",
                                     "--output", str(out), "--permutations", "2"])
    audit.main()
    protocol = json.loads((out / "protocol.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    assert protocol["source_files"][0]["available"] is False
    assert protocol["source_files"][0]["sha256"] is None
    assert all(item["requested"] == 2 and item["generator_infrastructure_missing"] == 2
               and item["observed_valid_fraction_completed"] is None for item in summary["methods"])
    assert (out / "preliminary_summary.json").is_file()
    assert (out / "permutation_distribution.csv").is_file()
