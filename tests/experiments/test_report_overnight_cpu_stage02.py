"""Reports must retain missing data and avoid conditioning artifacts."""
from __future__ import annotations

import csv
import json

from scripts import report_overnight_cpu_stage02 as reporter
from scripts.run_overnight_cpu_policy import file_sha


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_singletons_are_excluded_from_conditional_rank_mean():
    common = [
        {"family": "12x12x4", "controller": "cllp8", "L": "32", "status": "complete",
         "layout_id": "a", "common_success_targets": "1", "distinct_d": "1", "mean_abs_u_difference": "0"},
        {"family": "12x12x4", "controller": "cllp8", "L": "32", "status": "complete",
         "layout_id": "b", "common_success_targets": "2", "distinct_d": "2", "mean_abs_u_difference": "0.2"},
        {"family": "12x12x4", "controller": "cllp8", "L": "32", "status": "complete",
         "layout_id": "c", "common_success_targets": "3", "distinct_d": "1", "mean_abs_u_difference": "0.4"},
        {"family": "12x12x4", "controller": "cllp8", "L": "32", "status": "incomplete",
         "layout_id": "d", "common_success_targets": "", "distinct_d": "", "mean_abs_u_difference": ""},
    ]
    stats = reporter.conditional_ranks(common, "12x12x4", "cllp8", 32)
    assert stats["singleton_contexts"] == 1
    assert stats["multi_contexts"] == 2 and stats["multi_with_distinct_d"] == 1
    assert abs(stats["mean"] - 0.3) < 1e-12
    assert stats["n_layouts"] == 2 and stats["ci_low"] is not None


def test_findings_use_shared_successes_and_skip_incomplete_comparisons():
    rank_summaries, ranks = [], []
    for controller in ("cllp8", "cllp32"):
        rank_summaries.append({"scope": "family", "family": "12x12x4", "controller": controller, "L": 32,
                               "planned_contexts": 2, "complete_contexts": 2, "incomplete_contexts": 0,
                               "inversions": 1, "strict_comparable": 3})
        for index, difference in enumerate((0.2, 0.4)):
            ranks.append({"family": "12x12x4", "controller": controller, "L": "32", "status": "complete",
                          "layout_id": f"board-{index}", "common_success_targets": "2", "distinct_d": "2",
                          "mean_abs_u_difference": str(difference)})
    sensitivity = {"scope": "family", "family": "12x12x4", "controller": "cllp8", "L": 32,
                   "pool_variant": "demo4_bfs6", "reference": "fixed", "comparison": "reduced",
                   "planned_contexts": 2, "complete_pool_contexts": 2, "contexts_with_shared_success": 2,
                   "mean_abs_delta_u": 0.4, "ci_low": 0.3, "ci_high": 0.5}
    summary = {"rank_summary": rank_summaries, "sensitivity_summary": [sensitivity],
               "candidate_summary": [{"scope": "family", "L": 32, "completed_attempts": 4, "successes": 3}]}
    budget = [{"family": "12x12x4", "source_family": "all", "controller": "cllp8",
               "pool_variant": "demo4_bfs6", "short_L": "8", "long_L": "32",
               "planned_contexts": "2", "complete_contexts": "2", "contexts_with_shared_success": "2",
               "mean_abs_delta_u": "0.15", "ci_low": "0.1", "ci_high": "0.2"}]
    findings = "\n".join(reporter.scale_findings(summary, ranks, budget))
    assert "0.300 [" in findings and "1/3 strict inversions" in findings
    assert "0.150 [0.100, 0.200]" in findings and "0.400 [0.300, 0.500]" in findings
    assert "n=2 contexts" in findings and "1/4 completed candidate attempts" in findings
    assert "conditional on both controllers" in findings
    budget[0]["complete_contexts"] = "1"
    assert "when L changed" not in "\n".join(reporter.scale_findings(summary, ranks, budget))


def test_generator_partial_bounds_and_conditional_denominators_are_distinct():
    rows = [{"family": "12x12x4", "source_family": "all", "controller": "cllp8", "L": "32",
             "planned_outputs": "10", "valid_outputs": "6", "invalid_outputs": "2", "duplicate_outputs": "1",
             "generator_missing_outputs": "2", "valid_measured_outputs": "5", "valid_missing_controller": "1",
             "successful_outputs": "3", "complete_contexts": "3", "planned_contexts": "5",
             "success_per_requested_equal_context": "0.4", "ci_low": "0.1", "ci_high": "0.7",
             "bootstrap_layouts": "3"}]
    cells = reporter.generator_rows(rows)[0]
    assert cells[2:9] == ["6/10", "2", "1", "2/1", "3/10", "30.0–60.0%", "3/5"]
    assert cells[-2:] == ["3/5", "3"]


def test_bfs_rank_summary_preserves_variants_ties_and_denominators():
    row = {"scope": "family", "family": "12x12x4", "controller": "cllp8", "L": 32,
           "pool_variant": "demo4_bfs6", "complete_contexts": 10, "planned_contexts": 12,
           "contexts_with_bfs_success": 8, "success_layouts": 8, "mean_abs_rank_difference": 0.2,
           "ci_low": 0.1, "ci_high": 0.3, "inversions": 2, "strict_comparable": 20,
           "source_depth_tie_only": 3, "d_tie_only": 4, "both_tied": 5}
    cells = reporter.bfs_rank_rows({"bfs_rank_summary": [row]})[0]
    assert cells == ["12x12x4", "demo4_bfs6", "cllp8", "10/12", "8/8", "0.200 [0.100, 0.300]",
                     "2/20", "3/4/5"]


def test_partial_report_never_invents_absent_results_and_writes_atomically(tmp_path):
    body, status = reporter.report(tmp_path)
    assert status == "partial"
    assert "no result has been inferred" in body
    assert "not available yet" in body
    target = tmp_path / "stage_02_report.md"
    reporter.atomic_text(target, body)
    assert target.read_text() == body
    assert not list(tmp_path.glob(".*.tmp"))


def test_optional_pool_figure_checks_hash_without_blocking_absence(tmp_path):
    ready, message = reporter.optional_pool_figure(tmp_path)
    assert not ready and "not available" in message
    figure = tmp_path / "pool_budget_sensitivity.pdf"
    figure.write_bytes(b"figure fixture")
    ready, message = reporter.optional_pool_figure(tmp_path)
    assert not ready and "no readable hash manifest" in message
    save_json(tmp_path / "pool_budget_sensitivity_manifest.json", {"outputs_sha256": "not a mapping"})
    ready, message = reporter.optional_pool_figure(tmp_path)
    assert not ready and "no readable hash manifest" in message
    save_json(tmp_path / "pool_budget_sensitivity_manifest.json", {"outputs_sha256": {
        "pool_budget_sensitivity.pdf": file_sha(figure)}})
    ready, _ = reporter.optional_pool_figure(tmp_path)
    assert ready
    figure.write_bytes(b"changed")
    ready, message = reporter.optional_pool_figure(tmp_path)
    assert not ready and "does not match" in message


def test_hash_mismatch_cannot_publish_stale_findings(tmp_path):
    analysis = tmp_path / "stage_02" / "analysis_evaluation"
    save_json(analysis / "summaries.json", {"demo_reachability": [], "rank_summary": [],
                                           "pool_summary": [], "sensitivity_summary": []})
    save_csv(analysis / "common_ranks.csv", [{"family": "12x12x4"}])
    save_json(analysis / "analysis_manifest.json", {"planned_attempts": 10, "completed_attempts": 10,
                                                    "missing_attempts": 0,
                                                    "output_sha256": {"summaries.json": "incorrect",
                                                                      "common_ranks.csv": file_sha(
                                                                          analysis / "common_ranks.csv")}})
    body, status = reporter.report(tmp_path)
    assert status == "partial"
    assert "summary does not match" in body
    assert "hash-matched evaluation summary" in body
    assert "## Findings" not in body


def test_complete_report_uses_only_hash_matched_assessment(tmp_path):
    stage = tmp_path / "stage_02"
    analysis = stage / "analysis_evaluation"
    demo = {"scope": "family", "family": "12x12x4", "source_family": "all", "L": 32, "k": 32,
            "controller": "cllp8", "planned_contexts": 1, "completed_contexts": 1, "missing_contexts": 0,
            "reached_contexts": 1, "rate": 1.0, "ci_low": None, "ci_high": None}
    rank = {"scope": "family", "family": "12x12x4", "controller": "cllp8", "L": 32,
            "complete_contexts": 1, "contexts_with_common_success": 1, "common_targets": 2, "error_layouts": 1}
    pool = {"scope": "family", "family": "12x12x4", "controller": "cllp8", "L": 32,
            "pool": "fixed", "pool_variant": "demo4_bfs4", "planned_contexts": 1,
            "complete_contexts": 1, "informative_contexts": 1}
    sensitivity = {"scope": "family", "family": "12x12x4", "controller": "cllp8", "L": 32,
                   "pool_variant": "demo4_bfs4", "reference": "fixed", "comparison": "demo_only"}
    bfs_rank = {"scope": "family", "family": "12x12x4", "controller": "cllp8", "L": 32,
                "pool_variant": "demo4_bfs4", "complete_contexts": 1, "planned_contexts": 1,
                "contexts_with_bfs_success": 1, "success_layouts": 1, "mean_abs_rank_difference": 0.25,
                "inversions": 1, "strict_comparable": 2}
    save_json(analysis / "summaries.json", {"demo_reachability": [demo], "rank_summary": [rank],
                                           "pool_summary": [pool], "sensitivity_summary": [sensitivity],
                                           "bfs_rank_summary": [bfs_rank]})
    save_csv(analysis / "common_ranks.csv", [{"family": "12x12x4", "controller": "cllp8", "L": 32,
             "status": "complete", "layout_id": "a", "common_success_targets": 2, "distinct_d": 2,
             "mean_abs_u_difference": 0.1}])
    save_csv(analysis / "profiles.csv", [{"family": "12x12x4", "controller": "cllp8", "L": 32,
             "pool": "fixed", "status": "complete", "context_id": "first-context", "board_rank": 0,
             "median_d": 4}])
    labels = [{"family": "12x12x4", "controller": "cllp8", "L": 32, "pool": "fixed",
               "pool_status": "complete", "status": "complete", "context_id": "first-context",
               "board_rank": 0, "source_family": "real", "candidate_origin": "demonstration",
               "slot": "demo_01", "candidate_id": "a", "selected_demo_offset": 1, "bfs_depth": "",
               "p": 1, "d": 1, "d_over_median": 0.25, "u": 0.1},
              {"family": "12x12x4", "controller": "cllp8", "L": 32, "pool": "fixed",
               "pool_status": "complete", "status": "complete", "context_id": "first-context",
               "board_rank": 0, "source_family": "real", "candidate_origin": "bfs",
               "slot": "bfs_01", "candidate_id": "b", "selected_demo_offset": "", "bfs_depth": 8,
               "p": 0, "d": "", "d_over_median": "", "u": ""}]
    save_csv(analysis / "profile_labels.csv", labels)
    for name in ("demo_reachability.pdf", "scale_diagnostics.pdf"):
        (analysis / name).write_bytes(b"figure fixture")
    outputs = {name: file_sha(analysis / name) for name in ("summaries.json", "common_ranks.csv", "profiles.csv",
               "profile_labels.csv", "demo_reachability.pdf", "scale_diagnostics.pdf")}
    save_json(analysis / "analysis_manifest.json", {"planned_attempts": 2, "completed_attempts": 2,
                                                    "missing_attempts": 0, "output_sha256": outputs})
    save_csv(analysis / "generator_proposal_summary.csv", [{"family": "12x12x4", "source_family": "all",
             "controller": "cllp8", "L": 32, "planned_outputs": 2, "valid_outputs": 1,
             "invalid_outputs": 1, "duplicate_outputs": 0, "generator_missing_outputs": 0,
             "valid_measured_outputs": 1, "valid_missing_controller": 0, "successful_outputs": 1,
             "complete_contexts": 1, "planned_contexts": 1, "success_per_requested_equal_context": 0.5,
             "ci_low": "", "ci_high": "", "bootstrap_layouts": 1}])
    save_csv(analysis / "budget_sensitivity_summary.csv", [{"family": "12x12x4", "source_family": "all",
             "controller": "cllp8", "pool_variant": "demo4_bfs4", "short_L": 8, "long_L": 32,
             "planned_contexts": 1, "complete_contexts": 1, "contexts_with_shared_success": 1,
             "shared_success_targets": 2, "contexts_with_new_success": 1, "new_success_targets": 1,
             "mean_abs_delta_u": 0.15, "ci_low": "", "ci_high": "", "delta_layouts": 1}])
    status = {"state": "complete", "planned": 2, "completed": 2, "imported": 1, "new": 1}
    save_json(stage / "controller_evaluation/status.json", status)
    save_json(stage / "controller_train/status.json", status)
    save_jsonl(stage / "data_evaluation/contexts.jsonl", [{"family": "12x12x4", "layout_id": "a",
                                                            "pool_variant": "demo4_bfs4"},
                                                           {"family": "12x12x10", "layout_id": "c",
                                                            "pool_variant": "demo4_bfs6"}])
    save_jsonl(stage / "data_train/contexts.jsonl", [{"family": "12x12x4", "layout_id": "b"}])
    save_jsonl(stage / "data_train/candidates.jsonl", [{"candidate_origin": "demonstration"}])
    audit = {"state": "passed", "error_count": 0, "counts": {"profiles": 20, "checks_label_p": 42,
             "checks_label_d": 42, "checks_label_u": 30, "bfs_records_replayed": 5}}
    save_json(analysis / "independent_audit.json", audit)
    save_json(stage / "analysis_train/independent_audit.json", audit)
    save_json(stage / "independent_rank_audit.json", {"status": "passed", "results": {
        "demo": {"error_count": 0}, "bfs": {"error_count": 0}}})
    body, state = reporter.report(tmp_path)
    assert state == "complete"
    assert "1/2" in body and "50.0–50.0%" in body
    assert "1 / 1" in body and "0.100 [interval unavailable]" in body
    assert "Campaign-train candidate origins: demonstration: 1" in body
    assert "bfs (depth=8)" in body and "| 0 | null | 4.000 | null | null |" in body
    assert "demonstration (k=1)" in body and "| 1 | 1 | 4.000 | 0.250 | 0.100 |" in body
    assert "8 → 32" in body and "0.150 [interval unavailable]" in body
    assert "optional pool-and-budget sensitivity figure is not available" in body
    assert "BFS depth versus controller duration" in body and "0.250 [interval unavailable]" in body
    assert "20 profiles, 42 p checks, 42 d checks, 30 u checks" in body
    assert "Independent rank audit" in body and "demo errors 0, BFS errors 0" in body


def test_commands_are_exact_and_do_not_expose_unexpected_credentials():
    manifest = {"stages": [{"name": "stage02_controller_evaluation", "status": "completed",
                            "command": ["env", "OPENAI_API_KEY=offline-test-key", "python", "file with space.py"]},
                           {"name": "stage02_controller_train", "status": "failed",
                            "command": ["env", "OPENAI_API_KEY=secret", "python", "x.py"]}]}
    rows = reporter.commands(manifest)
    assert "'file with space.py'" in rows[0][2]
    assert "secret" not in rows[1][2] and rows[1][2].startswith("not shown:")
