"""Checks for paired, board-level stage-four summaries."""
import json

import pytest

from scripts import report_overnight_stage04 as report


def row(layout, method, transform="identity", *, valid=True, found=True, status="complete"):
    return {"family": "12x12x4", "source_family": "source", "layout_id": layout, "context_id": layout,
            "transform": transform, "condition": "original" if transform == "identity" else "d4",
            "method": method, "output_status": status, "valid_nontrivial": valid,
            "bounded_path_found": found, "json_valid": True,
            "parsed_intent": {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": []}}


def test_marginal_distance_and_missing_outcomes_are_explicit():
    assert report.total_variation({"a": 2}, {"b": 2}) == 1
    assert report.total_variation({"a": 1, "b": 1}, {"a": 1, "b": 1}) == 0
    assert report.total_variation({}, {"a": 1}) is None
    assert report.metric(row("a", "wide", valid=False), "valid") == 0
    assert report.metric(row("a", "wide", valid=False), "bounded_path") == 0
    assert report.metric(row("a", "wide", status="missing"), "valid") is None


def test_paired_statistics_resample_layouts_not_outputs():
    rows = []
    for index in range(6):
        layout = f"board-{index}"
        rows.extend([row(layout, "narrow", valid=False), row(layout, "wide", valid=True),
                     row(layout, "wide_aug", valid=index < 3)])
    result = report.paired_differences(rows)
    selected = next(item for item in result if item["first"] == "narrow" and item["metric"] == "valid")
    assert selected["paired_boards"] == 6 and selected["difference_second_minus_first"] == 1
    assert selected["ci_low"] == selected["ci_high"] == 1
    other = next(item for item in result if item["first"] == "wide" and item["metric"] == "valid")
    assert other["paired_boards"] == 6 and other["difference_second_minus_first"] == -.5
    with pytest.raises(ValueError, match="Duplicate method"):
        report.paired_differences(rows + [rows[0]])


def test_family_paired_bootstrap_keeps_source_counts_and_paired_boards():
    rows = []
    for index in range(6):
        for method, valid in (("narrow", False), ("wide", True), ("wide_aug", index < 3)):
            item = row(f"board-{index}", method, valid=valid)
            item["family"] = "12x12x10"
            item["source_family"] = "gamma2" if index < 3 else "reverse"
            rows.append(item)
    results = report.family_paired_differences(rows)
    selected = next(item for item in results if item["first"] == "narrow" and item["metric"] == "valid")
    assert selected["paired_boards"] == 6 and selected["difference_second_minus_first"] == 1
    assert selected["planned_source_board_counts"] == {"gamma2": 3, "reverse": 3}
    assert selected["paired_source_board_counts"] == {"gamma2": 3, "reverse": 3}
    for item in rows:
        if item["method"] == "narrow" and item["layout_id"] in ("board-3", "board-4"):
            item["output_status"] = "missing"
    selected = next(item for item in report.family_paired_differences(rows)
                    if item["first"] == "narrow" and item["metric"] == "valid")
    assert selected["planned_boards"] == 6 and selected["paired_boards"] == 4
    assert selected["planned_source_board_counts"] == {"gamma2": 3, "reverse": 3}
    assert selected["paired_source_board_counts"] == {"gamma2": 3, "reverse": 1}


def test_repetition_counts_invalid_structured_outputs():
    rows = [row("a", "wide", valid=True), row("b", "wide", valid=False),
            row("c", "wide", valid=False, status="missing")]
    item = report.repetition_diagnostics(rows)[0]
    assert item["requested_outputs"] == 3 and item["completed_outputs"] == 2
    assert item["single_box_at_outputs"] == 2 and item["most_common_box_coordinate"] == "2,3"
    assert item["most_common_box_coordinate_outputs"] == 2


def test_recorded_commands_include_real_evaluation_and_corrected_data_preparation():
    stages = [{"name": name, "status": "completed", "command": ["python", "-m", name]} for name in (
        "stage04_prepare_balanced_narrow", "stage04_evaluation", "stage04_eval_unit_tests")]
    assert [item["name"] for item in report.recorded_commands({"stages": stages})] == [
        "stage04_prepare_balanced_narrow", "stage04_evaluation"]


def test_watchdog_amendment_distinguishes_supervisor_and_unchanged_child_deadlines():
    stage = {"name": "stage04_evaluation", "stop_epoch": 90, "stop_monotonic": 9,
             "stop_amendment": {"amended_utc": "recorded", "reason": "reserve",
                                "original": {"stop_epoch": 120, "stop_monotonic": 12}},
             "command": ["python", "--stop-epoch", "110", "--stop-monotonic", "11"]}
    amendment = report.evaluation_stop_amendment({"stages": [stage]})
    assert amendment["original_watchdog_epoch"] == 120
    assert amendment["revised_watchdog_epoch"] == 90
    assert amendment["child_cutoff_epoch"] == 110
    assert report.evaluation_stop_amendment({"stages": []}) is None


def test_figure_data_includes_all_frozen_methods_and_both_metrics():
    rows = []
    for family in ("12x12x4", "12x12x10"):
        for method in report.METHODS:
            for transform in report.TRANSFORMS:
                item = row(f"{family}-board", method, transform)
                item["family"] = family
                rows.append(item)
    result = report.figure_data(rows)
    assert len(result) == 2 * 5 * 3 * 2
    assert {item["metric"] for item in result} == {"valid", "bfs"}
    assert all(item["planned_boards"] == 1 for item in result)


def test_family_repetition_uses_only_completed_outputs_and_preserves_sources():
    rows = []
    for family in ("12x12x4", "12x12x10"):
        for method in report.METHODS:
            for layout, source, status in (("a", "first", "complete"), ("b", "second", "missing")):
                item = row(layout, method, status=status, valid=False)
                item["family"], item["source_family"] = family, source
                rows.append(item)
    results = report.family_repetition(rows)
    assert len(results) == 10
    example = results[0]
    assert example["planned_boards"] == 2 and example["completed_boards"] == 1
    assert example["single_box_at_outputs"] == example["most_common_box_coordinate_outputs"] == 1
    assert example["single_box_at_share"] == example["most_common_box_coordinate_share"] == 1
    assert example["planned_source_board_counts"] == {"first": 1, "second": 1}
    assert example["completed_source_board_counts"] == {"first": 1, "second": 0}
    with pytest.raises(ValueError, match="one original output per board"):
        report.family_repetition(rows + [rows[0]])


def test_examples_are_selected_by_frozen_rank_and_keep_raw_fences():
    contexts, rows = [], []
    for family in ("12x12x4", "12x12x10"):
        for rank in (1, 0):
            context_id = f"{family}-{rank}"
            contexts.append({"context_id": context_id, "family": family, "source_family": "source",
                             "layout_id": context_id, "board_rank": rank, "original_rows": ["##", "#@"]})
            if rank:
                continue
            for method in report.METHODS:
                item = row(context_id, method, valid=False)
                item.update({"family": family, "source_family": "source", "source_rows": ["##", "#@"],
                             "raw_output": "```json\n{}\n```", "bfs_status": "not_applicable_invalid_intent",
                             "bounded_path_found": None})
                rows.append(item)
    examples = report.representative_examples({"context_manifest": contexts}, rows)
    assert [item["board_rank"] for item in examples] == [0, 0]
    assert all(len(item["outputs"]) == len(report.METHODS) for item in examples)
    lines = report.example_lines(examples)
    assert "````text" in lines and "```json\n{}\n```" in lines


def test_predicate_coverage_includes_invalid_parseable_conjunctions():
    rows = []
    for family in ("12x12x4", "12x12x10"):
        for method in report.METHODS:
            for index in range(2):
                item = row(f"{family}-{index}", method, valid=index == 0)
                item["family"] = family
                item["any_effect_initially_true"] = index == 1
                item["parsed_intent"]["effects"].append({"predicate": "boxes_on_goals_at_least", "count": 2})
                rows.append(item)
    coverage = report.predicate_coverage(rows)
    selected = next(item for item in coverage if item["family"] == "12x12x4" and item["method"] == "wide_aug")
    assert selected["parseable_outputs"] == 2 and selected["valid_outputs"] == 1
    assert selected["predicate_presence_parseable"] == {"box_at": 2, "boxes_on_goals_at_least": 2}
    assert selected["predicate_presence_valid"] == {"box_at": 1, "boxes_on_goals_at_least": 1}
    assert selected["box_and_goal_parseable"] == selected["conjunctions_parseable"] == 2
    assert selected["conjunctions_valid"] == selected["conjunctions_any_effect_initially_true"] == 1


def test_descriptive_reduction_requires_lower_single_box_share():
    coverage, repetitions, pairs = [], [], []
    for family in ("12x12x4", "12x12x10"):
        for method in ("wide", "wide_aug"):
            coverage.append({"family": family, "method": method, "parseable_outputs": 2,
                "completed_outputs": 2, "valid_outputs": 1, "requested_outputs": 2,
                "predicate_presence_parseable": {"box_at": 2}, "predicate_presence_valid": {"box_at": 1},
                "box_and_goal_parseable": 2 if method == "wide_aug" else 1,
                "conjunctions_parseable": 2, "conjunctions_valid": 1,
                "conjunctions_any_effect_initially_true": 1, "conjunctions_initial_truth_unknown": 0})
            repetitions.append({"family": family, "method": method, "single_box_at_outputs": 2,
                                "completed_boards": 2, "most_common_box_coordinate_outputs": 1})
        pairs.append({"family": family, "transform": "identity", "first": "wide", "second": "wide_aug",
                      "metric": "valid", "ci_low": -1, "ci_high": 1, "paired_boards": 2, "planned_boards": 2})
    assert not any("reduced single-box share" in line
                   for line in report.descriptive_findings_lines(coverage, repetitions, pairs))
    for item in repetitions:
        if item["method"] == "wide_aug":
            item["single_box_at_outputs"] = 1
    assert sum("reduced single-box share" in line
               for line in report.descriptive_findings_lines(coverage, repetitions, pairs)) == 2


def test_actual_sample_marginals_include_symmetry_and_repetitions(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    record = {"record_id": "r", "layout_id": "layout", "effect_type": "box", "k": 4,
              "start_observation": ["-" * 12] * 12,
              "intent": {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": []}}
    for arm in ("narrow", "wide"):
        (data / f"{arm}.jsonl").write_text(json.dumps(record) + "\n")
    arm = tmp_path / "wide_aug"
    arm.mkdir()
    samples = [{"record_id": "r", "layout_id": "layout", "symmetry": symmetry}
               for symmetry in ("identity", "rotate_90")]
    (arm / "training_curve.jsonl").write_text(json.dumps({"step": 1, "train_loss": .5, "samples": samples,
        "model_tokens": 20, "effective_tokens": 18, "supervised_tokens": 5}) + "\n")
    (arm / "status.json").write_text(json.dumps({"status": "finished"}))
    curves, summaries = report.training_curves({"wide_aug": arm}, data)
    assert len(curves) == 1
    summary = next(item for item in summaries if item["arm"] == "wide_aug")
    assert summary["sampled_examples"] == 2 and summary["sampled_effect_types"] == {"box": 2}
    assert summary["sampled_offsets"] == {"4": 2}
    assert summary["sampled_coordinates"] == {"2,3": 1, "8,2": 1}


def test_train_only_data_revision_accepts_frozen_chain_and_rejects_tampering(tmp_path):
    stage, old, new = tmp_path, tmp_path / "data_v2", tmp_path / "data_v3"
    old.mkdir()
    new.mkdir()

    def write(path, value):
        path.write_text(json.dumps(value))
        return report.sha256(path)

    for directory in (old, new):
        (directory / "dataset_wide.jsonl").write_text("wide")
        (directory / "validation.jsonl").write_text("validation")
    (old / "dataset_narrow.jsonl").write_text("old narrow")
    (new / "dataset_narrow.jsonl").write_text("new narrow")
    manifests = {"splits": {"train": ["a"], "validation": ["b"]},
                 "arms": {"narrow": {"train_layout_ids": ["a"]}, "wide": {"train_layout_ids": ["a"]}}}
    old_sha = write(old / "dataset_manifest.json", manifests)
    revised_manifest = {**manifests, "narrow_sampling_correction": {
        "source_dataset_manifest_sha256": old_sha,
        "dataset_wide_sha256_unchanged": report.sha256(old / "dataset_wide.jsonl")}}
    new_sha = write(new / "dataset_manifest.json", revised_manifest)
    freeze_sha = write(stage / "training_freeze.json", {"data_sha256": {"dataset_manifest": old_sha}})
    assert report.verify_dataset_revision(stage, old)["kind"] == "original_v2"
    decision_sha = write(stage / "training_freeze_amendment_narrow.json",
                         {"previous_training_plan_sha256": freeze_sha})
    audit_sha = write(stage / "data_v3_independent_audit.json", {"status": "pass", "dataset_manifest_sha256": new_sha})
    sample_sha = write(stage / "narrow_sampling_preflight_v3.json", {})
    token_sha = write(stage / "token_preflight_v3.json", {})
    realized = {
        "prior_training_freeze_sha256": freeze_sha, "decision_amendment_sha256": decision_sha,
        "old_dataset_manifest_sha256": old_sha, "new_dataset_manifest_sha256": new_sha,
        "old_narrow_sha256": report.sha256(old / "dataset_narrow.jsonl"),
        "new_narrow_sha256": report.sha256(new / "dataset_narrow.jsonl"),
        "wide_v2_sha256": report.sha256(old / "dataset_wide.jsonl"),
        "wide_v3_sha256": report.sha256(new / "dataset_wide.jsonl"),
        "validation_v2_sha256": report.sha256(old / "validation.jsonl"),
        "validation_v3_sha256": report.sha256(new / "validation.jsonl"),
        "data_v3_audit_sha256": audit_sha, "sampling_preflight_sha256": sample_sha,
        "token_preflight_sha256": token_sha, "no_new_test_results_used": True,
        "narrow_layouts_unchanged": True, "wide_and_validation_unchanged": True}
    realized_sha = write(stage / "training_freeze_amendment_narrow_realized.json", realized)
    write(stage / "training_freeze_v3.json", {"parent_training_freeze_sha256": freeze_sha,
        "realized_amendment_sha256": realized_sha, "data_sha256": {"dataset_manifest": new_sha}})
    assert report.verify_dataset_revision(stage, new)["kind"] == "balanced_narrow_v3"
    (new / "dataset_wide.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="recorded hashes"):
        report.verify_dataset_revision(stage, new)
