"""The controller addendum shares replay and keeps its exploratory provenance."""

import csv
import hashlib
import json
import sys
from argparse import Namespace

import pytest

import scripts.assess_controller_lora_addendum as addendum_entry
import scripts.assess_fixed_controller as assessment
from scripts.assess_fixed_controller import ADDENDUM_PAIRS
from scripts.assess_fixed_controller import ADDENDUM_STATUS
from scripts.assess_fixed_controller import AnalysisKind
from scripts.assess_fixed_controller import assess
from scripts.assess_fixed_controller import validate_addendum_plan


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def board(boxes):
    return (["############", "#" + "$" * boxes + "-" * (10 - boxes) + "#",
             "#@" + "." * min(boxes, 9) + "-" * (9 - min(boxes, 9)) + "#"]
            + ["#" + "." * (1 if boxes == 10 and index == 0 else 0)
               + "-" * (9 if boxes == 10 and index == 0 else 10) + "#" for index in range(8)]
            + ["############"])


@pytest.fixture
def study(tmp_path):
    run = tmp_path
    (run / "data").mkdir()
    contexts = [{"record_id": f"id-{index}", "layout_id": f"layout-{index}", "split": "test",
                 "evaluation_family": "real_12x12x4" if index < 20 else "proposal_gamma2_12x12x10",
                 "start_observation": board(4 if index < 20 else 10), "task_goal": "put boxes on goals"}
                for index in range(40)]
    data = run / "data/dataset.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in contexts))
    ids = run / "controller_test_ids.json"
    ids.write_text(json.dumps({"dataset_sha256": digest(data), "record_ids": [row["record_id"] for row in contexts]}))
    selection = {"model": "fixed-model", "action_cap": 16}
    selected = run / "controller_selection.json"
    selected.write_text(json.dumps({"selected": "mock", "selected_config": selection,
                                    "selected_configuration_sha256": "fixed-config"}))
    calibration = run / "policy_calibration_mock.manifest.json"
    calibration.write_text(json.dumps({"configuration_sha256": "fixed-config"}))
    assets = run / "controller_assets_snapshot.json"
    assets.write_text("{}")
    strict = {"frozen_ids": str(ids), "frozen_ids_sha256": digest(ids), "controller_selection_sha256": digest(selected),
              "selected_controller": selection, "decoding": "greedy", "action_parser": "fixed",
              "model_training": False, "primitive_action_budget": 16, "api_calls": 0}
    strict_path = run / "policy_evaluation_plan.json"
    strict_path.write_text(json.dumps(strict))
    methods = [{"name": name, "representation": representation, "support_set": "zero",
                "adapter": None if name.startswith("base_") else "fixed-adapter"}
               for representation, pair in zip(("json", "formal", "natural"), ADDENDUM_PAIRS, strict=True)
               for name in pair]
    matrix = {"methods": methods, "inputs": {"data": {"path": str(data), "sha256": digest(data)},
                                               "test_ids": {"path": str(ids), "sha256": digest(ids)}}}
    matrix_path = run / "matrix_plan.json"
    matrix_path.write_text(json.dumps(matrix))
    matrix_path.with_suffix(".sha256").write_text(digest(matrix_path))
    sources = ("scripts/run_fixed_local_controller.py", "carl/icl/local_intent_policy.py",
               "carl/icl/luna_intent_rollout.py", "carl/icl/intent_fence_audit.py")
    plan = {**strict, "status": ADDENDUM_STATUS, "generator_output_normalisation": "none",
            "variants": [name for pair in ADDENDUM_PAIRS for name in pair],
            "contrasts": [list(pair) for pair in ADDENDUM_PAIRS],
            "prediction_presence_at_freeze": {pair[1]: False for pair in ADDENDUM_PAIRS},
            "original_policy_plan_sha256": digest(strict_path), "matrix_plan_sha256": digest(matrix_path),
            "controller_calibration_manifest_sha256": digest(calibration),
            "controller_assets_snapshot_sha256": digest(assets),
            "controller_source_sha256": {name: digest(assessment.ROOT / name) for name in sources}}
    plan_path = run / "policy_lora_addendum_plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_path.with_suffix(".sha256").write_text(digest(plan_path))
    return run, plan, plan_path, matrix, matrix_path, contexts


def rewrite_plan(path, plan):
    path.write_text(json.dumps(plan))
    path.with_suffix(".sha256").write_text(digest(path))


def arguments(run, output):
    return Namespace(plan=run / "policy_lora_addendum_plan.json", data=None, inputs=None, results=None,
                     predictions=None, matrix_plan=None, output=output)


def test_addendum_requires_frozen_pairs_policy_and_actual_board_family(study):
    run, plan, path, matrix, matrix_path, contexts = study
    validate_addendum_plan(plan, path, run, matrix, matrix_path, contexts)
    bad = {**plan, "contrasts": [list(ADDENDUM_PAIRS[0])]}
    rewrite_plan(path, bad)
    with pytest.raises(ValueError, match="matched pairs"):
        validate_addendum_plan(bad, path, run, matrix, matrix_path, contexts)
    bad = {**plan, "generator_output_normalisation": "remove_one_outer_fence"}
    rewrite_plan(path, bad)
    with pytest.raises(ValueError, match="strict generator"):
        validate_addendum_plan(bad, path, run, matrix, matrix_path, contexts)
    rewrite_plan(path, plan)
    changed = [dict(row) for row in contexts]
    changed[0]["start_observation"] = board(10)
    with pytest.raises(ValueError, match="dimensions or box count"):
        validate_addendum_plan(plan, path, run, matrix, matrix_path, changed)
    path.with_suffix(".sha256").write_text("wrong")
    with pytest.raises(ValueError, match="checksum"):
        validate_addendum_plan(plan, path, run, matrix, matrix_path, contexts)


def test_missing_runs_have_null_outcomes_not_fabricated_failures(study):
    run = study[0]
    output = run / "addendum-assessment"
    assess(arguments(run, output))
    summary = json.loads((output / "summary.json").read_text())
    assert summary["config"]["analysis_kind"] == AnalysisKind.ADDENDUM
    assert len(summary["results"]) == 18
    overall = [row for row in summary["results"] if row["family"] == "all"]
    assert len(overall) == 6
    assert all(row["planned"] == 40 and row["statuses"] == {"missing_eligibility": 40} for row in overall)
    assert all(row["pipeline_evaluated"] == row["controller_evaluated"] == 0 for row in overall)
    assert len(summary["paired"]) == 3 * 3 * 2 * 17
    assert {(row["reference"], row["candidate"]) for row in summary["paired"]} == set(ADDENDUM_PAIRS)
    assert all(row["analysis_kind"] == "exploratory_addendum" and row["paired_contexts"] == 0
               and row["board_mean_delta"] is None for row in summary["paired"])
    assert {row["budget"] for row in summary["curves"]} == set(range(17))
    assert all(row["analysis_kind"] == "exploratory_addendum" for row in summary["curves"])
    with (output / "paired.csv").open(newline="", encoding="utf-8") as stream:
        assert {row["analysis_kind"] for row in csv.DictReader(stream)} == {"exploratory_addendum"}
    assert "Eksploracyjne porównanie LoRA" in (output / "controller_budget_all.svg").read_text()
    assert json.loads((output / "complete.json").read_text())["analysis_kind"] == "exploratory_addendum"


def test_three_matched_pairs_keep_invalid_and_infrastructure_separate(study, monkeypatch):
    run, _, _, _, _, contexts = study
    calls = []

    def decisions(_path, _predictions, variant, *_args):
        rows = {row["record_id"]: {"status": "missing_generation"} for row in contexts}
        rows["id-0"] = {"status": "invalid_representation"}
        rows["id-1"] = {"status": "ready"}
        rows["id-2"] = {"status": "ready"}
        rows["id-3"] = {"status": "invalid_intent"}
        return rows, {"status": "ready", "path": variant, "manifest": None, "records": {}}

    def rollouts(_path, variant, *_args):
        calls.append(variant)
        actions = ["RIGHT"] * (2 if variant.startswith("rep_lora_") else 1)
        good = {"result": {"status": "achieved", "achieved": True, "actions": actions,
                           "steps": [{"no_op": False} for _ in actions],
                           "model_calls": len(actions), "task_solved": False}}
        error = {"result": {"status": "api_error", "achieved": False, "actions": [], "steps": [],
                            "model_calls": 1, "task_solved": False}}
        return {"id-1": good, "id-2": error}, {"manifest": None, "audit": {"rollouts": 2}}

    monkeypatch.setattr(assessment, "eligibility_rows", decisions)
    monkeypatch.setattr(assessment, "controller_rows", rollouts)
    output = run / "paired-assessment"
    assess(arguments(run, output))
    summary = json.loads((output / "summary.json").read_text())
    assert calls == [name for pair in ADDENDUM_PAIRS for name in pair]
    by_budget = {(row["reference"], row["candidate"], row["metric"], row["family"], row["budget"]): row
                 for row in summary["paired"]}
    for before, after in ADDENDUM_PAIRS:
        first = by_budget[(before, after, "pipeline", "all", 1)]
        assert first["planned_pairs"] == 40 and first["paired_contexts"] == 3
        assert first["reference_success"] == 1 and first["candidate_success"] == 0
        assert first["board_mean_delta"] == pytest.approx(-1 / 3)
        second = by_budget[(before, after, "pipeline", "all", 2)]
        assert second["candidate_success"] == second["reference_success"] == 1
        assert second["board_mean_delta"] == 0
        conditional = by_budget[(before, after, "conditional", "all", 1)]
        assert conditional["paired_contexts"] == 1 and conditional["board_mean_delta"] == -1
        ten_boxes = by_budget[(before, after, "pipeline", "proposal_gamma2_12x12x10", 1)]
        assert ten_boxes["planned_pairs"] == 20 and ten_boxes["paired_contexts"] == 0
        assert ten_boxes["board_mean_delta"] is None


def test_addendum_cannot_write_into_preregistered_outputs(study):
    run = study[0]
    with pytest.raises(ValueError, match="separate"):
        assess(arguments(run, run / "controller_assessment" / "child"))


def test_cli_delegates_to_shared_assessor_with_separate_output(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(addendum_entry, "assess", lambda args: called.append(args))
    monkeypatch.setattr(sys, "argv", ["assessment", "--run", str(tmp_path), "--stop-epoch", "12"])
    addendum_entry.main()
    assert len(called) == 1
    assert called[0].plan == tmp_path / "policy_lora_addendum_plan.json"
    assert called[0].output == tmp_path / "controller_lora_addendum_assessment"
    assert called[0].stop_epoch == 12 and called[0].inputs is None and called[0].results is None
