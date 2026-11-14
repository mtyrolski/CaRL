"""Controller achievement curves distinguish model failures from missing work."""

import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.assess_fixed_controller as controller_assessment
from carl.icl.intent_representation_experiment import messages_for
from carl.icl.local_intent_policy import run_record
from scripts.assess_fixed_controller import AssessmentDeadline
from scripts.assess_fixed_controller import GeneratorNormalisation
from scripts.assess_fixed_controller import check_deadline
from scripts.assess_fixed_controller import check_test_subset
from scripts.assess_fixed_controller import classify
from scripts.assess_fixed_controller import compare
from scripts.assess_fixed_controller import curve
from scripts.assess_fixed_controller import main
from scripts.assess_fixed_controller import normalisation_from_plan
from scripts.assess_fixed_controller import reconstruct_generator_answer
from scripts.assess_fixed_controller import run_posthoc_fallback
from scripts.assess_fixed_controller import success
from scripts.assess_fixed_controller import validate_normalisation_metadata
from scripts.run_fixed_local_controller import evaluate_record

BOARD = ["#####", "#####", "#@$.#", "#####", "#####"]
INTENT = {"effects": [{"predicate": "boxes_on_goals_at_least", "count": 1}], "preserve": [], "instruction": None}
ACTION_PARSER = "Remove at most one complete outer plain or json Markdown fence, then require strict action JSON."


def record(identifier="a", layout="board"):
    return {"record_id": identifier, "layout_id": layout, "split": "test", "start_observation": BOARD,
            "task_goal": "Put every box on a goal.", "evaluation_family": "real_5x5x1"}


def item(identifier="a", layout="board"):
    return {"record_id": identifier, "layout_id": layout, "family": "real_5x5x1", "variant": "base",
            "split": "test", "start_observation": BOARD, "task_goal": "Put every box on a goal.", "intent": INTENT}


def test_invalid_generation_is_pipeline_failure_but_not_controller_trial():
    bad = classify(record(), {"status": "invalid_representation"}, None)
    missing = classify(record("b", "board2"), {"status": "missing_generation"}, None)
    unrun = classify(record("c", "board3"), {"status": "ready"}, None)
    pipeline = curve("base", [bad, missing, unrun], 16, "pipeline")
    conditional = curve("base", [bad, missing, unrun], 16, "conditional")
    assert (pipeline["planned"], pipeline["assessed"], pipeline["missing_infrastructure"], pipeline["successes"]) == (3, 1, 2, 0)
    assert pipeline["invalid_generation"] == 1 and conditional["invalid_generation"] == 1
    assert conditional["assessed"] == 0 and conditional["board_mean"] is None


def test_controller_subset_must_belong_to_the_full_frozen_matrix():
    check_test_subset(["a", "b", "c"], [record("a"), record("c")])
    with pytest.raises(ValueError, match="unique subset"):
        check_test_subset(["a", "b"], [record("a"), record("c")])
    with pytest.raises(ValueError, match="unique subset"):
        check_test_subset(["a", "a"], [record("a")])


def test_posthoc_removes_only_one_complete_fence_without_changing_raw_verification():
    raw = "```json\n" + json.dumps(INTENT) + "\n```"
    prediction = {"representation": "json", "raw_output": raw, "syntax_valid": False, "parsed_intent": None}
    strict, strict_details = reconstruct_generator_answer(prediction, GeneratorNormalisation.NONE)
    relaxed, details = reconstruct_generator_answer(prediction, GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE)
    assert strict is None and not strict_details["fence_removed"]
    assert relaxed == INTENT and details["fence_removed"] and details["raw_output"] == raw
    assert details["normalised_output"] == json.dumps(INTENT)
    with pytest.raises(ValueError, match="Recorded raw parse"):
        reconstruct_generator_answer({**prediction, "syntax_valid": True}, GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE)
    nested = {**prediction, "raw_output": "```json\n" + raw + "\n```"}
    result, nested_details = reconstruct_generator_answer(nested, GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE)
    assert result is None and not nested_details["fence_removed"]


def test_posthoc_manifest_binds_raw_data_and_normaliser():
    normaliser = Path("carl/icl/intent_fence_audit.py")
    value = {"mode": "remove_one_outer_fence", "posthoc": True,
             "normaliser_source_sha256": hashlib.sha256(normaliser.read_bytes()).hexdigest(),
             "raw_predictions_sha256": "raw-sha"}
    metadata = {"predictions_sha256": "raw-sha", "generator_output_normalisation": value}
    validate_normalisation_metadata(metadata, GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE)
    validate_normalisation_metadata({}, GeneratorNormalisation.NONE)
    with pytest.raises(ValueError, match="normalisation provenance"):
        validate_normalisation_metadata(metadata, GeneratorNormalisation.NONE)
    with pytest.raises(ValueError, match="normalisation provenance"):
        validate_normalisation_metadata({"predictions_sha256": "other", "generator_output_normalisation": value},
                                        GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE)


def test_posthoc_plan_must_bind_same_cases_controller_budget_and_hashes(tmp_path):
    strict = {"frozen_ids": "ids.json", "frozen_ids_sha256": "ids", "controller_selection_sha256": "selection",
              "selected_controller": {"model": "fixed"}, "decoding": "greedy", "action_parser": "fixed",
              "model_training": False, "primitive_action_budget": 16, "variants": ["base"]}
    strict_path = tmp_path / "policy_evaluation_plan.json"
    strict_path.write_text(json.dumps(strict))
    plan = {**strict, "status": "posthoc_exploratory", "generator_output_normalisation": "remove_one_outer_fence",
            "primary_policy_plan_sha256": hashlib.sha256(strict_path.read_bytes()).hexdigest()}
    path = tmp_path / "policy_evaluation_fence_plan.json"
    path.write_text(json.dumps(plan))
    path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    assert normalisation_from_plan(plan, path, tmp_path) == GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE
    modified = {**plan, "primitive_action_budget": 8}
    path.write_text(json.dumps(modified))
    path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="same cases, policy and budget"):
        normalisation_from_plan(modified, path, tmp_path)
    path.with_suffix(".sha256").write_text("wrong")
    with pytest.raises(ValueError, match="checksum"):
        normalisation_from_plan(modified, path, tmp_path)


def test_posthoc_fallback_runs_once_and_uses_recovery_for_incomplete_output(tmp_path, monkeypatch):
    strict = {"frozen_ids": "ids", "frozen_ids_sha256": "ids", "controller_selection_sha256": "selection",
              "selected_controller": {"model": "fixed"}, "decoding": "greedy", "action_parser": "fixed",
              "model_training": False, "primitive_action_budget": 16, "variants": ["base"]}
    strict_path = tmp_path / "policy_evaluation_plan.json"
    strict_path.write_text(json.dumps(strict))
    fence = {**strict, "status": "posthoc_exploratory", "generator_output_normalisation": "remove_one_outer_fence",
             "primary_policy_plan_sha256": hashlib.sha256(strict_path.read_bytes()).hexdigest()}
    fence_path = tmp_path / "policy_evaluation_fence_plan.json"
    fence_path.write_text(json.dumps(fence))
    fence_path.with_suffix(".sha256").write_text(hashlib.sha256(fence_path.read_bytes()).hexdigest())
    (tmp_path / "controller_assessment_fence").mkdir()  # Interrupted previous attempt.
    primary = tmp_path / "strict_assessment"
    primary.mkdir()
    args = Namespace(plan=strict_path, output=primary, fallback_max_seconds=30, inputs=None, results=None,
                     no_posthoc_fallback=False)
    calls = []
    def fake_assess(recovery):
        calls.append(recovery)
        recovery.output.mkdir()
        summary = recovery.output / "summary.json"
        summary.write_text("{}")
        (recovery.output / "complete.json").write_text(json.dumps({
            "status": "complete", "analysis_kind": "posthoc_exploratory",
            "plan_sha256": hashlib.sha256(fence_path.read_bytes()).hexdigest(),
            "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest()}))
        return GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE
    monkeypatch.setattr(controller_assessment, "assess", fake_assess)
    run_posthoc_fallback(args)
    report = json.loads((primary / "posthoc_fallback.json").read_text())
    assert report["status"] == "complete" and len(calls) == 1
    assert calls[0].output.name == "controller_assessment_fence_recovery" and calls[0].no_posthoc_fallback
    run_posthoc_fallback(args)
    assert json.loads((primary / "posthoc_fallback.json").read_text())["status"] == "already_complete"
    assert len(calls) == 1


def test_fallback_deadline_is_enforced_without_killing_unrelated_processes():
    with pytest.raises(AssessmentDeadline):
        check_deadline(Namespace(stop_monotonic=time.monotonic() - 1))


def test_first_satisfaction_sets_exact_integer_budget():
    rollout = run_record(item(), lambda _: '{"action":"RIGHT"}', 4)
    row = classify(record(), {"status": "ready"}, rollout)
    assert row["status"] == "achieved" and row["first_success_action"] == 1
    assert not success(row, 0) and success(row, 1) and success(row, 4)
    assert row["actions"] == 1 and row["no_ops"] == 0 and row["model_calls"] == 1


def test_invalid_response_counts_failure_and_local_inference_error_is_missing():
    invalid = run_record(item(), lambda _: "UP", 4)
    def fail(_):
        raise TimeoutError("local inference failed")
    failure = run_record(item(), fail, 4)
    model = classify(record(), {"status": "ready"}, invalid)
    infra = classify(record("b", "board2"), {"status": "ready"}, failure)
    assert model["status"] == "invalid_response" and model["controller_evaluated"]
    assert infra["status"] == "controller_infrastructure_error" and not infra["pipeline_evaluated"]


def test_paired_comparison_weights_layouts_and_excludes_unmeasured():
    rows = []
    for identifier, layout, left, right in [("a", "b1", None, 1), ("b", "b1", None, 1), ("c", "b2", 1, None)]:
        for variant, first in (("a", left), ("b", right)):
            rows.append({"variant": variant, "record_id": identifier, "layout_id": layout, "family": "x4",
                         "pipeline_evaluated": True, "controller_evaluated": True, "first_success_action": first})
    rows.extend([{"variant": "a", "record_id": "d", "layout_id": "b3", "family": "x4",
                  "pipeline_evaluated": True, "controller_evaluated": True, "first_success_action": 1},
                 {"variant": "b", "record_id": "d", "layout_id": "b3", "family": "x4",
                  "pipeline_evaluated": False, "controller_evaluated": False, "first_success_action": None}])
    result = compare(rows, "a", "b", 4, "pipeline")
    assert result["planned_pairs"] == 4 and result["paired_contexts"] == 3 and result["paired_boards"] == 2
    assert result["board_mean_delta"] == 0 and result["unpaired_contexts"] == 1


def test_end_to_end_verifies_records_and_replays_observed_controller(tmp_path, monkeypatch):
    run = tmp_path
    (run / "data").mkdir()
    inputs, outputs = run / "controller_inputs/base", run / "controller_results"
    inputs.mkdir(parents=True)
    outputs.mkdir()
    predictions_dir = run / "predictions"
    predictions_dir.mkdir()
    identifiers = ["a", "b", "c", "d", "e", "f"]
    rows = [record(key, "board_" + key) for key in identifiers]
    data = run / "data/dataset.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def entry(path):
        return {"path": str(path.resolve()), "sha256": digest(path)}
    ids = run / "ids.json"
    ids.write_text(json.dumps({"record_ids": identifiers, "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}))
    plan = {"frozen_ids": str(ids), "frozen_ids_sha256": hashlib.sha256(ids.read_bytes()).hexdigest(),
            "variants": ["base"], "primitive_action_budget": 4,
            "selected_controller": {"model": "local", "revision": "frozen", "device": "cpu", "max_new_tokens": 32,
                                    "max_context_tokens": 2048, "action_cap": 4}}
    plan_file = run / "plan.json"
    plan_file.write_text(json.dumps(plan))
    already_true = {"effects": [{"predicate": "box_at", "row": 2, "col": 2}], "preserve": [], "instruction": None}
    proposed = []
    for row in rows:
        if row["record_id"] == "c":
            continue
        intent = already_true if row["record_id"] == "b" else INTENT
        proposed.append({"record_id": row["record_id"], "layout_id": row["layout_id"], "variant": "base", "split": "test",
                         "representation": "json", "support_set": "zero", "raw_output": json.dumps(intent),
                         "syntax_valid": True, "parsed_intent": intent})
    prediction_path = predictions_dir / "base.jsonl"
    prediction_path.write_text("\n".join(map(json.dumps, proposed)) + "\n")
    protocol = Path("carl/icl/intent_representation_experiment.py").resolve()
    generator = Path("carl/icl/intent_generator_experiment.py").resolve()
    selector = Path("carl/icl/intent_in_context.py").resolve()
    support_data, support_manifest = run / "support.jsonl", run / "support.json"
    support_data.write_text("")
    support_manifest.write_text("{}")
    prompts_path = prediction_path.with_suffix(".prompts.jsonl")
    prompts = [{"record_id": row["record_id"], "layout_id": row["layout_id"],
                "messages": messages_for(row, [], "json")} for row in rows]
    prompts_path.write_text("\n".join(map(json.dumps, prompts)) + "\n")
    settings = {"seed": 1729, "dtype": "float16", "threads": 1, "max_length": 1024, "max_new_tokens": 128,
                "decoding": "greedy", "enable_thinking": False, "split": "test", "require_cuda": False}
    model = {"path": str((run / "fake-model").resolve()), "revision": "revision", "sha256": {
        "model.safetensors": "weights", "config.json": "config", "tokenizer.json": "tokenizer"}}
    method = {"name": "base", "representation": "json", "support_set": "zero", "adapter": None,
              "adapter_pending": False, "adapter_sha256": None, "output": str(prediction_path)}
    matrix = {"run": str(run), "inputs": {"data": entry(data), "test_ids": entry(ids),
              "support_data": entry(support_data), "support_manifest": entry(support_manifest),
              "protocol": entry(protocol), "generator": entry(generator), "support_selector": entry(selector)},
              "model": model, "settings": settings, "methods": [method]}
    matrix_path = run / "matrix_plan.json"
    matrix_path.write_text(json.dumps(matrix))
    matrix_path.with_suffix(".sha256").write_text(digest(matrix_path))
    run_metadata = {"variant": "base", "representation": "json", "support_set": "zero", "model": model["path"],
                    "revision": model["revision"], "model_weights_sha256": "weights", "model_config_sha256": "config",
                    "tokenizer_sha256": "tokenizer", "adapter": None, "adapter_weights_sha256": None,
                    "adapter_config_sha256": None, "data_sha256": digest(data), "test_ids_sha256": digest(ids),
                    "support_data_sha256": digest(support_data), "support_manifest_sha256": digest(support_manifest),
                    "protocol_sha256": digest(protocol), "generator_sha256": digest(generator),
                    "support_selection_sha256": digest(selector), "prompts_sha256": digest(prompts_path), **settings}
    run_path = prediction_path.with_suffix(".run.json")
    run_path.write_text(json.dumps(run_metadata))
    statuses = ["ready", "invalid_intent", "missing_generation", "ready", "ready", "ready"]
    decisions = [{"record_id": row["record_id"], "layout_id": row["layout_id"], "family": "real_5x5x1",
                  "variant": "base", "status": status} for row, status in zip(rows, statuses, strict=True)]
    # The final ready record is unstarted; earlier ready results are a prefix.
    cases = [item(key, "board_" + key) for key in ("a", "d", "e", "f")]
    decision_path, records_path = inputs / "eligibility.jsonl", inputs / "records.jsonl"
    decision_path.write_text("\n".join(map(json.dumps, decisions)) + "\n")
    records_path.write_text("\n".join(map(json.dumps, cases)) + "\n")
    (inputs / "manifest.json").write_text(json.dumps({"variant": "base", "test_ids_sha256": digest(ids),
                                                       "data_sha256": digest(data), "eligibility_sha256": digest(decision_path),
                                                       "records_sha256": digest(records_path),
                                                       "predictions_sha256": digest(prediction_path),
                                                       "matrix_plan_sha256": digest(matrix_path),
                                                       "prediction_run_sha256": digest(run_path),
                                                       "prompts_sha256": digest(prompts_path)}))
    good = evaluate_record(cases[0], lambda _: '{"action":"RIGHT"}', 4)
    invalid = evaluate_record(cases[1], lambda _: "UP", 4)
    calls = 0
    def fail(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            return '{"action":"UP"}'
        raise TimeoutError("local inference")
    infra = evaluate_record(cases[2], fail, 4)
    result_path = outputs / "base.jsonl"
    result_path.write_text("\n".join(map(json.dumps, [good, invalid, infra])) + "\n")
    result_path.with_suffix(".manifest.json").write_text(json.dumps({
        "config": plan["selected_controller"], "records_sha256": digest(records_path),
        "record_ids": [row["record_id"] for row in cases], "policy_config_sha256": "fixed",
        "weights_sha256": "fixed", "tokenizer_sha256": "fixed", "source_sha256": {}, "action_parser": ACTION_PARSER,
    }))
    output = run / "assessment"
    monkeypatch.setattr(sys, "argv", ["assessment", "--plan", str(plan_file), "--output", str(output)])
    main()
    summary = json.loads((output / "summary.json").read_text())
    overall = next(row for row in summary["results"] if row["family"] == "all")
    assert overall["planned"] == 6 and overall["pipeline_evaluated"] == 3
    assert overall["controller_evaluated"] == 2 and overall["controller_success"] == 1
    assert overall["statuses"]["controller_infrastructure_error"] == 1 and overall["statuses"]["missing_controller"] == 1
    assert (overall["attempted_actions"], overall["no_ops"], overall["model_calls"]) == (2, 1, 4)
    curves = [row for row in summary["curves"] if row["family"] == "all" and row["metric"] == "pipeline"]
    assert [row["successes"] for row in curves] == [0, 1, 1, 1, 1]
    assert (output / "curves.csv").exists() and (output / "controller_budget_all.svg").exists()

    # Run the same cases and controller through the separate post-hoc path:
    # one raw response becomes parseable only after a complete outer fence.
    strict_path = run / "policy_evaluation_plan.json"
    strict_path.write_text(json.dumps(plan))
    fence_plan = {**plan, "status": "posthoc_exploratory", "generator_output_normalisation": "remove_one_outer_fence",
                  "primary_policy_plan_sha256": digest(strict_path)}
    fence_plan_path = run / "policy_evaluation_fence_plan.json"
    fence_plan_path.write_text(json.dumps(fence_plan))
    fence_plan_path.with_suffix(".sha256").write_text(digest(fence_plan_path))
    original_predictions = prediction_path.read_text()
    fenced = [dict(row) for row in proposed]
    fenced[0].update(raw_output="```json\n" + fenced[0]["raw_output"] + "\n```", syntax_valid=False, parsed_intent=None)
    prediction_path.write_text("\n".join(map(json.dumps, fenced)) + "\n")
    fence_inputs = run / "controller_fence_inputs/base"
    fence_inputs.mkdir(parents=True)
    by_id = {row["record_id"]: row for row in fenced}
    fence_decisions = []
    for decision in decisions:
        decision = dict(decision)
        if decision["record_id"] in by_id:
            _, diagnostics = reconstruct_generator_answer(by_id[decision["record_id"]],
                                                           GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE)
            decision.update(diagnostics)
        fence_decisions.append(decision)
    fence_eligibility, fence_records = fence_inputs / "eligibility.jsonl", fence_inputs / "records.jsonl"
    fence_eligibility.write_text("\n".join(map(json.dumps, fence_decisions)) + "\n")
    fence_records.write_bytes(records_path.read_bytes())
    normaliser = Path("carl/icl/intent_fence_audit.py")
    fence_manifest = json.loads((inputs / "manifest.json").read_text())
    fence_manifest.update(eligibility_sha256=digest(fence_eligibility), records_sha256=digest(fence_records),
                          predictions_sha256=digest(prediction_path), generator_output_normalisation={
                              "mode": "remove_one_outer_fence", "posthoc": True,
                              "normaliser_source_sha256": digest(normaliser),
                              "raw_predictions_sha256": digest(prediction_path)})
    (fence_inputs / "manifest.json").write_text(json.dumps(fence_manifest))
    fence_results = run / "controller_fence_results"
    fence_results.mkdir()
    (fence_results / "base.jsonl").write_bytes(result_path.read_bytes())
    (fence_results / "base.manifest.json").write_bytes(result_path.with_suffix(".manifest.json").read_bytes())
    fence_output = run / "controller_fence_assessment"
    monkeypatch.setattr(sys, "argv", ["assessment", "--plan", str(fence_plan_path), "--output", str(fence_output)])
    main()
    fence_summary = json.loads((fence_output / "summary.json").read_text())
    assert fence_summary["config"]["analysis_kind"] == "posthoc_exploratory"
    assert fence_summary["config"]["generator_output_normalisation"] == "remove_one_outer_fence"
    assert fence_summary["results"][0]["controller_success"] == 1
    assert fence_summary["curves"][0]["analysis_kind"] == "posthoc_exploratory"

    # A strict assessment automatically recovers the independent post-hoc
    # report if its usual directory has not been completed by the runner.
    auto_output = run / "strict_with_fallback"
    monkeypatch.setattr(sys, "argv", ["assessment", "--plan", str(strict_path), "--output", str(auto_output),
                                      "--fallback-max-seconds", "60"])
    main()
    fallback = json.loads((auto_output / "posthoc_fallback.json").read_text())
    assert fallback["status"] == "complete"
    recovered = json.loads((run / "controller_assessment_fence/summary.json").read_text())
    assert recovered["config"]["analysis_kind"] == "posthoc_exploratory"
    assert recovered["results"][0]["controller_success"] == 1
    prediction_path.write_text(original_predictions)

    # Even if the preparer's hash is updated, a run using another support is
    # rejected against the independently frozen matrix settings.
    original_run = run_path.read_text()
    modified = dict(run_metadata, support_set="balanced0")
    run_path.write_text(json.dumps(modified))
    input_manifest = json.loads((inputs / "manifest.json").read_text())
    input_manifest["prediction_run_sha256"] = digest(run_path)
    (inputs / "manifest.json").write_text(json.dumps(input_manifest))
    monkeypatch.setattr(sys, "argv", ["assessment", "--plan", str(plan_file), "--output", str(run / "bad_metadata"),
                                      "--no-posthoc-fallback"])
    main()
    rejected = json.loads((run / "bad_metadata/rejected_variants.json").read_text())
    assert rejected[0]["variant"] == "base" and "differs in support_set" in rejected[0]["reason"]
    bad_summary = json.loads((run / "bad_metadata/summary.json").read_text())
    assert bad_summary["results"][0]["statuses"] == {"rejected_infrastructure": 6}

    # Replacing the input intent with a different valid target is rejected,
    # even after all local input hashes are recomputed.
    run_path.write_text(original_run)
    input_manifest["prediction_run_sha256"] = digest(run_path)
    cases[0]["intent"] = {"effects": [{"predicate": "player_at", "row": 2, "col": 3}],
                           "preserve": [], "instruction": None}
    records_path.write_text("\n".join(map(json.dumps, cases)) + "\n")
    input_manifest["records_sha256"] = digest(records_path)
    (inputs / "manifest.json").write_text(json.dumps(input_manifest))
    monkeypatch.setattr(sys, "argv", ["assessment", "--plan", str(plan_file), "--output", str(run / "forged_intent"),
                                      "--no-posthoc-fallback"])
    main()
    rejected = json.loads((run / "forged_intent/rejected_variants.json").read_text())
    assert "violates the board/intent contract" in rejected[0]["reason"]
