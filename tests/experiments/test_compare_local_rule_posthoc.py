"""Paired rule comparison must retain missing cases and layout independence."""

import argparse
import hashlib
import json

import pytest

from carl.icl.intent_representation_experiment import parse_representation
from scripts.compare_local_rule_posthoc import compare
from scripts.compare_local_rule_posthoc import hash_inputs
from scripts.compare_local_rule_posthoc import index_predictions
from scripts.compare_local_rule_posthoc import index_rows
from scripts.compare_local_rule_posthoc import input_paths
from scripts.compare_local_rule_posthoc import inspect_snapshot
from scripts.compare_local_rule_posthoc import rejected_methods
from scripts.compare_local_rule_posthoc import supervisor_provenance
from scripts.compare_local_rule_posthoc import verify_generator_sources
from scripts.compare_local_rule_posthoc import verify_links
from scripts.compare_local_rule_posthoc import write_provenance


def context(identifier, layout, family="real_12x12x4", prior=False):
    return {"record_id": identifier, "layout_id": layout, "split": "test", "evaluation_family": family,
            "previously_evaluated_in_lora_matrix": prior, "start_observation": ["#####", "#@$.#", "#####"]}


def valid_intent():
    return {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": [], "instruction": None}


def source_row(record, variant, *, valid=True, state="evaluated"):
    result = {"record_id": record["record_id"], "layout_id": record["layout_id"],
              "family": record["evaluation_family"], "variant": variant, "status": state}
    if state != "evaluated":
        return result
    raw = json.dumps(valid_intent()) if valid else 'not JSON'
    result.update(start_observation=record["start_observation"], raw_output=raw,
                  generation_seconds=None, generated_tokens=None)
    if variant == "local_rule":
        result.update(strict_valid=valid, strict_whole_goal=False, strict_intent=valid_intent() if valid else None)
    else:
        result.update(representation="json", support_set="zero", syntax_valid=valid, semantic_valid=valid,
                      valid_not_whole_goal=valid, intent=valid_intent() if valid else None)
    return result


def prediction_row(record, variant, *, valid=True):
    result = {"record_id": record["record_id"], "layout_id": record["layout_id"], "split": "test",
              "raw_output": json.dumps(valid_intent()) if valid else "not JSON",
              "generation_seconds": None, "generated_tokens": None}
    if variant == "local_rule":
        result.update(method=variant, json_valid=valid, parsed=valid_intent() if valid else None)
    else:
        result.update(variant=variant, representation="json", support_set="zero", syntax_valid=valid,
                      parsed_intent=valid_intent() if valid else None)
    return result


def test_paired_counts_treat_model_failures_and_missing_distinctly():
    cases = [context("a", "a"), context("b", "b"), context("c", "c"), context("d", "d"),
             context("e", "e"), context("f", "f")]
    rule = {r["record_id"]: source_row(r, "local_rule") for r in cases}
    model = {"a": source_row(cases[0], "model"), "b": source_row(cases[1], "model", valid=False),
             "c": source_row(cases[2], "model", state="missing"),
             "d": source_row(cases[3], "model", state="rejected_infrastructure"),
             "e": source_row(cases[4], "model"), "f": source_row(cases[5], "model")}
    summary, rows = compare("model", cases, rule, model, "all", "all")
    assert summary["planned_contexts"] == 6 and summary["paired_contexts"] == 4
    assert summary["generator_missing"] == 1 and summary["generator_infrastructure_errors"] == 1
    assert summary["both_valid"] == 3 and summary["local_only_valid"] == 1
    assert summary["generator_minus_local_board_mean"] == -.25
    assert summary["bootstrap_low"] is None and summary["bootstrap_high"] is None
    assert len(rows) == 6 and rows[2]["delta"] is None and rows[3]["delta"] is None


def test_bootstrap_uses_layout_means_even_with_multiple_fragments():
    cases = [context("a1", "a"), context("a2", "a"), *[context(c, c) for c in "bcde"]]
    rule = {r["record_id"]: source_row(r, "local_rule") for r in cases}
    model = {r["record_id"]: source_row(r, "model") for r in cases}
    model["a1"] = source_row(cases[0], "model", valid=False)
    summary, _ = compare("model", cases, rule, model, "all", "all")
    assert summary["paired_contexts"] == 6 and summary["paired_boards"] == 5
    assert summary["generator_minus_local_board_mean"] == pytest.approx(-.1)
    assert summary["bootstrap_low"] is not None and summary["bootstrap_high"] is not None


def test_degenerate_bootstrap_is_explicit_and_family_filter_is_paired():
    cases = [context(str(i), str(i), family="real_12x12x4" if i < 5 else "generated_v1_12x12x10")
             for i in range(8)]
    rule = {r["record_id"]: source_row(r, "local_rule") for r in cases}
    model = {r["record_id"]: source_row(r, "model", valid=False) for r in cases}
    overall, _ = compare("model", cases, rule, model, "all", "all")
    subset, _ = compare("model", cases, rule, model, "real_12x12x4", "all")
    assert overall["generator_minus_local_board_mean"] == -1
    assert overall["bootstrap_low"] == overall["bootstrap_high"] == -1 and overall["degenerate_interval"]
    assert subset["planned_boards"] == subset["paired_boards"] == 5


def test_assessment_provenance_and_raw_validity_are_checked():
    source = context("a", "a")
    expected = {"a": source}
    local = source_row(source, "local_rule")
    model = source_row(source, "model")
    assert index_rows([local], expected, local_rule=True)["local_rule"]["a"] == local
    assert index_rows([model], expected, local_rule=False, parser=parse_representation)["model"]["a"] == model
    with pytest.raises(ValueError, match="layout or family"):
        index_rows([{**model, "layout_id": "different"}], expected, local_rule=False, parser=parse_representation)
    with pytest.raises(ValueError, match="Duplicate"):
        index_rows([local, local], expected, local_rule=True)
    with pytest.raises(ValueError, match="validity"):
        index_rows([{**model, "semantic_valid": False}], expected, local_rule=False, parser=parse_representation)
    with pytest.raises(ValueError, match="current board"):
        index_rows([{**local, "start_observation": ["future"]}], expected, local_rule=True)


def test_provenance_hashes_complete_inputs_and_generated_files(tmp_path):
    data = tmp_path / "dataset.jsonl"
    identifiers = tmp_path / "test_ids.json"
    local = tmp_path / "local"
    generator = tmp_path / "generator"
    output = tmp_path / "output"
    for directory in (local, generator, output):
        directory.mkdir()
    rule_predictions = tmp_path / "local_predictions.jsonl"
    snapshot = tmp_path / "snapshot.py"
    matrix = tmp_path / "matrix.json"
    args = argparse.Namespace(data=data, test_ids=identifiers, local_assessment=local,
                              generator_assessment=generator)
    paths = input_paths(args, {"variants": {"local_rule": {"path": str(rule_predictions)}}},
                        {"parser_path": str(snapshot), "matrix_plan_path": str(matrix)})
    assert len(paths) == 18
    for path in paths:
        if path.is_relative_to(tmp_path):
            path.write_text(f"source: {path.name}", encoding="utf-8")
    hashes = hash_inputs(paths)
    (output / "paired_summary.csv").write_text("variant,count\nmodel,1\n", encoding="utf-8")
    write_provenance(output, hashes, supervisor=None)
    record = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert record["input_sha256"] == hashes and len(record["input_sha256"]) == 18
    assert record["output_sha256"] == {
        str(output / "paired_summary.csv"): hashlib.sha256(b"variant,count\nmodel,1\n").hexdigest()}
    assert record["analysis_command_argv"] and record["supervisor"] is None
    data.write_text("changed", encoding="utf-8")
    assert hash_inputs(paths) != hashes


def test_supervisor_command_is_recorded_and_requires_a_running_stage(tmp_path):
    stage = {"name": "paired", "status": "running", "kind": "evaluation", "timeout_seconds": 180,
             "started_utc": "2026-11-14T09:00:00+00:00", "output": "stage.log",
             "command": ["env", "MPLCONFIGDIR=/tmp/example", "python", "analysis.py"]}
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({"stages": [stage]}), encoding="utf-8")
    result = supervisor_provenance(tmp_path, "paired")
    assert result is not None and result["status_when_snapshotted"] == "running"
    assert result["launch_command_argv"][-4:] == stage["command"]
    assert "--timeout 180" in result["launch_command_shell"]
    with pytest.raises(ValueError, match="both"):
        supervisor_provenance(tmp_path, None)
    stage["status"] = "completed"
    manifest.write_text(json.dumps({"stages": [stage]}), encoding="utf-8")
    with pytest.raises(ValueError, match="running"):
        supervisor_provenance(tmp_path, "paired")


@pytest.mark.parametrize("field,value", [("record_id", "other"), ("layout_id", "other"), ("split", "train"),
                                       ("variant", "other"), ("representation", "formal"),
                                       ("support_set", "balanced0")])
def test_prediction_context_and_protocol_must_match(field, value):
    record = context("a", "a")
    prediction = {**prediction_row(record, "model"), field: value}
    with pytest.raises(ValueError, match="context|method"):
        index_predictions([prediction], {"a": record}, method={"name": "model", "representation": "json",
                                                                  "support_set": "zero"},
                          local_rule=False, parser=parse_representation)


def test_assessment_links_reject_raw_mismatches_and_false_missing():
    record = context("a", "a")
    assessment = source_row(record, "model")
    prediction = prediction_row(record, "model")
    method = {"name": "model", "representation": "json", "support_set": "zero"}
    indexed = index_predictions([prediction], {"a": record}, method=method, local_rule=False,
                                parser=parse_representation)
    verify_links({"a": assessment}, indexed, local_rule=False)
    altered_intent = {**valid_intent(), "effects": [{"predicate": "box_at", "row": 1, "col": 2}]}
    altered = {**prediction, "raw_output": json.dumps(altered_intent), "parsed_intent": altered_intent}
    altered_index = index_predictions([altered], {"a": record}, method=method, local_rule=False,
                                      parser=parse_representation)
    with pytest.raises(ValueError, match="differs"):
        verify_links({"a": assessment}, altered_index, local_rule=False)
    with pytest.raises(ValueError, match="existing prediction"):
        verify_links({"a": source_row(record, "model", state="missing")}, indexed, local_rule=False)
    with pytest.raises(ValueError, match="absent prediction"):
        verify_links({"a": assessment}, {}, local_rule=False)
    verify_links({"a": source_row(record, "model", state="missing")}, {}, local_rule=False)
    verify_links({}, indexed, local_rule=False)  # Interrupted assessment: prediction exists, assessment does not.
    failed = {**prediction, "status": "generation_error", "error": "out of memory"}
    failure = {**source_row(record, "model", state="infrastructure_error"), "error": "out of memory"}
    verify_links({"a": failure}, {"a": failed}, local_rule=False)
    with pytest.raises(ValueError, match="Infrastructure"):
        verify_links({"a": {**failure, "error": "other"}}, {"a": failed}, local_rule=False)
    rejected = {**source_row(record, "model", state="rejected_infrastructure"), "error": "bad metadata"}
    verify_links({"a": rejected}, {}, local_rule=False, rejected_reason="bad metadata")
    with pytest.raises(ValueError, match="contradictory"):
        verify_links({"a": rejected}, {}, local_rule=False, rejected_reason="other")


def test_local_assessment_is_linked_to_rule_prediction():
    record = context("a", "a")
    assessment = source_row(record, "local_rule")
    prediction = prediction_row(record, "local_rule")
    indexed = index_predictions([prediction], {"a": record}, method={"name": "local_rule"}, local_rule=True)
    verify_links({"a": assessment}, indexed, local_rule=True)
    with pytest.raises(ValueError, match="differs"):
        verify_links({"a": {**assessment, "raw_output": "different"}}, indexed, local_rule=True)
    with pytest.raises(ValueError, match="duplicate"):
        index_predictions([prediction, prediction], {"a": record}, method={"name": "local_rule"}, local_rule=True)


def test_snapshot_hash_manifest_and_missing_files_fail_closed(tmp_path):
    filename = "model.jsonl"
    payload = b'{"record_id":"a"}\n{"truncated"'
    (tmp_path / filename).write_bytes(payload)
    entries = {filename: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)},
               "model.run.json": {"sha256": None, "bytes": None},
               "model.prompts.jsonl": {"sha256": None, "bytes": None}}
    manifest = {"schema_version": 1, "matrix_plan_sha256": "plan", "variants": [
        {"variant": "model", "files": entries}]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {"matrix_plan_sha256": "plan", "input_snapshot_manifest_sha256": hashlib.sha256(
        manifest_path.read_bytes()).hexdigest(), "variants": {"model": {"sha256": entries[filename]["sha256"]}}}
    matrix = {"methods": [{"name": "model"}]}
    inspected, sources = inspect_snapshot(tmp_path, config, matrix)
    assert set(inspected) == {"model"} and set(sources) == {manifest_path, tmp_path / filename}
    (tmp_path / "model.run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected"):
        inspect_snapshot(tmp_path, config, matrix)
    (tmp_path / "model.run.json").unlink()
    (tmp_path / filename).write_bytes(payload + b"tampered")
    with pytest.raises(ValueError, match="hash or size"):
        inspect_snapshot(tmp_path, config, matrix)
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        inspect_snapshot(tmp_path, config, matrix)


def test_partial_prediction_and_rejection_handling(tmp_path, monkeypatch):
    record = context("a", "a")
    method = {"name": "model", "representation": "json", "support_set": "zero"}
    prediction = prediction_row(record, "model")
    (tmp_path / "model.jsonl").write_bytes((json.dumps(prediction) + '\n{"incomplete"').encode())
    (tmp_path / "model.run.json").write_text('{"variant":"model"}', encoding="utf-8")
    (tmp_path / "model.prompts.jsonl").write_text(json.dumps({"record_id": "a", "layout_id": "a"}) + '\n',
                                                  encoding="utf-8")
    config = {"variants": {"model": {"metadata": {"variant": "model"}, "partial_final_line": True}}}
    matrix = {"methods": [method]}
    monkeypatch.setattr("scripts.compare_local_rule_posthoc.validate_matrix_run", lambda *args: None)
    result, extras = verify_generator_sources(tmp_path, matrix, config, {},
                                              {"model": {"a": source_row(record, "model")}},
                                              {"a": record}, parse_representation)
    assert result["prediction_rows"] == 1 and result["partial_prediction_files"] == 1 and not extras
    config["variants"]["model"]["partial_final_line"] = False
    with pytest.raises(ValueError, match="Partial JSONL"):
        verify_generator_sources(tmp_path, matrix, config, {}, {}, {"a": record}, parse_representation)
    rejection_path = tmp_path / "rejected.json"
    rejection_path.write_text(json.dumps([{"variant": "model", "reason": "bad metadata"}]), encoding="utf-8")
    config["variants"]["model"]["rejected_reason"] = "bad metadata"
    assert rejected_methods(rejection_path, config) == {"model": "bad metadata"}
    rejected = {**source_row(record, "model", state="rejected_infrastructure"), "error": "bad metadata"}
    result, _ = verify_generator_sources(tmp_path, matrix, config, {"model": "bad metadata"},
                                         {"model": {"a": rejected}}, {"a": record}, parse_representation)
    assert result["rejected_variants"] == 1
    config["variants"]["model"]["rejected_reason"] = "other"
    with pytest.raises(ValueError, match="differ"):
        rejected_methods(rejection_path, config)


def test_saved_prompts_are_bound_to_generation_metadata(tmp_path):
    record = context("a", "a")
    method = {"name": "model", "representation": "json", "support_set": "zero", "adapter": None}
    names = ("data", "test_ids", "support_data", "support_manifest", "protocol", "generator", "support_selector")
    inputs = {name: {"sha256": name + "_hash"} for name in names}
    model = {"path": str(tmp_path / "weights"), "revision": "revision",
             "sha256": {"model.safetensors": "weights_hash", "config.json": "config_hash",
                        "tokenizer.json": "tokenizer_hash"}}
    matrix = {"methods": [method], "inputs": inputs, "model": model, "settings": {"seed": 1729}}
    prompt_path = tmp_path / "model.prompts.jsonl"
    prompt_path.write_text(json.dumps({"record_id": "a", "layout_id": "a"}) + "\n", encoding="utf-8")
    metadata = {"variant": "model", "representation": "json", "support_set": "zero",
                "model": model["path"], "revision": model["revision"], "model_weights_sha256": "weights_hash",
                "model_config_sha256": "config_hash", "tokenizer_sha256": "tokenizer_hash", "seed": 1729,
                "adapter": None, "adapter_weights_sha256": None, "adapter_config_sha256": None,
                "prompts_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()}
    for field, name in (("data_sha256", "data"), ("test_ids_sha256", "test_ids"),
                        ("support_data_sha256", "support_data"), ("support_manifest_sha256", "support_manifest"),
                        ("protocol_sha256", "protocol"), ("generator_sha256", "generator"),
                        ("support_selection_sha256", "support_selector")):
        metadata[field] = inputs[name]["sha256"]
    (tmp_path / "model.run.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "model.jsonl").write_text(json.dumps(prediction_row(record, "model")) + "\n", encoding="utf-8")
    config = {"variants": {"model": {"metadata": metadata, "partial_final_line": False}}}
    assessments = {"model": {"a": source_row(record, "model")}}
    result, _ = verify_generator_sources(tmp_path, matrix, config, {}, assessments, {"a": record},
                                         parse_representation)
    assert result["prediction_rows"] == 1
    prompt_path.write_text(json.dumps({"record_id": "a", "layout_id": "changed"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Saved prompts differ"):
        verify_generator_sources(tmp_path, matrix, config, {}, assessments, {"a": record}, parse_representation)
