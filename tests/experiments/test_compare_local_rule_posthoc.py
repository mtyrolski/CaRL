"""Paired rule comparison must retain missing cases and layout independence."""

import argparse
import hashlib
import json

import pytest

from carl.icl.intent_representation_experiment import parse_representation
from scripts.compare_local_rule_posthoc import compare
from scripts.compare_local_rule_posthoc import hash_inputs
from scripts.compare_local_rule_posthoc import index_rows
from scripts.compare_local_rule_posthoc import input_paths
from scripts.compare_local_rule_posthoc import supervisor_provenance
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
    result.update(start_observation=record["start_observation"], raw_output=raw)
    if variant == "local_rule":
        result.update(strict_valid=valid, strict_whole_goal=False)
    else:
        result.update(representation="json", syntax_valid=valid, semantic_valid=valid, valid_not_whole_goal=valid)
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
    assert len(paths) == 17
    for path in paths:
        if path.is_relative_to(tmp_path):
            path.write_text(f"source: {path.name}", encoding="utf-8")
    hashes = hash_inputs(paths)
    (output / "paired_summary.csv").write_text("variant,count\nmodel,1\n", encoding="utf-8")
    write_provenance(output, hashes, supervisor=None)
    record = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert record["input_sha256"] == hashes and len(record["input_sha256"]) == 17
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
