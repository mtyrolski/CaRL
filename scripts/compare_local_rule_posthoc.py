#!/usr/bin/env python3
"""Pair an existing graph-rule assessment with frozen generator assessments.

This post-hoc comparison measures validity of proposed intents on matched
boards.  The rule searches the current board's movement graph; it is not a
language model.  Its bounded-search measurements are not controller outcomes.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import carl.icl.intent_assessment as intent_assessment_module
import carl.icl.intent_data as intent_data_module
import carl.icl.intent_generator_experiment as intent_generator_module
import scripts.assess_intent_representations as representation_assessment_module
import scripts.assess_lora_matrix as lora_assessment_module
from carl.icl.intent_assessment import board_means
from carl.icl.intent_assessment import bootstrap
from carl.icl.intent_assessment import test_contexts
from carl.icl.intent_data import validate_prediction
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import parse_json
from scripts.assess_intent_representations import validate_matrix_run
from scripts.assess_lora_matrix import parse_partial
from scripts.assess_lora_matrix import whole_goal

Record = dict[str, Any]
Parser = Callable[[str, str], Record]
RNG_SEED = 20261114
INFRASTRUCTURE = {"infrastructure_error", "rejected_infrastructure", "generation_error"}
ROOT = Path(__file__).resolve().parents[1]


def snapshot_jsonl(path: Path) -> tuple[list[Record], dict[str, Any]]:
    data = path.read_bytes()
    rows, partial = parse_partial(data)
    return rows, {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                  "partial_final_line": partial, "captured_bytes": len(data)}


def read_json(path: Path) -> Record:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object: {path}")
    return data


def load_parser(config: Record) -> Parser:
    path = Path(config["parser_path"])
    if file_sha256(path) != config["parser_sha256"]:
        raise ValueError("Generator assessment parser differs from the recorded snapshot")
    specification = importlib.util.spec_from_file_location("recorded_local_rule_comparison_protocol", path)
    if specification is None or specification.loader is None:
        raise ValueError("Unable to import the recorded parser")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.parse_representation


def verify_config(config: Record, data: Path, identifiers: Path) -> None:
    if config.get("data_sha256") != file_sha256(data) or config.get("test_ids_sha256") != file_sha256(identifiers):
        raise ValueError("Assessment does not belong to the frozen data and IDs")


def inspect_snapshot(directory: Path, config: Record, matrix: Record) -> tuple[dict[str, Record], list[Path]]:
    """Verify every listed byte, including the absence of missing snapshot files."""
    manifest_path = directory / "manifest.json"
    if file_sha256(manifest_path) != config.get("input_snapshot_manifest_sha256"):
        raise ValueError("Snapshot manifest hash differs from the generator assessment")
    manifest = read_json(manifest_path)
    methods = [method["name"] for method in matrix["methods"]]
    entries = manifest.get("variants")
    if (manifest.get("schema_version") != 1 or not isinstance(entries, list)
            or manifest.get("matrix_plan_sha256") != config.get("matrix_plan_sha256")
            or [entry.get("variant") for entry in entries] != methods
            or len(set(methods)) != len(methods) or set(config["variants"]) != set(methods)):
        raise ValueError("Snapshot manifest differs from the frozen matrix")
    sources = [manifest_path]
    by_variant = {}
    listed = {"manifest.json"}
    for entry in entries:
        name = entry["variant"]
        if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
            raise ValueError("Unsafe snapshot variant name")
        files = entry.get("files")
        expected_files = {f"{name}.jsonl", f"{name}.run.json", f"{name}.prompts.jsonl"}
        if not isinstance(files, dict) or set(files) != expected_files:
            raise ValueError(f"Incomplete snapshot file listing: {name}")
        for filename, info in files.items():
            path = directory / filename
            digest, size = info.get("sha256"), info.get("bytes")
            if digest is None:
                if size is not None or os.path.lexists(path):
                    raise ValueError(f"Unexpected snapshot file: {filename}")
                continue
            if not path.is_file() or not isinstance(size, int) or isinstance(size, bool):
                raise ValueError(f"Missing or malformed snapshot file: {filename}")
            data = path.read_bytes()
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Snapshot file hash or size mismatch: {filename}")
            sources.append(path)
            listed.add(filename)
        if config["variants"][name].get("sha256") != files[f"{name}.jsonl"].get("sha256"):
            raise ValueError(f"Assessment prediction hash disagrees with snapshot: {name}")
        by_variant[name] = entry
    if {path.name for path in directory.iterdir()} != listed:
        raise ValueError("Snapshot contains unlisted files")
    return by_variant, sources


def index_predictions(rows: list[Record], expected: dict[str, Record], *, method: Record,
                      local_rule: bool, parser: Parser | None = None) -> dict[str, Record]:
    """Check frozen identifiers and the format assigned to each raw prediction."""
    if not local_rule and parser is None:
        raise ValueError("Generator prediction checking requires its recorded parser")
    indexed = {}
    for row in rows:
        identifier = row.get("record_id")
        if (not isinstance(identifier, str) or identifier not in expected or identifier in indexed
                or row.get("layout_id") != expected[identifier]["layout_id"] or row.get("split") != "test"):
            raise ValueError("Prediction has a foreign, duplicate or mismatched context")
        if local_rule:
            if row.get("method") != "local_rule":
                raise ValueError("Rule prediction has the wrong method")
        elif (row.get("variant") != method["name"] or row.get("representation") != method["representation"]
              or row.get("support_set") != method["support_set"]):
            raise ValueError("Prediction has the wrong method, representation or support set")
        prediction_status = row.get("status")
        if prediction_status not in (None, "infrastructure_error", "generation_error"):
            raise ValueError("Unknown prediction status")
        if prediction_status is None:
            raw = row.get("raw_output")
            if not isinstance(raw, str):
                raise ValueError("Prediction lacks raw output")
            try:
                if local_rule:
                    intent = parse_json(raw)
                else:
                    assert parser is not None
                    intent = parser(method["representation"], raw)
                syntax = intent is not None
            except ValueError:
                intent, syntax = None, False
            syntax_key, intent_key = ("json_valid", "parsed") if local_rule else ("syntax_valid", "parsed_intent")
            if row.get(syntax_key) is not syntax or row.get(intent_key) != intent:
                raise ValueError("Prediction parse differs from its raw output")
        indexed[identifier] = row
    return indexed


def verify_links(assessment: dict[str, Record], predictions: dict[str, Record], *, local_rule: bool,
                 rejected_reason: str | None = None) -> None:
    """Link each assessed row to its prediction without converting absences to failures."""
    for identifier, row in assessment.items():
        source = predictions.get(identifier)
        if rejected_reason is not None:
            if row.get("status") != "rejected_infrastructure" or row.get("error") != rejected_reason:
                raise ValueError("Rejected method has a contradictory assessment status")
            continue
        if source is None:
            if row.get("status") != "missing":
                raise ValueError("Assessment claims a result for an absent prediction")
            continue
        source_status = source.get("status")
        if source_status in ("infrastructure_error", "generation_error"):
            if row.get("status") != "infrastructure_error" or row.get("error") != source.get("error"):
                raise ValueError("Infrastructure status differs from the prediction")
            continue
        if row.get("status") != "evaluated":
            raise ValueError("Assessment marks an existing prediction as missing or rejected")
        fields = ("record_id", "layout_id", "raw_output", "generation_seconds", "generated_tokens")
        if not local_rule:
            fields += ("variant", "representation", "support_set", "syntax_valid")
        if any(row.get(key) != source.get(key) for key in fields):
            raise ValueError("Assessment differs from its source prediction")
        source_intent = source.get("parsed") if local_rule else source.get("parsed_intent")
        assessment_intent = row.get("strict_intent") if local_rule else row.get("intent")
        if source_intent != assessment_intent:
            raise ValueError("Assessment intent differs from its source prediction")


def verify_prompts(path: Path, expected: dict[str, Record]) -> None:
    prompts, info = snapshot_jsonl(path)
    if info["partial_final_line"]:
        raise ValueError("Prompt snapshot has an incomplete final line")
    seen = set()
    for row in prompts:
        identifier = row.get("record_id")
        if (not isinstance(identifier, str) or identifier not in expected or identifier in seen
                or row.get("layout_id") != expected[identifier]["layout_id"]):
            raise ValueError("Prompt snapshot contains a foreign, duplicate or mismatched context")
        seen.add(identifier)
    if seen != set(expected):
        raise ValueError("Prompt snapshot does not cover the frozen contexts")


def rejected_methods(path: Path, config: Record) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of rejected variants")
    result = {}
    for entry in data:
        name, reason = entry.get("variant"), entry.get("reason")
        if not isinstance(name, str) or not isinstance(reason, str) or name in result:
            raise ValueError("Invalid rejected-variant entry")
        result[name] = reason
    configured = {name: info["rejected_reason"] for name, info in config["variants"].items()
                  if info.get("rejected_reason")}
    if result != configured:
        raise ValueError("Rejected variants differ from assessment configuration")
    return result


def verify_generator_sources(directory: Path, matrix: Record, config: Record, rejected: dict[str, str],
                             assessments: dict[str, dict[str, Record]], expected: dict[str, Record],
                             parser: Parser) -> tuple[Record, list[Path]]:
    """Audit metadata and pair all extant assessment rows with frozen predictions."""
    counts: Counter[str] = Counter()
    extra_sources = []
    for method in matrix["methods"]:
        name = method["name"]
        assessment = assessments.get(name, {})
        reason = rejected.get(name)
        counts["assessment_rows"] += len(assessment)
        if reason is not None:
            verify_links(assessment, {}, local_rule=False, rejected_reason=reason)
            counts["rejected_variants"] += 1
            counts["rejected_assessment_rows"] += len(assessment)
            continue
        path = directory / f"{name}.jsonl"
        run_path = directory / f"{name}.run.json"
        if path.is_file():
            rows, info = snapshot_jsonl(path)
            partial = info["partial_final_line"]
        else:
            rows, partial = [], False
        config_info = config["variants"][name]
        if partial is not config_info.get("partial_final_line"):
            raise ValueError(f"Partial JSONL status differs from assessment: {name}")
        if run_path.is_file():
            metadata = read_json(run_path)
            if metadata != config_info.get("metadata"):
                raise ValueError(f"Run metadata differs from assessment: {name}")
            validate_matrix_run(matrix, method, metadata, path)
            verify_prompts(path.with_suffix(".prompts.jsonl"), expected)
            if method.get("adapter_pending") and method.get("adapter") is not None:
                extra_sources.append(Path(matrix["run"]) / "matrix_adapter_hashes" / f"{name}.json")
        elif rows or config_info.get("metadata") is not None:
            raise ValueError(f"Predictions or assessment metadata lack a run manifest: {name}")
        predictions = index_predictions(rows, expected, method=method, local_rule=False, parser=parser)
        verify_links(assessment, predictions, local_rule=False)
        counts["prediction_rows"] += len(predictions)
        counts["predictions_without_assessment"] += len(set(predictions) - set(assessment))
        counts["partial_prediction_files"] += int(partial)
    return dict(counts), extra_sources


def index_rows(rows: list[Record], expected: dict[str, Record], *, local_rule: bool,
               parser: Parser | None = None) -> dict[str, dict[str, Record]]:
    """Verify provenance and validity against current boards, never future states."""
    by_variant: dict[str, dict[str, Record]] = {}
    for row in rows:
        identifier, variant = row.get("record_id"), row.get("variant")
        if not isinstance(identifier, str) or identifier not in expected or not isinstance(variant, str):
            raise ValueError("Assessment has a foreign ID or missing variant")
        if local_rule and variant != "local_rule":
            raise ValueError("Rule assessment contains a different method")
        source = expected[identifier]
        family = source.get("evaluation_family") or source.get("family")
        if row.get("layout_id") != source["layout_id"] or row.get("family") != family:
            raise ValueError("Assessment layout or family differs from the frozen dataset")
        group = by_variant.setdefault(variant, {})
        if identifier in group:
            raise ValueError("Duplicate method/context in assessment")
        status = row.get("status")
        if status not in {"evaluated", "missing", *INFRASTRUCTURE}:
            raise ValueError(f"Unknown assessment status: {status}")
        if status == "evaluated":
            board = source["start_observation"]
            if row.get("start_observation") != board or not isinstance(row.get("raw_output"), str):
                raise ValueError("Assessment must retain the exact current board and raw output")
            if local_rule:
                try:
                    intent = parse_json(row["raw_output"])
                except ValueError:
                    intent = None
                valid = intent is not None and not validate_prediction(intent, board)
                whole = whole_goal(intent, board) if valid and intent is not None else False
                if row.get("strict_valid") is not valid or row.get("strict_whole_goal") is not whole:
                    raise ValueError("Rule validity differs from independent checking of its raw output")
            else:
                if parser is None or not isinstance(row.get("representation"), str):
                    raise ValueError("Generator parser and representation are required")
                try:
                    intent = parser(row["representation"], row["raw_output"])
                    syntax = True
                except ValueError:
                    intent, syntax = None, False
                valid = bool(syntax and not validate_prediction(intent, board))
                whole = whole_goal(intent, board) if valid and intent is not None else False
                if (row.get("syntax_valid") is not syntax or row.get("semantic_valid") is not valid
                        or row.get("valid_not_whole_goal") is not (valid and not whole)):
                    raise ValueError("Generator validity differs from independent checking of its raw output")
        group[identifier] = row
    return by_variant


def exposure(record: Record) -> str:
    value = record.get("previously_evaluated_in_lora_matrix")
    return "in_previous_lora_matrix" if value is True else "not_in_previous_lora_matrix" if value is False else "unknown"


def status(row: Record | None) -> str:
    if row is None:
        return "absent_assessment_row"
    value = row["status"]
    return "infrastructure_error" if value in INFRASTRUCTURE else value


def compare(variant: str, contexts: list[Record], reference: dict[str, Record],
            candidate: dict[str, Record], family: str, cohort: str) -> tuple[Record, list[Record]]:
    chosen = [r for r in contexts if (family == "all" or (r.get("evaluation_family") or r.get("family")) == family)
              and (cohort == "all" or exposure(r) == cohort)]
    paired = []
    pairs = []
    before_status: Counter[str] = Counter()
    after_status: Counter[str] = Counter()
    for record in chosen:
        identifier = record["record_id"]
        before, after = reference.get(identifier), candidate.get(identifier)
        left, right = status(before), status(after)
        before_status[left] += 1
        after_status[right] += 1
        pair = {"variant": variant, "record_id": identifier, "layout_id": record["layout_id"],
                "family": record.get("evaluation_family") or record.get("family"),
                "previous_lora_exposure": exposure(record), "local_status": left, "generator_status": right,
                "local_valid_not_whole": None, "generator_valid_not_whole": None, "delta": None}
        if left == right == "evaluated":
            assert before is not None and after is not None
            baseline = bool(before["strict_valid"] and not before["strict_whole_goal"])
            result = bool(after["valid_not_whole_goal"])
            pair.update(local_valid_not_whole=baseline, generator_valid_not_whole=result,
                        delta=int(result) - int(baseline))
            paired.append(pair)
        pairs.append(pair)
    means = board_means(paired, lambda row: float(row["delta"]))
    interval = bootstrap(means, random.Random(RNG_SEED)) if means else None
    counts = Counter((r["local_valid_not_whole"], r["generator_valid_not_whole"]) for r in paired)
    result = {"variant": variant, "family": family, "previous_lora_exposure": cohort,
              "planned_contexts": len(chosen), "planned_boards": len({r["layout_id"] for r in chosen}),
              "local_status_counts": dict(before_status), "generator_status_counts": dict(after_status),
              "local_evaluated": before_status["evaluated"], "generator_evaluated": after_status["evaluated"],
              "local_missing": before_status["missing"] + before_status["absent_assessment_row"],
              "generator_missing": after_status["missing"] + after_status["absent_assessment_row"],
              "local_missing_prediction": before_status["missing"],
              "generator_missing_prediction": after_status["missing"],
              "local_absent_assessment": before_status["absent_assessment_row"],
              "generator_absent_assessment": after_status["absent_assessment_row"],
              "local_infrastructure_errors": before_status["infrastructure_error"],
              "generator_infrastructure_errors": after_status["infrastructure_error"],
              "paired_contexts": len(paired), "paired_boards": len(means),
              "local_valid_paired": sum(bool(r["local_valid_not_whole"]) for r in paired),
              "generator_valid_paired": sum(bool(r["generator_valid_not_whole"]) for r in paired),
              "both_valid": counts[(True, True)], "local_only_valid": counts[(True, False)],
              "generator_only_valid": counts[(False, True)], "both_invalid": counts[(False, False)],
              "generator_minus_local_board_mean": sum(means) / len(means) if means else None,
              "bootstrap_low": interval[0] if interval else None, "bootstrap_high": interval[1] if interval else None,
              "uncertainty": "paired board percentile bootstrap, 2000 resamples" if interval
                             else "descriptive: fewer than five paired boards",
              "degenerate_interval": bool(interval is not None and interval[0] == interval[1])}
    return result, pairs


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def input_paths(args: argparse.Namespace, local_config: Record, generator_config: Record,
                additional_sources: list[Path] | None = None) -> list[Path]:
    """Identify data, code and recorded plans needed to reproduce this analysis."""
    paths = [
        Path(__file__).resolve(),
        ROOT / "tests/experiments/test_compare_local_rule_posthoc.py",
        ROOT / "scripts/intent_experiment_supervisor.py",
        Path(intent_assessment_module.__file__).resolve(),
        Path(intent_data_module.__file__).resolve(),
        Path(intent_generator_module.__file__).resolve(),
        Path(representation_assessment_module.__file__).resolve(),
        Path(lora_assessment_module.__file__).resolve(),
        args.data,
        args.test_ids,
        args.local_assessment / "config.json",
        args.local_assessment / "assessment.jsonl",
        args.generator_assessment / "config.json",
        args.generator_assessment / "assessment.jsonl",
        args.generator_assessment / "summary.json",
        Path(generator_config["parser_path"]),
        Path(local_config["variants"]["local_rule"]["path"]),
    ]
    if generator_config.get("matrix_plan_path"):
        paths.append(Path(generator_config["matrix_plan_path"]))
    if additional_sources:
        paths.extend(additional_sources)
    return paths


def hash_inputs(paths: list[Path]) -> dict[str, str]:
    result = {str(path): file_sha256(path) for path in paths}
    if len(result) != len(paths):
        raise ValueError("Duplicate provenance input paths")
    return result


def supervisor_provenance(run: Path | None, stage_name: str | None) -> Record | None:
    if run is None and stage_name is None:
        return None
    if run is None or stage_name is None:
        raise ValueError("Specify both --supervisor-run and --supervisor-stage")
    manifest = read_json(run / "run_manifest.json")
    stages = [stage for stage in manifest["stages"] if stage["name"] == stage_name]
    if len(stages) != 1 or stages[0]["status"] != "running":
        raise ValueError("The specified supervisor stage must be registered and running")
    stage = stages[0]
    command = [sys.executable, str(ROOT / "scripts/intent_experiment_supervisor.py"), "run",
               "--name", stage_name, "--kind", stage["kind"], "--timeout", str(stage["timeout_seconds"]),
               str(run), "--", *stage["command"]]
    return {"run_manifest": str(run / "run_manifest.json"), "stage": stage_name,
            "status_when_snapshotted": stage["status"], "started_utc": stage["started_utc"],
            "stage_log": stage["output"], "launch_command_argv": command,
            "launch_command_shell": shlex.join(command)}


def write_provenance(output: Path, hashes: dict[str, str], supervisor: Record | None) -> None:
    outputs = sorted(path for path in output.iterdir() if path.is_file() and path.name != "provenance.json")
    command = [sys.executable, *sys.argv]
    record = {"schema_version": 1, "posthoc": True, "working_directory": str(Path.cwd()),
              "analysis_command_argv": command, "analysis_command_shell": shlex.join(command),
              "environment": {key: os.environ.get(key) for key in ("MPLCONFIGDIR", "IPYTHONDIR")},
              "supervisor": supervisor, "input_sha256": hashes,
              "output_sha256": {str(path): file_sha256(path) for path in outputs}}
    atomic_text(output / "provenance.json", json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[Record]) -> None:
    import io

    if not rows:
        raise ValueError("No comparison rows")
    output = io.StringIO(newline="")
    fields = [key for key in rows[0] if key not in ("local_status_counts", "generator_status_counts")]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, output.getvalue())


def plot(output: Path, rows: list[Record]) -> None:
    """Display only measured comparisons; missing methods remain in the CSV."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["family"] == row["previous_lora_exposure"] == "all"
                and row["generator_minus_local_board_mean"] is not None]
    if not selected:
        return
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(9, max(3, len(selected) * .45 + 1.5)), constrained_layout=True)
    for index, row in enumerate(selected):
        value = row["generator_minus_local_board_mean"]
        low, high = row["bootstrap_low"], row["bootstrap_high"]
        axis.scatter(value, index, color="#315c7e", zorder=3)
        if low is not None and high is not None:
            axis.hlines(index, low, high, color="#315c7e")
        axis.annotate(f"{row['paired_boards']}/{row['planned_boards']} plansz", (value, index),
                      xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_yticks(range(len(selected)), [row["variant"] for row in selected])
    axis.invert_yaxis()
    axis.set(xlim=(-1.05, 1.05), xlabel="Qwen minus reguła: odsetek poprawnych intencji",
             title="Post-hoc: generator i reguła korzystająca z grafu planszy")
    axis.grid(axis="x", alpha=.2)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"paired_local_rule.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--test-ids", required=True, type=Path)
    parser.add_argument("--local-assessment", required=True, type=Path)
    parser.add_argument("--generator-assessment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--supervisor-run", type=Path)
    parser.add_argument("--supervisor-stage")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory; existing results are preserved")
    identifiers = read_json(args.test_ids)
    if identifiers.get("dataset_sha256") != file_sha256(args.data):
        raise ValueError("Frozen test IDs do not match the dataset")
    data_rows, data_info = snapshot_jsonl(args.data)
    if data_info["partial_final_line"]:
        raise ValueError("Frozen dataset has an incomplete line")
    contexts = test_contexts(data_rows, args.test_ids)
    if not contexts:
        raise ValueError("No frozen test contexts")
    expected = {row["record_id"]: row for row in contexts}
    if len(expected) != len(contexts):
        raise ValueError("Frozen contexts must have unique IDs")
    local_config, generator_config = (read_json(args.local_assessment / "config.json"),
                                      read_json(args.generator_assessment / "config.json"))
    verify_config(local_config, args.data, args.test_ids)
    verify_config(generator_config, args.data, args.test_ids)
    matrix_path = generator_config.get("matrix_plan_path")
    if not matrix_path or file_sha256(matrix_path) != generator_config.get("matrix_plan_sha256"):
        raise ValueError("Generator matrix plan is missing or changed after the source assessment")
    matrix = read_json(Path(matrix_path))
    snapshot_dir = args.generator_assessment / "input_snapshot"
    _, snapshot_sources = inspect_snapshot(snapshot_dir, generator_config, matrix)
    rejected_path = args.generator_assessment / "rejected_variants.json"
    rejected = rejected_methods(rejected_path, generator_config)
    adapter_sources = [Path(matrix["run"]) / "matrix_adapter_hashes" / f"{method['name']}.json"
                       for method in matrix["methods"] if method.get("adapter_pending")
                       and method.get("adapter") is not None and method["name"] not in rejected
                       and (snapshot_dir / f"{method['name']}.run.json").is_file()]
    sources = input_paths(args, local_config, generator_config, [*snapshot_sources, rejected_path, *adapter_sources])
    hashes = hash_inputs(sources)
    rule_record = local_config["variants"]["local_rule"]
    if hashes[str(Path(rule_record["path"]))] != rule_record["sha256"]:
        raise ValueError("Rule predictions changed after their assessment")
    supervisor = supervisor_provenance(args.supervisor_run, args.supervisor_stage)
    parser_fn = load_parser(generator_config)
    local_rows, local_info = snapshot_jsonl(args.local_assessment / "assessment.jsonl")
    generator_rows, generator_info = snapshot_jsonl(args.generator_assessment / "assessment.jsonl")
    if (data_info["sha256"] != hashes[str(args.data)]
            or local_info["sha256"] != hashes[str(args.local_assessment / "assessment.jsonl")]
            or generator_info["sha256"] != hashes[str(args.generator_assessment / "assessment.jsonl")]):
        raise ValueError("An input changed while it was being read")
    local = index_rows(local_rows, expected, local_rule=True).get("local_rule", {})
    generators = index_rows(generator_rows, expected, local_rule=False, parser=parser_fn)
    rule_rows, rule_info = snapshot_jsonl(Path(rule_record["path"]))
    if (rule_info["sha256"] != hashes[str(Path(rule_record["path"]))]
            or rule_info["partial_final_line"] is not rule_record.get("partial_final_line")):
        raise ValueError("Rule predictions changed or their partial-line status differs from the assessment")
    rule_predictions = index_predictions(rule_rows, expected, method={"name": "local_rule"}, local_rule=True)
    verify_links(local, rule_predictions, local_rule=True)
    generator_audit, verified_adapters = verify_generator_sources(snapshot_dir, matrix, generator_config, rejected,
                                                                   generators, expected, parser_fn)
    if set(verified_adapters) != set(adapter_sources):
        raise ValueError("Frozen adapter metadata changed while validating snapshot")
    generator_summary = read_json(args.generator_assessment / "summary.json")
    observed_variants = [row["variant"] for row in generator_summary.get("results", [])]
    variants = [row["name"] for row in matrix["methods"]]
    if set(observed_variants) - set(variants):
        raise ValueError("Assessment summary contains a foreign matrix variant")
    if not variants or len(set(variants)) != len(variants) or set(generators) - set(variants):
        raise ValueError("Generator assessment summary lacks unique, complete variant labels")
    families = sorted({str(row.get("evaluation_family") or row.get("family")) for row in contexts})
    cohorts = sorted({exposure(row) for row in contexts})
    results, all_pairs = [], []
    for variant in variants:
        for family in ("all", *families):
            for cohort in ("all", *cohorts):
                selected = [row for row in contexts
                            if (family == "all" or (row.get("evaluation_family") or row.get("family")) == family)
                            and (cohort == "all" or exposure(row) == cohort)]
                if not selected:
                    continue
                result, pairs = compare(variant, selected, local, generators.get(variant, {}), family, cohort)
                results.append(result)
                if family == cohort == "all":
                    all_pairs.extend(pairs)
    summary = {
        "schema_version": 1, "status": "complete", "posthoc": True,
        "metric": "generator valid_not_whole_goal minus (local strict_valid and not strict_whole_goal)",
        "whole_goal_definition": "An explicit boxes_on_goals_at_least effect for all boxes; other logically "
                                 "equivalent conjunctions are not recognised.",
        "baseline": "Deterministic rule searches the current-board walking graph for a legal push; not an LLM.",
        "feasibility": "Rule assessment's bounded BFS is a solver check; it is not execution by the Qwen controller.",
        "bootstrap": {"seed": RNG_SEED, "resamples": 2000, "unit": "independent layout",
                      "method": "paired percentile bootstrap", "minimum_boards": 5,
                      "caution": "A zero-width percentile interval can occur with no observed variation and does "
                                 "not establish zero uncertainty. Intervals are exploratory and unadjusted for "
                                 "multiple comparisons; layouts are a convenience sample. Missing or partial "
                                 "evaluations need not be representative of the planned test set."},
        "exposure": "Not in the earlier LoRA matrix is not proof of no prior model exposure; family and "
                    "exposure are confounded in this cohort.",
        "source_verification": {"rule_prediction_rows": len(rule_predictions),
                                "rule_assessment_rows": len(local), **generator_audit},
        "inputs": {"data": data_info, "test_ids_sha256": file_sha256(args.test_ids),
                   "local_assessment": local_info, "generator_assessment": generator_info,
                   "local_config_sha256": file_sha256(args.local_assessment / "config.json"),
                   "generator_config_sha256": file_sha256(args.generator_assessment / "config.json"),
                   "generator_summary_sha256": file_sha256(args.generator_assessment / "summary.json"),
                   "analysis_source_sha256": file_sha256(__file__)},
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output / "paired_rows.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_pairs))
    write_csv(args.output / "paired_summary.csv", results)
    atomic_text(args.output / "paired_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    plot(args.output, results)
    _, final_snapshot_sources = inspect_snapshot(snapshot_dir, generator_config, matrix)
    if final_snapshot_sources != snapshot_sources:
        raise ValueError("Snapshot file set changed during the analysis")
    if hashes != hash_inputs(sources):
        raise ValueError("An input changed during the analysis; output provenance is incomplete")
    write_provenance(args.output, hashes, supervisor)
    print(json.dumps({"status": "complete", "variants": len(variants), "contexts": len(contexts),
                      "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
