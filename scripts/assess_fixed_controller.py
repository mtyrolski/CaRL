#!/usr/bin/env python3
"""Assess a frozen intent controller at each primitive-action budget.

Invalid generator outputs count as failures of the generator-plus-controller
pipeline. Missing generation, missing rollouts, and local infrastructure
errors are excluded from observed denominators and reported separately. The
controller-only fraction is conditional on valid intents actually evaluated.
Every saved action and prompt is independently replayed in the real environment.
"""

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

import matplotlib

from carl.icl.intent_assessment import board_means
from carl.icl.intent_assessment import bootstrap
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_assessment import read_jsonl
from carl.icl.intent_assessment import test_contexts
from carl.icl.intent_data import validate_prediction
from carl.icl.intent_fence_audit import remove_one_outer_fence
from carl.icl.intent_representation_experiment import SupportSet
from carl.icl.intent_representation_experiment import load_supports
from carl.icl.intent_representation_experiment import messages_for
from carl.icl.intent_representation_experiment import parse_representation
from carl.icl.local_intent_policy import sha256_file
from carl.icl.luna_intent_rollout import POLICY_PROMPT
from scripts.assess_intent_representations import atomic_json
from scripts.assess_intent_representations import atomic_text
from scripts.assess_intent_representations import board_family
from scripts.assess_intent_representations import validate_matrix_run
from scripts.assess_intent_representations import write_csv
from scripts.assess_lora_matrix import parse_partial
from scripts.audit_luna_rollouts import audit_rollout

matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Record = dict[str, Any]
SEED = 20261114
ELIGIBILITY = {"missing_generation", "invalid_representation", "invalid_intent", "ready"}
CONTROLLER_STATUSES = {"achieved", "action_limit", "preservation_violated", "task_solved", "invalid_response",
                       "api_error", "invalid_intent"}


class GeneratorNormalisation(StrEnum):
    """Frozen rule applied to a generator answer before policy execution."""

    NONE = "none"
    REMOVE_ONE_OUTER_FENCE = "remove_one_outer_fence"


class AnalysisKind(StrEnum):
    STRICT = "strict_preregistered"
    FENCE = "posthoc_exploratory"
    ADDENDUM = "exploratory_addendum"


ADDENDUM_STATUS = "posthoc_icl_prior_to_matched_lora_predictions"
ADDENDUM_PAIRS = [(f"base_{name}_zero", f"rep_lora_{name}_zero") for name in ("json", "formal", "natural")]


class AssessmentDeadline(Exception):
    """A bounded in-process fallback reached its own deadline."""


def resolve(path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else ROOT / result


def read_snapshot(path: Path) -> tuple[list[Record], Record]:
    data = path.read_bytes() if path.exists() else None
    rows, partial = parse_partial(data) if data is not None else ([], False)
    return rows, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest() if data is not None else None,
                  "partial_final_line": partial}


def check_test_subset(full_ids: list[str], contexts: list[Record]) -> None:
    selected = [row["record_id"] for row in contexts]
    if (len(set(full_ids)) != len(full_ids) or len(set(selected)) != len(selected)
            or not set(selected) <= set(full_ids)):
        raise ValueError("Controller IDs are not a unique subset of the full frozen matrix IDs")


def reconstruct_generator_answer(prediction: Record, mode: GeneratorNormalisation) -> tuple[Record | None, Record]:
    """Verify raw parsing before optionally removing exactly one complete fence."""
    raw = prediction.get("raw_output")
    representation = prediction.get("representation")
    if not isinstance(raw, str) or not isinstance(representation, str):
        raise ValueError("Generator output or representation is missing")
    try:
        strict_intent = parse_representation(representation, raw)
        strict_valid = True
    except ValueError:
        strict_intent, strict_valid = None, False
    if prediction.get("syntax_valid") is not strict_valid or prediction.get("parsed_intent") != strict_intent:
        raise ValueError("Recorded raw parse differs from the original generator response")
    normalised, removed = (remove_one_outer_fence(raw) if mode == GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE
                           else (raw, False))
    try:
        intent = parse_representation(representation, normalised)
    except ValueError:
        intent = None
    return intent, {"raw_output": raw, "normalised_output": normalised, "fence_removed": removed,
                    "recorded_syntax_valid": strict_valid, "recorded_parsed_intent": strict_intent}


def verify_generator_link(path: Path, variant: str, metadata: Record, all_test: list[Record],
                          contexts: list[Record], data: Path, matrix: Record, matrix_path: Path,
                          mode: GeneratorNormalisation) -> tuple[dict[str, str], dict[str, Record], dict[str, Record]]:
    """Reconstruct eligibility and intents from the frozen raw model answers."""
    if not path.exists() or metadata.get("predictions_sha256") != file_hash(path):
        raise ValueError(f"Generator predictions differ from the frozen controller inputs: {variant}")
    run_path = path.with_suffix(".run.json")
    if not run_path.exists():
        raise ValueError(f"Generator metadata is missing: {variant}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    protocol = ROOT / "carl/icl/intent_representation_experiment.py"
    if (run.get("variant") != variant or run.get("data_sha256") != file_hash(data)
            or run.get("protocol_sha256") != file_hash(protocol)):
        raise ValueError(f"Generator metadata or parser protocol differs from frozen inputs: {variant}")
    prompts_path = path.with_suffix(".prompts.jsonl")
    if (metadata.get("matrix_plan_sha256") != file_hash(matrix_path)
            or metadata.get("prediction_run_sha256") != file_hash(run_path)
            or not prompts_path.exists() or metadata.get("prompts_sha256") != file_hash(prompts_path)):
        raise ValueError(f"Controller input provenance does not bind the matrix, run and prompts: {variant}")
    methods = [job for job in matrix["methods"] if job["name"] == variant]
    if len(methods) != 1:
        raise ValueError(f"Variant does not occur exactly once in the matrix plan: {variant}")
    method = methods[0]
    if Path(method["output"]).resolve() != path.resolve():
        raise ValueError(f"Generator path differs from the matrix plan: {variant}")
    validate_matrix_run(matrix, method, run, path)
    full_ids_path = Path(matrix["inputs"]["test_ids"]["path"])
    if (matrix["inputs"]["data"]["sha256"] != file_hash(data)
            or matrix["inputs"]["test_ids"]["sha256"] != file_hash(full_ids_path)):
        raise ValueError("Frozen full matrix data or IDs changed")
    full_manifest = json.loads(full_ids_path.read_text(encoding="utf-8"))
    if full_manifest.get("dataset_sha256") != file_hash(data):
        raise ValueError("Full matrix IDs refer to a different dataset")
    full_ids = full_manifest["record_ids"]
    check_test_subset(full_ids, contexts)
    source = {row["record_id"]: row for row in all_test}
    if len(source) != len(all_test):
        raise ValueError("Duplicate test IDs in the source dataset")
    if any(identifier not in source for identifier in full_ids):
        raise ValueError("A frozen matrix context is absent from the test dataset")
    prompts = read_jsonl(prompts_path)
    if [row.get("record_id") for row in prompts] != full_ids:
        raise ValueError(f"Prompt contexts differ from the full frozen matrix order: {variant}")
    for key in ("support_data", "support_manifest"):
        entry = matrix["inputs"][key]
        if file_hash(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Frozen matrix input changed: {key}")
    support = ([] if method["support_set"] == "zero" else
               load_supports(Path(matrix["inputs"]["support_data"]["path"]),
                             Path(matrix["inputs"]["support_manifest"]["path"]), SupportSet(method["support_set"])))
    for prompt in prompts:
        record = source[prompt["record_id"]]
        if (prompt.get("layout_id") != record["layout_id"]
                or prompt.get("messages") != messages_for(record, support, method["representation"])):
            raise ValueError(f"Saved generator prompt differs from the frozen protocol: {variant}/{record['record_id']}")
    predictions, snapshot = read_snapshot(path)
    if snapshot["partial_final_line"]:
        raise ValueError(f"Generator predictions end in a partial line: {variant}")
    by_id: dict[str, Record] = {}
    for row in predictions:
        identifier = row.get("record_id")
        if (not isinstance(identifier, str) or identifier not in source or identifier in by_id or row.get("variant") != variant
                or row.get("layout_id") != source[identifier]["layout_id"] or row.get("split") != "test"
                or row.get("representation") != method["representation"]
                or row.get("support_set") != method["support_set"]):
            raise ValueError(f"Generator prediction has foreign or duplicate provenance: {variant}/{identifier}")
        by_id[identifier] = row
    statuses: dict[str, str] = {}
    intents: dict[str, Record] = {}
    diagnostics: dict[str, Record] = {}
    for record in contexts:
        identifier = record["record_id"]
        prediction = by_id.get(identifier)
        if prediction is None:
            statuses[identifier] = "missing_generation"
            continue
        intent, details = reconstruct_generator_answer(prediction, mode)
        diagnostics[identifier] = details
        if intent is None:
            statuses[identifier] = "invalid_representation"
            continue
        if validate_prediction(intent, record["start_observation"]):
            statuses[identifier] = "invalid_intent"
        else:
            statuses[identifier] = "ready"
            intents[identifier] = intent
    return statuses, intents, diagnostics


def validate_normalisation_metadata(metadata: Record, mode: GeneratorNormalisation) -> None:
    value = metadata.get("generator_output_normalisation")
    if value is None and mode == GeneratorNormalisation.NONE:
        return  # Older strict-input manifests predate this field.
    if (not isinstance(value, dict) or value.get("mode") != mode
            or value.get("posthoc") is not (mode == GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE)
            or value.get("normaliser_source_sha256") != file_hash(ROOT / "carl/icl/intent_fence_audit.py")
            or value.get("raw_predictions_sha256") != metadata.get("predictions_sha256")):
        raise ValueError("Generator normalisation provenance differs from the frozen policy plan")


def eligibility_rows(path: Path, predictions_path: Path, variant: str, contexts: list[Record], all_test: list[Record],
                     ids_path: Path, data: Path, matrix: Record, matrix_path: Path,
                     mode: GeneratorNormalisation) -> tuple[dict[str, Record], Record]:
    """Accept only complete, hashed, frozen eligibility records or none at all."""
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {}, {"status": "missing_eligibility", "path": str(path), "manifest": None}
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    eligibility_path = path / "eligibility.jsonl"
    records_path = path / "records.jsonl"
    if (metadata.get("variant") != variant or metadata.get("test_ids_sha256") != file_hash(ids_path)
            or metadata.get("data_sha256") != file_hash(data) or metadata.get("eligibility_sha256") != file_hash(eligibility_path)
            or metadata.get("records_sha256") != file_hash(records_path)):
        raise ValueError(f"Controller input manifest differs from frozen sources: {variant}")
    validate_normalisation_metadata(metadata, mode)
    expected_statuses, expected_intents, diagnostics = verify_generator_link(predictions_path, variant, metadata, all_test,
                                                                               contexts, data, matrix, matrix_path, mode)
    expected = {r["record_id"]: r for r in contexts}
    rows = read_jsonl(eligibility_path)
    if [r.get("record_id") for r in rows] != [r["record_id"] for r in contexts]:
        raise ValueError(f"Eligibility records differ from the frozen order: {variant}")
    for row in rows:
        source = expected[row["record_id"]]
        if (row.get("variant") != variant or row.get("layout_id") != source["layout_id"]
                or row.get("family") != board_family(source) or row.get("status") not in ELIGIBILITY
                or row.get("status") != expected_statuses[row["record_id"]]):
            raise ValueError(f"Eligibility provenance differs for {variant}/{row['record_id']}")
        if mode == GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE and row["record_id"] in diagnostics:
            if any(row.get(key) != value for key, value in diagnostics[row["record_id"]].items()):
                raise ValueError(f"Recorded generator normalisation differs from raw output: {variant}/{row['record_id']}")
        elif mode == GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE and any(
                row.get(key) is not None for key in ("raw_output", "normalised_output", "fence_removed",
                                                     "recorded_syntax_valid", "recorded_parsed_intent")):
            raise ValueError(f"Missing generation contains fabricated normalisation data: {variant}/{row['record_id']}")
    policy_records = read_jsonl(records_path)
    ready_ids = [row["record_id"] for row in rows if row["status"] == "ready"]
    if [row.get("record_id") for row in policy_records] != ready_ids:
        raise ValueError(f"Controller records differ from eligible IDs: {variant}")
    for row in policy_records:
        source = expected[row["record_id"]]
        if (row.get("layout_id") != source["layout_id"] or row.get("variant") != variant
                or row.get("start_observation") != source["start_observation"]
                or row.get("task_goal") != source["task_goal"] or row.get("intent") != expected_intents[row["record_id"]]
                or validate_prediction(row.get("intent"), source["start_observation"])):
            raise ValueError(f"Controller input violates the board/intent contract: {variant}/{row['record_id']}")
        if set(row) - {"record_id", "layout_id", "family", "variant", "split", "start_observation", "task_goal", "intent"}:
            raise ValueError(f"Controller input contains extra fields: {variant}/{row['record_id']}")
    return {r["record_id"]: r for r in rows}, {"status": "ready", "path": str(path), "manifest": metadata,
                                                "generator_link_audit": "Raw predictions independently re-parsed; eligibility and controller intents match.",
                                                "records": {r["record_id"]: r for r in policy_records}}


def controller_rows(path: Path, variant: str, selection: Record, inputs: Record,
                    action_cap: int, stop_monotonic: float | None = None,
                    stop_epoch: float | None = None) -> tuple[dict[str, Record], Record]:
    rows, snapshot = read_snapshot(path)
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        if rows:
            raise ValueError(f"Controller results lack a manifest: {variant}")
        return {}, {**snapshot, "manifest": None, "audit": {"rollouts": 0, "actions": 0, "model_calls": 0}}
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared = inputs.get("manifest")
    if prepared is None:
        raise ValueError(f"Controller results exist without frozen inputs: {variant}")
    if (metadata.get("config") != selection or metadata.get("records_sha256") != prepared.get("records_sha256")
            or metadata.get("action_parser") != "Remove at most one complete outer plain or json Markdown fence, then require strict action JSON."):
        raise ValueError(f"Controller settings differ from the frozen selection: {variant}")
    cases = inputs["records"]
    if metadata.get("record_ids") != list(cases):
        raise ValueError(f"Controller manifest IDs differ from frozen eligible records: {variant}")
    if [row.get("record_id") for row in rows] != list(cases)[:len(rows)]:
        raise ValueError(f"Saved controller results are not a prefix of the frozen order: {variant}")
    result: dict[str, Record] = {}
    audited: Counter[str] = Counter()
    for row in rows:
        if ((stop_monotonic is not None and time.monotonic() >= stop_monotonic)
                or (stop_epoch is not None and time.time() >= stop_epoch)):
            raise AssessmentDeadline("Post-hoc recovery reached its time limit while replaying actions")
        identifier = row.get("record_id")
        if not isinstance(identifier, str) or identifier not in cases or identifier in result:
            raise ValueError(f"Duplicate or foreign controller result: {variant}/{identifier}")
        case = cases[identifier]
        result_data = row.get("result")
        if not isinstance(result_data, dict) or result_data.get("status") not in CONTROLLER_STATUSES:
            raise ValueError(f"Unknown controller result status: {variant}/{identifier}")
        if result_data["status"] == "invalid_intent":
            raise ValueError(f"The controller rejected an independently validated intent: {variant}/{identifier}")
        responses = row.get("raw_policy_responses")
        answered = [step for step in result_data.get("steps", []) if step.get("raw_response") is not None]
        if not isinstance(responses, list) or len(responses) != len(answered):
            raise ValueError(f"Missing raw controller responses: {variant}/{identifier}")
        for original, step in zip(responses, answered, strict=True):
            normalised, removed = remove_one_outer_fence(original["raw"])
            if (original.get("normalised") != normalised or original.get("fence_removed") != removed
                    or step["raw_response"] != normalised):
                raise ValueError(f"Controller response normalisation differs from frozen protocol: {variant}/{identifier}")
        audit_row = {**{key: case.get(key) for key in ("record_id", "phase", "family", "layout_id", "context_id")},
                     **row, "cap": action_cap}
        counts = audit_rollout(case, audit_row, POLICY_PROMPT)
        audited.update(counts)
        audited["rollouts"] += 1
        result[identifier] = row
    return result, {**snapshot, "manifest": metadata, "audit": dict(audited)}


def classify(record: Record, decision: Record | None, rollout: Record | None) -> Record:
    """Return a disjoint status, preserving actual controller attempt counts."""
    result: Record = {"record_id": record["record_id"], "layout_id": record["layout_id"],
                      "family": board_family(record), "first_success_action": None,
                      "actions": None, "no_ops": None, "model_calls": None, "task_solved": None}
    if decision is None:
        return {**result, "status": "missing_eligibility", "pipeline_evaluated": False, "controller_evaluated": False}
    status = decision["status"]
    if status == "missing_generation":
        return {**result, "status": status, "pipeline_evaluated": False, "controller_evaluated": False}
    if status in ("invalid_representation", "invalid_intent"):
        return {**result, "status": status, "pipeline_evaluated": True, "controller_evaluated": False}
    if status != "ready":
        raise ValueError("Unknown eligibility status")
    if rollout is None:
        return {**result, "status": "missing_controller", "pipeline_evaluated": False, "controller_evaluated": False}
    measured = rollout["result"]
    result.update({"actions": len(measured["actions"]),
                   "no_ops": sum(step.get("no_op") is True for step in measured["steps"]),
                   "model_calls": measured["model_calls"], "task_solved": measured["task_solved"]})
    if measured["status"] == "api_error":
        return {**result, "status": "controller_infrastructure_error", "pipeline_evaluated": False,
                "controller_evaluated": False}
    success = measured["status"] == "achieved" and measured["achieved"] is True
    if success:
        result["first_success_action"] = len(measured["actions"])
    return {**result, "status": measured["status"], "pipeline_evaluated": True, "controller_evaluated": True}


def success(row: Record, budget: int) -> bool:
    first = row.get("first_success_action")
    return isinstance(first, int) and not isinstance(first, bool) and first <= budget


def curve(variant: str, rows: list[Record], budget: int, metric: str, family: str | None = None) -> Record:
    selected = [row for row in rows if family is None or row["family"] == family]
    if metric == "pipeline":
        assessed = [row for row in selected if row["pipeline_evaluated"]]
    elif metric == "conditional":
        assessed = [row for row in selected if row["controller_evaluated"]]
    else:
        raise ValueError("Unknown controller metric")
    means = board_means(assessed, lambda row: int(success(row, budget)))
    interval = bootstrap(means, random.Random(SEED))
    infrastructure = {"missing_eligibility", "missing_generation", "missing_controller", "controller_infrastructure_error",
                      "rejected_infrastructure"}
    invalid_generation = {"invalid_representation", "invalid_intent"}
    return {"family": family or "all", "variant": variant, "metric": metric, "budget": budget,
            "planned": len(selected), "assessed": len(assessed),
            "missing_infrastructure": sum(row["status"] in infrastructure for row in selected),
            "invalid_generation": sum(row["status"] in invalid_generation for row in selected),
            "excluded_from_denominator": len(selected) - len(assessed),
            "boards": len(means), "successes": sum(success(row, budget) for row in assessed),
            "board_mean": sum(means) / len(means) if means else None,
            "bootstrap_low": interval[0] if interval else None, "bootstrap_high": interval[1] if interval else None}


def compare(rows: list[Record], before: str, after: str, budget: int, metric: str, family: str | None = None) -> Record:
    chosen = [row for row in rows if family is None or row["family"] == family]
    index: dict[str, dict[str, Record]] = {before: {}, after: {}}
    for row in chosen:
        if row["variant"] in index:
            target = index[row["variant"]]
            if row["record_id"] in target:
                raise ValueError("Duplicate record in controller comparison")
            target[row["record_id"]] = row
    left, right = index[before], index[after]
    common = sorted(left.keys() & right.keys())
    for key in common:
        if left[key]["layout_id"] != right[key]["layout_id"] or left[key]["family"] != right[key]["family"]:
            raise ValueError("Paired controller records have conflicting layouts")
    eligible_key = "pipeline_evaluated" if metric == "pipeline" else "controller_evaluated"
    paired = [key for key in common if left[key][eligible_key] and right[key][eligible_key]]
    changes = [{"layout_id": left[key]["layout_id"],
                "delta": int(success(right[key], budget)) - int(success(left[key], budget))} for key in paired]
    means = board_means(changes, lambda row: row["delta"])
    bounds = bootstrap(means, random.Random(SEED))
    return {"family": family or "all", "metric": metric, "budget": budget, "reference": before, "candidate": after,
            "planned_pairs": len(common), "paired_contexts": len(paired), "unpaired_contexts": len(common) - len(paired),
            "paired_boards": len(means), "reference_success": sum(success(left[key], budget) for key in paired),
            "candidate_success": sum(success(right[key], budget) for key in paired),
            "board_mean_delta": sum(means) / len(means) if means else None,
            "bootstrap_low": bounds[0] if bounds else None, "bootstrap_high": bounds[1] if bounds else None}


def plot_curves(output: Path, curves: list[Record], family: str, variants: list[str], *, posthoc: bool = False,
                addendum: bool = False) -> None:
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    colours = plt.get_cmap("tab10")
    max_budget = max(row["budget"] for row in curves)
    for axis, metric, title in zip(axes, ("pipeline", "conditional"),
                                    ("Generator i kontroler", "Kontroler na poprawnych intencjach")):
        for index, variant in enumerate(variants):
            rows = sorted((row for row in curves if row["family"] == family and row["variant"] == variant
                           and row["metric"] == metric), key=lambda row: row["budget"])
            actual = [row for row in rows if row["board_mean"] is not None]
            if not actual:
                continue
            axis.plot([row["budget"] for row in actual], [row["board_mean"] for row in actual],
                      color=colours(index), marker="o", markersize=2, linewidth=1, drawstyle="steps-post",
                      label=f"{variant} ({actual[-1]['assessed']}/{actual[-1]['planned']})")
            for row in actual:
                if (row["budget"] in {0, 4, 8, 12, 16} or row["budget"] == actual[-1]["budget"]) and row["bootstrap_low"] is not None:
                    axis.vlines(row["budget"], row["bootstrap_low"], row["bootstrap_high"],
                                color=colours(index), linewidth=.6, alpha=.65)
        axis.set(title=title, xlabel="Budżet ruchów prymitywnych", xlim=(-.5, max_budget + .5), ylim=(-.03, 1.05))
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=6, loc="best")
        else:
            axis.text(.5, .5, "Brak ocenionych prób", transform=axis.transAxes, ha="center", fontsize=9)
    axes[0].set_ylabel("Średni odsetek zrealizowanych intencji na planszę")
    if addendum:
        prefix = "Eksploracyjne porównanie LoRA — ścisły format — "
    else:
        prefix = "POST-HOC: po usunięciu jednej otoczki — " if posthoc else "Ścisły format — "
    fig.suptitle(f"{prefix}zamrożony lokalny kontroler — {family}")
    fig.supxlabel("W nawiasie: ocenione / zaplanowane; przedziały 95% bootstrap po planszach dla zaznaczonych budżetów.",
                  fontsize=8)
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in family)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"controller_budget_{safe}.{extension}", dpi=180)
    plt.close(fig)


def normalisation_from_plan(plan: Record, plan_path: Path, run: Path) -> GeneratorNormalisation:
    mode = GeneratorNormalisation(plan.get("generator_output_normalisation", GeneratorNormalisation.NONE))
    if mode == GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE:
        if plan.get("status") != "posthoc_exploratory":
            raise ValueError("The fence-normalised controller analysis must be labelled post-hoc")
        strict_path = run / "policy_evaluation_plan.json"
        if not strict_path.exists():
            raise ValueError("The corresponding strict controller plan is missing")
        checksum = plan_path.with_suffix(".sha256")
        if (not checksum.is_file() or checksum.read_text(encoding="utf-8").strip() != file_hash(plan_path)
                or plan.get("primary_policy_plan_sha256") != file_hash(strict_path)):
            raise ValueError("Post-hoc or primary controller plan checksum differs")
        motivation_path = plan.get("motivation_summary_path")
        if motivation_path is not None:
            source = resolve(motivation_path)
            if not source.is_file() or file_hash(source) != plan.get("motivation_summary_sha256"):
                raise ValueError("Post-hoc motivation summary differs from its recorded hash")
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        fields = ("frozen_ids", "frozen_ids_sha256", "controller_selection_sha256", "selected_controller",
                  "decoding", "action_parser", "model_training", "primitive_action_budget", "variants", "api_calls")
        if any(plan.get(key) != strict.get(key) for key in fields):
            raise ValueError("Post-hoc plan does not use the strict plan's same cases, policy and budget")
    return mode


def validate_addendum_plan(plan: Record, plan_path: Path, run: Path, matrix: Record,
                          matrix_path: Path, contexts: list[Record]) -> None:
    """Bind the exploratory addendum to its original policy, data and three fixed pairs."""
    checksum = plan_path.with_suffix(".sha256")
    if (plan.get("status") != ADDENDUM_STATUS or not checksum.is_file()
            or checksum.read_text(encoding="utf-8").strip() != file_hash(plan_path)):
        raise ValueError("Exploratory addendum plan or checksum is invalid")
    strict_path = run / "policy_evaluation_plan.json"
    if not strict_path.is_file() or file_hash(strict_path) != plan.get("original_policy_plan_sha256"):
        raise ValueError("Exploratory addendum differs from the original policy plan")
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    shared = ("frozen_ids", "frozen_ids_sha256", "controller_selection_sha256", "selected_controller",
              "decoding", "action_parser", "model_training", "primitive_action_budget", "api_calls")
    if any(plan.get(key) != strict.get(key) for key in shared):
        raise ValueError("Exploratory addendum changed the frozen policy, inputs or budget")
    if (plan.get("generator_output_normalisation") != GeneratorNormalisation.NONE
            or plan.get("primitive_action_budget") != 16 or plan.get("model_training") is not False
            or plan.get("decoding") != "greedy" or plan.get("api_calls") != 0):
        raise ValueError("Exploratory addendum must use the strict generator and fixed controller")
    pairs = [list(pair) for pair in ADDENDUM_PAIRS]
    if (plan.get("contrasts") != pairs or plan.get("variants") != [name for pair in pairs for name in pair]
            or plan.get("prediction_presence_at_freeze") != {pair[1]: False for pair in pairs}):
        raise ValueError("Exploratory addendum does not contain the three frozen matched pairs")
    if (len(contexts) != 40 or len({row["layout_id"] for row in contexts}) != 40
            or Counter(board_family(row) for row in contexts) != {"real_12x12x4": 20, "proposal_gamma2_12x12x10": 20}):
        raise ValueError("Exploratory addendum requires the 40 frozen independent x4/x10 boards")
    for row in contexts:
        board = row["start_observation"]
        boxes = 4 if board_family(row) == "real_12x12x4" else 10
        if (not isinstance(board, list) or len(board) != 12
                or any(not isinstance(line, str) or len(line) != 12 for line in board)
                or sum(line.count("$") + line.count("*") for line in board) != boxes):
            raise ValueError("Exploratory addendum board dimensions or box count differ from the recorded family")
    if plan.get("matrix_plan_sha256") != file_hash(matrix_path):
        raise ValueError("Exploratory addendum matrix plan differs from the frozen plan")
    matrix_data, matrix_ids = (matrix["inputs"][name] for name in ("data", "test_ids"))
    if (file_hash(Path(matrix_data["path"])) != matrix_data["sha256"]
            or file_hash(Path(matrix_ids["path"])) != matrix_ids["sha256"]):
        raise ValueError("Exploratory addendum full-matrix data or IDs changed")
    full_ids = json.loads(Path(matrix_ids["path"]).read_text(encoding="utf-8"))
    controller_ids = json.loads(resolve(plan["frozen_ids"]).read_text(encoding="utf-8"))
    if (full_ids.get("dataset_sha256") != matrix_data["sha256"]
            or controller_ids.get("dataset_sha256") != matrix_data["sha256"]):
        raise ValueError("Exploratory addendum full-matrix IDs refer to another dataset")
    check_test_subset(full_ids["record_ids"], contexts)
    jobs = {job["name"]: job for job in matrix["methods"]}
    if len(jobs) != len(matrix["methods"]):
        raise ValueError("Matrix plan contains duplicate methods")
    for representation, (base, adapted) in zip(("json", "formal", "natural"), ADDENDUM_PAIRS, strict=True):
        for name, expect_adapter in ((base, False), (adapted, True)):
            job = jobs.get(name)
            if (job is None or job.get("representation") != representation or job.get("support_set") != "zero"
                    or bool(job.get("adapter")) is not expect_adapter):
                raise ValueError(f"Exploratory addendum method differs from the matrix: {name}")
    selection_path = run / "controller_selection.json"
    if not selection_path.is_file() or file_hash(selection_path) != plan.get("controller_selection_sha256"):
        raise ValueError("Exploratory addendum controller selection is missing or changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    calibration = run / f"policy_calibration_{selection['selected']}.manifest.json"
    assets = run / "controller_assets_snapshot.json"
    for path, key in ((calibration, "controller_calibration_manifest_sha256"),
                      (assets, "controller_assets_snapshot_sha256")):
        if not path.is_file() or file_hash(path) != plan.get(key):
            raise ValueError(f"Exploratory addendum frozen input changed: {path.name}")
    source_hashes = plan.get("controller_source_sha256")
    expected_sources = {"scripts/run_fixed_local_controller.py", "carl/icl/local_intent_policy.py",
                        "carl/icl/luna_intent_rollout.py", "carl/icl/intent_fence_audit.py"}
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_sources:
        raise ValueError("Exploratory addendum controller-source hashes are incomplete")
    for name, digest in source_hashes.items():
        source = ROOT / name
        if not source.is_file() or file_hash(source) != digest:
            raise ValueError(f"Exploratory addendum controller source changed: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--matrix-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-posthoc-fallback", action="store_true", help="Do not recover a missing post-hoc report")
    parser.add_argument("--fallback-max-seconds", type=float, default=180,
                        help="Maximum additional wall time for the in-process post-hoc recovery")
    args = parser.parse_args()
    if args.fallback_max_seconds <= 0:
        parser.error("fallback-max-seconds must be positive")
    return args


def check_deadline(args: argparse.Namespace) -> None:
    stop = getattr(args, "stop_monotonic", None)
    epoch = getattr(args, "stop_epoch", None)
    if ((stop is not None and time.monotonic() >= stop) or (epoch is not None and time.time() >= epoch)):
        raise AssessmentDeadline("Post-hoc recovery reached its time limit")


def assess(args: argparse.Namespace) -> GeneratorNormalisation:
    check_deadline(args)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    run = args.plan.resolve().parent
    mode = normalisation_from_plan(plan, args.plan, run)
    posthoc = mode == GeneratorNormalisation.REMOVE_ONE_OUTER_FENCE
    addendum = plan.get("status") == ADDENDUM_STATUS or args.plan.name == "policy_lora_addendum_plan.json"
    data = args.data or run / "data/dataset.jsonl"
    ids = resolve(plan["frozen_ids"])
    inputs_name = "controller_lora_addendum_inputs" if addendum else "controller_fence_inputs" if posthoc else "controller_inputs"
    results_name = "controller_lora_addendum_results" if addendum else "controller_fence_results" if posthoc else "controller_results"
    inputs_root, results_root = args.inputs or run / inputs_name, args.results or run / results_name
    if addendum:
        protected = [run / name for name in ("controller_inputs", "controller_results", "controller_fence_inputs",
                                             "controller_fence_results", "controller_assessment", "controller_assessment_fence")]
        if any(path.resolve().is_relative_to(root.resolve()) for path in (inputs_root, results_root, args.output)
               for root in protected):
            raise ValueError("Exploratory addendum must use directories separate from existing policy analyses")
    if posthoc and (inputs_root.resolve() == (run / "controller_inputs").resolve()
                    or results_root.resolve() == (run / "controller_results").resolve()):
        raise ValueError("Post-hoc inputs and results must be separate from the strict protocol")
    predictions_root = args.predictions or run / "predictions"
    matrix_path = args.matrix_plan or run / "matrix_plan.json"
    matrix_checksum = matrix_path.with_suffix(".sha256")
    if not matrix_checksum.is_file() or matrix_checksum.read_text(encoding="utf-8").strip() != file_hash(matrix_path):
        raise ValueError("Frozen matrix plan checksum is missing or differs")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if file_hash(ids) != plan["frozen_ids_sha256"]:
        raise ValueError("Controller test-ID manifest has changed")
    frozen = json.loads(ids.read_text(encoding="utf-8"))
    if frozen.get("dataset_sha256") != file_hash(data):
        raise ValueError("Controller test IDs belong to a different dataset")
    source_records = read_jsonl(data)
    all_test = [row for row in source_records if row.get("split") == "test"]
    contexts = test_contexts(source_records, ids)
    variants, cap, selection = plan["variants"], plan["primitive_action_budget"], plan["selected_controller"]
    if not contexts or not variants or len(set(variants)) != len(variants) or not isinstance(cap, int) or cap < 1:
        raise ValueError("Invalid planned test contexts, variants or action cap")
    if selection.get("action_cap") != cap:
        raise ValueError("Selected controller action cap differs from plan")
    if addendum:
        validate_addendum_plan(plan, args.plan, run, matrix, matrix_path, contexts)
    analysis_kind = AnalysisKind.ADDENDUM if addendum else AnalysisKind.FENCE if posthoc else AnalysisKind.STRICT
    selection_path = run / "controller_selection.json"
    selected_calibration = None
    if plan.get("controller_selection_sha256") is not None:
        if not selection_path.exists() or file_hash(selection_path) != plan["controller_selection_sha256"]:
            raise ValueError("Controller selection is missing or changed after the plan was frozen")
        selection_record = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_record.get("selected_config") != selection:
            raise ValueError("Plan controller config differs from the selected calibration config")
        calibration_path = run / f"policy_calibration_{selection_record['selected']}.manifest.json"
        selected_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if selected_calibration.get("configuration_sha256") != selection_record.get("selected_configuration_sha256"):
            raise ValueError("Selected calibration manifest does not match the selection")
    loaded = []
    rejected = []
    for variant in variants:
        check_deadline(args)
        try:
            decisions, input_info = eligibility_rows(inputs_root / variant, predictions_root / f"{variant}.jsonl",
                                                      variant, contexts, all_test, ids, data, matrix, matrix_path, mode)
            results, output_info = controller_rows(results_root / f"{variant}.jsonl", variant, selection, input_info, cap,
                                                   getattr(args, "stop_monotonic", None), getattr(args, "stop_epoch", None))
            if selected_calibration is not None and output_info.get("manifest") is not None:
                for field in ("policy_config_sha256", "weights_sha256", "tokenizer_sha256", "source_sha256", "action_parser"):
                    if output_info["manifest"].get(field) != selected_calibration.get(field):
                        raise ValueError(f"Test controller {field} differs from the selected calibration")
        except (ValueError, KeyError, TypeError, OSError) as error:
            reason = f"{type(error).__name__}: {error}"
            decisions, results = {}, {}
            input_info = {"status": "rejected_infrastructure", "path": str(inputs_root / variant),
                          "manifest": None, "rejected_reason": reason}
            output_info = {"path": str(results_root / f"{variant}.jsonl"), "sha256": None, "manifest": None,
                           "rejected_reason": reason, "audit": {"rollouts": 0, "actions": 0, "model_calls": 0}}
            rejected.append({"variant": variant, "reason": reason})
        loaded.append((variant, decisions, input_info, results, output_info))
    comparable = [info["manifest"] for _, _, _, _, info in loaded if info.get("manifest") is not None]
    for field in ("policy_config_sha256", "weights_sha256", "tokenizer_sha256", "source_sha256", "action_parser"):
        known = {json.dumps(item.get(field), sort_keys=True) for item in comparable}
        if len(known) > 1 or (known and "null" in known):
            raise ValueError(f"Controller variants do not use the same frozen {field}")
        if selected_calibration is not None and any(item.get(field) != selected_calibration.get(field) for item in comparable):
            raise ValueError(f"Test controller {field} differs from the selected calibration")
    if comparable:
        for name, digest in comparable[0]["source_sha256"].items():
            source = ROOT / name
            if not source.is_file() or file_hash(source) != digest:
                raise ValueError(f"Local controller source differs from recorded execution: {name}")
    assets_path = run / "controller_assets_snapshot.json"
    assets = None
    if comparable and selected_calibration is not None:
        check_deadline(args)
        if not assets_path.exists():
            raise ValueError("Controller asset snapshot was not recorded before evaluation")
        assets = json.loads(assets_path.read_text(encoding="utf-8"))
        if (assets.get("model") != selection.get("model")
                or assets.get("selection_sha256") != plan["controller_selection_sha256"]):
            raise ValueError("Controller asset snapshot differs from the frozen selection")
        expected_assets = {"model.safetensors", "tokenizer.json", "config.json", "tokenizer_config.json", "generation_config.json"}
        if set(assets.get("files", {})) != expected_assets:
            raise ValueError("Controller asset snapshot is incomplete")
        for name, metadata in assets["files"].items():
            check_deadline(args)
            source = Path(selection["model"]) / name
            if not source.is_file() or sha256_file(source) != metadata.get("sha256"):
                raise ValueError(f"Controller asset changed since test freeze: {name}")
        if (any(item["weights_sha256"] != assets["files"]["model.safetensors"]["sha256"] for item in comparable)
                or any(item["tokenizer_sha256"] != assets["files"]["tokenizer.json"]["sha256"] for item in comparable)):
            raise ValueError("Controller run uses different model weights or tokenizer than frozen assets")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists() or (args.output / "assessment.jsonl").exists():
        raise FileExistsError("Controller assessment exists; choose a fresh output directory")
    config = {"plan_sha256": file_hash(args.plan), "data_sha256": file_hash(data), "test_ids_sha256": file_hash(ids),
              "analysis_kind": analysis_kind,
              "generator_output_normalisation": mode,
              "primary_policy_plan_sha256": plan.get("primary_policy_plan_sha256") if posthoc else None,
              "motivation_summary_sha256": plan.get("motivation_summary_sha256") if posthoc else None,
              "interpretation": ("Exploratory addendum chosen after ICL results, before matched LoRA predictions; "
                                 "strict generator parsing and the same frozen controller."
                                 if addendum else "Post-hoc after observed strict and fence results; not a preregistered comparison."
                                 if posthoc else "Original strict controller protocol; no generator formatting repairs."),
              "matrix_plan_sha256": file_hash(matrix_path),
              "assessment_source_sha256": {
                  str(path.relative_to(ROOT)): file_hash(path) for path in (
                      Path(__file__).resolve(), ROOT / "scripts/assess_intent_representations.py",
                      ROOT / "scripts/audit_luna_rollouts.py", ROOT / "carl/icl/intent_assessment.py",
                      ROOT / "carl/icl/intent_data.py")},
              "controller_selection": selection, "action_cap": cap, "seed": SEED, "bootstrap_resamples": 2000,
              "selected_calibration_configuration_sha256": (selected_calibration or {}).get("configuration_sha256"),
              "controller_assets_sha256": file_hash(assets_path) if assets is not None else None,
              "variants": {name: {"input": {k: v for k, v in inputs.items() if k != "records"}, "output": output}
                           for name, _, inputs, _, output in loaded},
              "denominators": {"pipeline": "completed model outputs with invalid intents counted as failures; missing controller and local errors excluded",
                               "conditional": "valid proposed intents actually evaluated by the controller"},
              "audit": "Every saved rollout replayed against the real environment and fixed prompt."}
    atomic_json(args.output / "config.json", config)
    atomic_json(args.output / "rejected_variants.json", rejected)
    assessed: list[Record] = []
    for variant, decisions, input_info, results, _ in loaded:
        for context in contexts:
            identifier = context["record_id"]
            classified = classify(context, decisions.get(identifier), results.get(identifier))
            if input_info.get("rejected_reason"):
                classified.update(status="rejected_infrastructure", error=input_info["rejected_reason"])
            assessed.append({"variant": variant, "analysis_kind": config["analysis_kind"],
                             "generator_output_normalisation": mode, **classified})
    atomic_text(args.output / "assessment.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in assessed))
    families = sorted({board_family(row) for row in contexts})
    curves = [curve(name, [row for row in assessed if row["variant"] == name], budget, metric, family)
              for name in variants for metric in ("pipeline", "conditional") for family in (None, *families)
              for budget in range(cap + 1)]
    for row in curves:
        row.update(analysis_kind=config["analysis_kind"], generator_output_normalisation=mode)
    summaries = []
    for variant in variants:
        rows = [row for row in assessed if row["variant"] == variant]
        for family in (None, *families):
            selected = [row for row in rows if family is None or row["family"] == family]
            observed = [row for row in selected if row["controller_evaluated"]]
            saved_attempts = [row for row in selected if row["actions"] is not None]
            summaries.append({"analysis_kind": config["analysis_kind"], "generator_output_normalisation": mode,
                              "variant": variant, "family": family or "all", "planned": len(selected),
                              "statuses": dict(Counter(row["status"] for row in selected)),
                              "pipeline_evaluated": sum(row["pipeline_evaluated"] for row in selected),
                              "controller_evaluated": len(observed), "controller_success": sum(success(row, cap) for row in observed),
                              "task_solved": sum(row["task_solved"] is True for row in observed),
                              "saved_rollouts": len(saved_attempts),
                              "attempted_actions": sum(row["actions"] for row in saved_attempts),
                              "no_ops": sum(row["no_ops"] for row in saved_attempts),
                              "model_calls": sum(row["model_calls"] for row in saved_attempts)})
    if addendum:
        contrasts = [compare(assessed, before, after, budget, metric, family)
                     for before, after in ADDENDUM_PAIRS for metric in ("pipeline", "conditional")
                     for family in (None, *families) for budget in range(cap + 1)]
    else:
        contrasts = [compare(assessed, variants[0], variant, cap, metric, family)
                     for variant in variants[1:] for metric in ("pipeline", "conditional") for family in (None, *families)]
    for row in contrasts:
        row.update(analysis_kind=config["analysis_kind"], generator_output_normalisation=mode)
    atomic_json(args.output / "summary.json", {"config": config, "results": summaries, "curves": curves,
                                                 "paired": contrasts, "rejected_variants": rejected})
    write_csv(args.output / "curves.csv", curves, list(curves[0]))
    if contrasts:
        write_csv(args.output / "paired.csv", contrasts, list(contrasts[0]))
    write_csv(args.output / "summary.csv", [{**row, "statuses": json.dumps(row["statuses"], sort_keys=True)}
                                            for row in summaries], list(summaries[0]))
    for family in ("all", *families):
        check_deadline(args)
        plot_curves(args.output, curves, family, variants, posthoc=posthoc, addendum=addendum)
    atomic_json(args.output / "complete.json", {"status": "complete", "plan_sha256": file_hash(args.plan),
                                                "summary_sha256": file_hash(args.output / "summary.json"),
                                                "analysis_kind": config["analysis_kind"]})
    print(json.dumps({"variants": len(variants), "planned_per_variant": len(contexts),
                      "saved_rollouts": sum(info["audit"].get("rollouts", 0) for _, _, _, _, info in loaded)}))
    return mode


def completed_posthoc(directory: Path, plan_hash: str) -> bool:
    marker_path, summary_path = directory / "complete.json", directory / "summary.json"
    if not marker_path.is_file() or not summary_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(marker.get("status") == "complete" and marker.get("analysis_kind") == "posthoc_exploratory"
                and marker.get("plan_sha256") == plan_hash and marker.get("summary_sha256") == file_hash(summary_path))


def fallback_seconds(run: Path, maximum: float) -> float:
    """Respect the earliest original, stage or local deadline with a small reserve."""
    limits = [maximum]
    manifest_path = run / "run_manifest.json"
    if manifest_path.exists():
        state = json.loads(manifest_path.read_text(encoding="utf-8"))
        clock = state.get("clock", {})
        for key, now in (("deadline_epoch", time.time()), ("deadline_monotonic", time.monotonic())):
            if isinstance(clock.get(key), (int, float)):
                limits.append(clock[key] - now - 5)
        for stage in state.get("stages", []):
            if stage.get("status") == "running" and stage.get("pid") == os.getpgrp():
                for key, now in (("stop_epoch", time.time()), ("stop_monotonic", time.monotonic())):
                    if isinstance(stage.get(key), (int, float)):
                        limits.append(stage[key] - now - 5)
    return max(0.0, min(limits))


def run_posthoc_fallback(args: argparse.Namespace) -> None:
    """Recover one missing post-hoc report in-process, without recursion or GPU."""
    run = args.plan.resolve().parent
    fence_plan = run / "policy_evaluation_fence_plan.json"
    if not fence_plan.is_file():
        return
    status_path = args.output / "posthoc_fallback.json"
    record: Record = {"plan": str(fence_plan), "analysis_kind": "posthoc_exploratory"}
    try:
        plan = json.loads(fence_plan.read_text(encoding="utf-8"))
        normalisation_from_plan(plan, fence_plan, run)
        if plan.get("primary_policy_plan_sha256") != file_hash(args.plan):
            raise ValueError("Post-hoc plan does not refer to this strict evaluation plan")
        plan_hash = file_hash(fence_plan)
        base = run / "controller_assessment_fence"
        existing = [base, *sorted(run.glob("controller_assessment_fence_recovery*"))]
        completed = next((path for path in existing if completed_posthoc(path, plan_hash)), None)
        if completed is not None:
            record.update(status="already_complete", output=str(completed))
        else:
            output = base
            for index in range(100):
                if not output.exists():
                    break
                output = run / ("controller_assessment_fence_recovery" + (f"_{index + 1}" if index else ""))
            else:
                raise ValueError("No free post-hoc recovery directory")
            budget = fallback_seconds(run, args.fallback_max_seconds)
            if budget <= 0:
                record.update(status="skipped_deadline", output=str(output), budget_seconds=budget)
            else:
                recovery = argparse.Namespace(**vars(args))
                recovery.plan, recovery.output = fence_plan, output
                recovery.inputs, recovery.results = None, None
                recovery.no_posthoc_fallback = True
                recovery.stop_monotonic = time.monotonic() + budget
                recovery.stop_epoch = time.time() + budget
                record.update(status="running", output=str(output), budget_seconds=budget)
                atomic_json(status_path, record)
                try:
                    assess(recovery)  # One bounded call; assess never invokes this fallback.
                    record["status"] = "complete"
                except AssessmentDeadline as error:
                    record.update(status="timed_out", error=str(error))
                except (ValueError, KeyError, TypeError, OSError) as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}")
    except (ValueError, KeyError, TypeError, OSError) as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    atomic_json(status_path, record)
    print(json.dumps({"posthoc_fallback": record}, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    mode = assess(args)
    if (mode == GeneratorNormalisation.NONE and not args.no_posthoc_fallback
            and args.plan.name != "policy_lora_addendum_plan.json"
            and json.loads(args.plan.read_text(encoding="utf-8")).get("status") != ADDENDUM_STATUS):
        run_posthoc_fallback(args)


if __name__ == "__main__":
    main()
