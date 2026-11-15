"""Render the completed or partial stage-two CPU study from saved artifacts.

The report does not run a controller or update the proposal. Missing artifacts
and infrastructure omissions are reported explicitly, never counted as failures.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.assess_overnight_cpu_stage02 import cluster_mean
from scripts.report_overnight_cpu_stage01 import table
from scripts.run_overnight_cpu_policy import file_sha

FAMILIES = ("12x12x4", "12x12x10")
CONTROLLERS = ("cllp8", "cllp32")
OFFSETS = (1, 2, 4, 8, 16, 32)
LIMITS = (8, 16, 32)


def json_object(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        errors.append(f"{path}: {type(error).__name__}: {error}")
        return None


def csv_rows(path: Path, errors: list[str]) -> list[dict[str, str]] | None:
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))
    except (OSError, UnicodeError, csv.Error) as error:
        errors.append(f"{path}: {type(error).__name__}: {error}")
        return None


def jsonl_rows(path: Path, errors: list[str]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as source:
            values = [json.loads(line) for line in source if line.strip()]
        if not all(isinstance(value, dict) for value in values):
            raise ValueError("expected JSON objects")
        return values
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        errors.append(f"{path}: {type(error).__name__}: {error}")
        return None


def number(value: Any) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def count(value: Any) -> str:
    parsed = number(value)
    return str(int(parsed)) if parsed is not None and parsed.is_integer() else "not available"


def scalar(value: Any, places: int = 3) -> str:
    parsed = number(value)
    return f"{parsed:.{places}f}" if parsed is not None else "not available"


def interval(value: Any, low: Any, high: Any, *, percent: bool = False) -> str:
    point, lower, upper = number(value), number(low), number(high)
    if point is None:
        return "not estimable"
    factor, suffix, places = (100, "%", 1) if percent else (1, "", 3)
    point_text = f"{point * factor:.{places}f}{suffix}"
    if lower is None or upper is None:
        return f"{point_text} [interval unavailable]"
    return (f"{point_text} [{lower * factor:.{places}f}{suffix}, "
            f"{upper * factor:.{places}f}{suffix}]")


def filtered(summary: dict[str, Any], key: str, **wanted: Any) -> list[dict[str, Any]]:
    values = summary.get(key, [])
    if not isinstance(values, list):
        return []
    return [row for row in values if isinstance(row, dict) and row.get("scope") == "family"
            and all(row.get(field) == value for field, value in wanted.items())]


def conditional_ranks(rows: list[dict[str, str]], family: str, controller: str, limit: int) -> dict[str, Any]:
    """Exclude singleton common-success sets and bootstrap whole layouts."""
    relevant = [row for row in rows if row.get("family") == family and row.get("controller") == controller
                and number(row.get("L")) == limit and row.get("status") == "complete"]
    singleton = sum(number(row.get("common_success_targets")) == 1 for row in relevant)
    multi = [row for row in relevant if (number(row.get("common_success_targets")) or 0) >= 2]
    valid = [(row["layout_id"], number(row.get("mean_abs_u_difference"))) for row in multi]
    if any(value is None for _, value in valid):
        raise ValueError("A multi-target context is missing its mean rank difference")
    stats = cluster_mean([(layout, value) for layout, value in valid if value is not None])
    return {"singleton_contexts": singleton, "multi_contexts": len(multi),
            "multi_with_distinct_d": sum((number(row.get("distinct_d")) or 0) >= 2 for row in multi), **stats}


def demo_rows(summary: dict[str, Any]) -> list[list[str]]:
    result = []
    for family in FAMILIES:
        for k in OFFSETS:
            for controller in CONTROLLERS:
                rows = filtered(summary, "demo_reachability", family=family, controller=controller, L=32, k=k)
                if len(rows) > 1:
                    raise ValueError("Duplicate demonstration summary")
                if not rows:
                    result.append([family, str(k), controller, "not available", "not available", "not available"])
                    continue
                row = rows[0]
                result.append([family, str(k), controller,
                               f"{count(row.get('reached_contexts'))}/{count(row.get('completed_contexts'))}",
                               f"{count(row.get('planned_contexts'))} / {count(row.get('missing_contexts'))}",
                               interval(row.get("rate"), row.get("ci_low"), row.get("ci_high"), percent=True)])
    return result


def ranks_rows(summary: dict[str, Any], raw: list[dict[str, str]]) -> list[list[str]]:
    result = []
    for family in FAMILIES:
        for controller in CONTROLLERS:
            rows = filtered(summary, "rank_summary", family=family, controller=controller, L=32)
            if len(rows) > 1:
                raise ValueError("Duplicate rank summary")
            if not rows:
                result.append([family, controller] + ["not available"] * 8)
                continue
            row = rows[0]
            conditional = conditional_ranks(raw, family, controller, 32)
            result.append([family, controller,
                           f"{count(row.get('contexts_with_common_success'))}/{count(row.get('complete_contexts'))}",
                           f"{count(row.get('common_targets'))} / {count(row.get('error_layouts'))}",
                           count(conditional["singleton_contexts"]),
                           f"{count(conditional['multi_contexts'])} / {count(conditional['multi_with_distinct_d'])}",
                           interval(conditional["mean"], conditional["ci_low"], conditional["ci_high"]),
                           interval(row.get("mean_abs_u_difference"), row.get("error_ci_low"), row.get("error_ci_high")),
                           f"{count(row.get('inversions'))}/{count(row.get('strict_comparable'))}; "
                           f"mean {interval(row.get('mean_inversion_rate'), row.get('inversion_ci_low'), row.get('inversion_ci_high'))}",
                           f"{count(row.get('k_tie_only'))}/{count(row.get('d_tie_only'))}/{count(row.get('both_tied'))}"])
    return result


def pools_rows(summary: dict[str, Any]) -> list[list[str]]:
    result = []
    for family in FAMILIES:
        for variant in ("demo4_bfs6", "demo4_bfs4", "underfilled"):
            for controller in CONTROLLERS:
                for limit in LIMITS:
                    rows = filtered(summary, "pool_summary", family=family, controller=controller, L=limit,
                                    pool="fixed", pool_variant=variant)
                    if len(rows) > 1:
                        raise ValueError("Duplicate fixed-pool summary")
                    if not rows:
                        continue
                    row = rows[0]
                    complete, informative = number(row.get("complete_contexts")), number(row.get("informative_contexts"))
                    fewer = complete - informative if complete is not None and informative is not None else None
                    result.append([family, variant, controller, str(limit),
                                   f"{count(row.get('complete_contexts'))}/{count(row.get('planned_contexts'))}",
                                   count(fewer), f"{count(row.get('all_fail_contexts'))}/"
                                   f"{count(row.get('singleton_success_contexts'))}/"
                                   f"{count(row.get('multiple_all_tied_contexts'))}",
                                   interval(row.get("informative_fraction"), row.get("ci_low"), row.get("ci_high"),
                                            percent=True),
                                   f"{count(row.get('incomplete_contexts'))}/"
                                   f"{count(row.get('underfilled_contexts'))}/"
                                   f"{count(row.get('unavailable_contexts'))}"])
    return result


def sensitivity_rows(summary: dict[str, Any]) -> list[list[str]]:
    result = []
    for family in FAMILIES:
        for variant in ("demo4_bfs6", "demo4_bfs4", "underfilled"):
            for controller in CONTROLLERS:
                for reference, comparison in (("fixed", "reduced"), ("fixed", "demo_only"),
                                              ("fixed", "bfs_only"),
                                              ("generator_base", "generator_augmented")):
                    rows = filtered(summary, "sensitivity_summary", family=family, controller=controller, L=32,
                                    pool_variant=variant, reference=reference, comparison=comparison)
                    if len(rows) > 1:
                        raise ValueError("Duplicate sensitivity summary")
                    if not rows:
                        continue
                    row = rows[0]
                    result.append([family, variant, controller, f"{reference} → {comparison}",
                                   f"{count(row.get('complete_pool_contexts'))}/{count(row.get('planned_contexts'))}",
                                   f"{count(row.get('contexts_with_shared_success'))} / "
                                   f"{count(row.get('shared_success_targets'))} / {count(row.get('n_layouts'))}",
                                   interval(row.get("mean_abs_delta_u"), row.get("ci_low"), row.get("ci_high"))])
    return result


def bfs_rank_rows(summary: dict[str, Any]) -> list[list[str]]:
    """Report BFS-depth and controller-duration ranks on successful BFS targets."""
    result = []
    for family in FAMILIES:
        for variant in ("demo4_bfs6", "demo4_bfs4"):
            for controller in CONTROLLERS:
                rows = filtered(summary, "bfs_rank_summary", family=family, controller=controller, L=32,
                                pool_variant=variant)
                if len(rows) > 1:
                    raise ValueError("Duplicate BFS rank summary")
                if not rows:
                    continue
                row = rows[0]
                result.append([family, variant, controller,
                               f"{count(row.get('complete_contexts'))}/{count(row.get('planned_contexts'))}",
                               f"{count(row.get('contexts_with_bfs_success'))}/{count(row.get('success_layouts'))}",
                               interval(row.get("mean_abs_rank_difference"), row.get("ci_low"), row.get("ci_high")),
                               f"{count(row.get('inversions'))}/{count(row.get('strict_comparable'))}",
                               f"{count(row.get('source_depth_tie_only'))}/{count(row.get('d_tie_only'))}/"
                               f"{count(row.get('both_tied'))}"])
    return result


def profile_examples(profiles: list[dict[str, str]], labels: list[dict[str, str]]) -> list[list[str]]:
    """Illustrate the first complete board per family/controller, with outcomes marked."""
    result = []
    for family in FAMILIES:
        for controller in CONTROLLERS:
            candidates = [row for row in profiles if row.get("family") == family
                          and row.get("controller") == controller and number(row.get("L")) == 32
                          and row.get("pool") == "fixed" and row.get("status") == "complete"]
            if not candidates:
                continue
            candidates.sort(key=lambda row: (number(row.get("board_rank"))
                                              if number(row.get("board_rank")) is not None else math.inf,
                                              row.get("context_id", "")))
            profile = candidates[0]
            selected = [row for row in labels if row.get("context_id") == profile.get("context_id")
                        and row.get("controller") == controller and number(row.get("L")) == 32
                        and row.get("pool") == "fixed" and row.get("pool_status") == "complete"
                        and row.get("status") == "complete"]
            selected.sort(key=lambda row: (row.get("slot", ""), row.get("candidate_id", "")))
            successes = [row for row in selected if number(row.get("p")) == 1]
            failures = [row for row in selected if number(row.get("p")) == 0]
            examples = ([successes[0], failures[0]] if successes and failures else
                        successes[:2] if successes else failures[:2])
            for row in examples:
                reached = number(row.get("p")) == 1
                if not reached and any(number(row.get(key)) is not None for key in ("d", "d_over_median", "u")):
                    raise ValueError("A failed example has a defined duration or scale")
                origin = row.get("candidate_origin", "unknown")
                detail = (f"k={row.get('selected_demo_offset') or 'unavailable'}" if origin == "demonstration" else
                          f"depth={row.get('bfs_depth') or 'unavailable'}" if origin == "bfs" else "generated")
                result.append([family, controller, f"{row.get('board_rank', '?')} / "
                               f"{row.get('context_id', '')[:12]}", str(row.get("source_family", "?")),
                               f"{origin} ({detail})", "1" if reached else "0",
                               count(row.get("d")) if reached else "null", scalar(profile.get("median_d")),
                               scalar(row.get("d_over_median")) if reached else "null",
                               scalar(row.get("u")) if reached else "null"])
    return result


def budget_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    """Compare the same achieved targets under two reference action limits."""
    result = []
    for row in sorted(rows, key=lambda item: (item.get("family", ""), item.get("pool_variant", ""),
                                               item.get("controller", ""), number(item.get("short_L")) or 0)):
        if (row.get("source_family") != "all" or number(row.get("short_L")) not in (8, 16)
                or number(row.get("long_L")) != 32):
            continue
        result.append([row.get("family", "?"), row.get("pool_variant", "?"), row.get("controller", "?"),
                       f"{count(row.get('short_L'))} → 32", f"{count(row.get('complete_contexts'))}/"
                       f"{count(row.get('planned_contexts'))}",
                       f"{count(row.get('contexts_with_shared_success'))} / {count(row.get('shared_success_targets'))}",
                       f"{count(row.get('contexts_with_new_success'))} / {count(row.get('new_success_targets'))}",
                       interval(row.get("mean_abs_delta_u"), row.get("ci_low"), row.get("ci_high")),
                       count(row.get("delta_layouts"))])
    return result


def primary_findings(summary: dict[str, Any]) -> list[str]:
    """Describe the comparable longest-offset cells without selecting a winner."""
    findings = []
    for family in FAMILIES:
        cells = [filtered(summary, "demo_reachability", family=family, controller=model, L=32, k=32)
                 for model in CONTROLLERS]
        if any(len(cell) != 1 for cell in cells):
            continue
        left, right = (cell[0] for cell in cells)
        if (left.get("missing_contexts") != 0 or right.get("missing_contexts") != 0
                or left.get("planned_contexts") != right.get("planned_contexts")):
            continue
        findings.append(f"- **{family}, demonstration k=32:** CLLP-8 reached "
                        f"{count(left.get('reached_contexts'))}/{count(left.get('completed_contexts'))}; "
                        f"CLLP-32 reached {count(right.get('reached_contexts'))}/{count(right.get('completed_contexts'))}. "
                        "Each denominator counts boards with that eligible offset.")
    return findings


def scale_findings(summary: dict[str, Any], ranks: list[dict[str, str]],
                   budgets: list[dict[str, str]]) -> list[str]:
    """Summarise matched diagnostics only when their complete denominators exist."""
    rank_details, sensitivity_details = [], []
    for family in FAMILIES:
        rank_pieces = []
        for controller in CONTROLLERS:
            matches = filtered(summary, "rank_summary", family=family, controller=controller, L=32)
            if len(matches) != 1:
                continue
            row = matches[0]
            if (not number(row.get("planned_contexts"))
                    or number(row.get("complete_contexts")) != number(row.get("planned_contexts"))
                    or number(row.get("incomplete_contexts")) != 0):
                continue
            stats = conditional_ranks(ranks, family, controller, 32)
            inversions, comparable = number(row.get("inversions")), number(row.get("strict_comparable"))
            if stats["mean"] is None or not stats["multi_contexts"] or inversions is None or not comparable:
                continue
            rank_pieces.append(f"{controller.upper()} |u(d)−u(k)| "
                               f"{interval(stats['mean'], stats['ci_low'], stats['ci_high'])} "
                               f"across {count(stats['multi_contexts'])} contexts with at least two shared successes; "
                               f"{count(inversions)}/{count(comparable)} strict inversions")
        if rank_pieces:
            rank_details.append(f"{family}: " + "; ".join(rank_pieces))
        budget = [row for row in budgets if row.get("family") == family and row.get("source_family") == "all"
                  and row.get("controller") == "cllp8" and row.get("pool_variant") == "demo4_bfs6"
                  and number(row.get("short_L")) == 8 and number(row.get("long_L")) == 32]
        pool = filtered(summary, "sensitivity_summary", family=family, controller="cllp8", L=32,
                        pool_variant="demo4_bfs6", reference="fixed", comparison="reduced")
        if len(budget) != 1 or len(pool) != 1:
            continue
        short, reduced = budget[0], pool[0]
        if (not number(short.get("planned_contexts")) or not number(reduced.get("planned_contexts"))
                or number(short.get("complete_contexts")) != number(short.get("planned_contexts"))
                or number(reduced.get("complete_pool_contexts")) != number(reduced.get("planned_contexts"))
                or not number(short.get("contexts_with_shared_success"))
                or not number(reduced.get("contexts_with_shared_success"))
                or number(short.get("mean_abs_delta_u")) is None
                or number(reduced.get("mean_abs_delta_u")) is None):
            continue
        sensitivity_details.append(f"{family}: "
            f"{interval(short['mean_abs_delta_u'], short.get('ci_low'), short.get('ci_high'))} "
            f"when L changed from 8 to 32 (n={count(short['contexts_with_shared_success'])} contexts), "
            f"and {interval(reduced['mean_abs_delta_u'], reduced.get('ci_low'), reduced.get('ci_high'))} "
            f"from fixed to reduced pool (n={count(reduced['contexts_with_shared_success'])} contexts)")
    findings = []
    if rank_details:
        findings.append("- **Demonstration ranks at L=32:** " + "; ".join(rank_details) + ". "
                        "This comparison is conditional on both controllers reaching the same targets.")
    if sensitivity_details:
        findings.append("- **CLLP-8, 4+6 pool:** for shared successful targets, mean |Δu| was "
                        + "; ".join(sensitivity_details) + ".")
    outcomes = filtered(summary, "candidate_summary", L=32)
    if outcomes and all(number(row.get("completed_attempts")) is not None
                        and number(row.get("successes")) is not None for row in outcomes):
        completed = sum(int(number(row["completed_attempts"])) for row in outcomes)
        missed = sum(int(number(row["completed_attempts"]) - number(row["successes"])) for row in outcomes)
        if completed > 0 and missed > 0:
            findings.append(f"- Across the two controllers, {missed}/{completed} completed candidate attempts "
                            "did not reach their target within 32 actions. The profiles describe dependence "
                            "on the candidate pool and budget; they do not show that u is better than absolute d, "
                            "or establish transfer or improved planning.")
    return findings


def generator_rows(rows: list[dict[str, str]]) -> list[list[str]]:
    result = []
    for row in sorted(rows, key=lambda item: (item.get("family", ""), item.get("controller", ""))):
        if row.get("source_family") != "all" or number(row.get("L")) != 32:
            continue
        planned, successes = number(row.get("planned_outputs")), number(row.get("successful_outputs"))
        missing_generator = number(row.get("generator_missing_outputs"))
        missing_controller = number(row.get("valid_missing_controller"))
        if (planned is not None and planned > 0 and successes is not None
                and missing_generator is not None and missing_controller is not None):
            bounds = f"{100 * successes / planned:.1f}–{100 * (successes + missing_generator + missing_controller) / planned:.1f}%"
        else:
            bounds = "not available"
        result.append([row.get("family", "?"), row.get("controller", "?"),
                       f"{count(row.get('valid_outputs'))}/{count(row.get('planned_outputs'))}",
                       count(row.get("invalid_outputs")), count(row.get("duplicate_outputs")),
                       f"{count(row.get('generator_missing_outputs'))}/{count(row.get('valid_missing_controller'))}",
                       f"{count(row.get('successful_outputs'))}/{count(row.get('planned_outputs'))}", bounds,
                       f"{count(row.get('successful_outputs'))}/{count(row.get('valid_measured_outputs'))}",
                       interval(row.get("success_per_requested_equal_context"), row.get("ci_low"), row.get("ci_high"),
                                percent=True), f"{count(row.get('complete_contexts'))}/"
                       f"{count(row.get('planned_contexts'))}", count(row.get("bootstrap_layouts"))])
    return result


def status_text(status: dict[str, Any] | None) -> str:
    if status is None:
        return "not available"
    state = status.get("state", "not available")
    planned, completed = status.get("planned"), status.get("completed")
    if completed is None:
        parts = [number(status.get(key)) for key in ("imported", "previously_saved", "new")]
        completed = sum(parts) if all(part is not None for part in parts) else None
        qualifier = "at least " if completed is not None else ""
    else:
        qualifier = ""
    return (f"{state}; {qualifier}{count(completed)}/{count(planned)} reported attempts; "
            f"reused {count(status.get('imported'))}, newly run {count(status.get('new'))}")


def commands(manifest: dict[str, Any] | None) -> list[list[str]]:
    names = ("stage02_bfs_evaluation", "stage02_generator_evaluation", "stage02_merge_evaluation",
             "stage02_controller_evaluation", "stage02_prepare_campaign_train", "stage02_controller_train",
             "stage02_assess_evaluation", "stage02_assess_campaign_train", "stage02_summarise_generator",
             "stage02_budget_evaluation", "stage02_sensitivity_figure", "stage02_independent_audit_evaluation",
             "stage02_independent_audit_train", "stage02_independent_rank_audit")
    stages = manifest.get("stages", []) if manifest is not None else []
    result = []
    relevant = [stage for stage in stages if isinstance(stage, dict) and (
        stage.get("name") in names or (isinstance(stage.get("command"), list) and any(
            module in stage["command"] for module in ("scripts.summarise_overnight_cpu_stage02_generator",
                                                       "scripts.summarise_overnight_cpu_stage02_budget"))))]
    for stage in relevant:
        name = str(stage.get("name", "unnamed"))
        argv = stage.get("command")
        if not isinstance(argv, list) or not all(isinstance(part, str) for part in argv):
            continue
        if any(part.startswith("OPENAI_API_KEY=") and part != "OPENAI_API_KEY=offline-test-key" for part in argv):
            rendered = "not shown: command contains a credential; consult the local supervisor manifest"
        else:
            rendered = shlex.join(argv)
        result.append([name, str(stage.get("status", "unknown")), rendered])
    return result


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def optional_pool_figure(analysis: Path) -> tuple[bool, str]:
    """Validate the separately produced figure without blocking a partial report."""
    figure = analysis / "pool_budget_sensitivity.pdf"
    if not figure.is_file():
        return False, "The optional pool-and-budget sensitivity figure is not available yet."
    issues: list[str] = []
    manifest = json_object(analysis / "pool_budget_sensitivity_manifest.json", issues)
    outputs = manifest.get("outputs_sha256") if manifest is not None else None
    expected = outputs.get(figure.name) if isinstance(outputs, dict) else None
    if issues or not isinstance(expected, str):
        return False, "The optional pool-and-budget sensitivity figure has no readable hash manifest."
    try:
        actual = file_sha(figure)
    except OSError:
        return False, "The optional pool-and-budget sensitivity figure could not be read."
    if actual != expected:
        return False, "The optional pool-and-budget sensitivity figure does not match its saved hash."
    return True, "The pool-and-budget sensitivity PDF matches its separate figure manifest."


def report(run_dir: Path) -> tuple[str, str]:
    stage, errors = run_dir / "stage_02", []
    analysis = stage / "analysis_evaluation"
    metadata = json_object(analysis / "analysis_manifest.json", errors)
    summary = json_object(analysis / "summaries.json", errors)
    ranks = csv_rows(analysis / "common_ranks.csv", errors)
    profiles = csv_rows(analysis / "profiles.csv", errors)
    labels = csv_rows(analysis / "profile_labels.csv", errors)
    generator = csv_rows(analysis / "generator_proposal_summary.csv", errors)
    budgets = csv_rows(analysis / "budget_sensitivity_summary.csv", errors)
    eval_status = json_object(stage / "controller_evaluation/status.json", errors)
    train_status = json_object(stage / "controller_train/status.json", errors)
    eval_contexts = jsonl_rows(stage / "data_evaluation/contexts.jsonl", errors)
    train_contexts = jsonl_rows(stage / "data_train/contexts.jsonl", errors)
    train_candidates = jsonl_rows(stage / "data_train/candidates.jsonl", errors)
    merge_data = json_object(stage / "merged_evaluation/generator_merge_manifest.json", errors)
    supervisor = json_object(run_dir / "stage_02_supervision/run_manifest.json", errors)
    audit_errors: list[str] = []
    evaluation_audit = json_object(analysis / "independent_audit.json", audit_errors)
    training_audit = json_object(stage / "analysis_train/independent_audit.json", audit_errors)
    rank_audit = json_object(stage / "independent_rank_audit.json", audit_errors)
    pool_figure_ready, pool_figure_note = optional_pool_figure(analysis)

    verified = metadata is not None and summary is not None
    if verified:
        expected = metadata.get("output_sha256", {}).get("summaries.json")
        if expected is None or file_sha(analysis / "summaries.json") != expected:
            errors.append("Evaluation summary does not match its analysis manifest")
            verified = False
        for name in ("common_ranks.csv", "profiles.csv", "profile_labels.csv"):
            expected_file = metadata.get("output_sha256", {}).get(name)
            result_path = analysis / name
            if expected_file is None or not result_path.is_file() or file_sha(result_path) != expected_file:
                errors.append(f"{name} does not match the analysis manifest")
                verified = False
        rollouts = stage / "controller_evaluation/rollouts.jsonl"
        if metadata.get("rollouts_sha256") and (not rollouts.is_file()
                or file_sha(rollouts) != metadata["rollouts_sha256"]):
            errors.append("Evaluation rollouts differ from the assessed snapshot")
            verified = False
    if not verified:
        summary, ranks, profiles, labels = None, None, None, None
    required_summaries = {"demo_reachability", "rank_summary", "pool_summary", "sensitivity_summary", "bfs_rank_summary"}
    summary_ready = (summary is not None and ranks is not None and profiles is not None and labels is not None
                     and all(isinstance(summary.get(key), list) and summary[key] for key in required_summaries))
    generator_ready = generator is not None and bool(generator_rows(generator))
    budget_ready = budgets is not None and bool(budget_rows(budgets))
    figure_names = ("demo_reachability.pdf", "scale_diagnostics.pdf")
    figure_hashes = metadata.get("output_sha256", {}) if metadata is not None else {}
    figures_ready = all((analysis / name).is_file() and figure_hashes.get(name) is not None
                        and file_sha(analysis / name) == figure_hashes[name] for name in figure_names)
    both_families = eval_contexts is not None and all(
        any(row.get("family") == family for row in eval_contexts) for family in FAMILIES)
    complete = (summary_ready and generator_ready and budget_ready and metadata is not None and figures_ready and not errors
                and metadata.get("missing_attempts") == 0 and (number(metadata.get("planned_attempts")) or 0) > 0
                and eval_status is not None
                and eval_status.get("state") == "complete" and train_status is not None
                and train_status.get("state") == "complete" and eval_contexts is not None and train_contexts is not None
                and train_candidates is not None and both_families)
    state = "complete" if complete else "partial"
    lines = ["# Stage 2 — fixed-controller temporal profiles", "",
             f"**Report status: {state}.** This is an exploratory full-state controller study, not predicate-intent "
             "execution or a complete-task planning experiment.", "",
             f"Evaluation: {status_text(eval_status)}. Campaign train: {status_text(train_status)}."]
    if metadata is not None:
        lines.append(f"Recorded assessment snapshot: {count(metadata.get('completed_attempts'))}/"
                     f"{count(metadata.get('planned_attempts'))} attempts; "
                     f"{count(metadata.get('missing_attempts'))} missing. Missing attempts are not controller failures.")
    else:
        lines.append("The evaluation assessment is not available yet; no result has been inferred.")
    if errors:
        lines.extend(["", "Artifact issues:", *(f"- {error}" for error in errors)])

    lines.extend(["", "## Data and method", "",
                  "CLLP-8 and CLLP-32 are frozen, deterministic greedy controllers receiving the current board "
                  "and the **complete target board**. The separate frozen BART generator receives the current "
                  "board only. Stage-one demonstration attempts are imported after replay and provenance checks; "
                  "new full targets are attempted once per controller. The cutoff is 32 primitive actions, "
                  "including no-ops; L=8/16/32 are prefixes of the same trace.", "",
                  "With M=1, observed p is 0 or 1; d is first-hit duration and is undefined on failure. "
                  "Within each complete reference pool, u is the midpoint of the empirical CDF jump over "
                  "successful d values; ties share their midpoint. No-success candidates have no u. The median "
                  "used for d/median is the conventional arithmetic median, including the average of the two "
                  "middle values for even samples. These measured labels are retrospective evaluator "
                  "information; a deployed generator can receive a requested scale, but does not observe the "
                  "held-out context's measured label profile or future states.", "",
                  "BFS enumerates valid states up to the recorded depth and node limits; its depth is neither "
                  "controller duration d nor a proof that an unobserved target is impossible. No model training "
                  "or fine-tuning was performed in this stage.", "",
                  "The evaluation cohort was already inspected and is exploratory. Source distributions differ "
                  "between families, so box count is confounded. New campaign-train boards are disjoint from the "
                  "recognised history and stage-one layouts, but the historical split and checkpoint training "
                  "exposure are unknown. These results do not establish unseen-board transfer, "
                  "scale-conditioned generator behaviour, or improved planning."])
    historical = run_dir.parent / "overnight_relative_scale_2026-11-07/REPORT.md"
    if historical.is_file():
        historical_text = historical.read_text(encoding="utf-8")
        if "90.6%" in historical_text and "98.1%" in historical_text:
            lines.extend(["", "The [7 November exploratory report](../overnight_relative_scale_2026-11-07/REPORT.md) "
                          "reported 90.6% and 98.1% label agreement on reached demonstration targets in "
                          "different cohorts and under different conventions. The present comparison "
                          "conditions on common successes of both controllers, so it is not a direct repeat "
                          "and does not establish an advantage of relative scale over absolute duration."])

    if eval_contexts is not None or train_contexts is not None:
        cohort_rows = []
        for label, contexts in (("evaluation", eval_contexts), ("campaign train", train_contexts)):
            for family in FAMILIES:
                selected = [row for row in contexts or [] if row.get("family") == family]
                cohort_rows.append([label, family, count(len(selected)) if contexts is not None else "not available",
                                    count(len({row.get('layout_id') for row in selected}))
                                    if contexts is not None else "not available"])
        lines.extend(["", table(["Role", "Family", "Contexts", "Canonical boards"], cohort_rows)])
    if train_candidates is not None:
        origins = Counter(row.get("candidate_origin", "unknown") for row in train_candidates)
        lines.extend(["", "Campaign-train candidate origins: " + "; ".join(
            f"{name}: {number}" for name, number in sorted(origins.items())) + "."])
    if eval_contexts is not None:
        coverage = Counter((row.get("family"), row.get("pool_variant")) for row in eval_contexts)
        lines.extend(["", "Frozen evaluation pool coverage (before policy outcomes):", "",
                      table(["Family", "Pool", "Contexts"], [[family, variant, n]
                          for (family, variant), n in sorted(coverage.items(), key=lambda item: str(item[0]))])])
    if merge_data is not None:
        pool_status = merge_data.get("pool_status_counts")
        if isinstance(pool_status, dict):
            lines.extend(["", "Generator pool coverage: " + "; ".join(f"{key}: {value}"
                         for key, value in sorted(pool_status.items())) + "."])
    if summary_ready and summary is not None and ranks is not None and profiles is not None and labels is not None:
        findings = primary_findings(summary) + scale_findings(summary, ranks, budgets or [])
        if findings:
            lines.extend(["", "## Findings", "", *findings])
        lines.extend(["", "## Demonstration targets", "",
                      "All available demonstration offsets are included at L=32, even when two offsets name "
                      "the same target. A missing offset is not a failed attempt. Rates and intervals are "
                      "board-weighted; intervals are 95% bootstrap over original canonical layouts. "
                      "Percentile bootstrap intervals can collapse to [0,0] or [1,1] when all observed "
                      "outcomes agree; this does not imply population certainty.", "",
                      table(["Family", "k", "Controller", "Reached / observed", "Planned / missing",
                             "Rate [95% CI]"], demo_rows(summary)), "",
                      "## Common-success rank diagnostic", "",
                      "For each context, both controllers must reach the same demonstration target within "
                      "L=32. The ranks u(d) and u(k) are recomputed on that intersection; the earliest k is "
                      "used for aliases. Single-target contexts have zero difference by construction, so the "
                      "primary conditional column excludes them. The all-n>0 column retains them. "
                      "Strict inversions exclude ties; tie counts are k only / d only / both. Bootstrap "
                      "resamples original boards, not targets.", "",
                      table(["Family", "Controller", "Contexts with common success / complete",
                             "Common targets / layouts", "Singleton contexts", "≥2 successes / ≥2 distinct d",
                             "|u(d)−u(k)| ≥2 successes [95% CI]", "All n>0 [95% CI]",
                             "Inversions / strict; mean rate [95% CI]", "Tie pairs k/d/both"],
                            ranks_rows(summary, ranks)), "",
                      "## BFS depth versus controller duration", "",
                      "For each enumerated target, source_depth is the shortest BFS depth in the explored "
                      "state graph under the recorded termination rules. It is neither demonstration offset k "
                      "nor controller duration d. Ranks below use only successful fixed-pool BFS targets; "
                      "the search cutoff does not establish that unobserved targets are impossible. Tie "
                      "counts are depth only / d only / both.", "",
                      table(["Family", "Pool variant", "Controller", "Complete / planned contexts",
                             "Contexts / layouts with BFS success", "Mean absolute rank difference [95% CI]",
                             "Inversions / strict", "Tie pairs depth/d/both"], bfs_rank_rows(summary)), "",
                      "## Fixed-pool information", "",
                      "A complete pool is informative here only when at least two successful targets have "
                      "different d. The 4+6 and 4+4 pool variants are separate; underfilled and incomplete "
                      "pools are reported rather than assigned scale labels. A context can have several "
                      "successful targets but still be tied. The 4+4 variant preserves the demo/BFS proportion, "
                      "but its available BFS strata may differ across contexts and need not have the same difficulty "
                      "distribution. Candidate selection deliberately prioritises the short demonstration "
                      "offsets k=1 and k=4 when available and includes shallow BFS strata; the resulting "
                      "informative fraction is not representative of arbitrary candidate pools or generator outputs.", "",
                      table(["Family", "Pool variant", "Controller", "L", "Complete / planned", "<2 distinct d",
                             "All fail / singleton / multiple tied", "Informative [95% CI]",
                             "Incomplete / underfilled / unavailable"], pools_rows(summary)), "",
                      "## Examples of measured targets", "",
                      "For each family/controller, select the lowest board_rank with a complete L=32 fixed "
                      "profile, breaking ties by context ID. Within that profile show the first successful "
                      "and first unsuccessful target in slot/ID order; if only one outcome exists, show up to "
                      "two of that type. These outcome-stratified examples illustrate the raw measurements; "
                      "they are not additional independent results. Failed targets have undefined d, "
                      "d/median and u.", "",
                      table(["Family", "Controller", "Board rank / context", "Source distribution", "Target",
                             "p", "d (actions)", "Profile median d", "d / median d", "u"],
                            profile_examples(profiles, labels)), "",
                      "## Sensitivity to candidate pool", "",
                      "At L=32, entries compare u for shared successful targets, averaged by context. "
                      "Fixed is compared with its declared reduced, demo-only and BFS-only subsets. "
                      "Reduced takes the first two demo slots and first three BFS slots (or two in the 4+4 "
                      "variant); it is an ordered subset, not a representative random sample. "
                      "The generator comparison adds one generated target to the exact same 4-demo+4-BFS "
                      "base. Results for L=8/16 and individual source distributions are in sensitivity_summary.csv.", "",
                      table(["Family", "Pool variant", "Controller", "Reference → comparison",
                             "Complete / planned contexts", "Shared contexts / targets / layouts",
                             "Mean |Δu| [95% CI]"], sensitivity_rows(summary))])
    else:
        lines.extend(["", "## Evaluation results", "", "A complete, hash-matched evaluation summary and "
                      "common-rank table are not yet available."])
    lines.extend(["", "## Sensitivity to action limit", "",
                  "For the same successful targets, compare u recomputed at the short and long action limits; "
                  "newly successful targets can change the reference distribution. Pool variants remain separate. "
                  "Intervals resample original boards."])
    if budgets and budget_rows(budgets):
        lines.extend(["", table(["Family", "Pool variant", "Controller", "Limit", "Complete / planned contexts",
                                 "Shared-success contexts / targets", "Contexts / new successes",
                                 "Mean |Δu| [95% CI]", "Boards"], budget_rows(budgets))])
    else:
        lines.extend(["", "The action-limit sensitivity summary is not available yet."])
    lines.extend(["", "## Raw generator outputs", "",
                  "Each context requested two raw outputs. Invalid outputs, duplicates and missing controller "
                  "traces remain visible. Duplicates can refer to the same attempted target, so raw output "
                  "counts are not independent controller trials. Structural validity does not prove reachability. "
                  "Invalid outputs count as unsuccessful proposals, not CLLP failures. Requested-output bounds "
                  "treat unobserved generator/controller outputs as unknown; they are per-output counts, "
                  "not confidence intervals for the context-weighted mean. The separate observed per-request "
                  "mean includes invalid outputs and averages only complete contexts, which can be selective "
                  "when work is partial. Its uncertainty resamples original boards."])
    if generator:
        lines.extend(["", table(["Family", "Controller", "Valid / requested", "Invalid", "Duplicates",
                                 "Missing generator / controller", "Observed reached / requested",
                                 "Requested-output bounds", "Reached / measured valid",
                                 "Observed complete-context mean [95% CI]", "Complete / planned contexts",
                                 "Boards"], generator_rows(generator))])
    else:
        lines.extend(["", "The generator proposal summary is not available yet."])
    lines.extend(["", "## Independent verification", ""])
    for name, audit, link in (("Evaluation profile audit", evaluation_audit,
                               "stage_02/analysis_evaluation/independent_audit.json"),
                              ("Training profile audit", training_audit,
                               "stage_02/analysis_train/independent_audit.json")):
        if audit is None:
            lines.append(f"- {name}: not available.")
            continue
        counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
        lines.append(f"- [{name}]({link}): {audit.get('state', 'unknown')}; "
                     f"{count(audit.get('error_count'))} errors; {count(counts.get('profiles'))} profiles, "
                     f"{count(counts.get('checks_label_p'))} p checks, {count(counts.get('checks_label_d'))} d checks, "
                     f"{count(counts.get('checks_label_u'))} u checks and "
                     f"{count(counts.get('bfs_records_replayed'))} replayed BFS paths.")
    if rank_audit is None:
        lines.append("- Independent rank audit: not available.")
    else:
        groups = rank_audit.get("results") if isinstance(rank_audit.get("results"), dict) else {}
        demo = groups.get("demo") if isinstance(groups.get("demo"), dict) else {}
        bfs = groups.get("bfs") if isinstance(groups.get("bfs"), dict) else {}
        lines.append(f"- [Independent rank audit](stage_02/independent_rank_audit.json): "
                     f"{rank_audit.get('status', 'unknown')}; demo errors {count(demo.get('error_count'))}, "
                     f"BFS errors {count(bfs.get('error_count'))}.")
    if audit_errors:
        lines.append("- Some optional audit artifacts could not be read: " + "; ".join(audit_errors) + ".")
    lines.extend(["", "## Files and reproduction", "",
                  "[Demo reachability PDF](stage_02/analysis_evaluation/demo_reachability.pdf) · "
                  + ("[Pool and budget sensitivity PDF](stage_02/analysis_evaluation/pool_budget_sensitivity.pdf) · "
                     if pool_figure_ready else "")
                  + "[Additional scale diagnostics PDF](stage_02/analysis_evaluation/scale_diagnostics.pdf) · "
                  "[Profiles JSONL](stage_02/analysis_evaluation/profiles.jsonl) · "
                  "[All summary CSVs](stage_02/analysis_evaluation/) · "
                  "[Generator proposal summary CSV](stage_02/analysis_evaluation/generator_proposal_summary.csv) · "
                  "[Action-limit sensitivity CSV](stage_02/analysis_evaluation/budget_sensitivity_summary.csv)", "",
                  pool_figure_note, "",
                  "[Evaluation raw attempts and contract](stage_02/controller_evaluation/) · "
                  "[Train raw attempts and contract](stage_02/controller_train/) · "
                  "[Evaluation freeze](stage_02/evaluation_freeze.json) · "
                  "[Candidate freeze](stage_02/controller_candidate_freeze.json) · "
                  "[Train exposure register](stage_02/data_train/exposure_register.json) · "
                  "[Supervisor manifest](stage_02_supervision/run_manifest.json) · "
                  "[Execution source snapshots](stage_02/source_snapshots/manifest.json)", "",
                  "The source snapshots preserve the exact files and hashes used for the completed runs. "
                  "The committed runner was subsequently changed only by import formatting; its different "
                  "hash cannot resume the original controller contract. The snapshot README gives the "
                  "matching executable path without resetting the campaign clock.", "",
                  "Recorded commands below are reproduced from the local supervisor manifest. "
                  "Resume only with the same frozen input and contract; consult the manifest for time limits, "
                  "stage status and any steps not listed here.", ""])
    recorded = commands(supervisor)
    if recorded:
        for name, command_status, command in recorded:
            lines.extend([f"**{name}** ({command_status})", "", "```sh", command, "```", ""])
    else:
        lines.extend(["Supervisor commands are not available yet.", ""])
    for name, status in (("stage02_controller_evaluation", eval_status), ("stage02_controller_train", train_status)):
        if status is None or status.get("state") == "complete":
            continue
        matches = [command for recorded_name, _, command in recorded if recorded_name == name]
        if matches and not matches[-1].startswith("not shown:"):
            lines.extend([f"To resume {name} if its original cutoff and campaign deadline still permit:", "", "```sh",
                          f"{matches[-1]} --resume", "```", ""])
    lines.extend(["Regenerate this report after evaluation, assessment and generator-summary completion:", "", "```sh",
                  f"uv run --offline --no-sync python -m scripts.report_overnight_cpu_stage02 --run-dir {shlex.quote(str(run_dir))}",
                  "```", ""])
    return "\n".join(lines), state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    body, state = report(run_dir)
    target = run_dir / "stage_02_report.md"
    atomic_text(target, body)
    print(json.dumps({"report": str(target), "state": state}))


if __name__ == "__main__":
    main()
