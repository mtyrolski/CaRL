#!/usr/bin/env python3
"""Assess frozen generator proposals with bounded search, not a controller."""

import argparse
import csv
import json
import os
import random
from collections import Counter
from collections import defaultdict
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from carl.icl.intent_assessment import bounded_witness
from carl.icl.intent_generator_experiment import read_records


class GeneratorMethod(StrEnum):
    """Generator variants included in the local fallback."""

    BASE = "B"
    DEMONSTRATION = "D"
    LOCAL_RULE = "R"


METHODS = tuple(GeneratorMethod)
LABELS = {GeneratorMethod.BASE: "Base Qwen", GeneratorMethod.DEMONSTRATION: "Qwen + LoRA",
          GeneratorMethod.LOCAL_RULE: "Local rule"}


def _goals(rows: list[str]) -> int:
    return sum(row.count("*") for row in rows)


def _interval(values: list[float], seed: int) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(2000))
    return [samples[49], samples[1949]]


def _local_rule_work(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("walk_graphs", "walk_nodes_expanded", "walk_edges_examined", "push_checks", "legal_pushes",
              "goal_distance_checks", "elapsed_seconds")
    metrics = {}
    for key in fields:
        values = [row["local_rule_work"].get(key) if isinstance(row.get("local_rule_work"), dict) else None
                  for row in rows]
        known = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        metrics[key] = {"total": sum(known) if len(known) == len(rows) else None,
                        "known_sum": sum(known) if known else None, "known_outputs": len(known)}
    return {"requested_outputs": len(rows), "metrics": metrics}


def validate_cases(cases: list[dict[str, Any]]) -> None:
    """Check pairing before any potentially expensive local search."""
    if not cases or len({case["record_id"] for case in cases}) != len(cases):
        raise ValueError("Nonempty distinct evaluation cases are required")
    statuses = {"valid_intent", "invalid_json", "invalid_intent", "infrastructure_error", "not_attempted"}
    layouts = {case["layout_id"] for case in cases}
    for layout in layouts:
        group = [case for case in cases if case["layout_id"] == layout]
        if len({tuple(case["start_observation"]) for case in group}) != 1:
            raise ValueError("Every method must receive the same state for each board")
        for method in METHODS:
            slots = [case for case in group if case["arm"] == method]
            if len(slots) != 2 or {case["requested_scale"] for case in slots} != {.25, .75}:
                raise ValueError("Every method must have two predeclared slots on every board")
    if any(case["arm"] not in METHODS or case["prediction_status"] not in statuses for case in cases):
        raise ValueError("Unexpected method or prediction status")


def analyse(cases: list[dict[str, Any]], *, max_depth: int, max_nodes: int,
            seconds: float, on_result: Callable[[dict[str, Any]], None] | None = None
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep every proposal and separately check valid intents under one solver budget."""
    validate_cases(cases)
    results = []
    for case in cases:
        method = case["arm"]
        if method not in METHODS:
            raise ValueError(f"Unexpected method: {method}")
        status = case["prediction_status"]
        valid = status == "valid_intent"
        search = None
        search_error = None
        if valid:
            try:
                search = bounded_witness(case["start_observation"], case["parsed"],
                                         max_depth=max_depth, max_nodes=max_nodes, seconds=seconds)
            except Exception as error:  # Keep all other cases if a local solver or replay fails.
                search_error = f"{type(error).__name__}: {error}"
        reached = search["reached_observation"] if search else None
        unavailable = search_error is not None or status in ("infrastructure_error", "not_attempted")
        result = {**case, "valid_intent": valid, "search": search, "solver_error": search_error,
                  "path_found": None if unavailable else bool(search and search["witness_found"]),
                  "goal_box_change": _goals(reached) - _goals(case["start_observation"])
                  if reached is not None else None}
        results.append(result)
        if on_result is not None:
            on_result(result)
    summary, paired = summarise(results)
    return results, summary, paired


def summarise(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate saved checks without repeating the bounded solver."""
    validate_cases(results)
    layouts = {row["layout_id"] for row in results}
    summary = []
    for method in METHODS:
        rows = [row for row in results if row["arm"] == method]
        missing = sum(row["prediction_status"] in ("infrastructure_error", "not_attempted") for row in rows)
        solver_errors = sum(row["solver_error"] is not None for row in rows)
        by_board: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_board[row["layout_id"]].append(row)
        valid_by_board = [sum(row["valid_intent"] for row in board) / len(board) for board in by_board.values()]
        found_by_board = [sum(row["path_found"] is True for row in board) / len(board) for board in by_board.values()]
        searches = [row["search"] for row in rows if row["search"] is not None]
        intents = [json.dumps(row["parsed"], sort_keys=True, separators=(",", ":"))
                   for row in rows if row["valid_intent"]]
        parsed_intents = [json.dumps(row["parsed"], sort_keys=True, separators=(",", ":"))
                          for row in rows if row.get("json_valid") is True]
        types: Counter[str] = Counter()
        sizes: Counter[int] = Counter()
        whole_goal = 0
        for row in rows:
            if not row["valid_intent"]:
                continue
            effects = row["parsed"]["effects"]
            types.update(item["predicate"] for item in effects)
            sizes[len(effects)] += 1
            box_count = sum(line.count("$") + line.count("*") for line in row["start_observation"])
            whole_goal += any(item["predicate"] == "boxes_on_goals_at_least" and item["count"] >= box_count
                              for item in effects)
        valid_n = sum(row["valid_intent"] for row in rows)
        found_n = sum(row["path_found"] is True for row in rows)
        summary.append({"method": method, "boards": len(by_board), "requested_outputs": len(rows),
                        "strict_valid_json": sum(row.get("strict_json_valid") is True for row in rows),
                        "valid_json": sum(row.get("json_valid") is True for row in rows),
                        "fences_removed": sum(row.get("normalization_applied") is True for row in rows),
                        "valid_intents": valid_n, "paths_found": found_n,
                        "no_path_found_among_valid": valid_n - found_n - solver_errors,
                        "infrastructure_missing": missing,
                        "solver_infrastructure_errors": solver_errors,
                        "valid_fraction": valid_n / len(rows), "path_fraction": found_n / len(rows),
                        "valid_fraction_of_completed": valid_n / (len(rows) - missing) if missing < len(rows) else None,
                        "path_fraction_of_completed": (found_n / (len(rows) - missing - solver_errors)
                                                       if missing + solver_errors < len(rows) else None),
                        "valid_board_bootstrap_95": _interval(valid_by_board, 20261115) if not missing else None,
                        "path_board_bootstrap_95": (_interval(found_by_board, 20261116)
                                                    if not missing and not solver_errors else None),
                        "solver_expanded_nodes": sum(item["expanded_nodes"] for item in searches),
                        "solver_primitive_edges_examined": sum(item.get("primitive_edges_examined", 0)
                                                               for item in searches),
                        "solver_seconds": sum(item["elapsed_seconds"] for item in searches),
                        "solver_verification_seconds": sum(item.get("verification_seconds") or 0
                                                           for item in searches),
                        "solver_statuses": dict(Counter(item["status"] for item in searches)),
                        "predicate_counts": dict(types), "conjunction_sizes": dict(sizes),
                        "whole_goal_requests": whole_goal,
                        "most_common_valid_intent_count": Counter(intents).most_common(1)[0][1] if intents else 0,
                        "most_common_parseable_intent_count": (Counter(parsed_intents).most_common(1)[0][1]
                                                               if parsed_intents else 0),
                        "local_rule_work": _local_rule_work(rows) if method == GeneratorMethod.LOCAL_RULE else None,
                        "paths_increasing_boxes_on_goals": sum(row["goal_box_change"] is not None
                                                               and row["goal_box_change"] > 0 for row in rows),
                        "paths_decreasing_boxes_on_goals": sum(row["goal_box_change"] is not None
                                                               and row["goal_box_change"] < 0 for row in rows)})
    paired = {}
    for metric in ("valid_intent", "path_found"):
        differences = []
        for layout in sorted(layouts):
            group = [row for row in results if row["layout_id"] == layout
                     and row["arm"] in (GeneratorMethod.BASE, GeneratorMethod.DEMONSTRATION)]
            if any(row["prediction_status"] in ("infrastructure_error", "not_attempted") for row in group):
                continue
            if metric == "path_found" and any(row["solver_error"] is not None for row in group):
                continue
            base = [row[metric] for row in group if row["arm"] == GeneratorMethod.BASE]
            adapted = [row[metric] for row in group if row["arm"] == GeneratorMethod.DEMONSTRATION]
            differences.append(sum(adapted) / len(adapted) - sum(base) / len(base))
        paired[metric] = {"difference_D_minus_B": sum(differences) / len(differences) if differences else None,
                          "board_bootstrap_95": _interval(differences, 20261117), "boards": len(differences)}
    return summary, paired


def plot(summary: list[dict[str, Any]], results: list[dict[str, Any]], output: Path) -> None:
    """Draw the saved counts; bounded search is labelled separately from Luna."""
    family_names = {"real_12x12x4": "Real Sokoban · 12×12 · 4 boxes",
                    "proposal_gamma2_12x12x10": "Synthetic Sokoban · 12×12 · 10 boxes"}
    families = {row.get("family") for row in results}
    family = next(iter(families)) if len(families) == 1 else None
    title = family_names.get(family, "Sokoban intent checks")
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    figure, axis = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
    for offset, key, count_key, label, colour in ((-.19, "valid_fraction", "valid_intents", "Valid intent", "#527da3"),
                                                  (.19, "path_fraction", "paths_found", "Bounded search found a path",
                                                   "#a94947")):
        for index, row in enumerate(summary):
            value = row[key]
            interval = row["valid_board_bootstrap_95" if key == "valid_fraction" else "path_board_bootstrap_95"]
            errors = [[max(0., value - interval[0])], [max(0., interval[1] - value)]] if interval else None
            axis.bar(index + offset, value, width=.36, color=colour, label=label if index == 0 else None,
                     yerr=errors, capsize=2, error_kw={"linewidth": .7})
            axis.text(index + offset, value / 2 if value > .02 else .07,
                      f"{row[count_key]}/{row['requested_outputs']}", ha="center", va="center",
                      color="white" if value > .02 else "#333333", fontsize=7)
    axis.set_xticks(range(len(summary)), [LABELS[row["method"]] for row in summary])
    axis.set(ylabel="Fraction of requested outputs", ylim=(0, 1.08))
    axis.set_title(f"{title}\n{summary[0]['boards']} boards · two deterministic slots each", pad=31)
    axis.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.005), ncol=2)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(output / f"validity_bounded_search.{suffix}", dpi=180)
    plt.close(figure)


def write(output: Path, results: list[dict[str, Any]], summary: list[dict[str, Any]],
          paired: dict[str, Any], *, max_depth: int, max_nodes: int, seconds: float) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty assessment directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results))
    (output / "summary.json").write_text(json.dumps({"methods": summary, "paired": paired,
        "solver": {"max_depth": max_depth, "max_nodes": max_nodes, "seconds_per_proposal": seconds},
        "slot_labels": "0.25 and 0.75 are pairing labels only; no scale was supplied to B, D or R. "
                       "Both Qwen slots use greedy decoding and can repeat the same output.",
        "interpretation": "Bounded search is a local solver, not Luna. Unresolved is not impossible. "
                          "Fractions over requested outputs are lower bounds if infrastructure outputs are missing; "
                          "paired intervals exclude boards with incomplete B or D outputs."}, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as stream:
        columns = ["method", "boards", "requested_outputs", "strict_valid_json", "valid_json", "fences_removed",
                   "valid_intents", "paths_found",
                   "no_path_found_among_valid", "infrastructure_missing", "solver_infrastructure_errors",
                   "valid_fraction", "path_fraction",
                   "valid_fraction_of_completed", "path_fraction_of_completed",
                   "solver_expanded_nodes", "solver_primitive_edges_examined", "solver_seconds",
                   "solver_verification_seconds", "whole_goal_requests", "most_common_valid_intent_count",
                   "most_common_parseable_intent_count", "local_rule_work"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    plot(summary, results, output)
    first_layouts = list(dict.fromkeys(row["layout_id"] for row in results))[:3]
    lines = ["# Examples from the local bounded search", "",
             "These are solver paths, not trajectories executed by Luna. Limits and raw cases are in summary.json.", ""]
    for layout in first_layouts:
        subset = [row for row in results if row["layout_id"] == layout and row["requested_scale"] == .25]
        if not subset:
            continue
        lines.extend([f"## Board {layout}", "", "```", *subset[0]["start_observation"], "```", ""])
        for row in subset:
            lines.extend([f"### {LABELS[row['arm']]}", "", f"Intent: `{json.dumps(row.get('parsed'))}`", "",
                          f"Prediction status: `{row['prediction_status']}`; path found: `{row['path_found']}`.", ""])
            search = row.get("search")
            if search and search.get("reached_observation"):
                lines.extend([f"Solver actions: `{search['witness_actions']}` (0=up, 1=down, 2=left, 3=right).",
                              "", "```", *search["reached_observation"], "```", ""])
    (output / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--plan", type=Path, nargs="+", help="Frozen proposals to assess")
    inputs.add_argument("--results", type=Path, nargs="+", help="Saved local checks to aggregate without rerunning")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=1.)
    parser.add_argument("--progress", type=Path, help="Append each case here before final aggregation")
    args = parser.parse_args()
    if args.max_depth < 1 or args.max_nodes < 1 or args.seconds <= 0:
        parser.error("Search limits must be positive")
    if args.progress is not None and args.progress.resolve().is_relative_to(args.output.resolve()):
        parser.error("The progress file must be outside the final output directory")
    cases = [case for path in (args.plan or args.results) for case in read_records(path)]
    def save_progress(row: dict[str, Any]) -> None:
        if args.progress is not None:
            args.progress.parent.mkdir(parents=True, exist_ok=True)
            with args.progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    if args.results and args.progress is not None:
        parser.error("Progress is only used when running new checks")
    if args.progress is not None and args.progress.exists():
        raise FileExistsError("Use a fresh progress file")
    if args.results:
        for case in cases:
            search = case.get("search")
            if search is not None and (search.get("depth_limit"), search.get("node_limit"),
                                       search.get("time_limit_seconds")) != (args.max_depth, args.max_nodes, args.seconds):
                raise ValueError("Saved checks use a different bounded-search budget")
        results = cases
        summary, paired = summarise(results)
    else:
        results, summary, paired = analyse(cases, max_depth=args.max_depth, max_nodes=args.max_nodes,
                                           seconds=args.seconds, on_result=save_progress)
    write(args.output, results, summary, paired, max_depth=args.max_depth, max_nodes=args.max_nodes, seconds=args.seconds)
    print(json.dumps({"boards": len({row['layout_id'] for row in results}), "methods": summary}, indent=2))


if __name__ == "__main__":
    main()
