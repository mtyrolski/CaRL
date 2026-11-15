#!/usr/bin/env python3
"""Post-hoc state-association diagnostic for saved generator proposals.

Each within-family permutation moves whole raw responses between boards. The
same mapping is used for every method. The first mapping also receives the
same bounded solver check as the original proposals; this is not a controller
measurement or evidence of useful planning.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.intent_assessment import bounded_witness
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records
from carl.icl.policy_relative_lora import PredictionStatus
from carl.icl.policy_relative_lora import assess_output
from scripts.assess_luna_policy_local import LABELS
from scripts.assess_luna_policy_local import METHODS

Record = dict[str, Any]
MISSING = {PredictionStatus.INFRASTRUCTURE_ERROR, PredictionStatus.NOT_ATTEMPTED}
SEED = 20261115


def derangements(family: str, size: int, seed: int, count: int) -> list[list[int]]:
    """Draw uniform derangements by rejection, with a separate family RNG."""
    if size < 2 or count < 1:
        raise ValueError("At least two boards and one permutation are required")
    material = f"{seed}\0{family}".encode("utf-8")
    family_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random(family_seed)
    result = []
    while len(result) < count:
        indices = list(range(size))
        rng.shuffle(indices)
        if all(target != source for target, source in enumerate(indices)):
            result.append(indices)
    return result


def select_slot(records: list[Record], slot: float, contexts: list[Record] | None = None
                ) -> dict[str, dict[str, dict[str, Record]]]:
    """Keep missing responses as infrastructure, using frozen contexts if supplied."""
    grouped: dict[str, dict[str, dict[str, Record]]] = defaultdict(lambda: defaultdict(dict))
    planned = {}
    for context in contexts or []:
        key = context["family"], context["layout_id"]
        if key in planned:
            raise ValueError("Duplicate layout in frozen contexts")
        planned[key] = context
        grouped[key[0]][key[1]]
    for row in records:
        if row.get("requested_scale") != slot:
            continue
        family, layout, arm = row["family"], row["layout_id"], row["arm"]
        if arm not in METHODS or arm in grouped[family][layout]:
            raise ValueError("Unexpected or duplicated selected-slot method")
        if contexts is not None and (family, layout) not in planned:
            raise ValueError("Result board is absent from the frozen contexts")
        if contexts is not None and row["start_observation"] != planned[family, layout]["start_observation"]:
            raise ValueError("Result board differs from the frozen context")
        input_status = row.get("prediction_status")
        if input_status is not None and input_status not in set(PredictionStatus):
            raise ValueError("Unexpected prediction status")
        raw = row.get("raw_output")
        reason = None
        if input_status is None:
            reason = "missing_prediction_status"
        elif input_status not in MISSING and not isinstance(raw, str):
            reason = "missing_or_nonstring_raw_output"
        normalized = dict(row, input_prediction_status=input_status, infrastructure_reason=reason)
        if reason is not None:
            normalized["prediction_status"] = PredictionStatus.INFRASTRUCTURE_ERROR
        grouped[family][layout][arm] = normalized
    if not grouped:
        raise ValueError("No selected-slot responses")
    for family, layouts in grouped.items():
        if len(layouts) < 2:
            raise ValueError(f"At least two boards are required for {family}")
        for layout, methods in layouts.items():
            exemplar = planned.get((family, layout)) or next(iter(methods.values()))
            if canonical_layout_id(from_rows(exemplar["start_observation"])) != layout:
                raise ValueError("Layout ID does not match the canonical D4 board layout")
            for arm in set(METHODS) - set(methods):
                methods[arm] = {"record_id": f"missing:{family}:{layout}:{arm}:{slot}", "family": family,
                                "layout_id": layout, "arm": arm, "requested_scale": slot,
                                "start_observation": exemplar["start_observation"], "raw_output": None,
                                "input_prediction_status": None, "infrastructure_reason": "missing_result_row",
                                "prediction_status": PredictionStatus.NOT_ATTEMPTED,
                                "search": None, "solver_error": None, "path_found": None}
            if len({tuple(row["start_observation"]) for row in methods.values()}) != 1:
                raise ValueError("All methods must have the same board for a layout")
    return grouped


def _quantile(values: list[float], probability: float) -> float:
    """Empirical quantile without interpolation."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _ratio(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def _known_total(values: list[Any]) -> Record:
    known = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {"requested_outputs": len(values), "known_outputs": len(known),
            "known_sum": sum(known) if known else None, "total": sum(known) if len(known) == len(values) else None}


def _generator_work(rows: list[Record], arm: str) -> Record:
    result = {"model_generation_seconds": _known_total([row.get("generation_seconds") for row in rows]),
              "new_calls_for_permutation": 0}
    if arm == "R":
        keys = ("walk_graphs", "walk_nodes_expanded", "walk_edges_examined", "push_checks", "legal_pushes",
                "goal_distance_checks", "elapsed_seconds")
        result["local_rule_work"] = {
            key: _known_total([work.get(key) if isinstance(work, dict) else None
                               for row in rows for work in [row.get("local_rule_work")]]) for key in keys}
    return result


def _search_unavailable(row: Record) -> bool:
    if row["prediction_status"] != PredictionStatus.VALID:
        return False
    search, found = row.get("search"), row.get("path_found")
    return (row.get("solver_error") is not None or not isinstance(search, dict) or not isinstance(found, bool)
            or search.get("witness_found") is not found)


def _search_budget_mismatch(row: Record, max_depth: int, max_nodes: int, seconds: float) -> bool:
    if row["prediction_status"] != PredictionStatus.VALID or _search_unavailable(row):
        return False
    search = row["search"]
    return (search.get("depth_limit"), search.get("node_limit"), search.get("time_limit_seconds")) != (
        max_depth, max_nodes, seconds)


def _search_summary(rows: list[Record], max_depth: int, max_nodes: int, seconds: float) -> Record:
    searches = [row["search"] for row in rows if row["prediction_status"] == PredictionStatus.VALID
                and not _search_unavailable(row) and not _search_budget_mismatch(row, max_depth, max_nodes, seconds)]
    errors = sum(_search_unavailable(row) and row.get("solver_error") is not None for row in rows)
    missing_results = sum(_search_unavailable(row) and row.get("solver_error") is None for row in rows)
    mismatches = sum(_search_budget_mismatch(row, max_depth, max_nodes, seconds) for row in rows)
    missing = sum(row["prediction_status"] in MISSING for row in rows)
    found = sum(row["prediction_status"] == PredictionStatus.VALID and not _search_unavailable(row)
                and not _search_budget_mismatch(row, max_depth, max_nodes, seconds)
                and row.get("path_found") is True for row in rows)
    return {
        "requested": len(rows), "generator_infrastructure_missing": missing,
        "solver_infrastructure_errors": errors, "solver_results_missing": missing_results,
        "solver_budget_mismatches": mismatches, "paths_found": found,
        "fraction_per_requested_lower_bound_if_missing": _ratio(found, len(rows)),
        "fraction_per_completed": _ratio(found, len(rows) - missing - errors - missing_results - mismatches),
        "solver_calls_completed": len(searches),
        "solver_statuses": dict(Counter(search["status"] for search in searches)),
        "solver_expanded_nodes_known": sum(search["expanded_nodes"] for search in searches),
        "solver_primitive_edges_known": sum(search["primitive_edges_examined"] for search in searches),
        "solver_seconds_known": sum(search["elapsed_seconds"] for search in searches),
        "replay_seconds_known": sum(search.get("verification_seconds") or 0 for search in searches),
        "known_costs_exclude_unavailable_or_different_budget": errors + missing_results + mismatches > 0,
    }


def analyse(records: list[Record], *, contexts: list[Record] | None = None, slot: float = .25,
            seed: int = SEED, permutations: int = 1000,
            max_depth: int = 24, max_nodes: int = 10000, seconds: float = 1.,
            on_plan: Callable[[Record], None] | None = None,
            on_validity: Callable[[list[Record], list[Record], float], None] | None = None,
            on_case: Callable[[Record], None] | None = None) -> tuple[list[Record], list[Record], Record]:
    """Return first-mapping solver cases, all validity draws, and summaries."""
    if max_depth < 1 or max_nodes < 1 or seconds <= 0:
        raise ValueError("Solver limits must be positive")
    groups = select_slot(records, slot, contexts)
    family_layouts = {family: sorted(layouts) for family, layouts in sorted(groups.items())}
    assignments = {family: derangements(family, len(layouts), seed, permutations)
                   for family, layouts in family_layouts.items()}
    plan = {"seed": seed, "slot": slot, "permutations": permutations,
            "mapping_convention": "for each sorted target index, read the source index in the mapping",
            "family_seed": "int.from_bytes(sha256(f'{seed}\\0{family}').digest()[:8], 'big')",
            "algorithm": "random.Random(family_seed).shuffle with rejection of fixed points",
            "families": {family: {"sorted_layout_ids": family_layouts[family], "source_indices": mappings}
                         for family, mappings in assignments.items()}}
    if on_plan is not None:
        on_plan(plan)

    @lru_cache(maxsize=None)
    def check(raw: str, board: tuple[str, ...]) -> Record:
        return assess_output(raw, list(board), normalize=True)

    # Evaluate all raw-output/board combinations once; permutations are then
    # integer lookups. Missing outputs move as missing, never as invalid JSON.
    started = time.monotonic()
    tables: dict[tuple[str, str], list[list[Record | None]]] = {}
    distributions = []
    diagnostics = {}
    for family, layouts in family_layouts.items():
        boards = [tuple(groups[family][layout][METHODS[0]]["start_observation"]) for layout in layouts]
        for arm in METHODS:
            original = [groups[family][layout][arm] for layout in layouts]
            for index, row in enumerate(original):
                if row["prediction_status"] not in MISSING:
                    result = check(row["raw_output"], boards[index])
                    if result["status"] != row["prediction_status"]:
                        raise ValueError("Saved prediction status does not match the frozen parser")
            table = [[None if source["prediction_status"] in MISSING else check(source["raw_output"], board)
                      for source in original] for board in boards]
            tables[family, arm] = table
            missing = sum(row["prediction_status"] in MISSING for row in original)
            observed = sum(table[i][i] is not None and table[i][i]["status"] == PredictionStatus.VALID
                           for i in range(len(layouts)))
            values = []
            for index, mapping in enumerate(assignments[family]):
                count = sum(table[target][source] is not None
                            and table[target][source]["status"] == PredictionStatus.VALID
                            for target, source in enumerate(mapping))
                values.append(count / len(layouts))
                distributions.append({"family": family, "arm": arm, "permutation_index": index,
                                      "requested": len(layouts), "completed": len(layouts) - missing,
                                      "valid_intents": count, "valid_fraction_requested": count / len(layouts),
                                      "valid_fraction_completed": _ratio(count, len(layouts) - missing)})
            diagnostics[family, arm] = {
                "requested": len(layouts), "completed": len(layouts) - missing,
                "generator_infrastructure_missing": missing, "observed_valid": observed,
                "generator_infrastructure_reasons": dict(Counter(
                    row.get("infrastructure_reason") or row["prediction_status"]
                    for row in original if row["prediction_status"] in MISSING)),
                "observed_valid_fraction_requested": observed / len(layouts),
                "observed_valid_fraction_completed": _ratio(observed, len(layouts) - missing),
                "permuted_valid_fraction_requested_mean": sum(values) / len(values),
                "permuted_valid_fraction_requested_quantiles": {
                    "q025": _quantile(values, .025), "median": _quantile(values, .5),
                    "q975": _quantile(values, .975)},
                "permutation_fraction_at_least_observed": (sum(v >= observed / len(layouts) for v in values)
                                                         / len(values) if not missing else None),
                "quantile_interpretation": "Quantiles of permuted assignments, not confidence intervals.",
            }
    validity_seconds = time.monotonic() - started
    if on_validity is not None:
        partial = [{"family": family, "arm": arm, **values} for (family, arm), values in diagnostics.items()]
        on_validity(distributions, partial, validity_seconds)

    cases = []
    summary = []
    for family, layouts in family_layouts.items():
        mapping = assignments[family][0]
        for arm in METHODS:
            original = [groups[family][layout][arm] for layout in layouts]
            relocated = [original[index] for index in mapping]
            def raw_key(row: Record) -> str:
                return json.dumps([row["prediction_status"], row.get("raw_output")], sort_keys=True)

            if Counter(raw_key(row) for row in original) != Counter(raw_key(row) for row in relocated):
                raise AssertionError("Permutation changed the raw response multiset")
            permuted = []
            for target_index, source_index in enumerate(mapping):
                target, source = original[target_index], original[source_index]
                checked = tables[family, arm][target_index][source_index]
                prediction_status = source["prediction_status"] if checked is None else checked["status"]
                search, error, wall_seconds = None, None, None
                if prediction_status == PredictionStatus.VALID:
                    clock = time.monotonic()
                    try:
                        search = bounded_witness(target["start_observation"], checked["parsed"],
                                                 max_depth=max_depth, max_nodes=max_nodes, seconds=seconds)
                    except Exception as exc:  # Preserve all other proposals after a solver/replay failure.
                        error = f"{type(exc).__name__}: {exc}"
                    wall_seconds = time.monotonic() - clock
                unavailable = prediction_status in MISSING or error is not None
                row = {"family": family, "arm": arm, "target_layout_id": layouts[target_index],
                       "source_layout_id": layouts[source_index], "source_case_id": source["record_id"],
                       "target_case_id": target["record_id"], "raw_output": source.get("raw_output"),
                       "source_prediction_status": source["input_prediction_status"],
                       "source_infrastructure_reason": source["infrastructure_reason"],
                       "prediction_status": prediction_status, "parsed": checked["parsed"] if checked else None,
                       "validation_errors": checked["validation_errors"] if checked else None,
                       "path_found": None if unavailable else bool(search and search["witness_found"]),
                       "solver_error": error, "search": search, "solver_call_wall_seconds": wall_seconds}
                permuted.append(row)
                cases.append(row)
                if on_case is not None:
                    on_case(row)
            original_search = _search_summary(original, max_depth, max_nodes, seconds)
            permuted_search = _search_summary(permuted, max_depth, max_nodes, seconds)
            permuted_search["solver_call_wall_seconds_known"] = sum(
                row["solver_call_wall_seconds"] or 0 for row in permuted)
            summary.append({"family": family, "arm": arm, **diagnostics[family, arm],
                            "original_solver": original_search, "first_permutation_solver": permuted_search,
                            "source_generator_work": _generator_work(original, arm),
                            "generator_cost_note": "Responses are reassigned without rerunning the generator; "
                                                   "unknown source costs remain null."})
    return cases, distributions, {"methods": summary, "validity_and_permutation_seconds": validity_seconds,
                                  "plan": plan}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[Record]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_distributions(output: Path, distribution: list[Record], methods: list[Record]) -> list[str]:
    """Plot exact permutation frequencies; marker areas reflect their counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    saved = []
    families = sorted({row["family"] for row in methods})
    slugs = [re.sub(r"[^a-zA-Z0-9_.-]+", "_", family) for family in families]
    if len(set(slugs)) != len(slugs):
        raise ValueError("Family names would produce duplicate figure filenames")
    for family, slug in zip(families, slugs, strict=True):
        summary = [row for row in methods if row["family"] == family]
        draws = [row for row in distribution if row["family"] == family]
        frequencies = {row["arm"]: Counter(draw["valid_intents"] for draw in draws if draw["arm"] == row["arm"])
                       for row in summary}
        largest = max(count for counter in frequencies.values() for count in counter.values())
        figure, axis = plt.subplots(figsize=(6.1, 4.2), layout="constrained")
        for position, row in enumerate(summary):
            counts = frequencies[row["arm"]]
            for valid, frequency in sorted(counts.items()):
                axis.scatter(position, valid / row["requested"], s=12 + 260 * frequency / largest,
                             color="#6b91a7", alpha=.68, linewidths=0, zorder=2)
            axis.scatter(position, row["observed_valid_fraction_requested"], marker="D", s=48,
                         facecolor="#9b3735", edgecolor="white", linewidth=.7, zorder=3)
        axis.set_xticks(range(len(summary)), [LABELS.get(row["arm"], str(row["arm"])) for row in summary])
        axis.set_xlim(-.3, len(summary) - .7)
        axis.set_ylim(-.03, 1.03)
        axis.set_ylabel("Valid intents / requested outputs")
        names = {"real_12x12x4": "Real Sokoban · 12×12 · 4 boxes",
                 "proposal_gamma2_12x12x10": "Synthetic Sokoban · 12×12 · 10 boxes"}
        count = sum(row["arm"] == summary[0]["arm"] for row in draws)
        figure.suptitle(f"{names.get(family, family)}\n{summary[0]['requested']} boards · {count} permutations",
                        fontsize=10)
        axis.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#6b91a7",
                                    markeredgecolor="none", markersize=8,
                                    label="Permutation distribution (size = frequency)"),
                             Line2D([], [], marker="D", color="none", markerfacecolor="#9b3735",
                                    markeredgecolor="white", markersize=7, label="Observed pairing")],
                    frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=2)
        if any(row["generator_infrastructure_missing"] for row in summary):
            axis.text(.5, -.18, "Infrastructure missing: fractions are lower bounds.",
                      transform=axis.transAxes, ha="center", fontsize=8)
        for suffix in ("pdf", "svg", "png"):
            path = output / f"permutation_distribution_{slug}.{suffix}"
            figure.savefig(path, dpi=180)
            saved.append(path.name)
        plt.close(figure)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--contexts", type=Path, nargs="+", help="Frozen board lists, including fully missing boards")
    parser.add_argument("--allow-missing-results", action="store_true",
                        help="Treat absent result files as infrastructure; requires frozen contexts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slot", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=1.)
    args = parser.parse_args()
    if args.allow_missing_results and args.contexts is None:
        parser.error("--allow-missing-results requires --contexts")
    missing_files = [path for path in args.results if not path.is_file()]
    if missing_files and not args.allow_missing_results:
        raise FileNotFoundError(f"Missing assessment files: {missing_files}")
    records = [record for path in args.results if path.is_file() for record in read_records(path)]
    contexts = [row for path in args.contexts for row in read_records(path)] if args.contexts is not None else None
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"classification": "post-hoc exploratory diagnostic, not a planning or controller test",
                "source_files": [{"path": str(path.resolve()), "sha256": file_sha256(path) if path.is_file() else None,
                                  "available": path.is_file()} for path in args.results],
                "context_files": ([{"path": str(path.resolve()), "sha256": file_sha256(path)}
                                   for path in args.contexts] if args.contexts is not None else None),
                "board_coverage": "frozen contexts" if contexts is not None else
                                  "inferred from results; completely absent boards cannot be counted",
                "slot": args.slot, "seed": args.seed, "permutations": args.permutations,
                "solver": {"max_depth": args.max_depth, "max_nodes": args.max_nodes, "seconds": args.seconds},
                "parser": "assess_output(normalize=True), the same as the source assessment",
                "missing": "Keep infrastructure failures in requested denominators; also report completed denominators.",
                "interpretation": "Null-distribution quantiles are not confidence intervals. The solver is not Luna."}
    atomic_json(args.output / "protocol.json", protocol)
    atomic_json(args.output / "status.json", {"state": "running", "completed_cases": 0})
    progress = args.output / "paired_cases.jsonl"
    completed = 0

    def save_case(row: Record) -> None:
        nonlocal completed
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        completed += 1
        atomic_json(args.output / "status.json", {"state": "running", "completed_cases": completed})

    def save_validity(distribution: list[Record], partial: list[Record], seconds: float) -> None:
        atomic_csv(args.output / "permutation_distribution.csv", distribution)
        atomic_json(args.output / "preliminary_summary.json",
                    {"methods": partial, "validity_and_permutation_seconds": seconds,
                     "note": "Permutation validity complete; bounded solver may still be running."})
        atomic_json(args.output / "status.json", {"state": "validity_complete", "completed_cases": 0})

    try:
        _, distributions, result = analyse(records, contexts=contexts, slot=args.slot,
                                           seed=args.seed, permutations=args.permutations,
                                           max_depth=args.max_depth, max_nodes=args.max_nodes, seconds=args.seconds,
                                           on_plan=lambda plan: atomic_json(args.output / "permutations.json", plan),
                                           on_validity=save_validity,
                                           on_case=save_case)
        result.pop("plan")
        atomic_json(args.output / "summary.json", result)
        result["figures"] = plot_distributions(args.output, distributions, result["methods"])
        atomic_json(args.output / "summary.json", result)
        atomic_json(args.output / "status.json", {"state": "complete", "completed_cases": completed})
    except Exception as exc:
        atomic_json(args.output / "status.json", {"state": "failed", "completed_cases": completed,
                                                   "error": f"{type(exc).__name__}: {exc}"})
        raise
    print(json.dumps({"output": str(args.output.resolve()), "cases": completed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
