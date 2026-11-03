"""Build component-probe tables and figures from saved evidence, without API calls."""

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.generator_probe.evaluate_artifacts import figures, ledger_costs, solver_result

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "analysis/luna_icl/campaign_100/generator_probe_v2"
Row = dict[str, Any]


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[Row]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def raw_boards(case: Row) -> list[list[str]]:
    """Extract model rows verbatim, including invalid or copied boards."""
    boards = []
    for call in case["generator_calls"]:
        try:
            candidates = json.loads(call.get("response", {}).get("output_text", ""))["subgoals"]
            for item in candidates:
                boards.append(item["rows"] if "rows" in item else
                              [item[f"row_{index:02d}"] for index in range(len(case["source"]))])
        except (KeyError, TypeError, ValueError):
            continue
    return boards


def flatten(case: Row, path: Path, base: Path, sources: dict[str, Row], experiment: str) -> Row:
    identifier = case["board_id"]
    source = sources.get(identifier, sources.get(f"{identifier}:position_00", {}))
    candidates = [item for call in case["classification"] for item in call["candidates"]]
    rollouts = [proposal["rollout"] for proposal in case["proposals"] if "rollout" in proposal]
    boards = raw_boards(case)
    if any(proposal["target"] not in boards for proposal in case["proposals"]):
        raise ValueError(f"Accepted target differs from every raw model board: {path}")
    first_hits = [rollout["first_hit"] for rollout in rollouts if rollout["exact_hit"]]
    metrics, generator = case["metrics"], case["generator_metrics"]
    return {
        "experiment": experiment, "variant": case["variant"], "board_id": source.get("board_id", identifier),
        "source_id": identifier, "trajectory_id": source.get("trajectory_id", ""),
        "source_path": source.get("source_path", ""), "source_index": source.get("source_index", ""),
        "position": source.get("position", ""), "u": case["scale"],
        "demonstration_count": case["config"].get("generator_example_limit") or case["config"]["few_shot"],
        "raw_candidates": len(candidates), "static_valid": sum(item["static_valid"] for item in candidates),
        "invalid": sum(not item["static_valid"] for item in candidates),
        "copies": sum(board == case["source"] for board in boards),
        "accepted": len(case["proposals"]), "hit": int(case["reached_any"]),
        "first_hit": min(first_hits) if first_hits else "", "actions": sum(rollout["actions"] for rollout in rollouts),
        "complete": case.get("complete", case.get("error") is None), "error": json.dumps(case.get("error")),
        "generator_calls": len(case["generator_calls"]), "physical_requests": metrics["requests"],
        "local_cache_hits": metrics["cache_hits"], "physical_input_tokens": metrics["input_tokens"],
        "physical_output_tokens": metrics["output_tokens"], "provider_cached_input_tokens": metrics["cached_input_tokens"],
        "generator_physical_requests": generator["requests"], "generator_local_cache_hits": generator["cache_hits"],
        "generator_physical_input_tokens": generator["input_tokens"],
        "generator_physical_output_tokens": generator["output_tokens"],
        "accepted_exact_raw": all(proposal["target"] in boards for proposal in case["proposals"]),
        "replay_verified": all(rollout["replay_verified"] for rollout in rollouts),
        "raw_file": str(path.relative_to(base)),
    }


def aggregate(rows: list[Row], keys: tuple[str, ...]) -> list[Row]:
    groups: dict[tuple[Any, ...], list[Row]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        hits = [row["first_hit"] for row in group if row["first_hit"] != ""]
        result.append({**dict(zip(keys, key)), "queries": len(group),
                       "layouts": len({row["board_id"] for row in group}),
                       "source_states": len({row["source_id"] for row in group}),
                       **{name: sum(row[name] for row in group) for name in (
                           "hit", "raw_candidates", "static_valid", "invalid", "copies", "accepted", "actions",
                           "physical_requests", "local_cache_hits", "physical_input_tokens", "physical_output_tokens")},
                       "first_hit_mean_conditional_on_hit": mean(hits) if hits else "",
                       "first_hit_min": min(hits) if hits else "", "first_hit_max": max(hits) if hits else ""})
    return result


def scale_pairs(cases: list[tuple[str, Path, Row]]) -> list[Row]:
    groups: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    for experiment, _, case in cases:
        # Only identical source and generator configuration count as a matched scale comparison.
        config = json.dumps(case["config"], sort_keys=True)
        key = experiment, case["variant"], json.dumps(case["source"]), config
        groups[key].append(case)
    rows = []
    for (experiment, variant, _, _), group in groups.items():
        for left, right in itertools.combinations(sorted(group, key=lambda row: row["scale"]), 2):
            if left["scale"] == right["scale"]:
                continue
            a, b = raw_boards(left), raw_boards(right)
            rows.append({"experiment": experiment, "variant": variant, "source_id": left["board_id"],
                         "lower_u": left["scale"], "higher_u": right["scale"],
                         "raw_single_board_pair": len(a) == len(b) == 1,
                         "identical_raw_endpoint": len(a) == len(b) == 1 and a == b,
                         "lower_hit": left["reached_any"], "higher_hit": right["reached_any"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    base = args.directory.resolve()
    output = base / "report"
    output.mkdir(exist_ok=True)
    sources = {row["id"]: row for row in read(base / "validation_sources_30.json")["instances"]}
    cases = []
    for experiment, relative in (("initial12", "openai"), ("expanded_comparison", "expanded_comparison/openai"),
                                 ("static_retry", "static_retry/openai"), ("visible_plan", "visible_plan/openai")):
        for path in sorted((base / relative).glob("*.json")):
            value = read(path)
            if isinstance(value, dict) and all(key in value for key in ("classification", "generator_calls", "metrics")):
                cases.append((experiment, path, value))
    rows = [flatten(case, path, base, sources, experiment) for experiment, path, case in cases]
    totals = aggregate(rows, ("experiment", "variant"))
    by_layout = aggregate(rows, ("experiment", "variant", "board_id"))
    by_scale = aggregate(rows, ("experiment", "variant", "u"))
    pairs = scale_pairs(cases)
    checkpoint = read(base / "checkpoint_balanced/generator.json")
    calibration = read(base / "expanded_calibration/openai/calibration.json")
    examples = [{"order": index, **{key: item[key] for key in (
        "context", "k", "d", "u", "hits", "trials", "p_hat", "a_c", "weight")}}
        for index, item in enumerate(checkpoint["examples"], 1)]
    for name, table in (("queries", rows), ("by_variant", totals), ("by_layout", by_layout),
                        ("by_scale", by_scale), ("matched_scale_pairs", pairs), ("checkpoint_examples", examples)):
        write_csv(output / f"{name}.csv", table)
    costs, phase_costs = ledger_costs()
    solver = solver_result(base)
    write_csv(output / "phase_costs.csv", phase_costs)
    write_csv(output / "solver_validation.csv", [solver])
    summary = {"observed_queries": len(rows), "groups": totals, "ledger": costs, "phase_costs": phase_costs,
               "full_solver_validation": solver,
               "calibration": {"targets": len(calibration["targets"]), "retained": len(examples),
                               "hits": sum(item["hits"] for item in calibration["targets"]),
                               "trials": sum(item["trials"] for item in calibration["targets"])},
               "inputs": {str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for _, path, _ in cases}}
    for relative in ("validation_sources_30.json", "checkpoint_balanced/generator.json",
                     "expanded_calibration/openai/calibration.json", "initial12_raw_output_diagnosis.json"):
        summary["inputs"][relative] = hashlib.sha256((base / relative).read_bytes()).hexdigest()
    for relative in ("solver_validation/manifest.json", "solver_validation/case.json", "solver_validation/trace.jsonl"):
        if (base / relative).exists():
            summary["inputs"][relative] = hashlib.sha256((base / relative).read_bytes()).hexdigest()
    figures(output, by_scale, totals, examples)
    lines = ["# Offline generator component report", "",
             "The paired comparison reaches a distinct generated endpoint on 23/30 source states with four "
             "demonstrations and 23/30 with 26 demonstrations. It does not establish an advantage for the larger "
             "context or demonstrate learned policy-relative scale control.", "",
             "## Evidence and denominators", "",
             "The 30 real source states come from recorded rl-data validation trajectories: three positions on "
             "each of ten layouts. Both variants use the same source and u. These are 30 paired source observations, "
             "not 30 independent layouts. Every scheduled query remains in the denominator, including invalid outputs, "
             "source copies and empty replies. The frozen controller receives an accepted complete board and has L=8 "
             "primitive actions. This measures component endpoint reachability, not full puzzle completion.", "",
             "| Experiment | Variant | Endpoint hits / queries | Layouts | Invalid raw | Raw copies | Accepted | Actions |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in totals:
        lines.append(f"| {row['experiment']} | {row['variant']} | {row['hit']}/{row['queries']} | {row['layouts']} | "
                     f"{row['invalid']} | {row['copies']} | {row['accepted']} | {row['actions']} |")
    lines += ["", "Invalid/copy columns count raw candidates across all attempts; retries can therefore exceed query counts. "
              "The initial 12 queries use two layouts at two u values per variant. Retry and visible-plan probes, when "
              "present, are separate post-hoc experiments and are not pooled into the paired comparison.", "",
              "## Raw-output and controller diagnosis", "",
              "The initial copies are exact source boards emitted by the model, not decoder replacements. All eight "
              "initial accepted targets equal the raw model rows. This report checks accepted-to-raw equality for every "
              "included case. The layout schema preserves row geometry but does not guarantee a single player, box count, "
              "or legal dynamics. The initial diagnosis separates two baseline targets unreachable within eight moves "
              "from the two identical planned targets reachable in eight moves that CLLP misses. The bounded offline "
              "search was diagnostic only: it supplied no query hints, generated candidates, repairs, or controller actions. "
              "See [raw diagnosis](../initial12_raw_output_diagnosis.json).", "",
              "## Checkpoint and scale interpretation", "",
              f"The saved checkpoint contains {len(examples)} genuinely calibrated demonstrations from ten training "
              f"contexts, retained from {len(calibration['targets'])} targets after "
              f"{summary['calibration']['hits']}/{summary['calibration']['trials']} exact-hit rollouts (R=2, L=8). "
              "Its first four ordered examples cover d=1,2,4. The u label is the within-context hit-weighted percentile "
              "of measured successful first-hit time; it is neither a move count nor a success probability. The reference "
              "population has three distinct offline targets per context, not every possible future target. Repeated "
              "seeded action draws can share cached policy distributions and are not independent provider samples.", "",
              "The initial low-reasoning hits require only 2,1,1 moves. The same f955 source gets the same low-reasoning "
              "endpoint at u=.5 and .85; the planned variant also repeats its endpoint there. Thus 3/4 initial hits do "
              "not establish scale adherence. The expanded comparison assigns only one u to each source. Its first-hit "
              "means by u are descriptive and conditional on success; different source difficulties and failures prevent "
              "interpreting them as a learned scale effect. matched_scale_pairs.csv compares only identical sources "
              "with identical saved generator configurations at different u.", "",
              "| Variant | Requested u | Endpoint hits / sources | Mean first-hit actions among hits | Range among hits |",
              "|---|---:|---:|---:|---:|"]
    for row in by_scale:
        if row["experiment"] == "expanded_comparison":
            lines.append(f"| {row['variant']} | {row['u']} | {row['hit']}/{row['queries']} | "
                         f"{row['first_hit_mean_conditional_on_hit']:.2f} | {row['first_hit_min']}–{row['first_hit_max']} |")
    lines += ["",
              "Inspect [ordered demonstration table](checkpoint_examples.csv), "
              "[checkpoint](../checkpoint_balanced/generator.json), and "
              "[exact comparison prompts](../expanded_comparison/openai/generator_inputs.json).", "",
              "## Cost and reproducibility", "",
              "The ledger snapshot uses tags beginning generator_probe_v2. Settled charges below are actual accounted "
              "new provider charges. Unsettled reservations are reported separately and are not called spending. "
              "The interrupted earlier call retains an unknown $0.5791896 reservation when still present in the ledger. "
              "Local cache replays incur no new provider charge; provider cached input tokens remain a separate metric.", "",
              "| Ledger status | Attempts | Settled charge USD | Unsettled reservation USD |",
              "|---|---:|---:|---:|"]
    for status, cost in costs.items():
        lines.append(f"| {status} | {cost['attempts']} | {cost['settled_charge_usd']:.9f} | {cost['unsettled_reservation_usd']:.9f} |")
    lines += ["", "| Phase | Status | New physical attempts | Settled charge USD | Unsettled reservation USD |",
              "|---|---|---:|---:|---:|"]
    for cost in phase_costs:
        lines.append(f"| {cost['phase']} | {cost['status']} | {cost['physical_attempts']} | "
                     f"{cost['settled_charge_usd']:.9f} | {cost['unsettled_reservation_usd']:.9f} |")
    retry_path = base / "static_retry/openai/summary.json"
    if retry_path.exists():
        retry = read(retry_path)
        summary["static_retry"] = retry
        lines += ["", "## Post-hoc static-feedback retry", "",
                  f"All {retry['observed']} selected failed 26-example queries reuse their cached original first reply "
                  f"and permit the existing production static-validation feedback with at most two generator attempts. "
                  f"The retry recovers {retry['recovered']}/{retry['expected']}, giving a retrospective replay-first cohort "
                  f"of {retry['revised_replay_first_hits']}/{retry['cohort_size']} endpoint hits. This is not a fresh "
                  "30-source retest or independent confirmation; the original single-attempt result remains 23/30. "
                  "The feedback checks static output validity and copying, without providing a future trajectory or "
                  "simulator search result. The visible-plan helper is prepared but unexecuted; no visible-plan result "
                  "is claimed. The runnable retry profile is configs/icl/experiments/luna_checkpoint26_retry2.json."]
    lines += ["", "## Exploratory full-board solver validation", ""]
    if solver["status"] == "recorded":
        lines += [f"The saved full solver case on `{solver['board_id']}` uses {solver['demonstration_count']} "
                  f"demonstrations, at most {solver['generator_max_attempts']} generator attempts, seed {solver['seed']} "
                  f"and B={solver['budget_actions']}. It {'solves' if solver['solved'] else 'does not solve'} the board "
                  f"and finishes with `{solver['finish_reason']}` after {solver['attempted_actions']} primitive actions "
                  f"and {solver['expansions']} queue expansions. Saved action replay is verified. "
                  f"There are {solver['accepted_subgoals']} accepted and {solver['rejected_subgoals']} rejected "
                  "subgoal attempts in the search result; these are not the component-query counts above.", "",
                  f"Saved-log diagnosis: {solver['no_op_actions']}/{solver['attempted_actions']} actions leave the state "
                  f"unchanged. The tree contains {solver['endpoint_status_new']} new, "
                  f"{solver['endpoint_status_rejected']} rejected and {solver['endpoint_status_duplicate']} duplicate "
                  f"endpoint records, plus {solver['endpoint_status_root']} root. "
                  f"Of {solver['saved_endpoint_samples']} saved endpoint samples including the root, "
                  f"{solver['saved_endpoints_zero_boxes_on_goals']} have zero boxes on goals, counted directly from "
                  "the ASCII boards. This describes saved endpoints only, not every intermediate action state. "
                  "Reaching generated endpoints therefore did not demonstrate box-on-goal progress in these samples.", "",
                  "This is one exploratory scale-validation board, previously available for scale selection. It is "
                  "neither a new pilot cohort nor a held-out generalization estimate. It stays outside the 30-source "
                  "component comparison and the seven-case retrospective retry. A component endpoint hit does not "
                  "establish full-board completion. See [saved case](../solver_validation/case.json), "
                  "[search tree](../solver_validation/search_tree.html), "
                  "[expansions](../solver_validation/expansions.html), and [solver table](solver_validation.csv)."]
        if solver["error"]:
            lines += ["", f"Recorded solver error: `{solver['error']}`."]
    elif solver["status"] == "awaiting_saved_case":
        lines += ["The full-solver run has been prepared and is awaiting its saved final case. No completion "
                  "outcome is reported from its partial trace. Rerun this offline exporter after case.json is saved."]
    else:
        lines += ["No full-solver validation case is available in this evidence directory."]
    lines += ["", "queries.csv includes raw evidence paths, source provenance, tokens and physical/cache counters. "
              "by_layout.csv retains the ten layout clusters. summary.json records raw-file hashes. Figures are available "
              "as self-contained HTML, SVG and PDF: endpoint_outcomes, scale_first_hit and checkpoint_calibration.", "",
              "The historical trained Torch reference at analysis/trained_reference/random100_k841 reports 91/100 "
              "full-board completions on a separate sample with no overlapping layouts. It is not a paired baseline "
              "for these component tests and its completion percentage must not be compared with endpoint-hit percentages.", "",
              "Rebuild offline: `uv run python -m scripts.generator_probe.evaluate`. No provider API, core implementation "
              "changes, simulator-derived query hints, or new experiment execution are performed by this report."]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "queries": len(rows), "ledger": costs}, indent=2))


if __name__ == "__main__":
    main()
