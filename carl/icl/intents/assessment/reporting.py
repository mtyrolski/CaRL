"""Summaries and examples for board-level intent evaluation."""

import json
import random
from collections import Counter
from pathlib import Path

from carl.icl.intent_data import predicate_value
from carl.icl.intents.assessment.io import AssessmentStatus
from carl.icl.intents.assessment.io import Method
from carl.icl.intents.assessment.io import Record
from carl.icl.intents.assessment.io import atomic_write
from carl.icl.intents.assessment.io import board_means
from carl.icl.intents.assessment.io import bootstrap


def has_witness(row: Record) -> bool:
    return bool(row.get("solver") and row["solver"]["witness_found"])


def paired_comparison(
    results: list[Record], reference: Method, candidate: Method
) -> Record:
    """Compare the same contexts and bootstrap their independent boards."""
    by_method = {
        method: {
            row["record_id"]: row
            for row in results
            if row["method"] == method and row["status"] == AssessmentStatus.EVALUATED
        }
        for method in (reference, candidate)
    }
    paired = []
    for record_id in sorted(by_method[reference].keys() & by_method[candidate].keys()):
        before, after = by_method[reference][record_id], by_method[candidate][record_id]
        if before["layout_id"] != after["layout_id"]:
            raise ValueError(f"Paired context has inconsistent layouts: {record_id}")
        paired.append(
            {
                "layout_id": before["layout_id"],
                "valid_delta": int(after["valid_nontrivial"])
                - int(before["valid_nontrivial"]),
                "witness_delta": int(has_witness(after)) - int(has_witness(before)),
            }
        )
    comparison: Record = {
        "contexts": len(paired),
        "boards": len({row["layout_id"] for row in paired}),
    }
    for metric in ("valid_delta", "witness_delta"):
        values = board_means(paired, lambda row, metric=metric: row[metric])
        comparison[metric] = sum(values) / len(values) if values else None
        comparison[f"{metric}_95_board_bootstrap"] = bootstrap(
            values, random.Random(20261114)
        )
    return comparison


def summarise(results: list[Record]) -> tuple[list[Record], Record]:
    summary = []
    methods = [Method.BASE, Method.ADAPTED]
    if any(row["method"] == Method.IN_CONTEXT for row in results):
        methods.append(Method.IN_CONTEXT)
    methods.append(Method.LOCAL_RULE)
    for method in methods:
        planned = [row for row in results if row["method"] == method]
        available = [
            row for row in planned if row["status"] == AssessmentStatus.EVALUATED
        ]
        good = [row for row in available if row["valid_nontrivial"]]
        witness = [row for row in good if has_witness(row)]
        valid_boards = board_means(available, lambda row: int(row["valid_nontrivial"]))
        witness_boards = board_means(available, lambda row: int(has_witness(row)))
        rng = random.Random(20261114)
        summary.append(
            {
                "method": method,
                "planned_contexts": len(planned),
                "planned_boards": len({row["layout_id"] for row in planned}),
                "outputs": len(available),
                "boards_with_outputs": len(valid_boards),
                "missing_infrastructure_outputs": len(planned) - len(available),
                "valid_json": sum(row["json_valid"] for row in available),
                "valid_nontrivial": len(good),
                "witness_found": len(witness),
                "unresolved_among_valid": len(good) - len(witness),
                "valid_json_fraction": sum(row["json_valid"] for row in available)
                / len(available)
                if available
                else None,
                "valid_fraction": len(good) / len(available) if available else None,
                "witness_fraction_all_outputs": len(witness) / len(available)
                if available
                else None,
                "witness_fraction_valid": len(witness) / len(good) if good else None,
                "board_mean_valid_fraction": sum(valid_boards) / len(valid_boards)
                if valid_boards
                else None,
                "board_mean_witness_fraction": sum(witness_boards) / len(witness_boards)
                if witness_boards
                else None,
                "valid_fraction_95_board_bootstrap": bootstrap(valid_boards, rng),
                "witness_fraction_95_board_bootstrap": bootstrap(witness_boards, rng),
                "effect_types": dict(
                    Counter(effect for row in good for effect in row["effect_types"])
                ),
                "conjunction_sizes": dict(
                    Counter(str(row["conjunction_size"]) for row in good)
                ),
                "solver_statuses": dict(
                    Counter(row["solver"]["status"] for row in good)
                ),
                "solver_expanded_nodes": sum(
                    row["solver"]["expanded_nodes"] for row in good
                ),
                "solver_seconds": sum(row["solver"]["elapsed_seconds"] for row in good),
            }
        )
    return summary, paired_comparison(results, Method.BASE, Method.ADAPTED)


def write_examples(results: list[Record], output: Path) -> None:
    """Record representative successes and failures without changing metrics."""
    selected: list[Record] = []
    for method in Method:
        eligible = [
            row
            for row in results
            if row["method"] == method and row["status"] == AssessmentStatus.EVALUATED
        ]
        for matches in (
            lambda row: not row["valid_nontrivial"],
            lambda row: row["valid_nontrivial"] and has_witness(row),
            lambda row: row["valid_nontrivial"] and not has_witness(row),
        ):
            example = next((row for row in eligible if matches(row)), None)
            if example is not None:
                selected.append(example)
    examples = []
    for row in selected:
        solver = row["solver"]
        witness = solver.get("reached_observation") if solver else None
        predicates = []
        if row["valid_nontrivial"]:
            for group in ("effects", "preserve"):
                for predicate in row["intent"].get(group, []):
                    predicates.append(
                        {
                            "group": group,
                            "predicate": predicate,
                            "start": predicate_value(
                                predicate, row["start_observation"]
                            ),
                            "demonstration_end": predicate_value(
                                predicate, row["demonstration_later_observation"]
                            ),
                            "solver_end": predicate_value(predicate, witness)
                            if witness
                            else None,
                        }
                    )
        examples.append(
            {
                "method": row["method"],
                "record_id": row["record_id"],
                "layout_id": row["layout_id"],
                "initial_board": row["start_observation"],
                "raw_output": row["raw_output"],
                "parsed_intent": row["intent"],
                "validation_errors": row["validation_errors"],
                "predicate_truth": predicates,
                "solver": solver,
                "demonstration_end_board": row["demonstration_later_observation"],
            }
        )
    atomic_write(
        output / "examples.json",
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n",
    )
    lines = [
        "# Selected held-out examples",
        "",
        "Illustrative examples are selected by method and outcome; all metrics use the full frozen subset.",
        "The demonstration end is shown for context and was not supplied to the generator. "
        "Any reached state is a bounded-search witness replayed in the environment, not a controller rollout.",
    ]
    for example in examples:
        solver = example["solver"]
        lines.extend(
            [
                "",
                f"## {example['method']} — {example['record_id']}",
                "",
                "Current board:",
                "",
                "```text",
                *example["initial_board"],
                "```",
                "",
                "Raw output:",
                "",
                "````text",
                example["raw_output"],
                "````",
                "",
                "Parsed intent and validation:",
                "",
                "```json",
                json.dumps(
                    {
                        "intent": example["parsed_intent"],
                        "errors": example["validation_errors"],
                        "predicate_truth": example["predicate_truth"],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
                "",
                "Recorded demonstration end:",
                "",
                "```text",
                *example["demonstration_end_board"],
                "```",
            ]
        )
        if solver and solver["witness_found"]:
            lines.extend(
                [
                    "",
                    f"Bounded-search witness: {solver['witness_depth']} primitive actions.",
                    "",
                    "```text",
                    *solver["reached_observation"],
                    "```",
                ]
            )
        elif solver:
            lines.extend(
                [
                    "",
                    f"Bounded search: {solver['status']}; no witness found within this search.",
                ]
            )
        else:
            lines.extend(["", "Bounded search was not run for this invalid intent."])
    atomic_write(output / "examples.md", "\n".join(lines) + "\n")
