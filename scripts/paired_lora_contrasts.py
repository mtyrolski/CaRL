#!/usr/bin/env python3
"""Compare LoRA variants on shared contexts using board-level bootstrap."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

from carl.icl.intent_assessment import board_means
from carl.icl.intent_assessment import bootstrap
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_assessment import read_jsonl

matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402


def compare(rows: list[dict[str, Any]], reference: str, candidate: str, metric: str,
            family: str | None = None) -> dict[str, Any]:
    """Pair evaluated contexts first; resample independent board averages."""
    selected = [row for row in rows if family is None or row.get("family") == family]
    indexed: dict[str, dict[str, dict[str, Any]]] = {reference: {}, candidate: {}}
    for row in selected:
        method = row.get("variant")
        if method not in indexed:
            continue
        identifier = row.get("record_id")
        if not isinstance(identifier, str) or identifier in indexed[method]:
            raise ValueError("A variant has a missing or duplicate context ID")
        indexed[method][identifier] = row
    before, after = indexed[reference], indexed[candidate]
    planned = before.keys() & after.keys()
    for identifier in planned:
        if (before[identifier].get("layout_id") != after[identifier].get("layout_id")
                or before[identifier].get("family") != after[identifier].get("family")):
            raise ValueError(f"Paired context has conflicting board or family: {identifier}")
    common = [identifier for identifier in sorted(planned)
              if before[identifier].get("status") == after[identifier].get("status") == "evaluated"]
    paired = []
    for identifier in common:
        left, right = before[identifier], after[identifier]
        if not isinstance(left.get(metric), bool) or not isinstance(right.get(metric), bool):
            raise ValueError(f"Paired metric is missing or not boolean: {identifier}")
        paired.append({"layout_id": left["layout_id"], "before": int(left[metric]), "after": int(right[metric]),
                       "delta": int(right[metric]) - int(left[metric])})
    means = board_means(paired, lambda row: row["delta"])
    bounds = bootstrap(means, random.Random(20261114))
    return {"family": family or "all", "reference": reference, "candidate": candidate, "metric": metric,
            "planned_paired_contexts": len(planned), "paired_contexts": len(paired), "paired_boards": len(means),
            "missing_paired_contexts": len(planned) - len(paired),
            "reference_valid": sum(row["before"] for row in paired),
            "candidate_valid": sum(row["after"] for row in paired),
            "board_mean_delta": sum(means) / len(means) if means else None,
            "bootstrap_low": bounds[0] if bounds else None, "bootstrap_high": bounds[1] if bounds else None,
            "uncertainty": "paired board bootstrap, 2000 resamples" if bounds else "descriptive: fewer than five boards"}


def plot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hide the aggregate on plots only when it duplicates the sole family."""
    strict = [row for row in rows if row["metric"] == "strict_valid"]
    families = {row["family"] for row in strict if row["family"] != "all"}
    return [row for row in strict if row["family"] != "all"] if len(families) == 1 else strict


def plot(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot strict differences with unsmoothed board-bootstrap intervals."""
    displayed = plot_rows(rows)
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8, max(3, .55 * len(displayed) + 1.8)), constrained_layout=True)
    labels = []
    for index, row in enumerate(displayed):
        labels.append(f"{row['candidate']} − {row['reference']}\n{row['family']}")
        point = row["board_mean_delta"]
        if point is None:
            ax.text(0, index, "brak wspólnych wyników", ha="center", fontsize=8)
            continue
        ax.scatter([point], [index], color="#325e82", s=35)
        if row["bootstrap_low"] is not None:
            ax.hlines(index, row["bootstrap_low"], row["bootstrap_high"], color="#325e82")
        ax.annotate(f"{row['paired_contexts']} kontekstów, {row['paired_boards']} plansz", (point, index),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axvline(0, color="0.35", linewidth=.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Różnica odsetka poprawnych intencji (kandydat − r4)",
           xlim=(-1.15, 1.15), ylim=(-.5, max(len(displayed) - .5, .5)),
           title="Sparowane konteksty; bootstrap według plansz")
    ax.grid(axis="x", alpha=.2)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(path / f"paired_ranks.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", default="r4_100")
    parser.add_argument("--candidate", action="append", default=None)
    args = parser.parse_args()
    candidates = args.candidate or ["r8_100", "r2_100"]
    if len(set(candidates)) != len(candidates) or args.reference in candidates:
        raise ValueError("Candidates must be distinct from each other and the reference")
    rows = read_jsonl(args.assessment)
    families = sorted({row["family"] for row in rows})
    output = [compare(rows, args.reference, candidate, metric, family)
              for candidate in candidates for metric in ("strict_valid", "fence_valid")
              for family in (None, *families)]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "paired_ranks.json"
    if path.exists():
        raise FileExistsError("Paired output already exists; use a new directory")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"assessment_sha256": file_hash(args.assessment), "seed": 20261114,
                                     "bootstrap_resamples": 2000, "results": output}, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    csv_path = args.output / "paired_ranks.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    csv_temporary.replace(csv_path)
    plot(args.output, output)
    print(path)


if __name__ == "__main__":
    main()
