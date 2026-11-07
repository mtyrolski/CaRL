"""Offline summaries and figures for the trajectory reachability experiment.

The profile uses trajectory offset k.  Relative labels instead rank measured
successful first-hit times d; the two axes must not be interchanged.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from carl.icl.calibration import weighted_percentiles


def summarize_trials(records: list[dict[str, Any]], expected_trials: int) -> dict[str, Any]:
    """Summarize a target only once all planned rollouts have finished.

    Missing and failed records remain visible; they are not failures of the
    controller and are never silently included in the reachability denominator.
    """
    if expected_trials < 1:
        raise ValueError("expected_trials must be positive")
    complete = []
    for row in records:
        if row.get("status", "complete") != "complete" or row.get("error"):
            continue
        hit = row.get("exact_hit")
        first_hit = row.get("first_hit")
        solved = row.get("solved_before_target")
        valid_time = (isinstance(first_hit, int) and not isinstance(first_hit, bool) and first_hit >= 0)
        if not isinstance(hit, bool) or not isinstance(solved, bool):
            continue
        if (hit and (not valid_time or solved)) or (not hit and first_hit is not None):
            continue
        if "actions_attempted" in row:
            attempted = row["actions_attempted"]
            if not isinstance(attempted, int) or isinstance(attempted, bool) or attempted < 0:
                continue
            # Rollout stops at the first exact hit; it cannot continue past it.
            if hit and first_hit != attempted:
                continue
        complete.append(row)
    trial_ids = [row["trial"] for row in records if "trial" in row]
    unique_trials = len(trial_ids) == len(set(trial_ids))
    finished = (len(records) == expected_trials and len(complete) == expected_trials and unique_trials)
    hits = sum(row["exact_hit"] for row in complete)
    times = [row["first_hit"] for row in complete if row["exact_hit"]]
    return {
        "status": "complete" if finished else "incomplete",
        "completed_trials": len(complete),
        "expected_trials": expected_trials,
        "hits": hits,
        "solved_before_target": sum(row["solved_before_target"] for row in complete),
        "p_hat": hits / expected_trials if finished else None,
        "d": sum(times) / len(times) if finished and times else None,
    }


def label_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign policy-relative labels within each fully observed model/context.

    Ties receive the midpoint of their shared weighted CDF interval. Zero-hit
    targets have no label. An incomplete target makes the entire context's
    reference distribution incomplete, so no labels are emitted for it.
    """
    result = [{**row, "weight": None, "u": None} for row in rows]
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in result:
        groups[(row.get("model"), row.get("context_id"))].append(row)
    for group in groups.values():
        if any(row.get("status") != "complete" for row in group):
            continue
        retained = [row for row in group if row.get("hits", 0) > 0 and row.get("d") is not None]
        if not retained:
            continue
        weights, labels = weighted_percentiles(
            [row["d"] for row in retained], [row["hits"] for row in retained]
        )
        for row, weight, label in zip(retained, weights, labels, strict=True):
            row["weight"], row["u"] = float(weight), float(label)
    return result


def _csv(path: Path, rows: list[dict[str, Any]], initial: list[str]) -> None:
    columns = list(dict.fromkeys([*initial, *(key for row in rows for key in row)]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def _figures(targets: list[dict[str, Any]], output: Path, provider: str) -> None:
    # Load plotting only when needed, allowing statistics to run without a GUI.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    for kind in ("reachability", "percentile"):
        fig, axis = plt.subplots(figsize=(6.8, 4.1), layout="constrained")
        groups: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for row in targets:
            groups[(row.get("model"), row.get("context_id"))].append(row)
        plotted = 0
        for (model, context), group in groups.items():
            if kind == "reachability":
                visible = [row for row in group if row.get("p_hat") is not None]
                xs, ys = [row["k"] for row in visible], [row["p_hat"] for row in visible]
            else:
                visible = [row for row in group if row.get("u") is not None]
                xs, ys = [row["d"] for row in visible], [row["u"] for row in visible]
            if visible:
                axis.scatter(xs, ys, s=18, alpha=0.75, label=f"{model} · {context}")
                plotted += 1
        if not plotted:
            axis.text(0.5, 0.5, "No complete observations", ha="center", va="center", transform=axis.transAxes)
        axis.set_ylim(-0.03, 1.03)
        if kind == "reachability":
            axis.set(xlabel="Trajectory offset k (actions)", ylabel="Exact-target reach rate", title=f"CLLP reachability by target offset [{provider.upper()}]")
        else:
            axis.set(xlabel="Mean successful first-hit time d (actions)", ylabel="Policy-relative percentile u", title=f"Within-context weighted percentiles [{provider.upper()}]")
        axis.grid(alpha=0.2)
        if 0 < plotted <= 8:
            axis.legend(fontsize="x-small")
        for suffix in ("pdf", "svg"):
            fig.savefig(output / f"{kind}.{suffix}")
        plt.close(fig)


def write_report(records: list[dict[str, Any]], output_dir: str | Path, expected_trials: int,
                 *, provider: str = "unknown") -> list[dict[str, Any]]:
    """Write raw trial rows, complete-only target summaries and static figures.

    The caller must include a row with status ``missing`` for every unattempted
    planned trial so that unobserved targets cannot disappear from the CDF.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if provider not in {"mock", "openai", "replay", "unknown"}:
        raise ValueError("Unknown report provider")
    if any("provider" in row and row["provider"] != provider for row in records):
        raise ValueError("Trial provider differs from report provider")
    records = [{**row, "provider": provider} for row in records]
    group_fields = ("model", "context_id", "target_hash", "k")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if any(key not in row for key in group_fields):
            raise ValueError("Every planned trial must identify model, context_id, target_hash and k")
        groups[tuple(row[key] for key in group_fields)].append(row)
    targets = []
    for key, group in groups.items():
        target = dict(zip(group_fields, key, strict=True))
        for field in ("provider", "source_hash", "family", "dataset", "trajectory_id", "target_id", "position"):
            if field in group[0]:
                target[field] = group[0][field]
        target.update(summarize_trials(group, expected_trials))
        targets.append(target)
    targets = label_targets(targets)
    _csv(output / "results.csv", records, ["provider", *group_fields, "trial", "status"])
    _csv(output / "targets.csv", targets, ["provider", *group_fields, "status", "completed_trials", "expected_trials",
                                           "hits", "p_hat", "d", "weight", "u"])
    _figures(targets, output, provider)
    return targets
