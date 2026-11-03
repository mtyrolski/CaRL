"""Summaries and Plotly figures for paired trajectory component studies."""
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go


def load_cases(directory: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read saved cases without making inference requests."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = [json.loads(path.read_text(encoding="utf-8"))
             for path in sorted((root / "cases").glob("shots_*/*/*.json"))]
    return manifest, cases


def wilson(successes: int, total: int) -> tuple[float, float]:
    """95% Wilson interval for a rate, in percentages."""
    if not total:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return 100 * (centre - margin), 100 * (centre + margin)


def _rate(shots: int, metric: str, success: int, observed: int, planned: int) -> dict[str, Any]:
    lower, upper = wilson(success, observed)
    return {"few_shot": shots, "metric": metric, "successes": success, "observed": observed,
            "planned": planned, "rate_pct": 100 * success / observed if observed else None,
            "ci_low_pct": lower if observed else None, "ci_high_pct": upper if observed else None}


def _mean_rate(shots: int, metric: str, scores: list[float], planned: int) -> dict[str, Any]:
    """Bootstrap complete trajectories, rather than correlated pairs within a trajectory."""
    rng = np.random.default_rng(20261102)
    if scores:
        samples = np.asarray(scores, dtype=float)
        bootstrap = samples[rng.integers(0, len(samples), size=(2000, len(samples)))].mean(axis=1)
        low, high = np.percentile(bootstrap, [2.5, 97.5])
        mean = float(np.mean(samples))
    else:
        low = high = mean = math.nan
    return {"few_shot": shots, "metric": metric, "successes": None, "observed": len(scores),
            "planned": planned, "rate_pct": 100 * mean if scores else None,
            "ci_low_pct": 100 * float(low) if scores else None,
            "ci_high_pct": 100 * float(high) if scores else None}


def summarize(manifest: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count known outcomes separately from errors and missing cases."""
    study = manifest["study"]
    planned = study["instances"]
    result = []
    for shots in study["few_shot"]:
        policy = [r for r in cases if r["few_shot"] == shots and r["component"] == "policy"]
        valid_policy = [r for r in policy if "error" not in r]
        result.append(_rate(shots, "policy_reached", sum(bool(r["reached"]) for r in valid_policy),
                            len(valid_policy), planned))
        generated = [r for r in cases if r["few_shot"] == shots and r["component"] == "generator"]
        valid_generation = [r for r in generated if "error" not in r["generator"]]
        valid_proposal = [r for r in valid_generation if r["generator"]["valid_count"] > 0]
        result.append(_rate(shots, "generator_valid", len(valid_proposal), len(valid_generation), planned))
        reached_evaluable = [r for r in valid_generation if r["generator"]["valid_count"] == 0
                             or ("rollout" in r and "error" not in r["rollout"])]
        reached = sum(bool(r.get("rollout", {}).get("reached")) for r in reached_evaluable)
        result.append(_rate(shots, "generated_reached_all", reached, len(reached_evaluable), planned))
        attempted = [r for r in valid_proposal if "rollout" in r and "error" not in r["rollout"]]
        result.append(_rate(shots, "generated_reached_given_valid",
                            sum(bool(r["rollout"]["reached"]) for r in attempted), len(attempted), planned))
        values = [r for r in cases if r["few_shot"] == shots and r["component"] == "value"]
        complete = [r for r in values if len(r["values"]) == len(study["value_positions"])
                    and all("value" in value for value in r["values"])]
        tolerance = study["monotonic_tolerance"]
        deltas = [[b["value"] - a["value"] for a, b in zip(row["values"][:-1], row["values"][1:])]
                  for row in complete]
        scores = [float(np.mean([delta >= -tolerance for delta in trajectory])) for trajectory in deltas]
        result.append(_rate(shots, "value_fully_nondecreasing", sum(score == 1 for score in scores), len(scores), planned))
        result.append(_mean_rate(shots, "value_mean_nondecreasing_edges", scores, planned))
        result.append(_mean_rate(shots, "value_mean_increasing_edges",
                                 [float(np.mean([delta > tolerance for delta in row])) for row in deltas], planned))
        result.append(_mean_rate(shots, "value_mean_tied_edges",
                                 [float(np.mean([abs(delta) <= tolerance for delta in row])) for row in deltas], planned))
    for row in result:
        metric = row["metric"]
        component = "value" if metric.startswith("value_") else "policy" if metric == "policy_reached" else "generator"
        matching = [case for case in cases if case["few_shot"] == row["few_shot"] and case["component"] == component]
        row.update(cases_recorded=len(matching), cases_with_errors=sum("error" in case for case in matching),
                   cases_missing=planned - len(matching))
    return result


def _rate_figure(summary: list[dict[str, Any]], title: str, labels: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    for metric, label in labels.items():
        rows = [row for row in summary if row["metric"] == metric]
        fig.add_trace(go.Bar(
            name=label, x=[str(row["few_shot"]) for row in rows],
            y=[row["rate_pct"] for row in rows],
            error_y={"type": "data", "symmetric": False,
                     "array": [row["ci_high_pct"] - row["rate_pct"] if row["rate_pct"] is not None else 0 for row in rows],
                     "arrayminus": [row["rate_pct"] - row["ci_low_pct"] if row["rate_pct"] is not None else 0 for row in rows]},
            customdata=[[row["successes"], row["observed"], row["planned"], row["cases_with_errors"], row["cases_missing"]] for row in rows],
            hovertemplate=("%{y:.1f}%<br>successes: %{customdata[0]}"
                           "<br>observed: %{customdata[1]} / %{customdata[2]} planned"
                           "<br>errors: %{customdata[3]}<br>missing: %{customdata[4]}<extra>%{fullData.name}</extra>"),
        ))
    fig.update_layout(title=title, xaxis_title="Few-shot examples", yaxis_title="Percent",
                      yaxis_range=[0, 105], barmode="group", template="plotly_white")
    return fig


def figures(manifest: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, go.Figure]:
    """Build all requested charts from saved records only."""
    summary = summarize(manifest, cases)
    plots = {
        "policy_reach": _rate_figure(summary, "CLLP: reached offline trajectory target within k actions",
                                      {"policy_reached": "Exact target reached"}),
        "generator_reach": _rate_figure(summary, "Generator + CLLP: valid and reached targets",
            {"generator_valid": "Valid generated board", "generated_reached_all": "Reached / all evaluable cases",
             "generated_reached_given_valid": "Reached / valid boards"}),
        "value_monotonicity": _rate_figure(summary, "Value monotonicity along held-out trajectories",
            {"value_mean_nondecreasing_edges": "Mean fraction of non-decreasing adjacent pairs",
             "value_fully_nondecreasing": "Entire trajectory non-decreasing",
             "value_mean_increasing_edges": "Strictly increasing pairs",
             "value_mean_tied_edges": "Tied pairs"}),
    }
    trajectory = go.Figure()
    for shots in manifest["study"]["few_shot"]:
        rows = [r for r in cases if r["few_shot"] == shots and r["component"] == "value"]
        xs, ys, counts = [], [], []
        for fraction in manifest["study"]["value_positions"]:
            values = [point["value"] for row in rows for point in row["values"]
                      if point["fraction"] == fraction and "value" in point]
            xs.append(100 * fraction)
            ys.append(float(np.median(values)) if values else None)
            counts.append(len(values))
        trajectory.add_trace(go.Scatter(x=xs, y=ys, customdata=counts, mode="lines+markers",
                                        name=f"{shots} examples", hovertemplate="progress %{x:.0f}%<br>median value %{y:.3f}<br>n=%{customdata}<extra>%{fullData.name}</extra>"))
    trajectory.update_layout(title="Value along offline solution trajectories (median)",
        xaxis_title="Progress through witness trajectory (%)", yaxis_title="Predicted discounted value",
        yaxis_range=[0, 1.02], template="plotly_white")
    plots["value_trajectory"] = trajectory
    return plots


def export(directory: str | Path) -> Path:
    """Save interactive HTML charts and machine-readable summaries beside cases."""
    root = Path(directory)
    manifest, cases = load_cases(root)
    summary = summarize(manifest, cases)
    report = root / "plots"
    report.mkdir(parents=True, exist_ok=True)
    for name, figure in figures(manifest, cases).items():
        figure.write_html(report / f"{name}.html", include_plotlyjs="directory")
    (report / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (report / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Luna component study", "", f"Provider: {manifest['provider']}; "
             f"planned trajectories per condition: {manifest['study']['instances']}.", "",
             "| Few-shot | Metric | Rate | Observed | Errors | Missing |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in summary:
        rate = f"{row['rate_pct']:.1f}%" if row["rate_pct"] is not None else "N/A"
        lines.append(f"| {row['few_shot']} | {row['metric']} | {rate} | {row['observed']} | "
                     f"{row['cases_with_errors']} | {row['cases_missing']} |")
    (report / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics = ("requests", "cache_hits", "input_tokens", "cached_input_tokens", "cache_write_tokens",
               "output_tokens", "logical_requests")
    usage = []
    for shots in manifest["study"]["few_shot"]:
        for component in ("policy", "generator", "value"):
            selected = [case for case in cases if case["few_shot"] == shots and case["component"] == component]
            attempts = [attempt["metrics"] for case in selected for attempt in case.get("prior_attempts", [])]
            attempts.extend(case["metrics"] for case in selected)
            usage.append({"few_shot": shots, "component": component, "cases": len(selected),
                          "cases_with_errors": sum("error" in case for case in selected),
                          **{metric: sum(record.get(metric, 0) for record in attempts) for metric in metrics}})
    with (report / "usage.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(usage[0]))
        writer.writeheader()
        writer.writerows(usage)
    return report
