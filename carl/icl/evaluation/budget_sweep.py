"""Compare separately executed action budgets without mixing experiment contracts."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from carl.icl.evaluation.data import EvaluationConfig, SavedRun
from carl.icl.evaluation.statistics import board_means, interval
from carl.icl.types import JSONObject


def _sweep_key(run: SavedRun) -> str:
    """Keep all settings matched except the action budget and method treatment."""
    ignored = {
        "name",
        "method",
        "scales",
        "fixed_cap",
        "action_budget",
        "output_dir",
        "cache_dir",
    }
    contract = {
        "config": run.settings.model_dump(exclude=ignored),
        "provider": run.record.provider,
        "policy": run.detail["frozen_policy"],
        "prompt_profile_recorded": run.detail.get("_prompt_profile_recorded", False),
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def actual_budget_results(
    runs: list[SavedRun], config: EvaluationConfig
) -> list[JSONObject]:
    """Estimate completion for each actually executed action cap; report coverage."""
    by_budget: dict[str, set[int]] = defaultdict(set)
    planned_budgets: dict[str, set[int]] = defaultdict(set)
    planned_conditions: dict[str, set[str]] = defaultdict(set)
    plans: dict[Path, JSONObject] = {}
    grouped: dict[
        tuple[str, str, str, tuple[float, ...], str, int], list[SavedRun]
    ] = defaultdict(list)
    for run in runs:
        key = _sweep_key(run)
        budget = run.settings.action_budget
        by_budget[key].add(budget)
        for parent in run.directory.parents:
            path = parent / "sweep_plan.json"
            if path.is_file():
                if path not in plans:
                    plans[path] = json.loads(path.read_text(encoding="utf-8"))
                plan = plans[path]
                planned_budgets[key].update(plan["budgets"])
                planned_conditions[key].update(
                    item["name"]
                    for item in plan["conditions"]
                    if item["family"] == run.record.family
                )
                break
        series = (
            key,
            run.record.config,
            run.record.method,
            tuple(run.settings.scales),
            run.settings.fixed_cap,
            budget,
        )
        grouped[series].append(run)
    series_per_sweep: dict[str, set[str]] = defaultdict(set)
    series_per_budget: dict[tuple[str, int], set[str]] = defaultdict(set)
    calibration_per_series: dict[
        tuple[str, str, str, tuple[float, ...], str], set[str | None]
    ] = defaultdict(set)
    for key, name, _method, _scales, _cap, budget in grouped:
        series_per_sweep[key].add(name)
        series_per_budget[(key, budget)].add(name)
        calibration_per_series[(key, name, _method, _scales, _cap)].update(
            run.detail.get("calibration_signature")
            for run in grouped[(key, name, _method, _scales, _cap, budget)]
        )
    for key, names in planned_conditions.items():
        series_per_sweep[key].update(names)
    if any(len(signatures) > 1 for signatures in calibration_per_series.values()):
        raise ValueError("A budget series contains conflicting calibration signatures")
    results = []
    for (key, name, method, scales, fixed_cap, budget), records in grouped.items():
        if max(len(by_budget[key]), len(planned_budgets[key])) < 2:
            continue
        signatures = {r.detail.get("calibration_signature") for r in records}
        if len(signatures) > 1:
            raise ValueError(
                f"Conflicting calibration signatures for {name} at budget {budget}"
            )
        values = board_means(records, lambda run: float(run.record.solved))
        low, high = interval(values, config)
        expected = records[0].settings.instances * len(records[0].settings.repeat_seeds)
        results.append(
            {
                "sweep": key,
                "config": name,
                "method": method,
                "scales": list(scales),
                "fixed_cap": fixed_cap,
                "provider": records[0].record.provider,
                "family": records[0].record.family,
                "split": records[0].record.split,
                "budget": budget,
                "runs": len(records),
                "expected_runs": expected,
                "missing_runs": max(0, expected - len(records)),
                "missing_conditions": sorted(
                    series_per_sweep[key] - series_per_budget[(key, budget)]
                ),
                "unstarted_budgets": sorted(planned_budgets[key] - by_budget[key]),
                "boards": len(values),
                "solved": sum(r.record.solved for r in records),
                "completion": float(np.mean(values)),
                "ci95_low": low,
                "ci95_high": high,
                "api_errors": sum(
                    r.record.finish_reason == "api_error" for r in records
                ),
                "api_budget_stops": sum(
                    r.record.finish_reason == "api_budget" for r in records
                ),
                "expansion_budget_stops": sum(
                    r.record.finish_reason == "expansion_budget" for r in records
                ),
            }
        )
    return sorted(results, key=lambda row: (row["sweep"], row["config"], row["budget"]))


def actual_budget_figure(rows: list[JSONObject]) -> go.Figure:
    """Show one point per independently run budget, with board-bootstrap intervals."""
    figure = go.Figure()
    names = list(
        dict.fromkeys(
            (row["config"], row["method"], tuple(row["scales"]), row["fixed_cap"])
            for row in rows
        )
    )
    for name, method, scales, cap in names:
        symbol = "diamond" if method == "fixed" else "circle"
        group = [
            row
            for row in rows
            if (row["config"], row["method"], tuple(row["scales"]), row["fixed_cap"])
            == (name, method, scales, cap)
        ]
        figure.add_trace(
            go.Scatter(
                x=[row["budget"] for row in group],
                y=[row["completion"] for row in group],
                name=f"{name} · {method} · {list(scales)}"
                + (f" · cap={cap}" if method == "fixed" else ""),
                mode="lines+markers",
                line={"dash": "dash" if method == "fixed" else "solid"},
                connectgaps=False,
                marker={
                    "size": 12 if method == "fixed" else 8,
                    "symbol": [
                        f"{symbol}-open"
                        if row["missing_runs"]
                        or row["missing_conditions"]
                        or row["unstarted_budgets"]
                        else symbol
                        for row in group
                    ],
                },
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [
                        row["ci95_high"] - row["completion"]
                        if row["ci95_high"] is not None
                        else None
                        for row in group
                    ],
                    "arrayminus": [
                        row["completion"] - row["ci95_low"]
                        if row["ci95_low"] is not None
                        else None
                        for row in group
                    ],
                },
                customdata=[
                    [
                        row["solved"],
                        row["runs"],
                        row["expected_runs"],
                        row["boards"],
                        row["api_errors"],
                        row["api_budget_stops"],
                        row["expansion_budget_stops"],
                        ", ".join(row["missing_conditions"]) or "none",
                        ", ".join(str(value) for value in row["unstarted_budgets"])
                        or "none",
                    ]
                    for row in group
                ],
                hovertemplate="Action budget: %{x}<br>Board-balanced solve rate: %{y:.1%}"
                "<br>Solved/runs: %{customdata[0]}/%{customdata[1]} (expected %{customdata[2]})"
                "<br>Boards: %{customdata[3]}<br>API errors: %{customdata[4]}"
                "<br>API budget stops: %{customdata[5]}<br>Expansion budget stops: %{customdata[6]}"
                "<br>Missing conditions: %{customdata[7]}<br>Unstarted budgets: %{customdata[8]}"
                "<extra>%{fullData.name}</extra>",
            )
        )
    context = f" · {rows[0]['provider']} · {rows[0]['family']}" if rows else ""
    figure.update_layout(
        template="plotly_white",
        title="Solve rate at independently executed action budgets · 95% board bootstrap"
        + context,
        xaxis_title="Action budget per run",
        yaxis_title="Board-balanced solve rate",
        height=500,
        legend={"orientation": "h", "y": -0.25},
        margin={"b": 115},
    )
    figure.update_yaxes(tickformat=".0%", range=[-0.05, 1.05], tick0=0, dtick=0.2)
    return figure
