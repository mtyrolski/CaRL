"""Editable Plotly graph_objects figures built only from saved observations."""
from collections.abc import Mapping
from typing import Final

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from carl.icl.evaluation.data import SavedRun
from carl.icl.evaluation.statistics import conditions
from carl.icl.types import JSONObject

PALETTE: Final = ("#2563eb", "#e76f51", "#16a085", "#9b59b6", "#d69e00", "#64748b")
TEMPLATE: Final[str] = "plotly_white"


def _style(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        template=TEMPLATE,
        title=title,
        height=440,
        margin={"t": 80, "b": 70},
        font={"family": "Arial, sans-serif", "size": 13},
        legend={"orientation": "h", "y": -0.22},
    )
    return figure


def _errors(rows: list[JSONObject], value: str) -> JSONObject:
    return {
        "type": "data",
        "symmetric": False,
        "array": [
            row["ci95_high"] - row[value] if row["ci95_high"] is not None else None for row in rows
        ],
        "arrayminus": [
            row[value] - row["ci95_low"] if row["ci95_low"] is not None else None for row in rows
        ],
    }


def completion_figure(summary: list[JSONObject]) -> go.Figure:
    """Show board-balanced completion with cluster-bootstrap intervals."""
    figure = go.Figure(
        go.Bar(
            x=[row["config"] for row in summary],
            y=[row["completion"] for row in summary],
            marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(summary))],
            error_y=_errors(summary, "completion"),
            texttemplate="%{y:.0%}<br>%{customdata[0]}/%{customdata[1]}",
            textposition="outside",
            cliponaxis=False,
            customdata=[
                [row["solved"], row["runs"], row["boards"], row["missing_runs"]] for row in summary
            ],
            hovertemplate="%{x}<br>Completion: %{y:.1%}<br>Solved/runs: %{customdata[0]}/%{customdata[1]}"
            "<br>Boards: %{customdata[2]}<br>Missing scheduled runs: %{customdata[3]}<extra></extra>",
        )
    )
    figure.update_yaxes(title="Board-balanced completion", tickformat=".0%", range=[0, 1.08])
    return _style(figure, "Task completion · 95% board-cluster bootstrap (no CI for one board)")


def budget_figure(curves: list[JSONObject]) -> go.Figure:
    """Use total search effort to completion, never the final solution length."""
    figure = go.Figure()
    for index, name in enumerate(dict.fromkeys(row["config"] for row in curves)):
        rows = [row for row in curves if row["config"] == name]
        figure.add_trace(
            go.Scatter(
                x=[row["budget"] for row in rows],
                y=[row["completion"] for row in rows],
                name=name,
                mode="lines",
                line={"shape": "hv", "color": PALETTE[index % len(PALETTE)]},
                customdata=[[row["ci95_low"], row["ci95_high"]] for row in rows],
                hovertemplate="Actions ≤ %{x}<br>Observed completion: %{y:.1%}"
                "<br>CI: %{customdata[0]} – %{customdata[1]}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_xaxes(title="Total attempted search actions (including failed attempts)")
    figure.update_yaxes(title="Observed completion", tickformat=".0%", range=[0, 1.05])
    return _style(figure, "Observed completion vs search effort · not counterfactual reruns")


def effort_figure(runs: list[SavedRun]) -> go.Figure:
    """Keep unconditional effort and conditional solution length visibly separate."""
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Attempted actions · all runs",
            "Seconds · all runs",
            "Plan length · solved only",
        ),
    )
    for index, (name, group) in enumerate(conditions(runs).items()):
        values = (
            [run.record.attempted_actions for run in group],
            [run.record.seconds for run in group],
            [run.record.solution_actions for run in group if run.record.solved],
        )
        for column, observations in enumerate(values, start=1):
            figure.add_trace(
                go.Box(
                    y=observations,
                    name=name,
                    legendgroup=name,
                    showlegend=column == 1,
                    boxpoints="all",
                    jitter=0.25,
                    pointpos=0,
                    marker_color=PALETTE[index % len(PALETTE)],
                ),
                row=1,
                col=column,
            )
    if not any(run.record.solved for run in runs):
        figure.add_annotation(
            text="No solved runs", x=0.86, y=0.5, xref="paper", yref="paper", showarrow=False
        )
    return _style(figure, "Effort distributions · replay/cache timings are not live API latency")


def usage_figures(runs: list[SavedRun]) -> dict[str, go.Figure]:
    """Separate logical workload from physical provider usage, without guessing prices."""
    grouped = conditions(runs)
    token_figure, call_figure = go.Figure(), go.Figure()
    for field, label in (
        ("logical_input_tokens", "Logical input (local estimate)"),
        ("logical_output_tokens", "Logical output"),
        ("input_tokens", "Physical input (provider)"),
        ("output_tokens", "Physical output (provider)"),
    ):
        token_figure.add_trace(
            go.Bar(
                name=label,
                x=list(grouped),
                y=[
                    float(np.mean([getattr(r.record, field) for r in group]))
                    for group in grouped.values()
                ],
            )
        )
    for field, label in (
        ("generator_calls", "Generator"),
        ("policy_calls", "Policy"),
        ("value_calls", "Value"),
    ):
        call_figure.add_trace(
            go.Bar(
                name=label,
                x=list(grouped),
                y=[
                    float(np.mean([getattr(r.record, field) for r in group]))
                    for group in grouped.values()
                ],
            )
        )
    call_figure.add_trace(
        go.Bar(
            name="Embeddings (logical)",
            x=list(grouped),
            y=[
                float(
                    np.mean(
                        [
                            r.record.logical_requests
                            - r.record.generator_calls
                            - r.record.policy_calls
                            - r.record.value_calls
                            for r in group
                        ]
                    )
                )
                for group in grouped.values()
            ],
        )
    )
    token_figure.update_layout(barmode="group", yaxis_title="Mean tokens per observed run")
    call_figure.update_layout(barmode="stack", yaxis_title="Mean logical calls per observed run")
    return {
        "tokens": _style(
            token_figure, "Token usage · mock physical zeros are not a live cost estimate"
        ),
        "calls": _style(call_figure, "Inference workload by component · cache hits count"),
    }


def termination_figure(runs: list[SavedRun]) -> go.Figure:
    """Retain API failures and exhausted budgets in the denominator."""
    grouped = conditions(runs)
    figure = go.Figure()
    for reason in sorted({run.record.finish_reason for run in runs}):
        figure.add_trace(
            go.Bar(
                name=reason,
                x=list(grouped),
                y=[
                    sum(r.record.finish_reason == reason for r in group) / len(group)
                    for group in grouped.values()
                ],
            )
        )
    figure.update_layout(barmode="stack", yaxis_title="Fraction of observed runs")
    figure.update_yaxes(tickformat=".0%")
    return _style(figure, "Termination reasons · failures are not filtered out")


def reachability_figure(attempts: list[JSONObject]) -> go.Figure:
    """Distinguish exact target hits from task completion at another endpoint."""
    figure = go.Figure()
    keys = list(dict.fromkeys((row["config"], row["scale"]) for row in attempts))
    outcomes = {
        "Exact subgoal hit": lambda row: row["reached"],
        "Task solved before target": lambda row: row["solved"] and not row["reached"],
        "Target not reached": lambda row: not row["solved"] and not row["reached"],
    }
    for label, predicate in outcomes.items():
        figure.add_trace(
            go.Bar(
                name=label,
                x=[f"{name} · scale={scale:g}" for name, scale in keys],
                y=[
                    sum(
                        bool(predicate(row))
                        for row in attempts
                        if (row["config"], row["scale"]) == key
                    )
                    for key in keys
                ],
            )
        )
    figure.update_layout(barmode="stack", yaxis_title="Saved completed subgoal attempts")
    if not attempts:
        figure.add_annotation(text="No saved subgoal attempts", showarrow=False)
    return _style(figure, "Subgoal execution · interrupted attempts may have no saved event")


def paired_figure(points: list[JSONObject], estimates: list[JSONObject]) -> go.Figure:
    """Show signed completion differences; smaller failed-run effort is not better performance."""
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Paired effort · hover for outcomes",
            "Completion difference · left − right",
        ),
    )
    for left, right in dict.fromkeys((row["left"], row["right"]) for row in points):
        rows = [row for row in points if (row["left"], row["right"]) == (left, right)]
        figure.add_trace(
            go.Scatter(
                x=[row["left_actions"] for row in rows],
                y=[row["right_actions"] for row in rows],
                name=f"{left} / {right}",
                mode="markers",
                customdata=[
                    [row["instance"], row["seed"], row["left_solved"], row["right_solved"]]
                    for row in rows
                ],
                hovertemplate="%{customdata[0]} · seed %{customdata[1]}<br>Left/right solved: %{customdata[2]}/%{customdata[3]}"
                "<br>Actions: %{x}/%{y}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )
    if points:
        maximum = max(max(row["left_actions"], row["right_actions"]) for row in points)
        figure.add_trace(
            go.Scatter(
                x=[0, maximum],
                y=[0, maximum],
                mode="lines",
                name="Equal effort",
                line={"dash": "dash", "color": "#94a3b8"},
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Bar(
            x=[f"{row['left']} − {row['right']}" for row in estimates],
            y=[row["completion_delta"] for row in estimates],
            error_y=_errors(estimates, "completion_delta"),
            name="Paired completion delta",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Left attempted actions", row=1, col=1)
    figure.update_yaxes(title_text="Right attempted actions", row=1, col=1)
    figure.update_yaxes(tickformat="+.0%", range=[-1.05, 1.05], row=1, col=2)
    if not points:
        figure.add_annotation(
            text="No unique matched pair. Set comparison_configs for ablations.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    return _style(figure, "Matched board/seed comparison · 95% board-cluster bootstrap")


def cohort_figures(
    runs: list[SavedRun], tables: Mapping[str, list[JSONObject]]
) -> dict[str, go.Figure]:
    """Return ordinary graph_objects figures for direct notebook modification."""
    return {
        "completion": completion_figure(tables["summary"]),
        "budget": budget_figure(tables["curves"]),
        "effort": effort_figure(runs),
        **usage_figures(runs),
        "termination": termination_figure(runs),
        "reachability": reachability_figure(tables["attempts"]),
        "paired": paired_figure(tables["points"], tables["pairs"]),
    }
