"""Inspect calibrated scales and actual proposal/endpoint boards without inference."""
import json
from pathlib import Path
from typing import Final

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from carl.icl.boards import SYMBOLS, from_rows
from carl.icl.evaluation.data import SavedRun
from carl.icl.types import JSONObject

TILE_COLORS: Final = ("#334155", "#f1f5f9", "#fbbf24", "#22c55e", "#b45309", "#2563eb", "#a78bfa")
TILE_LABELS: Final = ("Wall", "Floor", "Goal", "Box on goal", "Box", "Player", "Player on goal")


def load_calibrations(runs: list[SavedRun]) -> tuple[dict[Path, JSONObject], list[str]]:
    """Resolve only referenced signatures; never mix mock labels into live diagnostics."""
    found: dict[Path, JSONObject] = {}
    missing: set[str] = set()
    for run in runs:
        signature = run.detail.get("calibration_signature")
        if not signature:
            continue
        root = run.settings.path("data_dir") / "calibration"
        path = (root / signature / "calibration.json").resolve()
        if path.parent.parent != root.resolve():
            raise ValueError("Invalid calibration signature")
        if not path.exists():
            missing.add(str(path))
            continue
        payload = found.get(path) or json.loads(path.read_text(encoding="utf-8"))
        expected_provider = "mock" if run.record.provider == "mock" else "openai"
        if payload["signature"] != signature or payload["provider"] != expected_provider:
            raise ValueError("Calibration signature/provider disagrees with saved search")
        if payload["policy_fingerprint"] != run.detail["frozen_policy"]:
            raise ValueError("Calibration does not match the saved policy fingerprint")
        found[path] = payload
    return found, sorted(missing)


def calibration_figure(calibration: JSONObject) -> go.Figure:
    """Compare measured policy time with percentile labels and trajectory offsets."""
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Weighted local percentile labels",
            "Trajectory offset vs policy first-hit time",
        ),
    )
    examples = calibration["examples"]
    for context in dict.fromkeys(row["context"] for row in examples):
        rows = sorted(
            (row for row in examples if row["context"] == context), key=lambda row: row["d"]
        )
        marker = {"size": [8 + 24 * row["weight"] for row in rows]}
        metadata = [[row["hits"], row["success_rate"], row["weight"]] for row in rows]
        figure.add_trace(
            go.Scatter(
                x=[row["d"] for row in rows],
                y=[row["u"] for row in rows],
                name=context,
                mode="markers+lines",
                legendgroup=context,
                marker=marker,
                customdata=metadata,
                hovertemplate="Mean hit time: %{x}<br>Percentile: %{y:.3f}<br>Hits: %{customdata[0]}"
                "<br>Success: %{customdata[1]:.1%}<br>Weight: %{customdata[2]:.3f}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=[row["k"] for row in rows],
                y=[row["d"] for row in rows],
                name=context,
                mode="markers",
                legendgroup=context,
                showlegend=False,
                marker=marker,
            ),
            row=1,
            col=2,
        )
    figure.update_xaxes(title_text="Mean successful first-hit time d", row=1, col=1)
    figure.update_yaxes(title_text="Weighted tied midpoint u", range=[0, 1], row=1, col=1)
    figure.update_xaxes(title_text="Reference trajectory offset k", row=1, col=2)
    figure.update_yaxes(title_text="Measured policy time d", row=1, col=2)
    figure.update_layout(
        template="plotly_white",
        height=480,
        title=f"Calibration · {calibration['provider']} · {calibration['status']} · retained targets only",
        legend={"orientation": "h", "y": -0.28},
        margin={"b": 100},
    )
    return figure


def _board_trace(rows: list[str], show_scale: bool) -> go.Heatmap:
    indices = from_rows(rows).argmax(-1)
    colorscale = [
        (position, color)
        for index, color in enumerate(TILE_COLORS)
        for position in (index / 7, (index + 1) / 7)
    ]
    return go.Heatmap(
        z=indices,
        zmin=-0.5,
        zmax=6.5,
        colorscale=colorscale,
        text=[[SYMBOLS[int(tile)] for tile in row] for row in indices],
        texttemplate="%{text}",
        hovertemplate="row=%{y}, col=%{x}: %{text}<extra></extra>",
        showscale=show_scale,
        colorbar={"tickvals": list(range(7)), "ticktext": TILE_LABELS},
        xgap=1,
        ygap=1,
    )


def board_viewer(run: SavedRun, limit: int) -> go.Figure:
    """Animate saved source, requested target and actual endpoint side by side."""
    events = run.detail.get("events", [])[:limit]
    initial = run.detail["instance"]["state"]
    title = f"{run.record.config} · {run.record.instance} · seed {run.record.seed} · {run.record.provider}"
    frames = []
    labels = []
    for index, event in enumerate(events):
        label = (
            f"#{index + 1} scale={event['scale']:g} actions={event['attempted_actions']} "
            f"hit={event['reached']} solved={event['solved']}"
        )
        labels.append(label)
        frames.append(
            go.Frame(
                name=str(index),
                data=[
                    _board_trace(event[key], col == 2)
                    for col, key in enumerate(("source", "proposal", "endpoint"))
                ],
                traces=[0, 1, 2],
                layout=go.Layout(title=f"{title}<br><sup>{label}</sup>"),
            )
        )
    figure = make_subplots(
        rows=1, cols=3, subplot_titles=("Source", "Requested subgoal", "Actual endpoint")
    )
    first = events[0] if events else {key: initial for key in ("source", "proposal", "endpoint")}
    for column, key in enumerate(("source", "proposal", "endpoint"), start=1):
        figure.add_trace(_board_trace(first[key], column == 3), row=1, col=column)
        figure.update_yaxes(
            autorange="reversed",
            scaleanchor="x" if column == 1 else f"x{column}",
            row=1,
            col=column,
        )
        figure.update_xaxes(constrain="domain", row=1, col=column)
    figure.frames = frames
    if frames:
        figure.update_layout(
            sliders=[
                {
                    "currentvalue": {"prefix": "Attempt: "},
                    "steps": [
                        {
                            "label": str(index + 1),
                            "method": "animate",
                            "args": [
                                [str(index)],
                                {
                                    "mode": "immediate",
                                    "frame": {"duration": 0, "redraw": True},
                                    "transition": {"duration": 0},
                                },
                            ],
                        }
                        for index in range(len(labels))
                    ],
                }
            ]
        )
    else:
        figure.add_annotation(
            text="No saved subgoal events; showing initial state",
            x=0.5,
            y=-0.1,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    figure.update_layout(
        template="plotly_white",
        height=650,
        margin={"b": 180},
        title=title + (f"<br><sup>{labels[0]}</sup>" if labels else ""),
    )
    return figure
