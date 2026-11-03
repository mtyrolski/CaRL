"""Sokoban vector graphics matching the thesis's 2026/plots.ipynb palette and glyphs."""
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from carl.icl.boards import to_rows
from carl.icl.types import Board, JSONObject

COLORS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "wall": "#263e50",
        "floor": "#fff5d7",
        "box": "#efa44c",
        "cross": "#865121",
        "goal": "#32724b",
        "player": "#286bf4",
        "ink": "#13233e",
    }
)
TRANSPARENT: Final[str] = "rgba(0,0,0,0)"
TILE_NAMES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "#": "Wall",
        "-": "Floor",
        ".": "Goal",
        "$": "Box",
        "*": "Box on goal",
        "@": "Player",
        "+": "Player on goal",
    }
)


def _tile_shapes(symbol: str, column: int, row: int, stroke: float) -> list[JSONObject]:
    """Draw a unit tile; preserve the goal marker under an occupied square."""
    shapes: list[JSONObject] = []

    def shape(
        kind: str,
        bounds: tuple[float, float, float, float],
        fill: str,
        outline: str,
        width: float,
    ) -> None:
        x0, y0, x1, y1 = bounds
        item: JSONObject = {
            "type": kind,
            "x0": column + x0,
            "y0": row + y0,
            "x1": column + x1,
            "y1": row + y1,
            "line": {"color": outline, "width": width},
            "layer": "above",
        }
        if kind != "line":
            item["fillcolor"] = fill
        shapes.append(item)

    goal = symbol in ".*+"
    base = COLORS["wall"] if symbol == "#" else COLORS["goal" if goal else "floor"]
    shape("rect", (0, 0, 1, 1), base, "rgba(230,235,230,.2)", 0.2)
    if goal:
        shape("circle", (0.2, 0.2, 0.8, 0.8), TRANSPARENT, "white", stroke)
    if symbol in "$*":
        inset = 0.18 if symbol == "*" else 0.1
        shape(
            "rect",
            (inset, inset, 1 - inset, 1 - inset),
            COLORS["box"],
            COLORS["cross"],
            max(0.4, stroke * 0.7),
        )
        for y0, y1 in ((0.2, 0.8), (0.8, 0.2)):
            shape("line", (0.2, y0, 0.8, y1), TRANSPARENT, COLORS["cross"], stroke)
        if symbol == "*":
            shape(
                "circle", (0.38, 0.38, 0.62, 0.62), TRANSPARENT, "white", stroke * 0.8
            )
    if symbol in "@+":
        shape(
            "circle",
            (0.18, 0.18, 0.82, 0.82),
            COLORS["player"],
            COLORS["ink"],
            stroke * 0.7,
        )
    return shapes


def board_comparison(left: Board, right: Board, labels: tuple[str, str]) -> go.Figure:
    """Compare boards with square vector tiles, paper-style glyphs and tile hover labels."""
    figure = make_subplots(rows=1, cols=2, subplot_titles=labels)
    shapes: list[JSONObject] = []
    for panel, board in enumerate((left, right), start=1):
        rows = to_rows(board)
        suffix = "" if panel == 1 else str(panel)
        xref, yref = f"x{suffix}", f"y{suffix}"
        stroke = max(0.45, 20 / len(rows[0]))
        hover_x, hover_y, hover_text = [], [], []
        for row, cells in enumerate(rows):
            for column, symbol in enumerate(cells):
                shapes.extend(
                    {**item, "xref": xref, "yref": yref}
                    for item in _tile_shapes(symbol, column, row, stroke)
                )
                hover_x.append(column + 0.5)
                hover_y.append(row + 0.5)
                hover_text.append(f"{TILE_NAMES[symbol]} · row {row}, column {column}")
        figure.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                text=hover_text,
                mode="markers",
                marker={"opacity": 0, "size": 12},
                showlegend=False,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=panel,
        )
        figure.update_xaxes(
            range=[0, len(rows[0])], visible=False, constrain="domain", row=1, col=panel
        )
        figure.update_yaxes(
            range=[len(rows), 0],
            visible=False,
            scaleanchor=xref,
            scaleratio=1,
            constrain="domain",
            row=1,
            col=panel,
        )
    figure.update_layout(
        shapes=shapes,
        template="plotly_white",
        height=max(420, min(850, max(left.shape[0], right.shape[0]) * 24)),
        margin={"t": 55, "b": 15, "l": 10, "r": 10},
    )
    return figure
