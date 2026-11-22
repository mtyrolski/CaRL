"""Board descriptions, model schemas, and static Sokoban features."""
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import numpy as np
from scipy.optimize import linear_sum_assignment

from carl.icl.boards import from_rows, tiles, to_rows
from carl.icl.types import Board, BoardRows, JSONObject

FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {"fixed": "k", "relative": "u", "relative_u": "u", "absolute": "d", "absolute_d": "d", "ratio_d_over_ac": "ratio_d_over_ac", "normalized_offset": "normalized_offset"}
)
POLICY_HORIZONS: Final = (1, 4, 8, 16)
POLICY_COORDINATE_VERSION: Final[str] = "board-rc-v1"
POLICY_COORDINATE_INSTRUCTIONS: Final[str] = (
    "\nCoordinates are [row, column], zero-based, with row increasing DOWN "
    "and column increasing RIGHT. Use the coordinates and the complete boards "
    "to carefully compare movable pieces. They describe the same states."
)


def board_coordinates(rows: BoardRows) -> JSONObject:
    """Describe movable pieces and goals using only the supplied board."""
    board = tiles(from_rows(rows))
    players = np.argwhere(np.isin(board, [5, 6]))
    if len(players) != 1:
        raise ValueError("Expected exactly one player when describing a board")
    return {
        "player_rc": players[0].tolist(),
        "boxes_rc": np.argwhere(np.isin(board, [3, 4])).tolist(),
        "goals_rc": np.argwhere(np.isin(board, [2, 3, 6])).tolist(),
    }


def schema_object(properties: JSONObject) -> JSONObject:
    """Build the closed object schema required by structured Responses output."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def layout_generator_payload(payload: JSONObject, state: Board, count: int) -> JSONObject:
    """Constrain each complete output row to the input's walls and goal squares."""
    rows = to_rows(state)
    names = [f"row_{index:02d}" for index in range(len(rows))]
    properties = {
        name: {
            "type": "string",
            "pattern": "^" + "".join(
                "#" if symbol == "#" else "[.*+]" if symbol in ".*+" else "[-$@]"
                for symbol in row
            ) + "$",
        }
        for name, row in zip(names, rows, strict=True)
    }
    payload["text"]["format"]["schema"] = schema_object({
        "subgoals": {"type": "array", "maxItems": count, "items": schema_object(properties)}
    })
    boxes = int(np.isin(tiles(state), [3, 4]).sum())
    payload["input"][0]["content"][0]["text"] += (
        "\nEach subgoal is the COMPLETE board. "
        "Return it as an object with row_00, row_01, etc. in top-to-bottom order. "
        "Each string is one complete board row. Keep every wall and goal in place, "
        f"exactly {boxes} boxes and exactly one player."
    )
    for message in payload["input"]:
        if message["role"] == "assistant":
            content = message["content"][0]
            answer = json.loads(content["text"])
            content["text"] = json.dumps({
                "subgoals": [
                    {f"row_{index:02d}": row for index, row in enumerate(proposal["rows"])}
                    for proposal in answer["subgoals"]
                ]
            })
    return payload


def task(state: Board) -> JSONObject:
    """Describe only the public task goal, without reference solution actions."""
    return {
        "rule": "all boxes on goals",
        "goals": np.argwhere(np.isin(tiles(state), [2, 3, 6])).tolist(),
    }


def board_distance(state: Board, target: Board | None = None) -> float:
    """Manhattan matching for offline mocks and the optional explicit value feature."""
    board = tiles(state)
    boxes = np.argwhere(np.isin(board, [3, 4]))
    goals = np.argwhere(
        np.isin(tiles(target), [3, 4])
        if target is not None
        else np.isin(board, [2, 3, 6])
    )
    costs = np.abs(boxes[:, None] - goals[None, :]).sum(-1)
    row, col = linear_sum_assignment(costs)
    distance = float(costs[row, col].sum())
    if target is not None:
        player = np.argwhere(np.isin(board, [5, 6]))[0]
        target_player = np.argwhere(np.isin(tiles(target), [5, 6]))[0]
        distance += float(np.abs(player - target_player).sum()) * 0.2
    return distance


def value_question(state: Board, *, features: bool) -> JSONObject:
    """Describe the board; optional static features do not use a reference solution."""
    rows = to_rows(state)
    question: JSONObject = {"state": rows, "goal": task(state)}
    if features:
        board = tiles(state)
        details = board_coordinates(rows)
        details.update(
            boxes_on_goals=int((board == 3).sum()),
            boxes_off_goals=int((board == 4).sum()),
            box_to_goal_manhattan=float(board_distance(state)),
            task_solved=bool(not (board == 4).any()),
        )
        question["board_features"] = details
    return question
