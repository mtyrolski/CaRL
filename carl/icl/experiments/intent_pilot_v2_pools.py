"""Materialize the two predeclared Sokoban calibration groups."""

from carl.icl.boards import canonical_layout_id, from_rows
from carl.icl.experiments.intent_pilot_attempts import read
from carl.icl.experiments.intent_pilot_v2 import V2, _terrain, certify
from carl.icl.intent_domain import TASK_GOAL
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject


def calibration_cases() -> tuple[list[JSONObject], list[JSONObject]]:
    """Return public case inputs and separate private native certificates."""
    manifest = read(V2 / "splits.json")
    if manifest["schema"] != "intent-pilot-v2-splits":
        raise ValueError("Unsupported v2 split manifest")
    groups = sorted(
        (group for group in manifest["groups"]
         if group["domain"] == "sokoban" and group["partition"] == "calibration"),
        key=lambda group: group["index"],
    )
    if [group["index"] for group in groups] != [0, 1]:
        raise ValueError("Expected the straight and turning Sokoban calibration groups")

    cases: list[JSONObject] = []
    for group in groups:
        index = group["index"]
        terrain, bottom_row, tunnel_col, preserved_col, goal_col = _terrain("calibration", index, group["seed"])
        goal_row = next(row for row in range(2, bottom_row) if terrain[row][goal_col] == ".")
        player_row = bottom_row if index == 0 else goal_row + 2
        terrain[player_row][tunnel_col] = "@"
        terrain[player_row - 1][tunnel_col] = "$"
        terrain[1][preserved_col] = "*"
        board = ["".join(row) for row in terrain]
        layout = canonical_layout_id(from_rows(board))
        if layout != group["group_id"]:
            raise ValueError(f"Sokoban layout differs from frozen group {group['case_id']}")

        near = (player_row - 2, tunnel_col)
        far = (goal_row, goal_col)
        targets = (near, far) if index == 0 else (far, near)
        requests = [
            {
                "request_id": name,
                "intent": {
                    "effects": [{"predicate": "box_at", "row": row, "col": col}],
                    "preserve": [{"predicate": "box_at", "row": 1, "col": preserved_col}],
                    "instruction": None,
                },
            }
            for name, (row, col) in zip(("A", "B"), targets, strict=True)
        ]
        cases.append({**group, "observation": board, "task_goal": TASK_GOAL, "requests": requests,
                      "context_id": contract_hash([layout, board, TASK_GOAL])})

    return cases, certify(cases)
