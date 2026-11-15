"""Full-start selection must retain box counts and separate canonical layouts."""

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from scripts.prepare_overnight_stage05_boards import select


def sample(family: str, split: str, rank: int, wall_col: int) -> dict:
    boxes = 4 if family == "12x12x4" else 10
    cells = [list("############")] + [list("#----------#") for _ in range(10)] + [list("############")]
    cells[1][1] = "@"
    cells[10][wall_col] = "#"
    for index in range(boxes):
        cells[2][index + 1] = "$"
        cells[3][index + 1] = "."
    initial = ["".join(row) for row in cells]
    return {"family": family, "source_family": "fixture", "split": split, "board_rank": rank,
            "layout_id": canonical_layout_id(from_rows(initial)), "trajectory_id": f"{family}-{split}-{rank}",
            "initial_state": initial, "source_path": "fixture", "source_sha256": "fixture",
            "exposure_status": "fixture", "original_split": "fixture", "actions": [0, 1]}


def test_select_full_start_without_demonstration_actions() -> None:
    rows = [sample(family, split, 0, wall) for family in ("12x12x4", "12x12x10")
            for split, wall in (("development", 1), ("evaluation", 2))]
    selected = select(list(reversed(rows)), 1, 1)
    assert len(selected) == 4
    assert all("actions" not in row and not row["initially_solved"] for row in selected)
    assert {row["split"] for row in selected} == {"development", "evaluation"}


def test_reject_wrong_box_count_and_cross_split_layout() -> None:
    rows = [sample(family, split, 0, wall) for family in ("12x12x4", "12x12x10")
            for split, wall in (("development", 1), ("evaluation", 2))]
    wrong = [dict(row) for row in rows]
    wrong[2]["initial_state"] = wrong[0]["initial_state"]
    with pytest.raises(ValueError, match="box, goal or player"):
        select(wrong, 1, 1)
    duplicate = [dict(row) for row in rows]
    duplicate[1]["initial_state"] = duplicate[0]["initial_state"]
    duplicate[1]["layout_id"] = duplicate[0]["layout_id"]
    with pytest.raises(ValueError, match="Duplicate or cross-split"):
        select(duplicate, 1, 1)
