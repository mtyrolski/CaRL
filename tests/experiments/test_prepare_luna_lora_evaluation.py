"""Checks for the frozen, model-free Luna/LoRA evaluation selection."""

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from scripts.prepare_luna_lora_evaluation import EvaluationFamily
from scripts.prepare_luna_lora_evaluation import freeze_contexts


def board(boxes: int, wall_col: int) -> list[str]:
    rows = [list("############")] + [list("#----------#") for _ in range(10)] + [list("############")]
    rows[1][1] = "@"
    rows[2][wall_col] = "#"
    cells = [(r, c) for r in (4, 5) for c in range(1, 11)]
    for r, c in cells[:boxes]:
        rows[r][c] = "$"
    for r, c in cells[boxes:2 * boxes]:
        rows[r][c] = "."
    return ["".join(row) for row in rows]


def sources() -> list[dict]:
    records = []
    for index, family in enumerate(EvaluationFamily):
        rows = board(4 if family == EvaluationFamily.REAL else 10, index + 2)
        records.append({
            "record_id": str(index), "context_id": f"context-{index}", "split": "test", "family": family,
            "layout_id": canonical_layout_id(from_rows(rows)), "start_observation": rows,
            "task_goal": "Put every box on a goal.", "later_observation": ["private"],
            "intent": {"private": True}, "k": 99, "fragment_actions": [0],
        })
    return records


COUNTS = {family: 1 for family in EvaluationFamily}


def test_contexts_never_contain_demonstration_labels_or_actions() -> None:
    rows = sources()
    selected = freeze_contexts(rows, [row["record_id"] for row in rows], set(), COUNTS, seed=4)
    assert len(selected) == 3
    assert all({"intent", "later_observation", "k", "fragment_actions"}.isdisjoint(row) for row in selected)
    rows[0]["later_observation"] = ["changed"]
    rows[0]["intent"] = {"changed": True}
    rows[0]["k"] = 3
    assert selected == freeze_contexts(rows, [row["record_id"] for row in rows], set(), COUNTS, seed=4)


def test_rejects_layout_overlap_and_duplicate_test_ids() -> None:
    rows = sources()
    ids = [row["record_id"] for row in rows]
    with pytest.raises(ValueError, match="disjoint"):
        freeze_contexts(rows, ids, {rows[0]["layout_id"]}, COUNTS, seed=4)
    with pytest.raises(ValueError, match="IDs"):
        freeze_contexts(rows, ids + [ids[0]], set(), COUNTS, seed=4)


def test_rejects_wrong_box_family() -> None:
    rows = sources()
    rows[0]["family"] = EvaluationFamily.GENERATED
    with pytest.raises(ValueError, match="box count"):
        freeze_contexts(rows, [row["record_id"] for row in rows], set(), COUNTS, seed=4)
