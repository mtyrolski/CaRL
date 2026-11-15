"""The follow-up cohorts are frozen without relying on model outcomes."""

import copy

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from scripts.prepare_luna_policy_followup import select_calibration
from scripts.prepare_luna_policy_followup import select_test
from scripts.prepare_luna_policy_followup import semantic_intent_key


def candidate(context, layout, split, identifier, effects):
    return {"record_id": identifier, "context_id": context, "layout_id": layout, "split": split,
            "start_observation": ["#####", "#---#", "#.-@#", "#####"], "task_goal": "solve",
            "intent": {"effects": effects, "preserve": [], "instruction": None}}


def context(context_id, layout, split):
    return [candidate(context_id, layout, split, f"{context_id}-{column}",
                      [{"predicate": "box_at", "row": 1, "col": column}]) for column in range(1, 5)]


def test_goal_count_implied_by_box_on_goal_is_semantically_redundant():
    box = {"predicate": "box_at", "row": 2, "col": 1}
    count = {"predicate": "boxes_on_goals_at_least", "count": 1}
    a = candidate("a", "board", "train", "a1", [box, count])
    b = candidate("a", "board", "train", "a2", [box])
    assert semantic_intent_key(a) == semantic_intent_key(b)
    a["intent"]["preserve"] = [box]
    assert semantic_intent_key(a) != semantic_intent_key(b)


def test_duplicate_context_is_excluded_and_one_context_per_board_is_selected():
    bad = context("bad", "one", "train")
    bad[0]["intent"]["effects"] = [{"predicate": "box_at", "row": 2, "col": 1},
                                   {"predicate": "boxes_on_goals_at_least", "count": 1}]
    bad[1]["intent"]["effects"] = [{"predicate": "box_at", "row": 2, "col": 1}]
    rows = bad + context("good", "one", "train") + context("other", "two", "train")
    rows += context("val", "three", "validation")
    selected, rejected = select_calibration(rows, 20261115, 2, 1)
    assert [r["context_id"] for r in rejected] == ["bad"]
    assert {r["context_id"] for r in selected} == {"good", "other", "val"}
    assert len(selected) == 12
    repeated, _ = select_calibration(list(reversed(rows)), 20261115, 2, 1)
    assert [r["record_id"] for r in repeated] == [r["record_id"] for r in selected]


def board(goal_column):
    grid = [list("############")] + [list("#----------#") for _ in range(10)] + [list("############")]
    grid[1][1] = "@"
    for column in range(2, 6):
        grid[2][column] = "$"
    for column in (2, 4, 6, goal_column):
        grid[5][column] = "."
    return ["".join(row) for row in grid]


def instance(index, goal_column):
    state = board(goal_column)
    return {"id": f"case-{index}", "family": "real_12x12x4", "state": state,
            "layout_id": canonical_layout_id(from_rows(state)), "source_path": "source",
            "source_sha256": "a" * 64, "source_index": index}


def test_test_selection_excludes_known_layouts_and_is_order_independent():
    items = [instance(i, col) for i, col in enumerate((7, 8, 9))]
    blocked = {items[0]["layout_id"]}
    result = select_test(items, blocked, 42, 2)
    assert len(result) == 2 and all(item["split"] == "test" for item in result)
    assert not blocked & {item["layout_id"] for item in result}
    assert result == select_test(list(reversed(items)), blocked, 42, 2)
    corrupt = copy.deepcopy(items)
    corrupt[0]["layout_id"] = "a" * 64
    with pytest.raises(ValueError, match="canonical layout"):
        select_test(corrupt, blocked, 42, 2)
