"""The expanded evaluation must preserve frozen contexts and layout splits."""

import pytest

from scripts.audit_representation_data import audit_splits
from scripts.audit_representation_data import independent_d4
from scripts.prepare_representation_data import EvaluationFamily
from scripts.prepare_representation_data import checked_solution
from scripts.prepare_representation_data import frozen_order
from scripts.prepare_representation_data import validate_groups


def item(identifier, family, split="test", layout_id=None):
    return {"record_id": identifier, "layout_id": layout_id or identifier,
            "split": split, "evaluation_family": family}


def test_previous_evaluation_order_is_unchanged_and_new_families_interleave():
    rows = [item("old-x4", EvaluationFamily.REAL), item("old-x10", EvaluationFamily.PROPOSAL),
            item("new-b", EvaluationFamily.REAL), item("new-a", EvaluationFamily.REAL),
            item("new-gen", EvaluationFamily.GENERATED)]
    assert frozen_order(rows, ["old-x4", "old-x10"]) == ["old-x4", "old-x10", "new-a", "new-gen", "new-b"]


def test_order_rejects_missing_duplicate_or_unknown_contexts():
    rows = [item("known", EvaluationFamily.REAL)]
    with pytest.raises(ValueError, match="unique and present"):
        frozen_order(rows, ["missing"])
    with pytest.raises(ValueError, match="unique and present"):
        frozen_order(rows, ["known", "known"])
    with pytest.raises(ValueError, match="Unexpected"):
        frozen_order(rows + [item("unknown", "another-family")], ["known"])


def test_split_disjointness_and_one_test_context_per_layout():
    rows = [item("train-a", EvaluationFamily.REAL, split="train", layout_id="training"),
            item("train-b", EvaluationFamily.REAL, split="train", layout_id="training"),
            item("test-a", EvaluationFamily.REAL, layout_id="test-layout")]
    validate_groups(rows)
    with pytest.raises(ValueError, match="Overlapping layouts"):
        validate_groups(rows + [item("leak", EvaluationFamily.REAL, layout_id="training")])
    with pytest.raises(ValueError, match="one test context"):
        validate_groups(rows + [item("duplicate-layout", EvaluationFamily.REAL, layout_id="test-layout")])
    with pytest.raises(ValueError, match="Duplicate record"):
        validate_groups(rows + [item("test-a", EvaluationFamily.REAL)])


def test_solution_rejects_ambiguous_action_types_before_replay():
    with pytest.raises(ValueError, match="invalid primitive"):
        checked_solution([], [True])
    with pytest.raises(ValueError, match="invalid primitive"):
        checked_solution([], [4])


def test_independent_d4_identifies_rotations_and_reflections():
    board = ["#####", "#@$.#", "#---#", "#####", "#####"]
    reflected = [row[::-1] for row in board]
    rotated = ["".join(board[len(board) - 1 - col][row] for col in range(len(board)))
               for row in range(len(board))]
    assert independent_d4(board) == independent_d4(reflected) == independent_d4(rotated)
    changed_goal = ["#####", "#@..#", "#-$-#", "#####", "#####"]
    assert independent_d4(board) != independent_d4(changed_goal)


def test_independent_audit_rejects_rotated_split_leak():
    board = ["#####", "#@$.#", "#---#", "#####", "#####"]
    rows = [{"record_id": "a", "layout_id": "same", "split": "train", "start_observation": board},
            {"record_id": "b", "layout_id": "same", "split": "test",
             "start_observation": [row[::-1] for row in board]},
            {"record_id": "c", "layout_id": "different", "split": "validation",
             "start_observation": ["#####", "#@$.#", "#-.-#", "#####", "#####"]}]
    with pytest.raises(ValueError, match="overlap"):
        audit_splits(rows)
