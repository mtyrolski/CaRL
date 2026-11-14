"""Pair only common contexts and treat layouts as bootstrap units."""

import pytest

from scripts.paired_lora_contrasts import compare
from scripts.paired_lora_contrasts import plot_rows


def row(variant, identifier, layout, valid, *, family="x4", status="evaluated"):
    return {"variant": variant, "record_id": identifier, "layout_id": layout, "family": family,
            "status": status, "strict_valid": valid, "fence_valid": valid}


def test_missing_contexts_do_not_become_failures_or_pairs():
    rows = [row("r4", "a", "board1", False), row("r8", "a", "board1", True),
            row("r4", "b", "board2", True), row("r8", "b", "board2", None, status="missing")]
    result = compare(rows, "r4", "r8", "strict_valid")
    assert result["planned_paired_contexts"] == 2 and result["paired_contexts"] == 1
    assert result["paired_boards"] == 1 and result["board_mean_delta"] == 1
    assert result["missing_paired_contexts"] == 1 and result["bootstrap_low"] is None


def test_contexts_from_the_same_layout_receive_one_board_weight():
    rows = [row("r4", "a", "board1", False), row("r8", "a", "board1", True),
            row("r4", "b", "board1", False), row("r8", "b", "board1", True),
            row("r4", "c", "board2", True), row("r8", "c", "board2", False)]
    result = compare(rows, "r4", "r8", "strict_valid")
    assert result["paired_contexts"] == 3 and result["paired_boards"] == 2
    assert result["board_mean_delta"] == 0


def test_bootstrap_is_reproducible_and_family_is_filtered():
    rows = []
    for index in range(6):
        rows.extend([row("r4", str(index), str(index), False), row("r8", str(index), str(index), True)])
    rows.extend([row("r4", "other", "other", True, family="x10"),
                 row("r8", "other", "other", False, family="x10")])
    result = compare(rows, "r4", "r8", "strict_valid", "x4")
    assert result["paired_boards"] == 6 and result["board_mean_delta"] == 1
    assert result["bootstrap_low"] == result["bootstrap_high"] == 1
    assert result == compare(rows, "r4", "r8", "strict_valid", "x4")


def test_conflicting_layouts_are_rejected():
    with pytest.raises(ValueError, match="conflicting board"):
        compare([row("r4", "a", "board1", False), row("r8", "a", "board2", True)], "r4", "r8", "strict_valid")


def test_plot_hides_duplicate_aggregate_only_for_one_family():
    rows = [{"metric": "strict_valid", "family": "all"}, {"metric": "strict_valid", "family": "x10"},
            {"metric": "fence_valid", "family": "all"}]
    assert plot_rows(rows) == [rows[1]] and len(rows) == 3
    rows.append({"metric": "strict_valid", "family": "x4"})
    assert plot_rows(rows) == [rows[0], rows[1], rows[3]]
