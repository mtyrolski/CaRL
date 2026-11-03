"""Protect the historical reference cohort and missing/failure semantics."""
from pathlib import Path

import pytest

from carl.planners.base import Experience, SearchInfo, Solution
from scripts.trained_reference.common import selected_indices
from scripts.trained_reference.freeze import export_record, require_same_or_save


def test_selection_is_reproducible_without_outcomes() -> None:
    first = selected_indices(1000, 100, 20261102)
    assert first == selected_indices(1000, 100, 20261102)
    assert first == sorted(set(first)) and len(first) == 100
    assert first != selected_indices(1000, 100, 20261103)
    with pytest.raises(ValueError):
        selected_indices(10, 11, 20261102)


def test_frozen_selection_rejects_replacement(tmp_path: Path) -> None:
    path = tmp_path / 'boards.json'
    require_same_or_save(path, {'ids': [1, 2], 'per_k': {8: 3, 4: 2, 1: 0}})
    require_same_or_save(path, {'ids': [1, 2], 'per_k': {8: 3, 4: 2, 1: 0}})
    with pytest.raises(ValueError, match='Frozen artifact differs'):
        require_same_or_save(path, {'ids': [2, 3]})


def test_failed_historical_attempt_keeps_unknown_replay_and_runtime() -> None:
    record = Experience(Solution(solved=False), SearchInfo(
        finished_reason='budget_exceeded', low_level_nodes_visited=503,
        high_level_nodes_valid=4, high_level_nodes_unreachable=65,
        subgoals_selected_for_expansion={8: 68, 4: 1, 1: 0}))
    board = {'board_id': 'board', 'state_hash': 'hash', 'population_index': 4, 'state': ['#']}
    row, detail = export_record(record, Path('archive.joblib'), 'digest', 4, board, 500)
    assert row['solved'] is False and row['native_overshoot'] == 3
    assert row['solution_replayed'] is None and row['runtime_seconds'] is None
    assert detail['action_path'] is None and row['generator_calls'] == 69
