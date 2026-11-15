"""Coverage for frozen overnight board grouping and target availability."""

import json

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import state_hash
from scripts.prepare_overnight_cpu_data import OFFSETS
from scripts.prepare_overnight_cpu_data import check_board
from scripts.prepare_overnight_cpu_data import prepare
from scripts.prepare_overnight_cpu_data import select_context_indices
from scripts.prepare_overnight_cpu_data import to_records


def state(position: int) -> np.ndarray:
    cells = np.ones((12, 12), dtype=np.int8)
    cells[0, :] = 0
    cells[-1, :] = 0
    cells[:, 0] = 0
    cells[:, -1] = 0
    cells[1, 1:5] = 2
    cells[3, 1:5] = 4
    row, col = divmod(position, 10)
    cells[5 + row, 1 + col] = 5
    return np.eye(7, dtype=np.float32)[cells]


def test_canonical_layout_groups_rotations_and_validates_box_count():
    original = state(0)
    assert check_board(original, 4) == canonical_layout_id(np.rot90(original))
    with pytest.raises(ValueError, match='box, goal or player'):
        check_board(original, 10)
    invalid = original.copy()
    invalid[2, 2, :] = 0
    with pytest.raises(ValueError, match='one-hot'):
        check_board(invalid, 4)


def test_contexts_are_deterministic_and_distinct_even_after_a_cycle():
    trajectory = [state(0), state(1), state(0), state(2)]
    layout = canonical_layout_id(trajectory[0])
    a = select_context_indices(trajectory, layout, 42, 2)
    assert a == select_context_indices(trajectory, layout, 42, 2)
    assert len({state_hash(trajectory[i]) for i in a}) == 2
    assert all(i < len(trajectory) - 1 for i in a)


def test_contexts_prefer_two_starts_with_nontrivial_k32_targets():
    trajectory = [state(i) for i in range(45)]
    layout = canonical_layout_id(trajectory[0])
    selected = select_context_indices(trajectory, layout, 42, 2)
    assert len(selected) == 2
    assert all(i + 32 < len(trajectory) and state_hash(trajectory[i]) != state_hash(trajectory[i + 32])
               for i in selected)


def test_one_available_k32_start_is_filled_by_a_shorter_context():
    trajectory = [state(i) for i in range(33)]
    layout = canonical_layout_id(trajectory[0])
    selected = select_context_indices(trajectory, layout, 42, 2)
    assert selected[0] == 0
    assert len(selected) == 2 and selected[1] > 0
    assert len({state_hash(trajectory[i]) for i in selected}) == 2


def test_k32_return_to_start_is_not_preferred():
    trajectory = [state(i) for i in range(32)] + [state(0)]
    layout = canonical_layout_id(trajectory[0])
    selected = select_context_indices(trajectory, layout, 42, 2)
    assert len(selected) == 2
    assert all(i + 32 >= len(trajectory) or state_hash(trajectory[i]) == state_hash(trajectory[i + 32])
               for i in selected)


def test_short_trajectory_and_missing_second_context_are_retained():
    trajectory = [state(0), state(1)]
    layout = canonical_layout_id(trajectory[0])
    assert select_context_indices(trajectory, layout, 9, 2) == [0]
    board, pairs, unavailable = to_records(
        {'id': 'tiny'}, trajectory, [3], split='evaluation', family='12x12x4',
        source_family='synthetic_test_only', source_path='trusted', source_hash='abc',
        original_split='test', exposure_status='test', layout=layout, seed=9, contexts=2, board_rank=0,
    )
    assert board['available_contexts'] == 1
    assert board['training_eligibility'] == 'prohibited_evaluation'
    assert [item['k'] for item in pairs] == [1]
    assert {item['k'] for item in unavailable if item['reason'] == 'beyond_trajectory_end'} == set(OFFSETS[1:])
    assert any(item['k'] is None and item['reason'] == 'no_additional_distinct_context' for item in unavailable)


def test_return_to_start_records_unavailable_target_instead_of_a_free_success():
    trajectory = [state(0), state(1), state(0)]
    layout = canonical_layout_id(trajectory[0])
    _, pairs, unavailable = to_records(
        {'id': 'cycle'}, trajectory, [3, 2], split='development', family='12x12x4',
        source_family='synthetic_test_only', source_path='trusted', source_hash='abc',
        original_split='test', exposure_status='test', layout=layout, seed=1, contexts=2, board_rank=0,
    )
    assert all(item['source_hash'] != item['target_hash'] for item in pairs)
    assert any(item['k'] == 2 and item['reason'] == 'target_already_satisfied' for item in unavailable)
    assert all(item['demonstration_actions'] == [3, 2][item['i']:item['i'] + item['k']] for item in pairs)


def test_amendment_rejects_changed_previously_frozen_artifacts(tmp_path):
    (tmp_path / 'data_manifest.json').write_text('{}')
    (tmp_path / 'preparation_v1.py').write_text('# old source')
    amendment = tmp_path / 'context_selection_amendment.json'
    amendment.write_text(json.dumps({'prior_data_manifest_sha256': 'incorrect',
                                     'prior_preparation_script_sha256': 'incorrect'}))
    with pytest.raises(ValueError, match='pre-amendment artifacts'):
        prepare(tmp_path / 'new', eval4=1, development=1, contexts=2, seed=3, amendment=amendment)
    assert not (tmp_path / 'new').exists()
