"""Protect held-out exclusions and the validation source-only contract."""
import json
from pathlib import Path

import numpy as np
import pytest

from carl.icl.boards import from_rows, to_rows
from carl.icl.datasets import environment
from scripts.generator_probe import source_selection
from scripts.generator_probe.prepare_dataset import (
    VerifiedSource, selected_targets, source_record, training_context, validation_sources, write_once,
)
from scripts.generator_probe.source_selection import ExclusionSnapshot


def test_fresh_selection_excludes_protected_and_deduplicates_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh source selection must never load a protected or duplicate layout."""
    monkeypatch.setattr(source_selection, 'index_rows', lambda: [
        (0, 'protected', 'a', 30, 1), (1, 'duplicate', 'b', 30, 1),
        (2, 'duplicate', 'c', 30, 1), (3, 'independent', 'd', 30, 1),
        (4, 'short', 'e', 3, 1), (5, 'solved', 'f', 30, 0),
    ])
    snapshot = ExclusionSnapshot(
        version=1, selection_seed=source_selection.SEED, source_path=source_selection.PART1,
        source_sha256='source-digest', history_sha256='history-digest',
        public_manifest_sha256={}, observed_request_sha256={},
        excluded_layout_ids=['protected'], protected_layout_ids=['protected'],
    )
    selected = source_selection.fresh_candidates(snapshot)
    assert {row['layout_id'] for row in selected} == {'duplicate', 'independent'}
    assert len(selected) == 2
    assert selected == source_selection.fresh_candidates(snapshot)


def make_source(index: int) -> VerifiedSource:
    """Build one small legal training trajectory without a model or hidden target."""
    # Distinct corridor lengths make independently identifiable fixture layouts.
    width = 9 + index
    board = from_rows(['#' * width, '#@--$' + '-' * (width - 7) + '.#'] + ['#' * width] * (width - 2))
    env = environment(board)
    states = [board]
    actions = [3] * 4
    for action in actions:
        states.append(np.asarray(env.next_state(states[-1], action)))
    return VerifiedSource(source_record(index, states, 'source-digest'), states, actions)


def test_validation_export_cannot_contain_training_witness_or_future_target() -> None:
    """Enforce the whole exported field allowlist, not just known forbidden names."""
    sources = [make_source(index) for index in range(10)]
    exported = validation_sources(sources)
    permitted = {'id', 'board_id', 'trajectory_id', 'source_path', 'source_index',
                 'source_sha256', 'position', 'state', 'state_hash', 'layout_id'}
    assert len(exported) == 30
    assert all(set(row) == permitted for row in exported)
    assert all(row['state'] == to_rows(sources[row['source_index']].states[row['position']]) for row in exported)
    context = training_context(sources[0])
    assert [candidate['k'] for candidate in context['candidates']] == [1, 2, 4]
    assert all(set(candidate) == {'k', 'subgoal', 'actions'} for candidate in context['candidates'])
    for candidate in context['candidates']:
        assert candidate['actions'] == sources[0].actions[:candidate['k']]
        assert candidate['subgoal'] == to_rows(sources[0].states[candidate['k']])


def test_immutable_manifest_rejects_mismatched_rerun(tmp_path: Path) -> None:
    """A revised dataset must not silently replace an inspectable checkpoint input."""
    destination = tmp_path / 'manifest.json'
    write_once(destination, {'count': 30})
    write_once(destination, {'count': 30})
    with pytest.raises(ValueError, match='refusing to overwrite'):
        write_once(destination, {'count': 31})
    assert json.loads(destination.read_text()) == {'count': 30}


@pytest.mark.parametrize(('actions', 'offsets', 'source_equal_k'), [
    ([3, 3, 2, 2, 0], [1, 2, 5], 4),  # trajectory_138: a four-action loop
    ([3, 2, 0, 3], [1, 3, 4], 2),  # trajectory_682: a two-action loop
])
def test_training_loops_cannot_become_copy_demonstrations(
    actions: list[int], offsets: list[int], source_equal_k: int,
) -> None:
    """Real observed loop patterns must preserve actual k and reject current-state targets."""
    rows = ['#########'] + ['#-------#'] * 7 + ['#########']
    rows[4] = '#@---$-.#'
    board = from_rows(rows)
    env = environment(board)
    states = [board]
    for action in actions:
        states.append(np.asarray(env.next_state(states[-1], action)))
    source = VerifiedSource(source_record(0, states, 'source-digest'), states, actions)
    candidates, audit = selected_targets(source)
    assert [candidate['k'] for candidate in candidates] == offsets
    assert {'k': source_equal_k, 'reason': 'source_equal'} in audit['rejected_offsets']
    assert all(candidate['subgoal'] != rows for candidate in candidates)
    assert len({tuple(candidate['subgoal']) for candidate in candidates}) == 3
    for candidate in candidates:
        assert candidate['actions'] == actions[:candidate['k']]
        assert candidate['subgoal'] == to_rows(states[candidate['k']])
