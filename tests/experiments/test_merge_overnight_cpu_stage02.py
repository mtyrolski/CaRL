"""Mixed pools retain all distinct proposals without a reachability filter."""
import copy

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts.merge_overnight_cpu_stage02 import combine_context
from scripts.prepare_overnight_cpu_stage02 import BFS_SLOTS
from scripts.prepare_overnight_cpu_stage02 import DEMO_SLOTS


def source_board():
    tiles = np.ones((12, 12), dtype=int)
    tiles[[0, -1], :] = 0
    tiles[:, [0, -1]] = 0
    tiles[2, 2:5] = 3
    tiles[5, 3], tiles[5, 4], tiles[5, 5] = 5, 4, 2
    return np.eye(7, dtype=np.float32)[tiles]


def base_records():
    source = source_board()
    common = {'context_id': 'ctx', 'layout_id': canonical_layout_id(source), 'family': '12x12x4',
              'source_family': 'fixture', 'split': 'development', 'board_rank': 0, 'context_rank': 0,
              'source': to_rows(source), 'source_hash': state_hash(source), 'in_fixed_pool': True,
              'pool_variant': 'demo4_bfs4'}
    records = []
    for index, slot in enumerate((*DEMO_SLOTS, *BFS_SLOTS[:4])):
        records.append({**common, 'candidate_id': f'candidate-{index}', 'target_hash': f'existing-{index}',
                        'candidate_origin': 'demonstration' if index < 4 else 'bfs', 'slot': slot})
    return records


def proposal(target, rank):
    return {'rank': rank, 'valid_nontrivial': True, 'validation_error': None,
            'target': to_rows(target), 'target_hash': state_hash(target), 'raw_token_ids': [rank]}


def test_generator_pool_keeps_other_valid_unique_proposals():
    source = source_board()
    env = environment(source)
    outputs = [proposal(np.asarray(env.next_state(source, 2)), 0),
               proposal(np.asarray(env.next_state(source, 3)), 1)]
    raw = {'context_id': 'ctx', 'source_hash': state_hash(source), 'outputs': outputs}
    combined, coverage = combine_context(base_records(), raw)
    generated = [row for row in combined if row['candidate_origin'] == 'generator']
    assert len(generated) == 2
    assert sum(row['in_generator_pool'] for row in combined) == 9
    assert [row['in_generator_pool'] for row in generated] == [True, False]
    assert all(row['bfs_depth'] is None and row['demo_offsets'] == [] for row in generated)
    assert coverage['pool_status'] == 'complete_demo4_bfs4_generator1'
    assert [row['status'] for row in coverage['raw_proposals']] == ['valid_unique', 'valid_unique']


def test_duplicates_and_invalid_proposals_remain_in_coverage():
    source = source_board()
    target = np.asarray(environment(source).next_state(source, 2))
    base = base_records()
    base[0]['target_hash'] = state_hash(target)
    invalid = {'rank': 1, 'valid_nontrivial': False, 'validation_error': 'wrong box count',
               'target': None, 'target_hash': None, 'raw_token_ids': [1]}
    raw = {'context_id': 'ctx', 'source_hash': state_hash(source),
           'outputs': [proposal(target, 0), invalid]}
    combined, coverage = combine_context(base, raw)
    assert len(combined) == 8 and not any(row['in_generator_pool'] for row in combined)
    assert coverage['pool_status'] == 'underfilled_generator'
    assert [row['status'] for row in coverage['raw_proposals']] == ['duplicate', 'invalid']
    assert coverage['raw_proposals'][0]['candidate_id'] == base[0]['candidate_id']
    assert combined[0]['also_generated'] and combined[0]['generator_ranks'] == [0]
    missing = copy.deepcopy(raw)
    missing['source_hash'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        combine_context(base, missing)
