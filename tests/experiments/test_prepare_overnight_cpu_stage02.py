"""Checks for stage-two candidate identity, replay and split reservations."""

import argparse
import json

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from scripts.prepare_overnight_cpu_stage02 import BFS_SLOTS
from scripts.prepare_overnight_cpu_stage02 import PoolVariant
from scripts.prepare_overnight_cpu_stage02 import bfs_states
from scripts.prepare_overnight_cpu_stage02 import candidate_id
from scripts.prepare_overnight_cpu_stage02 import file_sha
from scripts.prepare_overnight_cpu_stage02 import path_to
from scripts.prepare_overnight_cpu_stage02 import prepare_context
from scripts.prepare_overnight_cpu_stage02 import replay
from scripts.prepare_overnight_cpu_stage02 import rows_for_state
from scripts.prepare_overnight_cpu_stage02 import run
from scripts.prepare_overnight_cpu_stage02 import select_contexts
from scripts.prepare_overnight_cpu_stage02 import select_pool_variant


def board() -> list[str]:
    cells = np.ones((12, 12), dtype=np.int8)
    cells[0, :] = cells[-1, :] = 0
    cells[:, 0] = cells[:, -1] = 0
    cells[1, 1:5] = 2
    cells[3, 1:5] = 4
    cells[5, 1] = 5
    return to_rows(np.eye(7, dtype=np.float32)[cells])


def pairs() -> list[dict]:
    source = board()
    initial = from_rows(source)
    actions = [3, 3, 3, 3] + [2, 3] * 14
    base = {'context_id': 'context-one', 'layout_id': canonical_layout_id(initial), 'family': '12x12x4',
            'source_family': 'fixture', 'split': 'development', 'board_rank': 0, 'context_rank': 0,
            'source': source, 'source_hash': state_hash(initial), 'source_path': 'fixture', 'source_sha256': 'sha',
            'trajectory_id': 'trajectory', 'i': 0, 'original_split': 'fixture', 'exposure_status': 'fixture'}
    result = []
    for offset in (1, 2, 4, 8, 16, 32):
        target = replay(source, actions[:offset])
        result.append({**base, 'pair_id': f'pair-{offset}', 'k': offset, 'target': target,
                       'target_hash': state_hash(from_rows(target)), 'demonstration_actions': actions[:offset]})
    return result


def test_bfs_paths_replay_and_count_pushes_with_one_shared_node_limit():
    source = board()
    nodes, info = bfs_states(source, max_depth=12, max_nodes=120)
    assert info['expanded_nodes'] <= 120
    assert info['edges_examined'] == 4 * info['expanded_nodes']
    assert info['status'] == 'node_limit'
    selected = [(state, node) for state, node in nodes.items() if node.depth]
    assert selected
    for state, node in selected[::max(1, len(selected) // 10)]:
        actions = path_to(nodes, state)
        assert len(actions) == node.depth
        assert replay(source, actions) == rows_for_state(source, state)
        pushes = 0
        previous = from_rows(source)
        for index in range(1, len(actions) + 1):
            current = from_rows(replay(source, actions[:index]))
            pushes += int(set(map(tuple, np.argwhere(np.isin(previous.argmax(-1), (3, 4))))) !=
                          set(map(tuple, np.argwhere(np.isin(current.argmax(-1), (3, 4))))))
            previous = current
        assert pushes == node.pushes


def test_demonstration_aliases_are_retained_and_fixed_slots_deduplicated():
    result = prepare_context(pairs(), max_depth=12, max_nodes=120, seed=42)
    context, records = result['context'], result['candidates']
    demo = [row for row in records if row['candidate_origin'] == 'demonstration']
    repeated = next(row for row in demo if 4 in row['demo_offsets'])
    assert repeated['demo_offsets'] == [4, 8, 16, 32]
    assert repeated['demo_pair_ids'] == ['pair-4', 'pair-8', 'pair-16', 'pair-32']
    assert context['pool_variant'] == 'underfilled'
    assert len(records) == len({row['target_hash'] for row in records})
    assert all(row['candidate_id'] == candidate_id(row['context_id'], row['target_hash']) for row in records)
    duplicate_offsets = [item['offset'] for item in context['demo_offset_availability']
                         if item['status'] == 'duplicate_target']
    assert duplicate_offsets == [16, 32, 8]
    demo_slots = context['slots'][:4]
    assert [slot['selected_offset'] for slot in demo_slots] == [1, 4, 2, None]
    assert demo_slots[-1]['reason'] == 'insufficient_distinct_demo_targets'
    for row in records:
        if row['candidate_origin'] == 'bfs':
            assert row['demo_offsets'] == row['demo_pair_ids'] == []
            assert replay(row['source'], row['bfs_actions']) == row['target']
            assert 'k' not in row


def test_missing_long_demo_offsets_are_filled_with_other_distinct_targets():
    rows = pairs()
    source = rows[0]['source']
    actions = [3] * 8
    for row in rows:
        if row['k'] in (1, 2, 4, 8):
            row['demonstration_actions'] = actions[:row['k']]
            row['target'] = replay(source, row['demonstration_actions'])
            row['target_hash'] = state_hash(from_rows(row['target']))
    rows = [row for row in rows if row['k'] in (1, 2, 4, 8)]
    result = prepare_context(rows, max_depth=12, max_nodes=120, seed=42)
    slots = result['context']['slots'][:4]
    assert [slot['slot'] for slot in slots] == ['demo_01', 'demo_02', 'demo_03', 'demo_04']
    assert [slot['selected_offset'] for slot in slots] == [1, 4, 8, 2]
    availability = {item['offset']: item['status'] for item in result['context']['demo_offset_availability']}
    assert availability[16] == availability[32] == 'not_available'


def test_pool_fallback_is_declared_and_does_not_use_policy_outcomes():
    assert select_pool_variant(4, set(BFS_SLOTS)) == (PoolVariant.COMPLETE, set(BFS_SLOTS))
    five = set(BFS_SLOTS[1:])
    assert select_pool_variant(4, five) == (PoolVariant.FALLBACK, set(BFS_SLOTS[1:5]))
    assert select_pool_variant(3, five) == (PoolVariant.UNDERFILLED, five)
    assert select_pool_variant(4, set(BFS_SLOTS[:3]))[0] == PoolVariant.UNDERFILLED
    with pytest.raises(ValueError, match='Unknown'):
        select_pool_variant(4, {'unknown'})


def test_recorded_paths_and_reservations_are_not_silently_ignored():
    rows = pairs()
    broken = [{**row} for row in rows]
    broken[0] = {**broken[0], 'demonstration_actions': [2]}
    with pytest.raises(ValueError, match='no-op'):
        prepare_context(broken, max_depth=12, max_nodes=20, seed=42)
    layout = rows[0]['layout_id']
    exposure = {'training_prohibited_layout_ids': [layout], 'campaign_evaluation_layout_ids': [layout]}
    with pytest.raises(ValueError, match='Protected'):
        select_contexts(rows, 'development', None, 1, exposure)
    evaluation = [{**row, 'split': 'evaluation'} for row in rows]
    assert len(select_contexts(evaluation, 'evaluation', None, 1, exposure)) == 1
    with pytest.raises(ValueError, match='reservation'):
        select_contexts(evaluation, 'evaluation', None, 1,
                        {'training_prohibited_layout_ids': [], 'campaign_evaluation_layout_ids': []})


def test_atomic_shards_resume_and_reject_changed_contract(tmp_path):
    source = tmp_path / 'input'
    source.mkdir()
    rows = pairs()
    (source / 'pairs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (source / 'exposure_register.json').write_text(json.dumps({'training_prohibited_layout_ids': [],
                                                              'campaign_evaluation_layout_ids': []}))
    hashes = {name: file_sha(source / name) for name in ('pairs.jsonl', 'exposure_register.json')}
    (source / 'data_manifest.json').write_text(json.dumps({'outputs_sha256': hashes}))
    args = argparse.Namespace(input_dir=source, output=tmp_path / 'output', split='development', max_boards=1,
                              max_contexts=1, max_depth=12, max_nodes=60, workers=1, seed=42, resume=False)
    first = run(args)
    assert first['state'] == 'complete' and first['completed_contexts'] == 1
    shard = args.output / 'shards' / 'context-one.json'
    assert shard.is_file()
    before = (args.output / 'candidates.jsonl').read_bytes()
    args.resume = True
    second = run(args)
    assert second['state'] == 'complete'
    assert (args.output / 'candidates.jsonl').read_bytes() == before
    args.max_nodes = 61
    with pytest.raises(ValueError, match='contract differs'):
        run(args)
