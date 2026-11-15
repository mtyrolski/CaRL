#!/usr/bin/env python3
"""Build matched Sokoban annotations from replayed, disjoint local layouts.

The canonical layout split is persisted before fragments are extracted. Both
wide arms use exactly the same original records; augmentation belongs to the
training sampler and never creates new layouts or extra optimiser steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import EffectType
from scripts.datasets.prepare_fresh_trajectory_holdout import verified_trajectory
from scripts.prepare_intent_generator_data import record
from scripts.prepare_intent_generator_data import sample
from scripts.prepare_overnight_cpu_data import check_board
from scripts.prepare_overnight_cpu_stage02 import atomic_json
from scripts.prepare_overnight_cpu_stage02 import atomic_jsonl
from scripts.prepare_overnight_cpu_stage02 import file_sha
from scripts.prepare_overnight_cpu_stage02 import read_jsonl
from scripts.prepare_overnight_cpu_stage02_train import INDEX_RELATIVE
from scripts.prepare_overnight_cpu_stage02_train import SOURCE_RELATIVE

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261115
WIDE_BOARDS = 64
NARROW_BOARDS = 9
CATEGORIES = (EffectType.BOX.value, EffectType.GOAL_CONJUNCTION.value,
              EffectType.BOX_CONJUNCTION.value, EffectType.WALK_ONLY.value)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def category_quotas(total: int, variant: int = 0) -> dict[str, int]:
    """Keep walk-only below 10%; adapt conjunction quotas only if needed."""
    walking = total * 9 // 100
    goal = max(0, total * (19 - variant * 2) // 100)
    box_conjunction = max(0, total * (16 - variant * 2) // 100)
    return {CATEGORIES[0]: total - walking - goal - box_conjunction,
            CATEGORIES[1]: goal, CATEGORIES[2]: box_conjunction, CATEGORIES[3]: walking}


def allocate(pools: dict[str, dict[str, list]], quotas: dict[str, int], cap: int) -> dict[str, list] | None:
    """Allocate category quotas with a per-board cap using integer max flow.

    One preallocated example per board guarantees that all 9/64 layouts
    contribute. Stable inputs and category ordering yield a stable selection.
    """
    if set(quotas) != set(CATEGORIES) or cap < 1 or any(n < 0 for n in quotas.values()):
        raise ValueError('Invalid allocation quotas or board cap')
    boards = sorted(pools)
    selected = {board: [] for board in boards}
    remaining = dict(quotas)
    buckets = {board: {kind: list(pools[board].get(kind, [])) for kind in CATEGORIES} for board in boards}
    for board in boards:
        chosen = next((kind for kind in CATEGORIES if buckets[board][kind] and remaining[kind] > 0), None)
        if chosen is None:
            return None
        selected[board].append(buckets[board][chosen].pop())
        remaining[chosen] -= 1
    source, sink = ('source', ''), ('sink', '')
    capacity: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    adjacency: dict[tuple[str, str], list[tuple[str, str]]] = {}

    def edge(a: tuple[str, str], b: tuple[str, str], n: int) -> None:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        capacity[a, b] = n
        capacity[b, a] = 0

    for kind in CATEGORIES:
        category = ('category', kind)
        edge(source, category, remaining[kind])
        for board in boards:
            edge(category, ('board', board), len(buckets[board][kind]))
    for board in boards:
        edge(('board', board), sink, cap - 1)
    residual = dict(capacity)
    flow = 0
    while True:
        parents: dict[tuple[str, str], tuple[str, str] | None] = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in parents and residual[node, neighbour] > 0:
                    parents[neighbour] = node
                    queue.append(neighbour)
        if sink not in parents:
            break
        amount = sum(remaining.values()) - flow
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            assert parent is not None
            amount = min(amount, residual[parent, node])
            node = parent
        node = sink
        while parents[node] is not None:
            parent = parents[node]
            assert parent is not None
            residual[parent, node] -= amount
            residual[node, parent] += amount
            node = parent
        flow += amount
    if flow != sum(remaining.values()):
        return None
    for kind in CATEGORIES:
        category = ('category', kind)
        for board in boards:
            node = ('board', board)
            number = capacity[category, node] - residual[category, node]
            selected[board].extend(buckets[board][kind][:number])
    if sum(map(len, selected.values())) != sum(quotas.values()):
        raise AssertionError('Allocation lost examples')
    return selected


def matching_allocations(narrow: dict[str, dict[str, list]], wide: dict[str, dict[str, list]]) -> tuple:
    """Largest declared budget with identical category quotas in both arms."""
    for total in (256, 224, 192, 160, 128):
        for variant in range(7):
            quotas = category_quotas(total, variant)
            first, second = allocate(narrow, quotas, 32), allocate(wide, quotas, 4)
            if first is not None and second is not None:
                return total, quotas, first, second
    raise ValueError('No matched category allocation with all layouts represented')


def replay_saved(row: dict[str, Any]) -> tuple[list[list[str]], list[int]]:
    """Replay all recorded primitive actions and validate every saved hash."""
    state = from_rows(row['initial_state'])
    if check_board(state, 4) != row['layout_id']:
        raise ValueError('Saved development board layout differs')
    env = environment(state)
    actions = row['actions']
    expected = row['state_hashes']
    if len(expected) != len(actions) + 1 or state_hash(state) != expected[0]:
        raise ValueError('Saved development trajectory state count or hash differs')
    states = [to_rows(state)]
    for index, action in enumerate(actions):
        if isinstance(action, bool) or action not in range(4) or env.is_solved(state):
            raise ValueError('Invalid action or action after completion')
        next_state = np.asarray(env.next_state(state, action))
        if np.array_equal(state, next_state) or state_hash(next_state) != expected[index + 1]:
            raise ValueError('Saved development transition fails replay')
        state = next_state
        states.append(to_rows(state))
    return states, actions


def candidate_pools(states_by_layout: dict[str, tuple[list[list[str]], list[int], dict]], seed: int) -> dict:
    pools = {}
    for layout, (states, _actions, _metadata) in sorted(states_by_layout.items()):
        derived_seed = int(digest([seed, layout])[:16], 16)
        groups = {kind: [] for kind in CATEGORIES}
        for item in sample(states, cap=160, seed=derived_seed):
            groups[item[2]].append(item)
        pools[layout] = groups
    return pools


def materialise(allocations: dict[str, list], trajectories: dict, split: str) -> list[dict]:
    result = []
    for layout, items in sorted(allocations.items()):
        states, actions, metadata = trajectories[layout]
        for i, k, kind, intent in items:
            row = record(metadata, split, states, actions, i, k, kind, intent,
                         {'kind': 'exact_local_trajectory_difference', 'teacher_model': None})
            row['record_id'] = digest(['stage04-intent-v1', layout, metadata['id'], i, k, intent])[:24]
            result.append(row)
    result.sort(key=lambda row: (row['layout_id'], row['record_id']))
    if len({row['record_id'] for row in result}) != len(result):
        raise ValueError('Duplicate annotation identity')
    return result


def describe_marginals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report residual differences beyond the matched intent categories."""
    predicates: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    goal_counts: Counter[str] = Counter()
    coordinates: Counter[str] = Counter()
    for row in rows:
        effects = row['intent']['effects']
        sizes[str(len(effects))] += 1
        for item in effects:
            name = item['predicate']
            predicates[name] += 1
            if name == 'boxes_on_goals_at_least':
                goal_counts[str(item['count'])] += 1
            if name in ('box_at', 'player_at'):
                coordinates[f"{item['row']},{item['col']}"] += 1
    return {'predicate_counts': dict(predicates), 'conjunction_sizes': dict(sizes),
            'goal_threshold_counts': dict(goal_counts), 'coordinate_counts': dict(coordinates)}


def prepare(run: Path, output: Path, seed: int) -> dict[str, Any]:
    stage1 = run / 'stage_01/data_v2'
    stage2 = run / 'stage_02/data_train'
    old = run / 'stage_03/data/contexts.jsonl'
    manifest1 = json.loads((stage1 / 'data_manifest.json').read_text())
    manifest2 = json.loads((stage2 / 'data_manifest.json').read_text())
    for path, expected in ((stage1 / 'trajectories.jsonl', manifest1['outputs_sha256']['trajectories.jsonl']),
                           (stage1 / 'exposure_register.json', manifest1['outputs_sha256']['exposure_register.json']),
                           (stage2 / 'trajectories.jsonl', manifest2['output_sha256']['trajectories.jsonl']),
                           (stage2 / 'exposure_register.json', manifest2['output_sha256']['exposure_register.json'])):
        if file_sha(path) != expected:
            raise ValueError(f'Frozen input hash differs: {path}')
    source, index = ROOT / SOURCE_RELATIVE, ROOT / INDEX_RELATIVE
    source_sha = file_sha(source)
    if source_sha != manifest2['contract']['source_sha256']:
        raise ValueError('Trusted trajectory shard changed since stage 2')
    exposure = json.loads((stage1 / 'exposure_register.json').read_text())
    old_train = read_jsonl(stage2 / 'trajectories.jsonl')
    stage1_rows = read_jsonl(stage1 / 'trajectories.jsonl')
    protected = (set(exposure['training_prohibited_layout_ids']) | set(exposure['observed_layout_ids'])
                 | {row['layout_id'] for row in stage1_rows} | {row['layout_id'] for row in read_jsonl(old)})
    existing = {int(row['trajectory_id'].rsplit('_', 1)[-1]): row['layout_id'] for row in old_train}
    if len(existing) != len(old_train) or set(existing.values()) & protected:
        raise ValueError('Previously assigned training layouts overlap protected data')
    with sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True) as db:
        source_entry = db.execute('SELECT sha256 FROM shards WHERE path=?', (SOURCE_RELATIVE,)).fetchone()
        indexed = db.execute('SELECT trajectory_index,layout_id,trajectory_states,unsolved FROM boards WHERE path=?',
                             (SOURCE_RELATIVE,)).fetchall()
    if source_entry is None or source_entry[0] != source_sha:
        raise ValueError('Trusted shard index is stale')
    candidates = [(int(i), layout) for i, layout, length, unsolved in indexed
                  if layout not in protected and length > 1 and unsolved and i not in existing]
    candidates.sort()
    random.Random(seed).shuffle(candidates)
    ordered = sorted(existing.items()) + candidates
    if len({layout for _index, layout in ordered}) != len(ordered):
        raise ValueError('Indexed source contains duplicated layouts')
    development = [row for row in stage1_rows if row['split'] == 'development' and row['family'] == '12x12x4']
    validation_ids = sorted(row['layout_id'] for row in development)
    evaluation_ids = sorted(exposure['campaign_evaluation_layout_ids'])
    if len(validation_ids) != 10 or len(set(validation_ids)) != 10:
        raise ValueError('Expected ten unique stage-one x4 development layouts')
    output.mkdir(parents=True, exist_ok=False)
    candidate_file = output / 'candidate_order.json'
    atomic_json(candidate_file, {'source_path': SOURCE_RELATIVE, 'source_sha256': source_sha,
                                 'seed': seed, 'candidate_order': ordered,
                                 'rule': 'existing stage-2 train, then seeded D4-disjoint indices; before fragment extraction'})
    raw = joblib.load(source)
    raw = list(raw.values()) if isinstance(raw, dict) else list(raw)
    trajectories: dict[str, tuple[list[list[str]], list[int], dict]] = {}
    accepted = []
    rejected = []
    for index_number, layout in ordered:
        try:
            states, actions = verified_trajectory(np.asarray(raw[index_number]))
            if check_board(states[0], 4) != layout or len(actions) < 2:
                raise ValueError('Invalid source layout or short trajectory')
            metadata = {'id': f'shard3_trajectory_{index_number}', 'layout_id': layout,
                        'path': SOURCE_RELATIVE, 'sha256': source_sha}
            trajectories[layout] = ([to_rows(state) for state in states], actions, metadata)
            accepted.append({'trajectory_index': index_number, 'layout_id': layout,
                             'previous_stage02_train': index_number in existing,
                             'primitive_actions': len(actions)})
            if len(accepted) == WIDE_BOARDS:
                break
        except (IndexError, TypeError, ValueError) as error:
            rejected.append({'trajectory_index': index_number, 'layout_id': layout,
                             'reason': f'{type(error).__name__}: {error}'})
    if len(accepted) != WIDE_BOARDS:
        atomic_jsonl(output / 'rejected_trajectories.jsonl', rejected)
        raise ValueError(f'Only {len(accepted)} valid independent training layouts')
    narrow_ids = sorted(random.Random(seed + 1).sample([row['layout_id'] for row in accepted], NARROW_BOARDS))
    train_ids = sorted(trajectories)
    if set(train_ids) & (set(validation_ids) | set(evaluation_ids) | protected):
        raise ValueError('Training layouts intersect development, evaluation, or protected data')
    split = {'seed': seed, 'canonicalization': 'carl-layout-d4-v1', 'source_sha256': source_sha,
             'train': accepted, 'narrow_train_layout_ids': narrow_ids, 'wide_train_layout_ids': train_ids,
             'validation_layout_ids': validation_ids, 'evaluation_layout_ids': evaluation_ids,
             'candidate_order_sha256': file_sha(candidate_file),
             'selection': 'Verified trajectories only, before annotation extraction; narrow is a subset of wide.',
             'historical_exposure': 'protected observed layouts excluded; old model/checkpoint pretraining exposure unknown.'}
    atomic_json(output / 'split_manifest.json', split)
    validation = {}
    for row in development:
        states, actions = replay_saved(row)
        metadata = {'id': row['trajectory_id'], 'layout_id': row['layout_id'],
                    'path': row['source_path'], 'sha256': row['source_sha256']}
        validation[row['layout_id']] = (states, actions, metadata)
    pools = candidate_pools(trajectories, seed)
    count, quotas, narrow, wide = matching_allocations({key: pools[key] for key in narrow_ids}, pools)
    narrow_rows = materialise(narrow, trajectories, 'train')
    wide_rows = materialise(wide, trajectories, 'train')
    val_pools = candidate_pools(validation, seed + 2)
    validation_caps = {CATEGORIES[0]: 5, CATEGORIES[1]: 2, CATEGORIES[2]: 2, CATEGORIES[3]: 1}
    validation_selection = {layout: [item for kind in CATEGORIES for item in groups[kind][:validation_caps[kind]]]
                            for layout, groups in val_pools.items()}
    validation_rows = materialise(validation_selection, validation, 'validation')
    if not validation_rows or {row['layout_id'] for row in validation_rows} != set(validation_ids):
        raise ValueError('Validation labels do not cover the ten frozen layouts')
    if sum(row['effect_type'] == EffectType.WALK_ONLY for row in validation_rows) > len(validation_rows) // 10:
        raise ValueError('Validation walking-only labels exceed ten percent')
    trajectory_records = [{'split': split_name, 'layout_id': layout, 'trajectory_id': metadata['id'],
                           'source_path': metadata['path'], 'source_sha256': metadata['sha256'],
                           'states': states, 'actions': actions}
                          for split_name, source in (('train', trajectories), ('validation', validation))
                          for layout, (states, actions, metadata) in sorted(source.items())]
    for name, rows in (('narrow.jsonl', narrow_rows), ('wide.jsonl', wide_rows),
                       ('validation.jsonl', validation_rows), ('dataset_narrow.jsonl', narrow_rows + validation_rows),
                       ('dataset_wide.jsonl', wide_rows + validation_rows), ('trajectories.jsonl', trajectory_records),
                       ('rejected_trajectories.jsonl', rejected)):
        atomic_jsonl(output / name, rows)
    distributions = {name: dict(Counter(row['effect_type'] for row in rows))
                     for name, rows in (('narrow', narrow_rows), ('wide', wide_rows), ('validation', validation_rows))}
    if distributions['narrow'] != distributions['wide'] or len(narrow_rows) != count or len(wide_rows) != count:
        raise AssertionError('Matched arm budgets or predicate marginals differ')
    output_names = ('candidate_order.json', 'split_manifest.json', 'narrow.jsonl', 'wide.jsonl',
                    'validation.jsonl', 'dataset_narrow.jsonl', 'dataset_wide.jsonl', 'trajectories.jsonl',
                    'rejected_trajectories.jsonl')
    result = {'schema_version': 1, 'state': 'complete', 'seed': seed,
              'splits': {'train_layout_ids': train_ids, 'validation_layout_ids': validation_ids,
                         'evaluation_layout_ids': evaluation_ids},
              'arms': {'narrow': {'train_layout_ids': narrow_ids, 'examples': len(narrow_rows), 'per_board_cap': 32},
                       'wide': {'train_layout_ids': train_ids, 'examples': len(wide_rows), 'per_board_cap': 4},
                       'wide_aug': {'original_dataset': 'dataset_wide.jsonl', 'same_originals_as': 'wide',
                                    'augmentation': 'sample one valid D4 transform during training, no extra updates'}},
              'predicate_distributions': distributions, 'target_quotas': quotas,
              'detailed_marginals': {name: describe_marginals(rows) for name, rows in
                                     (('narrow', narrow_rows), ('wide', wide_rows), ('validation', validation_rows))},
              'validation_examples': len(validation_rows), 'rejected_trajectories': len(rejected),
              'source_sha256': source_sha, 'original_split': 'unknown; train alias is byte-identical to validation shard',
              'adaptation_training_families': ['12x12x4'],
              'x10_exposure': 'No x10 adaptation records; base-model and historical checkpoint exposure unknown.',
              'old_pretraining_exposure': 'unknown',
              'input_fields': ['start_observation', 'task_goal'],
              'evidence_only_fields': ['later_observation', 'fragment_actions', 'i', 'k'],
              'output_sha256': {name: file_sha(output / name) for name in output_names},
              'script_sha256': file_sha(Path(__file__)), 'argv': sys.argv}
    atomic_json(output / 'dataset_manifest.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    result = prepare(args.campaign.resolve(), args.output.resolve(), args.seed)
    print(json.dumps({'state': result['state'], 'arms': result['arms'],
                      'validation_examples': result['validation_examples'],
                      'predicate_distributions': result['predicate_distributions']}), flush=True)


if __name__ == '__main__':
    main()
