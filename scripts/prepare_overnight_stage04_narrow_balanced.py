#!/usr/bin/env python3
"""Match narrow board weights to wide using the frozen training layouts.

Selection uses only the prior checked training trajectories, before test
evaluation. It does not change the sampler or either wide dataset.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import augmented_training_samples
from scripts.prepare_overnight_cpu_stage02 import atomic_json
from scripts.prepare_overnight_cpu_stage02 import atomic_jsonl
from scripts.prepare_overnight_cpu_stage02 import file_sha
from scripts.prepare_overnight_cpu_stage02 import read_jsonl
from scripts.prepare_overnight_cpu_stage04_data import CATEGORIES
from scripts.prepare_overnight_cpu_stage04_data import candidate_pools
from scripts.prepare_overnight_cpu_stage04_data import describe_marginals
from scripts.prepare_overnight_cpu_stage04_data import materialise

TRAINING_SEED = 1729
SAMPLE_COUNT = 600
QUOTAS = {'box': 145, 'goal_conjunction': 48, 'box_conjunction': 40, 'walk_only': 23}
COPIED = ('candidate_order.json', 'split_manifest.json', 'wide.jsonl', 'validation.jsonl',
          'dataset_wide.jsonl', 'trajectories.jsonl', 'rejected_trajectories.jsonl')


def select_exact(pools: dict[str, dict[str, list]], quotas: dict[str, int]) -> tuple[dict[str, list], list[str]]:
    """Use deterministic integer flow to assign 28/29 records per board."""
    boards = sorted(pools)
    if len(boards) != 9 or quotas != QUOTAS or sum(quotas.values()) != 256:
        raise ValueError('Expected nine layouts and the frozen 256-example quotas')
    source, sink = ('source', ''), ('sink', '')
    for extra in itertools.combinations(boards, 4):
        adjacency: dict[tuple[str, str], list[tuple[str, str]]] = {}
        capacity: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}

        def add_edge(first: tuple[str, str], second: tuple[str, str], number: int) -> None:
            adjacency.setdefault(first, []).append(second)
            adjacency.setdefault(second, []).append(first)
            capacity[first, second] = number
            capacity[second, first] = 0

        for kind in CATEGORIES:
            category = ('category', kind)
            add_edge(source, category, quotas[kind])
            for board in boards:
                add_edge(category, ('board', board), len(pools[board].get(kind, [])))
        for board in boards:
            add_edge(('board', board), sink, 29 if board in extra else 28)
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
            amount = sum(quotas.values()) - flow
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
        if flow != sum(quotas.values()):
            continue
        result = {}
        for board in boards:
            items = []
            for kind in CATEGORIES:
                category, node = ('category', kind), ('board', board)
                count = capacity[category, node] - residual[category, node]
                items.extend(pools[board].get(kind, [])[:count])
            result[board] = items
        if any(len(items) not in (28, 29) for items in result.values()):
            raise AssertionError('Flow did not fill every board')
        return result, list(extra)
    raise ValueError('No exact category allocation with 28/29 examples per board')


def sampled_categories(rows: list[dict[str, Any]], seed: int = TRAINING_SEED,
                       count: int = SAMPLE_COUNT) -> dict[str, int]:
    """Measure realised train-only sampling before any test evaluation."""
    stream = augmented_training_samples(rows, seed, 'none')
    return dict(Counter(next(stream)[0]['effect_type'] for _ in range(count)))


def copy_atomic(source: Path, target: Path) -> None:
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(source: Path, output: Path) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError('A new output directory is required')
    previous = json.loads((source / 'dataset_manifest.json').read_text())
    if previous.get('state') != 'complete':
        raise ValueError('Source dataset is incomplete')
    for name, expected in previous['output_sha256'].items():
        if file_sha(source / name) != expected:
            raise ValueError(f'Frozen source file changed: {name}')
    layouts = sorted(previous['arms']['narrow']['train_layout_ids'])
    if (len(layouts) != 9 or not set(layouts) <= set(previous['arms']['wide']['train_layout_ids'])
            or set(layouts) & (set(previous['splits']['validation_layout_ids']) |
                               set(previous['splits']['evaluation_layout_ids']))):
        raise ValueError('Frozen narrow layouts violate the original split')
    if previous['target_quotas'] != QUOTAS:
        raise ValueError('Source predicate quotas differ from the frozen comparison')
    output.mkdir(parents=True, exist_ok=False)
    contract = {'source_dataset_manifest_sha256': file_sha(source / 'dataset_manifest.json'),
                'source_trajectories_sha256': file_sha(source / 'trajectories.jsonl'),
                'source_narrow_sha256': file_sha(source / 'narrow.jsonl'),
                'original_nine_layouts': layouts, 'predicate_quotas': QUOTAS,
                'selection': 'First feasible max-flow over deterministic 28/29 per-board caps; '
                             'only previously assigned training trajectories; same train/validation/eval split.',
                'reason': 'The first narrow run drew 97/600 walking-only records despite a matched dataset quota. '
                          'This train-only correction was chosen before stage-four test outputs were opened.',
                'sampler': 'unchanged board_balanced, seed 1729; 600 samples for diagnostic only',
                'wide_and_validation': 'bit-identical copies from data_v2'}
    atomic_json(output / 'selection_contract.json', contract)
    trajectories = {}
    for row in read_jsonl(source / 'trajectories.jsonl'):
        if row['layout_id'] not in layouts:
            continue
        if row['split'] != 'train' or row['layout_id'] in trajectories:
            raise ValueError('Invalid frozen narrow trajectory identity')
        metadata = {'id': row['trajectory_id'], 'layout_id': row['layout_id'],
                    'path': row['source_path'], 'sha256': row['source_sha256']}
        trajectories[row['layout_id']] = (row['states'], row['actions'], metadata)
    if sorted(trajectories) != layouts:
        raise ValueError('The frozen nine training trajectories are incomplete')
    pool = candidate_pools(trajectories, previous['seed'])
    allocation, extra = select_exact(pool, QUOTAS)
    narrow = materialise(allocation, trajectories, 'train')
    old_narrow = read_jsonl(source / 'narrow.jsonl')
    wide = read_jsonl(source / 'wide.jsonl')
    validation = read_jsonl(source / 'validation.jsonl')
    if (len(narrow) != 256 or Counter(row['effect_type'] for row in narrow) != Counter(QUOTAS)
            or {row['layout_id'] for row in narrow} != set(layouts)):
        raise AssertionError('Selected examples do not match the frozen quotas and layouts')
    realised = {'original_narrow': sampled_categories(old_narrow), 'balanced_narrow': sampled_categories(narrow),
                'wide': sampled_categories(wide)}
    for name in COPIED:
        copy_atomic(source / name, output / name)
        if file_sha(source / name) != file_sha(output / name):
            raise AssertionError(f'Byte-identical retained dataset changed: {name}')
    atomic_jsonl(output / 'narrow.jsonl', narrow)
    atomic_jsonl(output / 'dataset_narrow.jsonl', narrow + validation)
    manifest = copy.deepcopy(previous)
    manifest['arms']['narrow'].update({'examples': 256, 'per_board_cap': 29,
                                       'selection': 'balanced_narrow_v3_28_or_29_per_layout'})
    manifest['predicate_distributions']['narrow'] = dict(Counter(row['effect_type'] for row in narrow))
    manifest['detailed_marginals']['narrow'] = describe_marginals(narrow)
    manifest['narrow_sampling_correction'] = {
        'selection_contract_sha256': file_sha(output / 'selection_contract.json'),
        'source_dataset_manifest_sha256': file_sha(source / 'dataset_manifest.json'),
        'original_narrow_sha256': file_sha(source / 'narrow.jsonl'),
        'wide_sha256_unchanged': file_sha(output / 'wide.jsonl'),
        'dataset_wide_sha256_unchanged': file_sha(output / 'dataset_wide.jsonl'),
        'validation_sha256_unchanged': file_sha(output / 'validation.jsonl'),
        'layouts_with_29_records': extra, 'realised_600_samples': realised,
        'new_vs_original_narrow_record_overlap': len({row['record_id'] for row in narrow} &
                                                     {row['record_id'] for row in old_narrow}),
        'new_narrow_vs_wide_record_overlap': len({row['record_id'] for row in narrow} &
                                                {row['record_id'] for row in wide}),
    }
    manifest['output_sha256'] = {name: file_sha(output / name) for name in
                                 (*COPIED, 'narrow.jsonl', 'dataset_narrow.jsonl', 'selection_contract.json')}
    manifest['script_sha256'] = file_sha(Path(__file__))
    manifest['argv'] = sys.argv
    atomic_json(output / 'dataset_manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output)
    print(json.dumps({'state': manifest['state'], 'narrow_examples': manifest['arms']['narrow']['examples'],
                      'correction': manifest['narrow_sampling_correction']}), flush=True)


if __name__ == '__main__':
    main()
