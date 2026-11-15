#!/usr/bin/env python3
"""Prepare a new, D4-disjoint campaign training cohort from trusted CaRL data.

The train-named source is a byte-identical alias of a validation-directory
shard. Its historical split and the old checkpoints' exposure are unknown;
the training role is assigned only for this campaign.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scripts.datasets.prepare_fresh_trajectory_holdout import verified_trajectory
from scripts.prepare_overnight_cpu_data import check_board
from scripts.prepare_overnight_cpu_data import history_register
from scripts.prepare_overnight_cpu_data import to_records
from scripts.prepare_overnight_cpu_stage02 import atomic_json
from scripts.prepare_overnight_cpu_stage02 import atomic_jsonl
from scripts.prepare_overnight_cpu_stage02 import file_sha
from scripts.prepare_overnight_cpu_stage02 import work

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE = 'rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_3.pkl'
ALIAS_RELATIVE = 'rl-data/validation/sokoban/offline/12-12-4-train/sokoban_12_12_4_trajectories_part_3.pkl'
INDEX_RELATIVE = 'data/icl/.private_campaign_shards.sqlite'
COHORT = (
    (582, '9bebab45c9a6f2d4e671b820dae779a154502d345b4cdad247686756f36d74a0'),
    (403, 'f85d0b5c00c77ec0f33b31b58a132bcd374a74872637b5d2189ca5d1734c1f53'),
    (729, '029fe2692af7d7be1b4f60fd0c9fb176b24027ee4a4647109e25e66c79a7c1ba'),
    (649, '5a4fb0e804525cbe797efc8932692167e883cb74d51db1bbbe21bde9f60873c7'),
    (874, '2adf88cb46b44072392c3fc7bf0ae306e29c4cd3520da4acd3988b1a99dae649'),
    (338, '8c68638cf18f9180021040c4bab0efd2f801a109603dabb73e9391f99538734a'),
    (282, '357d705eb9496f4ca55fd77d916af3809b24adefa89231e7a2ba17ccf42d634c'),
    (959, '030a25070b7af40a69162e33736dd741a453af54e3267cd40a41d183215ac200'),
    (409, '8ef9d5fb6de44ed39c595f81e1de1efe91a0a8f82cc23da7866e2d57666a2a82'),
    (533, 'cf56ffd5a84aa3c553fc28b6c3a7ef6c6aa40e986c935dfa7121866247c4b2f0'),
    (410, '92d92ed2e866ba24adaeac16ab443e5bdfe68042fc4423874bea6d81a3494dd1'),
    (239, '79cff8185761cbf53d10bb6b234395fbe7ede75bf9735977dcf27ffd97711b88'),
    (114, '56e863efc562bc4184a8285fc4cc773731e2b380ee69c1a67b2c706ba6cfac53'),
    (989, 'e41276a80cb2d5fb75213d4cc833c74faf57d379fa89c38bf5012a58c474a8d7'),
    (445, '9e685c3062d2991b4f9ecbb5a2e4a83400a4c81544e261954aeb99617ec79c7b'),
    (523, 'a8d0a1599902ad313936c775695b770b67c1a23adfcf3b8ad08ced0010ee310d'),
    (739, 'c8ce0fa7afabf4afe367a43aee344a430324cec9ad844cfd3d608058c8910383'),
    (656, '465327a80689a862673e7b97152fcdc0e2aa977c860926a2fc17ca3b00c63c3b'),
    (7, 'e507f4f85863ac22f5b38fd4a34722b0f2a0e329d927da4f182847bc4b05dd54'),
    (696, '4da8097e184a9e9fd9226108c1668fcaabbd8ce7f774af5ee04108810235b7ff'),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def json_sha(value: object) -> str:
    """Match the serialisation used by the shared atomic JSON writer."""
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    return hashlib.sha256(text.encode()).hexdigest()


def excluded_layouts(history: dict[str, Any], frozen: dict[str, Any],
                     stage01_trajectories: list[dict[str, Any]]) -> set[str]:
    return (set(history['observed_layout_ids']) | set(history['previously_protected_layout_ids'])
            | set(frozen['observed_layout_ids']) | set(frozen['training_prohibited_layout_ids'])
            | {row['layout_id'] for row in stage01_trajectories})


def verify_cohort(cohort: tuple[tuple[int, str], ...], excluded: set[str],
                  indexed: dict[int, str]) -> None:
    indices = [index for index, _ in cohort]
    layouts = [layout for _, layout in cohort]
    if len(set(indices)) != len(indices) or len(set(layouts)) != len(layouts):
        raise ValueError('Training cohort contains a duplicate index or canonical layout')
    if set(layouts) & excluded:
        raise ValueError('Training cohort intersects a known observed or protected layout')
    if any(indexed.get(index) != layout for index, layout in cohort):
        raise ValueError('Training cohort differs from the trusted source index')


def training_records(index: int, rank: int, states: list[np.ndarray], actions: list[int],
                     *, layout: str, source_hash: str, seed: int) -> tuple[dict, list[dict], list[dict]]:
    board, pairs, unavailable = to_records(
        {'id': f'shard3_trajectory_{index}', 'discarded_suffix': 0}, states, actions,
        split='train', family='12x12x4', source_family='rl_12x12x4_campaign_train',
        source_path=SOURCE_RELATIVE, source_hash=source_hash,
        original_split='unknown; train-named alias is byte-identical to validation source',
        exposure_status='not in checked historical observed or protected layouts',
        layout=layout, seed=seed, contexts=1, board_rank=rank,
    )
    # to_records was designed for the development/evaluation split, so its
    # default purpose cannot be used for this explicitly assigned train role.
    board.update(training_eligibility='campaign_train_assigned', purpose='training_only')
    return board, pairs, unavailable


def inspect_input(stage01: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str]:
    manifest = json.loads((stage01 / 'data_manifest.json').read_text())
    for name in ('exposure_register.json', 'trajectories.jsonl'):
        if file_sha(stage01 / name) != manifest['outputs_sha256'][name]:
            raise ValueError(f'Stage01 source does not match its frozen manifest: {name}')
    frozen = json.loads((stage01 / 'exposure_register.json').read_text())
    trajectories = read_jsonl(stage01 / 'trajectories.jsonl')
    history = history_register()
    canonical, alias = ROOT / SOURCE_RELATIVE, ROOT / ALIAS_RELATIVE
    source_hash = file_sha(canonical)
    if file_sha(alias) != source_hash:
        raise ValueError('The train-named source is no longer a byte-identical alias')
    index_path = ROOT / INDEX_RELATIVE
    with sqlite3.connect(index_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        entry = db.execute('SELECT sha256 FROM shards WHERE path=?', (SOURCE_RELATIVE,)).fetchone()
        indexed = dict(db.execute('SELECT trajectory_index,layout_id FROM boards WHERE path=?',
                                  (SOURCE_RELATIVE,)).fetchall())
    if entry is None or entry[0] != source_hash:
        raise ValueError('Source hash differs from the indexed shard')
    verify_cohort(COHORT, excluded_layouts(history, frozen, trajectories), indexed)
    return frozen, trajectories, history, source_hash


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.workers > len(os.sched_getaffinity(0)):
        raise ValueError('workers must fit available CPU affinity')
    stage01, output = args.stage01_data.resolve(), args.output.resolve()
    frozen, stage01_trajectories, history, source_hash = inspect_input(stage01)
    contract = {
        'schema_version': 1, 'role': 'campaign_train', 'source_path': str((ROOT / SOURCE_RELATIVE).resolve()),
        'source_sha256': source_hash, 'alias_source_path': str((ROOT / ALIAS_RELATIVE).resolve()),
        'alias_source_sha256': source_hash, 'indexed_source_path': SOURCE_RELATIVE,
        'indexed_source_sha256': source_hash, 'stage01_exposure_path': str(stage01 / 'exposure_register.json'),
        'stage01_exposure_sha256': file_sha(stage01 / 'exposure_register.json'),
        'stage01_trajectories_path': str(stage01 / 'trajectories.jsonl'),
        'stage01_trajectories_sha256': file_sha(stage01 / 'trajectories.jsonl'),
        'fresh_history_path': str(output / 'history_register.json'), 'fresh_history_sha256': json_sha(history),
        'script_sha256': file_sha(Path(__file__)),
        'stage02_preparation_sha256': file_sha(ROOT / 'scripts/prepare_overnight_cpu_stage02.py'),
        'planned_cohort': [{'trajectory_index': index, 'layout_id': layout} for index, layout in COHORT],
        'seed': args.seed, 'contexts_per_board': 1, 'bfs_max_nodes': 2000, 'bfs_max_depth': 32,
        'old_checkpoint_exposure': 'unknown',
        'historical_split': 'unknown; train-labelled alias is byte-identical to validation source',
    }
    contract_path = output / 'contract.json'
    if contract_path.exists():
        if not args.resume or json.loads(contract_path.read_text()) != contract:
            raise ValueError('Existing training contract differs or --resume was not requested')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Refusing to overwrite a nonempty output directory')
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(contract_path, contract)
    start = time.monotonic()
    atomic_json(output / 'history_register.json', history)
    if file_sha(output / 'history_register.json') != contract['fresh_history_sha256']:
        raise AssertionError('Saved historical exposure digest changed')
    atomic_json(output / 'data_manifest.json', {'state': 'running', 'contract': contract,
                                                'phase': 'trajectory_replay', 'argv': sys.argv})
    raw = joblib.load(ROOT / SOURCE_RELATIVE)
    raw = list(raw.values()) if isinstance(raw, dict) else list(raw)
    boards: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rank, (index, expected_layout) in enumerate(COHORT):
        try:
            array = np.asarray(raw[index])
            actual_layout = check_board(array[0], 4)
            if actual_layout != expected_layout:
                raise ValueError('Replayed initial state differs from the frozen D4 layout')
            states, actions = verified_trajectory(array)
            if len(actions) < 1:
                raise ValueError('Trajectory has no primitive action')
            board, current_pairs, missing = training_records(index, rank, states, actions, layout=actual_layout,
                                                              source_hash=source_hash, seed=args.seed)
            board['post_terminal_suffix_dropped'] = len(array) - len(states)
            boards.append(board)
            pairs.extend(current_pairs)
            unavailable.extend(missing)
        except (IndexError, TypeError, ValueError) as error:
            rejected.append({'trajectory_index': index, 'layout_id': expected_layout,
                             'reason': f'{type(error).__name__}: {error}'})
    train_ids = sorted(board['layout_id'] for board in boards)
    exposure = {**{key: contract[key] for key in (
                    'schema_version', 'role', 'source_path', 'source_sha256', 'alias_source_path',
                    'alias_source_sha256', 'indexed_source_path', 'indexed_source_sha256',
                    'stage01_exposure_path', 'stage01_exposure_sha256', 'stage01_trajectories_path',
                    'stage01_trajectories_sha256', 'fresh_history_path', 'fresh_history_sha256',
                    'old_checkpoint_exposure', 'historical_split')},
                'canonicalization': 'carl-layout-d4-v1', 'train_layout_ids': train_ids,
                'historical_observed_layout_ids': history['observed_layout_ids'],
                'historical_protected_layout_ids': history['previously_protected_layout_ids'],
                'stage01_development_layout_ids': frozen['development_layout_ids'],
                'stage01_evaluation_layout_ids': frozen['campaign_evaluation_layout_ids'],
                'stage01_training_prohibited_layout_ids': frozen['training_prohibited_layout_ids']}
    for name, value in (('trajectories.jsonl', boards), ('pairs.jsonl', pairs),
                        ('unavailable_offsets.jsonl', unavailable), ('rejected.jsonl', rejected)):
        atomic_jsonl(output / name, value)
    atomic_json(output / 'exposure_register.json', exposure)
    groups: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        groups.setdefault(pair['context_id'], []).append(pair)
    atomic_json(output / 'data_manifest.json', {'state': 'running', 'contract': contract,
                                                'phase': 'candidate_generation', 'planned_boards': len(COHORT),
                                                'replayed_boards': len(boards), 'rejected_boards': len(rejected),
                                                'planned_contexts': len(groups), 'argv': sys.argv})
    pending = []
    for context_id, group in groups.items():
        shard = output / 'shards' / f'{context_id}.json'
        if args.resume and shard.exists():
            old = json.loads(shard.read_text())
            if old['context']['context_id'] != context_id:
                raise ValueError('Saved training shard has a different context')
            if old['context']['status'] == 'complete':
                continue
        pending.append(group)
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[variable] = '1'
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(work, group, 32, 2000, args.seed): group for group in pending}
        for future in concurrent.futures.as_completed(futures):
            group = futures[future]
            atomic_json(output / 'shards' / f"{group[0]['context_id']}.json", future.result())
    contexts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for context_id in groups:
        result = json.loads((output / 'shards' / f'{context_id}.json').read_text())
        contexts.append(result['context'])
        candidates.extend(result['candidates'])
    atomic_jsonl(output / 'contexts.jsonl', contexts)
    atomic_jsonl(output / 'candidates.jsonl', candidates)
    counts = Counter(item['status'] for item in contexts)
    output_names = ('trajectories.jsonl', 'pairs.jsonl', 'unavailable_offsets.jsonl', 'rejected.jsonl',
                    'exposure_register.json', 'history_register.json', 'contexts.jsonl', 'candidates.jsonl')
    manifest = {
        'state': 'complete' if len(boards) == len(COHORT) and counts == {'complete': len(COHORT)} else 'partial',
        'contract': contract, 'planned_boards': len(COHORT), 'replayed_boards': len(boards),
        'rejected_boards': len(rejected), 'planned_contexts': len(groups), 'context_statuses': dict(counts),
        'pairs': len(pairs), 'candidates': len(candidates), 'elapsed_seconds': time.monotonic() - start,
        'workers': args.workers, 'threads_per_worker': 1, 'argv': sys.argv,
        'output_sha256': {name: file_sha(output / name) for name in output_names},
    }
    atomic_json(output / 'data_manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage01-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    print(json.dumps(prepare(parser.parse_args()), sort_keys=True))


if __name__ == '__main__':
    main()
