"""Freeze and replay source trajectories for the local five-stage campaign.

No model is loaded. Existing historical test layouts are never assigned to
training or development. All evaluation layouts remain reserved throughout
the campaign. The source directories named ``validation`` do not certify that
old checkpoints did not see their data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import tiles
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.data_sources.sokoban import verify_solution
from carl.icl.reachability_history import scan_history
from scripts.datasets.prepare_fresh_trajectory_holdout import verified_trajectory

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'data/icl/.private_campaign_shards.sqlite'
X4_SHARD = 'rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_2.pkl'
GAMMA = ROOT / 'data/icl/reachability_proposal_hard_12_v1/private/source_bundle.tar.gz'
REVERSE = ROOT / 'data/icl/reachability_generated_12x12x10_v1/private/source_bundle.tar.gz'
STANDALONE = ROOT.parent / 'data'
OFFSETS = (1, 2, 4, 8, 16, 32)
SEED = 20261115
EXTRA_JSONL = (
    'analysis/intent_generator_20261114T032232Z/dataset.jsonl',
    'analysis/luna_policy_relative_resume_20261115T0416Z/inputs/test_full_block.jsonl',
    'analysis/luna_policy_relative_resume_20261115T0416Z/inputs/test_second_block.jsonl',
    'analysis/luna_policy_relative_resume_20261115T0416Z/inputs/test_expansion_76.jsonl',
    'analysis/luna_policy_relative_resume_20261115T0416Z/inputs/test_historical_x10.jsonl',
    'analysis/luna_policy_relative_resume_20261115T0416Z/inputs/test_historical_x10_extra30.jsonl',
)
EXTRA_MANIFESTS = (
    'analysis/torch_reachability/run_v1/12x4/manifest.json',
    'analysis/torch_reachability/run_v1/12x10/manifest.json',
)


def sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def atomic_json(path: Path, data: object) -> None:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def stable_id(*parts: object) -> str:
    raw = json.dumps(parts, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def check_board(board: np.ndarray, boxes: int) -> str:
    """Check dimensions and exact object counts; return the D4 layout ID."""
    board = np.asarray(board)
    if board.shape != (12, 12, 7) or not np.isfinite(board).all():
        raise ValueError('expected finite 12x12x7 board')
    if not np.all(np.isin(board, (0, 1))) or not np.all(board.sum(axis=-1) == 1):
        raise ValueError('board is not one-hot')
    board_tiles = tiles(board)
    if (int(np.isin(board_tiles, (3, 4)).sum()) != boxes
            or int(np.isin(board_tiles, (2, 3, 6)).sum()) != boxes
            or int(np.isin(board_tiles, (5, 6)).sum()) != 1):
        raise ValueError('incorrect box, goal or player count')
    return canonical_layout_id(board)


def history_register(root: Path = ROOT) -> dict[str, Any]:
    """Inventory known exposures and conservatively protected old cohorts."""
    history = scan_history(root)
    if history['errors'] or history['missing']:
        raise ValueError('recognized historical sources contain errors or missing files')
    observed: set[str] = set()
    protected: set[str] = set()
    sources = []
    for name, row in history['sources'].items():
        ids = set(row.get('layout_ids', []))
        observed.update(ids)
        if any(term in name.lower() for term in ('/test', '/validation', 'holdout', 'pilot', 'reserved_layout_ids')):
            protected.update(ids)
        sources.append({'name': name, 'sha256': row.get('sha256'), 'count': len(ids), 'kind': row.get('kind')})
    for name in EXTRA_JSONL:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Prior exposure source is missing: {path}')
        count = 0
        with path.open(encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                layout = row.get('layout_id')
                if not isinstance(layout, str) or len(layout) != 64:
                    raise ValueError(f'Invalid historical layout ID in {path}')
                observed.add(layout)
                if row.get('split') in ('test', 'validation', 'evaluation'):
                    protected.add(layout)
                count += 1
        sources.append({'name': name, 'sha256': sha256(path), 'count': count, 'kind': 'prior_jsonl'})
    for name in EXTRA_MANIFESTS:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Prior exposure manifest is missing: {path}')
        rows = json.loads(path.read_text())['boards']
        ids = {row['layout_id'] for row in rows}
        observed.update(ids)
        protected.update(ids)
        sources.append({'name': name, 'sha256': sha256(path), 'count': len(ids), 'kind': 'prior_evaluation'})
    figure_csv = root / 'analysis/figure1_cllp_matched_20261115T0555Z/figure/paired_cases.csv'
    if not figure_csv.is_file():
        raise FileNotFoundError(figure_csv)
    with figure_csv.open(newline='', encoding='utf-8') as source:
        ids = {row['layout_id'] for row in csv.DictReader(source)}
    observed.update(ids)
    protected.update(ids)
    sources.append({'name': str(figure_csv.relative_to(root)), 'sha256': sha256(figure_csv),
                    'count': len(ids), 'kind': 'prior_evaluation'})
    return {'canonicalization': 'carl-layout-d4-v1', 'observed_layout_ids': sorted(observed),
            'previously_protected_layout_ids': sorted(protected), 'sources': sources,
            'limitations': history['limitations'] + ['Old checkpoint training exposure is not independently known.']}


def read_archive(path: Path, member_suffix: str) -> list[dict[str, Any]]:
    """Read the trusted local JSON bundle without extracting filesystem paths."""
    with tarfile.open(path, 'r:gz') as source:
        members = [item for item in source.getmembers() if item.name.endswith(member_suffix)]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError(f'Expected one regular {member_suffix} in {path}')
        stream = source.extractfile(members[0])
        if stream is None:
            raise ValueError(f'Cannot read {member_suffix}')
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError('source rows must be a list')
    return rows


def read_standalone(name: str) -> tuple[dict[str, Any], list[np.ndarray], list[int], Path]:
    """Replay a saved user board/action trace up to its first terminal state."""
    board_path = STANDALONE / 'boards' / f'{name}.txt'
    action_path = STANDALONE / 'witnesses' / f'{name}.json'
    rows = [line.replace(' ', '-') for line in board_path.read_text().splitlines()]
    initial = from_rows(rows)
    data = json.loads(action_path.read_text())
    if data.get('id') != name or data.get('action_encoding') != 'UDLR; row increases downward':
        raise ValueError('unexpected standalone trajectory identity or action encoding')
    encoded = data['actions']
    saved_states = data['states']
    if not isinstance(encoded, str) or len(saved_states) != len(encoded) + 1:
        raise ValueError('standalone trajectory state/action counts differ')
    env = environment(initial)
    states = [initial]
    actions: list[int] = []

    def compare(state: np.ndarray, saved: dict[str, Any]) -> None:
        board = tiles(state)
        player = np.argwhere(np.isin(board, (5, 6))).tolist()
        boxes = np.argwhere(np.isin(board, (3, 4))).tolist()
        if player != [saved['player']] or sorted(boxes) != sorted(saved['boxes']):
            raise ValueError('standalone recorded state differs from CaRL replay')

    compare(initial, saved_states[0])
    for index, character in enumerate(encoded):
        if env.is_solved(states[-1]):
            break
        try:
            action = 'UDLR'.index(character)
        except ValueError as error:
            raise ValueError('standalone trajectory contains an unknown action') from error
        successor = np.asarray(env.next_state(states[-1], action))
        if np.array_equal(successor, states[-1]):
            raise ValueError('standalone trajectory contains a no-op')
        compare(successor, saved_states[index + 1])
        actions.append(action)
        states.append(successor)
    checked = verify_solution(initial, actions)
    if len(checked) != len(states) or any(not np.array_equal(a, b) for a, b in zip(checked, states)):
        raise ValueError('standalone verification differs from recorded trajectory')
    row = {'id': name, 'discarded_suffix': len(encoded) - len(actions),
           'supplemental_source_sha256': {str(board_path): sha256(board_path)}}
    return row, states, actions, action_path


def select_context_indices(states: list[np.ndarray], layout: str, seed: int, count: int) -> list[int]:
    """Prefer distinct starts with a non-trivial k=32 target, then use shorter starts."""
    if count < 1:
        raise ValueError('context count must be positive')
    hashes = [state_hash(state) for state in states]
    eligible = [i for i in range(len(states) - 1) if hashes[i] != hashes[i + 1]]
    rng = random.Random(stable_id('contexts', seed, layout))
    long_window = [i for i in eligible if i + max(OFFSETS) < len(states)
                   and hashes[i] != hashes[i + max(OFFSETS)]]
    long_set = set(long_window)
    shorter = [i for i in eligible if i not in long_set]
    rng.shuffle(long_window)
    rng.shuffle(shorter)
    selected: list[int] = []
    seen_hashes: set[str] = set()
    for index in long_window + shorter:
        if hashes[index] in seen_hashes:
            continue
        selected.append(index)
        seen_hashes.add(hashes[index])
        if len(selected) == count:
            return selected
    if not selected:
        raise ValueError('trajectory has no distinct state with a successor')
    return selected


def to_records(row: dict[str, Any], states: list[np.ndarray], actions: list[int], *, split: str,
               family: str, source_family: str, source_path: str, source_hash: str,
               original_split: str, exposure_status: str, layout: str, seed: int,
               contexts: int, board_rank: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create immutable trajectory and paired-state records without model output."""
    initial_rows = to_rows(states[0])
    board_record = {'trajectory_id': row['id'], 'layout_id': layout, 'family': family,
                    'source_family': source_family, 'split': split, 'board_rank': board_rank,
                    'source_path': source_path, 'source_sha256': source_hash,
                    'original_split': original_split, 'exposure_status': exposure_status,
                    'training_eligibility': ('prohibited_evaluation' if split == 'evaluation' else
                                             'unresolved' if family == '12x12x4'
                                             or source_family == 'standalone_gamma2_12x12x10'
                                             else 'historical_training'),
                    'purpose': 'benchmark_only' if split == 'development' else 'evaluation_only',
                    'initial_state': initial_rows, 'actions': actions,
                    'state_hashes': [state_hash(state) for state in states],
                    'push_actions': sum(not np.array_equal(np.isin(tiles(a), (3, 4)),
                                                           np.isin(tiles(b), (3, 4)))
                                        for a, b in zip(states, states[1:])),
                    'primitive_actions': len(actions), 'post_terminal_suffix_dropped': row.get('discarded_suffix', 0),
                    'supplemental_source_sha256': row.get('supplemental_source_sha256')}
    indices = select_context_indices(states, layout, seed, contexts)
    pairs = []
    unavailable = []
    for context_rank, i in enumerate(indices):
        context_id = stable_id('context-v1', layout, i, state_hash(states[i]))
        for k in OFFSETS:
            missing = {'context_id': context_id, 'context_rank': context_rank, 'board_rank': board_rank,
                       'layout_id': layout, 'family': family, 'source_family': source_family,
                       'split': split, 'trajectory_id': row['id'], 'i': i, 'k': k}
            if i + k >= len(states):
                unavailable.append({**missing, 'reason': 'beyond_trajectory_end'})
                continue
            before_hash, after_hash = state_hash(states[i]), state_hash(states[i + k])
            if before_hash == after_hash:
                unavailable.append({**missing, 'reason': 'target_already_satisfied'})
                continue
            pairs.append({'pair_id': stable_id('pair-v1', context_id, k, after_hash),
                          'context_id': context_id, 'context_rank': context_rank, 'board_rank': board_rank,
                          'layout_id': layout, 'family': family, 'source_family': source_family,
                          'split': split, 'trajectory_id': row['id'], 'i': i, 'k': k,
                          'source': to_rows(states[i]), 'target': to_rows(states[i + k]),
                          'source_hash': before_hash, 'target_hash': after_hash,
                          'source_path': source_path, 'source_sha256': source_hash,
                          'original_split': original_split, 'exposure_status': exposure_status,
                          'demonstration_actions': actions[i:i + k]})
    for context_rank in range(len(indices), contexts):
        unavailable.append({'context_id': None, 'context_rank': context_rank, 'board_rank': board_rank,
                            'layout_id': layout, 'family': family, 'source_family': source_family,
                            'split': split, 'trajectory_id': row['id'], 'i': None, 'k': None,
                            'reason': 'no_additional_distinct_context'})
    board_record['requested_contexts'] = contexts
    board_record['available_contexts'] = len(indices)
    return board_record, pairs, unavailable


def prepare(out: Path, *, eval4: int, development: int, contexts: int, seed: int,
            amendment: Path | None = None) -> dict[str, Any]:
    if not 1 <= eval4 <= 250 or not 1 <= development <= 10 or contexts < 1:
        raise ValueError('eval4 must be 1..250, development 1..10 and contexts positive')
    amendment_metadata = None
    if amendment is not None:
        amendment = amendment.expanduser().resolve()
        details = json.loads(amendment.read_text(encoding='utf-8'))
        prior_manifest = amendment.parent / 'data_manifest.json'
        prior_script = amendment.parent / 'preparation_v1.py'
        if (sha256(prior_manifest) != details['prior_data_manifest_sha256']
                or sha256(prior_script) != details['prior_preparation_script_sha256']):
            raise ValueError('The pre-amendment artifacts do not match the frozen hashes')
        amendment_metadata = {'path': str(amendment), 'sha256': sha256(amendment),
                              'prior_data_manifest_sha256': sha256(prior_manifest),
                              'prior_preparation_script_sha256': sha256(prior_script)}
    out.mkdir(parents=True, exist_ok=True)
    names = ('pairs.jsonl', 'trajectories.jsonl', 'unavailable_offsets.jsonl', 'data_manifest.json',
             'exposure_register.json', 'rejected.jsonl')
    if any((out / name).exists() for name in names):
        raise FileExistsError('Data artifacts already exist; use a new stage directory')
    history = history_register()
    observed = set(history['observed_layout_ids'])
    protected = set(history['previously_protected_layout_ids'])
    selected: set[str] = set()
    rejected: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    checkpoint = out / 'data_preparation_progress.json'
    if checkpoint.exists():
        raise FileExistsError('A previous preparation left progress; preserve it and use a new directory')

    def reject(entry: dict[str, Any]) -> None:
        rejected.append(entry)
        path = out / 'rejected.partial.jsonl'
        with path.open('a', encoding='utf-8') as target:
            target.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n')
            target.flush()

    def progress() -> None:
        atomic_json(checkpoint, {'state': 'preparing', 'accepted_trajectories': len(trajectories),
                                 'prepared_pairs': len(pairs), 'rejected': len(rejected)})

    def accept(row: dict[str, Any], states: list[np.ndarray], actions: list[int], **kwargs: Any) -> bool:
        try:
            layout = check_board(states[0], 4 if kwargs['family'] == '12x12x4' else 10)
            if layout in selected:
                raise ValueError('canonical layout already selected')
            if kwargs['split'] == 'development' and layout in protected:
                raise ValueError('historically protected layout cannot be development')
            # Both trusted loaders above call verify_solution, which validates
            # walls, goals, box count and each primitive transition throughout.
            board, new_pairs, new_unavailable = to_records(row, states, actions, layout=layout, seed=seed,
                                                          contexts=contexts, **kwargs)
        except (ValueError, IndexError, KeyError, TypeError) as error:
            reject({'trajectory_id': row.get('id'), 'source_path': kwargs['source_path'],
                    'split': kwargs['split'], 'reason': f'{type(error).__name__}: {error}'})
            return False
        selected.add(layout)
        trajectories.append(board)
        pairs.extend(new_pairs)
        unavailable.extend(new_unavailable)
        if len(trajectories) % 10 == 0:
            progress()
        return True

    shard = ROOT / X4_SHARD
    shard_hash = sha256(shard)
    with sqlite3.connect(INDEX.resolve().as_uri() + '?mode=ro', uri=True) as db:
        indexed = db.execute('SELECT trajectory_index,layout_id,trajectory_states,unsolved FROM boards '
                             'WHERE path=? AND trajectory_states>1 AND unsolved=1', (X4_SHARD,)).fetchall()
        record = db.execute('SELECT sha256 FROM shards WHERE path=?', (X4_SHARD,)).fetchone()
    if not record or record[0] != shard_hash:
        raise ValueError('Stored x4 shard index does not match source hash')
    raw = joblib.load(shard)
    raw = list(raw.values()) if isinstance(raw, dict) else list(raw)
    random.Random(seed).shuffle(indexed)
    counts4 = Counter()
    for index, indexed_layout, _, _ in indexed:
        if indexed_layout in observed or indexed_layout in protected or indexed_layout in selected:
            continue
        split = 'development' if counts4['development'] < development else 'evaluation'
        if split == 'evaluation' and counts4['evaluation'] >= eval4:
            break
        try:
            array = np.asarray(raw[index])
            layout = check_board(array[0], 4)
            if layout != indexed_layout:
                raise ValueError('source layout differs from the indexed canonical layout')
            states, actions = verified_trajectory(array)
            discarded = len(array) - len(states)
            row = {'id': f'shard2_trajectory_{index}', 'discarded_suffix': discarded}
            if accept(row, states, actions, split=split, family='12x12x4',
                      source_family='rl_validation_12x12x4', source_path=X4_SHARD,
                      source_hash=shard_hash, original_split='rl-data/validation; not checkpoint training provenance',
                      exposure_status='not found in checked history; no guarantee', board_rank=counts4[split]):
                counts4[split] += 1
        except (ValueError, IndexError, KeyError, TypeError) as error:
            reject({'trajectory_id': f'shard2_trajectory_{index}', 'source_path': X4_SHARD,
                    'split': split, 'reason': f'{type(error).__name__}: {error}'})
    if counts4['development'] != development or counts4['evaluation'] != eval4:
        raise ValueError(f'Insufficient replayable x4 layouts: {dict(counts4)}')

    gamma_rows = read_archive(GAMMA, '/all.json')
    reverse_rows = read_archive(REVERSE, '/generated_12x12x10/test.json')
    gamma_hash, reverse_hash = sha256(GAMMA), sha256(REVERSE)
    gamma_train = [row for row in gamma_rows if row.get('split') == 'training']
    gamma_test = [row for row in gamma_rows if row.get('split') == 'main_exploratory']
    if len(gamma_train) != 12 or len(gamma_test) != 50 or len(reverse_rows) != 50:
        raise ValueError('Unexpected x10 source cohort counts')
    future_evaluation = {row['layout_id'] for row in gamma_test + reverse_rows}
    if len(future_evaluation) != 100:
        raise ValueError('x10 source cohorts overlap under their saved D4 IDs')
    protected.update(future_evaluation)
    gamma_train = [row for row in gamma_train if row['layout_id'] not in protected]
    random.Random(stable_id(seed, 'x10-development')).shuffle(gamma_train)
    rng_gamma = random.Random(stable_id(seed, 'x10-gamma-evaluation'))
    rng_reverse = random.Random(stable_id(seed, 'x10-reverse-evaluation'))
    rng_gamma.shuffle(gamma_test)
    rng_reverse.shuffle(reverse_rows)
    development_count = 0
    for row in gamma_train:
        if development_count >= development:
            break
        try:
            initial = from_rows(row['initial_state'])
            states = verify_solution(initial, row['actions'])
            actual = check_board(initial, 10)
            if actual != row['layout_id']:
                raise ValueError('gamma2 source layout hash mismatch')
            if not accept(row, states, row['actions'], split='development', family='12x12x10',
                          source_family='proposal_gamma2_12x12x10', source_path=str(GAMMA.relative_to(ROOT)),
                          source_hash=gamma_hash, original_split='training',
                          exposure_status='historical training; not evaluation', board_rank=development_count):
                raise ValueError('gamma2 development trajectory rejected')
            development_count += 1
        except (ValueError, IndexError, KeyError, TypeError) as error:
            reject({'trajectory_id': row.get('id'), 'source_path': str(GAMMA.relative_to(ROOT)),
                    'split': 'development', 'reason': f'{type(error).__name__}: {error}'})
    standalone_names = ('gamma2_s1212', 'gamma2_s1217', 'gamma2_s1223',
                        'gamma2_s5136', 'gamma2_s5580', 'gamma2_s6123')
    for name in standalone_names:
        if development_count >= development:
            break
        try:
            row, states, actions, source_path = read_standalone(name)
            if accept(row, states, actions, split='development', family='12x12x10',
                      source_family='standalone_gamma2_12x12x10', source_path=str(source_path),
                      source_hash=sha256(source_path), original_split='unassigned; protected_previous_assets',
                      exposure_status='known prior local asset; benchmark only', board_rank=development_count):
                development_count += 1
        except (ValueError, IndexError, KeyError, TypeError) as error:
            reject({'trajectory_id': name, 'source_path': str(STANDALONE),
                    'split': 'development', 'reason': f'{type(error).__name__}: {error}'})
    if development_count != development:
        raise ValueError(f'Insufficient non-protected x10 development layouts: {development_count}')
    for rank, (gamma, reverse) in enumerate(zip(gamma_test, reverse_rows)):
        for index, (row, archive, digest, family) in enumerate((
                (gamma, GAMMA, gamma_hash, 'proposal_gamma2_12x12x10'),
                (reverse, REVERSE, reverse_hash, 'synthetic_reverse_12x12x10'))):
            try:
                initial = from_rows(row['initial_state'])
                states = verify_solution(initial, row['actions'])
                actual = check_board(initial, 10)
                if actual != row['layout_id']:
                    raise ValueError('source layout hash mismatch')
                if not accept(row, states, row['actions'], split='evaluation', family='12x12x10',
                              source_family=family, source_path=str(archive.relative_to(ROOT)),
                              source_hash=digest, original_split='main_exploratory' if index == 0 else 'test',
                              exposure_status='historically inspected evaluation; exploratory', board_rank=2 * rank + index):
                    raise ValueError('x10 evaluation trajectory rejected')
            except (ValueError, IndexError, KeyError, TypeError) as error:
                reject({'trajectory_id': row.get('id'), 'source_path': str(archive.relative_to(ROOT)),
                        'split': 'evaluation', 'reason': f'{type(error).__name__}: {error}'})
                raise

    if len({r['layout_id'] for r in trajectories}) != len(trajectories):
        raise ValueError('Duplicate canonical layout in frozen split')
    dev_layouts = {r['layout_id'] for r in trajectories if r['split'] == 'development'}
    eval_layouts = {r['layout_id'] for r in trajectories if r['split'] == 'evaluation'}
    if dev_layouts & (eval_layouts | protected):
        raise ValueError('Development overlaps evaluation or previously protected layouts')
    if len({r['pair_id'] for r in pairs}) != len(pairs):
        raise ValueError('Duplicate pair ID')
    history.update({'seed': seed, 'development_layout_ids': sorted(dev_layouts),
                    'campaign_evaluation_layout_ids': sorted(eval_layouts),
                    'training_prohibited_layout_ids': sorted(protected | eval_layouts),
                    'evaluation_reservation': 'all campaign evaluation layouts are reserved from training, '
                                              'annotation training, calibration and checkpoint selection in all five stages',
                    'x4_development_training_eligibility': 'unresolved; benchmark only',
                    'x10_development_training_eligibility': 'nine historical training plus one benchmark-only '
                                                           'unresolved asset at default counts; check per board'})
    for name, rows in (('trajectories.jsonl', trajectories), ('pairs.jsonl', pairs),
                       ('unavailable_offsets.jsonl', unavailable), ('rejected.jsonl', rejected)):
        tmp = out / f'.{name}.{os.getpid()}.tmp'
        with tmp.open('w', encoding='utf-8') as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        os.replace(tmp, out / name)
    atomic_json(out / 'exposure_register.json', history)
    counts = Counter((r['family'], r['split'], r['source_family']) for r in trajectories)
    manifest = {'schema_version': 1, 'seed': seed, 'offsets': list(OFFSETS),
                'contexts_per_layout': contexts, 'planned_eval_prefix_x4': min(100, eval4),
                'max_eval_x4': eval4, 'eval_x10': 100, 'development_per_family': development,
                'ordering_rule': 'x4 deterministic seeded order; x10 evaluation alternates shuffled gamma2 and '
                                 'reverse-board trajectories, rank 0 onward',
                'context_rule': 'seed and canonical-layout-derived up to two distinct starts with a successor; '
                                'prefer starts with a non-trivial recorded k=32 target, and fill any remaining '
                                'slots from other eligible starts; retain each available non-trivial offset; '
                                'no board is excluded for a short trajectory and no model result is used',
                'context_selection_amendment': amendment_metadata,
                'evaluation_block_rule': 'freeze prefix and use context_rank 0 for all paired methods/k; '
                                         'context_rank 1 only for a complete matched block when throughput permits',
                'family_counts': [{'family': f, 'split': s, 'source_family': sf, 'boards': n}
                                  for (f, s, sf), n in sorted(counts.items())],
                'pairs': len(pairs), 'trajectories': len(trajectories), 'rejected': len(rejected),
                'unavailable_offsets_or_contexts': len(unavailable),
                'unavailable_reasons': dict(Counter(item['reason'] for item in unavailable)),
                'source_sha256': {X4_SHARD: shard_hash, str(GAMMA.relative_to(ROOT)): gamma_hash,
                                  str(REVERSE.relative_to(ROOT)): reverse_hash},
                'outputs_sha256': {name: sha256(out / name) for name in names if name != 'data_manifest.json'},
                'limitations': ['All x10 source cohorts were examined previously and are exploratory.',
                                'The x4 source is named validation; exact prior checkpoint exposure is unknown.',
                                'Different source distributions confound comparisons across box counts.',
                                'Offsets are recorded primitive-action counts, not pushes or shortest distances.'],
                'argv': sys.argv}
    atomic_json(out / 'data_manifest.json', manifest)
    atomic_json(checkpoint, {'state': 'complete', 'accepted_trajectories': len(trajectories),
                             'prepared_pairs': len(pairs), 'rejected': len(rejected)})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--eval4', type=int, default=250)
    parser.add_argument('--development', type=int, default=10)
    parser.add_argument('--contexts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--context-selection-amendment', type=Path)
    args = parser.parse_args()
    result = prepare(args.output.expanduser().resolve(), eval4=args.eval4, development=args.development,
                     contexts=args.contexts, seed=args.seed, amendment=args.context_selection_amendment)
    print(json.dumps({'boards': result['family_counts'], 'pairs': result['pairs'],
                      'rejected': result['rejected']}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
