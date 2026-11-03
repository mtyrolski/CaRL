"""Create a disjoint 100-board real 12x12 pilot without exposing its witnesses."""
import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows, state_hash, to_rows
from carl.icl.campaign_data import public_board, sha256_file, verify_record
from carl.icl.datasets import _public_instances, verify_solution
from scripts.datasets.audit_campaign_history import extract
from scripts.datasets.prepare_fresh_trajectory_holdout import verified_trajectory

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'data/icl/campaign_dataset_v2'
ORIGINAL_PRIVATE = ROOT / 'data/icl/campaign_dataset_v2_private'
OUTPUT = ROOT / 'data/icl/campaign_dataset_12x12_v1'
PRIVATE = ROOT / 'data/icl/campaign_dataset_12x12_v1_private'
REPORT = ROOT / 'analysis/luna_icl/campaign_100/small_boards_v1'
INDEX = ROOT / 'data/icl/.private_campaign_shards.sqlite'
HISTORY = ROOT / 'analysis/luna_icl/campaign_100/history_snapshot.json'
FAMILY = 'real_12x12x4'
SEED = 20261117
EXTRA_COUNT = 50
CALIBRATION_ID = '08e8317905e28ebc90073b94aad3ffea004ef265f5ad2d09065e44c2137d80ab'


def read(path: Path) -> Any:
    """Read JSON without emitting its contents."""
    return json.loads(path.read_text(encoding='utf-8'))


def observed_cache_layouts() -> tuple[set[str], int]:
    """Inspect saved requests only and retain their canonical layout IDs."""
    observed: set[str] = set()
    files = 0
    for path in (ROOT / 'analysis/luna_icl').rglob('*.json'):
        if path.parent.name != 'openai':
            continue
        record = read(path)
        if isinstance(record, dict) and 'request' in record:
            extract(record['request'], observed, set())
            files += 1
    return observed, files


def verify_group(job: tuple[str, list[dict[str, Any]]]) -> list[tuple[dict[str, Any], list[str], list[int]]]:
    """Replay source trajectories in a shard and return only verified initial states."""
    path, candidates = job
    raw = joblib.load(ROOT / path)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    verified = []
    for candidate in candidates:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                states, actions = verified_trajectory(np.asarray(trajectories[candidate['source_index']]))
            board = states[0]
            if (len(actions) < 16 or canonical_layout_id(board) != candidate['layout_id']
                    or state_hash(board) != candidate['state_hash']):
                continue
            verified.append((candidate, to_rows(board), actions))
        except (ValueError, IndexError, TypeError):
            continue
    return verified


def build(workers: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select new pilot layouts while leaving all existing test cohorts disjoint."""
    history = read(HISTORY)
    old_pilot = read(ORIGINAL / FAMILY / 'pilot100.json')['instances']
    old_private = read(ORIGINAL_PRIVATE / 'pilot100' / f'{FAMILY}.json')['witnesses']
    old_actions = {row['board_id']: row['actions'] for row in old_private}
    if len(old_pilot) != 50 or len(old_actions) != 50:
        raise ValueError('Existing real pilot must contain 50 boards and 50 witnesses')
    groups = {split: read(ORIGINAL / FAMILY / f'{split}.json')['instances']
              for split in ('component_validation', 'scale_validation', 'pilot100', 'holdout1000')}
    groups['demonstration'] = read(ORIGINAL / 'demonstrations.json')['instances']
    groups['policy_validation'] = read(ROOT / 'data/icl/campaign_policy_validation_v1' / FAMILY /
                                       'component_validation.json')['instances']
    group_ids = {key: {row['layout_id'] for row in rows} for key, rows in groups.items()}
    names = list(group_ids)
    if any(len(group_ids[key]) != len(groups[key]) for key in names):
        raise ValueError('An existing split has duplicate layouts')
    if any(group_ids[a] & group_ids[b] for i, a in enumerate(names) for b in names[i + 1:]):
        raise ValueError('Existing splits overlap')
    cached, cache_files = observed_cache_layouts()
    if group_ids['pilot100'] & (cached | set(history['observed_layout_ids'])):
        raise ValueError('An existing pilot layout appears in observed model requests')
    excluded = set(history['observed_layout_ids']) | set(history['reserved_layout_ids']) | cached
    for values in group_ids.values():
        excluded.update(values)
    with sqlite3.connect(INDEX.resolve().as_uri() + '?mode=ro', uri=True) as db:
        indexed = dict(db.execute('SELECT path, sha256 FROM shards'))
        directory = ROOT / 'rl-data/validation/sokoban/offline/12-12-4'
        paths = {str(path.relative_to(ROOT)): path for path in directory.glob('sokoban_12_12_4_trajectories_part_*.pkl')}
        if set(indexed) != set(paths) or not paths:
            raise ValueError('Indexed shards do not match the available source files')
        if any(sha256_file(path) != indexed[name] for name, path in paths.items()):
            raise ValueError('Source shard differs from the cross-shard index')
        rows = db.execute('SELECT boards.path, shards.sha256, trajectory_index, layout_id, state_hash '
                          'FROM boards JOIN shards ON boards.path=shards.path '
                          'WHERE trajectory_states > 16 AND unsolved=1').fetchall()
    candidates = []
    for path, digest, index, layout, state in rows:
        if layout not in excluded:
            identifier = f'{digest}:{index}'
            rank = hashlib.sha256(f'carl-small-pilot-v1:{SEED}:{identifier}'.encode()).hexdigest()
            candidates.append({'source_path': path, 'source_sha256': digest, 'source_index': index,
                               'layout_id': layout, 'state_hash': state, 'rank': rank})
    candidates.sort(key=lambda x: (x['rank'], x['source_sha256'], x['source_index']))
    unique = []
    seen = set(excluded)
    for candidate in candidates:
        if candidate['layout_id'] not in seen:
            seen.add(candidate['layout_id'])
            unique.append(candidate)
    print(f'Indexed distinct unexposed 12x12 candidate layouts: {len(unique)}; required: {EXTRA_COUNT}', flush=True)
    if len(unique) < EXTRA_COUNT:
        raise ValueError('Insufficient new distinct 12x12 layouts')
    accepted: list[tuple[dict[str, Any], list[str], list[int]]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(unique), 128):
            batch = unique[start:start + 128]
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in batch:
                grouped[item['source_path']].append(item)
            results = [result for part in executor.map(verify_group, grouped.items()) for result in part]
            by_id = {(row[0]['source_sha256'], row[0]['source_index']): row for row in results}
            accepted.extend(by_id[(item['source_sha256'], item['source_index'])] for item in batch
                            if (item['source_sha256'], item['source_index']) in by_id)
            if len(accepted) >= EXTRA_COUNT:
                break
    if len(accepted) < EXTRA_COUNT:
        raise ValueError(f'Only {len(accepted)} replay-verified fresh boards available')
    pilot = list(old_pilot)
    witnesses = [{'board_id': row['board_id'], 'actions': old_actions[row['board_id']]} for row in old_pilot]
    for candidate, rows, actions in accepted[:EXTRA_COUNT]:
        part = Path(candidate['source_path']).stem.rsplit('_', 1)[1]
        record = public_board(rows=rows, family=FAMILY, split='pilot100',
                              trajectory_id=f"part_{part}:{candidate['source_index']}",
                              source_path=candidate['source_path'], source_sha256=candidate['source_sha256'],
                              source_index=candidate['source_index'], generator_version=None, seed=None)
        if record['state_hash'] != candidate['state_hash'] or record['layout_id'] != candidate['layout_id']:
            raise ValueError('Selected initial board does not match its source index')
        pilot.append(record)
        witnesses.append({'board_id': record['board_id'], 'actions': actions})
    ids = {row['layout_id'] for row in pilot}
    if len(pilot) != 100 or len(ids) != 100 or any(ids & group_ids[key] for key in names if key != 'pilot100'):
        raise ValueError('Small-board pilot is not 100 disjoint layouts')
    if {row['board_id'] for row in pilot} != {row['board_id'] for row in witnesses}:
        raise ValueError('Pilot/witness identities differ')
    for row, witness in zip(pilot, witnesses, strict=True):
        verify_record(row, FAMILY, 'pilot100')
        with contextlib.redirect_stdout(io.StringIO()):
            verify_solution(from_rows(row['state']), witness['actions'])
    lengths = np.asarray([len(row['actions']) for row in witnesses], dtype=int)
    public = {'provenance': {'version': 1, 'family': FAMILY, 'split': 'pilot100',
                             'selection_seed': SEED, 'source': 'indexed real offline trajectories'},
              'instances': pilot}
    private = {'version': 1, 'witnesses': witnesses}
    report = {'version': 1, 'selection_seed': SEED, 'family': FAMILY, 'boards': len(pilot),
              'unique_layouts': len(ids), 'preserved_previous_pilot': 50, 'fresh_additions': EXTRA_COUNT,
              'indexed_shards': len(indexed), 'eligible_unexposed_unique_layouts': len(unique),
              'cached_request_files_checked': cache_files,
              'overlap_by_existing_split': {key: len(ids & value) for key, value in group_ids.items()
                                            if key != 'pilot100'},
              'overlap_observed_requests_or_history': len(ids & (cached | set(history['observed_layout_ids']))),
              'witnesses_replayed': len(witnesses), 'every_witness_solved': True,
              'known_witness_actions': {'min': int(lengths.min()), 'median': float(np.median(lengths)),
                                        'max': int(lengths.max()), 'mean': float(np.mean(lengths))},
              'witness_length_note': 'Known valid trajectories, not minimum solution lengths.',
              'holdout_note': 'Existing 500 real holdout boards remain separate; no 1000-board small holdout was created.',
              'previous_public_pilot_sha256': sha256_file(ORIGINAL / FAMILY / 'pilot100.json'),
              'history_snapshot_sha256': sha256_file(HISTORY), 'shard_index_sha256': sha256_file(INDEX)}
    return public, private, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--verify', action='store_true',
                        help='Recompute selection and witnesses and compare existing outputs without changing them')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be in 1..8')
    public, private, report = build(args.workers)
    if args.verify:
        public_path = OUTPUT / FAMILY / 'pilot100.json'
        private_path = PRIVATE / 'pilot100' / f'{FAMILY}.json'
        if read(public_path) != public or read(private_path) != private:
            raise ValueError('Rebuilt boards or witnesses differ from existing small-board pilot')
        summary = read(REPORT / 'integrity.json')
        if (summary['public_manifest_sha256'] != sha256_file(public_path)
                or summary['private_witness_sha256'] != sha256_file(private_path)):
            raise ValueError('Saved pilot or private witness checksum changed')
        _public_instances(public_path)
        print('Rebuilt all 100 identities and witnesses; public and private artifacts match exactly.', flush=True)
        return
    if OUTPUT.exists() or PRIVATE.exists():
        raise ValueError('Destination already exists; do not overwrite an existing cohort')
    OUTPUT.mkdir(parents=True)
    PRIVATE.mkdir(parents=True, mode=0o700)
    (PRIVATE / 'pilot100').mkdir(mode=0o700)
    (OUTPUT / FAMILY).mkdir()
    shutil.copy2(ORIGINAL / 'training.json', OUTPUT / 'training.json')
    shutil.copy2(ORIGINAL / 'value_examples.json', OUTPUT / 'value_examples.json')
    for split in ('component_validation', 'scale_validation'):
        shutil.copy2(ORIGINAL / FAMILY / f'{split}.json', OUTPUT / FAMILY / f'{split}.json')
    (OUTPUT / 'calibration').mkdir()
    shutil.copytree(ORIGINAL / 'calibration' / CALIBRATION_ID, OUTPUT / 'calibration' / CALIBRATION_ID)
    public_path = OUTPUT / FAMILY / 'pilot100.json'
    private_path = PRIVATE / 'pilot100' / f'{FAMILY}.json'
    write_json(public_path, public)
    write_json(private_path, private)
    private_path.chmod(0o600)
    _public_instances(public_path)
    manifest = {'version': 1, 'name': 'small_boards_real_12x12_v1',
                'manifest_sha256': {f'{FAMILY}/{split}.json': sha256_file(OUTPUT / FAMILY / f'{split}.json')
                                    for split in ('component_validation', 'scale_validation', 'pilot100')},
                'pilot_boards': 100, 'holdout1000_status': 'not_prepared', 'selection_seed': SEED}
    write_json(OUTPUT / 'campaign_dataset_manifest.json', manifest)
    report.update(public_manifest_sha256=sha256_file(public_path),
                  private_witness_sha256=sha256_file(private_path))
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(REPORT / 'integrity.json', report)
    print('Prepared 100 real 12x12 pilot boards; all witnesses replayed, existing holdout disjoint.', flush=True)
    print(f'Public data: {OUTPUT}; private witnesses: {PRIVATE}; integrity: {REPORT / "integrity.json"}', flush=True)


if __name__ == '__main__':
    main()
