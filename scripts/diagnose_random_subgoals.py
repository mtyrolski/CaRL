"""Measure short random-walk targets with parallel, deterministic simulator BFS.

This offline diagnostic never runs a full solver or calls an AI provider.  BFS
distance is not the policy-relative execution time used by the proposal.
"""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash, tiles, to_rows, validate_subgoal
from carl.icl.config import ROOT
from carl.icl.datasets import environment
from carl.icl.diagnostics import audit_subgoal

FAMILY = 'generated_32x32x10'
DEFAULT_SOURCE = ROOT / 'data/icl/campaign_dataset_v2' / FAMILY / 'component_validation.json'
DEFAULT_OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/random_walk_bfs_32'


def random_targets(rows: list[str], board_id: str, seed: int, offsets: list[int]) -> list[dict[str, Any]]:
    """Draw one action-uniform walk, keeping no-ops and stopping at task terminal."""
    initial = from_rows(rows)
    if initial.shape != (32, 32, 7):
        raise ValueError('Only full 32x32x10 validation boards are supported')
    validate_subgoal(initial, initial)
    if int(np.isin(tiles(initial), [3, 4]).sum()) != 10:
        raise ValueError('Expected exactly ten boxes')
    env = environment(initial)
    if env.is_solved(initial):
        raise ValueError('Random-walk source is already terminal')
    local_seed = int.from_bytes(hashlib.sha256(f'{seed}:{board_id}'.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(local_seed)
    current = initial.copy()
    actions: list[int] = []
    noops = 0
    targets: list[dict[str, Any]] = []
    for offset in offsets:
        while len(actions) < offset and not env.is_solved(current):
            action = int(rng.integers(0, 4))
            successor = np.asarray(env.next_state(current, action))
            noops += int(np.array_equal(current, successor))
            actions.append(action)
            current = successor
        if len(actions) < offset:
            targets.append({'board_id': board_id, 'sampling_seed': local_seed, 'offset': offset,
                            'status': 'terminal_before_offset', 'walk_actions': list(actions)})
            continue
        targets.append({'board_id': board_id, 'sampling_seed': local_seed, 'offset': offset,
                        'status': 'ready', 'source': rows, 'target': to_rows(current),
                        'source_hash': state_hash(initial), 'target_hash': state_hash(current),
                        'walk_actions': list(actions), 'walk_noops': noops,
                        'walk_endpoint_solved': bool(env.is_solved(current))})
    return targets


def audit_target(job: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic BFS on one pair and independently replay any witness."""
    if job['status'] != 'ready':
        return job
    source = from_rows(job['source'])
    target = from_rows(job['target'])
    env = environment(source)
    audit = audit_subgoal(env, source, target, cap=job['depth_limit'], max_states=job['max_states'])
    witness = list(audit.actions) if audit.actions is not None else None
    verified = False
    if witness is not None:
        current = source.copy()
        for action in witness:
            if env.is_solved(current):
                raise ValueError('BFS witness continued after task termination')
            current = np.asarray(env.next_state(current, action))
        verified = bool(np.array_equal(current, target))
        if not verified:
            raise ValueError('BFS witness did not replay to the requested target')
    return {**job, 'status': audit.status, 'bfs_distance': len(witness) if witness is not None else None,
            'bfs_witness_actions': witness, 'witness_replayed': verified,
            'expanded_states': audit.expanded_states, 'simulator_probes': audit.simulator_probes}


def run_jobs(jobs: list[dict[str, Any]], workers: int) -> list[dict[str, Any]]:
    """Parallelize independent searches while preserving deterministic result order."""
    if workers < 1:
        raise ValueError('workers must be positive')
    if workers == 1:
        return [audit_target(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(audit_target, jobs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                        help='Must be the campaign 32x32 component-validation manifest')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--boards', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--offsets', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--depth-limit', type=int, default=8)
    parser.add_argument('--max-states', type=int, default=2500)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.source.resolve() != DEFAULT_SOURCE.resolve():
        raise ValueError('Only the specified component-validation split is permitted')
    if args.boards < 1 or args.depth_limit < 0 or args.max_states < 1:
        raise ValueError('Invalid board, depth or state limit')
    if args.offsets != sorted(set(args.offsets)) or any(value < 1 for value in args.offsets):
        raise ValueError('Offsets must be distinct, positive and increasing')
    source_bytes = args.source.read_bytes()
    instances = json.loads(source_bytes)['instances']
    if args.boards > len(instances):
        raise ValueError('Requested more boards than available in validation')
    jobs: list[dict[str, Any]] = []
    for instance in instances[:args.boards]:
        if instance['family'] != FAMILY or instance['split'] != 'component_validation':
            raise ValueError('Unexpected board family or split')
        if state_hash(from_rows(instance['state'])) != instance['state_hash']:
            raise ValueError('Validation board state hash mismatch')
        for target in random_targets(instance['state'], instance['id'], args.seed, args.offsets):
            jobs.append({**target, 'depth_limit': args.depth_limit, 'max_states': args.max_states})
    results = run_jobs(jobs, args.workers)
    counts = Counter(row['status'] for row in results)
    report = {'description': 'offline simulator-only random walks and shortest-path bounded BFS',
              'family': FAMILY, 'split': 'component_validation', 'provider': 'none',
              'source_manifest': str(args.source.resolve()), 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
              'seed': args.seed, 'offsets': args.offsets, 'depth_limit': args.depth_limit,
              'max_states': args.max_states, 'workers': args.workers,
              'walk_semantics': 'uniform actions UP, DOWN, LEFT, RIGHT; no-ops counted; stop on task terminal',
              'bfs_semantics': 'exact shortest simulator distance when reachable; ordered actions 0,1,2,3',
              'warning': 'BFS distance is not mean policy first-hit time d or policy-relative percentile u.',
              'status_counts': dict(counts), 'results': results}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'report.json'
    if path.exists() and json.loads(path.read_text()) != report:
        raise ValueError('Existing report differs; choose a new output directory')
    if not path.exists():
        write_json(path, report)
    # This separate handoff excludes random actions, BFS distances and witnesses.
    # A CLLP must receive only source and target, not experiment metadata.
    policy_input = {'description': 'offline CLLP evaluation handoff; send only source and target to the policy',
                    'cases': [{'board_id': row['board_id'], 'offset_id': row['offset'],
                               'source': row['source'], 'target': row['target']}
                              for row in results if row['status'] != 'terminal_before_offset']}
    policy_path = args.output / 'policy_inputs.json'
    if policy_path.exists() and json.loads(policy_path.read_text()) != policy_input:
        raise ValueError('Existing policy handoff differs; choose a new output directory')
    if not policy_path.exists():
        write_json(policy_path, policy_input)
    print(json.dumps({'report': str(path), 'boards': args.boards, 'targets': len(results),
                      'status_counts': dict(counts), 'policy_inputs': str(policy_path)}, indent=2))


if __name__ == '__main__':
    main()
