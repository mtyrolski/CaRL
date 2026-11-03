"""Run the matched, resumable campaign policy comparison on fresh validation boards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows, state_hash, to_rows
from carl.icl.campaign_budget import CampaignLedger
from carl.icl.campaign_data import sha256_file
from carl.icl.config import Limits, load_config
from carl.icl.datasets import environment, load_data, verify_solution
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.search import rollout
from carl.icl.trace import RecordingEnvironment, RecordingPolicy

ROOT = Path('data/icl/campaign_policy_validation_v1')
PRIVATE = Path('data/icl/campaign_policy_validation_v1_private/component_validation/real_12x12x4.json')
OUTPUT = Path('analysis/luna_icl/campaign_100/policy_validation')
VARIANTS = {'baseline': 'policy_validation_base.json', 'coordinates': 'policy_validation_coordinates.json',
            'coordinates_low': 'policy_validation_coordinates_low.json'}
ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT')
OFFSETS = (1, 2, 4, 8)
CAP = 8
SEED = 11


def seed_for(board_id: str, position: int, offset: int) -> int:
    """Use common random numbers across variants, independent of execution order."""
    return int(hashlib.sha256(f'{SEED}:{board_id}:{position}:{offset}'.encode()).hexdigest()[:8], 16)


def result_row(components: LunaComponents, source: np.ndarray, target: np.ndarray, seed: int) -> dict[str, Any]:
    """Record the real solver rollout, including errors and all attempted transitions."""
    policy = RecordingPolicy(components)
    env = RecordingEnvironment(components.env, policy)
    result: dict[str, Any] = {'exact_hit': False, 'task_solved_early': False, 'error': None}
    try:
        attempt = rollout(env, policy, source, target, cap=CAP, rng=np.random.default_rng(seed))
        result['exact_hit'] = bool(attempt.reached)
        result['task_solved_early'] = bool(attempt.solved and not attempt.reached)
    except (RequestFailed, BudgetExceeded) as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    seen = {state_hash(source)}
    steps = []
    for step in env.steps:
        digest = state_hash(step.successor)
        steps.append({'action': ACTIONS[step.action], 'probabilities': step.probabilities.tolist(),
                      'no_op': not step.changed, 'repeated_state': digest in seen,
                      'successor': to_rows(step.successor)})
        seen.add(digest)
    result.update(steps=steps, attempts=len(steps), no_ops=sum(step['no_op'] for step in steps),
                  repeated_states=sum(step['repeated_state'] for step in steps),
                  first_hit=len(steps) if result['exact_hit'] else None)
    return result


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep incomplete and errored cases explicit; score only complete variants."""
    groups: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row['variant'] == variant]
        by_k = {}
        for k in OFFSETS:
            sample = [row for row in selected if row['offset'] == k]
            by_k[str(k)] = {'hits': sum(row['exact_hit'] for row in sample), 'cases': len(sample),
                            'errors': sum(row['error'] is not None for row in sample)}
        complete = len(selected) == 16 and all(row['error'] is None for row in selected)
        groups[variant] = {'by_offset': by_k, 'complete': complete,
                           'balanced_exact_hit_rate': sum(row['exact_hit'] for row in selected) / 16 if complete else None}
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('mock', 'replay', 'openai'), default='replay')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'report.json')
    args = parser.parse_args()
    if args.provider == 'openai':
        os.environ['CARL_CAMPAIGN_STAGE'] = 'policy_validation'
    credentials(args.provider)
    configs = {variant: load_config('configs/icl/campaign_data/' + filename) for variant, filename in VARIANTS.items()}
    for config in configs.values():
        config.limits = Limits(max_requests=8, max_input_tokens=20000, max_total_input_tokens=160000,
                               max_output_tokens=16384, wall_seconds=1200, timeout_seconds=90)
    training, instances = load_data(configs['baseline'])
    witnesses = {item['board_id']: item['actions'] for item in json.loads(PRIVATE.read_text())['witnesses']}
    selections = json.loads((ROOT / 'selection.json').read_text())['selections']
    contract = {'provider': args.provider, 'manifest_sha256': sha256_file(ROOT / 'campaign_dataset_manifest.json'),
                'selection_sha256': sha256_file(ROOT / 'selection.json'), 'private_sha256': sha256_file(PRIVATE),
                'configs': {name: config.model_dump(mode='json', exclude={'locked_policy_fingerprint'}) for name, config in configs.items()},
                'training_sha256': sha256_file(ROOT / 'training.json'), 'cap': CAP, 'seed': SEED}
    report = json.loads(args.output.read_text()) if args.output.exists() else {'contract': contract, 'cases': []}
    if report['contract'] != contract:
        raise ValueError('Existing report has a different provider or experiment contract')
    done = {(r['variant'], r['board_id'], r['offset']) for r in report['cases'] if r['error'] is None}
    instance_map = {item['id']: item for item in instances}
    for selected in selections:
        board_id, pos = selected['board_id'], selected['position']
        states = verify_solution(from_rows(instance_map[board_id]['state']), witnesses[board_id])
        for k in OFFSETS:
            for variant, config in configs.items():
                if (variant, board_id, k) in done:
                    continue
                os.environ['CARL_CAMPAIGN_TAG'] = variant
                api = ExperimentAPI(config.path('cache_dir'), args.provider, config.limits)
                components = LunaComponents(config, api, environment(states[pos]), training)
                row = {'variant': variant, 'board_id': board_id, 'layout_id': selected['layout_id'],
                       'position': pos, 'labels': selected['labels'], 'offset': k,
                       'sampling_seed': seed_for(board_id, pos, k), 'cap': CAP,
                       'source': to_rows(states[pos]), 'target': to_rows(states[pos + k]),
                       'policy_fingerprint': components.fingerprint()}
                row.update(result_row(components, states[pos], states[pos + k], row['sampling_seed']))
                row['metrics'] = dict(api.metrics)
                previous = [old for old in report['cases'] if (old['variant'], old['board_id'], old['offset']) == (variant, board_id, k)]
                report.setdefault('error_history', []).extend(old for old in previous if old['error'] is not None)
                report['cases'] = [old for old in report['cases'] if (old['variant'], old['board_id'], old['offset']) != (variant, board_id, k)]
                report['cases'].append(row)
                report['summary'] = summary(report['cases'])
                report['status'] = 'running' if row['error'] is None else 'interrupted_error'
                if args.provider == 'openai':
                    report['ledger'] = CampaignLedger().summary()
                write_json(args.output, report)
                print(f"{variant}: k={k}, board={board_id[-8:]}, hit={row['exact_hit']}, "
                      f"steps={row['attempts']}, error={row['error']}", flush=True)
                if row['error'] is not None:
                    return
    report['status'] = 'completed'
    write_json(args.output, report)


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / 'main.lock'), timeout=0):
        main()
