"""Evaluate frozen CLLP on random-walk 32x32 targets, with no generator or solver."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows, state_hash, to_rows
from carl.icl.config import ROOT, Limits, load_config
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.search import rollout
from carl.icl.trace import RecordingEnvironment, RecordingPolicy

DESIGN = ROOT / 'configs/icl/campaign_data/bfs32_cllp_diagnostic.json'
OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/random_walk_bfs_32/cllp_v3'
ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT')


def sha(path: Path) -> str:
    """Hash an experiment input without logging its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(seed: int, board_id: str, offset: int) -> int:
    return int.from_bytes(hashlib.sha256(f'{seed}:{board_id}:{offset}'.encode()).digest()[:8], 'big')


def selected_cases(design: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select known-reachable targets without feeding BFS metadata into inference."""
    handoff = json.loads((ROOT / design['policy_inputs']).read_text())['cases']
    audits = json.loads((ROOT / design['bfs_report']).read_text())['results']
    by_key = {(row['board_id'], row['offset']): row for row in audits}
    selected, excluded = [], []
    for case in handoff:
        key = (case['board_id'], case['offset_id'])
        source, target = from_rows(case['source']), from_rows(case['target'])
        if np.array_equal(source, target):
            excluded.append({'board_id': key[0], 'offset': key[1], 'reason': 'source_equals_target'})
            continue
        if case['offset_id'] not in design['offsets']:
            continue
        audit = by_key.get(key)
        if audit is None or audit['status'] != 'reachable' or not audit['witness_replayed']:
            raise ValueError(f'Missing verified reachable BFS target for {key}')
        if state_hash(source) != audit['source_hash'] or state_hash(target) != audit['target_hash']:
            raise ValueError('BFS report and policy input state hashes disagree')
        selected.append({**case, 'bfs_distance': audit['bfs_distance'], 'source_hash': state_hash(source),
                         'target_hash': state_hash(target), 'sampling_seed': seed_for(design['seed'], *key)})
    if not selected or len({(row['board_id'], row['offset_id']) for row in selected}) != len(selected):
        raise ValueError('No cases selected or repeated board/offset IDs')
    return selected, excluded


class DiagnosticAPI(ExperimentAPI):
    """Apply a small shared experiment cap in addition to the campaign ledger."""

    def __init__(self, *args: Any, cap_usd: float, tag_prefix: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cap_nanos = round(cap_usd * 1_000_000_000)
        self.tag_prefix = tag_prefix

    def call(self, endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
        if self.provider == 'mock':
            # Do not call the normal heuristic mock, which probes four successors.
            def uniform_response() -> dict[str, Any]:
                return {'output_text': '{"probabilities":[0.25,0.25,0.25,0.25]}',
                        'model': 'mock_all_components', 'usage': {}}

            mock = uniform_response
        if self.provider == 'openai':
            assert self.ledger is not None
            with closing(sqlite3.connect(self.ledger.path)) as db:
                used = db.execute('SELECT COALESCE(SUM(COALESCE(charged_nanos,reserved_nanos)),0) '
                                  'FROM attempts WHERE stage=? AND tag LIKE ?',
                                  (self.stage, self.tag_prefix + '%')).fetchone()[0]
            price = self.ledger.config['models'][payload['model']]
            estimate = (float(price['cache_write_12h_usd_per_million']) * 2 * price['max_input_tokens']
                        + float(price['output_usd_per_million']) * 1.5 * payload['max_output_tokens'])
            reserve_nanos = int(np.ceil(estimate * float(price['regional_multiplier']) * 1000))
            if used + reserve_nanos > self.cap_nanos:
                raise BudgetExceeded('BFS32 CLLP diagnostic cost cap would be exceeded')
        return super().call(endpoint, payload, mock)


def verify_trace(source: np.ndarray, target: np.ndarray, steps: list[dict[str, Any]], exact_hit: bool) -> bool:
    """Explicitly replay returned actions, including no-ops, after the rollout."""
    env = environment(source)
    state = source.copy()
    for index, step in enumerate(steps):
        if env.is_solved(state) or np.array_equal(state, target):
            raise ValueError('Trace continued after an exact hit or task terminal')
        state = np.asarray(env.next_state(state, ACTIONS.index(step['action'])))
        if to_rows(state) != step['successor']:
            raise ValueError(f'Trace endpoint mismatch at step {index}')
    if bool(np.array_equal(state, target)) != exact_hit:
        raise ValueError('Trace exact-hit flag mismatch')
    return True


def ledger_rows(api: DiagnosticAPI) -> dict[str, Any]:
    """Include physical attempts and cache events for this exact case."""
    if api.ledger is None:
        return {'attempts': [], 'events': []}
    with closing(sqlite3.connect(api.ledger.path)) as db:
        attempts = db.execute('SELECT status,request_hash,reserved_nanos,charged_nanos,usage_json FROM attempts '
                              'WHERE stage=? AND tag=? ORDER BY rowid', (api.stage, api.budget_tag)).fetchall()
        events = db.execute('SELECT kind,request_hash FROM events WHERE stage=? AND tag=? ORDER BY id',
                            (api.stage, api.budget_tag)).fetchall()
    return {'attempts': [{'status': status, 'request_hash': digest, 'reserved_nanos': reserve,
                          'charged_nanos': charged, 'usage': json.loads(usage) if usage else None}
                         for status, digest, reserve, charged, usage in attempts],
            'events': [{'kind': kind, 'request_hash': digest} for kind, digest in events]}


def run_case(case: dict[str, Any], design: dict[str, Any], provider: str, cache_dir: Path) -> dict[str, Any]:
    """Use the production rollout and frozen policy for one source/target pair."""
    config = load_config(design['policy_config'])
    config.limits = Limits(max_requests=16, max_input_tokens=16000, max_total_input_tokens=256000,
                           max_output_tokens=32768, wall_seconds=1200, timeout_seconds=90)
    training, _ = load_data(config)
    source, target = from_rows(case['source']), from_rows(case['target'])
    api = DiagnosticAPI(cache_dir, provider, config.limits, cap_usd=float(design['max_committed_usd']),
                        tag_prefix=design['ledger_tag_prefix'])
    api.budget_tag = f"{design['ledger_tag_prefix']}{case['board_id']}:{case['offset_id']}"
    component = LunaComponents(config, api, environment(source), training)
    payload = component.policy_payload(source, target)
    question = json.loads(payload['input'][-1]['content'][0]['text'])
    if set(question) != {'state', 'subgoal', 'state_coordinates', 'subgoal_coordinates'}:
        raise ValueError('Frozen policy query contains unexpected fields')
    if question['state'] != case['source'] or question['subgoal'] != case['target']:
        raise ValueError('Policy query does not contain the intended source and target')
    policy = RecordingPolicy(component)
    env = RecordingEnvironment(component.env, policy)
    attempt = None
    error = None
    try:
        attempt = rollout(env, policy, source, target, cap=design['action_cap'],
                          rng=np.random.default_rng(case['sampling_seed']))
    except (RequestFailed, BudgetExceeded, KeyboardInterrupt) as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        if api.client is not None:
            api.client.close()
    steps = [{'action': ACTIONS[step.action], 'probabilities': step.probabilities.tolist(),
              'no_op': not step.changed, 'successor': to_rows(step.successor)} for step in env.steps]
    exact_hit = bool(attempt.reached) if attempt is not None else False
    result = {'board_id': case['board_id'], 'offset_id': case['offset_id'], 'bfs_distance': case['bfs_distance'],
              'source_hash': case['source_hash'], 'target_hash': case['target_hash'], 'seed': case['sampling_seed'],
              'policy_fingerprint': component.fingerprint(), 'provider': provider, 'action_cap': design['action_cap'],
              'exact_hit': exact_hit, 'first_hit': len(steps) if exact_hit else None,
              'task_solved_early': bool(attempt.solved and not attempt.reached) if attempt is not None else False,
              'interrupted': error is not None, 'error': error, 'actions_attempted': len(steps),
              'no_ops': sum(step['no_op'] for step in steps), 'steps': steps,
              'metrics': dict(api.metrics), 'ledger': ledger_rows(api),
              'trace_replayed': verify_trace(source, target, steps, exact_hit)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('mock', 'replay', 'openai'), default='replay')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true', help='Write an immutable manifest without inference')
    mode.add_argument('--execute', action='store_true', help='Run the explicitly selected provider')
    args = parser.parse_args()
    design = json.loads(DESIGN.read_text())
    config = load_config(design['policy_config'])
    if (config.model != 'gpt-6-luna' or config.json_max_retries != 1 or
            design['action_cap'] != config.calibration.action_cap or not config.policy_coordinate_features):
        raise ValueError('The diagnostic must use the frozen retry CLLP and its L')
    cases, excluded = selected_cases(design)
    output = args.output.resolve() / args.provider
    paths = [DESIGN, ROOT / design['policy_config'], ROOT / design['policy_inputs'], ROOT / design['bfs_report'],
             ROOT / 'data/icl/campaign_dataset_v2/training.json', ROOT / 'carl/icl/luna.py',
             ROOT / 'carl/icl/search.py', ROOT / 'carl/icl/api.py', ROOT / 'carl/icl/experiment_api.py',
             ROOT / 'carl/icl/trace.py', Path(__file__)]
    manifest = {'version': 1, 'provider': args.provider, 'design': design,
                'resolved_policy_config': config.model_dump(mode='json'),
                'inputs_sha256': {str(path.resolve()): sha(path) for path in paths},
                'cases': [{k: row[k] for k in ('board_id', 'offset_id', 'source_hash', 'target_hash', 'sampling_seed')}
                          for row in cases], 'excluded': excluded,
                'prompt_contract': 'source and target plus coordinates computed only from those boards; '
                                   'no BFS metadata, random actions, simulator probes, distances or witnesses',
                'mock_note': 'uniform distribution, no hidden simulator probes; offline plumbing only'}
    output.mkdir(parents=True, exist_ok=True)
    # One shared lock covers the tag-based diagnostic spending limit, even with different output paths.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / '.diagnostic.lock'), timeout=0):
        manifest_path = output / 'manifest.json'
        if args.prepare:
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('Existing manifest differs; use a new output directory')
            if not manifest_path.exists():
                write_json(manifest_path, manifest)
            print(json.dumps({'manifest': str(manifest_path), 'cases': len(cases), 'excluded': excluded,
                              'max_committed_usd': design['max_committed_usd']}, indent=2))
            return
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Prepare the exact provider manifest before executing')
        if args.provider in {'openai', 'replay'}:
            os.environ['CARL_CAMPAIGN_STAGE'] = design['ledger_stage']
        credentials(args.provider)
        cache = output / 'mock_cache' if args.provider == 'mock' else config.path('cache_dir')
        completed = []
        summary: dict[str, Any] = {}
        for case in cases:
            path = output / f"{case['board_id']}_offset{case['offset_id']}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if (row['source_hash'], row['target_hash'], row['seed']) != (
                        case['source_hash'], case['target_hash'], case['sampling_seed']):
                    raise ValueError('Saved case does not match the manifest')
                if row['interrupted']:
                    raise ValueError('Saved interrupted case requires inspection; no automatic rerun')
            else:
                row = run_case(case, design, args.provider, cache)
                write_json(path, row)
            completed.append(row)
            summary = {'status': 'interrupted' if row['interrupted'] else 'running',
                       'expected': len(cases), 'completed': sum(not item['interrupted'] for item in completed),
                       'interrupted': sum(item['interrupted'] for item in completed),
                       'missing': len(cases) - len(completed),
                       'exact_hits': sum(item['exact_hit'] for item in completed),
                       'logical_requests': sum(item['metrics']['logical_requests'] for item in completed),
                       'physical_requests': sum(item['metrics']['requests'] for item in completed),
                       'input_tokens': sum(item['metrics']['input_tokens'] for item in completed),
                       'output_tokens': sum(item['metrics']['output_tokens'] for item in completed),
                       'cases': [{'board_id': item['board_id'], 'offset_id': item['offset_id'],
                                  'bfs_distance': item['bfs_distance'], 'exact_hit': item['exact_hit'],
                                  'interrupted': item['interrupted']} for item in completed]}
            write_json(output / 'summary.json', summary)
            print(f"{case['board_id']} offset={case['offset_id']} hit={row['exact_hit']} "
                  f"actions={row['actions_attempted']} interrupted={row['interrupted']}", flush=True)
            if row['interrupted']:
                return
        summary['status'] = 'completed'
        write_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
