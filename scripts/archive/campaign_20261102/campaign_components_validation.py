"""Matched generator and value validation on verified campaign validation trajectories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

from carl.icl.api import BudgetExceeded, RequestFailed, request_cache_digest, write_json
from carl.icl.boards import from_rows, state_hash, to_rows, validate_subgoal
from carl.icl.config import ExperimentConfig, Limits, load_config
from carl.icl.datasets import environment, load_data, verify_solution
from carl.icl.diagnostics import audit_subgoal
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from scripts.archive.campaign_20261102.campaign_policy_validation import result_row, seed_for

OUTPUT = Path('analysis/luna_icl/campaign_100/components_validation')
DATASETS = (
    ('data/icl/campaign_policy_validation_v1', 'data/icl/campaign_policy_validation_v1_private', 'real_12x12x4'),
    ('data/icl/campaign_dataset_v2', 'data/icl/campaign_dataset_v2_private', 'generated_32x32x10'),
)
GENERATOR = {'baseline': {'generator_reasoning_effort': 'none', 'generator_response_format': 'text',
                          'generator_layout_schema': False},
             'full_rows': {'generator_reasoning_effort': 'none', 'generator_response_format': 'json_schema',
                           'generator_layout_schema': True}}
VALUES = {'initial_examples': 'initial', 'trajectory_examples': 'trajectory'}


class CapturingAPI(ExperimentAPI):
    """Record the unfiltered response before component-level parsing and validation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.responses: list[dict[str, Any]] = []

    def call(self, endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
        try:
            result = super().call(endpoint, payload, mock)
        except (RequestFailed, BudgetExceeded):
            request = {'endpoint': endpoint, 'payload': payload}
            digest = request_cache_digest(request)
            path = self.cache_dir / f'{digest}.json'
            if path.exists():
                self.responses.append(json.loads(path.read_text()))
            raise
        self.responses.append(result)
        return result


def config_for(root: str, family: str, **overrides: Any) -> ExperimentConfig:
    base = load_config('configs/icl/campaign_data/frozen_policy_fixed.json')
    values = base.model_dump()
    values.update(data_dir=root, family=family, split='component_validation', instances=4,
                  limits=Limits(max_requests=32, max_input_tokens=20000, max_total_input_tokens=500000,
                                max_output_tokens=65536, wall_seconds=1800, timeout_seconds=90).model_dump(),
                  **overrides)
    return ExperimentConfig.model_validate(values)


def validation_data(root: str, private: str, family: str) -> tuple[Any, list[tuple[str, list[np.ndarray]]]]:
    config = config_for(root, family)
    training, instances = load_data(config)
    witness_path = Path(private) / 'component_validation' / f'{family}.json'
    witnesses = {item['board_id']: item['actions'] for item in json.loads(witness_path.read_text())['witnesses']}
    states = [(item['id'], verify_solution(from_rows(item['state']), witnesses[item['id']])) for item in instances]
    return training, states


def api_for(config: ExperimentConfig, provider: str, tag: str) -> CapturingAPI:
    os.environ['CARL_CAMPAIGN_TAG'] = tag
    return CapturingAPI(config.path('cache_dir'), provider, config.limits)


def classify(raw: list[dict[str, Any]], source: np.ndarray, layout_schema: bool) -> dict[str, Any]:
    """Classify every model output before filtering; invalid replies stay in the denominator."""
    result: dict[str, Any] = {'parse_ok': False, 'candidates': []}
    if not raw or not isinstance(raw[-1].get('output_text'), str):
        return result
    try:
        decoded = json.loads(raw[-1]['output_text'])
        items = decoded['subgoals']
        if not isinstance(items, list) or len(items) > 2:
            return result
    except (ValueError, TypeError, KeyError):
        return result
    result['parse_ok'] = True
    seen: set[str] = set()
    for item in items:
        row: dict[str, Any] = {'static_valid': False, 'source_copy': False, 'duplicate': False,
                               'distinct_valid': False}
        try:
            rows = [item[f'row_{i:02d}'] for i in range(source.shape[0])] if layout_schema else item['rows']
            candidate = from_rows(rows)
            validate_subgoal(source, candidate)
            row['static_valid'] = True
            row['source_copy'] = bool(np.array_equal(candidate, source))
            digest = state_hash(candidate)
            row['duplicate'] = digest in seen
            seen.add(digest)
            row['distinct_valid'] = not row['source_copy'] and not row['duplicate']
            row['rows'] = rows
        except (TypeError, ValueError, KeyError) as exc:
            row['invalid_reason'] = str(exc)
        result['candidates'].append(row)
    return result


def generator_case(provider: str, training: Any, board_id: str, states: list[np.ndarray], k: int, variant: str) -> dict[str, Any]:
    config = config_for(DATASETS[0][0], DATASETS[0][2], **GENERATOR[variant], num_subgoals=2,
                        generator_max_attempts=1, method='fixed', scales=[8, 4])
    source = states[0]
    api = api_for(config, provider, f'generator_{variant}')
    comp = LunaComponents(config, api, environment(source), training)
    row: dict[str, Any] = {'board_id': board_id, 'k': k, 'variant': variant, 'source': to_rows(source),
                           'policy_fingerprint': comp.fingerprint(), 'error': None}
    try:
        proposals = comp.propose(source, float(k))
    except (RequestFailed, BudgetExceeded) as exc:
        proposals = []
        row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    row['raw_responses'] = api.responses
    row.update(classify(api.responses, source, config.generator_layout_schema))
    row['distinct_valid_count'] = len(proposals)
    if sum(item['distinct_valid'] for item in row['candidates']) != len(proposals):
        if row['error'] is None:
            raise RuntimeError('Independent generator classification disagrees with production decoder')
    row['reached_any'] = False
    row['proposals'] = []
    row['generator_metrics'] = dict(api.metrics)
    # BFS is an offline audit. Its result is never sent to the policy or generator.
    for index, target in enumerate(proposals):
        audit = audit_subgoal(environment(source), source, target, cap=8, max_states=10000)
        policy_api = api_for(config, provider, f'generator_{variant}_cllp')
        policy = LunaComponents(config, policy_api, environment(source), training)
        seed = seed_for(board_id, k, index)
        result = result_row(policy, source, target, seed)
        endpoint = result['steps'][-1]['successor'] if result['steps'] else to_rows(source)
        row['proposals'].append({'target': to_rows(target), 'bfs': {'status': audit.status,
                                 'expanded_states': audit.expanded_states, 'simulator_probes': audit.simulator_probes,
                                 'witness_actions': list(audit.actions) if audit.actions is not None else None},
                                 'rollout': result, 'actual_endpoint': endpoint, 'sampling_seed': seed,
                                 'policy_metrics': dict(policy_api.metrics)})
        row['reached_any'] |= result['exact_hit']
        if result['error'] is not None:
            row['error'] = result['error']
            break
    return row


def value_case(provider: str, config: ExperimentConfig, training: Any, board_id: str,
               states: list[np.ndarray], position: int, variant: str) -> dict[str, Any]:
    api = api_for(config, provider, f'value_{variant}')
    comp = LunaComponents(config, api, environment(states[position]), training)
    remaining = len(states) - 1 - position
    row: dict[str, Any] = {'variant': variant, 'board_id': board_id, 'family': config.family,
                           'position': position, 'remaining': remaining, 'gamma': config.discount,
                           'reference_value': config.discount ** remaining, 'predicted_value': None, 'error': None}
    try:
        row['predicted_value'] = comp.value(states[position])
    except (RequestFailed, BudgetExceeded) as exc:
        row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    row['raw_responses'] = api.responses
    row['metrics'] = dict(api.metrics)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('mock', 'replay', 'openai'), default='replay')
    parser.add_argument('--phase', choices=('generator', 'value', 'all'), default='all')
    parser.add_argument('--retry-errors', action='store_true', help='Retry saved errored cases and preserve their history')
    args = parser.parse_args()
    os.environ['CARL_CAMPAIGN_STAGE'] = 'scale_validation'
    credentials(args.provider)
    output = OUTPUT / args.provider
    output.mkdir(parents=True, exist_ok=True)
    input_paths = [Path('configs/icl/campaign_data/frozen_policy_fixed.json'), Path(__file__),
                   Path('carl/icl/luna.py'), Path('carl/icl/search.py'), Path('carl/icl/diagnostics.py')]
    for root, private, family in DATASETS:
        input_paths.extend([Path(root) / 'campaign_dataset_manifest.json', Path(root) / 'training.json',
                            Path(root) / 'value_examples.json',
                            Path(root) / family / 'component_validation.json',
                            Path(private) / 'component_validation' / f'{family}.json'])
    contract = {'provider': args.provider, 'inputs_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                               for path in input_paths},
                'generator_variants': GENERATOR, 'value_variants': VALUES, 'cap': 8,
                'bfs_max_states': 10000, 'generator_offsets': [4, 8]}
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError('Validation contract changed; choose a new output directory')
    if not manifest.exists():
        write_json(manifest, contract)
    prepared = [(root, family, *validation_data(root, private, family)) for root, private, family in DATASETS]
    for root, family, training, boards in prepared:
        for board_id, states in boards:
            if args.phase in ('generator', 'all') and family == 'real_12x12x4':
                for k in (4, 8):
                    for variant in GENERATOR:
                        path = output / f'generator_{board_id}_{k}_{variant}.json'
                        previous = json.loads(path.read_text()) if path.exists() else None
                        if previous is not None and not (args.retry_errors and previous['error'] is not None):
                            continue
                        row = generator_case(args.provider, training, board_id, states, k, variant)
                        if previous is not None:
                            row['prior_attempts'] = [*previous.get('prior_attempts', []), previous]
                        write_json(path, row)
                        print('generator', board_id[-8:], k, variant, 'reached', row['reached_any'],
                              'error', row['error'], flush=True)
                        if row['error'] is not None and (row['error']['type'] == 'BudgetExceeded' or
                                                         'responses failed' in row['error']['message']):
                            return
            if args.phase in ('value', 'all'):
                last = len(states) - 1
                positions = [0, last // 4, last // 2, 3 * last // 4, last]
                if len(set(positions)) != 5:
                    raise ValueError('Expected five distinct trajectory positions')
                for position in positions:
                    for variant, examples in VALUES.items():
                        path = output / f'value_{board_id}_{position}_{variant}.json'
                        previous = json.loads(path.read_text()) if path.exists() else None
                        if previous is not None and not (args.retry_errors and previous['error'] is not None):
                            continue
                        config = config_for(root, family, value_target='return', value_examples=examples,
                                            value_board_features=False, value_response_format='text', value_tokens=128)
                        row = value_case(args.provider, config, training, board_id, states, position, variant)
                        if previous is not None:
                            row['prior_attempts'] = [*previous.get('prior_attempts', []), previous]
                        write_json(path, row)
                        print('value', board_id[-8:], position, variant, 'value', row['predicted_value'],
                              'error', row['error'], flush=True)
                        if row['error'] is not None and (row['error']['type'] == 'BudgetExceeded' or
                                                         'responses failed' in row['error']['message']):
                            return


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / 'run.lock'), timeout=0):
        main()
