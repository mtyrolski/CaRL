"""Matched exploratory generator-prompt comparison on full validation boards.

The additional 32x32 training illustration demonstrates only notation and a
verified action. It is not assigned a calibrated percentile or task label.
"""
import argparse
import copy
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import canonical_layout_id, from_rows, state_hash, to_rows, validate_subgoal
from carl.icl.config import ROOT, Limits, load_config
from carl.icl.datasets import environment, load_data, reverse_board, verify_solution
from carl.icl.experiment import calibration_path, components_for_run, credentials
from carl.icl.luna import LunaComponents
from scripts.archive.campaign_20261102.campaign_components_validation import CapturingAPI, classify
from scripts.archive.campaign_20261102.campaign_policy_validation import result_row, seed_for

DESIGN = ROOT / 'configs/icl/campaign_data/generator_prompt_v1.json'
DATA = ROOT / 'data/icl/campaign_dataset_v2'
OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/generator_prompt_v1'
EXAMPLE = OUTPUT / 'train32_illustration.json'
TAG_PREFIX = 'gen_prompt_v1:'


def sha(path: Path) -> str:
    """Hash an artifact without exposing it in logs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_illustration(design: dict[str, Any]) -> dict[str, Any]:
    """Create one independent training board and only its first legal transition."""
    board, actions, meta = reverse_board(32, 10, design['illustration_seed'])
    verify_solution(board, actions)
    env = environment(board)
    successor = np.asarray(env.next_state(board, actions[0]))
    if np.array_equal(board, successor):
        raise ValueError('Illustration action unexpectedly left the board unchanged')
    validate_subgoal(board, successor)
    layout = canonical_layout_id(board)
    # Manifest inspection is programmatic; no held-out boards or witnesses are printed.
    for family in design['families']:
        for split in ('component_validation', 'scale_validation', 'pilot100', 'holdout1000'):
            source = DATA / family / f'{split}.json'
            entries = json.loads(source.read_text())['instances']
            if any(item['layout_id'] == layout for item in entries):
                raise ValueError(f'Training illustration duplicates {family}/{split}')
    return {'role': 'training-only unscaled format and transition illustration', 'seed': design['illustration_seed'],
            'generator': 'carl.icl.datasets.reverse_board-v1', 'generator_metadata': meta,
            'layout_id': layout, 'state_hash': state_hash(board), 'action': int(actions[0]),
            'source': to_rows(board), 'successor': to_rows(successor)}


def variant_prompt(design: dict[str, Any], illustration: dict[str, Any]) -> str:
    """Append an explicitly unscaled illustration without inventing a u value."""
    example = {'source': illustration['source'], 'action': ('UP', 'DOWN', 'LEFT', 'RIGHT')[illustration['action']],
               'successor': illustration['successor']}
    return (design['generator_prompt'] + '\nThe following independent 32x32 training transition illustrates '
            'only board notation and a legal single move; it has NO percentile label and is NOT an example of '
            'the requested scale. Do not copy its board into your answer:\n' + json.dumps(example))


def reference_for(config: Any, provider: str, training: Any, source: np.ndarray) -> LunaComponents:
    """Use real frozen labels with mock output only for offline plumbing tests."""
    reference = components_for_run(config, 'replay' if provider == 'mock' else provider, training, source)
    if provider == 'mock':
        reference.calibrated = copy.deepcopy(reference.calibrated)
        if reference.calibrated is None:
            raise ValueError('Missing calibration')
        reference.calibrated['provider'] = 'mock'
    return reference


def summarize(rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    """Keep all requests in the denominator, including copies, empty output and errors."""
    summary: dict[str, Any] = {'complete': len(rows) == expected, 'cases': len(rows), 'expected': expected,
                               'variants': {}}
    for variant in ('baseline', 'clear_prompt_with_unscaled_train32'):
        selected = [row for row in rows if row['variant'] == variant]
        candidates = [item for row in selected for item in row['candidates']]
        summary['variants'][variant] = {'requests': len(selected),
            'reached_any': sum(bool(row['reached_any']) for row in selected),
            'errors': sum(row['error'] is not None for row in selected),
            'parse_failures': sum(not row['parse_ok'] for row in selected),
            'empty_responses': sum(row['parse_ok'] and not row['candidates'] for row in selected),
            'static_invalid': sum(not item['static_valid'] for item in candidates),
            'self_copies': sum(item['source_copy'] for item in candidates),
            'duplicates': sum(item['duplicate'] for item in candidates),
            'distinct_valid': sum(item['distinct_valid'] for item in candidates)}
    return summary


class LimitedCapture(CapturingAPI):
    """Use the campaign ledger plus a conservative experiment-specific spending ceiling."""

    def __init__(self, *args: Any, cap_usd: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cap_nanos = int(cap_usd * 1_000_000_000)
        self.payloads: list[dict[str, Any]] = []

    def call(self, endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
        self.payloads.append({'endpoint': endpoint, 'payload': payload})
        if self.ledger is not None and self.provider == 'openai':
            # This harness holds an exclusive lock, so no other instance can use its tags.
            # Include uncertain attempts, and reserve the same worst case as CampaignLedger.
            with sqlite3.connect(self.ledger.path) as db:
                used = db.execute("SELECT COALESCE(SUM(COALESCE(charged_nanos,reserved_nanos)),0) "
                                  "FROM attempts WHERE tag LIKE ?", (TAG_PREFIX + '%',)).fetchone()[0]
            price = self.ledger.config['models'][payload['model']]
            reserve = (float(price['cache_write_12h_usd_per_million']) * 2 * price['max_input_tokens'] +
                       float(price['output_usd_per_million']) * 1.5 * payload['max_output_tokens'])
            reserve *= float(price['regional_multiplier']) * 1000
            if used + int(np.ceil(reserve)) > self.cap_nanos:
                raise BudgetExceeded('Exploratory generator comparison spending limit would be exceeded')
        return super().call(endpoint, payload, mock)


def run_case(config: Any, training: Any, instance: Any, provider: str, variant: str,
             scale: float, design: dict[str, Any]) -> dict[str, Any]:
    """Capture raw generator output and evaluate all distinct candidates with frozen CLLP."""
    source = from_rows(instance['state'])
    api = LimitedCapture(config.path('cache_dir'), provider, config.limits, cap_usd=design['max_committed_usd'])
    api.budget_tag = f'{TAG_PREFIX}{variant}:{config.family}:{scale}'
    reference = reference_for(config, provider, training, source)
    comp = LunaComponents(config, api, environment(source), training, calibrated=reference.calibrated)
    row: dict[str, Any] = {'variant': variant, 'board_id': instance['id'], 'family': config.family,
                           'scale': scale, 'seed': design['seed'], 'error': None, 'reached_any': False}
    try:
        proposals = comp.propose(source, scale)
    except (RequestFailed, BudgetExceeded) as exc:
        proposals = []
        row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    row['generator_payloads'], row['raw_responses'] = api.payloads, api.responses
    row.update(classify(api.responses, source, config.generator_layout_schema))
    row['generator_metrics'] = dict(api.metrics)
    row['diagnostics'] = dict(comp.diagnostics)
    row['proposals'] = []
    if row['error'] is None and sum(item['distinct_valid'] for item in row['candidates']) != len(proposals):
        raise ValueError('Independent classification disagrees with production decoder')
    for index, target in enumerate(proposals):
        policy_api = LimitedCapture(config.path('cache_dir'), provider, config.limits, cap_usd=design['max_committed_usd'])
        policy_api.budget_tag = f'{TAG_PREFIX}{variant}:{config.family}:{scale}:cllp'
        policy = LunaComponents(config, policy_api, environment(source), training, calibrated=reference.calibrated)
        seed = seed_for(instance['id'], int(scale * 100), index + design['seed'])
        result = result_row(policy, source, target, seed)
        endpoint = result['steps'][-1]['successor'] if result['steps'] else instance['state']
        row['proposals'].append({'target': to_rows(target), 'sampling_seed': seed, 'rollout': result,
                                 'actual_endpoint': endpoint, 'metrics': dict(policy_api.metrics)})
        row['reached_any'] |= result['exact_hit']
        if result['error'] is not None:
            row['error'] = result['error']
            break
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('openai', 'replay', 'mock'), default='replay')
    parser.add_argument('--prepare', action='store_true', help='Create and verify manifest without model calls')
    parser.add_argument('--execute', action='store_true', help='Run the saved comparison and resume completed cases')
    args = parser.parse_args()
    if args.prepare == args.execute:
        parser.error('Choose exactly one of --prepare or --execute')
    os.environ['CARL_CAMPAIGN_STAGE'] = 'scale_validation'
    design = json.loads(DESIGN.read_text())
    output = OUTPUT / args.provider
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / 'run.lock'), timeout=1):
        illustration = prepare_illustration(design) if not EXAMPLE.exists() else json.loads(EXAMPLE.read_text())
        # Verify the saved transition even when the example has been prepared before.
        src, dst = from_rows(illustration['source']), from_rows(illustration['successor'])
        if (illustration['seed'] != design['illustration_seed'] or
                not np.array_equal(environment(src).next_state(src, illustration['action']), dst) or
                canonical_layout_id(src) != illustration['layout_id'] or state_hash(src) != illustration['state_hash']):
            raise ValueError('Saved training illustration failed integrity checks')
        if not EXAMPLE.exists():
            write_json(EXAMPLE, illustration)
        base = load_config('configs/icl/campaign_data/relative_u.json')
        cases = []
        for family in design['families']:
            config = base.model_copy(update={'family': family, 'split': design['split'], 'instances': 1,
                'scales': design['scales'], 'num_subgoals': design['subgoals'],
                'limits': Limits(max_requests=24, max_input_tokens=16000, max_total_input_tokens=384000,
                                 max_output_tokens=53248, wall_seconds=1200, timeout_seconds=90)})
            training, instances = load_data(config)
            reference = reference_for(config, args.provider, training, from_rows(instances[0]['state']))
            cases.append((family, config, training, instances[0], reference))
        paths = [DESIGN, EXAMPLE, Path(__file__), ROOT / 'carl/icl/luna.py', ROOT / 'carl/icl/search.py',
                 ROOT / 'carl/icl/campaign_budget.py', ROOT / 'scripts/archive/campaign_20261102/campaign_components_validation.py',
                 ROOT / 'scripts/archive/campaign_20261102/campaign_policy_validation.py', DATA / 'training.json',
                 calibration_path(cases[0][1], cases[0][4])]
        for family in design['families']:
            paths.append(DATA / family / f"{design['split']}.json")
        manifest = {'provider': args.provider, 'design': design, 'hashes': {str(p): sha(p) for p in paths},
                    'selected_boards': {family: {'id': instance['id'], 'state_hash': state_hash(from_rows(instance['state']))}
                                        for family, _, _, instance, _ in cases},
                    'policy_fingerprint': cases[0][4].fingerprint(),
                    'selected_example_ids': cases[0][4].calibrated['selected_example_ids'],
                    'illustration': 'Unscaled separate training example, not a calibrated trajectory target',
                    'variants': ['baseline', 'clear_prompt_with_unscaled_train32'],
                    'denominator': 'all requests, including empty, invalid, copies, and failures'}
        manifest_path = output / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Saved manifest does not match current inputs; do not overwrite it')
        if not manifest_path.exists():
            if not args.prepare:
                raise ValueError('Run --prepare before --execute')
            write_json(manifest_path, manifest)
        if args.prepare:
            print(f'Prepared immutable manifest: {manifest_path}', flush=True)
            return
        credentials(args.provider)
        rows: list[dict[str, Any]] = []
        for family, config, training, instance, _ in cases:
            for scale in design['scales']:
                for variant in manifest['variants']:
                    file = output / f'{family}_{scale}_{variant}.json'
                    if file.exists():
                        row = json.loads(file.read_text())
                        rows.append(row)
                        if row['error'] is not None:
                            print(f'Saved failure blocks automatic resume: {file}', flush=True)
                            return
                        continue
                    configured = config.model_copy(deep=True)
                    if variant != 'baseline':
                        configured = configured.model_copy(update={'prompts': configured.prompts.model_copy(
                            update={'generator': variant_prompt(design, illustration)})})
                    row = run_case(configured, training, instance, args.provider, variant, scale, design)
                    write_json(file, row)
                    rows.append(row)
                    write_json(output / 'summary.json', summarize(rows, len(cases) * len(design['scales']) * 2))
                    print(f"{family} u={scale} {variant}: reached={row['reached_any']} error={row['error']}", flush=True)
                    if row['error'] is not None:
                        return


if __name__ == '__main__':
    main()
