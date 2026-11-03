"""Prepare and resume two fixed-budget primitive-only validation runs."""
import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import calibration_path, components_for_run, credentials
from carl.icl.luna import LunaComponents
from carl.icl.search import SearchResult, search
from carl.icl.search_tree import SearchTreeRecorder, TreeNode
from scripts.tune_campaign_scales import CappedAPI, configured, failure_category, read_design, verify_tree

DESIGN = ROOT / 'configs/icl/campaign_data/scale_tuning/design.json'
OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/primitive'
FAMILIES = ('real_12x12x4', 'generated_32x32x10')
SEED = 11
BUDGET = 128


def sha(path: Path) -> str:
    """Hash a frozen input without exposing its contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primitive_config(design: dict[str, Any], family: str) -> ExperimentConfig:
    """Share all per-run limits and V settings with the preregistered scale grid."""
    if tuple(design['families']) != FAMILIES or design['seed'] != SEED or design['budget'] != BUDGET:
        raise ValueError('Primitive control requires the preregistered two families, seed 11 and B=128')
    base = configured(design, 'relative_u', 0, family, budget=BUDGET, instances=1)
    return ExperimentConfig.model_validate({**base.model_dump(), 'name': f'primitive_{family}', 'method': 'primitive'})


def build_manifest(design_path: Path, design: dict[str, Any], provider: str) -> dict[str, Any]:
    """Freeze validation IDs, policy, calibration, shared limits, dependencies and code."""
    reference_provider = 'replay' if provider == 'openai' else provider
    paths = [design_path, Path(__file__), ROOT / 'scripts/tune_campaign_scales.py']
    paths.extend(ROOT / path for path in (
        'carl/icl/search.py', 'carl/icl/search_tree.py', 'carl/icl/api.py', 'carl/icl/campaign_budget.py',
        'carl/icl/boards.py', 'carl/icl/calibration.py', 'carl/icl/config.py', 'carl/icl/datasets.py',
        'carl/icl/experiment.py', 'carl/icl/experiment_api.py', 'carl/icl/luna.py', 'carl/icl/responses.py',
        'carl/icl/types.py', 'carl/icl/prompt_defaults.py', 'carl/environment/env.py',
        'carl/environment/tokenizer.py', 'carl/environment/utilis.py', 'carl/environment/sokoban/core.py', 'carl/environment/sokoban/env.py', 'carl/environment/sokoban/tokenizer.py',
    ))
    cases, fingerprints, calibrations = [], set(), set()
    for family in FAMILIES:
        config = primitive_config(design, family)
        training, boards = load_data(config)
        item = boards[0]
        # Verify the existing frozen calibration independently; primitive search itself never requests a subgoal.
        reference = configured(design, 'relative_u', 0, family, budget=BUDGET, instances=1)
        calibrated = components_for_run(reference, reference_provider, training, from_rows(item['state']))
        policy = components_for_run(config, reference_provider, training, from_rows(item['state']))
        if calibrated.calibrated is None or policy.fingerprint() != calibrated.fingerprint():
            raise ValueError('Primitive policy/V provenance does not match the calibrated campaign')
        fingerprints.add(policy.fingerprint())
        calibrations.add(calibrated.calibrated['signature'])
        paths.extend([config.path('data_dir') / 'training.json', config.path('data_dir') / 'value_examples.json',
                      config.path('data_dir') / family / 'scale_validation.json', calibration_path(reference, calibrated),
                      ROOT / design['config_by_method']['relative_u']])
        tag = f'primitive_{family}_{item["id"]}_seed{SEED}'
        cases.append({'family': family, 'board_id': item['id'], 'state_hash': state_hash(from_rows(item['state'])),
                      'seed': SEED, 'budget': BUDGET, 'tag': tag, 'config': config.model_dump(mode='json')})
    if len(fingerprints) != 1 or len(calibrations) != 1:
        raise ValueError('Validation families do not share the same frozen policy and calibration')
    return {'version': 1, 'provider': provider, 'split': 'scale_validation', 'cases': cases,
            'policy_fingerprint': fingerprints.pop(), 'calibration_signature': calibrations.pop(),
            'max_generator_calls': design['max_generator_calls'], 'max_total_tokens': design['max_total_tokens'],
            'hashes': {str(path.resolve()): sha(path) for path in paths}}


def prepare_or_check(output: Path, manifest: dict[str, Any], *, prepare: bool) -> None:
    """Persist once, then reject any changed data, limits, provider or code."""
    path = output / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Primitive manifest changed; choose a new output directory')
    elif not prepare:
        raise ValueError('Run --prepare with the same provider before --execute')
    else:
        if any(item.name != '.lock' for item in output.iterdir()):
            raise ValueError('Output is not empty and has no manifest')
        write_json(path, manifest)


def run_one(config: ExperimentConfig, design: dict[str, Any], provider: str, training: Any,
            item: Any, tag: str) -> dict[str, Any]:
    """Record the real solver and persist its trace even if replay validation fails."""
    initial = from_rows(item['state'])
    api = CappedAPI(config.path('cache_dir'), provider, config.limits,
                    max_generator_calls=design['max_generator_calls'], max_total_tokens=design['max_total_tokens'])
    api.budget_tag = tag
    components = LunaComponents(config, api, environment(initial), training)
    tree = SearchTreeRecorder()
    try:
        outcome = search(components.env, components, initial, config, SEED, observer=tree)
    finally:
        if api.client is not None:
            api.client.close()
    try:
        verification = verify_tree(tree, initial, BUDGET)
    except (ValueError, AssertionError) as exc:
        verification = {'verified': False, 'error_type': type(exc).__name__, 'error': str(exc)}
    return {'instance': item, 'seed': SEED, 'provider': provider, 'budget_tag': tag,
            'config': config.model_dump(mode='json'), 'frozen_policy': components.fingerprint(),
            'outcome': asdict(outcome), 'failure_category': failure_category(outcome), 'api': dict(api.metrics),
            'logical_total_tokens': api.metrics['logical_input_tokens'] + api.metrics['logical_output_tokens'],
            'verification': verification, 'tree': {'nodes': [asdict(node) for node in tree.nodes],
                                                    'expansions': [asdict(entry) for entry in tree.expansions],
                                                    'solution_node': tree.solution_node}}


def restore(case: dict[str, Any], config: ExperimentConfig, item: Any, provider: str, tag: str) -> None:
    """Reject mismatched or non-replayable checkpoints; never silently retry a failure."""
    if (case['config'] != config.model_dump(mode='json') or case['instance'] != item or case['provider'] != provider
            or case['budget_tag'] != tag or case['seed'] != SEED):
        raise ValueError('Saved primitive case contract changed')
    if not case['verification'].get('verified'):
        raise ValueError('Saved primitive case failed replay; inspect its retained trace')
    tree = SearchTreeRecorder()
    tree.nodes = [TreeNode(**node) for node in case['tree']['nodes']]
    tree.result = SearchResult(**case['outcome'])
    if verify_tree(tree, from_rows(item['state']), BUDGET) != case['verification']:
        raise ValueError('Saved primitive case no longer replays')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=DESIGN)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--provider', choices=('openai', 'mock', 'replay'), default='replay')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true', help='Freeze and persist the exact manifest; never call the API')
    modes.add_argument('--execute', action='store_true', help='Run or resume only the frozen cases')
    args = parser.parse_args()
    design_path = args.design.expanduser().resolve()
    design = read_design(design_path)
    os.environ['CARL_CAMPAIGN_STAGE'] = 'scale_validation'
    credentials('replay' if args.prepare else args.provider)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.lock'), timeout=0):
        manifest = build_manifest(design_path, design, args.provider)
        prepare_or_check(output, manifest, prepare=args.prepare)
        if args.prepare:
            print(f'Frozen primitive manifest: {output / "manifest.json"}', flush=True)
            return
        cases = []
        for planned in manifest['cases']:
            family, tag = planned['family'], planned['tag']
            config = primitive_config(design, family)
            training, items = load_data(config)
            item = items[0]
            path = output / f'{tag}.json'
            if path.exists():
                case = json.loads(path.read_text())
                restore(case, config, item, args.provider, tag)
            else:
                case = run_one(config, design, args.provider, training, item, tag)
                write_json(path, case)
            cases.append(case)
            outcome = case['outcome']
            print(f"primitive {family}: {outcome['finish_reason']} actions={outcome['attempted_actions']} "
                  f"verified={case['verification']['verified']}", flush=True)
            if not case['verification']['verified'] or outcome['finish_reason'] == 'api_error' or case['failure_category'] == 'campaign_stage_guard':
                write_json(output / 'summary.json', {'status': 'interrupted', 'cases': len(cases),
                                                     'failure_category': case['failure_category']})
                return
        write_json(output / 'summary.json', {'status': 'completed', 'cases': len(cases),
                                             'solved': sum(case['outcome']['solved'] for case in cases),
                                             'actions': sum(case['outcome']['attempted_actions'] for case in cases)})


if __name__ == '__main__':
    main()
