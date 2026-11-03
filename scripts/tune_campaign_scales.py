"""Run/resume paired scale-validation solver trials with full recorded search traces."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from carl.icl.types import Instance

import numpy as np
from filelock import FileLock

from carl.icl.api import BudgetExceeded, write_json
from carl.icl.boards import from_rows, state_hash, to_rows
from carl.icl.config import ROOT, ExperimentConfig, load_config
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import calibration_path, components_for_run, credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.search import SearchResult, search
from carl.icl.search_tree import SearchTreeRecorder, TreeNode

DEFAULT_DESIGN = 'configs/icl/campaign_data/scale_tuning/design.json'
DEFAULT_OUTPUT = 'analysis/luna_icl/campaign_100/scale_tuning/solver'


class CappedAPI(ExperimentAPI):
    """Limit logical generator requests as well as normal per-run API limits."""

    def __init__(self, *args: Any, max_generator_calls: int, max_total_tokens: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_generator_calls = max_generator_calls
        self.max_total_tokens = max_total_tokens

    def call(self, endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
        if payload.get('metadata', {}).get('component') == 'generator' and self.metrics['generator_calls'] >= self.max_generator_calls:
            raise BudgetExceeded('Logical generator request limit exhausted')
        prompt_tokens = len(self.encoding.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=()))
        reserve = payload.get('max_output_tokens', 0) if endpoint == 'responses' else 0
        used = self.metrics['logical_input_tokens'] + self.metrics['logical_output_tokens']
        if used + prompt_tokens + reserve > self.max_total_tokens:
            raise BudgetExceeded('Combined logical input/output token limit exhausted')
        try:
            result = super().call(endpoint, payload, mock)
        except Exception:
            self._verify_total()
            raise
        self._verify_total()
        return result

    def _verify_total(self) -> None:
        if self.metrics['logical_input_tokens'] + self.metrics['logical_output_tokens'] > self.max_total_tokens:
            raise BudgetExceeded('Combined logical token limit exceeded after provider response')


def read_design(path: Path) -> dict[str, Any]:
    """Reject imbalanced grids and non-campaign validation scope."""
    design = json.loads(path.read_text())
    if design['version'] != 1 or set(design['config_by_method']) != {'relative_u', 'absolute_d', 'fixed'}:
        raise ValueError('Expected the three preregistered methods')
    if set(design['scale_pairs']) != set(design['config_by_method']) or any(len(v) != 2 for v in design['scale_pairs'].values()):
        raise ValueError('Each method must have exactly two scale pairs')
    if design['families'] != ['real_12x12x4', 'generated_32x32x10']:
        raise ValueError('Only the two prescribed validation families are supported')
    if not 0 < design['early_limit_block_fraction'] <= 1:
        raise ValueError('Early-limit gate must be a fraction in (0, 1]')
    if design['max_generator_calls'] < 1 or design['max_expansions'] < 1 or design['max_total_tokens'] < 1:
        raise ValueError('Per-run generation and expansion caps must be positive')
    return design


def configured(design: dict[str, Any], method: str, pair_index: int, family: str,
               *, budget: int, instances: int) -> ExperimentConfig:
    base = load_config(design['config_by_method'][method])
    settings = base.model_dump()
    settings.update(name=f'tune_{method}_{pair_index}_{family}', method=method,
                    scales=design['scale_pairs'][method][pair_index], family=family,
                    split='scale_validation', easy=False, instances=instances, repeat_seeds=[design['seed']],
                    action_budget=budget, max_expansions=design['max_expansions'], num_subgoals=design['num_subgoals'], limits=design['limits'])
    return ExperimentConfig.model_validate(settings)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(design_path: Path, design: dict[str, Any], *, budget: int, instances: int, provider: str,
                   workers: int) -> dict[str, Any]:
    if budget < 1 or instances < 1 or instances > 4 or not 1 <= workers <= 4:
        raise ValueError('Budget must be positive, instances in 1..4 and workers in 1..4')
    config = configured(design, 'relative_u', 0, design['families'][0], budget=budget, instances=instances)
    training, items = load_data(config)
    preflight_provider = 'replay' if provider == 'openai' else provider
    components = components_for_run(config, preflight_provider, training, from_rows(items[0]['state']))
    paths = [design_path, ROOT / 'carl/icl/search.py', ROOT / 'carl/icl/luna.py', ROOT / 'carl/icl/experiment.py',
             ROOT / 'carl/icl/calibration.py', ROOT / 'carl/icl/config.py', ROOT / 'carl/icl/experiment_api.py',
             ROOT / 'carl/icl/search_tree.py', ROOT / 'carl/icl/api.py', ROOT / 'carl/icl/campaign_budget.py',
             ROOT / 'carl/icl/responses.py', ROOT / 'carl/icl/types.py', ROOT / 'carl/icl/datasets.py',
             ROOT / 'carl/environment/sokoban/core.py', ROOT / 'carl/environment/sokoban/env.py',
             ROOT / 'carl/environment/sokoban/tokenizer.py', Path(__file__),
             ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/protocol.md', config.path('data_dir') / 'training.json',
             config.path('data_dir') / 'value_examples.json', calibration_path(config, components)]
    paths.extend(sorted((ROOT / 'carl/icl').rglob('*.py')))
    for family in design['families']:
        paths.append(config.path('data_dir') / family / 'scale_validation.json')
    selected_boards = {}
    for family in design['families']:
        for method in design['config_by_method']:
            for index in range(2):
                candidate = configured(design, method, index, family, budget=budget, instances=instances)
                contexts, boards = load_data(candidate)
                checked = components_for_run(candidate, preflight_provider, contexts, from_rows(boards[0]['state']))
                selected_boards[family] = [{'id': board['id'], 'state_hash': state_hash(from_rows(board['state']))}
                                           for board in boards]
                if checked.fingerprint() != components.fingerprint():
                    raise ValueError('Policy changed across conditions')
    for path in design['config_by_method'].values():
        paths.append(ROOT / path)
    if components.calibrated is None:
        raise ValueError('Completed calibration is required')
    contracts = [configured(design, method, index, family, budget=budget, instances=instances).model_dump(mode='json')
                 for family in design['families'] for method in design['config_by_method'] for index in range(2)]
    return {'version': 1, 'design': design, 'provider': provider, 'budget': budget, 'instances_per_family': instances,
            'split': 'scale_validation', 'seed': design['seed'], 'selected_board_ids': selected_boards,
            'selected_example_ids': components.calibrated['selected_example_ids'][:config.few_shot],
            'resolved_conditions': contracts,
            'policy_fingerprint': components.fingerprint(), 'calibration_signature': components.calibrated['signature'],
            'hashes': {str(path.resolve()): _sha(path) for path in paths},
            'cases': len(design['config_by_method']) * 2 * len(design['families']) * instances}


def verify_tree(tree: SearchTreeRecorder, initial: np.ndarray, budget: int) -> dict[str, Any]:
    """Replay recorded actions from actual source nodes; never access a witness."""
    if tree.result is None or not tree.nodes:
        raise ValueError('Missing search trace')
    env = environment(initial)
    if tree.nodes[0].state != to_rows(initial):
        raise ValueError('Trace root differs from initial board')
    actions = 0
    for node in tree.nodes[1:]:
        if node.parent_id is None or node.parent_id >= node.node_id:
            raise ValueError('Invalid trace parent')
        current = from_rows(tree.nodes[node.parent_id].state)
        for index, action in enumerate(node.actions):
            if action not in range(4) or env.is_solved(current):
                raise ValueError('Illegal trace action or post-terminal step')
            current = np.asarray(env.next_state(current, action))
            if env.is_solved(current) and index + 1 < len(node.actions):
                raise ValueError('Trace continued after solving')
        if to_rows(current) != node.state or bool(env.is_solved(current)) != node.solved:
            raise ValueError('Trace endpoint or solved flag does not replay')
        if node.proposal is not None and node.kind == 'subgoal' and node.reached != np.array_equal(current, from_rows(node.proposal)):
            raise ValueError('Trace exact-hit flag does not replay')
        actions += len(node.actions)
    if actions != tree.result.attempted_actions or actions > budget:
        raise ValueError('Trace actions disagree with accounting or budget')
    if tree.result.solved:
        current = initial.copy()
        for action in tree.result.action_path:
            if env.is_solved(current):
                raise ValueError('Solution path continues after terminal')
            current = np.asarray(env.next_state(current, action))
        if not env.is_solved(current) or len(tree.result.action_path) > actions:
            raise ValueError('Reported solution did not replay')
    return {'verified': True, 'trace_actions': actions, 'trace_edges': len(tree.nodes) - 1,
            'solution_replayed': bool(tree.result.solved)}


def failure_category(outcome: SearchResult) -> str | None:
    """Separate campaign reservation stops from per-run resource and parser failures."""
    error = outcome.error or ''
    if outcome.finish_reason == 'api_budget':
        if 'Campaign or stage ' in error:
            return 'campaign_stage_guard'
        if 'generator request limit' in error:
            return 'local_generator_limit'
        if 'Combined logical' in error:
            return 'local_combined_token_limit'
        return 'local_api_limit'
    if outcome.finish_reason == 'api_error':
        if 'Invalid JSON' in error or 'must return' in error or 'invalid subgoal' in error:
            return 'model_parse_or_validation_error'
        return 'provider_or_transport_error'
    if outcome.finish_reason == 'expansion_budget':
        return 'local_expansion_limit'
    return None


def run_case(config: ExperimentConfig, provider: str, item: Instance, training: Any,
             max_generator_calls: int, max_total_tokens: int, seed: int, tag: str, workers: int) -> dict[str, Any]:
    initial = from_rows(item['state'])
    reference = components_for_run(config, provider, training, initial)
    api = CappedAPI(config.path('cache_dir'), provider, config.limits, max_generator_calls=max_generator_calls, max_total_tokens=max_total_tokens)
    api.budget_tag = tag
    components = LunaComponents(config, api, environment(initial), training, calibrated=reference.calibrated)
    tree = SearchTreeRecorder()
    unexpected = None
    try:
        outcome = search(components.env, components, initial, config, seed, observer=tree)
    except Exception as error:  # Preserve any observed tree even for unexpected solver failures.
        unexpected = type(error).__name__
        outcome = tree.result if tree.result is not None else SearchResult()
        outcome.finish_reason = 'api_error'
        outcome.error = f'Unexpected search exception: {unexpected}'
        tree.result = outcome
    finally:
        if api.client is not None:
            api.client.close()
    try:
        verification = verify_tree(tree, initial, config.action_budget)
    except Exception as error:  # Record the actual failed tree before stopping the matrix.
        verification = {'verified': False, 'error_type': type(error).__name__, 'error': str(error)}
    return {'instance': item, 'seed': seed, 'provider': provider, 'budget_tag': tag, 'execution_workers': workers,
            'config': config.model_dump(mode='json'),
            'frozen_policy': components.fingerprint(), 'calibration_signature': (components.calibrated or {}).get('signature'),
            'outcome': asdict(outcome), 'unexpected_exception_type': unexpected,
            'failure_category': failure_category(outcome),
            'api': dict(api.metrics), 'diagnostics': dict(components.diagnostics),
            'logical_total_tokens': api.metrics['logical_input_tokens'] + api.metrics['logical_output_tokens'],
            'verification': verification, 'tree': {'nodes': [asdict(node) for node in tree.nodes],
                                                   'expansions': [asdict(entry) for entry in tree.expansions],
                                                   'solution_node': tree.solution_node}}


def summarize(design: dict[str, Any], cases: list[dict[str, Any]], instances: int) -> dict[str, Any]:
    results = {}
    for method in design['config_by_method']:
        rows = []
        for index in range(2):
            selected = [case for case in cases if case['config']['method'] == method and case['pair_index'] == index]
            expected = instances * len(design['families'])
            complete = len(selected) == expected
            solved = sum(case['outcome']['solved'] for case in selected)
            actions = sum(case['outcome']['attempted_actions'] for case in selected)
            errors = sum(case['outcome']['finish_reason'] == 'api_error' for case in selected)
            stage_stops = sum(case.get('failure_category') == 'campaign_stage_guard' for case in selected)
            early_stops = sum(case.get('failure_category') in {'local_generator_limit', 'local_combined_token_limit',
                              'local_api_limit', 'local_expansion_limit'}
                              and case['outcome']['attempted_actions'] < case['config'].get('action_budget', float('inf'))
                              for case in selected)
            rows.append({'pair_index': index, 'scales': design['scale_pairs'][method][index], 'completed': len(selected),
                         'expected': expected, 'solved': solved, 'attempted_actions': actions, 'api_errors': errors,
                         'api_budget_stops': sum(case['outcome']['finish_reason'] == 'api_budget' for case in selected),
                         'campaign_stage_guard_stops': stage_stops, 'early_local_limit_stops': early_stops,
                         'finish_reasons': {reason: sum(case['outcome']['finish_reason'] == reason for case in selected)
                                            for reason in sorted({case['outcome']['finish_reason'] for case in selected})},
                         'eligible_for_selection': complete and errors == 0 and stage_stops == 0 and early_stops / expected < design['early_limit_block_fraction']
                         and all(case['verification']['verified'] for case in selected)})
        eligible = [row for row in rows if row['eligible_for_selection']]
        best = max(eligible, key=lambda row: (row['solved'], -row['pair_index'])) if len(eligible) == 2 else None
        results[method] = {'pairs': rows, 'selected_pair': best['scales'] if best else None,
                           'all_zero': bool(best is not None and all(row['solved'] == 0 for row in rows))}
    return results


def family_limit_summary(cases: list[dict[str, Any]], families: list[str], threshold: float) -> dict[str, Any]:
    """Do not hide systemic early stops in one family behind another family."""
    result = {}
    for family in families:
        group = [case for case in cases if case['config']['family'] == family]
        stopped = sum(case.get('failure_category') in {'local_generator_limit', 'local_combined_token_limit',
                      'local_api_limit', 'local_expansion_limit'}
                      and case['outcome']['attempted_actions'] < case['config']['action_budget'] for case in group)
        result[family] = {'cases': len(group), 'early_local_limit_stops': stopped,
                          'blocked': bool(group and stopped / len(group) >= threshold)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, default=Path(DEFAULT_DESIGN))
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--budget', type=int, help='Override the design action budget before the first run')
    parser.add_argument('--instances', type=int, help='Override the design number of boards per family, 1..4')
    parser.add_argument('--provider', choices=('mock', 'replay', 'openai'), default='replay')
    parser.add_argument('--workers', type=int, default=2, help='At most 4 independent case workers; 2 recommended for ledger headroom')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--plan', action='store_true', help='Print preflight without API calls')
    modes.add_argument('--prepare', action='store_true', help='Persist the exact run manifest before live execution')
    args = parser.parse_args()
    design_path = args.design.resolve()
    design = read_design(design_path)
    args.budget = args.budget if args.budget is not None else design['budget']
    args.instances = args.instances if args.instances is not None else design['instances_per_family']
    credentials(args.provider if not (args.plan or args.prepare) else 'replay')
    manifest = build_manifest(design_path, design, budget=args.budget, instances=args.instances,
                              provider=args.provider, workers=args.workers)
    if args.plan:
        print(json.dumps(manifest, indent=2))
        return
    os.environ['CARL_CAMPAIGN_STAGE'] = 'scale_validation'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.lock'), timeout=0):
        manifest_path = output / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Frozen validation manifest differs; choose a new output directory')
        if not manifest_path.exists():
            if any(path.name != '.lock' for path in output.iterdir()):
                raise ValueError('Output is not empty and has no manifest')
            write_json(manifest_path, manifest)
        if args.prepare:
            print(f'Frozen manifest: {manifest_path}', flush=True)
            return
        os.environ['CARL_CAMPAIGN_TAG'] = 'scale_tuning_v1'
        runs_path = output / 'runtime.json'
        runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
        runs.append({'workers': args.workers, 'provider': args.provider})
        write_json(runs_path, runs)
        # Interleave board, scale pair, method, then family to balance partial runs.
        jobs = []
        for board_index in range(args.instances):
            for index in range(2):
                for method in design['config_by_method']:
                    for family in design['families']:
                        config = configured(design, method, index, family, budget=args.budget, instances=args.instances)
                        training, items = load_data(config)
                        item = items[board_index]
                        file = output / f"{config.name}_{item['id']}_seed{design['seed']}.json"
                        jobs.append((config, training, item, index, file))
        cases: list[dict[str, Any]] = []

        def execute(job: Any) -> dict[str, Any]:
            config, training, item, index, file = job
            tag_prefix = design.get('tag_prefix', 'tune')
            tag = f"{tag_prefix}_{config.method}_{index}_{config.family}_{item['id']}_seed{design['seed']}"
            if file.exists():
                case = json.loads(file.read_text())
                if (case['config'] != config.model_dump(mode='json') or case['instance'] != item
                        or case['provider'] != args.provider or case['seed'] != design['seed'] or case['pair_index'] != index
                        or case.get('budget_tag') != tag):
                    raise ValueError(f'Saved case contract mismatch: {file}')
                restored = SearchTreeRecorder()
                restored.nodes = [TreeNode(**node) for node in case['tree']['nodes']]
                restored.result = SearchResult(**case['outcome'])
                if verify_tree(restored, from_rows(item['state']), config.action_budget) != case['verification']:
                    raise ValueError(f'Saved case replay mismatch: {file}')
                return case
            case = run_case(config, args.provider, item, training, design['max_generator_calls'],
                            design['max_total_tokens'], design['seed'], tag, args.workers)
            case['pair_index'] = index
            write_json(file, case)
            return case

        # Small waves ensure that a parser/API failure never starts the rest of
        # the matrix; each worker owns its client, tree, metrics and case file.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(jobs), args.workers):
                batch = list(pool.map(execute, jobs[start:start + args.workers]))
                cases.extend(batch)
                write_json(output / 'summary.json', {'status': 'running', 'tie_rule': design['tie_rule'],
                                                     'methods': summarize(design, cases, args.instances)})
                for case in batch:
                    print(f"{case['config']['method']}[{case['pair_index']}] {case['config']['family']} "
                          f"{case['instance']['id']}: {case['outcome']['finish_reason']} "
                          f"actions={case['outcome']['attempted_actions']} gen={case['api']['generator_calls']}", flush=True)
                if any(case['outcome']['finish_reason'] == 'api_error' or not case['verification']['verified'] or
                       case.get('failure_category') == 'campaign_stage_guard' for case in batch):
                    stop = ('replay_error' if any(not case['verification']['verified'] for case in batch) else
                            'api_error' if any(case['outcome']['finish_reason'] == 'api_error' for case in batch) else
                            'campaign_stage_guard')
                    write_json(output / 'summary.json', {'status': stop, 'tie_rule': design['tie_rule'],
                                                         'methods': summarize(design, cases, args.instances)})
                    raise SystemExit(2)
        methods = summarize(design, cases, args.instances)
        early = sum(case.get('failure_category') in {'local_generator_limit', 'local_combined_token_limit',
                    'local_api_limit', 'local_expansion_limit'}
                    and case['outcome']['attempted_actions'] < case['config']['action_budget'] for case in cases)
        family_limits = family_limit_summary(cases, design['families'], design['early_limit_block_fraction'])
        blocked = (early / len(cases) >= design['early_limit_block_fraction']
                   or any(item['blocked'] for item in family_limits.values())
                   or any(result['selected_pair'] is None for result in methods.values()))
        write_json(output / 'summary.json', {'status': 'blocked_limits_or_incomplete' if blocked else 'complete',
                                             'tie_rule': design['tie_rule'], 'early_local_limit_stops': early,
                                             'family_limits': family_limits, 'methods': methods})
        if blocked:
            raise SystemExit(2)


if __name__ == '__main__':
    main()
