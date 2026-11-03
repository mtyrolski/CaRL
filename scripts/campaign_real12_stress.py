"""Prepare, run or replay one 12x12x4 validation stress at B=256."""
import argparse
import hashlib
import json
import os
import sqlite3
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
from scripts.campaign_scale_stress import (
    DurableEnvironment,
    DurableTree,
    ensure_budget_headroom,
    ledger_plan,
    sha256,
    source_cases,
    verify_actions,
)
from scripts.tune_campaign_scales import CappedAPI, failure_category, verify_tree
from scripts.tune_real12_campaign_scales import FAMILY, METHODS

TUNING = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/real12_v1/solver'
OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/real12_v1/stress_256'
BUDGET = 256
MAX_GENERATORS = 512
MAX_TOTAL_TOKENS = 16384000
LIMITS = {'max_requests': 1536, 'max_input_tokens': 16000, 'max_total_input_tokens': 16384000,
          'max_output_tokens': 2097152, 'wall_seconds': 3600, 'timeout_seconds': 90}


def checked_sources(tuning: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Require a complete, untampered and replayed 12x12-only scale matrix."""
    manifest = json.loads((tuning / 'manifest.json').read_text())
    summary = json.loads((tuning / 'summary.json').read_text())
    if (manifest.get('provider') != 'openai' or manifest.get('split') != 'scale_validation'
            or manifest.get('seed') != 11 or manifest['design']['families'] != [FAMILY]
            or summary.get('status') != 'complete' or set(summary.get('methods', {})) != METHODS
            or set(summary.get('family_limits', {})) != {FAMILY} or summary['family_limits'][FAMILY]['blocked']):
        raise ValueError('Stress requires completed and eligible 12x12-only live scale tuning')
    for method in summary['methods'].values():
        if (len(method.get('pairs', [])) != 2 or any(not row['eligible_for_selection'] for row in method['pairs'])
                or method.get('selected_pair') is None):
            raise ValueError('Both scale pairs must be eligible for all methods')
    cases = source_cases(tuning, manifest, summary)
    for case in cases:
        board = from_rows(case['instance']['state'])
        if case['config']['family'] != FAMILY or board.shape[:2] != (12, 12):
            raise ValueError('Stress sources may contain only 12x12 boards')
    return manifest, summary, cases


def estimate(cases: list[dict[str, Any]], selected: list[float]) -> dict[str, Any]:
    """Use the larger of two measured logical-token rates, with no local replay."""
    selected_cases = [case for case in cases if case['config']['method'] == 'relative_u'
                      and case['config']['scales'] == selected]
    if len(selected_cases) != 2:
        raise ValueError('Expected two selected relative-u validation cases')
    pricing = json.loads((ROOT / 'configs/icl/campaign_budget.json').read_text())['models']['gpt-6-luna']
    rates = []
    for case in selected_cases:
        outcome, api = case['outcome'], case['api']
        actions = outcome['attempted_actions']
        if outcome['finish_reason'] != 'action_budget' or actions != 128 or not case['verification']['verified']:
            continue
        proxy = (api['logical_input_tokens'] * float(pricing['cache_write_30m_usd_per_million'])
                 + api['logical_output_tokens'] * float(pricing['output_usd_per_million']))
        rates.append(proxy * float(pricing['regional_multiplier']) / 1e6 / actions)
    if not rates:
        raise ValueError('No uncensored full-budget validation case supports cost extrapolation')
    return {'sample_cases': len(rates), 'censored_cases': len(selected_cases) - len(rates),
            'max_proxy_usd_per_action': max(rates),
            'estimated_uncached_30m_write_usd': max(rates) * BUDGET,
            'note': 'Small-sample linear scenario with all logical input charged as 30m cache writes; not a bound.'}


def prepare_plan(tuning: Path, output: Path) -> tuple[dict[str, Any], ExperimentConfig, Any, Any]:
    manifest, summary, cases = checked_sources(tuning)
    selected = summary['methods']['relative_u']['selected_pair']
    conditions = [row for row in manifest['resolved_conditions'] if row['method'] == 'relative_u'
                  and row['family'] == FAMILY and row['scales'] == selected]
    if len(conditions) != 1:
        raise ValueError('Selected frozen relative-u condition is not unique')
    base = ExperimentConfig.model_validate(conditions[0])
    config = ExperimentConfig.model_validate({**base.model_dump(), 'name': 'stress_256_real12_relative_u',
                                              'action_budget': BUDGET, 'max_expansions': 512,
                                              'num_subgoals': 2, 'limits': LIMITS})
    training, items = load_data(config)
    item = items[0]
    initial = from_rows(item['state'])
    if initial.shape[:2] != (12, 12) or item['id'] != manifest['selected_board_ids'][FAMILY][0]['id']:
        raise ValueError('Stress target must be the first frozen 12x12 validation board')
    components = components_for_run(config, 'replay', training, initial)
    if (components.fingerprint() != manifest['policy_fingerprint'] or components.calibrated is None
            or components.calibrated['signature'] != manifest['calibration_signature']):
        raise ValueError('Frozen policy or calibration changed')
    files = [tuning / 'manifest.json', tuning / 'summary.json', ROOT / 'configs/icl/campaign_budget.json',
             Path(__file__), calibration_path(config, components), ROOT / 'scripts/campaign_scale_stress.py']
    files.extend(sorted(tuning.glob('tune_*.json')))
    tag = 'stress_real12_256_' + hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    plan = {'version': 1, 'provider': 'openai', 'purpose': '12x12 validation only; no pilot or freeze',
            'board_id': item['id'], 'state_hash': state_hash(initial), 'family': FAMILY, 'seed': 11,
            'action_budget': BUDGET, 'scales': selected, 'policy_fingerprint': components.fingerprint(),
            'calibration_signature': components.calibrated['signature'], 'resolved_config': config.model_dump(mode='json'),
            'max_generator_calls': MAX_GENERATORS, 'max_total_tokens': MAX_TOTAL_TOKENS, 'ledger_tag': tag,
            'cost_forecast': estimate(cases, selected), 'source_sha256': {str(path): sha256(path) for path in files}}
    return plan, config, item, training


def load_trace(path: Path, outcome: SearchResult) -> tuple[SearchTreeRecorder, list[dict[str, Any]]]:
    events = [json.loads(line) for line in path.read_text().splitlines()]
    tree = SearchTreeRecorder()
    tree.nodes = [TreeNode(**event['node']) for event in events if event['event'] in {'start', 'transition'}]
    tree.result = outcome
    return tree, events


def execute(plan: dict[str, Any], config: ExperimentConfig, item: Any, training: Any, output: Path) -> None:
    trace_path, case_path = output / 'trace.jsonl', output / 'case.json'
    if trace_path.exists() or case_path.exists():
        raise ValueError('Existing stress trace must be inspected; refusing to repeat work')
    credentials('openai')
    initial = from_rows(item['state'])
    reference = components_for_run(config, 'replay', training, initial)
    api = CappedAPI(config.path('cache_dir'), 'openai', config.limits,
                    max_generator_calls=MAX_GENERATORS, max_total_tokens=MAX_TOTAL_TOKENS)
    api.budget_tag = plan['ledger_tag']
    tree = DurableTree(trace_path)
    env = DurableEnvironment(environment(initial), tree)
    component = LunaComponents(config, api, env, training, calibrated=reference.calibrated)
    interrupted, unexpected = False, None
    try:
        outcome = search(component.env, component, initial, config, 11, observer=tree)
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        unexpected = type(error).__name__
        if interrupted and api.ledger is not None:
            # The provider may have billed the interrupted in-flight request;
            # retain its reservation and explicitly mark its usage unknown.
            with sqlite3.connect(api.ledger.path) as db:
                pending = db.execute("SELECT id FROM attempts WHERE stage=? AND tag=? AND status='reserved'",
                                     ('scale_validation', plan['ledger_tag'])).fetchall()
            for (reservation,) in pending:
                api.ledger.mark_unknown(reservation)
        outcome = tree.result if tree.result is not None else SearchResult()
        outcome.finish_reason = 'api_error'
        outcome.error = f'Unexpected search interruption: {unexpected}'
        outcome.attempted_actions = sum(1 for line in trace_path.read_text().splitlines()
                                        if json.loads(line).get('event') == 'action') if trace_path.exists() else 0
        tree.result = outcome
        tree.append({'event': 'interruption', 'error_type': unexpected})
    finally:
        if api.client is not None:
            api.client.close()
    try:
        verification = verify_tree(tree, initial, BUDGET)
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        verify_actions(tree, initial, events)
    except Exception as error:
        verification = {'verified': False, 'error_type': type(error).__name__, 'error': str(error)}
    full_budget = bool(unexpected is None and verification['verified'] and outcome.attempted_actions == BUDGET
                       and outcome.finish_reason in {'action_budget', 'solved'})
    case = {'provider': 'openai', 'instance': item, 'seed': 11, 'config': config.model_dump(mode='json'),
            'policy_fingerprint': component.fingerprint(), 'budget_tag': plan['ledger_tag'], 'outcome': asdict(outcome),
            'failure_category': failure_category(outcome), 'unexpected_exception_type': unexpected,
            'verification': verification, 'api': dict(api.metrics), 'diagnostics': dict(component.diagnostics),
            'evidence_full_budget': full_budget}
    write_json(case_path, case)
    print(json.dumps({'finish_reason': outcome.finish_reason, 'actions': outcome.attempted_actions,
                      'solved': outcome.solved, 'verification': verification, 'evidence_full_budget': full_budget}))
    if interrupted:
        raise SystemExit(130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning-dir', type=Path, default=TUNING)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--plan', action='store_true')
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    tuning, output = args.tuning_dir.resolve(), args.output.resolve()
    os.environ['CARL_CAMPAIGN_STAGE'] = 'scale_validation'
    plan, config, item, training = prepare_plan(tuning, output)
    if args.plan:
        print(json.dumps({'plan': plan, 'ledger_now': ledger_plan()}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / '.lock'), timeout=0):
        manifest = output / 'manifest.json'
        if args.prepare:
            if manifest.exists() and json.loads(manifest.read_text()) != plan:
                raise ValueError('Prepared stress differs; use a new directory')
            if not manifest.exists():
                if any(path.name != '.lock' for path in output.iterdir()):
                    raise ValueError('Stress directory is not empty')
                write_json(manifest, plan)
            print(json.dumps({'prepared': str(manifest), 'cost_forecast': plan['cost_forecast'],
                              'ledger_now': ledger_plan()}, indent=2))
            return
        if not manifest.is_file() or json.loads(manifest.read_text()) != plan:
            raise ValueError('Run --prepare with the same source inputs first')
        if args.replay:
            case = json.loads((output / 'case.json').read_text())
            if (case['provider'] != 'openai' or case['seed'] != 11 or case['instance'] != item
                    or case['config'] != config.model_dump(mode='json') or case['budget_tag'] != plan['ledger_tag']
                    or case['policy_fingerprint'] != plan['policy_fingerprint']):
                raise ValueError('Stress case differs from the frozen contract')
            tree, events = load_trace(output / 'trace.jsonl', SearchResult(**case['outcome']))
            finished = [event for event in events if event['event'] == 'finish']
            if len(finished) != 1 or finished[0]['outcome'] != case['outcome']:
                raise ValueError('No matching completed finish event in trace')
            verification = verify_tree(tree, from_rows(item['state']), BUDGET)
            verify_actions(tree, from_rows(item['state']), events)
            if verification != case['verification']:
                raise ValueError('Stress replay differs from saved verification')
            print(json.dumps({'replay': verification, 'evidence_full_budget': case['evidence_full_budget']}))
            return
        ledger = ledger_plan()
        try:
            ensure_budget_headroom(plan['cost_forecast'], ledger)
            forecast_exceeds_headroom = False
        except ValueError:
            # The forecast is not a hard bound: proceed until the durable
            # shared ledger enforces the actual stage/account/campaign limit.
            forecast_exceeds_headroom = True
        print(json.dumps({'cost_forecast': plan['cost_forecast'], 'ledger_now': ledger,
                          'forecast_exceeds_headroom': forecast_exceeds_headroom}, indent=2), flush=True)
        execute(plan, config, item, training, output)


if __name__ == '__main__':
    main()
