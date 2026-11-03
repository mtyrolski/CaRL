"""Prepare, execute or replay one 1024-action scale-validation stress run.

Preparation is read-only with respect to model inference. The source tuning must
be complete and replay-verified before the selected relative-u pair can be used.
"""
import argparse
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import calibration_path, components_for_run, credentials
from carl.icl.luna import LunaComponents
from carl.icl.search import SearchResult, search
from carl.icl.search_tree import SearchTreeRecorder, TreeNode
from scripts.archive.campaign_20261102.tune_campaign_scales import CappedAPI, failure_category, read_design, summarize, verify_tree

from scripts.experiment_support.trace import DurableEnvironment, DurableTree, verify_actions

DEFAULT_TUNING = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/solver'
DEFAULT_OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/stress_1024'
FAMILY = 'generated_32x32x10'
BUDGET = 1024
SEED = 11
MAX_GENERATORS = 2048
MAX_TOTAL_TOKENS = 76168960
LIMITS = {'max_requests': 4097, 'max_input_tokens': 16000, 'max_total_input_tokens': 65552000,
          'max_output_tokens': 10616960, 'wall_seconds': 7200, 'timeout_seconds': 90}


def sha256(path: Path) -> str:
    """Hash saved configuration, data and trace artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_summary(summary: dict[str, Any], manifest: dict[str, Any]) -> int:
    """Require the complete preregistered paired comparison before a stress plan exists."""
    if (summary.get('status') != 'complete' or manifest.get('provider') != 'openai'
            or manifest.get('split') != 'scale_validation' or manifest.get('seed') != SEED):
        raise ValueError('Stress preparation requires completed live scale tuning on validation with seed 11')
    methods = summary.get('methods', {})
    if set(methods) != {'relative_u', 'absolute_d', 'fixed'}:
        raise ValueError('The three required tuning conditions are not all present')
    family_limits = summary.get('family_limits', {})
    if set(family_limits) != {'real_12x12x4', FAMILY} or any(item.get('blocked') for item in family_limits.values()):
        raise ValueError('Both validation families must pass the tuning resource gate')
    for method in methods.values():
        pairs = method.get('pairs', [])
        if len(pairs) != 2 or any(not pair.get('eligible_for_selection') for pair in pairs):
            raise ValueError('Both scale pairs for every method must be eligible and complete')
        if method.get('selected_pair') is None:
            raise ValueError('The tuning summary has no selected scale pair')
    selected = methods['relative_u']['selected_pair']
    indices = [row['pair_index'] for row in methods['relative_u']['pairs'] if row['scales'] == selected]
    if len(indices) != 1:
        raise ValueError('Relative selection does not uniquely match one tuning pair')
    return int(indices[0])


def ledger_plan() -> dict[str, Any]:
    """Read the shared ledger and conservative single-request reservation without creating one."""
    prices = json.loads((ROOT / 'configs/icl/campaign_budget.json').read_text())
    path = ROOT / prices['ledger_path']
    uri = path.resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        rows = db.execute('SELECT stage,status,COUNT(*),SUM(COALESCE(charged_nanos,reserved_nanos)) FROM attempts '
                          'GROUP BY stage,status').fetchall()
        stored = db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()
    global_cap = min(float(prices['limit_usd']), int(stored[0]) / 1e9)
    local_env = dotenv_values(ROOT / '.env').get('CARL_CAMPAIGN_MAX_USD')
    for extra in (local_env, os.environ.get('CARL_CAMPAIGN_MAX_USD')):
        if extra is not None:
            global_cap = min(global_cap, float(extra))
    entries = [{'stage': stage, 'status': status, 'attempts': count, 'committed_usd': nanos / 1e9}
               for stage, status, count, nanos in rows]
    stage_used = sum(item['committed_usd'] for item in entries if item['stage'] == 'scale_validation')
    total_used = sum(item['committed_usd'] for item in entries)
    model = prices['models']['gpt-6-luna']
    reserve = (float(model['cache_write_12h_usd_per_million']) * 2 * model['max_input_tokens']
               + float(model['output_usd_per_million']) * 1.5 * 4096) * float(model['regional_multiplier']) / 1e6
    return {'stage_limit_usd': float(prices['stage_limits_usd']['scale_validation']), 'stage_committed_usd': stage_used,
            'global_effective_limit_usd': global_cap, 'global_committed_usd': total_used,
            'max_single_request_reservation_usd': reserve, 'entries': entries,
            'note': 'Committed includes unknown reservations. The live ledger enforces the actual limit atomically.'}


def ensure_budget_headroom(forecast: dict[str, Any], ledger: dict[str, Any]) -> None:
    """Do not start a stress run whose usage scenario already exceeds the remaining limits."""
    required = forecast['estimated_uncached_30m_write_usd'] + ledger['max_single_request_reservation_usd']
    stage_remaining = ledger['stage_limit_usd'] - ledger['stage_committed_usd']
    global_remaining = ledger['global_effective_limit_usd'] - ledger['global_committed_usd']
    if required > min(stage_remaining, global_remaining):
        raise ValueError('Estimated stress cost plus one request reservation exceeds remaining campaign or stage budget')


def estimate(cases: list[dict[str, Any]], selected_index: int) -> dict[str, Any]:
    """Project logical tokens per action; report observed cache and bias explicitly."""
    sample = [case for case in cases if case['config']['family'] == FAMILY
              and case['config']['method'] == 'relative_u' and case['pair_index'] == selected_index]
    if not sample:
        raise ValueError('Missing selected 32x32 tuning observations')
    actions = sum(case['outcome']['attempted_actions'] for case in sample)
    if actions <= 0:
        raise ValueError('Cannot estimate 1024-action cost from zero observed actions')
    incoming = sum(case['api']['logical_input_tokens'] for case in sample)
    outgoing = sum(case['api']['logical_output_tokens'] for case in sample)
    logical = sum(case['api']['logical_requests'] for case in sample)
    physical = sum(case['api']['requests'] for case in sample)
    cache = sum(case['api']['cache_hits'] for case in sample)
    budget_data = json.loads((ROOT / 'configs/icl/campaign_budget.json').read_text())
    ledger_path = ROOT / budget_data['ledger_path']
    uri = ledger_path.resolve().as_uri() + '?mode=ro'
    tags = [case['budget_tag'] for case in sample]
    with closing(sqlite3.connect(uri, uri=True)) as db:
        attempts = [row for tag in tags for row in db.execute(
            'SELECT status,charged_nanos,reserved_nanos,usage_json FROM attempts WHERE stage=? AND tag=?',
            ('scale_validation', tag))]
    if any(row[0] != 'settled' or row[1] is None for row in attempts):
        raise ValueError('Selected tuning sample has an unsettled or unknown physical attempt')
    physical_cost = sum(row[1] for row in attempts) / 1e9
    physical_usage = [json.loads(row[3]) for row in attempts if row[3]]
    if (len(attempts) != physical or len(physical_usage) != len(attempts)
            or sum(item.get('input_tokens', 0) for item in physical_usage) != sum(case['api']['input_tokens'] for case in sample)
            or sum(item.get('output_tokens', 0) for item in physical_usage) != sum(case['api']['output_tokens'] for case in sample)):
        raise ValueError('Selected 32x32 tuning usage disagrees with the shared cost ledger')
    budget = budget_data['models']['gpt-6-luna']
    input_rate = float(budget['cache_write_30m_usd_per_million'])
    output_rate = float(budget['output_usd_per_million'])
    multiplier = float(budget['regional_multiplier'])
    projected_input, projected_output = BUDGET * incoming / actions, BUDGET * outgoing / actions
    usd = (projected_input * input_rate + projected_output * output_rate) * multiplier / 1e6
    return {'sample_cases': len(sample), 'sample_actions': actions, 'sample_logical_requests': logical,
            'sample_physical_requests': physical, 'sample_local_cache_hits': cache,
            'sample_ledger_attempts': len(attempts), 'sample_ledger_charged_usd': physical_cost,
            'sample_ledger_input_tokens': sum(item.get('input_tokens', 0) for item in physical_usage),
            'sample_ledger_output_tokens': sum(item.get('output_tokens', 0) for item in physical_usage),
            'sample_logical_input_tokens': incoming, 'sample_logical_output_tokens': outgoing,
            'projected_logical_requests': BUDGET * logical / actions,
            'projected_input_tokens': projected_input, 'projected_output_tokens': projected_output,
            'estimated_uncached_30m_write_usd': usd,
            'note': 'Linear extrapolation from a small selected-pair 32x32 validation sample; assumes every input token '
                    'is a paid 30m cache write and adds regional uplift. Early termination, cache, larger trees, retries, '
                    'unknown billing, long contexts and rate changes invalidate it as a guarantee.'}


def source_cases(tuning: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Verify immutable source code/data and replay all paired tuning cases."""
    for filename, expected in manifest['hashes'].items():
        path = Path(filename)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Tuning source changed since its manifest: {path}')
    files = sorted(tuning.glob('tune_*.json'))
    if len(files) != manifest['cases']:
        raise ValueError('Tuning matrix is incomplete')
    cases = [json.loads(path.read_text()) for path in files]
    seen: set[tuple[str, int, str, str]] = set()
    for case in cases:
        config = case['config']
        key = (config['method'], case['pair_index'], config['family'], case['instance']['id'])
        if key in seen or case['provider'] != 'openai' or case['seed'] != SEED or config['split'] != 'scale_validation':
            raise ValueError('Duplicated or mismatched tuning case')
        seen.add(key)
        matching = [candidate for candidate in manifest['resolved_conditions']
                    if candidate['method'] == config['method'] and candidate['family'] == config['family']
                    and candidate['scales'] == config['scales']]
        if len(matching) != 1 or matching[0] != config:
            raise ValueError('Tuning case config differs from frozen condition')
        expected_boards = {row['id']: row['state_hash'] for row in manifest['selected_board_ids'][config['family']]}
        if expected_boards.get(case['instance']['id']) != state_hash(from_rows(case['instance']['state'])):
            raise ValueError('Tuning case board does not match frozen manifest')
        tree = SearchTreeRecorder()
        tree.nodes = [TreeNode(**node) for node in case['tree']['nodes']]
        tree.result = SearchResult(**case['outcome'])
        verification = verify_tree(tree, from_rows(case['instance']['state']), config['action_budget'])
        if verification != case['verification'] or not verification['verified']:
            raise ValueError('Tuning trace does not replay')
    if summarize(manifest['design'], cases, manifest['instances_per_family']) != summary['methods']:
        raise ValueError('Tuning summary does not match recorded cases')
    return cases


def prepare_plan(tuning: Path, output: Path) -> tuple[dict[str, Any], ExperimentConfig, Any, Any]:
    """Use the selected relative-u pair and first full validation board only."""
    manifest_path, summary_path = tuning / 'manifest.json', tuning / 'summary.json'
    manifest, summary = json.loads(manifest_path.read_text()), json.loads(summary_path.read_text())
    pair_index = checked_summary(summary, manifest)
    cases = source_cases(tuning, manifest, summary)
    design_dir = ROOT / 'configs/icl/campaign_data/scale_tuning'
    design_paths = [Path(path) for path in manifest['hashes']
                    if Path(path).parent == design_dir and Path(path).suffix == '.json']
    if len(design_paths) != 1:
        raise ValueError('Tuning manifest must identify exactly one scale design')
    design = read_design(design_paths[0])
    if design != manifest['design']:
        raise ValueError('Design changed after tuning')
    selected_scales = design['scale_pairs']['relative_u'][pair_index]
    resolved = [item for item in manifest['resolved_conditions']
                if item['method'] == 'relative_u' and item['family'] == FAMILY and item['scales'] == selected_scales]
    if len(resolved) != 1:
        raise ValueError('Tuning manifest must contain exactly one resolved selected condition')
    base = ExperimentConfig.model_validate(resolved[0])
    config = ExperimentConfig.model_validate({**base.model_dump(), 'name': 'stress_1024_relative_u_generated',
                                              'action_budget': BUDGET, 'max_expansions': 2048,
                                              'num_subgoals': 2, 'limits': LIMITS})
    training, instances = load_data(config)
    item = instances[0]
    if item['id'] != manifest['selected_board_ids'][FAMILY][0]['id']:
        raise ValueError('Stress board is not the first tuned validation board')
    components = components_for_run(config, 'replay', training, from_rows(item['state']))
    if components.fingerprint() != manifest['policy_fingerprint'] or components.calibrated is None:
        raise ValueError('Frozen policy or calibration is missing')
    if components.calibrated['signature'] != manifest['calibration_signature']:
        raise ValueError('Frozen calibration differs from tuning')
    files = [ROOT / 'scripts/experiment_support/api.py', ROOT / 'scripts/experiment_support/trace.py',
              manifest_path, summary_path, ROOT / 'scripts/archive/campaign_20261102/tune_campaign_scales.py', Path(__file__),
             ROOT / 'configs/icl/campaign_budget.json', calibration_path(config, components)]
    files.extend(sorted(tuning.glob('tune_*.json')))
    case_hash = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    tag = f'stress_1024_relative_u_{case_hash}'
    plan = {'version': 1, 'provider': 'openai', 'purpose': 'single validation stress; no pilot or freeze',
            'method': 'relative_u', 'selected_pair_index': pair_index, 'scales': config.scales,
            'board_id': item['id'], 'state_hash': state_hash(from_rows(item['state'])), 'family': FAMILY,
            'split': 'scale_validation', 'seed': SEED, 'action_budget': BUDGET, 'max_expansions': 2048,
            'max_generator_calls': MAX_GENERATORS, 'max_total_tokens': MAX_TOTAL_TOKENS, 'limits': LIMITS,
            'num_subgoals': 2, 'policy_fingerprint': components.fingerprint(),
            'calibration_signature': components.calibrated['signature'], 'ledger_stage': 'scale_validation', 'ledger_tag': tag,
            'resolved_config': config.model_dump(mode='json'), 'cost_forecast': estimate(cases, pair_index),
            'source_sha256': {str(path.resolve()): sha256(path) for path in files}}
    return plan, config, item, training








def budget_evidence(outcome: SearchResult, verification: dict[str, Any], unexpected: str | None) -> bool:
    """Require a full-budget run; an early solution does not exercise the limits."""
    normal = unexpected is None and failure_category(outcome) is None and bool(verification.get('verified'))
    return bool(normal and outcome.finish_reason in {'action_budget', 'solved'} and outcome.attempted_actions == BUDGET)


def execute(plan: dict[str, Any], config: ExperimentConfig, item: Any, training: Any, output: Path) -> None:
    """Run one search; retain every real transition and never use a reference witness."""
    trace_path, case_path = output / 'trace.jsonl', output / 'case.json'
    if trace_path.exists() or case_path.exists():
        raise ValueError('A stress trace already exists; inspect/replay it before any new attempt')
    credentials('openai')
    reference = components_for_run(config, 'replay', training, from_rows(item['state']))
    api = CappedAPI(config.path('cache_dir'), 'openai', config.limits,
                    max_generator_calls=MAX_GENERATORS, max_total_tokens=MAX_TOTAL_TOKENS)
    api.budget_tag = plan['ledger_tag']
    tree = DurableTree(trace_path)
    env = DurableEnvironment(environment(from_rows(item['state'])), tree)
    component = LunaComponents(config, api, env, training, calibrated=reference.calibrated)
    unexpected = None
    try:
        outcome = search(component.env, component, from_rows(item['state']), config, SEED, observer=tree)
    except Exception as error:  # Keep the partial trace, then classify it as lack of evidence.
        unexpected = type(error).__name__
        outcome = tree.result if tree.result is not None else SearchResult()
        outcome.finish_reason, outcome.error = 'api_error', f'Unexpected search exception: {unexpected}'
        tree.result = outcome
        tree.append({'event': 'exception', 'error_type': unexpected})
    finally:
        if api.client is not None:
            api.client.close()
    try:
        verification = verify_tree(tree, from_rows(item['state']), BUDGET)
        events = [json.loads(line) for line in trace_path.read_text().splitlines()]
        verify_actions(tree, from_rows(item['state']), events)
    except Exception as error:
        verification = {'verified': False, 'error_type': type(error).__name__, 'error': str(error)}
    evidence = budget_evidence(outcome, verification, unexpected)
    case = {'provider': 'openai', 'seed': SEED, 'instance': item, 'config': config.model_dump(mode='json'),
            'frozen_policy': component.fingerprint(), 'calibration_signature': (component.calibrated or {}).get('signature'),
            'budget_tag': plan['ledger_tag'], 'outcome': asdict(outcome), 'failure_category': failure_category(outcome),
            'unexpected_exception_type': unexpected, 'verification': verification, 'api': dict(api.metrics),
            'diagnostics': dict(component.diagnostics), 'evidence_of_usable_budget': evidence,
            'note': 'A failed replay, parser/API/stage/local limit, interruption, or frontier exhaustion before B does not establish usable B.'}
    write_json(case_path, case)
    print(json.dumps({'outcome': case['outcome']['finish_reason'], 'actions': outcome.attempted_actions,
                      'verified': verification['verified'], 'evidence_of_usable_budget': evidence,
                      'failure_category': case['failure_category']}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning-dir', type=Path, default=DEFAULT_TUNING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--plan', action='store_true', help='Print costs and limits; no inference')
    modes.add_argument('--prepare', action='store_true', help='Freeze plan after complete tuning; no inference')
    modes.add_argument('--execute', action='store_true', help='Run the prepared live stress; incurs charges')
    modes.add_argument('--replay', action='store_true', help='Replay a completed trace without API calls')
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
                raise ValueError('Prepared stress differs; use another output directory')
            if not manifest.exists():
                if any(path.name != '.lock' for path in output.iterdir()):
                    raise ValueError('Stress directory is nonempty without a manifest')
                write_json(manifest, plan)
            print(json.dumps({'prepared': str(manifest), 'cost_forecast': plan['cost_forecast'],
                              'ledger_now': ledger_plan()}, indent=2))
            return
        if not manifest.is_file() or json.loads(manifest.read_text()) != plan:
            raise ValueError('Run --prepare first; prepared inputs must remain unchanged')
        if args.replay:
            case = json.loads((output / 'case.json').read_text())
            tree = SearchTreeRecorder()
            # Replay the independently stored append-only trace, not a reference witness.
            events = [json.loads(line) for line in (output / 'trace.jsonl').read_text().splitlines()]
            nodes = [event['node'] for event in events if event['event'] in {'start', 'transition'}]
            tree.nodes = [TreeNode(**node) for node in nodes]
            if (case['provider'] != 'openai' or case['seed'] != SEED or case['instance'] != item
                    or case['config'] != config.model_dump(mode='json') or case['budget_tag'] != plan['ledger_tag']
                    or case['frozen_policy'] != plan['policy_fingerprint']
                    or case['calibration_signature'] != plan['calibration_signature']):
                raise ValueError('Stress case does not match the prepared contract')
            finished = [event for event in events if event['event'] == 'finish']
            if len(finished) != 1 or finished[0]['outcome'] != case['outcome']:
                raise ValueError('Append-only trace does not match the completed outcome')
            tree.result = SearchResult(**case['outcome'])
            result = verify_tree(tree, from_rows(item['state']), BUDGET)
            verify_actions(tree, from_rows(item['state']), events)
            if result != case['verification']:
                raise ValueError('Saved stress trace does not reproduce its verification')
            evidence = budget_evidence(tree.result, result, case['unexpected_exception_type'])
            if evidence != case['evidence_of_usable_budget']:
                raise ValueError('Saved budget-evidence flag does not match replay')
            print(json.dumps({'replay': result, 'evidence_of_usable_budget': evidence}, indent=2))
            return
        ledger = ledger_plan()
        ensure_budget_headroom(plan['cost_forecast'], ledger)
        print(json.dumps({'cost_forecast': plan['cost_forecast'], 'ledger_now': ledger}, indent=2), flush=True)
        execute(plan, config, item, training, output)


if __name__ == '__main__':
    main()
