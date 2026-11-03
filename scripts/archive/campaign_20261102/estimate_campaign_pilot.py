"""Project independent pilot costs from completed validation cases without opening pilot data."""
import argparse
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from carl.icl.api import write_json
from carl.icl.config import ROOT

BUDGETS = (256, 1024, 2048)
FAMILIES = ('real_12x12x4', 'generated_32x32x10')
METHOD_STAGES = {'relative_u': 'pilot_relative', 'absolute_d': 'pilot_absolute', 'fixed': 'pilot_fixed'}
DEFAULT_RESULTS = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/solver'
DEFAULT_OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/pilot_cost_projection'
DEFAULT_PRICING = ROOT / 'configs/icl/campaign_budget.json'


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger_snapshot(path: Path, budget: dict[str, Any]) -> dict[str, Any]:
    """Read committed reservations without creating or modifying the real ledger."""
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('BEGIN')
        effective = int(db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()[0])
        rows = db.execute('SELECT stage,status,COUNT(*),SUM(COALESCE(charged_nanos,reserved_nanos)) '
                          'FROM attempts GROUP BY stage,status').fetchall()
        case_rows = db.execute('SELECT tag,status,COUNT(*),SUM(COALESCE(charged_nanos,reserved_nanos)), '
                               'SUM(CASE WHEN usage_json IS NOT NULL THEN '
                               "CAST(json_extract(usage_json,'$.input_tokens') AS INTEGER) ELSE 0 END), "
                               'SUM(CASE WHEN usage_json IS NOT NULL THEN '
                               "CAST(json_extract(usage_json,'$.output_tokens') AS INTEGER) ELSE 0 END) "
                               "FROM attempts WHERE stage='scale_validation' GROUP BY tag,status").fetchall()
    limits = [budget['limit_usd'], os.environ.get('CARL_CAMPAIGN_MAX_USD'),
              dotenv_values(ROOT / '.env').get('CARL_CAMPAIGN_MAX_USD')]
    for limit in limits:
        if limit is not None:
            effective = min(effective, int(Decimal(str(limit)) * 1_000_000_000))
    committed = sum(int(row[3]) for row in rows)
    stage_totals: defaultdict[str, int] = defaultdict(int)
    for stage, _status, _count, amount in rows:
        stage_totals[stage] += int(amount)
    stages = {stage: {'limit_usd': float(limit), 'committed_usd': stage_totals[stage] / 1e9,
                      'remaining_usd': float(limit) - stage_totals[stage] / 1e9}
              for stage, limit in budget['stage_limits_usd'].items()}
    return {'snapshot_utc': datetime.now(timezone.utc).isoformat(),
            'effective_limit_usd': effective / 1e9, 'committed_usd': committed / 1e9,
            'remaining_usd': (effective - committed) / 1e9, 'stages': stages,
            'by_status': [{'stage': stage, 'status': status, 'count': count, 'committed_usd': int(amount) / 1e9}
                          for stage, status, count, amount in rows],
            'case_usage': [{'tag': tag, 'status': status, 'attempts': count, 'committed_usd': int(amount) / 1e9,
                            'input_tokens': int(input_tokens or 0), 'output_tokens': int(output_tokens or 0)}
                           for tag, status, count, amount, input_tokens, output_tokens in case_rows],
            'note': 'Unknown and in-flight reservations remain committed; active reservations may later settle lower.'}


def observation(case: dict[str, Any], pricing: dict[str, Any]) -> dict[str, Any]:
    """Keep every case, but censor failures, resource stops and short solved searches."""
    if case.get('provider') != 'openai':
        raise ValueError('Only actual live OpenAI validation results are eligible')
    config, outcome, metrics = case['config'], case['outcome'], case['api']
    if config.get('split') != 'scale_validation' or config.get('family') not in FAMILIES:
        raise ValueError('Only public scale-validation cases may be analyzed')
    if config.get('model') != 'gpt-6-luna':
        raise ValueError('Pricing applies only to gpt-6-luna')
    if case.get('seed') != 11:
        raise ValueError('Pilot extrapolation is preregistered for seed 11')
    actions, budget = int(outcome['attempted_actions']), int(config['action_budget'])
    inputs = int(metrics['logical_input_tokens'])
    outputs = int(metrics['logical_output_tokens'])
    if min(actions, inputs, outputs) < 0 or actions > budget:
        raise ValueError('Invalid action or token accounting')
    proxy = (inputs * float(pricing['cache_write_30m_usd_per_million'])
             + outputs * float(pricing['output_usd_per_million'])) * float(pricing['regional_multiplier']) / 1e6
    reason = outcome['finish_reason']
    verified = bool(case.get('verification', {}).get('verified'))
    uncensored = reason == 'action_budget' and actions == budget and actions > 0 and verified
    if uncensored:
        censorship = None
    elif not verified:
        censorship = 'replay_not_verified'
    elif outcome.get('solved'):
        censorship = 'solved_early_no_evidence_for_longer_search'
    elif reason == 'frontier_exhausted':
        censorship = 'frontier_exhausted'
    elif reason in {'api_budget', 'api_error', 'expansion_budget'}:
        censorship = 'resource_or_api_stop:' + str(case.get('failure_category') or reason)
    else:
        censorship = 'not_observed_to_full_budget:' + reason
    return {'method': config['method'], 'family': config['family'], 'pair_index': case['pair_index'],
            'scales': config['scales'], 'board_id': case['instance']['id'], 'seed': case['seed'],
            'budget_tag': case.get('budget_tag'), 'budget': budget, 'attempted_actions': actions,
            'finish_reason': reason, 'solved': bool(outcome.get('solved')),
            'replay_verified': verified, 'censored': not uncensored, 'censor_reason': censorship,
            'logical_input_tokens': inputs, 'logical_output_tokens': outputs,
            'logical_requests': int(metrics['logical_requests']), 'local_cache_hits': int(metrics['cache_hits']),
            'physical_requests': int(metrics['requests']),
            'uncached_30m_uplift_proxy_usd': proxy,
            'proxy_usd_per_action': proxy / actions if uncensored else None}


def project(observations: list[dict[str, Any]], summary: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Use winning validation cases only; incomplete or censored groups stay unknown."""
    selected = {method: summary.get('methods', {}).get(method, {}).get('selected_pair') for method in METHOD_STAGES}
    completed_selection = summary.get('status') == 'complete' and all(selected.values())
    samples: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if completed_selection:
        for row in observations:
            if row['method'] in selected and row['scales'] == selected[row['method']]:
                samples[(row['method'], row['family'])].append(row)
    groups = []
    for method in METHOD_STAGES:
        for family in FAMILIES:
            rows = samples[(method, family)]
            rates = [row['proxy_usd_per_action'] for row in rows if row['proxy_usd_per_action'] is not None]
            for scenario in ('lowest_observed_rate', 'mean_observed_rate'):
                rate = (min(rates) if scenario == 'lowest_observed_rate' else sum(rates) / len(rates)) if rates else None
                projections = {str(b): (50 * b * rate if rate is not None else None) for b in BUDGETS}
                groups.append({'method': method, 'family': family, 'stage': METHOD_STAGES[method],
                               'scenario': scenario, 'selected_scales': selected[method] if completed_selection else None,
                               'validation_cases': len(rows), 'uncensored_cases': len(rates),
                               'censored_cases': sum(row['censored'] for row in rows),
                               'censor_reasons': [row['censor_reason'] for row in rows if row['censored']],
                               'usd_per_action': rate, 'pilot_boards': 50,
                               'independent_budget_cost_usd': projections,
                               'three_independent_runs_usd': 50 * sum(BUDGETS) * rate if rate is not None else None})
    totals = []
    for scenario in ('lowest_observed_rate', 'mean_observed_rate'):
        chosen = [row for row in groups if row['scenario'] == scenario]
        for method, stage in METHOD_STAGES.items():
            rows = [row for row in chosen if row['method'] == method]
            values = [row['three_independent_runs_usd'] for row in rows]
            total = sum(values) if all(value is not None for value in values) else None
            remaining = ledger['stages'][stage]['remaining_usd']
            totals.append({'scenario': scenario, 'method': method, 'stage': stage,
                           'three_budgets_total_usd': total, 'stage_remaining_usd': remaining,
                           'exceeds_stage_remaining': total > remaining if total is not None else None})
        values = [row['three_independent_runs_usd'] for row in chosen]
        total = sum(values) if all(value is not None for value in values) else None
        totals.append({'scenario': scenario, 'method': 'all', 'stage': 'global', 'three_budgets_total_usd': total,
                       'stage_remaining_usd': ledger['remaining_usd'],
                       'exceeds_stage_remaining': total > ledger['remaining_usd'] if total is not None else None})
    status = ('waiting_for_complete_tuning_selection' if not completed_selection else
              'ready_scenario_not_statistical_forecast' if all(row['usd_per_action'] is not None for row in groups) else
              'incomplete_evidence_due_to_censoring_or_missing_cases')
    return {'status': status,
            'selected_scales': selected, 'groups': groups, 'totals': totals,
            'assumptions': ['Each B is an independent run: 50 validation-like boards per family, one seed 11.',
                            'All episodes exhaust B; cost scales linearly with proxy USD per observed action.',
                            'Every logical input token is billed as a 30m cache write, every output at output rate, with configured regional uplift; no local/provider cache discounts.',
                            'Lowest observed rate is only an illustrative optimistic scenario; mean is arithmetic over uncensored validation cases, not a statistical forecast.',
                            'Early solved, frontier-exhausted and resource/API stops are censored; missing groups are never extrapolated.',
                            'The actual pilot can differ substantially in puzzle difficulty, token lengths, generator failure, API pricing, retries and cache reuse.',
                            'The case files and ledger are read separately; while live runs continue, rerun after completion for a coherent view.',
                            'Logical input token counts are a local serialized-payload tokenizer proxy, not guaranteed provider billable token counts.']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--pricing', type=Path, default=DEFAULT_PRICING)
    parser.add_argument('--ledger', type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    budget = read_json(args.pricing)
    price = budget['models']['gpt-6-luna']
    ledger_path = args.ledger or ROOT / budget['ledger_path']
    ledger = ledger_snapshot(ledger_path, budget)
    summary_path = args.results_dir / 'summary.json'
    summary = read_json(summary_path) if summary_path.exists() else {'status': 'missing', 'methods': {}}
    files = sorted(args.results_dir.glob('tune_*.json'))
    if summary.get('status') == 'complete':
        # Never project from a stale or edited summary: independently replay
        # every validation trace and check the manifest before using selection.
        from scripts.archive.campaign_20261102.campaign_scale_stress import checked_summary, source_cases  # pylint: disable=import-outside-toplevel

        manifest = read_json(args.results_dir / 'manifest.json')
        checked_summary(summary, manifest)
        source_cases(args.results_dir, manifest, summary)
    observations = [observation(read_json(path), price) for path in files]
    for row in observations:
        row['physical_ledger_usage'] = [entry for entry in ledger['case_usage'] if entry['tag'] == row['budget_tag']]
    result = {'version': 1, 'pilot_scope': {'boards_per_family': 50, 'budgets': BUDGETS, 'seed': 11,
                                             'independent_runs_per_board': len(BUDGETS)},
              'source': {'results_dir': str(args.results_dir.resolve()), 'ledger_path': str(ledger_path.resolve()),
                         'summary_sha256': sha(summary_path) if summary_path.exists() else None,
                         'case_sha256': {path.name: sha(path) for path in files}, 'pricing_sha256': sha(args.pricing)},
              'pricing': price, 'validation_observations': observations, 'ledger': ledger,
              'projection': project(observations, summary, ledger)}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'estimate.json', result)
    lines = ['# Pilot cost projection', '', f"Status: **{result['projection']['status']}**.",
             'Projection covers 50+50 boards, seed 11, with separate independent runs at B=256,1024,2048.', '',
             'All costs are estimated; the ledger has the actual conservative committed accounting, not an invoice. '
             'No pilot boards or trajectories were opened.', '',
             '| Scenario | Method | Family | Uncensored / selected cases | USD/action | B256 | B1024 | B2048 | Total |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    def money(value: float | None) -> str:
        return 'unknown' if value is None else f'${value:.4f}'
    def rate_string(value: float | None) -> str:
        return 'unknown' if value is None else f'${value:.9f}'
    for row in result['projection']['groups']:
        values = row['independent_budget_cost_usd']
        lines.append(f"| {row['scenario']} | {row['method']} | {row['family']} | "
                     f"{row['uncensored_cases']}/{row['validation_cases']} | {rate_string(row['usd_per_action'])} | "
                     f"{money(values['256'])} | {money(values['1024'])} | {money(values['2048'])} | "
                     f"{money(row['three_independent_runs_usd'])} |")
    lines += ['', '| Scenario | Method | Projected total | Stage/global remaining | Exceeds |', '|---|---|---:|---:|---|']
    for row in result['projection']['totals']:
        lines.append(f"| {row['scenario']} | {row['method']} | {money(row['three_budgets_total_usd'])} | "
                     f"{money(row['stage_remaining_usd'])} | {row['exceeds_stage_remaining']} |")
    lines += ['', '## Observed validation cases', '',
              '| Method | Family | Pair | Actions / budget | Finish | Proxy cost | Ledger committed | Censor reason |',
              '|---|---|---|---:|---|---:|---:|---|']
    for row in observations:
        usage = row['physical_ledger_usage']
        committed = sum(entry['committed_usd'] for entry in usage) if usage else (0.0 if row['physical_requests'] == 0 else None)
        lines.append(f"| {row['method']} | {row['family']} | {row['pair_index']} | "
                     f"{row['attempted_actions']}/{row['budget']} | {row['finish_reason']} | "
                     f"{money(row['uncached_30m_uplift_proxy_usd'])} | {money(committed)} | "
                     f"{row['censor_reason'] or 'none'} |")
    lines += ['', '## Assumptions and limitations', ''] + [f'- {item}' for item in result['projection']['assumptions']]
    lines += ['', 'Every censored case, reason, its logical token counts, per-tag physical usage and the source hashes are in `estimate.json`.']
    (args.output / 'estimate.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"{result['projection']['status']}: {args.output / 'estimate.md'}")


if __name__ == '__main__':
    main()
