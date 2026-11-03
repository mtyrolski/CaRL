"""Estimate the two core methods on 100 unseen 12x12 boards; never open pilot data."""
import argparse
import json
from pathlib import Path
from typing import Any

from carl.icl.api import write_json
from carl.icl.config import ROOT
from scripts.archive.campaign_20261102.campaign_real12_stress import TUNING, checked_sources
from scripts.archive.campaign_20261102.estimate_campaign_pilot import ledger_snapshot, observation

OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/real12_v1/pilot_cost'
BUDGETS = (128, 256, 512)
METHOD_STAGES = {'relative_u': 'pilot_relative', 'absolute_d': 'pilot_absolute'}
BOARDS = 100


def project(cases: list[dict[str, Any]], summary: dict[str, Any], price: dict[str, Any],
            ledger: dict[str, Any]) -> dict[str, Any]:
    """Keep incomplete or censored inputs unknown; avoid inferring zero spending."""
    groups = []
    for method, stage in METHOD_STAGES.items():
        selected = summary['methods'][method]['selected_pair']
        rows = [observation(case, price) for case in cases
                if case['config']['method'] == method and case['config']['scales'] == selected]
        usable = [row['proxy_usd_per_action'] for row in rows if not row['censored']]
        complete = len(rows) == 2 and len(usable) == 2
        rates = [float(value) for value in usable if value is not None]
        for scenario, rate in (('mean', sum(rates) / len(rates) if complete else None),
                               ('max_observed', max(rates) if complete else None)):
            costs = {str(b): None if rate is None else rate * b * BOARDS for b in BUDGETS}
            total = None if rate is None else sum(float(value) for value in costs.values() if value is not None)
            remaining = ledger['stages'][stage]['remaining_usd']
            groups.append({'method': method, 'scales': selected, 'scenario': scenario, 'validation_cases': len(rows),
                           'uncensored_cases': len(rates), 'usd_per_action': rate, 'independent_budgets_usd': costs,
                           'total_usd': total, 'stage_remaining_usd': remaining,
                           'exceeds_stage': None if total is None else total > remaining})
    totals = []
    for scenario in ('mean', 'max_observed'):
        values = [group['total_usd'] for group in groups if group['scenario'] == scenario]
        total = sum(values) if len(values) == 2 and all(value is not None for value in values) else None
        totals.append({'scenario': scenario, 'total_usd': total, 'global_remaining_usd': ledger['remaining_usd'],
                       'exceeds_global': None if total is None else total > ledger['remaining_usd']})
    return {'status': 'cost_scenario_not_statistical_forecast', 'boards': BOARDS, 'families_observed': ['real_12x12x4'],
            'independent_budgets': list(BUDGETS), 'groups': groups, 'totals': totals,
            'assumptions': ['No pilot data read; 100 future 12x12 boards assumed validation-like.',
                            'Each budget is an independent run, all exhaust B; linear observed tokens/action.',
                            'Every logical input token priced as a 30m cache write with configured regional uplift; '
                            'actual local/provider cache discounts and costs can differ.',
                            'Two validation boards cannot yield a statistically reliable forecast.',
                            'The exact 100-board cohort, B*, stress outcome and costs must be frozen separately; '
                            'the projected cost alone does not prevent work within the hard ledger cap.',
                            'Other baselines, further tuning, failures and retries require additional funds.']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuning-dir', type=Path, default=TUNING)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    budget = json.loads((ROOT / 'configs/icl/campaign_budget.json').read_text())
    ledger = ledger_snapshot(ROOT / budget['ledger_path'], budget)
    try:
        manifest, summary, cases = checked_sources(args.tuning_dir.resolve())
    except FileNotFoundError as error:
        result = {'status': 'waiting_for_complete_verified_12x12_validation', 'reason': str(error), 'ledger': ledger}
    except ValueError as error:
        summary_path = args.tuning_dir / 'summary.json'
        status = json.loads(summary_path.read_text()).get('status') if summary_path.exists() else None
        state = ('validation_integrity_error' if status == 'complete'
                 else 'waiting_for_complete_verified_12x12_validation')
        result = {'status': state, 'reason': str(error), 'ledger': ledger}
    else:
        result = {'manifest_policy_fingerprint': manifest['policy_fingerprint'], 'ledger': ledger,
                  'projection': project(cases, summary, budget['models']['gpt-6-luna'], ledger)}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'estimate.json', result)
    if 'projection' not in result:
        body = f"# 12x12 pilot cost scenario\n\nStatus: **{result['status']}**.\n\n{result['reason']}\n"
    else:
        projection = result['projection']
        lines = ['# 12x12 pilot cost scenario', '', f"Status: **{projection['status']}**.", '',
                 '| Method | Scenario | Cases | $/action | B128 | B256 | B512 | Total | Stage remaining | Exceeds |',
                 '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
        def money(value: Any) -> str:
            return 'unknown' if value is None else f'${value:.4f}'
        for row in projection['groups']:
            costs = row['independent_budgets_usd']
            lines.append(f"| {row['method']} | {row['scenario']} | {row['uncensored_cases']}/{row['validation_cases']} | "
                         f"{money(row['usd_per_action'])} | {money(costs['128'])} | {money(costs['256'])} | "
                         f"{money(costs['512'])} | {money(row['total_usd'])} | {money(row['stage_remaining_usd'])} | "
                         f"{row['exceeds_stage']} |")
        lines += ['', '| Scenario | Two-method total | Global remaining | Exceeds |', '|---|---:|---:|---|']
        for row in projection['totals']:
            lines.append(f"| {row['scenario']} | {money(row['total_usd'])} | {money(row['global_remaining_usd'])} | "
                         f"{row['exceeds_global']} |")
        lines += ['', '## Assumptions'] + [f'- {text}' for text in projection['assumptions']]
        body = '\n'.join(lines) + '\n'
    (args.output / 'estimate.md').write_text(body)
    print(f"Saved: {args.output / 'estimate.md'}")


if __name__ == '__main__':
    main()
