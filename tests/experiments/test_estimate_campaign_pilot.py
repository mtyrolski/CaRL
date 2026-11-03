"""The pilot estimator remains offline, explicit and conservative about censoring."""
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.estimate_campaign_pilot import FAMILIES, METHOD_STAGES, ledger_snapshot, observation, project

PRICE = {'cache_write_30m_usd_per_million': '0.125', 'output_usd_per_million': '0.50',
         'regional_multiplier': '1.10'}


def case(method: str, family: str, *, reason: str = 'action_budget', actions: int = 128) -> dict:
    return {'provider': 'openai', 'seed': 11, 'config': {'method': method, 'family': family,
            'split': 'scale_validation', 'model': 'gpt-6-luna', 'action_budget': 128, 'scales': [8.0, 4.0]},
            'outcome': {'attempted_actions': actions, 'finish_reason': reason, 'solved': reason == 'solved'},
            'api': {'logical_input_tokens': 1000, 'logical_output_tokens': 100, 'logical_requests': 4, 'cache_hits': 2,
                    'requests': 2},
            'verification': {'verified': True}, 'pair_index': 0, 'instance': {'id': family}, 'budget_tag': 'tag'}


def selection() -> dict:
    return {'status': 'complete', 'methods': {method: {'selected_pair': [8, 4]} for method in METHOD_STAGES}}


def ledger() -> dict:
    return {'remaining_usd': 40, 'stages': {name: {'remaining_usd': limit} for name, limit in
            [('pilot_relative', 12), ('pilot_absolute', 12), ('pilot_fixed', 4)]}}


def test_proxy_accounts_all_logical_tokens_and_three_independent_budgets() -> None:
    row = observation(case('relative_u', FAMILIES[0]), PRICE)
    assert row['uncached_30m_uplift_proxy_usd'] == pytest.approx((1000 * .125 + 100 * .5) * 1.1 / 1e6)
    assert not row['censored'] and row['local_cache_hits'] == 2
    observations = [observation(case(method, family), PRICE) for method in METHOD_STAGES for family in FAMILIES]
    result = project(observations, selection(), ledger())
    group = next(item for item in result['groups'] if item['scenario'] == 'mean_observed_rate'
                 and item['method'] == 'relative_u' and item['family'] == FAMILIES[0])
    expected_rate = row['uncached_30m_uplift_proxy_usd'] / 128
    assert group['independent_budget_cost_usd']['256'] == pytest.approx(50 * 256 * expected_rate)
    assert group['three_independent_runs_usd'] == pytest.approx(50 * (256 + 1024 + 2048) * expected_rate)
    assert result['status'] == 'ready_scenario_not_statistical_forecast'


@pytest.mark.parametrize('reason, actions', [('frontier_exhausted', 20), ('api_budget', 35),
                                               ('expansion_budget', 100), ('api_error', 20), ('solved', 20)])
def test_early_stops_and_solutions_are_censored(reason: str, actions: int) -> None:
    row = observation(case('relative_u', FAMILIES[0], reason=reason, actions=actions), PRICE)
    assert row['censored'] and row['proxy_usd_per_action'] is None and row['censor_reason']


def test_missing_winner_or_censored_family_never_yields_full_forecast() -> None:
    rows = [observation(case('relative_u', FAMILIES[0], reason='frontier_exhausted', actions=12), PRICE)]
    partial = project(rows, {'status': 'running', 'methods': {}}, ledger())
    assert partial['status'] == 'waiting_for_complete_tuning_selection'
    assert all(row['three_independent_runs_usd'] is None for row in partial['groups'])
    censored = project(rows, selection(), ledger())
    assert censored['status'] == 'incomplete_evidence_due_to_censoring_or_missing_cases'
    assert all(row['three_budgets_total_usd'] is None for row in censored['totals'])


def test_rejects_pilot_or_mock_data() -> None:
    row = case('fixed', FAMILIES[0])
    row['config']['split'] = 'pilot100'
    with pytest.raises(ValueError, match='scale-validation'):
        observation(row, PRICE)
    row = case('fixed', FAMILIES[0])
    row['provider'] = 'mock'
    with pytest.raises(ValueError, match='actual live'):
        observation(row, PRICE)


def test_readonly_ledger_includes_unknown_and_reservations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'budget.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE metadata (key TEXT,value TEXT)')
        db.execute("INSERT INTO metadata VALUES ('effective_limit_nanos','40000000000')")
        db.execute('CREATE TABLE attempts (stage TEXT,tag TEXT,status TEXT,charged_nanos INTEGER,reserved_nanos INTEGER,usage_json TEXT)')
        db.executemany('INSERT INTO attempts VALUES (?,?,?,?,?,?)', [
            ('scale_validation', 'case', 'settled', 100000000, 600000000, json.dumps({'input_tokens': 50, 'output_tokens': 3})),
            ('calibration', 'case', 'unknown', None, 500000000, None),
            ('pilot_fixed', 'case', 'reserved', None, 600000000, None),
        ])
    budget = {'limit_usd': '40.00', 'stage_limits_usd': {'scale_validation': '4.00', 'pilot_fixed': '4.00'}}
    data = ledger_snapshot(path, budget)
    assert data['committed_usd'] == pytest.approx(1.2) and data['remaining_usd'] == pytest.approx(38.8)
    assert data['stages']['pilot_fixed']['remaining_usd'] == pytest.approx(3.4)
    assert data['case_usage'][0]['input_tokens'] == 50
    monkeypatch.setenv('CARL_CAMPAIGN_MAX_USD', '5.00')
    assert ledger_snapshot(path, budget)['remaining_usd'] == pytest.approx(3.8)
