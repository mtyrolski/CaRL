"""The smaller-board stress has bounded, replayable inputs and honest cost evidence."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from carl.icl.boards import from_rows
from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.search import search
from carl.icl.types import SearchComponents
from scripts.archive.campaign_20261102.campaign_real12_stress import checked_sources, estimate, load_trace
from scripts.archive.campaign_20261102.campaign_scale_stress import DurableEnvironment, DurableTree, verify_actions
from scripts.archive.campaign_20261102.estimate_real12_pilot import METHOD_STAGES, project
from scripts.archive.campaign_20261102.tune_campaign_scales import verify_tree


def sample(method: str = 'relative_u', *, reason: str = 'action_budget') -> dict[str, Any]:
    return {'provider': 'openai', 'seed': 11, 'pair_index': 0, 'instance': {'id': 'board'},
            'config': {'method': method, 'family': 'real_12x12x4', 'scales': [8, 4],
                       'split': 'scale_validation', 'model': 'gpt-6-luna', 'action_budget': 128},
            'outcome': {'attempted_actions': 128 if reason == 'action_budget' else 20,
                        'finish_reason': reason, 'solved': reason == 'solved'},
            'verification': {'verified': True},
            'api': {'logical_input_tokens': 1000, 'logical_output_tokens': 100, 'logical_requests': 4,
                    'cache_hits': 2, 'requests': 2}}


def test_refuses_incomplete_validation(tmp_path: Path) -> None:
    (tmp_path / 'manifest.json').write_text(json.dumps({'provider': 'openai', 'split': 'scale_validation',
                                                        'seed': 11, 'design': {'families': ['real_12x12x4']}}))
    (tmp_path / 'summary.json').write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='completed and eligible'):
        checked_sources(tmp_path)


def test_stress_cost_excludes_censored_runs_but_does_not_hide_them() -> None:
    cases = [sample(), sample(reason='solved')]
    projection = estimate(cases, [8, 4])
    assert projection['sample_cases'] == 1 and projection['censored_cases'] == 1
    assert projection['estimated_uncached_30m_write_usd'] > 0
    with pytest.raises(ValueError, match='No uncensored'):
        estimate([sample(reason='solved'), sample(reason='solved')], [8, 4])


def test_durable_trace_replays_every_action_and_terminal_status(tmp_path: Path) -> None:
    board = from_rows(['#####', '#@--#', '#-$-#', '#-.-#', '#####'])
    path = tmp_path / 'trace.jsonl'
    tree = DurableTree(path)
    env = DurableEnvironment(environment(board), tree)
    component = cast(SearchComponents, SimpleNamespace(value=lambda _state: 0.5))
    result = search(env, component, board, ExperimentConfig(method='primitive', action_budget=4, max_expansions=10),
                    11, observer=tree)
    restored, events = load_trace(path, result)
    verify_actions(restored, board, events)
    assert verify_tree(restored, board, 4)['trace_actions'] == result.attempted_actions == 4
    assert sum(event['event'] == 'action' for event in events) == 4


def test_12x12_cost_scenario_never_declares_budget_projection_a_freeze() -> None:
    cases = [sample(method) for method in METHOD_STAGES for _ in range(2)]
    summary = {'methods': {method: {'selected_pair': [8, 4]} for method in METHOD_STAGES}}
    price = {'cache_write_30m_usd_per_million': .125, 'output_usd_per_million': .5, 'regional_multiplier': 1.1}
    ledger = {'remaining_usd': 40, 'stages': {stage: {'remaining_usd': 12} for stage in METHOD_STAGES.values()}}
    data = project(cases, summary, price, ledger)
    assert data['status'] == 'cost_scenario_not_statistical_forecast' and data['boards'] == 100
    assert data['independent_budgets'] == [128, 256, 512]
    assert len(data['groups']) == 4 and all(group['uncensored_cases'] == 2 for group in data['groups'])
    cases[0] = sample(reason='solved')
    incomplete = project(cases, summary, price, ledger)
    assert next(group for group in incomplete['groups'] if group['method'] == 'relative_u')['total_usd'] is None
