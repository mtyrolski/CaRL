"""Scale tuning uses matched caps and audits the actual search trace."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from carl.icl.api import BudgetExceeded
from carl.icl.boards import from_rows
from carl.icl.config import ExperimentConfig, Limits
from carl.icl.datasets import environment
from carl.icl.search import search
from carl.icl.search_tree import SearchTreeRecorder
from carl.icl.types import SearchComponents
from scripts.tune_campaign_scales import CappedAPI, read_design, summarize, verify_tree


def test_generator_cap_counts_logical_cache_hits(tmp_path: Path) -> None:
    limits = Limits(max_requests=5, max_input_tokens=1000, max_total_input_tokens=5000,
                    max_output_tokens=1000, wall_seconds=30, timeout_seconds=10)
    api = CappedAPI(tmp_path, 'mock', limits, max_generator_calls=2, max_total_tokens=5000)
    payload = {'input': 'x', 'metadata': {'component': 'generator'}, 'max_output_tokens': 32}
    def mock() -> dict:
        return {'output_text': '{}', 'usage': {'output_tokens': 1}}
    api.call('responses', payload, mock)
    api.call('responses', payload, mock)
    assert api.metrics['generator_calls'] == 2 and api.metrics['cache_hits'] == 1
    with pytest.raises(BudgetExceeded, match='generator request limit'):
        api.call('responses', payload, mock)


def test_combined_token_cap_reserves_output_before_query(tmp_path: Path) -> None:
    limits = Limits(max_requests=5, max_input_tokens=1000, max_total_input_tokens=5000,
                    max_output_tokens=1000, wall_seconds=30, timeout_seconds=10)
    payload = {'input': 'x', 'metadata': {'component': 'policy'}, 'max_output_tokens': 32}
    api = CappedAPI(tmp_path, 'mock', limits, max_generator_calls=3, max_total_tokens=1)
    with pytest.raises(BudgetExceeded, match='Combined logical input/output'):
        api.call('responses', payload, lambda: {'output_text': '{}'})
    assert api.metrics['logical_requests'] == 0
    exact_prompt = len(api.encoding.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=()))
    api.max_total_tokens = exact_prompt + 32
    api.call('responses', payload, lambda: {'output_text': '{}', 'usage': {'output_tokens': 1}})
    with pytest.raises(BudgetExceeded, match='Combined logical input/output'):
        api.call('responses', payload, lambda: {'output_text': '{}', 'usage': {'output_tokens': 1}})
    assert api.metrics['logical_requests'] == 1


def test_trace_replay_accounts_noops_and_rejects_corruption() -> None:
    board = from_rows(['#####', '#@--#', '#-$-#', '#-.-#', '#####'])
    components = cast(SearchComponents, SimpleNamespace(value=lambda _state: 0.5))
    config = ExperimentConfig(method='primitive', action_budget=4, max_expansions=10)
    tree = SearchTreeRecorder()
    outcome = search(environment(board), components, board, config, 11, observer=tree)
    assert outcome.attempted_actions == 4
    result = verify_tree(tree, board, config.action_budget)
    assert result['verified'] and result['trace_actions'] == 4
    tree.nodes[1].state = tree.nodes[0].state[:-1]
    with pytest.raises(ValueError, match='endpoint'):
        verify_tree(tree, board, config.action_budget)


def test_zero_success_uses_first_preregistered_pair() -> None:
    design = read_design(Path('configs/icl/campaign_data/scale_tuning/design.json'))
    cases = []
    for method in design['config_by_method']:
        for index in range(2):
            for _family in design['families']:
                cases.append({'config': {'method': method}, 'pair_index': index,
                              'outcome': {'solved': False, 'attempted_actions': 1 if index == 0 else 20,
                                          'finish_reason': 'action_budget'}, 'verification': {'verified': True}})
    result = summarize(design, cases, 1)
    assert all(row['all_zero'] and row['selected_pair'] == design['scale_pairs'][name][0]
               for name, row in result.items())


def test_resource_failure_categories_and_early_limit_gate() -> None:
    from carl.icl.search import SearchResult
    from scripts.tune_campaign_scales import failure_category

    assert failure_category(SearchResult(finish_reason='api_budget', error='Campaign or stage scale_validation budget would be exceeded')) == 'campaign_stage_guard'
    assert failure_category(SearchResult(finish_reason='api_budget', error='Logical generator request limit exhausted')) == 'local_generator_limit'
    assert failure_category(SearchResult(finish_reason='api_error', error='Invalid JSON from model component')) == 'model_parse_or_validation_error'
    design = read_design(Path('configs/icl/campaign_data/scale_tuning/design.json'))
    cases = []
    for method in design['config_by_method']:
        for index in range(2):
            for _family in design['families']:
                cases.append({'config': {'method': method, 'action_budget': 128}, 'pair_index': index,
                              'outcome': {'solved': False, 'attempted_actions': 5, 'finish_reason': 'api_budget'},
                              'failure_category': 'local_api_limit', 'verification': {'verified': True}})
    result = summarize(design, cases, 1)
    assert all(row['selected_pair'] is None for row in result.values())


def test_run_case_saves_tree_even_if_replay_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import tune_campaign_scales as tuning
    from carl.icl.types import Instance, TrainingData

    item = cast(Instance, {'id': 'case', 'state': ['#####', '#@--#', '#-$-#', '#-.-#', '#####'], 'level_key': 'x'})
    training = cast(TrainingData, {'contexts': []})
    config = ExperimentConfig(method='primitive', few_shot=0, action_budget=1, cache_dir=str(tmp_path))
    monkeypatch.setattr(tuning, 'components_for_run', lambda *_args: SimpleNamespace(calibrated=None))
    def fail(*_args: object) -> None:
        raise RuntimeError('simulator replay failed')
    monkeypatch.setattr(tuning, 'verify_tree', fail)
    case = tuning.run_case(config, 'mock', item, training, 10, 10000, 11, 'case-tag', 1)
    assert not case['verification']['verified'] and case['verification']['error_type'] == 'RuntimeError'
    assert case['outcome']['attempted_actions'] == 1 and len(case['tree']['nodes']) > 1


def test_family_gate_detects_32x32_failure_despite_global_success() -> None:
    from scripts.tune_campaign_scales import family_limit_summary

    cases = []
    for family in ('real', 'large'):
        for index in range(6):
            cases.append({'config': {'family': family, 'action_budget': 128},
                          'outcome': {'attempted_actions': 5 if family == 'large' and index < 5 else 128},
                          'failure_category': 'local_api_limit' if family == 'large' and index < 5 else None})
    groups = family_limit_summary(cases, ['real', 'large'], 0.75)
    assert not groups['real']['blocked']
    assert groups['large'] == {'cases': 6, 'early_local_limit_stops': 5, 'blocked': True}
