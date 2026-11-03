"""The 1024 stress cannot run before qualified tuning or hide cost/trace failures."""
import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from carl.icl.boards import from_rows
from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.search import SearchResult, search
from carl.icl.search_tree import SearchTreeRecorder, TreeNode
from carl.icl.types import SearchComponents
from scripts import campaign_scale_stress as stress
from scripts.tune_campaign_scales import verify_tree


def completed_summary() -> tuple[dict[str, Any], dict[str, Any]]:
    def pair(index: int, scales: list[float]) -> dict[str, Any]:
        return {'pair_index': index, 'scales': scales, 'eligible_for_selection': True}
    methods = {'relative_u': {'pairs': [pair(0, [0.85, 0.5]), pair(1, [0.65, 0.25])], 'selected_pair': [0.85, 0.5]},
               'absolute_d': {'pairs': [pair(0, [8, 4]), pair(1, [6, 3])], 'selected_pair': [8, 4]},
               'fixed': {'pairs': [pair(0, [8, 4]), pair(1, [6, 3])], 'selected_pair': [8, 4]}}
    summary = {'status': 'complete', 'methods': methods, 'family_limits': {
        'real_12x12x4': {'blocked': False}, 'generated_32x32x10': {'blocked': False}}}
    return summary, {'provider': 'openai', 'split': 'scale_validation', 'seed': 11}


def test_stress_requires_completed_eligible_validation() -> None:
    summary, manifest = completed_summary()
    assert stress.checked_summary(summary, manifest) == 0
    summary['status'] = 'running'
    with pytest.raises(ValueError, match='completed live'):
        stress.checked_summary(summary, manifest)
    summary['status'] = 'complete'
    summary['methods']['fixed']['pairs'][1]['eligible_for_selection'] = False
    with pytest.raises(ValueError, match='eligible and complete'):
        stress.checked_summary(summary, manifest)
    summary['methods']['fixed']['pairs'][1]['eligible_for_selection'] = True
    summary['family_limits']['generated_32x32x10']['blocked'] = True
    with pytest.raises(ValueError, match='families'):
        stress.checked_summary(summary, manifest)


def test_stress_requires_headroom_in_both_budget_limits() -> None:
    forecast = {'estimated_uncached_30m_write_usd': 1.0}
    ledger = {'stage_limit_usd': 4.0, 'stage_committed_usd': 2.0,
              'global_effective_limit_usd': 40.0, 'global_committed_usd': 38.0,
              'max_single_request_reservation_usd': 0.6}
    stress.ensure_budget_headroom(forecast, ledger)
    with pytest.raises(ValueError, match='remaining campaign or stage'):
        stress.ensure_budget_headroom(forecast, {**ledger, 'stage_committed_usd': 2.5})
    with pytest.raises(ValueError, match='remaining campaign or stage'):
        stress.ensure_budget_headroom(forecast, {**ledger, 'global_committed_usd': 38.5})


def test_api_parser_and_stage_stops_cannot_be_budget_evidence() -> None:
    verified = {'verified': True}
    assert stress.budget_evidence(SearchResult(finish_reason='action_budget', attempted_actions=1024), verified, None)
    assert not stress.budget_evidence(SearchResult(finish_reason='solved', solved=True), verified, None)
    assert not stress.budget_evidence(SearchResult(finish_reason='action_budget', attempted_actions=20), verified, None)
    assert not stress.budget_evidence(SearchResult(finish_reason='api_error', solved=True), verified, None)
    assert not stress.budget_evidence(SearchResult(finish_reason='api_budget', attempted_actions=1024), verified, None)
    assert not stress.budget_evidence(SearchResult(finish_reason='expansion_budget', attempted_actions=1024), verified, None)
    assert not stress.budget_evidence(SearchResult(finish_reason='solved', solved=True), {'verified': False}, None)


def test_projection_uses_selected_32x32_actual_cost_and_rejects_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stress, 'ROOT', tmp_path)
    budget_path = tmp_path / 'configs/icl/campaign_budget.json'
    budget_path.parent.mkdir(parents=True)
    budget_path.write_text(json.dumps({'ledger_path': 'budget.sqlite3', 'models': {'gpt-6-luna': {
        'cache_write_30m_usd_per_million': '.125', 'output_usd_per_million': '.5', 'regional_multiplier': '1.1'}}}))
    with sqlite3.connect(tmp_path / 'budget.sqlite3') as db:
        db.execute('CREATE TABLE attempts (stage TEXT, tag TEXT, status TEXT, charged_nanos INTEGER, reserved_nanos INTEGER, usage_json TEXT)')
        usage = json.dumps({'input_tokens': 100, 'output_tokens': 20})
        db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?)', ('scale_validation', 'sample', 'settled', 1000, 9999, usage))
    sample = {'config': {'family': 'generated_32x32x10', 'method': 'relative_u'}, 'pair_index': 0,
              'outcome': {'attempted_actions': 64}, 'budget_tag': 'sample',
              'api': {'logical_input_tokens': 6400, 'logical_output_tokens': 1280, 'logical_requests': 64,
                      'requests': 1, 'cache_hits': 63, 'input_tokens': 100, 'output_tokens': 20}}
    result = stress.estimate([sample], 0)
    assert result['sample_ledger_charged_usd'] == 0.000001
    assert result['projected_input_tokens'] == 102400 and result['projected_output_tokens'] == 20480
    assert result['estimated_uncached_30m_write_usd'] == pytest.approx(0.025344)
    sample['api']['input_tokens'] = 101
    with pytest.raises(ValueError, match='disagrees'):
        stress.estimate([sample], 0)


def test_append_only_trace_contains_actual_search_and_replays(tmp_path: Path) -> None:
    board = from_rows(['#####', '#@--#', '#-$-#', '#-.-#', '#####'])
    class Primitive:
        def value(self, _state: Any) -> float:
            return 0.5
        def probabilities(self, _state: Any, _target: Any) -> np.ndarray:
            raise AssertionError('Primitive search must not call policy')
        def propose(self, _state: Any, _scale: float) -> list[Any]:
            raise AssertionError('Primitive search must not call generator')
    config = ExperimentConfig(method='primitive', action_budget=1, max_expansions=4)
    recorder = stress.DurableTree(tmp_path / 'trace.jsonl')
    env = stress.DurableEnvironment(environment(board), recorder)
    outcome = search(env, cast(SearchComponents, Primitive()), board, config, 11, observer=recorder)
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert events[0]['event'] == 'start' and events[-1]['event'] == 'finish'
    assert any(event['event'] == 'transition' for event in events)
    restored = SearchTreeRecorder()
    restored.nodes = [TreeNode(**event['node']) for event in events if event['event'] in {'start', 'transition'}]
    restored.result = SearchResult(**events[-1]['outcome'])
    assert verify_tree(restored, board, 1)['verified']
    stress.verify_actions(restored, board, events)
    events = [event for event in events if event['event'] != 'action']
    with pytest.raises(ValueError, match='action records'):
        stress.verify_actions(restored, board, events)
    assert outcome.attempted_actions == 1
