"""Offline reports keep interrupted attempts and cost categories separate."""
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.report_calibration_v2 import ledger_summary, summarize


def test_ledger_snapshot_filters_campaign_stage_and_preserves_unsettled(tmp_path: Path) -> None:
    ledger = tmp_path / 'ledger.sqlite3'
    with sqlite3.connect(ledger) as db:
        db.execute('CREATE TABLE attempts(stage TEXT, tag TEXT, status TEXT, charged_nanos INT, reserved_nanos INT, usage_json TEXT)')
        db.execute('CREATE TABLE events(stage TEXT, tag TEXT, kind TEXT)')
        usage = json.dumps({'input_tokens': 100, 'output_tokens': 20,
                            'input_tokens_details': {'cached_tokens': 30, 'cache_write_tokens': 50}})
        db.executemany('INSERT INTO attempts VALUES (?,?,?,?,?,?)', [
            ('calibration', 'calibration_v2', 'settled', 1300, 9000, usage),
            ('calibration', 'calibration_v2', 'unknown', None, 2000, None),
            ('calibration', 'calibration_v2', 'reserved', None, 3000, None),
            ('calibration', 'other', 'settled', 5000, 9000, usage),
        ])
        db.executemany('INSERT INTO events VALUES (?,?,?)', [
            ('calibration', 'calibration_v2', 'logical_request'),
            ('calibration', 'calibration_v2', 'live_cache_hit'),
            ('calibration', 'other', 'logical_request'),
        ])
    result = ledger_summary(ledger)
    assert result['total_attempts'] == 3
    assert result['attempts']['settled'] == {'attempts': 1, 'committed_usd': 0.0000013}
    assert result['attempts']['unknown']['committed_usd'] == 0.000002
    assert result['attempts']['reserved']['committed_usd'] == 0.000003
    assert result['events']['logical_request'] == result['events']['live_cache_hit'] == 1
    assert result['events']['local_replay_hit'] == 0
    assert result['tokens'] == {'input': 100, 'output': 20, 'cache_read': 30, 'cache_write': 50}
    assert result['attempts_without_usage'] == 2


def test_calibration_summary_keeps_target_failures_and_early_task_solve_separate() -> None:
    summary = {'status': 'partial', 'signature': 'a', 'provider': 'mock', 'context_count': 2,
               'config': {'trials': 2, 'sampled_targets': 2, 'reference_window': 4, 'action_cap': 4},
               'contexts': [{'context': 'a', 'excluded': False, 'distinct_d': 1}], 'contexts_completed': 1}
    targets = [{'context': 'a', 'target_index': 0, 'k': 1, 'trials': 2, 'hits': 1, 'p_hat': .5, 'excluded_reason': None},
               {'context': 'a', 'target_index': 1, 'k': 2, 'trials': 2, 'hits': 0, 'p_hat': 0, 'excluded_reason': 'zero_exact_hits'},
               {'context': 'b', 'target_index': 0, 'k': 3, 'trials': 1, 'hits': 0, 'p_hat': None, 'excluded_reason': 'incomplete'}]
    trials = [{'exact_hit': True, 'task_solved': True, 'solved_before_target': False, 'attempted_actions': 1},
              {'exact_hit': False, 'task_solved': True, 'solved_before_target': True, 'attempted_actions': 2}]
    report = summarize(summary, targets, trials)
    coverage = report['coverage']
    assert coverage['planned_max_targets'] == 4 and coverage['targets_completed'] == 2
    assert coverage['zero_hit_targets'] == 1 and coverage['targets_incomplete'] == 1
    assert coverage['exact_hits'] == 1 and coverage['solved_before_target'] == 1
    assert coverage['single_time_contexts'] == 1 and report['p_hat_distribution'] == {'0': 1, '0.5': 1}
    summary.update(status='completed', trials=3)
    with pytest.raises(ValueError, match='artifact disagree'):
        summarize(summary, targets, trials)


def test_policy_support_counts_distinct_cached_inputs(tmp_path: Path) -> None:
    from scripts.report_calibration_v2 import policy_support

    ledger = tmp_path / 'ledger.sqlite3'
    cache = tmp_path / 'cache' / 'openai'
    cache.mkdir(parents=True)
    with sqlite3.connect(ledger) as db:
        db.execute('CREATE TABLE events(stage TEXT, tag TEXT, kind TEXT, request_hash TEXT)')
        db.executemany('INSERT INTO events VALUES (?,?,?,?)', [
            ('calibration', 'calibration_v2', 'logical_request', 'a'),
            ('calibration', 'calibration_v2', 'logical_request', 'a'),
            ('calibration', 'calibration_v2', 'logical_request', 'b'),
            ('calibration', 'calibration_v2', 'logical_request', 'missing'),
            ('calibration', 'other', 'logical_request', 'outside'),
        ])
    for digest, probabilities in [('a', [0, 1, 0, 0]), ('b', [0.2, 0.8, 0, 0])]:
        (cache / f'{digest}.json').write_text(json.dumps({'request_hash': digest,
                                                          'output_text': json.dumps({'probabilities': probabilities})}))
    result = policy_support(ledger, tmp_path / 'cache')
    assert result['unique_logical_policy_inputs'] == 3
    assert result['point_mass'] == result['multiple_actions'] == result['missing'] == 1
    assert result['invalid'] == 0
