"""Report must never turn unavailable targets or missing shards into failures."""

import csv
import json
import sys
from pathlib import Path

import pytest

from scripts.report_overnight_cpu_stage01 import MODELS
from scripts.report_overnight_cpu_stage01 import OFFSETS
from scripts.report_overnight_cpu_stage01 import cell
from scripts.report_overnight_cpu_stage01 import cpu_usage
from scripts.report_overnight_cpu_stage01 import diagnostics
from scripts.report_overnight_cpu_stage01 import failure_locations
from scripts.report_overnight_cpu_stage01 import findings
from scripts.report_overnight_cpu_stage01 import main
from scripts.report_overnight_cpu_stage01 import make_report
from scripts.report_overnight_cpu_stage01 import planned_rows
from scripts.report_overnight_cpu_stage01 import source_breakdown


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row(model='cllp8', family='12x12x4', k=1, *, source='all', budget=32, **overrides):
    item = {'family': family, 'source_family': source, 'model_id': model, 'k': str(k),
            'budget': str(budget), 'hits': '3', 'planned': '5', 'complete': '5', 'missing': '0',
            'p_hat': '0.6', 'bootstrap_low': '0.2', 'bootstrap_high': '0.8',
            'bound_low': '0.6', 'bound_high': '0.6', 'unavailable_target_slots': '2',
            'mean_success_actions': '4', 'task_solved_by_budget': '1', 'total_no_ops_full_rollout': '2',
            'trials_with_nontrivial_cycle_full_rollout': '1', 'failures_at_action_limit_full_rollout': '2',
            'failures_at_task_terminal_full_rollout': '0'}
    item.update(overrides)
    return item


def test_empty_campaign_report_is_explicitly_partial(tmp_path):
    text = make_report(tmp_path)
    assert '**Report status: partial.**' in text
    assert '**Assessment is not yet available.**' in text
    assert 'No per-model rates' in text
    assert 'not available' in text
    assert 'policy94820' in text and 'current board only' in text
    assert 'eight-hour deadline' in text
    assert 'two historically inspected source distributions (gamma2 and synthetic reverse-board)' in text


def test_cli_requires_explicit_run_dir(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['report_overnight_cpu_stage01'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_partial_cell_never_turns_missing_into_failures_or_confidence_interval():
    partial = row(hits='2', planned='5', complete='3', missing='2', p_hat='',
                  bootstrap_low='', bootstrap_high='', bound_low='0.4', bound_high='0.8')
    result = cell(partial)
    assert result == 'not estimable; missing-data bounds 40.0–80.0%'
    assert 'CI' not in result
    table = planned_rows([partial], '12x12x4', ['cllp8'])
    assert table[0][2] == '2 / 5 / 3'
    assert table[0][-2:] == ['2', '2']
    assert len(table) == len(OFFSETS)
    assert table[1][2] == 'not available'


def test_success_duration_is_conditional_and_full_rollouts_are_counted_once_per_k():
    a = row(k=1, hits='3', complete='5', mean_success_actions='4', task_solved_by_budget='1',
            total_no_ops_full_rollout='2', trials_with_nontrivial_cycle_full_rollout='1')
    b = row(k=2, hits='1', complete='4', mean_success_actions='8', task_solved_by_budget='2',
            total_no_ops_full_rollout='3', trials_with_nontrivial_cycle_full_rollout='2')
    ignored = row(k=1, budget=8, hits='5', complete='5', mean_success_actions='1')
    result = diagnostics([a, b, ignored], '12x12x4', ['cllp8'])[0]
    assert result == ['cllp8', '9', '4', '5.00', '3', '5', '3', '4 / 0']
    assert diagnostics([row(hits='0', mean_success_actions='')], '12x12x4', ['cllp8'])[0][3] == 'not available'
    assert diagnostics([row(hits='1', mean_success_actions='')], '12x12x4', ['cllp8'])[0][3] == 'not available'


def test_failure_locations_keep_completed_nonhits_separate_from_infrastructure():
    records = [
        {'family': '12x12x4', 'model_id': 'cllp8', 'status': 'complete',
         'stop_reason': 'action_limit', 'final_player': '[2, 3]'},
        {'family': '12x12x4', 'model_id': 'cllp8', 'status': 'missing',
         'stop_reason': 'action_limit', 'final_player': '[2, 3]'},
    ]
    assert failure_locations(records) == [['12x12x4', 'cllp8', 'action_limit: 1', '(2, 3): 1']]


def test_source_family_breakdown_does_not_include_the_pooled_duplicate():
    rows = [row(k=16), row(k=16, source='proposal_gamma2_12x12x10', family='12x12x10')]
    result = source_breakdown(rows, ['cllp8'])
    assert len(result) == 1
    assert result[0][:3] == ['12x12x10', 'proposal_gamma2_12x12x10', 'cllp8']


def test_completed_report_requires_all_expected_cells_and_a_finished_runner(tmp_path):
    run = tmp_path
    models = ['cllp8', 'policy94820']
    write_json(run / 'stage_01/evaluation/plan.json', {'model_ids': models, 'split': 'evaluation',
                                                       'family_layouts': {'12x12x4': ['a'], '12x12x10': ['b']},
                                                       'contexts_per_layout': 1, 'planned_trials': 24})
    write_json(run / 'stage_01/evaluation/status.json', {'state': 'complete', 'missing_shards': 0,
                                                         'completed_shards': 4, 'planned_shards': 4})
    write_json(run / 'stage_01/assessment/assessment.json', {'state': 'complete', 'missing_model_trials': 0,
                                                              'completed_model_trials': 24, 'planned_model_trials': 24,
                                                              'invalid_shards': []})
    rows = [row(model, family, k) for family in ('12x12x4', '12x12x10') for model in models for k in OFFSETS]
    rows += [row(model, '12x12x10', 32, source=source)
             for model in ('cllp8', 'cllp32')
             for source in ('proposal_gamma2_12x12x10', 'synthetic_reverse_12x12x10')]
    rows += [row('cllp32', family, k) for family in ('12x12x4', '12x12x10') for k in (16, 32)]
    write_csv(run / 'stage_01/assessment/summary.csv', rows)
    differences = [{'family': family, 'source_family': 'all', 'left': 'cllp8', 'right': 'cllp32',
                    'k': str(k), 'budget': '32', 'planned_pairs': '5', 'paired_pairs': '5', 'missing_pairs': '0',
                    'paired_layouts': '5', 'difference': '0', 'bootstrap_low': '0', 'bootstrap_high': '0'}
                   for family in ('12x12x4', '12x12x10') for k in (16, 32)]
    write_csv(run / 'stage_01/assessment/paired_differences.csv', differences)
    text = make_report(run)
    assert '**Report status: complete.**' in text
    assert '## Findings' in text and 'synthetic reverse-board: CLLP-8 3/5' in text
    assert '| cllp8 | 1 | 3 / 5 / 5 | 60.0% [20.0%, 80.0%] | 2 | 0 |' in text
    assert 'missing targets or contexts' in text
    assert 'not a causal box-count' in text
    write_json(run / 'stage_01/evaluation/status.json', {'state': 'running', 'missing_shards': 1})
    assert '**Report status: partial.**' in make_report(run)


def test_incomplete_or_duplicated_assessment_is_never_reported_complete(tmp_path):
    write_json(tmp_path / 'stage_01/evaluation/status.json', {'state': 'complete', 'missing_shards': 0})
    write_json(tmp_path / 'stage_01/assessment/assessment.json', {'state': 'complete', 'missing_model_trials': 0})
    write_csv(tmp_path / 'stage_01/assessment/summary.csv', [row(), row()])
    text = make_report(tmp_path)
    assert '**Report status: partial.**' in text
    assert 'duplicated or incomplete cell identity' in text
    assert 'No outcome table can be computed safely' in text


def test_supervised_resume_keeps_original_deadline_and_is_quoted(tmp_path):
    write_json(tmp_path / 'stage_01_supervision/run_manifest.json', {'stages': [{
        'name': 'frozen_policy_evaluation', 'timeout_seconds': 123, 'command': [
            'python', '-m', 'example', '--output', 'a path', '--stop-epoch', '123456']}]})
    write_json(tmp_path / 'campaign_manifest.json', {'commands': {
        'stage01_automatic_assessment': ['python', '-m', 'assessment', '--output', 'a path']}})
    text = make_report(tmp_path)
    assert "--output 'a path' --stop-epoch 123456 --resume" in text
    assert 'scripts.intent_experiment_supervisor run --name frozen_policy_evaluation_resume' in text
    assert "python -m assessment --output 'a path'" in text
    assert 'do not reset the eight-hour clock' in text


def test_intentionally_replaced_finalizer_is_not_a_model_failure(tmp_path):
    write_json(tmp_path / 'campaign_manifest.json', {'stages': {'01': {'finalizer_replacement': {
        'previous_stage': 'stage01_automatic_reporting', 'new_stage': 'stage01_automatic_reporting_v2'}}}})
    write_json(tmp_path / 'stage_01_supervision/run_manifest.json', {'stages': [
        {'name': 'stage01_automatic_reporting', 'status': 'failed', 'exit_code': 143},
        {'name': 'stage01_automatic_reporting_v2', 'status': 'completed', 'exit_code': 0}]})
    text = make_report(tmp_path)
    assert 'exited with code 143 during its intentional replacement' in text
    assert 'replacement status: `completed`' in text
    assert 'not a controller or rollout failure' in text


def test_all_default_model_ids_are_unique():
    assert len(MODELS) == len(set(MODELS)) == 6


def test_findings_require_complete_paired_cells_and_handle_reverse_orientation():
    primary = [row(model, family, k, hits='3' if model == 'cllp8' else '2',
                   p_hat='0.6' if model == 'cllp8' else '0.4')
               for family in ('12x12x4', '12x12x10') for model in ('cllp8', 'cllp32') for k in (16, 32)]
    sources = [row(model, '12x12x10', 32, source=source)
               for model in ('cllp8', 'cllp32')
               for source in ('proposal_gamma2_12x12x10', 'synthetic_reverse_12x12x10')]
    differences = [{'family': family, 'source_family': 'all', 'left': 'cllp32', 'right': 'cllp8',
                    'k': str(k), 'budget': '32', 'planned_pairs': '5', 'paired_pairs': '5',
                    'missing_pairs': '0', 'paired_layouts': '5', 'difference': '-0.2',
                    'bootstrap_low': '-0.4', 'bootstrap_high': '0.1'}
                   for family in ('12x12x4', '12x12x10') for k in (16, 32)]
    text = findings(primary + sources, differences)
    assert text is not None
    assert 'CLLP-8 3/5 (60.0% board-weighted)' in text
    assert '+20.0 percentage points' in text and '[-10.0, +40.0] pp' in text
    assert 'gamma2: CLLP-8 3/5' in text and 'synthetic reverse-board: CLLP-8 3/5' in text
    assert 'not a minimum distance' in text
    incomplete = [dict(row) for row in differences]
    incomplete[0]['missing_pairs'] = '1'
    assert findings(primary + sources, incomplete) is None
    assert findings(primary + sources[:-1], differences) is None


def test_cpu_usage_uses_final_total_or_partial_worker_sum():
    assert cpu_usage({'elapsed_cpu_seconds': 45.0, 'elapsed_seconds': 10.0,
                      'resources': {'worker_cpu_seconds': {'1': 1.0}}}) == (45.0, 4.5)
    assert cpu_usage({'elapsed_seconds': 5, 'resources': {'worker_cpu_seconds': {'1': 10, '2': 7}}}) == (17.0, 3.4)
    assert cpu_usage({'elapsed_seconds': 0, 'elapsed_cpu_seconds': 10}) == (10.0, None)
    assert cpu_usage({'elapsed_seconds': 5, 'resources': {'worker_cpu_seconds': {'1': 10, '2': 'unknown'}}}) == (
        None, None)


def test_report_describes_tokenizer_and_actual_cpu_use(tmp_path):
    write_json(tmp_path / 'stage_01/evaluation/status.json', {'state': 'partial', 'elapsed_seconds': 10,
                                                              'elapsed_cpu_seconds': 36, 'resources': {}})
    text = make_report(tmp_path)
    assert 'carl/environment/sokoban/tokenizer.py' in text
    assert '291 tokens' in text and '146 tokens' in text
    assert 'worker CPU seconds: 36.00; average logical cores used' in text
    assert '(CPU seconds / elapsed wall seconds): 3.60' in text
