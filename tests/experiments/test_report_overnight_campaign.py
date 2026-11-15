"""Focused accounting tests for the local overnight campaign report."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.report_overnight_campaign import at_budget
from scripts.report_overnight_campaign import checked_cases
from scripts.report_overnight_campaign import checked_shard_files
from scripts.report_overnight_campaign import curve_line_style
from scripts.report_overnight_campaign import dataset_summary
from scripts.report_overnight_campaign import evaluation_contract_audit
from scripts.report_overnight_campaign import generation_horizons
from scripts.report_overnight_campaign import history_lines
from scripts.report_overnight_campaign import html_document
from scripts.report_overnight_campaign import initial_board_inventory
from scripts.report_overnight_campaign import input_token_sum
from scripts.report_overnight_campaign import joint_success
from scripts.report_overnight_campaign import lifecycle_snapshot
from scripts.report_overnight_campaign import methods_from_freeze
from scripts.report_overnight_campaign import paired_comparisons
from scripts.report_overnight_campaign import planned_cases
from scripts.report_overnight_campaign import prior_stages_audit
from scripts.report_overnight_campaign import replay_audit_summary
from scripts.report_overnight_campaign import solved_cell
from scripts.report_overnight_campaign import success_curves
from scripts.report_overnight_campaign import summarize_cases
from scripts.report_overnight_campaign import trace_comparisons


def board(index: int) -> dict:
    return {'family': '12x12x4', 'source_family': 'rl_validation', 'split': 'evaluation',
            'layout_id': f'board-{index}', 'board_rank': index}


def result(index: int, method: str, status: str, *, solved: bool | None = False) -> dict:
    return {**board(index), 'method': method, 'status': status, 'solved': solved,
            'transitions': 5, 'solve_at_transition': 5 if solved else None,
            'solution_length': 3 if solved else None, 'solution_verified': bool(solved),
            'solve_wall_seconds': 2.0 if solved else None, 'wall_seconds': 3.0,
            'failure_reason': None if solved else status}


FREEZE = {'methods': ['adasubs', 'ksubs', 'direct_policy'],
          'transition_budget': 10, 'per_job_wall_seconds': 10}


def test_missing_infrastructure_and_timeout_are_distinct() -> None:
    boards = [board(1), board(2)]
    shards = [result(1, 'adasubs', 'completed', solved=True), result(2, 'adasubs', 'timeout'),
              result(1, 'ksubs', 'infra_error', solved=None), result(2, 'ksubs', 'completed')]
    cases = planned_cases(boards, FREEZE, shards)
    assert len(cases) == 6
    summary = {(row['source_family'], row['method']): row for row in summarize_cases(cases, FREEZE)}
    ada = summary[('all', 'adasubs')]
    assert (ada['planned'], ada['known'], ada['solved'], ada['timeouts'], ada['missing']) == (2, 2, 1, 1, 0)
    ksubs = summary[('all', 'ksubs')]
    assert (ksubs['known'], ksubs['infra_errors'], ksubs['solved']) == (1, 1, 0)
    direct = summary[('all', 'direct_policy')]
    assert (direct['known'], direct['missing']) == (0, 2)
    assert solved_cell(direct) == 'n/a (0/2 observed)'
    assert direct['solved_per_known'] is None
    assert direct['total_transitions'] is None
    pair = next(row for row in paired_comparisons(cases, FREEZE) if row['left'] == 'adasubs' and row['right'] == 'ksubs')
    assert (pair['known_pairs'], pair['planned_pairs']) == (1, 2)


def test_reject_duplicate_and_unverified_solutions() -> None:
    row = result(1, 'adasubs', 'completed', solved=True)
    with pytest.raises(ValueError, match='Duplicate'):
        planned_cases([board(1)], FREEZE, [row, row])
    with pytest.raises(ValueError, match='verification'):
        planned_cases([board(1)], FREEZE, [{**row, 'solution_verified': False}])
    with pytest.raises(ValueError, match='over-budget'):
        planned_cases([board(1)], FREEZE, [{**row, 'transitions': 11}])


def test_curves_use_observed_event_times_and_planned_denominator() -> None:
    cases = planned_cases([board(1), board(2)], FREEZE, [result(1, 'adasubs', 'completed', solved=True)])
    first = next(row for row in cases if row['layout_id'] == 'board-1' and row['method'] == 'adasubs')
    assert not at_budget(first, 'transitions', 4, FREEZE)
    assert at_budget(first, 'transitions', 5, FREEZE)
    curves = success_curves(cases, FREEZE)
    selected = [row for row in curves if row['method'] == 'adasubs' and row['axis'] == 'transitions']
    assert [(row['budget'], row['solved']) for row in selected] == [(0, 0), (5, 1), (10, 1)]
    assert selected[-1]['fraction_of_planned_lower_bound'] == .5
    wall = [row for row in curves if row['method'] == 'adasubs' and row['axis'] == 'wall_seconds']
    assert [(row['budget'], row['solved']) for row in wall] == [(0, 0), (2, 1), (10, 1)]


def test_curve_styles_distinguish_fixed_k_and_incomplete_results() -> None:
    assert curve_line_style('adasubs', False) == '-'
    assert curve_line_style('ksubs', False) == '--'
    assert curve_line_style('direct_policy', False) == '-'
    assert curve_line_style('ksubs', True) == ':'
    assert curve_line_style('adasubs', True) == ':'


def test_html_is_small_and_references_local_asset() -> None:
    document = html_document('# Results\n\n![Figure](stage_05/figure.png)\n')
    assert 'stage_05/figure.png' in document
    assert 'http://' not in document and 'https://' not in document
    assert len(document) < 4000


def test_lifecycle_reports_only_saved_process_status() -> None:
    assert lifecycle_snapshot({})['owned_processes_stopped'] is None
    result = lifecycle_snapshot({'stages': {'05': {'status': 'complete', 'supervisor_status': 'complete',
                                                    'owned_processes_stopped': True}}})
    assert result['owned_processes_stopped'] is True
    assert result['supervisor_status'] == 'complete'


def test_input_tokens_use_recorded_counts_or_the_verified_fixed_length() -> None:
    rows = [{'policy_calls': 2, 'policy_input_tokens': 582}, {'policy_calls': 3}]
    assert input_token_sum(rows, 'policy_input_tokens', 'policy_calls', 291) == 1455
    assert input_token_sum([{'policy_calls': None}], 'policy_input_tokens', 'policy_calls', 291) is None


def test_generation_horizon_counts_actual_trace_and_flags_mismatch() -> None:
    rows = [{'candidate_generation_calls': 3, 'trace': [{'event': 'generation', 'k': 8},
                                                        {'event': 'candidate'},
                                                        {'event': 'generation', 'k': 8},
                                                        {'event': 'generation', 'k': 4}]},
            {'candidate_generation_calls': 2, 'trace': [{'event': 'generation', 'k': 1}]},
            {'candidate_generation_calls': 1, 'trace': None}]
    result = generation_horizons(rows)
    assert result['calls_by_k'] == {'1': 1, '4': 1, '8': 2}
    assert result['boards_using_k'] == {'1': 1, '4': 1, '8': 1}
    assert result['boards_using_k1_or_k4'] == 2
    assert (result['trace_events'], result['reported_calls']) == (4, 5)
    assert (result['missing_traces'], result['count_mismatches']) == (1, 1)
    assert generation_horizons([])['calls_by_k'] is None


def test_trace_comparison_ignores_timing_but_not_actions() -> None:
    left = result(1, 'adasubs', 'completed', solved=True)
    right = result(1, 'ksubs', 'completed', solved=True)
    left['trace'] = [{'event': 'candidate', 'actions': [0, 1], 'elapsed_seconds': 1.0, 'seconds': .1}]
    right['trace'] = [{'event': 'candidate', 'actions': [0, 1], 'elapsed_seconds': 5.0, 'seconds': .3}]
    rows = trace_comparisons(planned_cases([board(1)], FREEZE, [left, right]), FREEZE)
    x4 = next(row for row in rows if row['family'] == '12x12x4')
    assert x4['same_trace_ignoring_times'] == 1 and x4['trace_pairs_available'] == 1
    right['trace'][0]['actions'] = [1, 0]
    rows = trace_comparisons(planned_cases([board(1)], FREEZE, [left, right]), FREEZE)
    assert next(row for row in rows if row['family'] == '12x12x4')['same_trace_ignoring_times'] == 0


def test_trace_comparison_distinguishes_shorter_horizon_and_timeout() -> None:
    a1 = {**result(1, 'adasubs', 'completed', solved=True), 'trace': [{'event': 'generation', 'k': 4}]}
    b1 = {**result(1, 'ksubs', 'completed'), 'failure_reason': 'queue_exhausted', 'trace': []}
    a2 = {**result(2, 'adasubs', 'completed', solved=True), 'trace': [{'event': 'generation', 'k': 8}]}
    b2 = {**result(2, 'ksubs', 'timeout'), 'failure_reason': 'wall_clock_limit', 'trace': []}
    a3 = {**result(3, 'adasubs', 'completed'), 'trace': []}
    b3 = {**result(3, 'ksubs', 'completed', solved=True), 'trace': []}
    rows = trace_comparisons(planned_cases([board(1), board(2), board(3)], FREEZE, [a1, b1, a2, b2, a3, b3]), FREEZE)
    value = next(row for row in rows if row['family'] == '12x12x4')
    assert (value['adasubs_only_success'], value['ksubs_only_success']) == (2, 1)
    assert value['adasubs_only_used_k1_or_k4'] == 1
    assert value['adasubs_only_used_shorter_and_ksubs_queue_exhausted'] == 1
    assert value['adasubs_only_ksubs_timeout'] == 1


def test_secondary_method_does_not_change_primary_pairs() -> None:
    freeze = {**FREEZE, 'methods': [*FREEZE['methods'], 'primitive_astar']}
    assert len(methods_from_freeze(freeze)) == 4
    cases = planned_cases([board(1)], freeze, [])
    assert len(cases) == 4
    assert {(row['left'], row['right']) for row in paired_comparisons(cases, freeze)} == {
        ('adasubs', 'ksubs'), ('adasubs', 'direct_policy'), ('ksubs', 'direct_policy')}


def test_merge_verification_rejects_changed_or_unlisted_files(tmp_path: Path) -> None:
    shards = tmp_path / 'shards'
    shards.mkdir()
    item = shards / 'one.json'
    item.write_text('{}')
    path = tmp_path / 'merge_manifest.json'
    digest = hashlib.sha256(item.read_bytes()).hexdigest()
    path.write_text(json.dumps({'state': 'complete', 'planned_jobs': 1, 'merged_jobs': 1,
                                'shards_sha256': {'one.json': digest}, 'source_paths': {}}))
    selected, info = checked_shard_files(shards, path)
    assert selected == [item] and info['hash_verified']
    (shards / 'extra.json').write_text('{}')
    item.write_text('{"changed":true}')
    selected, info = checked_shard_files(shards, path)
    assert not selected and not info['hash_verified']
    assert len(info['rejected']) == 1 and info['unlisted'] == ['extra.json']


def test_bad_and_conflicting_shards_do_not_hide_valid_results(tmp_path: Path) -> None:
    row = result(1, 'adasubs', 'completed', solved=True)
    paths = [tmp_path / f'{index}.json' for index in range(4)]
    paths[0].write_text(json.dumps(row))
    paths[1].write_text(json.dumps(row))
    paths[2].write_text('{broken')
    paths[3].write_text(json.dumps(result(1, 'ksubs', 'completed', solved=True)))
    cases, invalid = checked_cases([board(1)], FREEZE, paths)
    assert len(invalid) == 3
    assert next(case for case in cases if case['method'] == 'adasubs')['result'] is None
    assert next(case for case in cases if case['method'] == 'ksubs')['result']['solved'] is True


def test_report_end_to_end_with_missing_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import report_overnight_campaign as report

    campaign = tmp_path / 'campaign'
    stage = campaign / 'stage_05'
    shards = stage / 'evaluation/shards'
    shards.mkdir(parents=True)
    (campaign / 'campaign_manifest.json').write_text(json.dumps({'campaign_id': 'fixture'}))
    expanded = stage / 'expanded_20'
    expanded.mkdir()
    (expanded / 'boards.jsonl').write_text(json.dumps(board(1)) + '\n')
    freeze = {**FREEZE, 'methods': [*FREEZE['methods'], 'primitive_astar'],
              'input_boards': str(expanded / 'boards.jsonl'),
              'checkpoints': {'value': {'path': '/fixture/value', 'weights_sha256': 'abc', 'config_sha256': 'def'}}}
    (stage / 'evaluation_freeze.json').write_text(json.dumps(freeze))
    audits = {}
    for method, name in (('adasubs', 'final_audit_adasubs.json'), ('ksubs', 'final_audit_ksubs.json'),
                         ('direct_policy', 'final_audit_baselines.json')):
        audit_file = stage / name
        audit_file.write_text('{}')
        audits[method] = {'audit_file': str(audit_file), 'audit_sha256': hashlib.sha256(audit_file.read_bytes()).hexdigest()}
    (stage / 'final_audit_manifest.json').write_text(json.dumps(
        {'state': 'partial', 'board_sha256': None, 'expected_methods': freeze['methods'],
         'methods': audits, 'issues': [], 'offline_audit_transitions': 0}))
    (shards / 'adasubs').mkdir()
    (shards / 'adasubs/one.json').write_text(json.dumps(result(1, 'adasubs', 'completed', solved=True)))
    monkeypatch.setattr(report, 'repository_snapshot', lambda: {
        'branch': 'icl-search', 'head': 'abc', 'changed_files': [], 'status_available': True,
        'head_stat': 'abc stage five\n scripts/runner.py | 2 ++',
    })
    monkeypatch.setattr('sys.argv', ['report', '--campaign', str(campaign)])
    report.main()
    summary = json.loads((campaign / 'campaign_summary.json').read_text())
    assert (summary['planned_cases'], summary['known_cases'], summary['missing_cases']) == (4, 1, 3)
    assert summary['stage_05_snapshot'] == 'partial'
    assert (campaign / 'OVERNIGHT_REPORT.md').is_file()
    assert 'scripts/runner.py | 2 ++' in (campaign / 'OVERNIGHT_REPORT.md').read_text()
    assert '`{8}`, `{8,4}`, `{8,1}` and `{8,4,1}`' in (campaign / 'OVERNIGHT_REPORT.md').read_text()
    assert (campaign / 'OVERNIGHT_REPORT.html').is_file()
    assert (stage / 'stage_05_success.pdf').is_file()
    assert all(str(stage / name) in summary['input_sha256'] for name in
               ('final_audit_adasubs.json', 'final_audit_ksubs.json', 'final_audit_baselines.json'))
    assert '[Frozen boards](stage_05/expanded_20/boards.jsonl)' in (campaign / 'stage_05_report.md').read_text()
    assert '`/fixture/value`' in (campaign / 'stage_05_report.md').read_text()


def test_bad_board_hash_produces_partial_report_instead_of_fabricated_counts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import report_overnight_campaign as report

    stage = tmp_path / 'stage_05'
    stage.mkdir()
    (tmp_path / 'campaign_manifest.json').write_text(json.dumps({'campaign_id': 'fixture'}))
    (stage / 'boards.jsonl').write_text(json.dumps(board(1)) + '\n')
    (stage / 'evaluation_freeze.json').write_text(json.dumps({**FREEZE, 'input_boards_sha256': 'wrong'}))
    monkeypatch.setattr('sys.argv', ['report', '--campaign', str(tmp_path)])
    report.main()
    output = json.loads((tmp_path / 'campaign_summary.json').read_text())
    assert output['stage_05_snapshot'] == 'partial'
    assert 'hash differs' in output['stage_05_plan_error']
    assert 'outcome rates are unavailable, not zero' in (tmp_path / 'stage_05_report.md').read_text()


def test_initial_board_inventory_checks_box_counts() -> None:
    value = {**board(1), 'initial_state': ['#@$$$...*--#'] + ['############'] * 11,
             'initial_boxes_on_goals': 1, 'initially_solved': False}
    result = initial_board_inventory([value])
    assert result[0]['initial_boxes_on_goals_distribution'] == {'1': 1}
    with pytest.raises(ValueError, match='Incorrect initial goal count'):
        initial_board_inventory([{**value, 'initial_boxes_on_goals': 2}])


def test_combined_audit_hash_and_unknown_interruption(tmp_path: Path) -> None:
    stage = tmp_path / 'stage_05'
    stage.mkdir()
    first, second, prior = stage / 'first.json', stage / 'second.json', stage / 'prior.json'
    first.write_text('{}')
    second.write_text('{}')
    prior.write_text(json.dumps({'partial_offline_transition_count': None}))

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    methods = {
        'adasubs': {'audit_file': str(first), 'audit_sha256': sha(first),
                    'recorded_transitions_replayed': 2, 'solution_transitions_replayed': 3},
        'ksubs': {'audit_file': str(second), 'audit_sha256': sha(second),
                  'recorded_transitions_replayed': 4, 'solution_transitions_replayed': 0},
    }
    manifest = {'state': 'pass', 'board_sha256': 'board', 'expected_methods': ['adasubs', 'ksubs'],
                'methods': methods, 'issues': [], 'offline_audit_transitions': 9,
                'interrupted_prior_audit': {'path': str(prior), 'sha256': sha(prior), 'offline_transitions': None}}
    (stage / 'final_audit_manifest.json').write_text(json.dumps(manifest))
    freeze = {'methods': ['adasubs', 'ksubs'], 'input_boards_sha256': 'board'}
    output = replay_audit_summary(tmp_path, freeze)
    assert output['state'] == 'pass' and not output['issues']
    assert output['interrupted_prior_audit']['offline_transitions'] is None
    first.write_text('{"changed":true}')
    output = replay_audit_summary(tmp_path, freeze)
    assert output['state'] == 'partial' and output['issues']


def test_prior_stages_audit_checks_named_sources(tmp_path: Path) -> None:
    stage = tmp_path / 'stage_05'
    stage.mkdir()
    script = stage / 'prior_stages_independent_audit.py'
    script.write_text('pass\n')
    source = tmp_path / 'result.csv'
    source.write_text('count\n1\n')
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {'status': 'pass', 'script_sha256': digest(script), 'source_sha256': {'result.csv': digest(source)}}
    (stage / 'prior_stages_independent_audit.json').write_text(json.dumps(audit))
    assert prior_stages_audit(tmp_path)['state'] == 'pass'
    source.write_text('count\n2\n')
    result = prior_stages_audit(tmp_path)
    assert result['state'] == 'partial' and result['issues']


def test_execution_contract_audit_checks_freeze_and_runner_hashes(tmp_path: Path) -> None:
    stage = tmp_path / 'stage_05'
    stage.mkdir()
    freeze_path = stage / 'evaluation_freeze.json'
    freeze = {'methods': ['adasubs'], 'input_boards_sha256': 'board'}
    freeze_path.write_text(json.dumps(freeze))
    contract = stage / 'contract.json'
    contract.write_text('{}')

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    audit = {'state': 'pass', 'boards_sha256': 'board', 'frozen_evaluation_sha256': digest(freeze_path),
             'contracts': {'adasubs': {'path': str(contract), 'sha256': digest(contract)}}}
    (stage / 'evaluation_contract_audit.json').write_text(json.dumps(audit))
    assert evaluation_contract_audit(tmp_path, freeze_path, freeze)['state'] == 'pass'
    contract.write_text('{"changed":true}')
    result = evaluation_contract_audit(tmp_path, freeze_path, freeze)
    assert result['state'] == 'partial' and result['issues']


def test_joint_budget_excludes_late_but_keeps_result_known() -> None:
    late = {**result(1, 'adasubs', 'completed', solved=True), 'solve_wall_seconds': 10.1, 'wall_seconds': 10.2}
    on_time = {**late, 'method': 'ksubs', 'solve_wall_seconds': 10.0, 'wall_seconds': 10.0}
    cases = planned_cases([board(1)], FREEZE, [late, on_time])
    a = next(case for case in cases if case['method'] == 'adasubs')
    b = next(case for case in cases if case['method'] == 'ksubs')
    assert not joint_success(a, FREEZE) and joint_success(b, FREEZE)
    assert not at_budget(a, 'transitions', 10, FREEZE)
    assert not at_budget(a, 'wall_seconds', 10, FREEZE)
    rows = summarize_cases(cases, FREEZE)
    found = next(row for row in rows if row['source_family'] == 'all' and row['method'] == 'adasubs')
    assert (found['known'], found['solved'], found['solutions_found_after_joint_budget']) == (1, 0, 1)
    assert found['infra_errors'] == 0
    pair = paired_comparisons(cases, FREEZE)[0]
    assert pair['difference_right_minus_left'] == 1.0 and pair['known_pairs'] == 1


def test_stage2_reports_mean_and_distinct_duration_denominators_separately(tmp_path: Path) -> None:
    path = tmp_path / 'stage_02/analysis_evaluation/rank_summary.csv'
    path.parent.mkdir(parents=True)
    fields = ['scope', 'family', 'source_family', 'controller', 'L', 'contexts_with_two_distinct_d',
              'error_contexts', 'mean_abs_u_difference', 'strict_comparable', 'inversions']
    row = dict(zip(fields, ['family', '12x12x4', 'all', 'cllp8', '32', '249', '250', '0.0046', '2035', '10']))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    history = dataset_summary(tmp_path)
    rank = history['stage_02']['conditional_rank'][0]
    assert rank['error_contexts'] == '250' and rank['contexts_with_two_distinct_d'] == '249'
    assert '| cllp8 | 250 | 249 | 0.005 | 10/2035 |' in '\n'.join(history_lines(history))
