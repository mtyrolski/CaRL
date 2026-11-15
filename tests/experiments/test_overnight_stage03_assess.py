"""Tests for grounded intent diagnostics and frozen board permutations."""

import json

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from scripts.overnight_stage03_assess import Condition
from scripts.overnight_stage03_assess import Settings
from scripts.overnight_stage03_assess import bootstrap_salt
from scripts.overnight_stage03_assess import check_multisets
from scripts.overnight_stage03_assess import digest
from scripts.overnight_stage03_assess import evaluate_job
from scripts.overnight_stage03_assess import freeze_plan
from scripts.overnight_stage03_assess import index_predictions
from scripts.overnight_stage03_assess import make_jobs
from scripts.overnight_stage03_assess import metric
from scripts.overnight_stage03_assess import recover_rows
from scripts.overnight_stage03_assess import summarize
from scripts.overnight_stage03_generate import transform_rows


def board(extra_wall):
    grid = [['#' if r in (0, 11) or c in (0, 11) else '-' for c in range(12)] for r in range(12)]
    grid[1][1], grid[1][3], grid[1][6] = '@', '$', '.'
    grid[9][extra_wall] = '#'
    return [''.join(row) for row in grid]


def context(i, source='real', transforms=None):
    rows = board(i + 2)
    return {'cohort_id': 'audit', 'context_id': f'ctx{i}', 'layout_id': canonical_layout_id(from_rows(rows)),
            'family': '12x12x4', 'source_family': source, 'split': 'test', 'expected_methods': ['base', 'icl'],
            'expected_transforms': transforms or ['identity'], 'board': rows}


def output(context_record, method='base', raw=None, status='complete', transform='identity'):
    if raw is None and status == 'complete':
        raw = json.dumps({'effects': [{'predicate': 'box_at', 'row': 1, 'col': 4}], 'preserve': [],
                          'instruction': None})
    return {**context_record, 'method': method, 'transform': transform,
            'source_rows': transform_rows(context_record['board'], transform), 'raw_output': raw,
            'prediction_status': status}


def job(context_record, *, method='base', transform='identity'):
    return {'cohort_id': context_record['cohort_id'], 'context_id': context_record['context_id'], 'method': method,
            'transform': transform, 'condition': Condition.ORIGINAL if transform == 'identity' else Condition.D4,
            'permutation_seed': None, 'donor_context_id': context_record['context_id'],
            'donor_layout_id': context_record['layout_id'], 'job_id': 'test'}


def test_derangements_are_shared_between_methods_and_preserve_missing_raw_frequencies():
    contexts = [context(0), context(1), context(2, 'synthetic'), context(3, 'synthetic')]
    plan = freeze_plan(contexts, Settings())
    jobs = make_jobs(contexts, plan)
    for group in plan['groups']:
        for permutation in group['permutations']:
            assert all(row['target_layout_id'] != row['donor_layout_id'] for row in permutation['mapping'])
            for mapping in permutation['mapping']:
                same = [item for item in jobs if item['condition'] == Condition.PERMUTED
                        and item['context_id'] == mapping['target_context_id']
                        and item['permutation_seed'] == permutation['seed']]
                assert {item['donor_context_id'] for item in same} == {mapping['donor_context_id']}
    predictions = [output(c, method) for c in contexts for method in c['expected_methods']]
    predictions[1] = output(contexts[0], 'icl', status='missing')
    checks = check_multisets(plan, index_predictions(predictions, contexts))
    assert len(checks) == 12
    assert all(row['status'] == 'frequency_preserved' for row in checks)


def test_single_board_stratum_is_unavailable_not_missing_model_output():
    ctx = context(0)
    plan = freeze_plan([ctx], Settings())
    jobs = make_jobs([ctx], plan)
    shuffled = [item for item in jobs if item['condition'] == Condition.PERMUTED]
    assert len(shuffled) == 6 and all(item['donor_context_id'] is None for item in shuffled)
    record = evaluate_job(shuffled[0], ctx, None, Settings(max_nodes=100), {})
    assert record['output_status'] == 'permutation_unavailable' and record['valid_nontrivial'] is None


def test_parser_rejects_duplicate_keys_and_classifies_fence_wall_range_and_satisfied():
    ctx = context(0)
    raw = json.dumps({'effects': [{'predicate': 'box_at', 'row': 1, 'col': 4}], 'preserve': [],
                      'instruction': None})
    fenced = evaluate_job(job(ctx), ctx, output(ctx, raw=f'```json\n{raw}\n```'), Settings(max_nodes=500), {})
    assert not fenced['strict_json_valid'] and fenced['json_valid'] and fenced['fence_removed']
    assert fenced['valid_nontrivial'] and fenced['bounded_path_found']
    for col, flag in ((0, 'wall_error'), (12, 'range_error'), (3, 'any_effect_initially_true')):
        malformed = json.dumps({'effects': [{'predicate': 'box_at', 'row': 1, 'col': col}], 'preserve': []})
        result = evaluate_job(job(ctx), ctx, output(ctx, raw=malformed), Settings(max_nodes=100), {})
        assert result[flag] and not result['valid_nontrivial']
    duplicate = '{"effects":[],"effects":[{"predicate":"box_at","row":1,"col":4}],"preserve":[]}'
    result = evaluate_job(job(ctx), ctx, output(ctx, raw=duplicate), Settings(max_nodes=100), {})
    assert not result['json_valid'] and result['bfs_status'] == 'not_applicable_invalid_intent'


def test_support_coordinate_copy_and_whole_goal_flags_are_explicit():
    ctx = context(0)
    raw = json.dumps({'effects': [{'predicate': 'box_at', 'row': 1, 'col': 4},
                                  {'predicate': 'boxes_on_goals_at_least', 'count': 1}],
                      'preserve': [], 'instruction': None})
    prediction = output(ctx, 'icl', raw)
    prediction['prompt'] = {'messages': [{'role': 'assistant', 'content': raw},
                                         {'role': 'assistant', 'content': raw}]}
    result = evaluate_job(job(ctx, method='icl'), ctx, prediction, Settings(max_nodes=500), {})
    assert result['requests_whole_goal'] and not result['whole_goal_only']
    assert result['support_status'] == 'two_support_examples' and result['matches_support_coordinate']
    assert result['matches_support_predicate_coordinate'] and result['effect_count'] == 2


def test_inverse_d4_path_replayed_and_natural_instruction_marked_untranslated():
    ctx = context(0, transforms=['identity', 'rot90_ccw'])
    value = {'effects': [{'predicate': 'box_at', 'row': 7, 'col': 1}], 'preserve': [],
             'instruction': 'Move the box towards row 7 column 1.'}
    prediction = output(ctx, raw=json.dumps(value), transform='rot90_ccw')
    result = evaluate_job(job(ctx, transform='rot90_ccw'), ctx, prediction, Settings(max_nodes=500), {})
    assert result['valid_nontrivial'] and result['bounded_path_found'] and result['inverse_replay_verified']
    assert result['instruction_mapping'] == 'untranslated_natural_instruction'
    assert result['inverse_intent']['effects'][0] == {'predicate': 'box_at', 'row': 1, 'col': 4}


def test_missing_and_invalid_are_distinct_in_pairing_and_summary():
    ctx = context(0)
    done = evaluate_job(job(ctx), ctx, output(ctx), Settings(max_nodes=500), {})
    missing = evaluate_job(job(ctx), ctx, None, Settings(max_nodes=500), {})
    assert metric(missing, 'valid') is None
    assert missing['output_status'] == 'missing' and done['valid_nontrivial']
    summaries, paired = summarize([done], Settings(max_nodes=500))
    assert summaries[0]['completed_outputs'] == 1 and summaries[0]['bounded_paths_found'] == 1
    assert not paired


def test_resume_refuses_changed_donor_even_if_record_digest_is_recomputed(tmp_path):
    ctx = context(0)
    original_job = job(ctx)
    prediction = output(ctx)
    row = evaluate_job(original_job, ctx, prediction, Settings(max_nodes=500), {})
    row['record_sha256'] = digest(row)
    path = tmp_path / 'assessments.jsonl'
    path.write_text(json.dumps(row) + '\n', encoding='utf-8')
    contexts = {(ctx['cohort_id'], ctx['context_id']): ctx}
    indexed = {(ctx['cohort_id'], ctx['context_id'], 'base', 'identity'): prediction}
    assert len(recover_rows(path, [original_job], contexts, indexed)) == 1
    row['raw_output'] = 'tampered'
    row['record_sha256'] = digest({key: value for key, value in row.items() if key != 'record_sha256'})
    path.write_text(json.dumps(row) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='donor raw output'):
        recover_rows(path, [original_job], contexts, indexed)


def test_bootstrap_intervals_are_identical_after_jsonl_resume():
    ctx = context(0)
    template = evaluate_job(job(ctx), ctx, output(ctx), Settings(max_nodes=500), {})
    rows = []
    for index in range(24):
        for draw, condition in enumerate((Condition.ORIGINAL, Condition.PERMUTED,
                                          Condition.PERMUTED, Condition.PERMUTED)):
            valid = ((index % 5) < 2 if draw == 0 else (index * (draw + 1) + draw) % 7 < 3)
            found = valid and (index + draw) % 3 == 0
            rows.append({**template, 'layout_id': f'layout-{index}', 'context_id': f'ctx-{index}',
                         'condition': condition, 'permutation_seed': None if draw == 0 else 20261114 + draw,
                         'strict_json_valid': valid, 'strict_semantic_valid': valid, 'json_valid': valid,
                         'valid_nontrivial': valid, 'bounded_path_found': found,
                         'bfs_search': {'expanded_nodes': 1, 'primitive_edges_examined': 4} if valid else None,
                         'bfs_actual_search_seconds': 0.01 if valid else None,
                         'bfs_status': ('witness_found' if found else 'node_limit') if valid else
                                       'not_applicable_invalid_intent'})
    restored = json.loads(json.dumps(rows))
    settings = Settings(bootstrap_draws=199)
    assert bootstrap_salt(('test', Condition.ORIGINAL), 'valid') == bootstrap_salt(('test', 'original'), 'valid')
    summary, paired = summarize(rows, settings)
    resumed_summary, resumed_paired = summarize(restored, settings)
    assert summary == resumed_summary and paired == resumed_paired
    assert any(row['ci_low'] is not None and row['ci_low'] != row['ci_high'] for row in paired)
