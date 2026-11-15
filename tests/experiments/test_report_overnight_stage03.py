"""Tests for the aggregate report's denominators and provenance checks."""

import json

import pytest

from scripts.report_overnight_stage03 import board_stats
from scripts.report_overnight_stage03 import chronology_excerpt
from scripts.report_overnight_stage03 import command_to_new_output
from scripts.report_overnight_stage03 import figure_data
from scripts.report_overnight_stage03 import file_sha
from scripts.report_overnight_stage03 import mode_diagnostics
from scripts.report_overnight_stage03 import overlap_diagnostics
from scripts.report_overnight_stage03 import plot
from scripts.report_overnight_stage03 import representative_examples
from scripts.report_overnight_stage03 import symmetry_degeneracy
from scripts.report_overnight_stage03 import token_cap_diagnostics
from scripts.report_overnight_stage03 import unresolved_count
from scripts.report_overnight_stage03 import verified_assessment


def record(cohort, layout, family, source, method, condition='original', transform='identity', *, valid=True,
           status='complete', seed=None, signature='signature'):
    return {'cohort_id': cohort, 'layout_id': layout, 'context_id': f'ctx-{layout}', 'family': family,
            'source_family': source, 'method': method, 'condition': condition, 'transform': transform,
            'permutation_seed': seed, 'output_status': status, 'valid_nontrivial': valid if status == 'complete' else None,
            'json_valid': True if status == 'complete' else None, 'intent_signature': signature,
            'support_status': 'no_examples', 'coordinate_effect_count': 1, 'requests_whole_goal': False,
            'bounded_path_found': valid if status == 'complete' else None, 'source_rows': ['########']*12,
            'raw_output': '{}', 'job_id': f'{cohort}-{layout}-{method}-{condition}-{transform}-{seed}'}


def test_modes_include_semantically_invalid_and_exclude_permutation_and_d4():
    rows = [record('fresh', 'a', 'x4', 'real', 'icl', valid=True),
            record('fresh', 'b', 'x4', 'real', 'icl', valid=False),
            {**record('fresh', 'c', 'x4', 'real', 'icl', valid=False), 'json_valid': False, 'intent_signature': None},
            record('fresh', 'a', 'x4', 'real', 'icl', condition='permuted', seed=1),
            record('fresh', 'a', 'x4', 'real', 'icl', condition='d4', transform='rot90_ccw')]
    result = mode_diagnostics(rows)
    assert len(result) == 1
    assert (result[0]['completed_outputs'], result[0]['parseable_outputs'], result[0]['most_common_count']) == (3, 2, 2)
    assert result[0]['most_common_share_of_completed'] == pytest.approx(2/3)
    assert result[0]['most_common_share_of_parseable'] == 1.0


def test_three_permutations_average_per_board_and_never_triple_board_count():
    rows = [record('legacy', 'a', 'x10', 'gamma', 'base', condition='permuted', valid=value, seed=seed)
            for seed, value in ((1, True), (2, True), (3, False))]
    rows += [record('legacy', 'b', 'x10', 'gamma', 'base', condition='permuted', valid=False, seed=seed)
             for seed in (1, 2, 3)]
    stats = board_stats(rows, 'valid')
    assert (stats['planned_outputs'], stats['planned_boards'], stats['known_boards']) == (6, 2, 2)
    assert stats['mean'] == pytest.approx(1/3)
    assert stats['ci_low'] is None and stats['ci_high'] is None


def test_figure_separates_cohorts_and_reports_mixed_sources(tmp_path):
    history, fresh = [], []
    for index in range(2):
        for method in ('base', 'icl_legacy', 'lora_validation_220'):
            history.append(record('legacy_transfer_40', f'h{index}', '12x12x10', 'proposal_gamma2_12x12x10', method))
            for seed in (1, 2, 3):
                history.append(record('legacy_transfer_40', f'h{index}', '12x12x10', 'proposal_gamma2_12x12x10',
                                      method, condition='permuted', seed=seed))
        for family in ('12x12x4', '12x12x10'):
            source = ('rl_validation_12x12x4' if family.endswith('x4') else
                      'proposal_gamma2_12x12x10' if index == 0 else 'synthetic_reverse_12x12x10')
            for method in ('base', 'in_context', 'checkpoint220'):
                for transform in ('identity', 'rot90_ccw', 'horizontal_reflection'):
                    fresh.append(record('stage03_fresh_evaluation', f'{family}-{index}', family, source, method,
                                        condition='original' if transform == 'identity' else 'd4', transform=transform))
    data = figure_data(history, fresh)
    assert len(data) == 24
    row = next(item for item in data if item['panel'] == 'fresh_x10')
    assert row['source_layout_counts'] == {'proposal_gamma2_12x12x10': 1, 'synthetic_reverse_12x12x10': 1}
    plot(tmp_path, data)
    for extension in ('pdf', 'svg', 'png'):
        assert (tmp_path / f'stage_03_figure.{extension}').stat().st_size > 500


def test_overlap_symmetries_and_token_cap_keep_their_denominators():
    history = [{'cohort_id': 'old', 'layout_id': 'a'}, {'cohort_id': 'old', 'layout_id': 'b'}]
    fresh = [{'layout_id': 'a', 'family': 'x10', 'source_family': 'gamma'},
             {'layout_id': 'c', 'family': 'x4', 'source_family': 'real'}]
    overlap = overlap_diagnostics(history, fresh)
    assert overlap['fresh_overlap_union'] == overlap['fresh_not_in_historical_union'] == 1
    plan = {'cases': [{'context_id': 'one', 'transform': 'rot90_ccw', 'original_rows': ['x'], 'source_rows': ['y']},
                      {'context_id': 'one', 'transform': 'rot90_ccw', 'original_rows': ['x'], 'source_rows': ['y']},
                      {'context_id': 'one', 'transform': 'horizontal_reflection', 'original_rows': ['x'], 'source_rows': ['x']}]}
    assert symmetry_degeneracy(plan)['planned_transformed_boards'] == 2
    assert symmetry_degeneracy(plan)['identical_to_original'] == 1
    samples = [{'family': 'x4', 'source_family': 'real', 'method': 'base', 'transform': 'identity',
                'prediction_status': 'complete', 'generated_tokens': n} for n in (128, 42, None)]
    cap = token_cap_diagnostics(samples, 128)[0]
    assert (cap['completed_outputs'], cap['token_counts_known'], cap['at_token_cap']) == (3, 2, 1)
    samples[0]['generated_tokens'] = 129
    with pytest.raises(ValueError, match='outside the frozen cap'):
        token_cap_diagnostics(samples, 128)


def test_examples_are_selected_in_id_order_without_using_success():
    first = record('stage03_fresh_evaluation', 'a', 'x4', 'real', 'base', valid=True)
    first.update(bounded_path_found=False, bfs_status='node_limit')
    second = record('stage03_fresh_evaluation', 'b', 'x4', 'real', 'base', valid=True)
    second.update(bounded_path_found=True, bfs_status='witness_found')
    examples = representative_examples([], [second, first])
    assert examples[2]['record']['layout_id'] == 'a'
    assert examples[2]['record']['bounded_path_found'] is False


def test_assessment_and_chronology_must_match_manifest(tmp_path):
    output = tmp_path / 'assessment'
    output.mkdir()
    names = ('assessments.jsonl', 'summary.csv', 'paired_comparisons.csv', 'equivariance_summary.csv')
    (output/'assessments.jsonl').write_text('{}\n', encoding='utf-8')
    for name in names[1:]:
        (output/name).write_text('field\n', encoding='utf-8')
    manifest = {'finished_jobs': 1, 'planned_jobs': 1, 'inputs_sha256': {},
                'outputs_sha256': {name: file_sha(output/name) for name in names}}
    (output/'assessment_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (output/'status.json').write_text('{"state":"complete"}', encoding='utf-8')
    assert len(verified_assessment(output)[1]) == 1
    (output/'summary.csv').write_text('changed\n', encoding='utf-8')
    with pytest.raises(ValueError, match='differs from manifest'):
        verified_assessment(output)
    chron = tmp_path/'report.md'
    chron.write_text('# Preliminary\n\n## Decision chronology\n08:54:56 after 08:54:05.\n\n## Other\nignore\n')
    assert chronology_excerpt(chron)['description'] == '08:54:56 after 08:54:05.'


def test_reproduction_changes_only_the_output_directory(tmp_path):
    command = ['env', 'OMP_NUM_THREADS=1', 'python', '-m', 'scripts.overnight_stage03_assess',
               '--contexts', 'source.jsonl', '--resume', '--output', 'existing']
    reproduced = command_to_new_output(command, tmp_path / 'new')
    assert reproduced.endswith(f'--output {tmp_path / "new"}')
    assert '--resume' not in reproduced
    assert command[-1] == 'existing'
    with pytest.raises(ValueError, match='exactly one'):
        command_to_new_output(['python'], tmp_path)


def test_unresolved_excludes_found_paths_and_infrastructure_errors():
    row = {'valid_nontrivial': '11', 'bounded_paths_found': '7', 'bfs_infrastructure_errors': '1'}
    assert unresolved_count(row) == 3
    row['bfs_infrastructure_errors'] = '5'
    with pytest.raises(ValueError, match='exceed'):
        unresolved_count(row)
