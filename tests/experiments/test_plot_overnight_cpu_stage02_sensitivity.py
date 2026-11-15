"""Checks for reading and plotting the frozen scale-sensitivity summaries."""

import json

import pytest

from scripts.plot_overnight_cpu_stage02_sensitivity import build_figure_data
from scripts.plot_overnight_cpu_stage02_sensitivity import file_sha
from scripts.plot_overnight_cpu_stage02_sensitivity import verified_inputs


@pytest.fixture
def inputs():
    pool = {'scope': 'family', 'source_family': 'all', 'family': '12x12x4', 'controller': 'cllp8',
            'L': '32', 'pool_variant': 'demo4_bfs6', 'reference': 'fixed', 'comparison': 'reduced',
            'planned_contexts': '12', 'complete_pool_contexts': '10', 'contexts_with_shared_success': '8',
            'shared_success_targets': '24', 'n_layouts': '7', 'mean_abs_delta_u': '0.12',
            'ci_low': '0.08', 'ci_high': '0.18'}
    budget = {'source_family': 'all', 'family': '12x12x4', 'controller': 'cllp32',
              'pool_variant': 'demo4_bfs4', 'short_L': '8', 'long_L': '16', 'planned_contexts': '8',
              'complete_contexts': '7', 'contexts_with_shared_success': '6', 'shared_success_targets': '18',
              'delta_layouts': '6', 'mean_abs_delta_u': '0.06', 'ci_low': '0.01', 'ci_high': '0.1'}
    return pool, budget


def test_exact_predeclared_comparisons_and_no_data_are_retained(inputs):
    pool, budget = inputs
    source_specific = {**pool, 'scope': 'source', 'source_family': 'some_distribution', 'mean_abs_delta_u': '0.9'}
    other_limit = {**pool, 'L': '16', 'mean_abs_delta_u': '0.9'}
    other_variant = {**pool, 'pool_variant': 'underfilled', 'mean_abs_delta_u': '0.9'}
    rows = build_figure_data([pool, source_specific, other_limit, other_variant], [budget])
    assert len(rows) == 48
    assert sum(row['status'] == 'available' for row in rows) == 2
    first = next(row for row in rows if row['panel'] == 'pool' and row['status'] == 'available')
    assert first['mean_abs_delta_u'] == pytest.approx(0.12)
    assert (first['planned_contexts'], first['complete_contexts'], first['contexts_with_shared_success'],
            first['shared_success_targets'], first['independent_layouts']) == (12, 10, 8, 24, 7)
    second = next(row for row in rows if row['panel'] == 'budget' and row['status'] == 'available')
    assert (second['reference'], second['comparison'], second['mean_abs_delta_u']) == (8, 16, 0.06)
    assert all(row['mean_abs_delta_u'] is None for row in rows if row['status'] == 'no_data')


def test_duplicate_and_invalid_denominators_are_rejected(inputs):
    pool, budget = inputs
    with pytest.raises(ValueError, match='Duplicate pool'):
        build_figure_data([pool, pool], [budget])
    with pytest.raises(ValueError, match='denominators'):
        build_figure_data([{**pool, 'contexts_with_shared_success': '11'}], [budget])
    with pytest.raises(ValueError, match='confidence'):
        build_figure_data([{**pool, 'ci_low': '0.6', 'ci_high': '0.2'}], [budget])


def test_inputs_are_bound_to_same_frozen_profiles(tmp_path):
    analysis, budget = tmp_path / 'analysis', tmp_path / 'budget'
    analysis.mkdir()
    budget.mkdir()
    profile_path = analysis / 'profiles.jsonl'
    profile_path.write_text('{}\n', encoding='utf-8')
    pool_path = analysis / 'sensitivity_summary.csv'
    pool_path.write_text('test\n', encoding='utf-8')
    (budget / 'budget_sensitivity_summary.csv').write_text('test\n', encoding='utf-8')
    (analysis / 'analysis_manifest.json').write_text(json.dumps({'output_sha256': {
        'profiles.jsonl': file_sha(profile_path), 'sensitivity_summary.csv': file_sha(pool_path)}}), encoding='utf-8')
    budget_manifest = budget / 'budget_sensitivity_manifest.json'
    budget_manifest.write_text(json.dumps({'profiles_sha256': file_sha(profile_path)}), encoding='utf-8')
    _, _, hashes = verified_inputs(analysis, budget)
    assert len(hashes) == 4
    budget_manifest.write_text(json.dumps({'profiles_sha256': 'incorrect'}), encoding='utf-8')
    with pytest.raises(ValueError, match='same frozen profiles'):
        verified_inputs(analysis, budget)
