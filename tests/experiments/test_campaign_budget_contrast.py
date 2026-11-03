"""The campaign compares matched relative-u and absolute-d without mixing controls."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carl.icl.api import write_json
from carl.icl.budget_sweep import BudgetSweepConfig, _manifest, load_sweep
from carl.icl.config import ExperimentConfig
from carl.icl.reporting import report

SCALES = {'relative_u': [0.9, 0.5], 'absolute_d': [8, 4], 'relative': [0.9, 0.5],
          'absolute': [8, 4], 'ratio_d_over_ac': [1.5, 0.5], 'normalized_offset': [0.75, 0.25], 'fixed': [8, 4]}
BOARD = ['#####', '#---#', '#@$.#', '#---#', '#####']


def configuration(method: str, data: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate({'name': method, 'method': method, 'scales': SCALES[method],
                                            'data_dir': str(data), 'family': 'real_12x12x4', 'split': 'test',
                                            'value_examples': 'initial'})


def build_sweep(tmp_path: Path, methods: list[str]) -> Path:
    files = []
    for i, method in enumerate(methods):
        config = configuration(method, tmp_path)
        config.name = f'{method}_{i}'
        path = tmp_path / f'{method}_{i}.json'
        write_json(path, config.model_dump(mode='json'))
        files.append(path.name)
    write_json(tmp_path / 'matrix.json', {'configs': files})
    path = tmp_path / 'sweep.json'
    write_json(path, {'matrix': 'matrix.json', 'budgets': [1, 2], 'output_dir': str(tmp_path / 'results')})
    return path


def test_sweep_supports_main_pair_and_optional_controls_but_not_ambiguous_methods(tmp_path: Path) -> None:
    path = build_sweep(tmp_path, ['relative_u', 'absolute_d', 'ratio_d_over_ac', 'normalized_offset', 'fixed'])
    _, configs = load_sweep(path)
    assert {item.method for item in configs} == {'relative_u', 'absolute_d', 'ratio_d_over_ac', 'normalized_offset', 'fixed'}
    assert {item.method for item in load_sweep(build_sweep(tmp_path, ['relative', 'fixed']))[1]} == {'relative', 'fixed'}
    with pytest.raises(ValueError, match='multiple conditions'):
        load_sweep(build_sweep(tmp_path, ['relative_u', 'absolute_d', 'fixed', 'fixed']))
    with pytest.raises(ValueError, match='must contain relative_u and absolute_d'):
        load_sweep(build_sweep(tmp_path, ['relative_u', 'fixed']))


def test_sweep_manifest_hashes_calibration_for_all_calibrated_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import carl.icl.budget_sweep as sweep
    methods = ['relative', 'relative_u', 'absolute', 'absolute_d', 'ratio_d_over_ac', 'normalized_offset', 'fixed']
    configs = [configuration(method, tmp_path) for method in methods]
    write_json(tmp_path / 'training.json', {'contexts': []})
    write_json(tmp_path / 'real_12x12x4' / 'test.json', {'instances': []})
    for name in methods:
        write_json(tmp_path / f'cal_{name}.json', {'name': name})
    monkeypatch.setattr(sweep, 'load_data', lambda _config: ({'contexts': []}, [{'id': 'x', 'state': BOARD}]))
    monkeypatch.setattr(sweep, 'components_for_run', lambda *_args: SimpleNamespace())
    monkeypatch.setattr(sweep, 'calibration_path', lambda config, _components: tmp_path / f'cal_{config.name}.json')
    spec = BudgetSweepConfig(matrix='unused', budgets=[1, 2], output_dir='unused')
    manifest = _manifest(spec, configs, 'mock')
    included = set(manifest['inputs_sha256'])
    assert all(str((tmp_path / f'cal_{name}.json').resolve()) in included for name in methods if name != 'fixed')
    assert str((tmp_path / 'cal_fixed.json').resolve()) not in included


def row(method: str, board: str, solved: bool, *, config: str | None = None) -> dict[str, Any]:
    return {'config': config or method, 'method': method, 'family': 'real', 'instance': board, 'seed': 11,
            'solved': solved, 'finish_reason': 'solved' if solved else 'action_budget', 'attempted_actions': 4}


def test_reporting_uses_only_main_pair_and_lists_controls_separately(tmp_path: Path) -> None:
    rows = [row('relative_u', 'a', True), row('absolute_d', 'a', False),
            row('relative_u', 'b', True), row('absolute_d', 'b', True),
            row('relative_u', 'unpaired', False), row('fixed', 'a', False),
            row('ratio_d_over_ac', 'a', True), row('normalized_offset', 'a', False)]
    report(rows, tmp_path)
    pairs = json.loads((tmp_path / 'paired_comparison.json').read_text())
    assert len(pairs) == 1 and pairs[0]['comparison'] == 'relative_u_minus_absolute_d'
    assert pairs[0]['success_delta_relative_u_minus_absolute_d'] == 0.5
    assert pairs[0]['paired_boards'] == 2 and pairs[0]['unpaired_left'] == 1
    text = (tmp_path / 'results.md').read_text()
    assert 'relative_u − absolute_d' in text and 'fixed' in text and 'ratio_d_over_ac' in text
    assert 'normalized_offset' in text and 'separate conditions' in text


def test_reporting_preserves_legacy_pair_and_rejects_incomplete_or_duplicate_main(tmp_path: Path) -> None:
    legacy = [row('relative', 'a', True), row('fixed', 'a', False)]
    report(legacy, tmp_path)
    pairs = json.loads((tmp_path / 'paired_comparison.json').read_text())
    assert pairs[0]['success_delta_relative_minus_fixed'] == 1
    assert (tmp_path / 'paired_comparison.csv').exists()
    # New main condition exists: never substitute relative-vs-fixed during partial execution.
    report(legacy + [row('relative_u', 'a', True)], tmp_path)
    assert json.loads((tmp_path / 'paired_comparison.json').read_text()) == []
    assert not (tmp_path / 'paired_comparison.csv').exists()
    report([row('relative_u', 'a', True), row('relative_u', 'a', False), row('absolute_d', 'a', True)], tmp_path)
    assert json.loads((tmp_path / 'paired_comparison.json').read_text()) == []
    assert 'incomplete or ambiguous' in (tmp_path / 'results.md').read_text()


def test_control_alone_never_implies_a_legacy_main_contrast(tmp_path: Path) -> None:
    report([row('fixed', 'a', False), row('ratio_d_over_ac', 'a', True)], tmp_path)
    assert json.loads((tmp_path / 'paired_comparison.json').read_text()) == []
    assert 'paired comparison (relative' not in (tmp_path / 'results.md').read_text()


def test_distinct_families_can_report_new_and_legacy_contrasts_together(tmp_path: Path) -> None:
    new = [row('relative_u', 'a', True), row('absolute_d', 'a', False)]
    old = [row('relative', 'b', False), row('fixed', 'b', True)]
    for item in old:
        item['family'] = 'legacy_family'
    report(new + old, tmp_path)
    pairs = json.loads((tmp_path / 'paired_comparison.json').read_text())
    assert {pair['comparison'] for pair in pairs} == {'relative_u_minus_absolute_d', 'relative_minus_fixed'}
    header = (tmp_path / 'paired_comparison.csv').read_text().splitlines()[0]
    assert 'success_delta_relative_u_minus_absolute_d' in header
    assert 'success_delta_relative_minus_fixed' in header


def test_paired_api_failures_remain_visible_in_the_matched_denominator(tmp_path: Path) -> None:
    failed = row('relative_u', 'a', False)
    failed['finish_reason'] = 'api_error'
    budget = row('absolute_d', 'a', False)
    budget['finish_reason'] = 'api_budget'
    report([failed, budget], tmp_path)
    pair = json.loads((tmp_path / 'paired_comparison.json').read_text())[0]
    assert pair['paired_runs'] == 1
    assert pair['paired_api_errors_left'] == 1 and pair['paired_api_errors_right'] == 0
    assert pair['paired_api_budget_left'] == 0 and pair['paired_api_budget_right'] == 1
    assert 'Paired API errors: 1 / 0' in (tmp_path / 'results.md').read_text()
