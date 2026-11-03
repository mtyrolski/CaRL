"""Export measured campaign results to CSV without model calls or notebook execution."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.plot_data_export.calibration import build_calibration
from scripts.plot_data_export.common import CAMPAIGN, DATA, OUTPUT, PILOT, read_json, sha, write_csv
from scripts.plot_data_export.comparison import comparison
from scripts.plot_data_export.components import component_costs, components, write_search_attempts
from scripts.plot_data_export.pilot import aggregate, load_pilot


def column_names(rows: list[dict[str, Any]], fallback: list[str] | None = None) -> list[str]:
    names = list(fallback or [])
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def other_boards(pilot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = list(pilot)
    for family in ('real_12x12x4', 'generated_32x32x10'):
        for split in ('component_validation', 'scale_validation'):
            path = DATA / family / f'{split}.json'
            if not path.is_file():
                continue
            for item in read_json(path)['instances']:
                result.append({'campaign_id': 'campaign_100_20261102', 'split': split, 'family': family,
                               'board_id': item['board_id'], 'layout_id': item['layout_id'],
                               'state_hash': item['state_hash']})
    policy_boards = DATA.parent / 'campaign_policy_validation_v1/real_12x12x4/component_validation.json'
    for item in read_json(policy_boards)['instances']:
        result.append({'campaign_id': 'campaign_100_20261102', 'split': 'component_validation',
                       'family': 'real_12x12x4', 'board_id': item['board_id'],
                       'layout_id': item['layout_id'], 'state_hash': item['state_hash']})
    for item in read_json(DATA / 'demonstrations.json')['instances']:
        result.append({'campaign_id': 'campaign_100_20261102', 'split': 'train',
                       'family': item['family'], 'board_id': item['board_id'],
                       'layout_id': item['layout_id'], 'state_hash': item['state_hash']})
    keys = [(row['split'], row['family'], row['board_id']) for row in result]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate board ID within a split')
    return result


def validate(runs: list[dict[str, Any]], curve: list[dict[str, Any]],
             pairs: list[dict[str, Any]], boards: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [row['run_key'] for row in runs]
    if len(keys) != len(set(keys)) or len(keys) != 600:
        raise ValueError('Expected 600 unique frozen pilot cells')
    known = {(row['split'], row['family'], row['board_id']) for row in boards}
    if any((r['split'], r['family'], r['board_id']) not in known for r in runs):
        raise ValueError('Pilot run does not join to boards')
    for row in runs:
        if row['execution_status'] == 'complete' and row['replay_verified'] is not True:
            raise ValueError('Completed pilot run lacks a successful trace replay')
        if row['execution_status'] != 'complete' and row['solved'] is not None:
            raise ValueError('Incomplete pilot cells must have NULL solved')
    observed = [r for r in runs if r['budget_actions'] == 256 and r['method'] == 'relative_u'
                and r['execution_status'] == 'complete' and r['replay_verified']
                and r['completion_class'] != 'local_limit_before_B']
    successes = sum(bool(row['solved']) for row in observed)
    point = next(row for row in curve if row['budget_actions'] == 256 and row['method'] == 'relative_u')
    if point['verified_successes'] != successes or point['denominator_boards'] != len(observed):
        raise ValueError('Solve-rate point disagrees with raw runs')
    maps = {method: {r['board_id']: r for r in runs if r['budget_actions'] == 256 and r['method'] == method
                     and r['execution_status'] == 'complete' and r['replay_verified']
                     and r['completion_class'] != 'local_limit_before_B'}
            for method in ('relative_u', 'absolute_d')}
    common = sorted(set(maps['relative_u']) & set(maps['absolute_d']))
    raw_delta = (sum(int(maps['relative_u'][key]['solved']) - int(maps['absolute_d'][key]['solved'])
                     for key in common) / len(common)) if common else None
    difference = next(row for row in pairs if row['budget_actions'] == 256)
    if difference['complete_pairs'] != len(common) or difference['delta_relative_minus_absolute'] != raw_delta:
        raise ValueError('Paired delta disagrees with raw runs')
    return {'planned_cells': len(runs), 'completed_cells': sum(r['execution_status'] == 'complete' for r in runs),
            'failed_cells': sum(r['execution_status'] == 'failed' for r in runs),
            'interrupted_cells': sum(r['execution_status'] == 'interrupted' for r in runs),
            'missing_cells': sum(r['execution_status'] == 'missing' for r in runs),
            'completed_actions': sum(r['attempted_actions'] or 0 for r in runs if r['execution_status'] == 'complete'),
            'manual_B256_relative': {'successes': successes, 'denominator': len(observed)},
            'manual_B256_paired_delta': {'pairs': len(common), 'delta': raw_delta}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pilot_boards, runs, freeze = load_pilot()
    boards = other_boards(pilot_boards)
    curve, paired = aggregate(runs, freeze)
    validation = validate(runs, curve, paired, boards)
    percentiles = freeze['conditions']['relative_u']['config']['scales']
    tables = {'boards': boards, 'runs': runs, 'solve_rate_by_budget': curve, 'paired_deltas': paired}
    tables.update(build_calibration(percentiles))
    tables.update(components())
    tables.update(comparison())
    targets = {(row['context_id'], row['target_id']): row for row in tables['calibration_targets']}
    if len(targets) != len(tables['calibration_targets']):
        raise ValueError('Duplicate calibration target')
    observed_trials: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in tables['calibration_trials']:
        key = (row['context_id'], row['target_id'])
        if key not in targets or row['board_id'] != targets[key]['board_id']:
            raise ValueError('Calibration trial does not join its target')
        observed_trials.setdefault(key, []).append(row)
    for key, target in targets.items():
        group = observed_trials.get(key, [])
        if len(group) != target['R_attempted'] or sum(bool(row['exact_hit']) for row in group) != target['hits']:
            raise ValueError('Calibration hit/attempt totals do not match trials')
    board_keys = {(row['split'], row['family'], row['board_id']) for row in boards}
    for name in ('component_policy', 'component_generator', 'component_value'):
        if any((row['split'], row['family'], row['board_id']) not in board_keys for row in tables[name]):
            raise ValueError(f'{name} contains a board missing from boards.csv')
    tables['component_costs'] = component_costs()
    fallbacks = {'figure1_ecdf': ['split', 'board_id', 'context_id', 'd', 'F_right', 'target_ids'],
                 'figure1_quantiles': ['split', 'board_id', 'context_id', 'percentile', 'quantile_d', 'rule'],
                 'component_generator': ['split', 'family', 'board_id', 'variant', 'k', 'reached_any'],
                 'component_value': ['split', 'family', 'board_id', 'variant', 'position', 'predicted_value']}
    for name, rows in tables.items():
        write_csv(output / f'{name}.csv', rows, column_names(rows, fallbacks.get(name)))
    validation.update(write_search_attempts(output / 'search_attempts.jsonl.gz'))
    if validation['trace_completed_actions'] != validation['completed_actions']:
        raise ValueError('Completed action total disagrees with raw traces')
    (output / 'validation.json').write_text(json.dumps(validation, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    sources = [CAMPAIGN / 'campaign.md', CAMPAIGN / 'scale_tuning/real12_v1/freeze.json',
               PILOT / 'pilot_manifest.json', CAMPAIGN / 'policy_validation/report.json',
               DATA / 'calibration/08e8317905e28ebc90073b94aad3ffea004ef265f5ad2d09065e44c2137d80ab/calibration.json']
    manifest = {'source_sha256': {str(path): sha(path) for path in sources},
                'export_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(output.glob('*.csv'))},
                'validation': validation, 'note': 'Offline snapshot; rerun after new checkpoints or ledger activity.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
