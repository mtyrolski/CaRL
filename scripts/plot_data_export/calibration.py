"""Export training-only calibrated targets, trials and empirical Figure 1 data."""
from collections import defaultdict
from typing import Any

import numpy as np

from scripts.plot_data_export.common import CALIBRATION, DATA, read_json


def build_calibration(percentiles: list[float]) -> dict[str, list[dict[str, Any]]]:
    summary = read_json(CALIBRATION / 'calibration.json')
    targets = read_json(CALIBRATION / 'calibration_targets.json')
    trials = read_json(CALIBRATION / 'calibration_trials.json')
    demos = read_json(DATA / 'demonstrations.json')['instances']
    boards = {f"trajectory_{row['source_index']}": row['board_id'] for row in demos}
    contexts = {row['context']: row for row in summary['contexts']}
    if any(row['context'] not in boards for row in targets):
        raise ValueError('Calibration context has no source board ID')
    target_rows = []
    for row in targets:
        context = row['context']
        target_rows.append({'split': 'train', 'board_id': boards[context], 'context_id': context,
                            'target_id': row['target_index'], 'k': row['k'],
                            'R_attempted': row['trials'], 'hits': row['hits'], 'p_hat': row['p_hat'],
                            'd': row.get('d'), 'w': row.get('weight'), 'F_left': row.get('cdf_before'),
                            'F_right': row.get('cdf_at'), 'u': row.get('u'),
                            'a_c': contexts[context].get('a_c'), 'excluded_reason': row.get('excluded_reason'),
                            'state': row['state'], 'target': row['subgoal'], 'target_hash': row['target_hash']})
    trial_rows = [{'split': 'train', 'board_id': boards[row['context']], 'context_id': row['context'],
                   'target_id': row['target_index'], 'k': row['k'], 'trial': row['trial'],
                   'seed_entropy': row['seed_entropy'], 'rng': row['rng'], 'cap': summary['config']['action_cap'],
                   'attempted_actions': row['attempted_actions'], 'exact_hit': row['exact_hit'],
                   'first_hit': row.get('first_hit'), 'task_solved_early': row['solved_before_target'],
                   'actions': row['actions'], 'endpoint_hash': row['endpoint']} for row in trials]
    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        by_k[row['k']].append(row)
    reachability = []
    for k, group in sorted(by_k.items()):
        board_means = [np.mean([r['p_hat'] for r in group if r['board_id'] == board])
                       for board in sorted({r['board_id'] for r in group})]
        hits = sum(row['hits'] for row in group)
        attempts = sum(row['R_attempted'] for row in group)
        reachability.append({'split': 'train', 'k': k, 'targets': len(group), 'boards': len(board_means),
                             'attempts': attempts, 'exact_hits': hits,
                             'hit_rate_per_attempt': hits / attempts if attempts else None,
                             'mean_target_p_hat': float(np.mean([r['p_hat'] for r in group])),
                             'mean_board_p_hat': float(np.mean(board_means))})
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        groups[row['context_id']].append(row)
    # Deterministic illustration rule: most distinct successful times, then
    # most successful targets, then context ID. No test-board data is used.
    selected = sorted(groups, key=lambda context: (
        -len({r['d'] for r in groups[context] if r['d'] is not None}),
        -sum(r['d'] is not None for r in groups[context]), context))[:3]
    figure_targets, ecdf, quantiles = [], [], []
    for context in selected:
        group = groups[context]
        figure_targets.extend(group)
        kept = [r for r in group if r['d'] is not None and r['w'] is not None]
        for time in sorted({r['d'] for r in kept}):
            cumulative = sum(r['w'] for r in kept if r['d'] <= time)
            ecdf.append({'split': 'train', 'board_id': boards[context], 'context_id': context,
                         'd': time, 'F_right': cumulative,
                         'target_ids': [r['target_id'] for r in kept if r['d'] == time]})
        for level in percentiles:
            reached = [r for r in ecdf if r['context_id'] == context and r['F_right'] >= level - 1e-12]
            quantiles.append({'split': 'train', 'board_id': boards[context], 'context_id': context,
                              'percentile': level, 'quantile_d': reached[0]['d'] if reached else None,
                              'rule': 'min d with F(d) >= percentile'})
    return {'calibration_targets': target_rows, 'calibration_trials': trial_rows,
            'reachability_by_offset': reachability, 'figure1_targets': figure_targets,
            'figure1_ecdf': ecdf, 'figure1_quantiles': quantiles}
