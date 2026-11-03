"""Read the frozen pilot matrix, including missing and censored cells."""
import hashlib
import json
import sqlite3
from collections import defaultdict
from typing import Any

import numpy as np

from scripts.plot_data_export.common import CAMPAIGN, FREEZE, PILOT, read_json, sha

METHODS = ('relative_u', 'absolute_d')


def cost_by_tag() -> dict[str, dict[str, Any]]:
    path = CAMPAIGN / 'cost_ledger.sqlite3'
    totals: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('BEGIN')
        rows = db.execute("SELECT tag,status,COUNT(*),SUM(charged_nanos),SUM(reserved_nanos) "
                          "FROM attempts WHERE tag LIKE 'small_pilot:%' GROUP BY tag,status").fetchall()
    for tag, status, count, charged, reserved in rows:
        row = totals.setdefault(tag, {'cost_settled_usd': 0.0, 'cost_unknown_reserved_usd': 0.0,
                                      'physical_attempts_ledger': 0})
        row['physical_attempts_ledger'] += count
        row['cost_settled_usd'] += (charged or 0) / 1e9
        if status != 'settled':
            row['cost_unknown_reserved_usd'] += (reserved or 0) / 1e9
    return totals


def trial_key(freeze_hash: str, board: str, method: str, seed: int, budget: int) -> str:
    value = [freeze_hash, board, method, seed, budget]
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def load_pilot() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    freeze = read_json(FREEZE)
    freeze_hash = sha(FREEZE)
    manifest = read_json(PILOT / 'pilot_manifest.json')
    if manifest['freeze_hash'] != freeze_hash:
        raise ValueError('Pilot manifest and frozen file do not match')
    tags = cost_by_tag()
    boards = [{'campaign_id': 'campaign_100_20261102', 'split': freeze['split'],
               'family': freeze['family'], 'board_id': item['id'], 'layout_id': item['layout_id'],
               'state_hash': item['state_hash']} for item in freeze['boards']]
    if len(boards) != 100 or len({row['layout_id'] for row in boards}) != len(boards):
        raise ValueError('Frozen pilot must contain 100 distinct layouts')
    rows = []
    for budget in freeze['budgets']:
        for board in boards:
            for method in METHODS:
                key = trial_key(freeze_hash, board['board_id'], method, freeze['seed'], budget)
                path = PILOT / 'trials' / f'{key}.json'
                started = PILOT / 'trials' / f'{key}.started.json'
                record = read_json(path) if path.is_file() else None
                job = record['job'] if record else None
                if job and (job['key'] != key or job['board_id'] != board['board_id']
                            or job['method'] != method or job['budget'] != budget):
                    raise ValueError(f'Mismatched pilot checkpoint: {path}')
                outcome = record.get('outcome', {}) if record else {}
                api = record.get('api', {}) if record else {}
                verification = record.get('verification', {}) if record else {}
                status = record['status'] if record else 'interrupted' if started.exists() else 'missing'
                complete = status == 'complete'
                replay = verification.get('verified') if complete else None
                solved = outcome.get('solved') if complete else None
                if solved is True and not verification.get('solution_replayed'):
                    raise ValueError(f'Unverified solved checkpoint: {path}')
                tag = f'small_pilot:{key}'
                cost = tags.get(tag, {})
                row = {**board, 'freeze_hash': freeze_hash, 'run_key': key, 'method': method,
                       'seed': freeze['seed'], 'budget_actions': budget, 'execution_status': status,
                       'solved': solved, 'replay_verified': replay,
                       'solution_replayed': verification.get('solution_replayed') if complete else None,
                       'completion_class': record.get('completion_class') if record else None,
                       'failure_category': record.get('failure_category') if record else None,
                       'finish_reason': outcome.get('finish_reason'), 'error': outcome.get('error'),
                       'unexpected_exception_type': record.get('unexpected_exception_type') if record else None,
                       'attempted_actions': outcome.get('attempted_actions'),
                       'solution_length': len(outcome.get('action_path', [])) if solved else None,
                       'expansions': outcome.get('expansions'), 'accepted': outcome.get('accepted'),
                       'rejected': outcome.get('rejected'), 'fallback_successors': outcome.get('primitive_successors'),
                       'wall_seconds': outcome.get('seconds'), 'logical_requests': api.get('logical_requests'),
                       'provider_requests': api.get('requests'), 'local_cache_hits': api.get('cache_hits'),
                       'input_tokens': api.get('input_tokens'), 'output_tokens': api.get('output_tokens'),
                       'cached_input_tokens': api.get('cached_input_tokens'), 'cache_write_tokens': api.get('cache_write_tokens'),
                       'logical_input_tokens': api.get('logical_input_tokens'),
                       'logical_output_tokens': api.get('logical_output_tokens'), **cost}
                if row['attempted_actions'] is not None and row['attempted_actions'] > budget:
                    raise ValueError(f'Action budget exceeded: {path}')
                rows.append(row)
    return boards, rows, freeze


def wilson(successes: int, count: int) -> tuple[float | None, float | None]:
    if not count:
        return None, None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return float(center - half), float(center + half)


def bootstrap(values: list[float], *, seed: int = 20261029) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(2000, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def usable(row: dict[str, Any]) -> bool:
    return (row['execution_status'] == 'complete' and row['replay_verified'] is True
            and row['solved'] is not None and row['completion_class'] != 'local_limit_before_B')


def aggregate(rows: list[dict[str, Any]], freeze: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve, paired = [], []
    for budget in freeze['budgets']:
        group = [r for r in rows if r['budget_actions'] == budget]
        for method in METHODS:
            method_rows = [r for r in group if r['method'] == method]
            by_board: dict[str, list[float]] = defaultdict(list)
            for row in method_rows:
                if usable(row):
                    by_board[row['board_id']].append(float(row['solved']))
            values = [float(np.mean(by_board[key])) for key in sorted(by_board)]
            count = len(values)
            successes = sum(values)
            low, high = bootstrap(values)
            one_seed = len({r['seed'] for r in method_rows}) == 1
            wilson_low, wilson_high = wilson(int(successes), count) if one_seed else (None, None)
            expected = len({r['board_id'] for r in method_rows})
            curve.append({'family': freeze['family'], 'method': method, 'budget_actions': budget,
                          'expected_boards': expected, 'observed_boards': count,
                          'observed_runs': sum(usable(r) for r in method_rows),
                          'verified_successes': successes, 'denominator_boards': count,
                          'solve_rate': successes / count if count else None,
                          'bootstrap_low': low, 'bootstrap_high': high, 'wilson_low': wilson_low,
                          'wilson_high': wilson_high,
                          'full_cohort_lower': successes / expected if one_seed else None,
                          'full_cohort_upper': (successes + expected - count) / expected if one_seed else None})
        maps = {}
        for method in METHODS:
            observations: dict[str, list[float]] = defaultdict(list)
            for row in group:
                if row['method'] == method and usable(row):
                    observations[row['board_id']].append(float(row['solved']))
            maps[method] = {key: float(np.mean(value)) for key, value in observations.items()}
        keys = sorted(set(maps[METHODS[0]]) & set(maps[METHODS[1]]))
        differences = [maps[METHODS[0]][key] - maps[METHODS[1]][key] for key in keys]
        low, high = bootstrap(differences)
        binary = all(maps[m][key] in (0, 1) for m in METHODS for key in keys)
        counts = joint_counts(maps, keys) if binary else {}
        paired.append({'family': freeze['family'], 'budget_actions': budget,
                       'primary_budget': budget == freeze['main_budget'], 'expected_pairs': len(freeze['boards']),
                       'complete_pairs': len(keys), 'missing_pairs': len(freeze['boards']) - len(keys),
                       'delta_relative_minus_absolute': float(np.mean(differences)) if differences else None,
                       'bootstrap_low': low, 'bootstrap_high': high,
                       'full_cohort_delta_lower': (sum(differences) - (len(freeze['boards']) - len(keys))) / len(freeze['boards']),
                       'full_cohort_delta_upper': (sum(differences) + (len(freeze['boards']) - len(keys))) / len(freeze['boards']),
                       **counts})
    return curve, paired


def joint_counts(maps: dict[str, dict[str, float]], keys: list[str]) -> dict[str, int]:
    """Count board-level binary joint outcomes for the available common seed."""
    pairs = [(int(maps[METHODS[0]][key]), int(maps[METHODS[1]][key])) for key in keys]
    return {'both_solved': pairs.count((1, 1)), 'only_relative': pairs.count((1, 0)),
            'only_absolute': pairs.count((0, 1)), 'neither': pairs.count((0, 0))}
