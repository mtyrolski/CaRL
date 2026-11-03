"""Summarize a completed or partial calibration and its shared ledger offline."""
import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

from carl.icl.api import write_json
from carl.icl.config import ROOT, load_config
from carl.icl.experiment import calibration_components, calibration_path

STAGE = 'calibration'
TAG = 'calibration_v2'
DEFAULT_CONFIG = 'configs/icl/campaign_data/calibration_live_v2.json'
DEFAULT_OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/calibration_v2'
TARGET_FIELDS = ('context', 'target_index', 'target_hash', 'k', 'trials', 'hits', 'p_hat', 'd', 'weight',
                 'cdf_before', 'cdf_at', 'u', 'a_c', 'ratio_d_over_ac', 'normalized_offset', 'excluded_reason',
                 'target_sampling_entropy')
TRIAL_FIELDS = ('context', 'target_index', 'k', 'trial', 'seed_entropy', 'rng', 'actions', 'attempted_actions',
                'exact_hit', 'first_hit', 'task_solved', 'solved_before_target', 'endpoint', 'local_policy_cache_hits')


def _json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger_summary(path: Path, *, stage: str = STAGE, tag: str = TAG) -> dict[str, Any]:
    """Read a consistent SQLite snapshot without creating or modifying the ledger."""
    if not path.is_file():
        raise FileNotFoundError(f'Campaign ledger not found: {path}')
    uri = path.resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute('BEGIN')
        attempts = db.execute(
            'SELECT status, charged_nanos, reserved_nanos, usage_json FROM attempts WHERE stage=? AND tag=?',
            (stage, tag),
        ).fetchall()
        events = db.execute(
            'SELECT kind, COUNT(*) FROM events WHERE stage=? AND tag=? GROUP BY kind', (stage, tag),
        ).fetchall()
    statuses = {name: {'attempts': 0, 'committed_usd': 0.0} for name in ('settled', 'unknown', 'reserved')}
    nanos: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    usage_missing = missing_details = 0
    for status, charged, reserved, raw in attempts:
        if status not in statuses:
            raise ValueError(f'Unrecognized ledger status: {status}')
        if status == 'settled' and charged is None:
            raise ValueError('Settled attempt lacks its recorded cost')
        statuses[status]['attempts'] += 1
        nanos[status] += charged if charged is not None else reserved
        if not raw:
            usage_missing += 1
            continue
        usage = json.loads(raw)
        tokens['input'] += int(usage.get('input_tokens') or 0)
        tokens['output'] += int(usage.get('output_tokens') or 0)
        details = usage.get('input_tokens_details')
        if not isinstance(details, dict) or details.get('cached_tokens') is None or details.get('cache_write_tokens') is None:
            missing_details += 1
        if isinstance(details, dict):
            tokens['cache_read'] += int(details.get('cached_tokens') or 0)
            tokens['cache_write'] += int(details.get('cache_write_tokens') or 0)
    for status, amount in nanos.items():
        statuses[status]['committed_usd'] = amount / 1_000_000_000
    event_counts = dict(events)
    for kind in ('logical_request', 'live_cache_hit', 'local_replay_hit', 'replay_miss', 'provider_error'):
        event_counts.setdefault(kind, 0)
    return {'stage': stage, 'tag': tag, 'attempts': statuses, 'total_attempts': len(attempts),
            'events': event_counts, 'tokens': {name: tokens[name] for name in ('input', 'output', 'cache_read', 'cache_write')},
            'attempts_without_usage': usage_missing, 'usage_records_without_complete_cache_details': missing_details,
            'note': 'Conservative configured prices and regional multiplier, not an invoice; unknown/reserved remain committed.'}


def policy_support(ledger_path: Path, cache_dir: Path) -> dict[str, Any]:
    """Count support sizes of distinct cached policy distributions used by calibration."""
    uri = ledger_path.resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        hashes = [row[0] for row in db.execute(
            "SELECT DISTINCT request_hash FROM events WHERE stage=? AND tag=? AND kind='logical_request'",
            (STAGE, TAG),
        )]
    counts: Counter[str] = Counter()
    for digest in hashes:
        path = cache_dir / 'openai' / f'{digest}.json'
        if not path.is_file():
            counts['missing'] += 1
            continue
        try:
            record = _json_file(path)
            probabilities = json.loads(record['output_text'])['probabilities']
            if (record.get('request_hash') != digest or not isinstance(probabilities, list) or len(probabilities) != 4
                    or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 or p > 1
                           for p in probabilities) or sum(probabilities) <= 0):
                raise ValueError('Invalid probabilities')
            counts['point_mass' if sum(p > 0 for p in probabilities) == 1 else 'multiple_actions'] += 1
        except (OSError, KeyError, TypeError, ValueError):
            counts['invalid'] += 1
    return {'unique_logical_policy_inputs': len(hashes), 'point_mass': counts['point_mass'],
            'multiple_actions': counts['multiple_actions'], 'invalid': counts['invalid'], 'missing': counts['missing'],
            'note': 'Support of final cached distributions per distinct full input; not independent LLM samples.'}


def summarize(summary: dict[str, Any], targets: list[dict[str, Any]], trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Report completed, failed, excluded and not-yet-started cases separately."""
    settings = summary['config']
    if summary['status'] == 'completed' and int(summary['trials']) != len(trials):
        raise ValueError('Completed calibration summary and trial artifact disagree')
    repeats = int(settings['trials'])
    context_count = int(summary.get('context_count', len(summary.get('contexts', []))))
    target_count = settings.get('sampled_targets')
    planned = context_count * int(target_count) if target_count is not None else None
    completed = [row for row in targets if row.get('trials') == repeats and row.get('excluded_reason') != 'incomplete']
    by_k: dict[str, dict[str, int]] = {}
    for k in sorted({int(row['k']) for row in targets}):
        group = [row for row in targets if int(row['k']) == k]
        by_k[str(k)] = {'targets': len(group), 'completed': sum(row in completed for row in group),
                        'hit_targets': sum((row.get('hits') or 0) > 0 for row in group),
                        'zero_hit_targets': sum(row.get('excluded_reason') == 'zero_exact_hits' for row in group),
                        'trials': sum(int(row.get('trials') or 0) for row in group),
                        'exact_hits': sum(int(row.get('hits') or 0) for row in group)}
    p_hat = Counter(str(row['p_hat']) for row in completed if row.get('p_hat') is not None)
    contexts = summary.get('contexts', [])
    sequences: dict[tuple[str, int], list[tuple[int, ...]]] = defaultdict(list)
    for trial in trials:
        if all(key in trial for key in ('context', 'target_index', 'actions')):
            sequences[(trial['context'], trial['target_index'])].append(tuple(trial['actions']))
    complete_sequences = [items for items in sequences.values() if len(items) == repeats]
    identical_sequences = sum(len(set(items)) == 1 for items in complete_sequences)
    return {'status': summary['status'], 'signature': summary['signature'], 'provider': summary['provider'],
            'policy_fingerprint': summary.get('policy_fingerprint'),
            'parameters': {'reference_window': settings['reference_window'], 'L': settings['action_cap'],
                           'R': repeats, 'sampled_targets_per_context': target_count,
                           'contexts_configured': context_count, 'seed': summary.get('target_sampling_seed')},
            'coverage': {'planned_max_targets': planned, 'targets_recorded': len(targets),
                         'targets_completed': len(completed), 'hit_targets': sum((row.get('hits') or 0) > 0 for row in completed),
                         'targets_incomplete': len(targets) - len(completed),
                         'targets_not_recorded_vs_planned_max': max(0, planned - len(targets)) if planned is not None else None,
                         'completed_trials': len(trials), 'planned_max_trials': planned * repeats if planned is not None else None,
                         'contexts_completed': int(summary.get('contexts_completed', len(contexts))),
                         'contexts_with_hit': sum(not row['excluded'] for row in contexts),
                         'contexts_without_hit': sum(row['excluded'] for row in contexts),
                         'zero_hit_targets': sum(row.get('excluded_reason') == 'zero_exact_hits' for row in targets),
                         'single_time_contexts': sum(row['distinct_d'] == 1 for row in contexts),
                         'multi_time_contexts': sum(row['distinct_d'] >= 2 for row in contexts),
                         'exact_hits': sum(bool(row.get('exact_hit')) for row in trials),
                         'task_solved': sum(bool(row.get('task_solved')) for row in trials),
                         'solved_before_target': sum(bool(row.get('solved_before_target')) for row in trials),
                         'attempted_actions': sum(int(row.get('attempted_actions', 0)) for row in trials),
                         'total_executed_actions_including_interrupted': summary.get('total_executed_actions'),
                         'interrupted_attempts': sum(len(row.get('interrupted', [])) for row in targets),
                         'targets_with_identical_action_sequences': identical_sequences,
                         'targets_with_complete_action_sequences': len(complete_sequences)},
            'offset_distribution': by_k, 'p_hat_distribution': dict(sorted(p_hat.items(), key=lambda item: float(item[0]))),
            'contexts': contexts, 'selected_example_ids': summary.get('selected_example_ids', []),
            'randomness_note': summary.get('randomness_note', ''),
            'coverage_note': 'Planned targets/trials are maxima; fewer are expected when fewer distinct future states exist.'}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _markdown(report: dict[str, Any]) -> str:
    p, c, cost = report['parameters'], report['coverage'], report['cost']
    lines = ['# Kalibracja policy-relative', '', f"Status: **{report['status']}**, provider: `{report['provider']}`.",
             f"D_ref={p['reference_window']}, L={p['L']}, R={p['R']}, "
             f"cele/kontekst≤{p['sampled_targets_per_context']}, konteksty={p['contexts_configured']}, seed={p['seed']}.", '',
             'Punktacja jest warunkowa na rzeczywiście uzyskanych i cache’owanych rozkładach policy. '
             'Niezależne seedy ponownie losują akcje; odtworzenie całego rollouta z cache nie jest kolejną próbą.', '',
             f"Cele: {c['targets_completed']}/{c['planned_max_targets']} ukończone, "
             f"{c['zero_hit_targets']} bez trafień, {c['targets_incomplete']} nieukończonych; "
             f"próby: {c['completed_trials']}/{c['planned_max_trials']}.",
             f"Konteksty: {c['contexts_completed']}/{p['contexts_configured']} ukończonych, "
             f"{c['contexts_without_hit']} bez hitów, {c['single_time_contexts']} z jednym czasem, "
             f"{c['multi_time_contexts']} z ≥2 czasami.",
             f"Exact hits: {c['exact_hits']}; rozwiązania zadania: {c['task_solved']}, "
             f"w tym przed innym targetem: {c['solved_before_target']}; akcje ukończonych prób: {c['attempted_actions']}; "
             f"wszystkie wykonane akcje, także przerwane próby: {c['total_executed_actions_including_interrupted']}; "
             f"przerwane próby: {c['interrupted_attempts']}.",
             f"Cele z identyczną sekwencją akcji we wszystkich R próbach: "
             f"{c['targets_with_identical_action_sequences']}/{c['targets_with_complete_action_sequences']}.", '',
             '| k | Cele | Ukończone | Z hitem | Zero hit | Próby | Exact hits |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for k, row in report['offset_distribution'].items():
        lines.append(f"| {k} | {row['targets']} | {row['completed']} | {row['hit_targets']} | "
                     f"{row['zero_hit_targets']} | {row['trials']} | {row['exact_hits']} |")
    lines += ['', f"Rozkład p_hat ukończonych celów: `{report['p_hat_distribution']}`."]
    if 'policy_support' in report:
        support = report['policy_support']
        lines.append(f"Rozkłady policy dla różnych wejść: {support['point_mass']} punktowych i "
                     f"{support['multiple_actions']} z więcej niż jedną akcją "
                     f"({support['missing']} brakujących, {support['invalid']} niepoprawnych w aktywnym cache).")
    lines += ['', '## Koszt i cache', '']
    if 'attempts' in cost:
        lines.append(f"Ledger: stage=`{cost['stage']}`, tag=`{cost['tag']}`; {cost['total_attempts']} fizycznych prób.")
        for status, row in cost['attempts'].items():
            lines.append(f"- {status}: {row['attempts']} prób; zajęte ${row['committed_usd']:.9f}.")
        lines += [f"- Zdarzenia: `{cost['events']}`.", f"- Tokeny z dostępnego usage: `{cost['tokens']}`.",
                  f"- Brak usage: {cost['attempts_without_usage']} prób; brak pełnego podziału cache: "
                  f"{cost['usage_records_without_complete_cache_details']} zapisów.",
                  '- Kwoty to konserwatywne oszacowanie według skonfigurowanych stawek, nie faktura.']
    else:
        lines.append('- Mock: brak płatnych żądań; nie używać do wniosków o Lunie.')
    lines += ['', 'Pełne dane: `report.json`, `calibration_targets.csv`, `calibration_trials.csv`. '
              'Gdy brakuje dostatecznej liczby różnych przyszłych stanów, planowana liczba celów jest górnym limitem.']
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--provider', choices=('mock', 'openai'), default='openai')
    parser.add_argument('--input-path', type=Path, help='Existing calibration directory or calibration.json')
    parser.add_argument('--output-dir', type=Path, help='Destination for reports, separate from calibration data')
    parser.add_argument('--ledger-path', type=Path, help='Existing SQLite ledger (read-only)')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.input_path is None:
        components = calibration_components(config, 'mock' if args.provider == 'mock' else 'replay')
        directory = calibration_path(config, components).parent
    else:
        directory = args.input_path.expanduser()
        if directory.name == 'calibration.json':
            directory = directory.parent
    files = [directory / name for name in ('calibration.json', 'calibration_targets.json', 'calibration_trials.json')]
    summary, targets, trials = (_json_file(path) for path in files)
    if summary['provider'] != args.provider:
        raise ValueError('Requested provider differs from calibration artifact provider')
    report = summarize(summary, targets, trials)
    if args.provider == 'openai':
        budget = _json_file(ROOT / 'configs/icl/campaign_budget.json')
        ledger = args.ledger_path or Path(budget['ledger_path'])
        ledger = ledger if ledger.is_absolute() else ROOT / ledger
        report['cost'] = ledger_summary(ledger)
        report['policy_support'] = policy_support(ledger, config.path('cache_dir'))
    else:
        report['cost'] = {'provider': 'mock', 'note': 'No paid API calls.'}
    report['input_sha256'] = {str(path): _sha(path) for path in files}
    output = args.output_dir or (DEFAULT_OUTPUT if args.provider == 'openai' else DEFAULT_OUTPUT.with_name('calibration_v2_mock'))
    if output.resolve() == directory.resolve() or output.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Report output must be separate from calibration data')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'report.json', report)
    (output / 'report.md').write_text(_markdown(report), encoding='utf-8')
    _write_csv(output / 'calibration_targets.csv', targets, TARGET_FIELDS)
    _write_csv(output / 'calibration_trials.csv', trials, TRIAL_FIELDS)
    print(f'Report: {output / "report.md"}')


if __name__ == '__main__':
    main()
