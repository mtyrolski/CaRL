"""Compare the same pool's measured scales across action-budget prefixes."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.assess_overnight_cpu_stage02 import cluster_mean
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl

PAIRS = ((8, 16), (8, 32), (16, 32))


def compare(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare only identical targets in complete profiles, keeping failures."""
    grouped: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for profile in profiles:
        if profile['pool'] != 'fixed':
            continue
        key = (profile['context_id'], profile['controller'], profile['pool_variant'])
        limit = profile['L']
        if limit in grouped[key] or limit not in (8, 16, 32):
            raise ValueError('Duplicate or unsupported action-budget profile')
        grouped[key][limit] = profile
    contexts, targets = [], []
    for by_limit in grouped.values():
        for short, long in PAIRS:
            if short not in by_limit or long not in by_limit:
                raise ValueError('Missing a planned action-budget profile')
            one, two = by_limit[short], by_limit[long]
            keys = ('context_id', 'layout_id', 'family', 'source_family', 'split', 'controller', 'pool_variant')
            if any(one[key] != two[key] for key in keys):
                raise ValueError('Action-budget profiles identify different contexts')
            meta = {key: one[key] for key in keys}
            if one['status'] != 'complete' or two['status'] != 'complete':
                contexts.append({**meta, 'short_L': short, 'long_L': long, 'status': 'unavailable',
                                 'shared_success_targets': None, 'new_success_targets': None,
                                 'mean_abs_delta_u': None, 'max_abs_delta_u': None})
                continue
            left = {row['candidate_id']: row for row in one['candidates']}
            right = {row['candidate_id']: row for row in two['candidates']}
            if len(left) != len(one['candidates']) or set(left) != set(right):
                raise ValueError('Budget comparison requires the same distinct pool targets')
            differences = []
            newly_reached = 0
            for ident, a in left.items():
                b = right[ident]
                if a['p'] not in (0, 1) or b['p'] not in (0, 1) or a['p'] > b['p']:
                    raise ValueError('A shorter deterministic prefix cannot succeed when the longer one fails')
                if a['p'] == 0 and b['p'] == 1:
                    newly_reached += 1
                if a['p'] == b['p'] == 1:
                    if a['d'] != b['d'] or a['u'] is None or b['u'] is None:
                        raise ValueError('A shared hit must retain its duration and have both scale labels')
                    change = b['u'] - a['u']
                    differences.append(abs(change))
                    targets.append({**meta, 'short_L': short, 'long_L': long, 'candidate_id': ident,
                                    'd': a['d'], 'u_short': a['u'], 'u_long': b['u'], 'delta_u': change})
            contexts.append({**meta, 'short_L': short, 'long_L': long, 'status': 'complete',
                             'shared_success_targets': len(differences), 'new_success_targets': newly_reached,
                             'mean_abs_delta_u': sum(differences) / len(differences) if differences else None,
                             'max_abs_delta_u': max(differences) if differences else None})
    return contexts, targets


def summarise(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in contexts:
        base = (row['family'], row['controller'], row['pool_variant'], row['short_L'], row['long_L'])
        grouped[(base[0], 'all', *base[1:])].append(row)
        grouped[(base[0], row['source_family'], *base[1:])].append(row)
    output = []
    for (family, source, controller, variant, short, long), rows in sorted(grouped.items()):
        completed = [row for row in rows if row['status'] == 'complete']
        defined = [row for row in completed if row['mean_abs_delta_u'] is not None]
        delta = cluster_mean([(row['layout_id'], row['mean_abs_delta_u']) for row in defined])
        gained = cluster_mean([(row['layout_id'], float(row['new_success_targets'] > 0)) for row in completed])
        output.append({'family': family, 'source_family': source, 'controller': controller,
                       'pool_variant': variant, 'short_L': short, 'long_L': long,
                       'planned_contexts': len(rows), 'complete_contexts': len(completed),
                       'contexts_with_shared_success': len(defined),
                       'shared_success_targets': sum(row['shared_success_targets'] for row in completed),
                       'new_success_targets': sum(row['new_success_targets'] for row in completed),
                       'contexts_with_new_success': sum(row['new_success_targets'] > 0 for row in completed),
                       'mean_abs_delta_u': delta['mean'], 'ci_low': delta['ci_low'], 'ci_high': delta['ci_high'],
                       'delta_layouts': delta['n_layouts'], 'new_success_fraction': gained['mean'],
                       'new_success_ci_low': gained['ci_low'], 'new_success_ci_high': gained['ci_high']})
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profiles = args.analysis / 'profiles.jsonl'
    manifest = json.loads((args.analysis / 'analysis_manifest.json').read_text())
    if manifest.get('output_sha256', {}).get('profiles.jsonl') != file_sha(profiles):
        raise ValueError('Profiles differ from their completed analysis manifest')
    contexts, targets = compare(read_jsonl(profiles))
    summary = summarise(contexts)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (('budget_sensitivity_contexts.csv', contexts), ('budget_sensitivity_targets.csv', targets),
                       ('budget_sensitivity_summary.csv', summary)):
        write_csv(args.output / name, rows)
    atomic_json(args.output / 'budget_sensitivity_manifest.json', {'profiles_sha256': file_sha(profiles),
                'source_sha256': file_sha(Path(__file__)), 'contexts': len(contexts), 'targets': len(targets),
                'summary_rows': len(summary), 'bootstrap_unit': 'canonical layout; equal context weight'})
    print(json.dumps({'contexts': len(contexts), 'targets': len(targets), 'summary_rows': len(summary)}))


if __name__ == '__main__':
    main()
