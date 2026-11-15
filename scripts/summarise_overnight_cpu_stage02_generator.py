"""Join every raw generator proposal to its single measured controller target.

Structurally invalid outputs remain separate from controller failures. Duplicate
outputs link to the same attempted target; bootstrap units remain layouts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.assess_overnight_cpu_stage02 import CONTROLLERS
from scripts.assess_overnight_cpu_stage02 import LIMITS
from scripts.assess_overnight_cpu_stage02 import cluster_mean
from scripts.assess_overnight_cpu_stage02 import prefix
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl


def proposal_outcomes(coverage: list[dict[str, Any]], plan: dict[str, Any],
                      attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scheduled = {(row['controller'], row['candidate_id']) for row in plan['planned_attempts']}
    observed = {(row['controller'], row['candidate_id']): row for row in attempts}
    if len(observed) != len(attempts) or not set(observed) <= scheduled:
        raise ValueError('Duplicate or unplanned saved controller attempts')
    rows = []
    for context in coverage:
        raw = context['raw_proposals']
        if context['generator_shard_missing']:
            if raw:
                raise ValueError('A missing generator shard cannot contain output')
            raw = [{'rank': rank, 'status': 'generator_missing', 'candidate_id': None,
                    'target_hash': None, 'validation_error': None} for rank in range(2)]
        if sorted(item['rank'] for item in raw) != [0, 1]:
            raise ValueError('Each context needs two declared proposal slots')
        for output in raw:
            is_valid = output['status'] in {'valid_unique', 'duplicate'}
            ident = output['candidate_id']
            if is_valid and (not ident or any((model, ident) not in scheduled for model in CONTROLLERS)):
                raise ValueError('A valid generated target was omitted from the frozen controller plan')
            if not is_valid and ident:
                raise ValueError('An invalid or missing output cannot name a controller target')
            for model in CONTROLLERS:
                saved = observed.get((model, ident)) if is_valid else None
                for limit in LIMITS:
                    outcome = prefix(saved, limit) if is_valid else {'status': 'not_applicable', 'p': None, 'd': None}
                    rows.append({**{key: context[key] for key in ('context_id', 'layout_id', 'family',
                                                                  'source_family', 'split', 'board_rank')},
                                 'rank': output['rank'], 'proposal_status': output['status'],
                                 'validation_error': output.get('validation_error'), 'candidate_id': ident,
                                 'target_hash': output.get('target_hash'), 'controller': model, 'L': limit,
                                 'controller_status': outcome['status'], 'p': outcome['p'], 'd': outcome['d']})
    return rows


def summarise(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row['family'], 'all', row['controller'], row['L'])].append(row)
        groups[(row['family'], row['source_family'], row['controller'], row['L'])].append(row)
    summaries = []
    for (family, source, model, limit), items in sorted(groups.items()):
        contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            contexts[item['context_id']].append(item)
        metrics = []
        for pair in contexts.values():
            valid = [item for item in pair if item['proposal_status'] in {'valid_unique', 'duplicate'}]
            if (len(pair) == 2 and all(item['proposal_status'] != 'generator_missing' for item in pair)
                    and all(item['controller_status'] == 'complete' for item in valid)):
                metrics.append((pair[0]['layout_id'], sum(item['p'] for item in valid) / 2))
        interval = cluster_mean(metrics)
        valid = [item for item in items if item['proposal_status'] in {'valid_unique', 'duplicate'}]
        measured = [item for item in valid if item['controller_status'] == 'complete']
        counts = Counter(item['proposal_status'] for item in items)
        summaries.append({'family': family, 'source_family': source, 'controller': model, 'L': limit,
                          'planned_contexts': len(contexts), 'planned_outputs': len(items),
                          'valid_outputs': len(valid), 'invalid_outputs': counts['invalid'],
                          'generator_missing_outputs': counts['generator_missing'],
                          'duplicate_outputs': counts['duplicate'], 'valid_measured_outputs': len(measured),
                          'valid_missing_controller': len(valid) - len(measured),
                          'successful_outputs': sum(item['p'] for item in measured),
                          'complete_contexts': len(metrics), 'success_per_requested_equal_context': interval['mean'],
                          'ci_low': interval['ci_low'], 'ci_high': interval['ci_high'],
                          'bootstrap_layouts': interval['n_layouts']})
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError('Refusing to write an empty proposal table')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', newline='', encoding='utf-8') as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coverage', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan_path, contract_path, rollout_path = (args.run / name for name in ('plan.json', 'contract.json', 'rollouts.jsonl'))
    merge_path = args.coverage.parent / 'generator_merge_manifest.json'
    plan = json.loads(plan_path.read_text())
    contract = json.loads(contract_path.read_text())
    merge = json.loads(merge_path.read_text())
    if contract.get('plan_sha256') != digest(plan):
        raise ValueError('Controller plan differs from its frozen contract')
    if (merge.get('coverage_sha256') != file_sha(args.coverage)
            or merge.get('candidates_sha256') != contract.get('candidates_sha256')):
        raise ValueError('Generator coverage and frozen controller candidates differ')
    attempts = read_jsonl(rollout_path)
    if any(row.get('contract_sha256') != digest(contract) for row in attempts):
        raise ValueError('Saved attempt belongs to a different controller contract')
    rows = proposal_outcomes(read_jsonl(args.coverage), plan, attempts)
    summary = summarise(rows)
    write_csv(args.output / 'generator_proposal_outcomes.csv', rows)
    write_csv(args.output / 'generator_proposal_summary.csv', summary)
    atomic_json(args.output / 'generator_proposal_summary.json', {'rows': summary,
                'meaning': 'Raw output counts include duplicates; controller attempts are reused by candidate ID. '
                           'Confidence intervals resample original layouts, not outputs.'})
    atomic_json(args.output / 'generator_join_manifest.json', {
        'coverage_path': str(args.coverage.resolve()), 'coverage_sha256': file_sha(args.coverage),
        'generator_merge_manifest_sha256': file_sha(merge_path),
        'plan_sha256': file_sha(plan_path), 'contract_sha256': file_sha(contract_path),
        'rollouts_sha256': file_sha(rollout_path), 'source_sha256': file_sha(Path(__file__)),
        'outputs_sha256': {name: file_sha(args.output / name) for name in (
            'generator_proposal_outcomes.csv', 'generator_proposal_summary.csv', 'generator_proposal_summary.json')},
    })
    print(json.dumps({'proposal_rows': len(rows), 'summary_rows': len(summary)}))


if __name__ == '__main__':
    main()
