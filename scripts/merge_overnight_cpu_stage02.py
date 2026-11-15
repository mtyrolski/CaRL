"""Add checked generator proposals to frozen demonstration/BFS pools.

Every structurally valid distinct proposal is retained for controller trials.
The optional nine-target pool uses four demonstration targets, the first four
selected BFS slots, and the first distinct generated target in decoding order.
No reachability or controller outcome is used to select a proposal.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import validate_subgoal
from scripts.prepare_overnight_cpu_stage02 import BFS_SLOTS
from scripts.prepare_overnight_cpu_stage02 import CONTEXT_FIELDS
from scripts.prepare_overnight_cpu_stage02 import DEMO_SLOTS
from scripts.prepare_overnight_cpu_stage02 import PROVENANCE_FIELDS
from scripts.prepare_overnight_cpu_stage02 import atomic_text
from scripts.prepare_overnight_cpu_stage02 import candidate_id
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl


def combine_context(base: list[dict[str, Any]], raw: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not base:
        raise ValueError('Context has no demonstration/BFS records')
    rows = [dict(item) for item in base]
    context = rows[0]
    for row in rows:
        if (row['context_id'] != context['context_id'] or row['source_hash'] != context['source_hash']
                or row['source'] != context['source']):
            raise ValueError('Candidates do not share a current state')
        row['in_generator_pool'] = False
        row['also_generated'] = False
        row['generator_ranks'] = []
    source = from_rows(context['source'])
    if state_hash(source) != context['source_hash']:
        raise ValueError('Incorrect current-state hash')
    by_hash = {row['target_hash']: row for row in rows}
    if len(by_hash) != len(rows):
        raise ValueError('Base candidate targets were not deduplicated')
    outcomes = []
    fresh = []
    if raw is not None:
        if (raw.get('context_id') != context['context_id'] or raw.get('source_hash') != context['source_hash']
                or len(raw.get('outputs', [])) != 2):
            raise ValueError('Generator shard identity or proposal count differs')
        for proposal in sorted(raw['outputs'], key=lambda item: item['rank']):
            result = {'rank': proposal['rank'], 'target_hash': proposal.get('target_hash'),
                      'validation_error': proposal.get('validation_error'), 'candidate_id': None}
            if not proposal.get('valid_nontrivial'):
                result['status'] = 'invalid'
                outcomes.append(result)
                continue
            target = from_rows(proposal['target'])
            validate_subgoal(source, target)
            target_hash = state_hash(target)
            if target_hash != proposal.get('target_hash') or target_hash == context['source_hash']:
                raise ValueError('Generator output validation or target hash differs')
            if target_hash in by_hash:
                by_hash[target_hash]['also_generated'] = True
                by_hash[target_hash]['generator_ranks'].append(proposal['rank'])
                result.update(status='duplicate', candidate_id=by_hash[target_hash]['candidate_id'])
                outcomes.append(result)
                continue
            item = {**{key: context[key] for key in CONTEXT_FIELDS},
                    **{key: context.get(key) for key in PROVENANCE_FIELDS},
                    'candidate_id': candidate_id(context['context_id'], target_hash),
                    'target': proposal['target'], 'target_hash': target_hash, 'candidate_origin': 'generator',
                    'demo_offsets': [], 'demo_pair_ids': [], 'bfs_depth': None, 'bfs_pushes': None,
                    'bfs_actions': None, 'in_fixed_pool': False, 'in_generator_pool': False,
                    'pool_variant': context.get('pool_variant'), 'slot': None, 'generator_rank': proposal['rank'],
                    'also_generated': True, 'generator_ranks': [proposal['rank']]}
            rows.append(item)
            by_hash[target_hash] = item
            fresh.append(item)
            result.update(status='valid_unique', candidate_id=item['candidate_id'])
            outcomes.append(result)
    demo = sorted((row for row in rows if row['in_fixed_pool'] and row['candidate_origin'] == 'demonstration'),
                  key=lambda row: DEMO_SLOTS.index(row['slot']))
    bfs = sorted((row for row in rows if row['in_fixed_pool'] and row['candidate_origin'] == 'bfs'),
                 key=lambda row: BFS_SLOTS.index(row['slot']))
    if len(demo) != 4 or len(bfs) < 4:
        pool_status = 'underfilled_base_pool'
    elif not fresh:
        pool_status = 'underfilled_generator'
    else:
        pool_status = 'complete_demo4_bfs4_generator1'
        for row in [*demo, *bfs[:4], fresh[0]]:
            row['in_generator_pool'] = True
    coverage = {key: context[key] for key in ('context_id', 'layout_id', 'family', 'source_family',
                                              'split', 'board_rank')}
    coverage.update({'pool_status': pool_status, 'generator_shard_missing': raw is None,
                     'raw_proposals': outcomes, 'valid_unique_count': len(fresh),
                     'selected_candidate_id': fresh[0]['candidate_id'] if pool_status.startswith('complete') else None})
    return rows, coverage


def run(base_data: Path, generator_dir: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    paths = ('candidates.jsonl', 'generator_pool_coverage.jsonl', 'generator_merge_manifest.json')
    if any((output / name).exists() for name in paths):
        raise FileExistsError('Use a fresh merge directory to preserve previous output')
    candidates = read_jsonl(base_data / 'candidates.jsonl')
    plan = json.loads((generator_dir / 'plan.json').read_text())
    contract = json.loads((generator_dir / 'contract.json').read_text())
    if plan['contract_sha256'] != digest(contract):
        raise ValueError('Generator plan/contract fingerprint differs')
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[row['context_id']].append(row)
    expected = {row['context_id'] for row in plan['contexts']}
    if set(grouped) != expected:
        raise ValueError('Generator plan and BFS context lists differ')
    results, coverage = [], []
    for context_id in sorted(grouped):
        path = generator_dir / 'shards' / f'{context_id}.json'
        raw = json.loads(path.read_text()) if path.is_file() else None
        if raw is not None and raw.get('contract_sha256') != digest(contract):
            raise ValueError('Generator shard belongs to a different contract')
        rows, one = combine_context(grouped[context_id], raw)
        results.extend(rows)
        coverage.append(one)
    if len({row['candidate_id'] for row in results}) != len(results):
        raise ValueError('Duplicate candidate identity after merge')
    def jsonl(items: list[dict[str, Any]]) -> str:
        return ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in items)
    atomic_text(output / 'candidates.jsonl', jsonl(results))
    atomic_text(output / 'generator_pool_coverage.jsonl', jsonl(coverage))
    manifest = {'schema_version': 1, 'base_candidates_sha256': file_sha(base_data / 'candidates.jsonl'),
                'generator_plan_sha256': file_sha(generator_dir / 'plan.json'),
                'generator_contract_sha256': file_sha(generator_dir / 'contract.json'),
                'merge_source_sha256': file_sha(Path(__file__)), 'contexts': len(coverage),
                'pool_rule': '4 selected demo + first 4 selected BFS + first valid unique generated state; '
                             'no reachability filter; incomplete pools are reported separately',
                'pool_status_counts': dict(Counter(item['pool_status'] for item in coverage)),
                'generator_output_status_counts': dict(Counter(result['status'] for item in coverage
                                                               for result in item['raw_proposals'])),
                'candidate_counts': dict(Counter(item['candidate_origin'] for item in results)),
                'candidates_sha256': file_sha(output / 'candidates.jsonl'),
                'coverage_sha256': file_sha(output / 'generator_pool_coverage.jsonl')}
    atomic_json(output / 'generator_merge_manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-data', type=Path, required=True)
    parser.add_argument('--generator-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.base_data, args.generator_dir, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
