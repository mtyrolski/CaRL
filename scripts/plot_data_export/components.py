"""Read observed component trials, production traces and campaign cost ledger."""
import gzip
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

from scripts.plot_data_export.common import CAMPAIGN, PILOT, read_json


def components() -> dict[str, list[dict[str, Any]]]:
    report = read_json(CAMPAIGN / 'policy_validation/report.json')
    policy = []
    for row in report['cases']:
        steps = row.get('steps', [])
        policy.append({'split': 'component_validation', 'family': 'real_12x12x4',
                       'board_id': row['board_id'], 'layout_id': row['layout_id'],
                       'variant': row['variant'], 'position': row['position'], 'k': row['offset'],
                       'seed': row['sampling_seed'], 'cap': row['cap'], 'attempts': row.get('attempts'),
                       'exact_hit': row.get('exact_hit'), 'first_hit': row.get('first_hit'),
                       'task_solved_early': row.get('task_solved_early'), 'no_ops': row.get('no_ops'),
                       'repeated_states': row.get('repeated_states'), 'error': row.get('error'),
                       'source': row['source'], 'target': row['target'],
                       'endpoint': steps[-1]['successor'] if steps else row['source'],
                       'actions': [step['action'] for step in steps],
                       'probabilities': [step['probabilities'] for step in steps]})
    generated, value = [], []
    directory = CAMPAIGN / 'components_validation/openai'
    for path in sorted(directory.glob('generator_*.json')):
        row = read_json(path)
        generated.append({'split': 'component_validation', 'family': 'real_12x12x4',
                          'board_id': row['board_id'], 'variant': row['variant'], 'k': row['k'],
                          'parse_ok': row['parse_ok'], 'error': row['error'],
                          'candidate_count': len(row['candidates']),
                          'static_invalid': sum(not c['static_valid'] for c in row['candidates']),
                          'source_copies': sum(c['source_copy'] for c in row['candidates']),
                          'duplicates': sum(c['duplicate'] for c in row['candidates']),
                          'distinct_valid_count': row['distinct_valid_count'], 'reached_any': row['reached_any'],
                          'source': row['source'], 'candidates': row['candidates'],
                          'proposals': row['proposals']})
    for path in sorted(directory.glob('value_*.json')):
        row = read_json(path)
        value.append({key: row.get(key) for key in ('variant', 'board_id', 'family', 'position',
                     'remaining', 'gamma', 'reference_value', 'predicted_value', 'error')}
                     | {'split': 'component_validation'})
    return {'component_policy': policy, 'component_generator': generated, 'component_value': value}


def write_search_attempts(path: Path) -> dict[str, int]:
    """Write recorded transitions only; unfinished traces retain their status."""
    count = 0
    completed_actions = 0
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding='utf-8') as output:
            for trace in sorted((PILOT / 'trials').glob('*.trace.jsonl')):
                key = trace.name.removesuffix('.trace.jsonl')
                checkpoint = trace.with_name(f'{key}.json')
                started = trace.with_name(f'{key}.started.json')
                if checkpoint.exists():
                    record = read_json(checkpoint)
                elif started.exists():
                    record = read_json(started)
                else:
                    raise ValueError(f'Trace without checkpoint or started marker: {trace}')
                job = record['job']
                nodes: dict[int, dict[str, Any]] = {}
                action_events = 0
                transition_actions = 0
                with trace.open(encoding='utf-8') as source:
                    for line in source:
                        event = json.loads(line)
                        if event['event'] == 'action':
                            action_events += 1
                        if event['event'] not in {'start', 'transition'}:
                            continue
                        node = event['node']
                        nodes[node['node_id']] = node
                        if event['event'] != 'transition':
                            continue
                        parent = nodes.get(node['parent_id'])
                        if parent is None:
                            raise ValueError(f'Missing source node in {trace}')
                        result = {'run_key': key, 'board_id': job['board_id'], 'method': job['method'],
                                  'seed': job['seed'], 'budget_actions': job['budget'],
                                  'run_status': record['status'], 'node_id': node['node_id'],
                                  'parent_id': node['parent_id'], 'kind': node['kind'], 'scale': node['scale'],
                                  'status': node['status'], 'source': parent['state'],
                                  'proposal': node.get('proposal'), 'actual_endpoint': node['state'],
                                  'actions': node['actions'], 'reached': node['reached'], 'solved': node['solved']}
                        output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + '\n')
                        count += 1
                        transition_actions += len(node['actions'])
                if record['status'] == 'complete':
                    expected = record['outcome']['attempted_actions']
                    if action_events != expected or transition_actions != expected:
                        raise ValueError(f'Completed actions disagree with raw trace: {trace}')
                    completed_actions += expected
    return {'search_transitions': count, 'trace_completed_actions': completed_actions}


def component_costs() -> list[dict[str, Any]]:
    """Group physical attempts by recorded component; unknown billing stays reserved."""
    ledger = CAMPAIGN / 'cost_ledger.sqlite3'
    cache = CAMPAIGN.parent / 'cache/openai'
    totals: dict[tuple[str, str, str], dict[str, Any]] = {}
    with sqlite3.connect(ledger.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.execute('BEGIN')
        attempts = db.execute('SELECT stage,tag,endpoint,request_hash,status,charged_nanos,reserved_nanos,usage_json '
                              'FROM attempts ORDER BY rowid').fetchall()
        events = db.execute('SELECT stage,tag,kind,request_hash,COUNT(*) FROM events '
                            'GROUP BY stage,tag,kind,request_hash').fetchall()
    components_by_hash: dict[str, str] = {}

    def component(digest: str) -> str:
        if digest not in components_by_hash:
            path = cache / f'{digest}.json'
            if path.exists():
                record = read_json(path)
                components_by_hash[digest] = record.get('request', {}).get('payload', {}).get('metadata', {}).get('component', 'unknown')
            else:
                components_by_hash[digest] = 'unknown'
        return components_by_hash[digest]

    def get(stage: str, tag: str, name: str) -> dict[str, Any]:
        key = (stage, tag, name)
        if key not in totals:
            totals[key] = {'stage': stage, 'tag': tag, 'component': name, 'provider_attempts': 0,
                           'logical_requests': 0, 'local_replay_hits': 0, 'live_cache_hits': 0,
                           'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0,
                           'cache_write_tokens': 0, 'cost_settled_usd': 0.0, 'cost_unknown_reserved_usd': 0.0}
        return totals[key]

    for stage, tag, endpoint, digest, status, charged, reserved, usage_json in attempts:
        row = get(stage, tag, component(digest) if endpoint == 'responses' else endpoint)
        usage = json.loads(usage_json) if usage_json else {}
        details = usage.get('input_tokens_details') or {}
        row['provider_attempts'] += 1
        row['input_tokens'] += usage.get('input_tokens', 0) or 0
        row['output_tokens'] += usage.get('output_tokens', 0) or 0
        row['cached_input_tokens'] += details.get('cached_tokens', 0) or 0
        row['cache_write_tokens'] += details.get('cache_write_tokens', 0) or 0
        row['cost_settled_usd'] += (charged or 0) / 1e9
        if status != 'settled':
            row['cost_unknown_reserved_usd'] += (reserved or 0) / 1e9
    for stage, tag, kind, digest, count in events:
        if kind in {'logical_request', 'local_replay_hit', 'live_cache_hit'}:
            row = get(stage, tag, component(digest))
            key = {'logical_request': 'logical_requests', 'local_replay_hit': 'local_replay_hits',
                   'live_cache_hit': 'live_cache_hits'}[kind]
            row[key] += count
    return [totals[key] for key in sorted(totals)]
