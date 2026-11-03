"""Audit cached 32x32 validation requests and complete-board generator outputs offline."""
import argparse
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from carl.icl.api import write_json
from carl.icl.config import ROOT, load_config

DEFAULT_OUTPUT = ROOT / 'analysis/luna_icl/campaign_100/scale_tuning/io_audit.json'


def audit(ledger_path: Path, cache_dir: Path) -> dict[str, Any]:
    """Inspect only validation-request cache entries identified by the campaign ledger."""
    uri = ledger_path.resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        hashes = [row[0] for row in db.execute("SELECT DISTINCT request_hash FROM events "
                  "WHERE stage='scale_validation' AND tag LIKE 'tune_%generated_32x32x10%' AND kind='logical_request'")]
    counts: Counter[str] = Counter()
    usage: dict[str, dict[str, int]] = {}
    for digest in hashes:
        path = cache_dir / 'openai' / f'{digest}.json'
        if not path.exists():
            counts['missing_cache'] += 1
            continue
        try:
            record = json.loads(path.read_text())
            payload = record['request']['payload']
            role = payload['metadata']['component']
            counts[f'{role}_requests'] += 1
            query = json.loads(payload['input'][-1]['content'][0]['text'])
            required = ['state', 'subgoal'] if role == 'policy' else ['state']
            full_input = all(len(query[key]) == 32 and all(isinstance(row, str) and len(row) == 32 for row in query[key])
                             for key in required)
            counts[f'{role}_full_inputs' if full_input else f'{role}_invalid_inputs'] += 1
            token_usage = record.get('usage', {})
            for kind in ('input_tokens', 'output_tokens'):
                n = token_usage.get(kind)
                if isinstance(n, int):
                    label = f'{role}_{kind}'
                    item = usage.setdefault(label, {'min': n, 'max': n, 'count': 0})
                    item['min'], item['max'], item['count'] = min(item['min'], n), max(item['max'], n), item['count'] + 1
            if role == 'generator':
                output = json.loads(record['output_text'])
                boards = output['subgoals']
                if not isinstance(boards, list):
                    raise ValueError('Expected a subgoal list')
                counts['generator_nonempty' if boards else 'generator_empty'] += 1
                for board in boards:
                    full = isinstance(board, dict) and all(
                        isinstance(board.get(f'row_{i:02d}'), str) and len(board[f'row_{i:02d}']) == 32
                        for i in range(32))
                    counts['generator_full_boards' if full else 'generator_incomplete_boards'] += 1
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            counts['invalid_or_missing_response'] += 1
    return {'unique_request_hashes': len(hashes), 'counts': dict(counts), 'usage': usage,
            'scope': '32x32 scale-validation tuning cache; no API calls, no held-out boards',
            'note': 'Complete dimensions alone do not establish legal or reachable subgoals.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_config('configs/icl/campaign_data/relative_u.json')
    budget = json.loads((ROOT / 'configs/icl/campaign_budget.json').read_text())
    path = Path(budget['ledger_path'])
    result = audit(path if path.is_absolute() else ROOT / path, config.path('cache_dir'))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
