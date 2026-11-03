"""Resume calibration, archiving invalid policy responses before bounded retries."""
import argparse
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from typing import Any

from filelock import FileLock

from carl.icl.campaign_budget import CampaignLedger
from carl.icl.config import ROOT, load_config
from carl.icl.experiment import calibration_components, calibration_path

STAGE = 'calibration'
TAG = 'calibration_v2'


def invalid_policy_output(record: dict[str, Any]) -> bool:
    """Match the same probability validation as the production CLLP."""
    if record.get('status') != 'completed' or record.get('request', {}).get('payload', {}).get('metadata', {}).get('component') != 'policy':
        return False
    try:
        value = json.loads(record['output_text'])
        probabilities = value['probabilities']
        return (not isinstance(probabilities, list) or len(probabilities) != 4
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(item) or not 0 <= item <= 1 for item in probabilities)
                or sum(probabilities) <= 0)
    except (ValueError, TypeError, KeyError):
        return True


def archive_invalid(config_path: str) -> str:
    """Preserve a bad response and remove only its active cache entry, without making a call."""
    config = load_config(config_path)
    components = calibration_components(config, 'replay')
    directory = calibration_path(config, components).parent
    with FileLock(str(directory / '.calibration.lock'), timeout=0):
        summary = json.loads((directory / 'calibration.json').read_text())
        error = summary.get('error', '')
        if summary.get('status') != 'partial' or not (error == 'Invalid JSON from model component'
                or error.startswith('Policy must return')):
            raise ValueError('Calibration did not stop on an invalid policy distribution')
        ledger = CampaignLedger()
        with closing(sqlite3.connect(ledger.path)) as db:
            row = db.execute("SELECT request_hash FROM events WHERE stage=? AND tag=? AND kind='logical_request' "
                             'ORDER BY id DESC LIMIT 1', (STAGE, TAG)).fetchone()
        if row is None:
            raise ValueError('No matching calibration request in the campaign ledger')
        digest = row[0]
        if len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
            raise ValueError('Invalid request digest')
        active = config.path('cache_dir') / 'openai' / f'{digest}.json'
        with FileLock(str(active) + '.lock', timeout=0):
            raw = active.read_bytes()
            record = json.loads(raw)
            stored_request = record.get('request')
            request_hash = hashlib.sha256(json.dumps(stored_request, sort_keys=True).encode()).hexdigest()
            if record.get('request_hash') != digest or request_hash != digest or not invalid_policy_output(record):
                raise ValueError('Latest cached response is not an invalid completed policy response')
            archive = directory / 'rejected_policy_responses'
            archive.mkdir(parents=True, exist_ok=True)
            destination = archive / f'{digest}.{hashlib.sha256(raw).hexdigest()[:16]}.json'
            if destination.exists() and destination.read_bytes() != raw:
                raise ValueError('Archived response hash collision')
            if not destination.exists():
                destination.write_bytes(raw)
            if destination.read_bytes() != raw:
                raise OSError('Failed to preserve rejected response')
            active.unlink()
            ledger.event(STAGE, 'invalid_policy_output_archived', 'responses', digest, TAG)
    return str(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/icl/campaign_data/calibration_live_v2.json')
    parser.add_argument('--max-invalid-retries', type=int, default=4)
    args = parser.parse_args()
    if args.max_invalid_retries < 0:
        raise ValueError('Retry bound must be nonnegative')
    os.environ['CARL_CAMPAIGN_STAGE'] = STAGE
    os.environ['CARL_CAMPAIGN_TAG'] = TAG
    for attempt in range(args.max_invalid_retries + 1):
        config = load_config(args.config)
        components = calibration_components(config, 'replay')
        summary_path = calibration_path(config, components)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            error = summary.get('error', '')
            if summary.get('status') == 'partial' and (error == 'Invalid JSON from model component'
                    or error.startswith('Policy must return')):
                if attempt == args.max_invalid_retries:
                    print('Invalid-output retry limit reached; calibration remains resumable.', flush=True)
                    return 1
                print(f'Preserved invalid policy response: {archive_invalid(args.config)}', flush=True)
        completed = subprocess.run([sys.executable, '-m', 'scripts.run_icl_search', 'calibrate',
                                    '--config', args.config, '--execute'], cwd=ROOT, check=False)
        if completed.returncode == 0:
            return 0
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        error = summary.get('error', '')
        if not (error == 'Invalid JSON from model component' or error.startswith('Policy must return')):
            print(f'Calibration stopped without an invalid-output retry: {error}', flush=True)
            return 1
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
