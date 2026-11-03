"""Evaluate the existing configured value contract on the same validation states."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.experiment import credentials
from scripts.archive.campaign_20261102.campaign_components_validation import DATASETS, OUTPUT, config_for, validation_data, value_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('mock', 'replay', 'openai'), default='replay')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    os.environ['CARL_CAMPAIGN_STAGE'] = 'scale_validation'
    credentials(args.provider)
    output = OUTPUT / args.provider
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), Path('configs/icl/campaign_data/frozen_policy_fixed.json'),
             Path('carl/icl/luna.py'), Path('scripts/archive/campaign_20261102/campaign_components_validation.py')]
    for root, private, family in DATASETS:
        paths.extend([Path(root) / 'campaign_dataset_manifest.json', Path(root) / 'training.json',
                      Path(root) / 'value_examples.json', Path(root) / family / 'component_validation.json',
                      Path(private) / 'component_validation' / f'{family}.json'])
    contract = {'provider': args.provider, 'inputs_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                                               for path in paths}, 'variant': 'current_frozen'}
    manifest = output / 'current_value_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError('Current value validation contract changed')
    if not manifest.exists():
        write_json(manifest, contract)
    for root, private, family in DATASETS:
        training, boards = validation_data(root, private, family)
        # This is the actual previous frozen config, not another prompt tune.
        config = config_for(root, family)
        for board_id, states in boards:
            last = len(states) - 1
            for position in (0, last // 4, last // 2, 3 * last // 4, last):
                path = output / f'value_{board_id}_{position}_current_frozen.json'
                previous = json.loads(path.read_text()) if path.exists() else None
                if previous is not None and not (args.retry_errors and previous['error'] is not None):
                    continue
                row = value_case(args.provider, config, training, board_id, states, position, 'current_frozen')
                if previous is not None:
                    row['prior_attempts'] = [*previous.get('prior_attempts', []), previous]
                write_json(path, row)
                print('value', board_id[-8:], position, 'current_frozen', row['predicted_value'], row['error'], flush=True)
                if row['error'] is not None and (row['error']['type'] == 'BudgetExceeded' or
                                                 'responses failed' in row['error']['message']):
                    return


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / 'current_value.lock'), timeout=0):
        main()
