"""Inspect calibration progress and ledger accounting without issuing API calls."""
import argparse
import json
from collections import Counter

from carl.icl.campaign_budget import CampaignLedger
from carl.icl.config import load_config
from carl.icl.experiment import calibration_components, calibration_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/icl/campaign_data/calibration_live_v2.json')
    args = parser.parse_args()
    config = load_config(args.config)
    components = calibration_components(config, 'replay')
    directory = calibration_path(config, components).parent
    files = sorted(directory.glob('context_*_candidate_*.json'))
    records = [json.loads(path.read_text()) for path in files]
    trials = [row for data in records for row in data['records']]
    ledger = CampaignLedger().summary()
    expected_targets = len(components.training['contexts']) * (config.calibration.sampled_targets or config.calibration.candidates)
    print(json.dumps({'directory': str(directory), 'status': 'completed' if (directory / 'calibration.json').exists()
                     and json.loads((directory / 'calibration.json').read_text())['status'] == 'completed' else 'running_or_partial',
                     'targets_started': len(files), 'targets_expected': expected_targets,
                     'trials_completed': len(trials), 'trials_expected': expected_targets * config.calibration.trials,
                     'exact_hits': sum(row['exact_hit'] for row in trials),
                     'solved_before_target': sum(row['solved_before_target'] for row in trials),
                     'target_trials': dict(Counter(len(data['records']) for data in records)),
                     'ledger': [row for row in ledger['attempts'] if row['stage'] == 'calibration']}, indent=2))


if __name__ == '__main__':
    main()
