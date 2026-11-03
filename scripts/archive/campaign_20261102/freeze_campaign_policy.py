"""Freeze the best completed policy validation run, with cost as tie-breaker."""
from __future__ import annotations

import json
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.campaign_budget import CampaignLedger
from carl.icl.campaign_data import sha256_file
from carl.icl.config import load_config
from carl.icl.datasets import environment, load_data
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from scripts.archive.campaign_20261102.campaign_policy_validation import OUTPUT, VARIANTS


def main() -> None:
    report_path = OUTPUT / 'report.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'completed' or not all(row['complete'] for row in report['summary'].values()):
        raise ValueError('Cannot freeze an incomplete or errored comparison')
    ledger = CampaignLedger().summary()
    costs = {name: sum(item['committed_usd'] for item in ledger['attempts']
                       if item['stage'] == 'policy_validation' and item['tag'] == name)
             for name in VARIANTS}
    winner = min(VARIANTS, key=lambda name: (-report['summary'][name]['balanced_exact_hit_rate'], costs[name], name))
    selected = load_config('configs/icl/campaign_data/' + VARIANTS[winner])
    # Keep the separately tuned generator/value and transplant only the selected policy contract.
    target = load_config('configs/icl/experiments/tuned_fixed.json')
    fields = ('model', 'few_shot', 'policy_coordinate_features', 'policy_reasoning_effort',
              'policy_response_format', 'policy_tokens', 'policy_temperature', 'seed', 'reasoning_effort',
              'response_format', 'selection')
    for field in fields:
        setattr(target, field, getattr(selected, field))
    target.prompts = target.prompts.model_copy(update={"policy": selected.prompts.policy})
    target.calibration.action_cap = selected.calibration.action_cap
    target.data_dir = 'data/icl/campaign_dataset_v2'
    target.family = 'real_12x12x4'
    target.split = 'component_validation'
    target.instances = 4
    target.name = 'campaign_frozen_policy_fixed'
    training, instances = load_data(target)
    board = from_rows(instances[0]['state'])
    components = LunaComponents(target, ExperimentAPI(target.path('cache_dir'), 'mock', target.limits), environment(board), training)
    fingerprints = {row['policy_fingerprint'] for row in report['cases'] if row['variant'] == winner}
    if fingerprints != {components.fingerprint()}:
        raise ValueError('Frozen policy does not match the evaluated policy fingerprint')
    target.locked_policy_fingerprint = components.fingerprint()
    config_path = Path('configs/icl/campaign_data/frozen_policy_fixed.json')
    write_json(config_path, target.model_dump(mode='json'))
    write_json(OUTPUT / 'selection.json', {
        'winner': winner, 'criterion': 'mean exact-hit over four boards and four offsets; lower committed cost breaks ties',
        'cost_note': 'Includes conservative unknown reservations; 10% regional uplift; not an invoice.',
        'costs_committed_usd': costs, 'summary': report['summary'], 'policy_fingerprint': components.fingerprint(),
        'report_sha256': sha256_file(report_path), 'config_sha256': sha256_file(config_path),
        'cap': target.calibration.action_cap, 'config': str(config_path),
    })
    print(json.dumps({'winner': winner, 'balanced_exact_hit_rate': report['summary'][winner]['balanced_exact_hit_rate'],
                      'config': str(config_path), 'costs_committed_usd': costs}, indent=2))


if __name__ == '__main__':
    main()
