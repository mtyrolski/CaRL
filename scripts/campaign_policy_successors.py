"""Run the separate four-simulator-probes-per-decision diagnostic, never policy selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows, to_rows
from carl.icl.campaign_budget import CampaignLedger
from carl.icl.campaign_data import sha256_file
from carl.icl.config import Limits, load_config
from carl.icl.datasets import environment, load_data, verify_solution
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents, board_coordinates
from carl.icl.types import Board, JSONObject
from scripts.campaign_policy_validation import ACTIONS, CAP, OFFSETS, OUTPUT, PRIVATE, ROOT, result_row, seed_for


class SuccessorDiagnostic(LunaComponents):
    """Diagnostic with explicitly disclosed access to four extra simulator transitions."""

    probes = 0

    def policy_payload(self, state: Board, subgoal: Board) -> JSONObject:
        payload = super().policy_payload(state, subgoal)
        payload['input'][0]['content'][0]['text'] += (
            '\nPossible next-state coordinates are computed by the Sokoban simulator. '
            'A move with changed=false is blocked. Choose the action whose next state '
            'best advances toward the exact target; avoid blocked moves.'
        )
        content = payload['input'][-1]['content'][0]
        question = json.loads(content['text'])
        options = []
        for action, name in enumerate(ACTIONS):
            after = np.asarray(self.env.next_state(state, action))
            self.probes += 1
            options.append({'action': name, 'changed': not np.array_equal(state, after),
                            'result_coordinates': board_coordinates(to_rows(after))})
        question['possible_next_states'] = options
        content['text'] = json.dumps(question)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('mock', 'replay', 'openai'), default='replay')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'simulator_successors.json')
    args = parser.parse_args()
    os.environ['CARL_CAMPAIGN_STAGE'] = 'policy_validation'
    os.environ['CARL_CAMPAIGN_TAG'] = 'simulator_successors'
    credentials(args.provider)
    config = load_config('configs/icl/campaign_data/policy_validation_coordinates.json')
    config.limits = Limits(max_requests=8, max_input_tokens=20000, max_total_input_tokens=160000,
                           max_output_tokens=16384, wall_seconds=1200, timeout_seconds=90)
    training, instances = load_data(config)
    witnesses = {item['board_id']: item['actions'] for item in json.loads(PRIVATE.read_text())['witnesses']}
    selections = json.loads((ROOT / 'selection.json').read_text())['selections']
    contract = {'provider': args.provider, 'config': config.model_dump(mode='json', exclude={'locked_policy_fingerprint'}),
                'selection_sha256': sha256_file(ROOT / 'selection.json'),
                'manifest_sha256': sha256_file(ROOT / 'campaign_dataset_manifest.json'),
                'private_sha256': sha256_file(PRIVATE), 'cap': CAP, 'probes_per_decision': 4,
                'note': 'Extra simulator information is diagnostic only; excluded from winner selection.'}
    report: dict[str, Any] = json.loads(args.output.read_text()) if args.output.exists() else {'contract': contract, 'cases': []}
    if report['contract'] != contract:
        raise ValueError('Existing diagnostic report has a different contract')
    done = {(r['board_id'], r['offset']) for r in report['cases'] if r['error'] is None}
    instance_map = {item['id']: item for item in instances}
    for selected in selections:
        board_id, pos = selected['board_id'], selected['position']
        states = verify_solution(from_rows(instance_map[board_id]['state']), witnesses[board_id])
        for k in OFFSETS:
            if (board_id, k) in done:
                continue
            api = ExperimentAPI(config.path('cache_dir'), args.provider, config.limits)
            comp = SuccessorDiagnostic(config, api, environment(states[pos]), training)
            row = {'board_id': board_id, 'layout_id': selected['layout_id'], 'position': pos,
                   'labels': selected['labels'], 'offset': k, 'sampling_seed': seed_for(board_id, pos, k),
                   'cap': CAP, 'source': to_rows(states[pos]), 'target': to_rows(states[pos + k])}
            row.update(result_row(comp, states[pos], states[pos + k], row['sampling_seed']))
            row.update(extra_simulator_probes=comp.probes, metrics=dict(api.metrics))
            previous = [old for old in report['cases'] if (old['board_id'], old['offset']) == (board_id, k)]
            report.setdefault('error_history', []).extend(old for old in previous if old['error'] is not None)
            report['cases'] = [old for old in report['cases'] if (old['board_id'], old['offset']) != (board_id, k)]
            report['cases'].append(row)
            report['summary'] = {str(offset): {'hits': sum(r['exact_hit'] for r in report['cases'] if r['offset'] == offset),
                                'cases': sum(r['offset'] == offset for r in report['cases']),
                                'errors': sum(r['error'] is not None for r in report['cases'] if r['offset'] == offset)}
                                 for offset in OFFSETS}
            report['status'] = 'running' if row['error'] is None else 'interrupted_error'
            if args.provider == 'openai':
                report['ledger'] = CampaignLedger().summary()
            write_json(args.output, report)
            print(f"simulator_successors: k={k}, board={board_id[-8:]}, hit={row['exact_hit']}, "
                  f"steps={row['attempts']}, probes={comp.probes}, error={row['error']}", flush=True)
            if row['error'] is not None:
                return
    report['status'] = 'completed'
    write_json(args.output, report)


if __name__ == '__main__':
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / 'successors.lock'), timeout=0):
        main()
