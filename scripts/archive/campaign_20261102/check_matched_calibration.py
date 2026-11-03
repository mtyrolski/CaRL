"""Verify frozen calibration IDs, demonstrations and query fields without inference."""
import hashlib
import json
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.config import load_config
from carl.icl.datasets import load_data
from carl.icl.experiment import calibration_path, components_for_run
from carl.icl.luna import FIELDS

METHODS = ('relative_u', 'absolute_d', 'ratio_d_over_ac', 'normalized_offset')
OUTPUT = Path('analysis/luna_icl/campaign_100/calibration_v2/demo_equivalence.json')


def main() -> None:
    selected, contracts, signatures = [], {}, set()
    for method in METHODS:
        config = load_config(f'configs/icl/campaign_data/{method}.json')
        training, _ = load_data(config)
        state = from_rows(training['contexts'][0]['state'])
        components = components_for_run(config, 'replay', training, state)
        examples = components.generator_examples(state, config.scales[0], resolve_embeddings=False)
        identities = [(example['context'], example.get('target_index')) for example in examples]
        shared = [(example['context'], example.get('target_index'), example['state'], example['subgoal'], example.get('weight'))
                  for example in examples]
        selected.append(shared)
        payload = components.generator_payload(state, config.scales[0], resolve_embeddings=False)
        query = json.loads(payload['input'][-1]['content'][0]['text'])
        if set(query) != {'state', 'goal', 'num_subgoals', FIELDS[method]}:
            raise ValueError(f'Unexpected query fields for {method}')
        assert components.calibrated is not None
        signature = components.calibrated['signature']
        signatures.add(signature)
        path = calibration_path(config, components)
        contracts[method] = {'selected_ids': identities, 'query_fields': sorted(query),
                             'calibration_signature': signature, 'calibration_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                             'example_labels': [example['k'] / config.offset_denominator if method == 'normalized_offset'
                                                else example[FIELDS[method]] for example in examples]}
    if len(signatures) != 1 or any(value != selected[0] for value in selected[1:]):
        raise ValueError('Methods disagree on calibration or weighted demonstrations')
    write_json(OUTPUT, {'status': 'matched', 'methods': contracts,
                        'note': 'Query uses current state, task goal and fixed requested scale only; no test-board reference.'})
    print(f'Matched {len(METHODS)} methods and {len(selected[0])} demonstrations: {OUTPUT}')


if __name__ == '__main__':
    main()
