"""Freeze measured generator and value contracts atop the already frozen policy."""
import hashlib
import json
from pathlib import Path
from typing import Any

from carl.icl.api import write_json
from carl.icl.config import ExperimentConfig, load_config
from scripts.archive.campaign_20261102.campaign_components_validation import GENERATOR, OUTPUT, VALUES


def main() -> None:
    summary_path = OUTPUT / 'summary.json'
    summary = json.loads(summary_path.read_text())
    generator, value = summary['generator_winner'], summary['value_winner']
    base = load_config('configs/icl/campaign_data/frozen_policy_fixed.json')
    settings: dict[str, Any] = base.model_dump(mode='json')
    settings.update(GENERATOR[generator], generator_max_attempts=1, num_subgoals=2,
                    name='campaign_frozen_components_fixed')
    if value in VALUES:
        settings.update(value_target='return', value_examples=VALUES[value], value_board_features=False,
                        value_response_format='text', value_tokens=128)
    elif value != 'current_frozen':
        raise ValueError('Unknown measured value variant')
    config = ExperimentConfig.model_validate(settings)
    root = Path('configs/icl/campaign_data')
    fixed = root / 'frozen_components_fixed.json'
    relative = root / 'frozen_components_relative.json'
    absolute = root / 'frozen_components_absolute.json'
    write_json(fixed, config.model_dump(mode='json'))
    write_json(relative, {'extends': fixed.name, 'name': 'campaign_frozen_components_relative_provisional',
                          'method': 'relative', 'scales': [0.9, 0.5]})
    write_json(absolute, {'extends': fixed.name, 'name': 'campaign_frozen_components_absolute_provisional',
                          'method': 'absolute', 'scales': [8, 4]})
    loaded = [load_config(path) for path in (fixed, relative, absolute)]
    shared = ('model', 'locked_policy_fingerprint', 'generator_reasoning_effort', 'generator_response_format',
              'generator_layout_schema', 'generator_max_attempts', 'generator_tokens', 'num_subgoals',
              'value_target', 'value_examples', 'value_board_features', 'value_reasoning_effort',
              'value_response_format', 'value_tokens', 'few_shot', 'discount', 'seed')
    for field in shared:
        if len({str(getattr(item, field)) for item in loaded}) != 1:
            raise ValueError(f'Frozen component differs across methods: {field}')
    if len({item.prompts.model_dump_json() for item in loaded}) != 1:
        raise ValueError('Frozen component prompts differ across methods')
    source_files = [summary_path, root / 'frozen_policy_fixed.json', Path('data/icl/campaign_dataset_v2/training.json'),
                    Path('data/icl/campaign_dataset_v2/value_examples.json'), fixed, relative, absolute,
                    OUTPUT / 'openai/manifest.json', OUTPUT / 'openai/current_value_manifest.json']
    write_json(OUTPUT / 'frozen_components.json', {
        'generator': generator, 'value': value, 'policy_fingerprint': config.locked_policy_fingerprint,
        'selection_rule': {'generator': summary['generator_selection'], 'value': summary['value_selection']},
        'sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        'calibration_cap': config.calibration.action_cap,
        'note': 'Relative and absolute scales are provisional until separate scale validation.',
    })
    print(json.dumps({'generator': generator, 'value': value, 'config': str(fixed)}, indent=2))


if __name__ == '__main__':
    main()
