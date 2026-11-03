"""The launch adapter preserves measured components and separate resource bounds."""
import json
from pathlib import Path

import pytest

from carl.icl.boards import from_rows
from carl.icl.config import ExperimentConfig
from carl.icl.experiment_api import ExperimentAPI
from scripts.verbal_icl.launch import load_bundle, run_case


def bundle_file(tmp_path: Path) -> Path:
    state = ['#####', '#@--#', '#-$-#', '#.--#', '#####']
    config = ExperimentConfig(model='gpt-6-luna', few_shot=1, num_subgoals=1)
    training = {'contexts': [{'id': 'synthetic-train', 'state': state, 'remaining_actions': 3,
                             'candidates': [], 'calibration_indices': []}]}
    payload = {'schema': 'carl-verbal-launch-v1', 'calibration_status': 'completed',
               'config': config.model_dump(mode='json'),
               'prompts': {'generator': 'Grounded generator', 'policy': 'Frozen policy', 'judge': 'Frozen judge'},
               'examples': [{'context': 'synthetic-train', 'target_index': 0, 'source': state,
                             'goal': 'Move player to [1,2].', 'u': 0.5}],
               'demonstrations': [{'source': state, 'state': state, 'after': state,
                                   'goal': 'Move player to [1,2].', 'action': 3, 'reached': False}],
               'generator_examples': 1, 'reference': {'config': config.model_dump(mode='json'),
                                                      'training': training},
               'provenance': {'purpose': 'synthetic offline test only'}}
    path = tmp_path / 'verbal.json'
    path.write_text(json.dumps(payload))
    return path


def test_node_limit_is_independent_of_action_limit(tmp_path: Path) -> None:
    bundle = load_bundle(bundle_file(tmp_path))
    state = from_rows(bundle.examples[0]['source'])
    api = ExperimentAPI(tmp_path / 'cache', 'mock', bundle.config.limits)
    events: list[dict[str, object]] = []
    result = run_case(bundle, api, state, action_budget=20, node_budget=1, trace=events.append)
    assert result.finish_reason == 'expansion_budget' and result.expansions == 1
    assert result.attempted_actions == bundle.config.calibration.action_cap
    assert len(events) == result.attempted_actions
    result = run_case(bundle, api, state, action_budget=2, node_budget=500)
    assert result.finish_reason == 'action_budget' and result.attempted_actions == 2
    assert 'code_sha256' in bundle.contract


def test_incomplete_calibration_cannot_launch(tmp_path: Path) -> None:
    path = bundle_file(tmp_path)
    payload = json.loads(path.read_text())
    payload['calibration_status'] = 'insufficient_coverage'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='complete'):
        load_bundle(path)
