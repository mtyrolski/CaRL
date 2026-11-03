"""Search must reject calibration labels not tied to its frozen training pool."""
from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from carl.icl.boards import from_rows, to_rows
from carl.icl.calibration import calibrate
from carl.icl.config import Calibration, ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment import calibration_path, components_for_run
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.types import Board, TrainingData


def prepared_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ExperimentConfig, TrainingData, Board, dict]:
    board = from_rows(['#####', '#@--#', '#-$-#', '#-.-#', '#####'])
    successor = np.asarray(environment(board).next_state(board, 3))
    training = cast(TrainingData, {'contexts': [{
        'id': 'training-context', 'state': to_rows(board), 'remaining_actions': 5,
        'candidates': [{'k': 1, 'actions': [3], 'subgoal': to_rows(successor)}],
        'calibration_indices': [0],
    }]})
    config = ExperimentConfig(method='relative_u', few_shot=1, data_dir=str(tmp_path / 'data'),
                              cache_dir=str(tmp_path / 'cache'),
                              calibration=Calibration(trials=2, action_cap=2, sampled_targets=1))
    api = ExperimentAPI(config.path('cache_dir'), 'mock', config.calibration.limits)
    components = LunaComponents(config, api, environment(board), training)
    monkeypatch.setattr(components, 'probabilities', lambda _state, _target: np.array([0., 0., 0., 1.]))
    result = calibrate(config, components, config.path('data_dir') / 'calibration')
    assert result['status'] == 'completed' and len(result['examples']) == 1
    return config, training, board, result


def test_valid_calibration_is_accepted_without_api_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, training, board, result = prepared_calibration(tmp_path, monkeypatch)
    monkeypatch.setattr(ExperimentAPI, 'call', lambda *_args, **_kwargs: pytest.fail('Unexpected inference call'))
    components = components_for_run(config, 'mock', training, board)
    assert components.calibrated == result
    assert components.api.metrics['requests'] == components.api.metrics['logical_requests'] == 0


@pytest.mark.parametrize('tamper, message', [
    ('signature', 'signature'), ('state', 'differs from its training trajectory'),
    ('subgoal', 'differs from its training trajectory'), ('context', 'unknown training context'),
    ('candidate_index', 'unselected training target'), ('actions', 'differs from its training trajectory'),
    ('offset', 'differs from its training trajectory'), ('label', 'integrity check'),
    ('weight', 'integrity check'), ('median', 'integrity check'),
])
def test_tampered_calibration_is_rejected_before_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           tamper: str, message: str) -> None:
    from carl.icl.api import write_json

    config, training, board, result = prepared_calibration(tmp_path, monkeypatch)
    tampered = deepcopy(result)
    example = tampered['examples'][0]
    if tamper == 'signature':
        tampered['signature'] = 'f' * 64
    elif tamper == 'state':
        example['state'] = ['#####', '#--@#', '#-$-#', '#-.-#', '#####']
    elif tamper == 'subgoal':
        example['subgoal'] = ['#####', '#--@#', '#-$-#', '#-.-#', '#####']
    elif tamper == 'context':
        example['context'] = 'foreign-context'
    elif tamper == 'candidate_index':
        example['target_index'] = 99
    elif tamper == 'actions':
        example['actions'] = [2]
    elif tamper == 'label':
        example['u'] = 0.1
    elif tamper == 'weight':
        example['weight'] = 0.3
    elif tamper == 'median':
        example['a_c'] = 100.0
    else:
        example['k'] = 2
    api = ExperimentAPI(config.path('cache_dir'), 'mock', config.limits)
    reader = LunaComponents(config, api, environment(board), training)
    write_json(calibration_path(config, reader), tampered)
    monkeypatch.setattr(ExperimentAPI, 'call', lambda *_args, **_kwargs: pytest.fail('Unexpected inference call'))
    with pytest.raises(ValueError, match=message):
        components_for_run(config, 'mock', training, board)
    assert api.metrics['requests'] == 0
