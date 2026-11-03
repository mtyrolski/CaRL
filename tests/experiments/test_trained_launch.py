"""Technical shared-search launch checks with a tiny scripted model; no weights."""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from carl.icl.boards import from_rows
from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.types import Board, Probabilities
from scripts.trained_reference.run import execute_case


class ScriptedModel:
    def __init__(self, state: Board) -> None:
        self.env = environment(state)
        self.diagnostics: Counter[str] = Counter()

    def propose(self, state: Board, scale: float) -> list[Board]:
        del scale
        self.diagnostics['propose'] += 1
        return [np.asarray(self.env.next_state(state, 3))]

    def probabilities(self, state: Board, subgoal: Board) -> Probabilities:
        del state, subgoal
        return np.array([0., 0., 0., 1.])

    def value(self, state: Board) -> float:
        del state
        return 0.5


def test_completed_success_resume_is_replay_verified(tmp_path: Path) -> None:
    state = from_rows(['#####', '#---#', '#@$.#', '#---#', '#####'])
    model = ScriptedModel(state)
    config = ExperimentConfig(method='fixed', scales=[8, 4], fixed_cap='horizon',
                              action_budget=500, max_expansions=None)
    result = execute_case(model, state, config, 11, tmp_path, {'fixture': True}, resume=False)
    assert result['outcome']['solved'] and result['verification']['solution_replayed']
    calls = model.diagnostics['propose']
    assert execute_case(model, state, config, 11, tmp_path, {'fixture': True}, resume=True) == json.loads(json.dumps(result))
    assert model.diagnostics['propose'] == calls
    saved = json.loads((tmp_path / 'case.json').read_text())
    saved['outcome']['action_path'] = [2]
    (tmp_path / 'case.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='solution does not replay'):
        execute_case(model, state, config, 11, tmp_path, {'fixture': True}, resume=True)


def test_interrupted_case_never_reexecutes_automatically(tmp_path: Path) -> None:
    state = from_rows(['#####', '#---#', '#@$.#', '#---#', '#####'])
    (tmp_path / 'started.json').write_text('{}')
    model = ScriptedModel(state)
    with pytest.raises(ValueError, match='Interrupted'):
        execute_case(model, state, ExperimentConfig(), 11, tmp_path, {}, resume=True)
    assert not model.diagnostics
