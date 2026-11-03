"""The campaign policy study records actual production rollout semantics."""
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from carl.icl.boards import from_rows
from carl.icl.datasets import environment
from carl.icl.luna import LunaComponents
from scripts.campaign_policy_validation import result_row, seed_for, summary


def components_for(board: np.ndarray, action: int) -> tuple[LunaComponents, Any]:
    env = environment(board)
    probabilities = np.zeros(4)
    probabilities[action] = 1
    policy = SimpleNamespace(env=env, probabilities=lambda _state, _target: probabilities)
    return cast(LunaComponents, policy), env


def test_exact_hit_is_measured_by_environment_rollout() -> None:
    board = from_rows(['#####', '#@--#', '#-$-#', '#-.-#', '#####'])
    components, env = components_for(board, 3)
    target = np.asarray(env.next_state(board, 3))
    outcome = result_row(components, board, target, 11)
    assert outcome['exact_hit'] and outcome['first_hit'] == 1
    assert outcome['attempts'] == 1 and outcome['steps'][0]['action'] == 'RIGHT'
    assert outcome['no_ops'] == outcome['repeated_states'] == 0


def test_blocked_moves_count_as_attempts_and_repeated_states() -> None:
    board = from_rows(['#####', '#@--#', '#-$-#', '#-.-#', '#####'])
    components, env = components_for(board, 2)
    target = np.asarray(env.next_state(board, 3))
    outcome = result_row(components, board, target, 11)
    assert not outcome['exact_hit'] and outcome['first_hit'] is None
    assert outcome['attempts'] == outcome['no_ops'] == outcome['repeated_states'] == 8


def test_incomplete_results_cannot_be_selected_and_seeds_are_paired() -> None:
    row = {'variant': 'baseline', 'offset': 2, 'exact_hit': True, 'error': None}
    result = summary([row])['baseline']
    assert result['by_offset']['2'] == {'hits': 1, 'cases': 1, 'errors': 0}
    assert not result['complete'] and result['balanced_exact_hit_rate'] is None
    assert seed_for('board', 10, 2) == seed_for('board', 10, 2)
    assert seed_for('board', 10, 2) != seed_for('board', 10, 4)
