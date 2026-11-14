"""Independent checks must reject malformed intents and invalid search paths."""

import pytest

from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts.audit_lora_transfer_results import independently_valid
from scripts.audit_lora_transfer_results import parse_strict
from scripts.audit_lora_transfer_results import replay_found


def sample():
    board = ["######", "#@$.##", "#-*--#", "#----#", "#----#", "######"]
    intent = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}],
              "preserve": [{"predicate": "boxes_on_goals_at_least", "count": 1}], "instruction": None}
    return board, intent


def test_independent_checker_rejects_ambiguous_or_trivial_predictions():
    board, intent = sample()
    assert independently_valid(intent, board)
    assert not parse_strict('{"effects":[],"effects":[]}')[0]
    assert not parse_strict('{"count":NaN}')[0]
    already_true = {"effects": [{"predicate": "box_at", "row": 1, "col": 2}], "preserve": []}
    assert not independently_valid(already_true, board)
    duplicate = {"effects": intent["effects"] * 2, "preserve": []}
    assert not independently_valid(duplicate, board)


def test_search_replay_checks_actual_state_noops_and_preservation():
    board, intent = sample()
    state = from_rows(board)
    reached = to_rows(environment(state).next_state(state, 3))
    solver = {"witness_actions": [3], "witness_depth": 1, "depth_limit": 24, "reached_observation": reached}
    replay_found(board, intent, solver)
    with pytest.raises(ValueError, match="no-op"):
        replay_found(board, intent, {**solver, "witness_actions": [0]})
    with pytest.raises(ValueError, match="Preservation fails"):
        replay_found(board, intent, {**solver, "witness_actions": [1, 3], "witness_depth": 2})
    with pytest.raises(ValueError, match="disagrees"):
        replay_found(board, intent, {**solver, "reached_observation": board})
