"""Independently replay complete policy and search traces in real Sokoban."""
from collections import namedtuple

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts.audit_overnight_stage05_solutions import IncompleteTrace
from scripts.audit_overnight_stage05_solutions import audit_job
from scripts.run_overnight_stage05_policy import run_board
from scripts.run_overnight_stage05_primitive import search

Action = namedtuple("Action", "action generation_metadata")


def board() -> tuple[dict, np.ndarray]:
    cells = np.ones((12, 12), dtype=int)
    cells[0, :] = cells[-1, :] = cells[:, 0] = cells[:, -1] = 0
    cells[1, 1] = 5
    cells[1, 2] = 4
    cells[1, 3] = 2
    cells[4, 1:4] = 3
    state = np.eye(7, dtype=np.float32)[cells]
    record = {"family": "12x12x4", "source_family": "fixture", "split": "evaluation", "board_rank": 0,
              "layout_id": canonical_layout_id(state), "trajectory_id": "fixture", "initial_state": to_rows(state)}
    return record, state


class Policy:
    env = None

    def get_actions(self, state):
        return [Action(3, {})]


def search_job(record, start):
    reached = np.asarray(environment(start).next_state(start, 3))
    source_hash, target_hash = state_hash(start), state_hash(reached)
    return {**{key: record[key] for key in ("layout_id", "family", "source_family", "split", "board_rank")},
            "method": "adasubs", "status": "completed", "solved": True, "initial_state_hash": source_hash,
            "transitions": 1, "noop_actions": 0, "controller_attempts": 1, "controller_hits": 1,
            "candidate_generation_calls": 1, "candidates_proposed": 1, "candidates_assessed": 1,
            "candidate_tokens": 146, "candidates_valid": 1, "policy_calls": 1, "value_calls": 0,
            "solution_actions": [3], "solution_length": 1, "solve_at_transition": 1,
            "trace": [{"event": "generation", "source_rows": to_rows(start), "source_hash": source_hash,
                       "output_lengths": [146]},
                      {"event": "candidate", "source_hash": source_hash, "target_rows": to_rows(reached),
                       "start_transitions": 0, "end_transitions": 1, "actions": [3],
                       "state_hashes": [source_hash, target_hash], "noop_actions": 0,
                       "reached_rows": to_rows(reached), "hit": True, "task_solved": True}]}


def test_policy_path_and_success_replayed_in_real_environment():
    record, _ = board()
    job = run_board(record, Policy(), budget=10, wall_seconds=10)
    result = audit_job(record, job)
    assert result == {"recorded_transitions_replayed": 1, "solution_transitions_replayed": 0,
                      "solutions_verified": 1, "noops_verified": 0}


def test_full_search_solution_is_replayed_separately_from_charged_edge():
    record, start = board()
    job = search_job(record, start)
    result = audit_job(record, job)
    assert result == {"recorded_transitions_replayed": 1, "solution_transitions_replayed": 1,
                      "solutions_verified": 1, "noops_verified": 0}


def test_audit_rejects_tampered_state_hash():
    record, start = board()
    job = search_job(record, start)
    job["trace"][1]["state_hashes"][1] = "incorrect"
    with pytest.raises(ValueError, match="transition"):
        audit_job(record, job)


def test_audit_rejects_uncharged_or_wrong_transition_count():
    record, start = board()
    job = search_job(record, start)
    job["transitions"] = 2
    with pytest.raises(ValueError, match="aggregate"):
        audit_job(record, job)


def test_replay_rejects_plan_that_does_not_solve():
    record, start = board()
    job = search_job(record, start)
    job["solution_actions"] = [0]
    with pytest.raises(ValueError, match="does not solve"):
        audit_job(record, job)


def test_interrupted_infrastructure_trace_is_incomplete_not_model_failure():
    record, start = board()
    job = search_job(record, start)
    job["status"] = "infra_error"
    job["solved"] = None
    del job["trace"][1]["end_transitions"]
    with pytest.raises(IncompleteTrace, match="interrupted"):
        audit_job(record, job)


def test_primitive_search_replay_charges_discarded_moves_separately():
    class Value:
        env = None

        def get_value(self, state):
            return -0.5

    record, _ = board()
    job = search(record, Value(), budget=10, wall_seconds=10)
    result = audit_job(record, job)
    assert result == {"recorded_transitions_replayed": 4, "solution_transitions_replayed": 1,
                      "solutions_verified": 1, "noops_verified": 2}
