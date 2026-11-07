"""Hand-calculated checks of the measured object, independent of runner helpers."""
import itertools
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from carl.icl.boards import from_rows, state_hash, to_rows
from carl.icl.datasets import environment
from carl.icl.search import rollout
from scripts.reachability_runner import engine
from scripts.reachability_runner.independent_verify import check_action_rng, replay_actions

SOURCE = ["#####", "#---#", "#@$.#", "#---#", "#####"]
UP = ["#####", "#@--#", "#-$.#", "#---#", "#####"]
UP_RIGHT = ["#####", "#-@-#", "#-$.#", "#---#", "#####"]
UP_RIGHT_RIGHT = ["#####", "#--@#", "#-$.#", "#---#", "#####"]
SOLVED = ["#####", "#---#", "#-@*#", "#---#", "#####"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: pytest.fail("network unavailable offline"))


class Oracle:
    def __init__(self, actions):
        self.actions = iter(actions)

    def probabilities(self, _state, _target):
        action = next(self.actions)
        return np.eye(4)[action]


def trace(source, actions):
    state = from_rows(source)
    env = environment(state)
    records = []
    for action in actions:
        successor = np.asarray(env.next_state(state, action))
        records.append({"action": action, "probabilities": np.eye(4)[action].tolist(),
                        "before_hash": state_hash(state), "after_hash": state_hash(successor),
                        "changed": not np.array_equal(state, successor)})
        state = successor
    return records


@pytest.mark.parametrize(("target", "actions", "cap", "reason", "first_hit", "solved", "noops"), [
    (SOURCE, [], 3, "exact_hit", 0, False, 0),
    (UP, [0], 2, "exact_hit", 1, False, 0),
    (UP_RIGHT, [0, 3], 2, "exact_hit", 2, False, 0),
    (UP_RIGHT_RIGHT, [0, 3], 2, "miss_cap", None, False, 0),
    (UP, [2], 1, "miss_cap", None, False, 1),
    (UP_RIGHT, [0, 1], 2, "miss_cap", None, False, 0),
    (UP, [3], 3, "solved_before_target", None, True, 0),
    (SOLVED, [3], 3, "exact_hit", 1, True, 0),
])
def test_manual_sokoban_examples(target, actions, cap, reason, first_hit, solved, noops):
    source, goal = from_rows(SOURCE), from_rows(target)
    env = environment(source)
    oracle = Oracle(actions)
    outcome = rollout(env, oracle, source, goal, cap=cap, rng=np.random.default_rng(123))
    records = trace(SOURCE, actions)
    replay = replay_actions(source, goal, records, cap)
    check_action_rng(records, 123)
    assert replay["stop_reason"] == reason
    assert replay["first_hit"] == first_hit and replay["task_solved"] == solved
    assert replay["no_op_count"] == noops
    assert bool(outcome.reached) == (first_hit is not None)
    assert len(outcome.actions) == len(actions)
    assert to_rows(replay["states"][-1]) == to_rows(outcome.state)


def test_manual_states_and_oracle_path_are_not_just_self_consistent():
    source = from_rows(SOURCE)
    env = environment(source)
    up = np.asarray(env.next_state(source, 0))
    assert to_rows(up) == UP
    one_right = np.asarray(env.next_state(up, 3))
    assert to_rows(one_right) == UP_RIGHT
    assert to_rows(env.next_state(one_right, 3)) == UP_RIGHT_RIGHT
    assert to_rows(env.next_state(source, 3)) == SOLVED
    assert not np.array_equal(source, up)  # same boxes; player location is part of state
    assert np.array_equal(env.next_state(up, 1), source)  # genuine loop
    with pytest.raises(ValueError, match="after first hit"):
        replay_actions(source, up, trace(SOURCE, [0, 1]), 2)


def test_stochastic_policy_probability_by_enumeration():
    # UP or blocked LEFT, each 1/2; two independent draws hit UP with probability 3/4.
    source, target = from_rows(SOURCE), from_rows(UP)
    hits = 0
    for first, second in itertools.product([0, 2], repeat=2):
        actions = [first] if first == 0 else [first, second]
        result = replay_actions(source, target, trace(SOURCE, actions), 2)
        hits += result["exact_hit"]
    assert hits == 3
    assert hits / 4 == .75


@pytest.mark.skipif(
    not Path("data/icl/reachability_exploratory_v2/private/source_bundle.tar.gz").is_file(),
    reason="requires the local trajectory witness bundle",
)
def test_evaluation_witness_sentinel_cannot_enter_policy_payload(tmp_path):
    raw = json.loads(Path("configs/icl/reachability_runner/offline_smoke.json").read_text())
    raw.update(output_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"))
    study = engine.Study.model_validate(raw)
    _manifest, training, core, _reference, boards = engine.data_inputs(study)
    pair = dict(core[0], witness_actions=["SECRET_FUTURE_ACTION"], future_states=["SECRET_FUTURE_STATE"])
    query = engine.policy_query(pair, boards)
    assert set(query) == {"state", "subgoal"}
    payloads = [engine._payload(study, profile, training, from_rows(query["state"]), from_rows(query["subgoal"]))
                for profile in study.arms.values()]
    for payload in payloads:
        assert "SECRET_FUTURE" not in json.dumps(payload)
        messages = payload["input"]
        final_question = json.loads([m for m in messages if m["role"] == "user"][-1]["content"][0]["text"])
        assert final_question["state"] == query["state"] and final_question["subgoal"] == query["subgoal"]
        assert set(final_question) <= {"state", "subgoal", "state_coordinates", "subgoal_coordinates"}
    assert payloads[0]["input"] == payloads[1]["input"]


@pytest.mark.skipif(
    not Path("data/icl/reachability_exploratory_v2/private/source_bundle.tar.gz").is_file(),
    reason="requires the local trajectory witness bundle",
)
def test_corrupt_actions_identity_config_response_or_usage_are_detected(tmp_path, monkeypatch):
    raw = json.loads(Path("configs/icl/reachability_runner/offline_smoke.json").read_text())
    raw.update(output_dir=str(tmp_path / "out"), cache_dir=str(tmp_path / "cache"), action_cap=2)
    study = engine.Study.model_validate(raw)
    monkeypatch.setattr(engine, "verify_dataset", lambda _root: {})
    engine.prepare(study)
    engine.run(study, max_trials=1)
    assert engine.verify(study)["counts"]["complete"] == 1
    file = next((Path(study.output_dir) / "trials").glob("*.json"))
    original = json.loads(file.read_text())
    for field in ("action", "after_hash"):
        changed = json.loads(json.dumps(original))
        changed["steps"][0][field] = (changed["steps"][0][field] + 1) % 4 if field == "action" else "0" * 64
        file.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            engine.verify(study)
    changed = json.loads(json.dumps(original))
    changed["identity"]["target_hash"] = "0" * 64
    file.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity"):
        engine.verify(study)
    file.write_text(json.dumps(original))
    unexpected = file.with_name("unexpected-trial-id.json")
    unexpected.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="identifier"):
        engine.verify(study)
    unexpected.unlink()
    modified = study.model_copy(deep=True)
    modified.policy_temperature = 2.0
    with pytest.raises(ValueError, match="contract"):
        engine.verify(modified)
    trace_row = original["requests"][0]
    cache = Path(study.cache_dir) / "mock" / (trace_row["request_hash"] + ".json")
    saved = cache.read_text()
    cached = json.loads(saved)
    cached["usage"]["input_tokens"] = 1
    cache.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="response cache"):
        engine.verify(study)
    cache.write_text(saved)
    with sqlite3.connect(Path(study.output_dir) / "ledger.sqlite3") as db:
        db.execute("UPDATE calls SET usage_json=? WHERE id=?", ('{"input_tokens":123}', trace_row["ledger_id"]))
    with pytest.raises(ValueError, match="Ledger usage"):
        engine.verify(study)
