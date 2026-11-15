"""Small local checks for Figure 1 pair selection and CLLP stopping rules."""

import copy
import json
from argparse import Namespace

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts import figure1_cllp_matched as runner
from scripts.figure1_cllp_matched import Pair
from scripts.figure1_cllp_matched import atomic_jsonl
from scripts.figure1_cllp_matched import prepare_pairs
from scripts.figure1_cllp_matched import rollout_pairs
from scripts.figure1_cllp_matched import verify_saved_trial


@pytest.fixture
def sample():
    rows = ["#####", "#@--#", "#-$-#", "#-.-#", "#####"]
    source = from_rows(rows)
    env = environment(source)
    middle = np.asarray(env.next_state(source, 3))
    target = np.asarray(env.next_state(middle, 1))
    source_hash, middle_hash, target_hash = map(state_hash, (source, middle, target))
    layout = canonical_layout_id(source)
    pairs = [{"pair_id": f"p{k}", "rank": 0, "i": 0, "k": k, "layout_id": layout,
              "trajectory_id": "t0", "source_hash": source_hash, "target_hash": end}
             for k, end in ((1, middle_hash), (2, target_hash))]
    states = {state_hash(item): to_rows(item) for item in (source, middle, target)}
    trajectories = [{"id": "t0", "initial_state": rows, "actions": [3, 1]}]
    return source, middle, target, pairs, states, trajectories


def prepared(sample):
    _, _, _, pairs, states, trajectories = sample
    return prepare_pairs(pairs, states, trajectories, layout_count=1, offsets=(1, 2), board_size=5, boxes=1)


def test_selects_each_pair_and_replays_recorded_trajectory(sample):
    pairs = prepared(sample)
    assert [pair.k for pair in pairs] == [1, 2]
    assert len({pair.layout_id for pair in pairs}) == 1
    assert np.array_equal(pairs[1].target, sample[2])


def test_rejects_changed_target_hash_and_missing_offset(sample):
    _, _, _, pairs, states, trajectories = sample
    bad = copy.deepcopy(pairs)
    bad[1]["target_hash"] = bad[0]["target_hash"]
    with pytest.raises(ValueError, match="endpoint hashes"):
        prepare_pairs(bad, states, trajectories, layout_count=1, offsets=(1, 2), board_size=5, boxes=1)
    with pytest.raises(ValueError, match="pair count"):
        prepare_pairs(pairs[:1], states, trajectories, layout_count=1, offsets=(1, 2), board_size=5, boxes=1)


def test_noop_counts_and_exact_state_includes_player(sample):
    pair = prepared(sample)[0]
    actions = iter([0, 3])  # Blocked by the upper wall, then reach the requested player position.
    result = rollout_pairs([pair], lambda _states, _targets: [next(actions)], "CLLP-8", cap=3)[0]
    assert result["actions"] == [0, 3]
    assert result["no_op_count"] == 1
    assert result["first_hit"] == 2 and result["stop_reason"] == "exact_target"
    verify_saved_trial(pair, result, "CLLP-8", cap=3)


def test_terminal_and_exact_hit_precede_action_cap(sample):
    pair = prepared(sample)[1]
    actions = iter([3, 1])
    result = rollout_pairs([pair], lambda _states, _targets: [next(actions)], "CLLP-32", cap=2)[0]
    assert result["exact_hit"] is True and result["first_hit"] == 2
    assert result["stop_reason"] == "exact_target"
    verify_saved_trial(pair, result, "CLLP-32", cap=2)


def test_terminal_before_a_different_target_is_not_a_hit(sample):
    source, _, _, _, _, _ = sample
    env = environment(source)
    other_target = np.asarray(env.next_state(source, 1))
    original = prepared(sample)[0]
    pair = Pair(original.pair_id, original.layout_id, original.trajectory_id, original.rank,
                original.position, original.k, original.source_hash, state_hash(other_target), source, other_target)
    actions = iter([3, 1])
    result = rollout_pairs([pair], lambda _states, _targets: [next(actions)], "CLLP-8", cap=3)[0]
    assert result["exact_hit"] is False and result["first_hit"] is None
    assert result["stop_reason"] == "task_terminal"
    verify_saved_trial(pair, result, "CLLP-8", cap=3)


def test_cap_stops_repeated_noops_and_replay_rejects_tampering(sample):
    pair = prepared(sample)[0]
    result = rollout_pairs([pair], lambda _states, _targets: [0], "CLLP-8", cap=3)[0]
    assert result["stop_reason"] == "action_limit" and result["no_op_count"] == 3
    verify_saved_trial(pair, result, "CLLP-8", cap=3)
    result["state_hashes"][-1] = "changed"
    with pytest.raises(ValueError, match="independent replay"):
        verify_saved_trial(pair, result, "CLLP-8", cap=3)


def test_atomic_jsonl_keeps_all_completed_rows(tmp_path):
    path = tmp_path / "trials.jsonl"
    atomic_jsonl(path, [{"pair_id": "one"}])
    atomic_jsonl(path, [{"pair_id": "one"}, {"pair_id": "two"}])
    assert [json.loads(line)["pair_id"] for line in path.read_text().splitlines()] == ["one", "two"]


def test_rejects_invalid_action_without_making_a_trial(sample):
    with pytest.raises(ValueError, match="one action"):
        rollout_pairs(prepared(sample)[:1], lambda _states, _targets: [4], "CLLP-8", cap=3)


def test_resume_preserves_trials_and_rejects_changed_checkpoint(sample, tmp_path, monkeypatch):
    pairs = prepared(sample)
    data = tmp_path / "data"
    (data / "private").mkdir(parents=True)
    (data / "core_pairs.json").write_text('{"pairs": []}')
    (data / "states.json").write_text('{"states": {}}')
    (data / "private/source_bundle.tar.gz").write_bytes(b"fake bundle")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "pytorch_model.bin").write_bytes(b"fake weights")
    monkeypatch.setattr(runner, "DATA", data)
    monkeypatch.setattr(runner, "CHECKPOINTS", {"CLLP-8": checkpoint})
    monkeypatch.setattr(runner, "load_source_rows", lambda _path: [])
    monkeypatch.setattr(runner, "prepare_pairs", lambda *_args, **_kwargs: pairs)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    class FakeModel:
        def to(self, _device):
            return self

        def float(self):
            return self

        def eval(self):
            return self

        def requires_grad_(self, _value):
            return self

    class FakeBert:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return FakeModel()

    def predict(states, _targets):
        return [3 if state_hash(state) == pairs[0].source_hash else 1 for state in states]

    monkeypatch.setattr(runner, "BertForSequenceClassification", FakeBert)
    monkeypatch.setattr(runner, "model_predictor", lambda *_args: predict)
    output = tmp_path / "run"
    output.mkdir()
    args = Namespace(model="CLLP-8", device="cpu", threads=1, batch_size=2, limit=2, resume=False)
    runner.run(args, output)
    path = output / "trials.jsonl"
    original = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(original) == 2
    atomic_jsonl(path, original[:1])
    args.resume = True
    runner.run(args, output)
    assert [json.loads(line) for line in path.read_text().splitlines()] == original
    assert json.loads((output / "status.json").read_text())["state"] == "partial"

    status_before = (output / "status.json").read_bytes()
    trials_before = path.read_bytes()
    (checkpoint / "pytorch_model.bin").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Resume contract differs"):
        runner.run(args, output)
    assert (output / "status.json").read_bytes() == status_before
    assert path.read_bytes() == trials_before
