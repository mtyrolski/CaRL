"""Policy examples must be valid training witnesses, not evaluation trajectories."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id, from_rows, state_hash
from carl.icl.components import ICLComponents
from carl.icl.config import ExperimentConfig
from carl.icl.data_sources.sokoban import environment
from carl.icl.few_shot_selection import diverse_training_examples

TRAINING = Path("configs/icl/checkpoints/launch100/assets/data/training.json")


@pytest.fixture(scope="module")
def training():
    return json.loads(TRAINING.read_text())


@pytest.mark.parametrize("count", [8, 32, 64])
def test_balanced_examples_are_deterministic_and_replay_from_initial(training, count):
    first = diverse_training_examples(training, count)
    assert first == diverse_training_examples(copy.deepcopy(training), count)
    assert len(first) == count
    metadata = [row["metadata"] for row in first]
    assert len({row["layout_id"] for row in metadata}) == 8
    assert Counter((row["first_action"], row["push"]) for row in metadata) == {
        (direction, is_push): count // 8 for direction in ("UP", "DOWN", "LEFT", "RIGHT") for is_push in (False, True)
    }
    assert len({(row["source_hash"], row["target_hash"]) for row in metadata}) == count
    assert len({row["source_hash"] for row in metadata}) >= {8: 8, 32: 32, 64: 60}[count]
    assert set(row["k_actions"] for row in metadata) == {1, 2, 4, 8, 12, 16}
    contexts = {row["id"]: row for row in training["contexts"]}
    for example in first:
        info = example["metadata"]
        context = contexts[info["trajectory_id"]]
        source, target = (from_rows(example["question"][key]) for key in ("state", "subgoal"))
        assert canonical_layout_id(source) == info["layout_id"]
        assert state_hash(source) == info["source_hash"]
        assert state_hash(target) == info["target_hash"]
        assert not np.array_equal(source, target)
        assert info["target_index"] - info["anchor_index"] == info["k_actions"]
        actions = max(context["candidates"], key=lambda row: row["k"])["actions"]
        state = from_rows(context["state"])
        env = environment(state)
        if info["anchor_index"] == 0:
            assert np.array_equal(state, source)
        for i, action in enumerate(actions[:info["target_index"]]):
            next_state = env.next_state(state, action)
            assert not np.array_equal(state, next_state)
            state = next_state
            if i + 1 == info["anchor_index"]:
                assert np.array_equal(state, source)
        assert np.array_equal(state, target)
        assert example["answer"]["probabilities"][actions[info["anchor_index"]]] == 1.0


def test_corrupt_pre_anchor_state_fails_even_if_suffix_is_unchanged(training):
    bad = copy.deepcopy(training)
    # A candidate prior to a selected interior anchor is no longer the state
    # reached by the witness. Its retained suffix is left untouched.
    context = next(c for c in bad["contexts"] if c["id"] == "trajectory_186")
    context["candidates"][0]["subgoal"] = copy.deepcopy(context["candidates"][1]["subgoal"])
    with pytest.raises(ValueError, match="prefix fails exact simulator replay"):
        diverse_training_examples(bad, 32)


@pytest.mark.parametrize("count", [8, 32, 64])
def test_frozen_manifest_matches_training_and_selected_examples(training, count):
    manifest = json.loads(Path(f"configs/icl/figure1/diverse_policy_demonstrations_{count}_v2.json").read_text())
    assert manifest["training_source_sha256"] == hashlib.sha256(TRAINING.read_bytes()).hexdigest()
    expected = []
    for example in diverse_training_examples(training, count):
        payload = {"question": example["question"], "answer": example["answer"]}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode()).hexdigest()
        expected.append({"metadata": example["metadata"], "query_answer_sha256": digest})
    assert manifest["examples"] == expected


@pytest.mark.parametrize("count", [8, 32])
def test_balanced_policy_sends_same_examples_to_both_models(training, count):
    source = from_rows(training["contexts"][0]["state"])
    target = from_rows(training["contexts"][0]["candidates"][0]["subgoal"])
    env = environment(source)
    client = type("MockClient", (), {"provider": "mock"})()
    payloads = []
    for model in ("gpt-6-luna", "gpt-6.1-terra"):
        config = ExperimentConfig(model=model, few_shot=count, policy_demo_selection="diverse_training_v2",
                                  policy_coordinate_features=True)
        payloads.append(ICLComponents(config, client, env, training).policy_payload(source, target))
    assert len(payloads[0]["input"]) == 2 + 2 * count  # developer + pairs + query
    assert payloads[0]["input"] == payloads[1]["input"]
    for message in payloads[0]["input"][1:-1:2]:
        assert set(json.loads(message["content"][0]["text"])) == {
            "state", "subgoal", "state_coordinates", "subgoal_coordinates"
        }
    assert set(json.loads(payloads[0]["input"][-1]["content"][0]["text"])) == {
        "state", "subgoal", "state_coordinates", "subgoal_coordinates"
    }
    bad = ExperimentConfig(model="gpt-6-luna", few_shot=7, policy_demo_selection="diverse_training_v2")
    with pytest.raises(ValueError, match="exactly 8, 32 or 64"):
        ICLComponents(bad, client, env, training).policy_examples()


def test_32_is_fixed_prefix_of_64_and_short_supply_fails(training):
    assert diverse_training_examples(training, 8) == diverse_training_examples(training, 32)[:8]
    assert diverse_training_examples(training, 32) == diverse_training_examples(training, 64)[:32]
    sparse = copy.deepcopy(training)
    # Repeating a training layout cannot manufacture distinct layout supply.
    sparse["contexts"] = sparse["contexts"][:7]
    with pytest.raises(ValueError, match="at least eight"):
        diverse_training_examples(sparse, 32)


def test_short_category_supply_fails_without_duplicates(training, monkeypatch):
    import carl.icl.few_shot_selection as selector

    selector._select.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(selector, "OFFSETS", (16,))
        with pytest.raises(ValueError, match="Insufficient unique training pairs"):
            diverse_training_examples(training, 32)
    selector._select.cache_clear()
