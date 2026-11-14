"""Checks for the frozen Luna/LoRA calibration-candidate preparation."""

import json

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import validate_intent
from scripts.prepare_luna_lora_data import candidate_pool
from scripts.prepare_luna_lora_data import prepare_board
from scripts.prepare_luna_lora_data import validate_split
from scripts.prepare_luna_lora_data import write_frozen


@pytest.fixture
def trajectory():
    initial = ["#######", "#@-$-.#", "#--$-.#", "#-----#", "#-----#", "#-----#", "#######"]
    state = from_rows(initial)
    env = environment(state)
    actions = [3, 3, 3, 1]
    states = [initial]
    for action in actions:
        state = env.next_state(state, action)
        states.append(to_rows(state))
    return states, actions


def source(trajectory):
    return {
        "id": "toy",
        "path": "toy.json",
        "sha256": "a" * 64,
        "split": "train",
        "layout_id": canonical_layout_id(from_rows(trajectory[0][0]))
    }


def test_candidates_are_distinct_and_checked_on_the_recorded_fragment(trajectory):
    states, _ = trajectory
    pool, rejected = candidate_pool(states, 0)
    assert not rejected
    assert {item.kind for item in pool} == {"box", "goal_conjunction"}
    assert len({json.dumps(item.intent["effects"], sort_keys=True) for item in pool}) == len(pool)
    for item in pool:
        assert not validate_intent(item.intent, states[:item.offset + 1])
        assert item.intent["preserve"] == []


def test_checked_candidates_share_start_and_replay_actions_deterministically(trajectory):
    states, actions = trajectory
    kwargs = {"contexts_per_board": 2, "candidates_per_context": 2, "seed": 17}
    records, rejected, stats = prepare_board(source(trajectory), states, actions, **kwargs)
    assert (records, rejected, stats) == prepare_board(source(trajectory), states, actions, **kwargs)
    assert len(records) == 4
    assert len({row["context_id"] for row in records}) == 2
    assert stats["selected"] == 2
    for context_id in {row["context_id"] for row in records}:
        group = [row for row in records if row["context_id"] == context_id]
        assert len(group) == 2
        assert group[0]["start_observation"] == group[1]["start_observation"]
        assert group[0]["intent"] != group[1]["intent"]
    for row in records:
        index, offset = row["i"], row["k"]
        assert row["split"] == "train" and row["phase"] == "calibration"
        assert row["fragment_actions"] == actions[index:index + offset]
        assert row["fragment_observations"] == states[index:index + offset + 1]
        assert row["later_observation"] == states[index + offset]
        assert not validate_intent(row["intent"], row["fragment_observations"])
        assert row["policy_measurements"] is None


def test_context_shortages_are_reported_without_fabricating_records(trajectory):
    states, actions = trajectory
    records, rejected, stats = prepare_board(source(trajectory), states, actions, 1, 4, 17)
    assert records == []
    assert stats["examined"] == len(actions)
    assert stats["insufficient_candidates"] == len(actions)
    assert stats["selected"] == 0
    assert sum(row["reason"] == "too few distinct non-walking intents" for row in rejected) == len(actions)
    assert rejected[-1]["reason"] == "insufficient qualifying contexts"


def test_checker_failures_are_retained_and_not_selected(trajectory, monkeypatch):
    states, _ = trajectory
    monkeypatch.setattr("scripts.prepare_luna_lora_data.validate_intent", lambda *_: ["bad predicate"])
    pool, rejected = candidate_pool(states, 0)
    assert not pool
    assert rejected and all(item["reasons"] == ["bad predicate"] for item in rejected)


def test_split_rejects_related_layouts_and_prior_test_sources():
    rows = [
        {
            "id": "a",
            "layout_id": "a",
            "split": "train",
            "historical_cohort": "train"
        },
        {
            "id": "b",
            "layout_id": "b",
            "split": "validation",
            "historical_cohort": "train"
        },
        {
            "id": "c",
            "layout_id": "c",
            "split": "test",
            "historical_cohort": "test"
        },
    ]
    validate_split(rows, expected=(1, 1))
    with pytest.raises(ValueError, match="repeated trajectory or canonical layout"):
        validate_split([rows[0], rows[1], {**rows[2], "layout_id": "a"}], expected=(1, 1))
    with pytest.raises(ValueError, match="historical training cohort"):
        validate_split([{**rows[0], "historical_cohort": "test"}, *rows[1:]], expected=(1, 1))


def test_frozen_output_is_reproducible_and_never_replaces_different_data(tmp_path):
    records = [{"record_id": "one"}]
    write_frozen(tmp_path, records, [], {"status": "complete"})
    write_frozen(tmp_path, records, [], {"status": "complete"})
    original = (tmp_path / "calibration_candidates.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        write_frozen(tmp_path, [{"record_id": "other"}], [], {"status": "complete"})
    assert (tmp_path / "calibration_candidates.jsonl").read_bytes() == original
