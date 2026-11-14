"""Focused checks for the observable-effect contract and actual environment."""
import json

import pytest

from carl.icl.intent_data import SokobanPredicate
from carl.icl.intent_data import predicate_value
from carl.icl.intent_data import validate_intent
from carl.icl.intent_data import validate_prediction
from scripts import prepare_intent_generator_data as data_module
from scripts.prepare_intent_generator_data import replay

START = ["#####", "#@$.#", "#---#", "#---#", "#####"]
PUSH = ["#####", "#-@*#", "#---#", "#---#", "#####"]
RETURN = ["#####", "#@-*#", "#---#", "#---#", "#####"]


def test_zero_based_row_column_and_goal_truth():
    box = {"predicate": "box_at", "row": 1, "col": 3}
    goal = {"predicate": "boxes_on_goals_at_least", "count": 1}
    assert not predicate_value(box, START)
    assert predicate_value(box, PUSH)
    assert not predicate_value(goal, START)
    assert predicate_value(goal, PUSH)
    assert validate_intent({"effects": [box, goal], "preserve": [], "instruction": None}, [START, PUSH]) == []


def test_predicate_values_and_candidate_keep_the_json_contract():
    assert json.loads(json.dumps([predicate for predicate in SokobanPredicate
                                 ])) == ["box_at", "player_at", "boxes_on_goals_at_least"]
    result = data_module.candidate(START, PUSH, 0)
    assert result is not None
    kind, intent = result
    assert kind == "goal_conjunction"
    assert intent == {
        "effects": [
            {
                "predicate": "box_at",
                "row": 1,
                "col": 3
            },
            {
                "predicate": "boxes_on_goals_at_least",
                "count": 1
            },
        ],
        "preserve": [],
        "instruction": None,
    }


def test_config_resolves_paths_from_repository_root(tmp_path):
    config = {
        "source_dir": "sources",
        "prior_bundle": "data/bundle.tar.gz",
        "prior_pairs": "data/pairs.json",
        "prior_states": "data/states.json",
        "historical_labels": "../annotations/labels.jsonl",
        "craftax_manifest": "../annotations/craftax.json",
        "training_archive_member": "training.json",
        "witness_archive_member": "witness.json",
        "annotated_trajectory_id": "example",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = data_module.load_data_sources(path, root=tmp_path)
    assert result.source_dir == tmp_path / "sources"
    assert result.prior_bundle == tmp_path / "data/bundle.tar.gz"
    assert result.historical_labels == tmp_path.parent / "annotations/labels.jsonl"
    assert result.annotated_trajectory_id == "example"


def test_conjunction_requires_every_effect_to_change():
    unchanged = {"predicate": "box_at", "row": 1, "col": 2}
    changed = {"predicate": "box_at", "row": 1, "col": 3}
    assert any("already true" in error for error in validate_intent({
        "effects": [changed, unchanged],
        "preserve": []
    }, [START, PUSH]))


def test_preserve_is_required_as_an_array_even_when_empty():
    intent = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}]}
    assert "preserve must be an array" in validate_prediction(intent, START)
    intent["preserve"] = []
    assert not validate_prediction(intent, START)


def test_conjunction_rejects_incompatible_player_and_box_positions():
    impossible = {
        "effects": [{
            "predicate": "player_at",
            "row": 1,
            "col": 3
        }, {
            "predicate": "box_at",
            "row": 1,
            "col": 3
        }],
        "preserve": []
    }
    assert any("same position" in error for error in validate_prediction(impossible, START))
    impossible = {
        "effects": [{
            "predicate": "player_at",
            "row": 2,
            "col": 1
        }, {
            "predicate": "player_at",
            "row": 2,
            "col": 2
        }],
        "preserve": []
    }
    assert any("multiple positions" in error for error in validate_prediction(impossible, START))


def test_preservation_is_checked_on_intervening_states():
    intent = {
        "effects": [{
            "predicate": "box_at",
            "row": 1,
            "col": 3
        }],
        "preserve": [{
            "predicate": "player_at",
            "row": 1,
            "col": 1
        }]
    }
    assert predicate_value(intent["preserve"][0], START)
    assert predicate_value(intent["preserve"][0], RETURN)
    assert any("fragment state 1" in error for error in validate_intent(intent, [START, PUSH, RETURN]))


@pytest.mark.parametrize("predicate", [
    {
        "predicate": "box_at",
        "row": 0,
        "col": 0
    },
    {
        "predicate": "box_at",
        "row": 5,
        "col": 1
    },
    {
        "predicate": "player_at",
        "row": True,
        "col": 1
    },
    {
        "predicate": "boxes_on_goals_at_least",
        "count": 2
    },
    {
        "predicate": "unsupported",
        "row": 1,
        "col": 3
    },
    {
        "predicate": [],
        "row": 1,
        "col": 3
    },
    {
        "predicate": {
            "name": "box_at"
        },
        "row": 1,
        "col": 3
    },
])
def test_reject_invalid_predicate(predicate):
    assert validate_prediction({"effects": [predicate], "preserve": []}, START)


def test_replay_checks_primitive_direction_and_saved_target(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"id": "test", "ascii": "\n".join(START), "actions": "R"}))
    prior = [{"id": "test-0", "state": START, "candidates": [{"k": 1, "actions": [3], "subgoal": PUSH}]}]
    states, actions, audit = replay(path, prior, [], {})
    assert states == [START, PUSH]
    assert actions == [3]
    assert audit["primitive_actions"] == 1
    assert audit["push_actions"] == 1
    assert audit["no_ops"] == 0
    assert audit["historically_recorded_states_checked"] == 2
    prior[0]["candidates"][0]["actions"] = [2]
    with pytest.raises(ValueError, match="target/actions mismatch"):
        replay(path, prior, [], {})


def test_only_documented_witness_prefix_may_exclude_post_terminal_suffix(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"id": "test", "ascii": "\n".join(START), "actions": "RL"}))
    with pytest.raises(ValueError, match="after solved state"):
        replay(path, [], [], {})
    witness = {"id": "test", "initial_state": START, "actions": [3]}
    states, actions, audit = replay(path, [], [], {}, witness)
    assert states == [START, PUSH]
    assert actions == [3]
    assert audit["raw_source_actions"] == 2
    assert audit["discarded_post_terminal_actions"] == 1
    witness["actions"] = [2]
    with pytest.raises(ValueError, match="not an exact source prefix"):
        replay(path, [], [], {}, witness)


def test_offsets_rotate_deterministically_across_available_candidates(monkeypatch):
    monkeypatch.setattr(
        data_module, "candidate", lambda start, end, chooser: ("box", {
            "effects": [{
                "predicate": "box_at",
                "row": 1,
                "col": 1
            }],
            "preserve": [],
            "instruction": None
        }))
    states = [[str(i)] for i in range(80)]
    selected = data_module.sample(states, 12, 20261114)
    assert {k for _, k, _, _ in selected} == set(data_module.OFFSETS)
    assert selected == data_module.sample(states, 12, 20261114)
