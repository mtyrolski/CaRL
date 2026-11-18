"""Tests for representation parity, train-only support, and restart safety."""

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from carl.icl import intent_representation_experiment
from carl.icl.intent_representation_experiment import Representation
from carl.icl.intent_representation_experiment import configuration
from carl.icl.intent_representation_experiment import messages_for
from carl.icl.intent_representation_experiment import parse_representation
from carl.icl.intent_representation_experiment import prepare_output
from carl.icl.intent_representation_experiment import render_representation
from carl.icl.intent_representation_experiment import support_bank


def intent():
    return {
        "effects": [
            {"predicate": "box_at", "row": 2, "col": 3},
            {"predicate": "boxes_on_goals_at_least", "count": 2},
        ],
        "preserve": [{"predicate": "player_at", "row": 1, "col": 1}],
        "instruction": None,
    }


def row(layout, number, typ, whole_goal=False):
    value = (
        {
            "effects": [{"predicate": "box_at", "row": 2, "col": 3}],
            "preserve": [],
            "instruction": None,
        }
        if typ == "box"
        else {
            "effects": [
                {"predicate": "box_at", "row": 2, "col": 3},
                {
                    "predicate": "boxes_on_goals_at_least",
                    "count": 3 if whole_goal else 2,
                },
            ],
            "preserve": [],
            "instruction": None,
        }
    )
    return {
        "layout_id": layout,
        "record_id": str(number),
        "split": "train",
        "validation": {"accepted": True},
        "effect_type": typ,
        "intent": value,
        "task_goal": "Put boxes on goals",
        "start_observation": ["#####", "#...#", "#---@", "#####"],
        "later_observation": ["secret"],
    }


@pytest.mark.parametrize("representation", list(Representation))
def test_round_trip_preserves_predicate_arrays_and_conditions(representation):
    target = intent()
    assert (
        parse_representation(
            representation, render_representation(representation, target)
        )
        == target
    )
    target["preserve"] = []
    assert (
        parse_representation(
            representation, render_representation(representation, target)
        )
        == target
    )


@pytest.mark.parametrize(
    "representation,raw",
    [
        ("json", '```json\n{"effects":[],"preserve":[],"instruction":null}\n```'),
        ("json", '{"effects":[],"effects":[],"preserve":[],"instruction":null}'),
        ("formal", "E[box(1,2)] P[] plus extra"),
        ("formal", "E[] P[]"),
        ("natural", "Achieve: box at row 1 column 2. Preserve: nothing. Extra."),
    ],
)
def test_rejects_malformed_or_empty_output(representation, raw):
    with pytest.raises(ValueError):
        parse_representation(representation, raw)


def test_instruction_is_explicitly_out_of_scope():
    target = intent()
    target["instruction"] = "do not discard me"
    for representation in Representation:
        with pytest.raises(ValueError):
            render_representation(representation, target)


def test_support_selection_is_train_only_balanced_and_deterministic():
    rows = []
    for i in range(8):
        rows.extend(
            [
                row(f"layout-{i}", i * 2, "box"),
                row(f"layout-{i}", i * 2 + 1, "goal_conjunction", whole_goal=i == 1),
            ]
        )
    hidden = row("000-test", 100, "box")
    hidden["split"] = "test"
    rows.append(hidden)
    bank = support_bank(rows)
    assert bank == support_bank(list(reversed(rows)))
    assert not bank["zero"]
    used = [
        entry["layout_id"]
        for name in ("balanced0", "balanced1", "balanced2")
        for entry in bank[name]
    ]
    assert len(set(used)) == 6
    assert all(
        entry["split"] == "train" for examples in bank.values() for entry in examples
    )
    assert all(
        [entry["effect_type"] for entry in bank[name]] == ["box", "goal_conjunction"]
        for name in ("balanced0", "balanced1", "balanced2")
    )
    assert all(
        entry["intent"]["effects"][-1].get("count", 0) < 3
        for name in ("balanced0", "balanced1", "balanced2")
        for entry in bank[name]
    )


def test_hidden_query_and_support_fields_do_not_enter_prompts():
    support = row("a", 1, "box")
    query = row("b", 2, "box")
    query["split"] = "test"
    before = messages_for(query, [support], "formal")
    query.update(
        later_observation=["other secret"],
        fragment_actions=[999],
        k=99,
        intent={"secret": True},
    )
    support.update(
        later_observation=["different"], fragment_actions=[777], trajectory_id="private"
    )
    assert messages_for(query, [support], "formal") == before
    assert "secret" not in json.dumps(before)


def test_resume_rejects_changed_selection_and_recovers_partial_line(tmp_path):
    output = tmp_path / "predictions.jsonl"
    config = {
        "split": "test",
        "variant": "formal_zero",
        "representation": "formal",
        "support_set": "zero",
    }
    records = [
        {"record_id": "a", "layout_id": "board"},
        {"record_id": "b", "layout_id": "other"},
    ]
    assert prepare_output(output, config, records, False) == []
    value = {
        "effects": [{"predicate": "box_at", "row": 1, "col": 2}],
        "preserve": [],
        "instruction": None,
    }
    saved = {
        "record_id": "a",
        "layout_id": "board",
        **config,
        "raw_output": "E[box(1,2)] P[]",
        "syntax_valid": True,
        "parsed_intent": value,
    }
    output.write_text(json.dumps(saved) + '\n{"incomplete"')
    assert prepare_output(output, config, records, True) == [saved]
    assert list(tmp_path.glob("predictions.jsonl.partial-*"))
    with pytest.raises(ValueError):
        prepare_output(output, dict(config, variant="changed"), records, True)
    saved["parsed_intent"] = None
    output.write_text(json.dumps(saved) + "\n")
    with pytest.raises(ValueError):
        prepare_output(output, config, records, True)


def test_local_model_revision_is_verified_before_running(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_text("test")
    (model / "experiment_revision.json").write_text(json.dumps({"revision": "fixed"}))
    data = tmp_path / "data"
    data.write_text("frozen")
    args = Namespace(
        adapter=None,
        model=str(model),
        revision="fixed",
        variant=None,
        representation="json",
        support_set="zero",
        data=str(data),
        test_ids=str(data),
        support_data=str(data),
        support_manifest=str(data),
        split="test",
        dtype="bfloat16",
        seed=7,
        threads=2,
        max_length=512,
        max_new_tokens=64,
        require_cuda=True,
    )
    saved = configuration(args, data)
    assert saved["revision_verification"] == "experiment_revision.json"
    assert saved["threads"] == 2
    root = Path(intent_representation_experiment.__file__).parent
    for field in ("protocol_components_sha256", "generator_components_sha256"):
        assert saved[field]
        for name, digest in saved[field].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    assert "intents/representation/formats.py" in saved["protocol_components_sha256"]
    assert "intents/generation/backend.py" in saved["generator_components_sha256"]
    args.revision = "wrong"
    with pytest.raises(ValueError, match="revision"):
        configuration(args, data)
