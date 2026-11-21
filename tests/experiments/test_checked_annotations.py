"""Checked Sokoban annotations, source isolation, and legacy prompt inputs."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from carl.icl.boards import canonical_layout_id, from_rows, level_key, to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_annotations import AnnotationRequest, prepare_annotations
from carl.icl.intent_augmentation import BoardSymmetry, transform_actions, transform_rows
from carl.icl.intent_domain import (
    Action,
    SokobanObservation,
    SokobanSnapshot,
    TASK_GOAL,
    check_intent,
    decode_intent,
    render_instruction,
    replay_fragment,
)
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_in_context import in_context_messages
from carl.icl.intents.generation.data import prompt_messages
from carl.icl.intents.representation.prompts import messages_for
from carl.icl.local_intent_policy import run_record
from carl.icl.luna_intent_rollout import build_prompt

BOARD = [
    "#######",
    "#@$-.-#",
    "#-$-.-#",
    "#-----#",
    "#--#--#",
    "#-----#",
    "#######",
]
PUSHED = [BOARD[0], "#-@$.-#", *BOARD[2:]]
MOVED_BACK = [BOARD[0], "#@-$.-#", *BOARD[2:]]
VALID_CONJUNCTION = {
    "effects": [
        {"predicate": "box_at", "row": 1, "col": 3},
        {"predicate": "player_at", "row": 1, "col": 2},
    ],
    "preserve": [{"predicate": "box_at", "row": 2, "col": 2}],
}
BROKEN_PRESERVATION = {
    "effects": [{"predicate": "box_at", "row": 1, "col": 3}],
    "preserve": [{"predicate": "player_at", "row": 1, "col": 1}],
}
WALL_PREDICATE = {
    "effects": [{"predicate": "box_at", "row": 4, "col": 3}],
    "preserve": [],
}


def _trajectory(source_id: str, split: str, rows: list[str], actions: Sequence[int]) -> dict:
    return {"source_id": source_id, "split": split, "start_observation": rows, "actions": actions}


def _source(tmp_path: Path, *trajectories: dict) -> Path:
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"trajectories": trajectories}), encoding="utf-8")
    return path


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_checked_labels_and_prompt_inputs_are_distinct(tmp_path: Path) -> None:
    source = _source(tmp_path, _trajectory("demo", "train", BOARD, [Action.RIGHT]))
    calls = []

    def annotator(request: AnnotationRequest) -> dict:
        calls.append(request)
        assert request.start_observation.rows == tuple(BOARD)
        assert request.later_observation.rows == tuple(PUSHED)
        assert request.domain
        assert not any(hasattr(request, key) for key in ("split", "source_id", "actions", "record_id"))
        return VALID_CONJUNCTION

    output = tmp_path / "out"
    prepare_annotations(source, output, max_offset=1, seed=21, annotator=annotator, annotator_id="fixture")
    row, = _records(output / "D_A.jsonl")
    audit, = _records(output / "annotation_audit.jsonl")
    assert len(calls) == 1
    assert row["validation"]["accepted"] is True and audit["validation"]["accepted"] is True
    assert row["start_observation"] == BOARD and row["task_goal"] == TASK_GOAL
    assert row["layout_id"] == canonical_layout_id(from_rows(BOARD))
    assert row["intent"]["effects"] == VALID_CONJUNCTION["effects"]
    assert row["instruction"] == render_instruction(decode_intent(VALID_CONJUNCTION))
    assert not ({"evidence", "later_observation", "observations", "actions", "fragment_actions"} & row.keys())
    assert audit["source_id"] == row["source_id"] == "demo"
    assert audit["evidence"]["start_index"] == 0
    assert audit["evidence"]["actions"] == [Action.RIGHT]
    assert audit["evidence"]["observations"] == [BOARD, PUSHED]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 21 and manifest["source_ids"] == ["demo"]

    poisoned = {
        **row,
        "split": "test",
        "later_observation": ["SECRET_FUTURE"],
        "fragment_actions": ["SECRET_ACTION"],
        "intent": {"instruction": "SECRET_LABEL"},
    }
    assert prompt_messages(poisoned) == prompt_messages(row)
    policy = build_prompt(from_rows(BOARD), {
        **row["intent"], "instruction": "SECRET_TEACHER_PROSE", "evidence": "SECRET_WITNESS",
    }, TASK_GOAL)
    assert "SECRET_" not in policy
    policy_input = json.loads(policy.split("\n\n")[-1])
    assert policy_input["state"] == BOARD
    assert policy_input["intent"] == {key: row["intent"][key] for key in ("effects", "preserve")}

    repeated = tmp_path / "repeat"
    prepare_annotations(source, repeated, max_offset=1, seed=21, annotator=annotator, annotator_id="fixture")
    assert (output / "D_A.jsonl").read_bytes() == (repeated / "D_A.jsonl").read_bytes()
    assert (output / "annotation_audit.jsonl").read_bytes() == (repeated / "annotation_audit.jsonl").read_bytes()


@pytest.mark.parametrize("sketch,valid_json,reason", [
    pytest.param(json.dumps({**WALL_PREDICATE, "instruction": None}), True, "wall", id="illegal-intent"),
    pytest.param("{", False, "invalid json", id="invalid-json"),
])
def test_full_fragment_rejects_transient_violation_and_bad_sketch(
    tmp_path: Path, sketch: str, valid_json: bool, reason: str,
) -> None:
    source = _source(tmp_path, _trajectory("demo", "train", BOARD, [Action.RIGHT, Action.LEFT]))

    def annotator(request: AnnotationRequest) -> dict | str:
        start, end = request.start_observation.rows, request.later_observation.rows
        if start == tuple(BOARD) and end == tuple(MOVED_BACK):
            # This condition is true again at the endpoint after being broken midway.
            return BROKEN_PRESERVATION
        if start == tuple(BOARD) and end == tuple(PUSHED):
            return VALID_CONJUNCTION
        return sketch

    output = tmp_path / "out"
    prepare_annotations(source, output, max_offset=2, seed=21, annotator=annotator, annotator_id="fixture")
    accepted, = _records(output / "D_A.jsonl")
    assert accepted["intent"]["effects"] == VALID_CONJUNCTION["effects"]
    rejected = [row["validation"] for row in _records(output / "annotation_audit.jsonl")
                if not row["validation"]["accepted"]]
    assert len(rejected) == 2
    preservation, = [check for check in rejected if check["reasons"] == ["preserve[0]: false at fragment state 1"]]
    assert preservation["json_valid"] is True
    invalid, = [check for check in rejected if reason in " ".join(check["reasons"]).lower()]
    assert invalid["json_valid"] is valid_json


@pytest.mark.parametrize("other_split", ["validation", "test"])
def test_d4_layout_overlap_rejected_before_annotating(tmp_path: Path, other_split: str) -> None:
    rotated = transform_rows(BOARD, BoardSymmetry.ROTATE_90)
    assert level_key(from_rows(BOARD)) != level_key(from_rows(rotated))
    assert canonical_layout_id(from_rows(BOARD)) == canonical_layout_id(from_rows(rotated))
    source = _source(
        tmp_path,
        _trajectory("train-demo", "train", BOARD, [Action.RIGHT]),
        _trajectory("held-out-demo", other_split, rotated, transform_actions([Action.RIGHT], BoardSymmetry.ROTATE_90)),
    )

    def forbidden(_request: AnnotationRequest) -> dict:
        pytest.fail("the annotator ran before validating the canonical split")

    with pytest.raises(ValueError, match="layout"):
        prepare_annotations(source, tmp_path / "out", max_offset=1, seed=21, annotator=forbidden)
    assert not (tmp_path / "out" / "D_A.jsonl").exists()


def test_recorded_observation_must_match_simulated_action(tmp_path: Path) -> None:
    forged = {**_trajectory("demo", "train", BOARD, [Action.RIGHT]), "observations": [BOARD, MOVED_BACK]}
    source = _source(tmp_path, forged)

    def forbidden(_request: AnnotationRequest) -> dict:
        pytest.fail("the annotator ran before checking the replay witness")

    with pytest.raises(ValueError):
        prepare_annotations(source, tmp_path / "out", max_offset=1, seed=21, annotator=forbidden)
    assert not (tmp_path / "out" / "D_A.jsonl").exists()


def test_repeated_fragment_does_not_multiply_training_rows(tmp_path: Path) -> None:
    actions = [Action.DOWN, Action.UP, Action.DOWN]
    source = _source(tmp_path, _trajectory("demo", "train", BOARD, actions))

    def annotator(request: AnnotationRequest) -> dict:
        row = 1 if request.later_observation.rows[1][1] == "@" else 2
        return {"effects": [{"predicate": "player_at", "row": row, "col": 1}], "preserve": []}

    output = tmp_path / "out"
    prepare_annotations(source, output, max_offset=1, seed=21, annotator=annotator, annotator_id="fixture")
    records = _records(output / "D_A.jsonl")
    audit = _records(output / "annotation_audit.jsonl")
    assert len(records) == 2
    assert len({row["record_id"] for row in records}) == 2
    assert len(audit) == 3
    assert sum(bool(row["validation"]["accepted"]) for row in audit) == 2
    assert any("duplicate" in str(row["validation"]).lower() for row in audit)


@pytest.mark.parametrize("index,row,col", [(0, 1, 3), (1, 2, 2)])
def test_partial_observation_cannot_certify_annotation(index: int, row: int, col: int) -> None:
    states = [BOARD.copy(), PUSHED.copy()]
    cells = states[index][row]
    states[index][row] = cells[:col] + "?" + cells[col + 1:]
    intent = decode_intent(VALID_CONJUNCTION)
    assert check_intent(intent, [SokobanObservation(tuple(state)) for state in states]).status == "unknown"
    complete = [SokobanObservation(tuple(state)) for state in (BOARD, PUSHED)]
    assert check_intent(intent, complete).status == "accepted"


def test_snapshot_replay_matches_existing_full_state_environment() -> None:
    start = SokobanSnapshot(tuple(BOARD))
    actions = [Action.RIGHT, Action.LEFT]
    snapshots = replay_fragment(start, actions)
    state = from_rows(BOARD)
    env = environment(state)
    assert snapshots[0] == start
    for action, expected in zip(actions, snapshots[1:], strict=True):
        state = env.next_state(state, action)
        assert expected.observation.rows == tuple(to_rows(state))
    assert snapshots[-1].observation.rows == tuple(MOVED_BACK)
    assert start.observation.rows == tuple(BOARD)
    assert replay_fragment(start, [Action.RIGHT]) == snapshots[:2]


def test_checked_loader_rejects_forged_label_and_keeps_legacy_reader(tmp_path: Path) -> None:
    source = _source(tmp_path, _trajectory("demo", "train", BOARD, [Action.RIGHT]))
    output = tmp_path / "out"
    prepare_annotations(source, output, max_offset=1, seed=21)
    checked = output / "D_A.jsonl"
    row, = read_records(checked, split="train")

    forged = tmp_path / "forged.jsonl"
    forged.write_text(json.dumps({**row, "intent": {**WALL_PREDICATE, "instruction": None}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="intent|wall"):
        read_records(forged, split="train")

    legacy = tmp_path / "legacy.jsonl"
    old = {key: row[key] for key in ("record_id", "start_observation", "intent", "task_goal", "split")}
    legacy.write_text(json.dumps(old) + "\n", encoding="utf-8")
    assert read_records(legacy, split="train") == [old]


def test_checked_loader_checks_held_out_layouts_before_filtering(tmp_path: Path) -> None:
    train = tmp_path / "train"
    held_out = tmp_path / "test"
    source = _source(tmp_path, _trajectory("train-demo", "train", BOARD, [Action.RIGHT]))
    prepare_annotations(source, train, max_offset=1, seed=21)
    rotated = transform_rows(BOARD, BoardSymmetry.ROTATE_90)
    actions = transform_actions([Action.RIGHT], BoardSymmetry.ROTATE_90)
    source = _source(tmp_path, _trajectory("test-demo", "test", rotated, actions))
    prepare_annotations(source, held_out, max_offset=1, seed=21)
    merged = tmp_path / "merged.jsonl"
    merged.write_bytes((train / "D_A.jsonl").read_bytes() + (held_out / "D_A.jsonl").read_bytes())
    with pytest.raises(ValueError, match="layout"):
        read_records(merged, split="train")


def test_controller_requires_a_proposal_for_held_out_annotations(tmp_path: Path) -> None:
    source = _source(tmp_path, _trajectory("test-demo", "test", BOARD, [Action.RIGHT]))
    output = tmp_path / "out"
    prepare_annotations(source, output, max_offset=1, seed=21)
    record, = read_records(output / "D_A.jsonl", split="test")
    prompts = []

    def decide(prompt: str) -> str:
        prompts.append(json.loads(prompt.split("\n\n")[-1]))
        return '{"action":"RIGHT"}'

    with pytest.raises(ValueError, match="proposal|proposed|test|held"):
        run_record(record, decide, 1)
    assert prompts == []
    proposal = {"effects": [{"predicate": "player_at", "row": 1, "col": 2}], "preserve": []}
    result = run_record(record, decide, 1, proposed_intent=proposal)
    assert result["result"]["status"] == "achieved"
    prompt, = prompts
    assert prompt["intent"] == proposal
    assert prompt["intent"]["effects"] != record["intent"]["effects"]
    assert prompt["state"] == BOARD

    # Legacy support from a separate file may claim an unrelated layout ID.
    support = {
        "split": "train",
        "start_observation": transform_rows(BOARD, BoardSymmetry.ROTATE_90),
        "layout_id": "different-legacy-id",
        "task_goal": TASK_GOAL,
        "intent": record["intent"],
        "validation": {"accepted": True},
    }
    with pytest.raises(ValueError, match="share the query layout"):
        in_context_messages(record, [support])
    with pytest.raises(ValueError, match="distinct layout"):
        messages_for(record, [support], "json")
