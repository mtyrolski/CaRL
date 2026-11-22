"""Fixtures for policy-relative generator tests."""

import copy
import json

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows

BOARDS = [
    ["######", "#@-$-#", "#---.#", "######"],
    ["#######", "#@-$--#", "#----.#", "#######"],
    ["######", "#@-$-#", "#-.--#", "######"],
]
CONTROLLER = "a" * 64


class FakeTokenizer:
    eos_token = "<end>"

    def __init__(self):
        self.messages = []
        self.answers = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        return [10, 11, 12]

    def encode(self, answer, **kwargs):
        self.answers.append(answer)
        assert kwargs["add_special_tokens"] is False
        return [20 + index for index in range(len(answer))]


def attempt(status, count):
    return {"status": status, "primitive_actions_attempted": count}


def measured(
    identifier,
    context,
    split,
    board,
    predicate,
    attempts,
    fraction,
    duration,
    u,
    weight,
    size,
):
    return {
        "record_id": identifier,
        "context_id": context,
        "layout_id": canonical_layout_id(from_rows(board)),
        "split": split,
        "family": "synthetic_unit_test",
        "start_observation": board,
        "task_goal": "Put boxes on goals.",
        "intent": {"effects": [predicate], "preserve": [], "instruction": None},
        "validation": {"accepted": True},
        "later_observation": ["SECRET_FUTURE"],
        "k": 987654321,
        "fragment_actions": [87654321],
        "source_path": "SECRET_PATH",
        "trajectory_id": "SECRET_TRAJECTORY",
        "policy_measurements": {
            "controller_hash": CONTROLLER,
            "action_cap": 8,
            "expected_candidates": size,
            "complete": True,
            "attempts": attempts,
            "success_fraction": fraction,
            "mean_success_actions": duration,
            "u": u,
            "weight": weight,
        },
    }


def group(prefix, split, board):
    context = prefix + "-context"
    return [
        measured(
            prefix + "-fast",
            context,
            split,
            board,
            {"predicate": "box_at", "row": 1, "col": 4},
            [attempt("achieved", 1), attempt("action_limit", 8)],
            0.5,
            1,
            0.125,
            0.25,
            4,
        ),
        measured(
            prefix + "-tie-heavy",
            context,
            split,
            board,
            {"predicate": "box_at", "row": 2, "col": 1},
            [attempt("achieved", 2), attempt("achieved", 2)],
            1,
            2,
            0.625,
            0.5,
            4,
        ),
        measured(
            prefix + "-tie-light",
            context,
            split,
            board,
            {"predicate": "player_at", "row": 2, "col": 2},
            [attempt("achieved", 2), attempt("invalid_response", 3)],
            0.5,
            2,
            0.625,
            0.25,
            4,
        ),
        measured(
            prefix + "-failed",
            context,
            split,
            board,
            {"predicate": "box_at", "row": 2, "col": 3},
            [attempt("action_limit", 8), attempt("action_limit", 8)],
            0,
            None,
            None,
            None,
            4,
        ),
    ]


def make_dataset(tmp_path):
    rows = group("train-a", "train", BOARDS[0]) + group("train-b", "train", BOARDS[1])
    rows += group("validation", "validation", BOARDS[2])
    path = tmp_path / "calibration.jsonl"
    write_rows(path, rows)
    return path, rows


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
