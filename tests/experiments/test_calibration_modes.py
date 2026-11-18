"""Matched generator conditions reuse saved weighted demonstration identities."""
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.config import ExperimentConfig
from carl.icl.config import load_config
from carl.icl.datasets import environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.types import BoardEnvironment
from carl.icl.types import TrainingData

METHODS = ("relative_u", "absolute_d", "ratio_d_over_ac", "normalized_offset")
FIELDS = {"relative_u": "u", "absolute_d": "d", "ratio_d_over_ac": "ratio_d_over_ac",
          "normalized_offset": "normalized_offset"}


def fixture_data() -> tuple[np.ndarray, TrainingData, list[dict[str, Any]]]:
    board = from_rows(["#####", "#@--#", "#-$-#", "#-.-#", "#####"])
    env = environment(board)
    targets = [to_rows(np.asarray(env.next_state(board, action))) for action in (3, 1)]
    contexts = [{"id": context, "state": to_rows(board), "remaining_actions": 3,
                 "calibration_indices": [0, 1], "candidates": [
                     {"k": 1, "actions": [3], "subgoal": targets[0]},
                     {"k": 2, "actions": [1, 3], "subgoal": targets[1]}]}
                for context in ("first", "second")]
    training = cast(TrainingData, {"contexts": contexts})
    examples = [dict(candidate, state=context["state"], context=context["id"], target_index=index,
                     weight=[0.1, 0.9][index], u=[0.05, 0.55][index], d=[1.0, 2.0][index],
                     a_c=2.0, ratio_d_over_ac=[0.5, 1.0][index], normalized_offset=[1 / 16, 2 / 16][index])
                for context in contexts for index, candidate in enumerate(context["candidates"])]
    return board, training, examples


def components(tmp_path: Path, method: str, *, denominator: float = 16.0) -> tuple[LunaComponents, np.ndarray]:
    board, training, examples = fixture_data()
    scales = [0.9, 0.5] if method == "relative_u" else [2.0, 1.0]
    config = ExperimentConfig.model_validate({"method": method, "scales": scales, "few_shot": 2,
                                              "offset_denominator": denominator})
    api = ExperimentAPI(tmp_path, "mock", config.limits)
    agent = LunaComponents(config, api, environment(board), training)
    agent.calibrated = {"status": "completed", "provider": "mock", "policy_fingerprint": agent.fingerprint(),
                        "examples": examples, "selected_example_ids": [
                            {"context": "second", "target_index": 0},
                            {"context": "first", "target_index": 1}]}
    return agent, board


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_matched_modes_use_saved_ids_and_change_only_label_and_instruction(tmp_path: Path) -> None:
    questions = []
    answers = []
    selected = []
    for method in METHODS:
        agent, board = components(tmp_path, method)
        selected.append([(row["context"], row.get("target_index")) for row in agent.generator_examples(board, 0.5, False)])
        payload = agent.generator_payload(board, 0.5, False)
        questions.append([json.loads(payload["input"][index]["content"][0]["text"]) for index in (1, 3, 5)])
        answers.append([payload["input"][index]["content"][0]["text"] for index in (2, 4)])
    assert all(ids == [("second", 0), ("first", 1)] for ids in selected)
    assert all(answer == answers[0] for answer in answers)
    stripped = [[{k: v for k, v in question.items() if k != FIELDS[method]} for question in values]
                for method, values in zip(METHODS, questions, strict=True)]
    assert all(value == stripped[0] for value in stripped)
    assert [question[FIELDS[method]] for method, question in zip(METHODS, (v[0] for v in questions), strict=True)] == [
        0.05, 1.0, 0.5, 1 / 16]


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_offset_denominator_is_fixed_config_not_a_test_board_statistic(tmp_path: Path) -> None:
    first, board = components(tmp_path, "normalized_offset", denominator=16)
    second, _ = components(tmp_path, "normalized_offset", denominator=8)
    class NoSimulator:
        def next_state(self, *_args: Any) -> Any:
            pytest.fail("Generator payload must not probe the test board")
        def is_solved(self, *_args: Any) -> bool:
            pytest.fail("Generator payload must not reconstruct the test-board reference")
    first.env = cast(BoardEnvironment, NoSimulator())
    a = first.generator_payload(board, 0.5, False)
    b = second.generator_payload(board, 0.5, False)
    a_example = json.loads(a["input"][1]["content"][0]["text"])
    b_example = json.loads(b["input"][1]["content"][0]["text"])
    assert a_example["normalized_offset"] == 1 / 16
    assert b_example["normalized_offset"] == 1 / 8
    query = json.loads(a["input"][-1]["content"][0]["text"])
    assert set(query) == {"state", "goal", "normalized_offset", "num_subgoals"}
    assert "16" in a["input"][0]["content"][0]["text"]


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_fixed_k_keeps_trajectory_offset_and_needs_no_calibration(tmp_path: Path) -> None:
    board, training, _ = fixture_data()
    config = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=2)
    agent = LunaComponents(config, ExperimentAPI(tmp_path, "mock", config.limits), environment(board), training)
    payload = agent.generator_payload(board, 1.0, False)
    assert json.loads(payload["input"][1]["content"][0]["text"])["k"] == 1
    assert "k" in json.loads(payload["input"][-1]["content"][0]["text"])


def test_campaign_mode_configs_share_frozen_policy_and_calibration() -> None:
    configs = [load_config(f"configs/icl/campaign_data/{method}.json") for method in METHODS]
    assert len({config.locked_policy_fingerprint for config in configs}) == 1
    assert len({config.calibration.model_dump_json() for config in configs}) == 1
    assert all(config.data_dir == "data/icl/campaign_dataset_v2" for config in configs)


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_saved_selection_cannot_reference_missing_or_duplicate_targets(tmp_path: Path) -> None:
    agent, board = components(tmp_path, "relative_u")
    assert agent.calibrated is not None
    agent.calibrated["selected_example_ids"] = [{"context": "first", "target_index": 999}]
    with pytest.raises(ValueError, match="missing from retained"):
        agent.generator_examples(board, 0.5, False)
    agent.calibrated["selected_example_ids"] = [
        {"context": "first", "target_index": 0}, {"context": "first", "target_index": 1}]
    with pytest.raises(ValueError, match="duplicate"):
        agent.generator_examples(board, 0.5, False)
