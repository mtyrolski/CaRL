"""Probe denominators and calibration labels retain failed observations."""

import json
import copy
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from carl.icl.boards import from_rows
from carl.icl.datasets import environment
from carl.icl.luna import LunaComponents
from carl.icl.types import JSONObject
from scripts.generator_probe.calibrate import aggregate
from scripts.generator_probe.cases import classify, execute_target
from scripts.generator_probe.integrity import trial_seed, verify_target


def test_classification_keeps_source_copy_and_invalid_board() -> None:
    rows = ["#####", "#@$.#", "#####"]
    source = from_rows(rows)
    calls = [{"response": {"output_text": json.dumps({"subgoals": [
        {"rows": rows}, {"rows": ["#####", "#@@.#", "#####"]}
    ]})}}, {"response": {"output_text": "{broken"}},
        {"response": {"output_text": '{"subgoals": []}'}}]
    observations = classify(calls, source, False)
    assert len(observations) == 3
    assert observations[0]["candidates"][0]["source_copy"]
    assert not observations[0]["candidates"][1]["static_valid"]
    assert not observations[1]["parse_ok"]
    assert observations[2]["empty"]


def test_calibration_uses_successful_execution_times_not_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.generator_probe.calibrate.costs", lambda: {})
    source = ["######", "#@-$.#", "######"]
    target = ["######", "#-@$.#", "######"]
    context = {"id": "train", "state": source,
               "candidates": [{"k": k, "subgoal": target, "actions": [3] * k} for k in (1, 2, 4)]}
    records = [
        [{"exact_hit": True, "first_hit": 3}, {"exact_hit": True, "first_hit": 5}],
        [{"exact_hit": True, "first_hit": 6}, {"exact_hit": False, "first_hit": None}],
        [{"exact_hit": False, "first_hit": None}, {"exact_hit": False, "first_hit": None}],
    ]
    rows: list[JSONObject] = [{"context": "train", "target_index": index, "records": trials, "complete": True}
                              for index, trials in enumerate(records)]
    result = aggregate([context], rows, {"provider": "openai", "policy_fingerprint": "a" * 64})
    assert result["status"] == "completed"
    assert len(result["examples"]) == 2
    first, second = result["examples"]
    assert first["d"] == 4 and first["k"] == 1
    assert first["u"] == pytest.approx(1 / 3)
    assert second["u"] == pytest.approx(5 / 6)
    assert first["a_c"] == 4
    assert result["targets"][2]["excluded_reason"] == "zero_exact_hits"
    assert result["targets"][2]["p_hat"] == 0


def test_recorded_progress_and_saved_trial_replay_reject_stale_seed() -> None:
    rows = ["#######", "#@--$.#", *(["#######"] * 5)]
    target_rows = ["#######", "#--@$.#", *(["#######"] * 5)]
    source, target = from_rows(rows), from_rows(target_rows)
    components = cast(LunaComponents, SimpleNamespace(
        env=environment(source), config=SimpleNamespace(calibration=SimpleNamespace(action_cap=8)),
        probabilities=lambda _state, _target: np.array([0.0, 0.0, 0.0, 1.0])))
    context = {"id": "train", "state": rows,
               "candidates": [{"k": 2, "subgoal": target_rows, "actions": [3, 3]}]}
    progress: list[JSONObject] = []
    records = []
    for trial in range(2):
        seed = trial_seed("train", 0, trial)
        record = execute_target(components, source, target, seed,
                                lambda value: progress.append(copy.deepcopy(value)))
        record.update(seed=seed, trial=trial, cap=8)
        records.append(record)
    assert [row["actions"] for row in progress] == [1, 2, 1, 2]
    row = {"context": "train", "target_index": 0, "k": 2, "source": rows,
           "target": target_rows, "policy_fingerprint": "a" * 64, "complete": True,
           "error": None, "records": records}
    verify_target(context, 0, row, "a" * 64)
    records[1]["seed"] += 1
    with pytest.raises(ValueError, match="seed"):
        verify_target(context, 0, row, "a" * 64)
