"""Verify generator snapshots, calibrated provenance and independent example counts."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from carl.icl.boards import from_rows, to_rows
from carl.icl.calibration import calibrate, calibration_result_digest
from carl.icl.checkpoints import export_generator_checkpoint, load_generator_checkpoint
from carl.icl.checkpoints.io import checkpoint_digest
from carl.icl.checkpoints.types import GeneratorCheckpoint
from carl.icl.config import Calibration, ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment import components_for_run
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.types import Board, Candidate, JSONObject, Probabilities, TrainingData


@dataclass
class CheckpointCase:
    config: ExperimentConfig
    board: Board
    training: TrainingData
    calibration: JSONObject
    fingerprint: str
    source_hashes: dict[str, str]
    path: Path

    def export(self) -> GeneratorCheckpoint:
        return export_generator_checkpoint(
            self.path, config=self.config, policy_fingerprint=self.fingerprint,
            calibration=self.calibration, source_hashes=self.source_hashes,
        )


@pytest.fixture
def checkpoint_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CheckpointCase:
    state = from_rows(["#####", "#---#", "#-$-#", "#@.-#", "#####"])
    env = environment(state)
    candidates: list[Candidate] = [{"k": 1, "actions": [action], "subgoal": to_rows(env.next_state(state, action))}
                  for action in (0, 3)]
    training: TrainingData = {"contexts": [{"id": "train-0", "state": to_rows(state), "remaining_actions": 4,
        "candidates": candidates, "calibration_indices": [0, 1]}]}
    config = ExperimentConfig(few_shot=1, train_boards=1, cache_dir=str(tmp_path / "cache"),
        calibration=Calibration(reference_window=2, candidates=2, trials=2, action_cap=2))
    components = LunaComponents(config, ExperimentAPI(config.path("cache_dir"), "mock", config.limits), env, training)

    def probabilities(_state: Board, target: Board) -> Probabilities:
        action = 0 if to_rows(target) == candidates[0]["subgoal"] else 3
        result = np.zeros(4, dtype=np.float64)
        result[action] = 1
        return result

    monkeypatch.setattr(components, "probabilities", probabilities)
    summary = calibrate(config, components, tmp_path / "calibration")
    assert summary["status"] == "completed" and len(summary["examples"]) == 2
    summary["selected_example_ids"] = [{"context": row["context"], "target_index": row["target_index"]}
                                        for row in summary["examples"]]
    summary["result_digest"] = calibration_result_digest(summary)
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(training))
    trials_path = tmp_path / "calibration" / summary["signature"] / "calibration_trials.json"
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (training_path, trials_path)}
    return CheckpointCase(config, state, training, summary, components.fingerprint(), source_hashes, tmp_path / "checkpoint.json")


def test_reload_is_immutable_and_keeps_multiple_targets_per_context(checkpoint_case: CheckpointCase) -> None:
    case = checkpoint_case
    original = case.export()
    restored = load_generator_checkpoint(case.path, verify_sources=True)
    assert restored == original and restored.actual_example_count == 2
    assert [row.context for row in restored.examples] == ["train-0", "train-0"]
    assert [row.target_index for row in restored.examples] == [0, 1]
    with pytest.raises(ValidationError, match="frozen"):
        setattr(restored.examples[0], "u", 0.1)
    copy = restored.examples[0].as_example()
    copy["state"][0] = "corrupted"
    assert restored.examples[0].state[0] == "#####"
    assert case.export() == original


@pytest.mark.parametrize("expected", [
    {"model": "different-model"}, {"policy_fingerprint": "f" * 64}, {"action_cap": 3}, {"provider": "openai"},
])
def test_reload_rejects_incompatible_contract(checkpoint_case: CheckpointCase, expected: JSONObject) -> None:
    checkpoint_case.export()
    with pytest.raises(ValueError, match="mismatch"):
        load_generator_checkpoint(checkpoint_case.path, **expected)


def test_corruption_and_recomputed_forgery_fail(checkpoint_case: CheckpointCase) -> None:
    case = checkpoint_case
    checkpoint = case.export()
    raw = checkpoint.model_dump(mode="json")
    raw["examples"][0]["u"] = 0.2
    case.path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="content digest"):
        load_generator_checkpoint(case.path)
    forged = GeneratorCheckpoint.model_validate(raw)
    raw["sha256"] = checkpoint_digest(forged)
    case.path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="differs from calibration evidence"):
        load_generator_checkpoint(case.path)


def test_corrupted_evidence_and_source_hashes_fail(checkpoint_case: CheckpointCase) -> None:
    case = checkpoint_case
    checkpoint = case.export()
    source = Path(next(iter(case.source_hashes)))
    source.write_text("changed source")
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_generator_checkpoint(case.path, verify_sources=True)
    with pytest.raises(ValueError, match="source hash mismatch"):
        case.export()
    evidence = case.path.parent / checkpoint.calibration_evidence.path
    evidence.write_text("changed evidence")
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        load_generator_checkpoint(case.path)


@pytest.mark.parametrize("mutation", ["copy", "duplicate", "labels", "witness", "incomplete"])
def test_export_rejects_invalid_calibration(checkpoint_case: CheckpointCase, mutation: str) -> None:
    case = checkpoint_case
    rows = case.calibration["examples"]
    if mutation == "copy":
        rows[0]["subgoal"] = rows[0]["state"]
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "labels":
        rows[0]["u"] = 0.2
    elif mutation == "witness":
        rows[0]["actions"] = [2]
    else:
        rows[0]["trials"] = 1
    case.calibration["result_digest"] = calibration_result_digest(case.calibration)
    with pytest.raises(ValueError):
        case.export()
    assert not case.path.exists()


def test_checkpoint_cannot_be_overwritten_with_other_defaults(checkpoint_case: CheckpointCase) -> None:
    case = checkpoint_case
    case.export()
    saved = case.path.read_bytes()
    case.config = case.config.model_copy(update={"generator_tokens": 512})
    with pytest.raises(ValueError, match="overwrite immutable"):
        case.export()
    assert case.path.read_bytes() == saved


def test_counts_are_independent_and_labels_match_across_methods(checkpoint_case: CheckpointCase) -> None:
    case = checkpoint_case
    checkpoint = case.export()
    original = LunaComponents(case.config, ExperimentAPI(case.config.path("cache_dir"), "mock", case.config.limits),
                              environment(case.board), case.training)
    for method, scales, field in (("relative_u", [0.8, 0.2], "u"), ("absolute_d", [2, 1], "d"),
                                  ("ratio_d_over_ac", [2, 1], "ratio_d_over_ac"), ("fixed", [2, 1], "k")):
        config = ExperimentConfig.model_validate({**case.config.model_dump(), "method": method, "scales": scales,
                                                 "generator_checkpoint": str(case.path)})
        components = components_for_run(config, "mock", case.training, case.board)
        assert components.fingerprint() == case.fingerprint
        assert components.policy_payload(case.board, case.board) == original.policy_payload(case.board, case.board)
        examples = components.generator_examples(case.board, scales[0], False)
        assert examples == [row.as_example() for row in checkpoint.examples]
        payload = components.generator_payload(case.board, scales[0])
        questions = [json.loads(message["content"][0]["text"]) for message in payload["input"][1:-1:2]]
        assert len(questions) == 2
        assert [question[field] for question in questions] == [getattr(row, field) for row in checkpoint.examples]
    config = ExperimentConfig.model_validate({**case.config.model_dump(), "generator_checkpoint": str(case.path),
                                             "generator_example_limit": 1,
                                             "prompts": {**case.config.prompts.model_dump(), "generator": "Explicit variant"}})
    components = components_for_run(config, "mock", case.training, case.board)
    assert len(components.generator_examples(case.board, 0.8, False)) == 1
    assert components.fingerprint() == case.fingerprint
    assert components.policy_payload(case.board, case.board) == original.policy_payload(case.board, case.board)
    assert "Explicit variant" in components.generator_payload(case.board, 0.8)["input"][0]["content"][0]["text"]


def test_limit_overflow_and_missing_checkpoint_fail(checkpoint_case: CheckpointCase) -> None:
    case = checkpoint_case
    case.export()
    with pytest.raises(ValueError, match="requires a generator_checkpoint"):
        ExperimentConfig(generator_example_limit=1)
    config = case.config.model_copy(update={"generator_checkpoint": str(case.path), "generator_example_limit": 3})
    with pytest.raises(ValueError, match="fewer than requested"):
        components_for_run(config, "mock", case.training, case.board)
