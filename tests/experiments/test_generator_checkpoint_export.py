"""Keep the inspection bundle faithful to production prompts without inference."""
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carl.icl.boards import from_rows, to_rows
from carl.icl.calibration import calibration_result_digest
from carl.icl.checkpoints import export_generator_checkpoint
from carl.icl.config import Calibration, ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.types import Candidate, JSONObject, TrainingData
from scripts.generator_probe import export_checkpoint


def test_bundle_renders_real_messages_and_preserves_order_without_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect generator, policy and value examples through real payload builders."""
    monkeypatch.setattr("carl.icl.api._campaign_ledger", lambda: None)

    def forbidden(*_args: Any, **_kwargs: Any) -> JSONObject:
        raise AssertionError("Offline inspection must never call inference")

    monkeypatch.setattr(ExperimentAPI, "call", forbidden)
    state = from_rows(["#####", "#---#", "#-$-#", "#@.-#", "#####"])
    env = environment(state)
    candidates: list[Candidate] = [{"k": 1, "actions": [action], "subgoal": to_rows(env.next_state(state, action))}
                  for action in (0, 3)]
    training: TrainingData = {"contexts": [{"id": "training-only", "state": to_rows(state),
        "remaining_actions": 4, "candidates": candidates, "calibration_indices": [0, 1]}]}
    config = ExperimentConfig(method="relative_u", few_shot=1, train_boards=1,
                              cache_dir=str(tmp_path / "cache"),
                              calibration=Calibration(action_cap=2, trials=2))
    components = LunaComponents(config, ExperimentAPI(config.path("cache_dir"), "replay", config.limits),
                                env, training)
    examples = [{**candidate, "state": to_rows(state), "context": "training-only", "target_index": index,
                 "u": 0.5, "d": 1.0, "weight": 0.5, "a_c": 1.0, "ratio_d_over_ac": 1.0,
                 "normalized_offset": 1 / 16, "hits": 2, "trials": 2, "success_rate": 1.0, "p_hat": 1.0}
                for index, candidate in enumerate(candidates)]
    calibration: JSONObject = {"status": "completed", "provider": "openai", "signature": "a" * 64,
        "policy_fingerprint": components.fingerprint(), "config": {"action_cap": 2, "trials": 2},
        "examples": examples, "contexts": [],
        "selected_example_ids": [{"context": "training-only", "target_index": index} for index in (1, 0)]}
    calibration["result_digest"] = calibration_result_digest(calibration)
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(training))
    checkpoint_path = tmp_path / "generator.json"
    checkpoint = export_generator_checkpoint(checkpoint_path, config=config,
        policy_fingerprint=components.fingerprint(), calibration=calibration,
        source_hashes={str(training_path): hashlib.sha256(training_path.read_bytes()).hexdigest()})
    monkeypatch.setattr(export_checkpoint, "OUTPUT", tmp_path)
    monkeypatch.setattr(export_checkpoint, "load_data", lambda _config: (training, []))
    (tmp_path / "validation_sources_30.json").write_text(json.dumps({
        "instances": [{"id": "validation-source", "state": to_rows(state)}]}))
    configured = config.model_copy(update={"generator_checkpoint": str(checkpoint_path)})
    bundle = export_checkpoint.prompt_bundle(configured, checkpoint)
    assert bundle["inference_calls"] == 0
    assert len(bundle["generator"]) == 2  # Duplicate small/all counts are collapsed.
    for row in bundle["generator"]:
        payload = row["payload"]
        assert row["source_id"] == "validation-source" and row["example_count"] == 2
        assert len(payload["input"]) == 6
        answers = [json.loads(message["content"][0]["text"]) for message in payload["input"][2:-1:2]]
        assert [answer["subgoals"][0]["rows"] for answer in answers] == [
            list(example.subgoal) for example in checkpoint.examples]
        query = json.loads(payload["input"][-1]["content"][0]["text"])
        assert set(query) == {"state", "goal", "u", "num_subgoals"}
    assert bundle["policy"]["query_context"] == bundle["value"]["query_context"] == "training-only"
    assert len(bundle["policy"]["payload"]["input"]) == len(bundle["value"]["payload"]["input"]) == 4
    markdown = export_checkpoint.inspection_markdown(checkpoint, configured)
    assert markdown.index("1. training-only, target 1") < markdown.index("2. training-only, target 0")
    assert "2 successful demonstrations" in markdown
    assert config.prompts.policy in markdown and config.prompts.generator in markdown
    assert "Reachability alone does not establish" in markdown


def test_export_rejects_changed_prepared_protocol_before_reading_trials(tmp_path: Path) -> None:
    """A changed manifest cannot acquire a seemingly valid exported snapshot."""
    (tmp_path / "manifest.json").write_text(json.dumps({"provider": "openai", "R": 2}))
    with pytest.raises(ValueError, match="signature differs"):
        export_checkpoint.source_hashes(tmp_path / "calibration.json", {"signature": "b" * 64})
