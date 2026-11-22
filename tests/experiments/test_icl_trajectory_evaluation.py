"""Trajectory diagnostics use actual witnesses and never require live credentials."""
import json
from pathlib import Path

import numpy as np
import pytest

from carl.icl.api import CachedOpenAI
from carl.icl.boards import from_rows, level_key
from carl.icl.config import load_config
from carl.icl.datasets import environment, load_data
from scripts.evaluate_trajectory_components import parser, run
from scripts.datasets.prepare_fresh_trajectory_holdout import prepare

ROOT = Path(__file__).resolve().parents[2]


def test_value_demonstrations_are_training_layouts_only():
    if not (ROOT / "data/icl/value_examples.json").exists():
        pytest.skip("Run prepare-value to create the trajectory training examples")
    training, instances = load_data(load_config("configs/icl/experiments/tuned_fixed.json"))
    assert "value_examples" in training
    examples = training["value_examples"]
    contexts = {item["id"]: level_key(from_rows(item["state"])) for item in training["contexts"]}
    assert len(examples) == 4 * len(contexts)
    assert all(level_key(from_rows(item["state"])) == contexts[item["context"]] for item in examples)
    assert not ({level_key(from_rows(item["state"])) for item in examples} &
                {item["level_key"] for item in instances})


@pytest.mark.parametrize("component", ["policy", "generator", "value"])
@pytest.mark.usefixtures("deterministic_token_estimates")
def test_trajectory_diagnostic_runs_from_verified_data(tmp_path, monkeypatch, component):
    if not (ROOT / "data/icl/value_examples.json").exists():
        pytest.skip("Run prepare-value to create the trajectory training examples")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Mock diagnostics must not create an OpenAI client")

    monkeypatch.setattr(CachedOpenAI, "_client", forbidden)
    monkeypatch.chdir(ROOT)
    args = parser().parse_args([
        "--component", component, "--split", "validation", "--indices", "0",
        "--positions", "start", "--horizons", "4", "--max-requests", "20",
        "--output", str(tmp_path / "results"), "--cache-dir", str(tmp_path / "cache"), "--mock",
    ])
    path = run(args)
    report = json.loads(path.read_text())
    assert report["status"] == "completed"
    assert report["provider"] == "mock"
    assert report["summary"]["cases"] == len(report["cases"])
    assert report["metrics"]["logical_requests"] <= 20
    if component == "value":
        terminal = report["cases"][-1]
        assert terminal["terminal"] and terminal["remaining"] == 0 and terminal["value"] == 1.0
        assert report["metrics"]["logical_requests"] == len(report["cases"])
    else:
        row = report["cases"][0]
        board = from_rows(row["state"])
        env = environment(board)
        for name in row["reference_actions"]:
            action = ("UP", "DOWN", "LEFT", "RIGHT").index(name)
            board = np.asarray(env.next_state(board, action))
        np.testing.assert_array_equal(board, from_rows(row["target"]))
        if component == "generator":
            assert row["valid_count"] == 1
            assert "reachability" not in row
            composite = parser().parse_args([
                "--component", "generated_policy", "--generator-report", str(path),
                "--max-requests", "20", "--output", str(tmp_path / "results"),
                "--cache-dir", str(tmp_path / "cache"), "--mock",
            ])
            composite_report = json.loads(run(composite).read_text())
            assert composite_report["summary"]["cases"] == 1
            assert composite_report["generator_provider"] == "mock"


def test_fresh_holdout_excludes_prepared_and_prior_layouts(tmp_path, monkeypatch):
    if not (ROOT / "data/icl/value_examples.json").exists():
        pytest.skip("Run prepare-value to create the trajectory training examples")
    monkeypatch.chdir(ROOT)
    first = tmp_path / "first"
    second = tmp_path / "second"
    ids1 = prepare(first, 1, 31, "configs/icl/experiments/tuned_fixed.json")
    ids2 = prepare(second, 1, 31, "configs/icl/experiments/tuned_fixed.json", (first,))
    assert ids1 != ids2
    first_data = json.loads((first / "real_12x12x4/test.json").read_text())
    second_data = json.loads((second / "real_12x12x4/test.json").read_text())
    assert first_data["instances"][0]["level_key"] != second_data["instances"][0]["level_key"]
    with pytest.raises(ValueError, match="not empty"):
        prepare(first, 1, 31, "configs/icl/experiments/tuned_fixed.json")
