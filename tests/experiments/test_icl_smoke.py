"""Exercise the public Sokoban quickstart and its saved sweep design."""

import json
from collections import Counter
from pathlib import Path

import pytest

from carl.icl.config import ROOT
from carl.icl.evaluation.matched import MatchedSweep
from carl.icl.types import JSONObject
from scripts.run_icl_smoke import DEFAULT_CONFIG, main


def _read(path: Path) -> JSONObject:
    return json.loads(path.read_text(encoding="utf-8"))


def _arguments(output: Path) -> list[str]:
    return ["--config", str(ROOT / DEFAULT_CONFIG), "--output", str(output)]


def test_clean_fixture_produces_paired_trials_with_fixed_support(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    output = tmp_path / "first"
    assert main([*_arguments(output), "--profile", "four-shot"]) == 0
    sweep = MatchedSweep.model_validate_json((output / "sweep.json").read_bytes())
    assert len(sweep.trials) == 24
    assert sorted(Counter(instance.group for instance in sweep.instances).values()) == [2, 2]
    assert {instance.seed for instance in sweep.instances} == {73, 101}
    assert len({instance.policy_fingerprint for instance in sweep.instances}) == 1
    assert sweep.offline_compute_seconds is None and sweep.adaptation_seconds is None
    assert all(trial.accounting["physical_requests"] == 0 and trial.cost_usd == 0.0 for trial in sweep.trials)
    assert all((output / trial.artifact).is_file() for trial in sweep.trials)
    successes = Counter((trial.budget_id, trial.mode) for trial in sweep.trials if trial.solved)
    assert successes == {
        ("two-actions", "direct"): 4, ("two-actions", "one-step"): 2, ("two-actions", "full-search"): 4,
        ("four-actions", "direct"): 4, ("four-actions", "one-step"): 4, ("four-actions", "full-search"): 4,
    }
    assert {trial.finish_reason for trial in sweep.trials if not trial.solved} == {"action_budget"}
    assert _read(output / "annotations" / "manifest.json")["accepted"] == 5
    assert {role["shots"] for role in _read(output / "profile" / "preview.json")["components"].values()} == {4}


def test_override_preserves_prior_results_and_starts_in_a_fresh_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "existing"
    command = [*_arguments(output), "--prepare-only"]
    assert main(command) == 0
    paths = ("frozen-config.json", "annotations/D_A.jsonl", "profile/held-out.json")
    original = {path: (output / path).read_bytes() for path in paths}
    with pytest.raises(SystemExit) as error:
        main([*command, "--profile", "four-shot"])
    assert error.value.code == 1
    assert "select a fresh output" in capsys.readouterr().err
    assert {path: (output / path).read_bytes() for path in paths} == original

    rerun = tmp_path / "override"
    assert main([*_arguments(rerun), "--prepare-only", "--profile", "four-shot"]) == 0
    assert (rerun / "annotations/D_A.jsonl").read_bytes() == original["annotations/D_A.jsonl"]
    assert _read(output / "frozen-config.json")["fingerprint"] != _read(rerun / "frozen-config.json")["fingerprint"]
    assert {role["shots"] for role in _read(output / "profile/preview.json")["components"].values()} == {0}
    assert {role["shots"] for role in _read(rerun / "profile/preview.json")["components"].values()} == {4}


def test_invalid_sweep_schema_fails_before_creating_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad_config = tmp_path / "bad.json"
    config = _read(ROOT / DEFAULT_CONFIG)
    config["budgets"][1]["action_budget"] = config["budgets"][0]["action_budget"]
    bad_config.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "bad-run"
    with pytest.raises(SystemExit) as error:
        main(["--config", str(bad_config), "--output", str(output)])
    assert error.value.code == 1
    assert "action budgets must be distinct" in capsys.readouterr().err
    assert not output.exists()
