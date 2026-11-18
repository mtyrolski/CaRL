"""Cache location overrides are applied consistently to experiment commands."""
import sys
from pathlib import Path

import pytest

from carl.icl.config import ExperimentConfig
from carl.icl.budget_sweep import BudgetSweepConfig
from scripts import run_budget_sweep, run_icl_search


@pytest.mark.parametrize("command", ["preview", "matrix"])
def test_cache_dir_override_reaches_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str) -> None:
    config = ExperimentConfig(method="fixed", scales=[2, 1])
    seen = []
    monkeypatch.setattr(run_icl_search, "load_config", lambda _path: config)
    monkeypatch.setattr(run_icl_search, "matrix_configs", lambda _path: [config, config])
    monkeypatch.setattr(run_icl_search, "preview", lambda item, _provider: seen.append(item) or tmp_path)
    monkeypatch.setattr(run_icl_search, "run_matrix", lambda items, _provider, _directory: seen.extend(items))
    monkeypatch.setattr(sys, "argv", ["run_icl_search", command, "--replay", "--output", str(tmp_path),
                                      "--cache-dir", str(tmp_path / "restored")])
    assert run_icl_search.main() == 0
    assert len(seen) == (1 if command == "preview" else 2)
    assert all(item.cache_dir == str(tmp_path / "restored") for item in seen)


def test_cache_dir_override_reaches_sweep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = ExperimentConfig(method="fixed", scales=[2, 1])
    spec = BudgetSweepConfig(matrix="unused", budgets=[1, 2], output_dir=str(tmp_path))
    seen = []
    monkeypatch.setattr(run_budget_sweep, "load_sweep", lambda _path: (spec, [config]))
    monkeypatch.setattr(run_budget_sweep, "run_sweep", lambda _spec, items, *_args, **_kwargs: seen.extend(items))
    monkeypatch.setattr(sys, "argv", ["run_budget_sweep", "--replay", "--cache-dir", str(tmp_path / "restored")])
    assert run_budget_sweep.main() == 0
    assert seen[0].cache_dir == str(tmp_path / "restored")
