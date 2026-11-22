"""Independent budget estimates and restart-safe sweep orchestration."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from carl.icl.api import write_json
from carl.icl.budget_sweep import (
    BudgetSweepConfig,
    describe_plan,
    load_sweep,
    run_sweep,
)
from carl.icl.config import ExperimentConfig
from carl.icl.evaluation.budget_sweep import actual_budget_figure, actual_budget_results
from carl.icl.evaluation.data import EvaluationConfig
from carl.icl.experiment import run_condition
from carl.icl.search import SearchResult
from scripts import run_budget_sweep
from tests.experiments.test_icl_evaluation import saved_run


def test_experiment_plan_has_matched_conditions_and_real_budget_grid():
    spec, configs = load_sweep("configs/icl/budget_sweeps/pilot.json")
    plan = describe_plan(spec, configs)
    assert plan["budgets"] == [16, 32, 64, 128, 256]
    assert plan["solver_runs"] == 240
    assert {config.method for config in configs} == {"relative", "fixed"}
    assert all(config.instances == 8 and not config.easy for config in configs)
    quick_spec, quick_configs = load_sweep("configs/icl/budget_sweeps/quick.json")
    assert describe_plan(quick_spec, quick_configs)["solver_runs"] == 8
    assert all(config.few_shot == 4 and not config.easy for config in quick_configs)
    with pytest.raises(ValueError, match="strictly increasing"):
        BudgetSweepConfig(matrix="m.json", budgets=[8, 4], output_dir="out")


def test_actual_budget_points_preserve_outcomes_coverage_and_nonmonotonicity(tmp_path):
    runs = [
        saved_run(tmp_path, budget=16, actions=8, solved=True),
        saved_run(tmp_path, budget=16, actions=16, solved=False, instance="board_b"),
        saved_run(tmp_path, budget=64, actions=64, solved=False),
        saved_run(tmp_path, budget=64, actions=64, solved=False, instance="board_b"),
        saved_run(tmp_path, budget=16, actions=16, solved=False, name="fixed"),
        saved_run(tmp_path, budget=64, actions=7, solved=True, name="fixed"),
    ]
    config = EvaluationConfig(run_dirs=[str(tmp_path)], bootstrap_replicates=100)
    rows = actual_budget_results(runs, config)
    relative = [row for row in rows if row["method"] == "relative"]
    assert [row["budget"] for row in relative] == [16, 64]
    assert [row["completion"] for row in relative] == [0.5, 0.0]
    assert relative[0]["runs"] == 2 and relative[0]["expected_runs"] == 8
    assert relative[0]["missing_runs"] == 6 and relative[1]["api_budget_stops"] == 2
    figure = actual_budget_figure(rows)
    traces = figure.to_plotly_json()["data"]
    assert len(traces) == 2
    assert all(
        symbol.endswith("-open")
        for trace in traces
        for symbol in trace["marker"]["symbol"]
    )
    assert actual_budget_results(runs[:2], config) == []


def test_different_budgets_do_not_merge_incompatible_policy_or_provider(tmp_path):
    runs = [
        saved_run(tmp_path, budget=16, actions=8),
        saved_run(tmp_path, budget=64, actions=8),
    ]
    runs[1].detail["frozen_policy"] = "another-policy"
    assert actual_budget_results(runs, EvaluationConfig(run_dirs=[str(tmp_path)])) == []


def test_missing_condition_is_visible_in_other_methods_points(tmp_path):
    runs = [
        saved_run(tmp_path, budget=16, actions=8),
        saved_run(tmp_path, budget=16, actions=8, name="fixed"),
        saved_run(tmp_path, budget=64, actions=8),
    ]
    rows = actual_budget_results(runs, EvaluationConfig(run_dirs=[str(tmp_path)]))
    at_64 = next(row for row in rows if row["budget"] == 64)
    assert at_64["missing_conditions"] == ["fixed"]


def test_unstarted_budget_and_method_are_taken_from_saved_plan(tmp_path):
    write_json(
        tmp_path / "sweep_plan.json",
        {
            "budgets": [16, 64],
            "conditions": [
                {"name": "relative", "family": "real_12x12x4"},
                {"name": "fixed", "family": "real_12x12x4"},
            ],
        },
    )
    rows = actual_budget_results(
        [saved_run(tmp_path, budget=16, actions=8)],
        EvaluationConfig(run_dirs=[str(tmp_path)]),
    )
    assert len(rows) == 1
    assert rows[0]["missing_conditions"] == ["fixed"]
    assert rows[0]["unstarted_budgets"] == [64]


def test_concurrent_sweep_lock_is_rejected(tmp_path):
    from carl.icl.budget_sweep import _sweep_lock

    with _sweep_lock(tmp_path):
        with pytest.raises(ValueError, match="Another process"):
            with _sweep_lock(tmp_path):
                pytest.fail("Concurrent writer acquired the sweep lock")


def test_resume_skips_saved_trials_and_recovers_atomic_orphan(tmp_path, monkeypatch):
    import carl.icl.experiment as experiment

    config = ExperimentConfig(
        name="fixed",
        method="fixed",
        scales=[8, 4],
        instances=2,
        repeat_seeds=[11],
        action_budget=8,
    )
    training = {"contexts": []}
    instances = [
        {
            "id": name,
            "state": ["#####", "#---#", "#@$.#", "#---#", "#####"],
            "level_key": name,
        }
        for name in ("a", "b")
    ]
    component = SimpleNamespace(
        env=object(),
        api=SimpleNamespace(metrics={}),
        calibrated=None,
        fingerprint=lambda: "frozen",
        diagnostics={},
    )
    calls = []
    monkeypatch.setattr(experiment, "credentials", lambda provider: None)
    monkeypatch.setattr(experiment, "load_data", lambda _config: (training, instances))
    monkeypatch.setattr(experiment, "components_for_run", lambda *_args: component)
    monkeypatch.setattr(
        experiment,
        "search",
        lambda *_args: calls.append(1)
        or SearchResult(finish_reason="action_budget", attempted_actions=8),
    )
    rows = run_condition(config, "mock", tmp_path, resume=True)
    assert len(rows) == len(calls) == 2
    write_json(tmp_path / "results.json", rows[:1])
    restored = run_condition(config, "mock", tmp_path, resume=True)
    assert restored == rows and len(calls) == 2
    assert json.loads((tmp_path / "results.json").read_text()) == rows
    changed = ExperimentConfig.model_validate(
        {**config.model_dump(), "action_budget": 9}
    )
    with pytest.raises(ValueError, match="changed condition"):
        run_condition(changed, "mock", tmp_path, resume=True)
    with pytest.raises(ValueError, match="another provider"):
        run_condition(config, "replay", tmp_path, resume=True)


def test_api_error_stops_and_does_not_silently_retry(tmp_path, monkeypatch):
    import carl.icl.experiment as experiment

    config = ExperimentConfig(
        name="fixed",
        method="fixed",
        scales=[8, 4],
        instances=1,
        repeat_seeds=[11],
        action_budget=8,
    )
    instance = {
        "id": "a",
        "state": ["#####", "#---#", "#@$.#", "#---#", "#####"],
        "level_key": "a",
    }
    component = SimpleNamespace(
        env=object(),
        api=SimpleNamespace(metrics={}),
        calibrated=None,
        fingerprint=lambda: "frozen",
        diagnostics={},
    )
    monkeypatch.setattr(experiment, "credentials", lambda provider: None)
    monkeypatch.setattr(
        experiment, "load_data", lambda _config: ({"contexts": []}, [instance])
    )
    monkeypatch.setattr(experiment, "components_for_run", lambda *_args: component)
    monkeypatch.setattr(
        experiment, "search", lambda *_args: SearchResult(finish_reason="api_error")
    )
    with pytest.raises(ValueError, match="sweep stopped"):
        run_condition(config, "mock", tmp_path, resume=True, stop_on_api_error=True)
    assert len(json.loads((tmp_path / "results.json").read_text())) == 1
    with pytest.raises(ValueError, match="Saved API error"):
        run_condition(config, "mock", tmp_path, resume=True, stop_on_api_error=True)


def test_sweep_never_overwrites_or_resumes_changed_contract(tmp_path, monkeypatch):
    import carl.icl.budget_sweep as sweep

    spec = BudgetSweepConfig(matrix="unused", budgets=[4, 8], output_dir=str(tmp_path))
    configs = [ExperimentConfig(method="fixed", scales=[8, 4])]
    monkeypatch.setattr(
        sweep, "_manifest", lambda *_args: {"provider": "mock", "version": 1}
    )
    monkeypatch.setattr(sweep, "credentials", lambda *_args: None)
    monkeypatch.setattr(sweep, "run_condition", lambda *_args, **_kwargs: [])
    run_sweep(spec, configs, "mock", tmp_path, resume=False)
    assert json.loads((tmp_path / "sweep_plan.json").read_text())["output_dir"] == str(
        tmp_path
    )
    with pytest.raises(ValueError, match="not empty"):
        run_sweep(spec, configs, "mock", tmp_path, resume=False)
    monkeypatch.setattr(
        sweep, "_manifest", lambda *_args: {"provider": "replay", "version": 1}
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        run_sweep(spec, configs, "replay", tmp_path, resume=True)


def test_cache_dir_override_reaches_sweep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = ExperimentConfig(method="fixed", scales=[2, 1])
    spec = BudgetSweepConfig(matrix="unused", budgets=[1, 2], output_dir=str(tmp_path))
    seen = []
    monkeypatch.setattr(run_budget_sweep, "load_sweep", lambda _path: (spec, [config]))
    monkeypatch.setattr(run_budget_sweep, "run_sweep", lambda _spec, items, *_args, **_kwargs: seen.extend(items))
    monkeypatch.setattr(sys, "argv", ["run_budget_sweep", "--replay", "--cache-dir", str(tmp_path / "restored")])
    assert run_budget_sweep.main() == 0
    assert seen[0].cache_dir == str(tmp_path / "restored")
