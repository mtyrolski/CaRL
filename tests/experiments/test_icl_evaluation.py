"""Evaluation semantics: effort budgets, sampling units and artifact isolation."""
import json
from pathlib import Path

import pytest

from carl.icl.api import write_json
from carl.icl.config import ExperimentConfig
from carl.icl.evaluation.data import EvaluationConfig, RunRecord, SavedRun, load_runs
from carl.icl.evaluation.figures import cohort_figures
from carl.icl.evaluation.report import build_tables, evaluate
from carl.icl.evaluation.statistics import budget_curves, groups, paired, summarize
from carl.icl.types import Provider

BOARD = ["#####", "#---#", "#@$.#", "#---#", "#####"]


def saved_run(
    tmp_path: Path,
    *,
    name="relative",
    instance="board_a",
    seed=0,
    solved=True,
    actions=7,
    provider: Provider = "mock",
    easy=False,
    budget=100,
):
    method = "fixed" if name.startswith("fixed") else "relative"
    settings = ExperimentConfig(
        name=name,
        method=method,
        scales=[8, 4] if method == "fixed" else [0.9, 0.5],
        repeat_seeds=[0, 1, 2, 3],
        instances=2,
        action_budget=budget,
        easy=easy,
    )
    record = RunRecord(
        config=name,
        method=method,
        family=settings.family,
        split="test",
        instance=instance,
        seed=seed,
        provider=provider,
        tuning_status="pilot_unvalidated",
        solved=solved,
        finish_reason="solved" if solved else "api_budget",
        attempted_actions=actions,
        solution_actions=1 if solved else None,
        expansions=2,
        seconds=0.5,
        requests=1,
        response_requests=1,
        embedding_requests=0,
        cache_hits=2,
        input_tokens=100,
        output_tokens=20,
        cached_input_tokens=0,
        cache_write_tokens=0,
        api_seconds=0.2,
        logical_requests=3,
        logical_input_tokens=300,
        logical_output_tokens=60,
        generator_calls=1,
        policy_calls=1,
        value_calls=1,
    )
    detail = {
        "solved": solved,
        "finish_reason": record.finish_reason,
        "attempted_actions": actions,
        "instance": {"id": instance, "state": BOARD},
        "frozen_policy": "same-frozen-policy",
        "events": [],
        "action_path": [3] if solved else [],
    }
    return SavedRun(record, settings, tmp_path / name, detail)


def export_run(run):
    write_json(run.directory / "config.json", run.settings.model_dump())
    write_json(run.directory / "results.json", [run.record.model_dump()])
    write_json(
        run.directory / f"{run.record.instance}_seed{run.record.seed}.json", run.detail
    )


def test_budget_curve_uses_search_effort_not_solution_length(tmp_path):
    runs = [
        saved_run(tmp_path, actions=70),
        saved_run(tmp_path, instance="board_b", solved=False, actions=100),
    ]
    config = EvaluationConfig(run_dirs=[str(tmp_path)], budget_points=101)
    curve = {row["budget"]: row for row in budget_curves(runs, config)}
    assert curve[1]["completion"] == 0
    assert curve[69]["completion"] == 0
    assert curve[70]["completion"] == 0.5
    assert max(curve) == 100


def test_repeated_seeds_do_not_create_independent_boards(tmp_path):
    runs = [saved_run(tmp_path, seed=seed) for seed in range(4)]
    config = EvaluationConfig(run_dirs=[str(tmp_path)])
    assert summarize(runs, config)[0]["ci95_low"] is None
    runs.append(saved_run(tmp_path, instance="board_b", solved=False))
    summary = summarize(runs, config)[0]
    assert summary["completion"] == 0.5  # Equal boards, not 4/5 nominal trials.
    assert summary["boards"] == 2 and summary["missing_runs"] == 3


def test_incompatible_runs_are_separate_cohorts(tmp_path):
    runs = [
        saved_run(tmp_path),
        saved_run(tmp_path, name="fixed"),
        saved_run(tmp_path, provider="openai"),
        saved_run(tmp_path, easy=True),
        saved_run(tmp_path, budget=200),
    ]
    assert len(groups(runs)) == 4
    assert runs[0].cohort == runs[1].cohort


def test_legacy_unsaved_prompts_are_not_mixed_with_recorded_profiles(tmp_path):
    run = saved_run(tmp_path / "recorded")
    export_run(run)
    recorded = load_runs(EvaluationConfig(run_dirs=[str(run.directory)]))[0]
    legacy = saved_run(tmp_path / "legacy")
    export_run(legacy)
    path = legacy.directory / "config.json"
    old_config = json.loads(path.read_text(encoding="utf-8"))
    old_config.pop("prompts")
    write_json(path, old_config)
    old = load_runs(EvaluationConfig(run_dirs=[str(legacy.directory)]))[0]
    assert recorded.detail["_prompt_profile_recorded"] is True
    assert old.detail["_prompt_profile_recorded"] is False
    assert recorded.cohort != old.cohort


def test_pairing_reports_unpaired_boards_and_ambiguity(tmp_path):
    runs = [
        saved_run(tmp_path),
        saved_run(tmp_path, instance="board_b"),
        saved_run(tmp_path, name="fixed", solved=False),
        saved_run(tmp_path, name="fixed", instance="board_c"),
    ]
    config = EvaluationConfig(run_dirs=[str(tmp_path)])
    estimates, points = paired(runs, config)
    assert len(points) == 1 and estimates[0]["completion_delta"] == 1
    assert estimates[0]["unpaired_left"] == estimates[0]["unpaired_right"] == 1
    assert estimates[0]["ci95_low"] is None
    runs.append(saved_run(tmp_path, name="fixed_shared"))
    assert paired(runs, config) == ([], [])


def test_discovery_deduplicates_paths_but_rejects_repeated_trials(tmp_path):
    run = saved_run(tmp_path)
    export_run(run)
    config = EvaluationConfig(run_dirs=[str(tmp_path), str(run.directory)])
    assert len(load_runs(config)) == 1
    duplicate = SavedRun(run.record, run.settings, tmp_path / "rerun", run.detail)
    export_run(duplicate)
    with pytest.raises(ValueError, match="Duplicate logical trial"):
        load_runs(config)


def test_inconsistent_solution_lengths_are_rejected(tmp_path):
    run = saved_run(tmp_path)
    record = run.record.model_dump()
    record["solved"] = False
    record["finish_reason"] = "action_budget"
    with pytest.raises(ValueError, match="Only solved"):
        RunRecord.model_validate(record)


def test_figures_and_offline_report_export(tmp_path):
    runs = [saved_run(tmp_path), saved_run(tmp_path, name="fixed", solved=False)]
    for run in runs:
        export_run(run)
    config = EvaluationConfig(
        run_dirs=[str(tmp_path)],
        output_dir=str(tmp_path / "report"),
        max_board_viewers=0,
    )
    figures = cohort_figures(runs, build_tables(runs, config))
    assert len(figures) == 8
    budget_json = figures["budget"].to_json()
    assert isinstance(budget_json, str)
    assert json.loads(budget_json)["data"][0]["line"]["shape"] == "hv"
    dashboard = evaluate(config)
    assert dashboard.is_file()
    assert (dashboard.parent / "plotly.min.js").is_file()
    assert 'src="https://cdn.plot.ly' not in dashboard.read_text(encoding="utf-8")
    assert len(list(dashboard.parent.glob("*.plotly.json"))) == 8
    assert json.loads(
        (dashboard.parent / "input_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        (dashboard.parent / "attempts.csv")
        .read_text(encoding="utf-8")
        .startswith("cohort,")
    )
