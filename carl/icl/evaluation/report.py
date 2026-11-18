"""Publish local interactive figures, tidy tables and an auditable input manifest."""
import csv
import hashlib
import html
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from carl.icl.api import write_json
from carl.icl.evaluation.budget_sweep import actual_budget_figure, actual_budget_results
from carl.icl.evaluation.data import EvaluationConfig, SavedRun, load_runs, resolve_path
from carl.icl.evaluation.diagnostics import (
    board_viewer,
    calibration_figure,
    load_calibrations,
)
from carl.icl.evaluation.figures import cohort_figures
from carl.icl.evaluation.schema import EMPTY_COLUMNS, NOTES
from carl.icl.evaluation.statistics import (
    attempt_rows,
    budget_curves,
    groups,
    paired,
    summarize,
)
from carl.icl.types import JSONObject

__all__ = ["EMPTY_COLUMNS", "NOTES", "build_tables", "evaluate"]


def _csv(path: Path, rows: list[JSONObject]) -> None:
    fields = (
        list(dict.fromkeys(key for row in rows for key in row))
        if rows
        else list(EMPTY_COLUMNS[path.stem])
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_tables(
    runs: list[SavedRun], config: EvaluationConfig
) -> dict[str, list[JSONObject]]:
    """Expose tidy data directly for notebook edits and independent analysis."""
    estimates, points = paired(runs, config)
    return {
        "summary": summarize(runs, config),
        "curves": budget_curves(runs, config),
        "pairs": estimates,
        "points": points,
        "attempts": attempt_rows(runs),
        "runs": [
            {
                **run.record.model_dump(),
                "cohort": run.cohort,
                "source": str(run.directory),
            }
            for run in runs
        ],
    }


def _export_figure(
    name: str, figure: go.Figure, output: Path, config: EvaluationConfig
) -> str:
    figure.write_json(output / f"{name}.plotly.json")
    figure.write_html(
        output / f"{name}.html",
        include_plotlyjs="directory",
        full_html=True,
        auto_play=False,
    )
    for extension in config.static_formats:
        # Fail explicitly on a broken static renderer; never silently omit requested artifacts.
        figure.write_image(output / f"{name}.{extension}", width=1400, scale=2)
    links = f'<a href="{name}.html">Open figure</a> · <a href="{name}.plotly.json">Plotly JSON</a>'
    fragment = figure.to_html(
        include_plotlyjs=False,
        full_html=False,
        div_id=name,
        auto_play=False,
        config={"responsive": True, "displaylogo": False},
    )
    return f'<div class="figure">{links}{fragment}</div>'


def _manifest(
    runs: list[SavedRun], calibrations: dict[Path, JSONObject]
) -> list[JSONObject]:
    paths = set(calibrations)
    for run in runs:
        paths.update(
            [
                run.directory / "config.json",
                run.directory / "results.json",
                run.directory / f"{run.record.instance}_seed{run.record.seed}.json",
            ]
        )
        for parent in run.directory.parents:
            sweep_manifest = parent / "sweep_manifest.json"
            if sweep_manifest.is_file():
                paths.add(sweep_manifest)
                sweep_plan = parent / "sweep_plan.json"
                if sweep_plan.is_file():
                    paths.add(sweep_plan)
                break
    return [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(paths)
    ]


def _calibration_usage(calibration: JSONObject) -> JSONObject:
    """Sum recorded invocation counters without inventing missing historical cost."""
    history = calibration.get("invocations")
    records = (
        [entry["api"] for entry in history] if history else [calibration.get("api", {})]
    )
    return {
        "api_cost_scope": "recorded_invocations"
        if history
        else "last_saved_invocation",
        **{
            key: sum(record.get(key, 0) for record in records)
            for key in (
                "requests",
                "input_tokens",
                "output_tokens",
                "cache_hits",
                "logical_requests",
                "logical_input_tokens",
                "logical_output_tokens",
            )
        },
    }


def _markdown(
    tables: dict[str, list[JSONObject]], missing_calibrations: list[str]
) -> str:
    lines = [
        "# Luna ICL evaluation",
        "",
        *[f"- {note}" for note in NOTES],
        "",
        "| Cohort | Condition | Runs / expected | Boards | Solved | Board-balanced completion |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['cohort']} | {row['config']} | {row['runs']} / {row['expected_runs']} | "
        f"{row['boards']} | {row['solved']} | {row['completion']:.1%} |"
        for row in tables["summary"]
    )
    lines += [
        "",
        "Open index.html for interactive plots. Tables and Plotly JSON files are editable independently.",
        "Paired comparisons require unique methods per cohort or explicit comparison_configs.",
        "Missing calibration artifacts: " + str(len(missing_calibrations)) + ".",
    ]
    return "\n".join(lines) + "\n"


def evaluate(config: EvaluationConfig) -> Path:
    """Build a complete offline report without instantiating an inference client."""
    runs = load_runs(config)
    tables = build_tables(runs, config)
    calibrations, missing = load_calibrations(runs)
    output = resolve_path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation_config.json", config.model_dump())
    write_json(output / "input_manifest.json", _manifest(runs, calibrations))
    write_json(output / "missing_calibrations.json", missing)
    for name, rows in tables.items():
        _csv(output / f"{name}.csv", rows)
        write_json(output / f"{name}.json", rows)
    sweep_rows = actual_budget_results(runs, config)
    _csv(output / "budget_sweep.csv", sweep_rows)
    write_json(output / "budget_sweep.json", sweep_rows)
    # One local bundle serves the dashboard and every separately exported figure.
    (output / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    sections = [
        "<h1>Luna ICL evaluation</h1>",
        '<p>Tables: <a href="summary.csv">summary</a> · <a href="runs.csv">runs</a> · '
        '<a href="pairs.csv">paired comparisons</a> · <a href="budget_sweep.csv">actual budget sweep</a> · '
        '<a href="attempts.csv">subgoal attempts</a> · '
        '<a href="calibration_costs.csv">calibration costs</a> · <a href="input_manifest.json">input manifest</a></p>',
        "<ul>" + "".join(f"<li>{html.escape(note)}</li>" for note in NOTES) + "</ul>",
    ]
    for sweep in dict.fromkeys(row["sweep"] for row in sweep_rows):
        selected = [row for row in sweep_rows if row["sweep"] == sweep]
        sections.append(
            f"<h2>Independently executed budgets · {html.escape(selected[0]['family'])}</h2>"
            "<p>Open markers indicate missing runs, conditions or unstarted planned budgets. API/expansion budget stops "
            "remain failures; see hover details. Budgets with no records do not appear.</p>"
        )
        sections.append(
            _export_figure(
                f"sweep_{sweep}", actual_budget_figure(selected), output, config
            )
        )
    for cohort, matched in groups(runs).items():
        sections.append(
            f"<h2>{html.escape(matched[0].label)}</h2><p>Cohort: {cohort}</p>"
        )
        cohort_tables = {
            name: [row for row in rows if row["cohort"] == cohort]
            for name, rows in tables.items()
        }
        sections.extend(
            _export_figure(f"{cohort}_{name}", figure, output, config)
            for name, figure in cohort_figures(matched, cohort_tables).items()
        )
    calibration_targets: list[JSONObject] = []
    calibration_costs: list[JSONObject] = []
    for index, (path, calibration) in enumerate(calibrations.items()):
        sections.append(
            _export_figure(
                f"calibration_{index}", calibration_figure(calibration), output, config
            )
        )
        calibration_targets.extend(
            {
                "signature": calibration["signature"],
                "provider": calibration["provider"],
                **{
                    key: row[key]
                    for key in (
                        "context",
                        "k",
                        "d",
                        "u",
                        "hits",
                        "success_rate",
                        "weight",
                    )
                },
            }
            for row in calibration["examples"]
        )
        calibration_costs.append(
            {
                "signature": calibration["signature"],
                "source": str(path),
                "provider": calibration["provider"],
                "status": calibration["status"],
                "trials": calibration["trials"],
                "completed_trial_actions": calibration["calibration_actions"],
                "all_invocation_actions": calibration.get("total_executed_actions"),
                "excluded_targets": calibration["excluded_targets"],
                "excluded_contexts": calibration["excluded_contexts"],
                **_calibration_usage(calibration),
            }
        )
    _csv(output / "calibration_targets.csv", calibration_targets)
    _csv(output / "calibration_costs.csv", calibration_costs)
    # Round-robin across cohorts so small easy fixtures do not hide transfer boards.
    viewers = [matched[0] for matched in groups(runs).values()]
    viewers.extend(
        run for run in runs if all(run is not selected for selected in viewers)
    )
    for index, run in enumerate(viewers[: config.max_board_viewers]):
        sections.append(
            _export_figure(
                f"boards_{index}",
                board_viewer(run, config.max_viewer_attempts),
                output,
                config,
            )
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Luna ICL evaluation</title>'
        '<script src="plotly.min.js"></script><style>body{font:15px Arial;margin:32px auto;max-width:1500px;'
        "padding:0 24px;color:#172033;background:#f8fafc}li{margin:8px 0}.figure{background:white;"
        "padding:16px;margin:20px 0;border:1px solid #dbe2eb;border-radius:8px}a{color:#2563eb}</style>"
        "<body>" + "\n".join(sections) + "</body></html>"
    )
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "README.md").write_text(_markdown(tables, missing), encoding="utf-8")
    return output / "index.html"
