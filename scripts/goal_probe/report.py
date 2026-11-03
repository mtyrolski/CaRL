"""Export observed goal-placement and model-judged outcomes without API access."""

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from carl.icl.config import ROOT
from scripts.goal_probe.data import OUTPUT, goals

Row = dict[str, Any]


def write_csv(path: Path, rows: list[Row]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def explicit_rows() -> list[Row]:
    result = []
    directory = OUTPUT / "explicit/openai"
    queries = json.loads((OUTPUT / "data_validation_queries.json").read_text())["instances"]
    for index, query in enumerate(queries):
        for variant in ("baseline26", "goal_examples"):
            path = directory / f"case_{index:02d}_{variant}.json"
            saved = json.loads(path.read_text()) if path.exists() else {}
            attempts = [item["rollout"] for item in saved.get("proposals", []) if "rollout" in item]
            complete = bool(saved.get("complete"))
            endpoints = [attempt["actual_endpoint"] for attempt in attempts]
            result.append({"query_id": query["id"], "board_id": query["provenance"]["board_id"],
                           "variant": variant, "status": "completed" if complete else "incomplete" if saved else "missing",
                           "exact_hit": saved.get("reached_any") if complete else None,
                           "reached_goal_progress": saved.get("reached_goal_progress") if complete else None,
                           "goals_before": goals(query["state"]),
                           "endpoint_goals_max": max(map(goals, endpoints)) if endpoints else None,
                           "actions": sum(item["actions"] for item in attempts) if complete else None,
                           "no_ops": sum(step["no_op"] for item in attempts for step in item["steps"]) if complete else None,
                           "invalid": saved.get("diagnostics", {}).get("invalid_proposals"),
                           "copies": saved.get("diagnostics", {}).get("self_copies"),
                           "physical_requests": saved.get("metrics", {}).get("requests"),
                           "cache_hits": saved.get("metrics", {}).get("cache_hits"),
                           "raw_file": str(path.relative_to(ROOT))})
    return result


def verbal_rows(group: str = "verbal") -> list[Row]:
    result = []
    for directory in sorted((OUTPUT / group).glob("*")):
        if not (directory / "raw_calls.jsonl").exists():
            continue  # Unexecuted plans are not observations.
        frozen = json.loads((directory / "checkpoint.json").read_text())
        if frozen["provider"] != "openai":
            continue
        for position, entry in enumerate(frozen["queries"]):
            index, query = (entry["index"], entry["query"]) if "query" in entry else (position, entry)
            path = directory / f"query_{index:02d}.json"
            saved = json.loads(path.read_text()) if path.exists() else {}
            attempts = saved.get("attempts", [])
            steps = [step for item in attempts for step in item["steps"]]
            endpoints = [item["steps"][-1]["after"] for item in attempts if item["steps"]]
            complete = bool(saved) and not any(item["error"] for item in attempts)
            result.append({"signature": directory.name, "query_id": query["id"],
                           "board_id": query["provenance"]["board_id"],
                           "status": "completed" if complete else "incomplete" if saved else "missing",
                           "scale": saved.get("scale"), "goals": json.dumps([item["goal"] for item in attempts]),
                           "judge_claimed_hit": any(item["judged_reached"] for item in attempts) if complete else None,
                           "task_solved": any(item["solved"] for item in attempts) if complete else None,
                           "goals_before": goals(query["state"]),
                           "endpoint_goals_max": max(map(goals, endpoints)) if endpoints else None,
                           "goal_progress": any(goals(end) > goals(query["state"]) for end in endpoints) if complete else None,
                           "actions": len(steps) if complete else None,
                           "no_ops": sum(step["before"] == step["after"] for step in steps) if complete else None,
                           "judge_reasons": json.dumps([
                               {key: step["judgment"][key] for key in ("reached", "reason")}
                               if step["judgment"] else None for step in steps]),
                           "raw_file": str(path.relative_to(ROOT))})
    return result


def costs() -> list[Row]:
    path = ROOT / "analysis/luna_icl/campaign_100/cost_ledger.sqlite3"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        records = db.execute(
            "SELECT CASE WHEN tag = 'goal_progress_probe:full_board_32' THEN 'full_board_32' "
            "WHEN tag LIKE 'goal_progress_probe:explicit:baseline26:%' THEN 'explicit_baseline' "
            "WHEN tag LIKE 'goal_progress_probe:%' THEN 'explicit_goal_examples' "
            "WHEN tag LIKE 'verbal_grounded_retry_%' THEN 'verbal_grounded_retry' "
            "WHEN tag LIKE 'verbal_placement_calibrate_%' THEN 'verbal_calibration' ELSE 'verbal_queries' END, "
            "status,count(*),coalesce(sum(charged_nanos),0)/1e9, "
            "sum(CASE WHEN status!='settled' THEN reserved_nanos ELSE 0 END)/1e9 "
            "FROM attempts WHERE tag LIKE 'goal_progress_probe:%' OR tag LIKE 'verbal_placement_%' "
            "OR tag LIKE 'verbal_grounded_retry_%' GROUP BY 1,2"
        ).fetchall()
    return [dict(zip(("phase", "status", "attempts", "charged_usd", "unsettled_reserved_usd"), row))
            for row in records]


def main() -> None:
    explicit, verbal, charges = explicit_rows(), verbal_rows(), costs()
    grounded = verbal_rows("verbal_grounded_retry")
    directory = OUTPUT / "report"
    directory.mkdir(exist_ok=True)
    write_csv(directory / "explicit.csv", explicit)
    write_csv(directory / "verbal.csv", verbal)
    write_csv(directory / "verbal_grounded_retry.csv", grounded)
    write_csv(directory / "costs.csv", charges)
    calibration = []
    for path in sorted((OUTPUT / "verbal").glob("*/calibration/*/calibration.json")):
        frozen = json.loads((path.parents[2] / "checkpoint.json").read_text())
        if frozen["provider"] != "openai":
            continue
        saved = json.loads(path.read_text())
        for target in saved["targets"]:
            calibration.append({"signature": saved["signature"], "status": saved["status"],
                                **{key: target.get(key) for key in ("context", "target_index", "goal", "k",
                                    "attempts", "hits", "p_hat", "d", "weight", "u", "excluded_reason")}})
    write_csv(directory / "verbal_calibration.csv", calibration)
    groups = []
    for variant in ("baseline26", "goal_examples"):
        selected = [row for row in explicit if row["variant"] == variant]
        completed = [row for row in selected if row["status"] == "completed"]
        groups.append({"variant": variant, "expected": len(selected), "completed": len(completed),
                       "exact_hits": sum(bool(row["exact_hit"]) for row in completed),
                       "reached_goal_progress": sum(bool(row["reached_goal_progress"]) for row in completed)})
    observed = [row for row in verbal if row["status"] == "completed"]
    verbal_summary = {"expected": len(verbal), "completed": len(observed),
                      "judge_claims": sum(bool(row["judge_claimed_hit"]) for row in observed),
                      "goal_progress": sum(bool(row["goal_progress"]) for row in observed),
                      "task_solved": sum(bool(row["task_solved"]) for row in observed)}
    completed_retries = [row for row in grounded if row["status"] == "completed"]
    retry_summary = {"expected": len(grounded), "completed": len(completed_retries),
                     "judge_claims": sum(bool(row["judge_claimed_hit"]) for row in completed_retries),
                     "goal_progress": sum(bool(row["goal_progress"]) for row in completed_retries)}
    full_path = OUTPUT / "full_board_32/case.json"
    full = json.loads(full_path.read_text()) if full_path.exists() else {}
    full_summary = {"status": "completed" if full else "not_run",
                    **({key: full["outcome"][key] for key in
                        ("solved", "finish_reason", "attempted_actions", "expansions", "accepted", "rejected")}
                       if full else {}), "max_endpoint_goals": full.get("max_endpoint_goals")}
    summary = {"explicit": groups, "verbal": verbal_summary, "costs": charges,
               "verbal_calibration_targets": calibration, "full_board_32": full_summary,
               "verbal_grounded_retry": retry_summary}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure = make_subplots(rows=1, cols=3, subplot_titles=(
        "Explicit endpoints", "Text goals + Luna judge", "Post-hoc generator retries"))
    for metric, title in (("exact_hits", "Exact hit"), ("reached_goal_progress", "Exact hit with additional box on goal")):
        figure.add_trace(go.Bar(name=title, x=[row["variant"] for row in groups],
                               y=[row[metric] for row in groups],
                               text=[f"{row[metric]}/{row['completed']}" for row in groups],
                               textposition="outside"), row=1, col=1)
    figure.add_trace(go.Bar(name="Observed textual outcomes", x=["Judge claims", "Additional box on goal"],
                           y=[verbal_summary["judge_claims"], verbal_summary["goal_progress"]],
                           text=[f"{verbal_summary[key]}/{len(observed)} observed" for key in ("judge_claims", "goal_progress")],
                           textposition="outside"), row=1, col=2)
    figure.add_trace(go.Bar(name="Previously failed queries only", x=["Judge claims", "Additional box on goal"],
                           y=[retry_summary["judge_claims"], retry_summary["goal_progress"]],
                           text=[f"{retry_summary[key]}/{len(completed_retries)} observed"
                                 for key in ("judge_claims", "goal_progress")],
                           textposition="outside"), row=1, col=3)
    figure.update_layout(title="Targeted goal-placement diagnostic — six validation layouts, no full-board benchmark",
                         template="plotly_white", width=1550, height=650, barmode="group",
                         legend={"orientation": "h", "y": -0.25}, margin={"b": 150})
    figure.update_yaxes(title_text="Queries", range=[0, 7], dtick=1)
    figure.update_traces(textfont_size=14)
    figure.write_html(directory / "goal_progress.html", include_plotlyjs=True)
    figure.write_json(directory / "goal_progress.json")
    for extension in ("svg", "pdf"):
        figure.write_image(directory / f"goal_progress.{extension}")
    table = "\n".join(f"| {row['variant']} | {row['completed']}/{row['expected']} | {row['exact_hits']} | {row['reached_goal_progress']} |"
                      for row in groups)
    spent = sum(row["charged_usd"] for row in charges)
    text = f"""# Goal-placement and verbal-goal diagnostic

| Explicit variant | Completed/planned | Exact endpoint hits | Exact hits with additional box on goal |
|---|---:|---:|---:|
{table}

All scheduled queries remain in the denominator. Validation sources were deliberately selected
near a known goal-placement event in six previously permitted validation trajectories.
They are not fresh full-board solves or an unbiased solve-rate sample. The intervention combines
stronger task-progress instructions with six unlabelled train-only examples; the 26 calibrated
scale examples and CLLP are unchanged. This does not isolate the effect of wording from examples.

The verbal experiment has {len(observed)}/{len(verbal)} completed queries, {verbal_summary['judge_claims']}
positive judge claims and {verbal_summary['goal_progress']} endpoints with additional boxes on goals.
The controller and stopping rule differ from the explicit variant. Judge claims are not ground
truth and a preparation subgoal may be correctly reached without placing a box. Each judgment,
its reason, and the actual simulator transition remain in the raw records. Task completion is
checked independently by Sokoban. No query witness is given to either model. Verbal u labels
come from a separate tiny controller/judge calibration, not the exact-state checkpoint.
Its target counts and measured labels are in verbal_calibration.csv when available.
Calibration uses one deterministic rollout per target, so its percentiles are coarse.

The separate post-hoc generator retry revisits only failures 01/03 with explicit input-derived
box/goal coordinates and stronger grounding instructions. Result: {json.dumps(retry_summary)}.
Its controller and judge are unchanged; this is not an independent retest of four queries.
See verbal_grounded_retry.csv for the new descriptions and actions.
The [separate retry audit](../grounded_judge_audit.md) verifies seven actions and six of seven
judge decisions. One false negative delayed the first stop by one move; no false positives
were observed. Actual occupied-goal counts increase 0→1 and 1→2 on the two queries.

Post-hoc inspection of the original six calibration and four validation rollouts found all
46 per-step reached claims consistent with their literal coordinate conditions; all actions
replay. This small audit does not establish general judge accuracy. Query 01 originally
misnamed a player square as a box and floor as a goal. Query 03's controller alternated moves
without moving the named box to its goal. See [judge audit](../judge_audit.md).
In 41 raw judge replies, progress was absent. Historical saved zeros are decoder placeholders,
not model scores; the audit exports absent progress as null. No audit result is fed to inference.

Separate full-board diagnostic: {json.dumps(full_summary)}. A completed result here uses
B=32 on one already known validation layout. It is outside the targeted component denominator
and is not a paired comparison with the historical B=128 run.
Its [search tree](../full_board_32/search_tree.html) includes a retained path with two occupied
goals after seven path actions; total attempted work is 32 actions including failed attempts.
The full four-box task remains unsolved. No further full-board experiment is inferred from this.

New settled charges for this study: **${spent:.9f}**. Reservations with unknown usage are separate
in costs.csv. These costs exclude the prior $0.197832468 study. Missing results are not invented.

Data: [explicit.csv](explicit.csv), [verbal.csv](verbal.csv), [costs.csv](costs.csv).
Figure: [HTML](goal_progress.html), [SVG](goal_progress.svg), [PDF](goal_progress.pdf).
Public sources, train examples and private validation witnesses are separate data_*.json files
one directory above. Raw explicit requests are in ../explicit/openai; verbal checkpoints and
request/response logs are in ../verbal. Rebuild offline:

```bash
uv run --extra icl python -m scripts.goal_probe.report
```
"""
    (directory / "README.md").write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
