"""Read saved solver/ledger evidence and render offline component figures."""

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
Row = dict[str, Any]


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def ledger_costs() -> tuple[Row, list[Row]]:
    config = read(ROOT / "configs/icl/campaign_budget.json")
    path = ROOT / config["ledger_path"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as database:
        database.execute("BEGIN")  # Keep phase totals and status totals in one read snapshot.
        rows = database.execute(
            "SELECT status, COUNT(*), SUM(COALESCE(charged_nanos,0)), "
            "SUM(CASE WHEN charged_nanos IS NULL THEN reserved_nanos ELSE 0 END) "
            "FROM attempts WHERE tag LIKE 'generator_probe_v2%' GROUP BY status").fetchall()
        tagged = database.execute(
            "SELECT tag, status, COUNT(*), SUM(COALESCE(charged_nanos,0)), "
            "SUM(CASE WHEN charged_nanos IS NULL THEN reserved_nanos ELSE 0 END) "
            "FROM attempts WHERE tag LIKE 'generator_probe_v2%' GROUP BY tag, status").fetchall()
    phases: dict[tuple[str, str], Row] = {}
    for tag, status, count, charged, reserved in tagged:
        if ":calibration:" in tag:
            phase = "expanded_calibration"
        elif tag.endswith("_static_retry"):
            phase = "static_retry"
        elif tag.endswith((":few_shot_4", ":few_shot_26")):
            phase = "expanded_comparison"
        elif tag.endswith((":baseline", ":planned_board", ":planned_board_low")):
            phase = "initial12"
        elif tag.endswith(":full_solver_validation"):
            phase = "full_solver_validation"
        else:
            phase = "other_probe"
        row = phases.setdefault((phase, status), {"phase": phase, "status": status, "physical_attempts": 0,
                                                 "settled_charge_usd": 0.0, "unsettled_reservation_usd": 0.0})
        row["physical_attempts"] += count
        row["settled_charge_usd"] += charged / 1e9
        row["unsettled_reservation_usd"] += reserved / 1e9
    totals = {status: {"attempts": count, "settled_charge_usd": charged / 1e9,
                      "unsettled_reservation_usd": reserved / 1e9} for status, count, charged, reserved in rows}
    for row in phases.values():
        for key in ("settled_charge_usd", "unsettled_reservation_usd"):
            row[key] = round(row[key], 9)
    return totals, sorted(phases.values(), key=lambda row: (row["phase"], row["status"]))


def solver_result(base: Path) -> Row:
    """Keep the exploratory full-board case separate from component denominators."""
    directory = base / "solver_validation"
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {"status": "not_prepared"}
    manifest = read(manifest_path)
    row: Row = {"status": "awaiting_saved_case", "board_id": manifest["board_id"],
                "split": manifest["split"], "seed": manifest["seed"], "budget_actions": manifest["B"],
                "demonstration_count": manifest["config"]["generator_example_limit"],
                "generator_max_attempts": manifest["config"]["generator_max_attempts"]}
    case_path = directory / "case.json"
    if not case_path.exists():
        return row
    case = read(case_path)
    outcome, verification, metrics = case["outcome"], case["verification"], case["metrics"]
    actions = [event for line in (directory / "trace.jsonl").read_text().splitlines()
               if (event := json.loads(line))["event"] == "action"]
    endpoints = case["tree"]["nodes"]
    statuses = Counter(node["status"] for node in endpoints)
    boxes_on_goals = [sum(line.count("*") for line in node["state"]) for node in endpoints]
    if (case["instance"]["id"] != row["board_id"] or not verification["verified"]
            or outcome["attempted_actions"] != verification["trace_actions"]
            or len(actions) != outcome["attempted_actions"]
            or outcome["attempted_actions"] > row["budget_actions"]):
        raise ValueError("Saved full-solver case disagrees with its manifest or replay accounting")
    row.update(status="recorded", solved=outcome["solved"], finish_reason=outcome["finish_reason"],
               attempted_actions=outcome["attempted_actions"], expansions=outcome["expansions"],
               accepted_subgoals=outcome["accepted"], rejected_subgoals=outcome["rejected"],
               replay_verified=verification["verified"], solution_replayed=verification["solution_replayed"],
               solution_length=len(outcome["action_path"]) if outcome["solved"] else None,
               error=outcome.get("error"), physical_requests=metrics["requests"],
               local_cache_hits=metrics["cache_hits"], input_tokens=metrics["input_tokens"],
               output_tokens=metrics["output_tokens"], raw_file=str(case_path.relative_to(base)),
               no_op_actions=sum(event["source_hash"] == event["endpoint_hash"] for event in actions),
               saved_endpoint_samples=len(endpoints), saved_endpoints_zero_boxes_on_goals=boxes_on_goals.count(0),
               endpoint_status_new=statuses["new"], endpoint_status_rejected=statuses["rejected"],
               endpoint_status_duplicate=statuses["duplicate"], endpoint_status_root=statuses["root"])
    return row


def figures(output: Path, scales: list[Row], totals: list[Row], examples: list[Row]) -> None:
    comparison = [row for row in totals if row["experiment"] == "expanded_comparison"]
    comparison.sort(key=lambda row: int(row["variant"].rsplit("_", 1)[1]))
    figure = go.Figure()
    for label, key in (("Distinct endpoint reached", "hit"), ("Raw invalid candidates", "invalid"),
                       ("Raw source copies", "copies")):
        figure.add_bar(name=label, x=[row["variant"] for row in comparison], y=[row[key] for row in comparison],
                       text=[f"{row[key]}/{row['queries']}" for row in comparison], textposition="inside", textangle=0,
                       textfont=dict(size=12), constraintext="none")
    misses = [row["queries"] - row["hit"] - row["invalid"] - row["copies"] for row in comparison]
    figure.add_bar(name="Accepted target missed by CLLP", x=[row["variant"] for row in comparison],
                   y=misses, text=[f"{value}/30" if value else "" for value in misses], textposition="inside",
                   textangle=0, textfont=dict(size=12), constraintext="none")
    figure.update_layout(barmode="stack", title="Generator + frozen CLLP: 30 paired sources on 10 layouts",
                         yaxis_title="Queries (all 30 scheduled per variant)", template="plotly_white",
                         annotations=[dict(text="Three positions per layout are dependent; these are component endpoint hits.",
                                           x=0, y=-0.2, xref="paper", yref="paper", showarrow=False)])
    scale = go.Figure()
    grouped = [row for row in scales if row["experiment"] == "expanded_comparison"]
    for variant in sorted({row["variant"] for row in grouped}, key=lambda value: int(value.rsplit("_", 1)[1])):
        selected = [row for row in grouped if row["variant"] == variant]
        scale.add_scatter(name=variant, mode="markers+text", x=[row["u"] for row in selected],
                          y=[row["first_hit_mean_conditional_on_hit"] for row in selected],
                          text=[f"{row['hit']}/{row['queries']} hits" for row in selected],
                          textposition=["top center" if row["first_hit_mean_conditional_on_hit"] == max(
                              other["first_hit_mean_conditional_on_hit"] for other in grouped if other["u"] == row["u"])
                              else "bottom center" for row in selected], marker=dict(size=9))
    scale.update_layout(title="Successful first-hit time by requested u (different source states)",
                        xaxis_title="Requested policy-relative percentile u", yaxis_title="Mean actions, conditional on exact hit",
                        template="plotly_white", yaxis=dict(range=[0, 8.5]),
                        annotations=[dict(text="Descriptive only: each source receives one u; no paired scale-effect estimate.",
                                          x=0, y=-0.2, xref="paper", yref="paper", showarrow=False)])
    labels: dict[tuple[float, float], list[Row]] = defaultdict(list)
    for example in examples:
        labels[(example["u"], example["d"])].append(example)
    points = sorted(labels)
    calibration = go.Figure(go.Scatter(
        mode="markers+text", x=[point[0] for point in points], y=[point[1] for point in points],
        marker=dict(size=[10 + 2 * len(labels[point]) for point in points]),
        text=[f"n={len(labels[point])}" for point in points], textposition="top center",
        customdata=["<br>".join(item["context"] for item in labels[point]) for point in points],
        hovertemplate="u=%{x:.3f}<br>d=%{y}<br>%{text}<br>%{customdata}<extra></extra>"))
    calibration.update_layout(title="26 retained training demonstrations: measured labels and multiplicities",
                              xaxis_title="Within-context weighted percentile u", yaxis_title="Mean successful first-hit actions d",
                              template="plotly_white", annotations=[dict(
                                  text="Each label is defined within its own training context; coincident (u,d) values are grouped.",
                                  x=0, y=-0.2, xref="paper", yref="paper", showarrow=False)])
    for name, plot in (("endpoint_outcomes", figure), ("scale_first_hit", scale), ("checkpoint_calibration", calibration)):
        plot.update_layout(margin=dict(l=90, r=40, t=80, b=130))
        plot.write_html(output / f"{name}.html", include_plotlyjs=True)
        for extension in ("svg", "pdf"):
            plot.write_image(output / f"{name}.{extension}", width=1050, height=650)
