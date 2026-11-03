"""Export launch coverage, observed outcomes and figures without constructing a client."""

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path

import plotly.graph_objects as go

from carl.icl.api import write_json
from carl.icl.config import ROOT
from carl.icl.types import JSONObject
from scripts.launch100.plan import OUTPUT
from scripts.launch100.runtime import job_key, status


def costs(freeze_id: str) -> dict[str, JSONObject]:
    """Read settled charges separately from unresolved reservations; never create a ledger."""
    configuration = json.loads((ROOT / "configs/icl/campaign_budget.json").read_text())
    ledger = ROOT / configuration["ledger_path"]
    if not ledger.is_file():
        return {}
    prefix = f"launch100:{freeze_id}:"
    with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True) as database:
        rows = database.execute(
            "SELECT tag,COUNT(*),SUM(COALESCE(charged_nanos,0)),"
            "SUM(CASE WHEN charged_nanos IS NULL THEN reserved_nanos ELSE 0 END) "
            "FROM attempts WHERE tag LIKE ? GROUP BY tag", (prefix + "%",)
        ).fetchall()
    return {tag.removeprefix(prefix): {"paid_attempts": count, "settled_usd": charged / 1e9,
                                     "unsettled_reserved_usd": reserved / 1e9}
            for tag, count, charged, reserved in rows}


def run_rows(plan: JSONObject, output: Path, charges: dict[str, JSONObject]) -> list[JSONObject]:
    """Keep one row per planned cell, with NULL solved for an unobserved final outcome."""
    rows: list[JSONObject] = []
    for board in plan["cohort"]:
        for method in plan["methods"]:
            key = job_key(plan, board, method)
            directory = output / "jobs" / key
            path = directory / "case.json"
            record = json.loads(path.read_text()) if path.is_file() else None
            if record and (record["job_key"] != key or record["freeze_id"] != plan["freeze_id"]
                           or record["board_id"] != board["id"] or record["method"] != method):
                raise ValueError(f"Run identity mismatch: {path}")
            outcome = record["outcome"] if record else {}
            metrics = record["metrics"] if record else {}
            completed = bool(record and record["complete"])
            solved = outcome.get("solved") if completed else None
            replay = record.get("solution_replay_verified") if record else None
            if solved and not (record and replay and record["all_actions_replay_verified"]):
                raise ValueError(f"Solved case lacks verified replay: {path}")
            row: JSONObject = {
                "freeze_id": plan["freeze_id"], "job_key": key, "board_id": board["id"],
                "state_hash": board["state_hash"], "family": "real_12x12x4", "split": "pilot100",
                "method": method, "seed": plan["seed"], "node_budget": plan["node_budget"],
                "action_budget": plan["action_budget"], "status": "complete" if completed else
                "blocked" if record else "interrupted" if directory.exists() else "missing",
                "solved": solved, "solution_replay_verified": replay,
                "all_actions_replay_verified": record.get("all_actions_replay_verified") if record else None,
                "finish_reason": outcome.get("finish_reason"), "attempted_actions": outcome.get("attempted_actions"),
                "expansions": outcome.get("expansions"),
                "solution_length": len(outcome["action_path"]) if solved else None,
                "accepted": outcome.get("accepted"), "rejected": outcome.get("rejected"),
                "fallback": outcome.get("primitive_successors", outcome.get("primitive_actions")),
                "response_replay_calls_this_invocation": record.get("response_replay_calls_this_invocation") if record else None,
            }
            for field in ("requests", "logical_requests", "cache_hits", "input_tokens", "output_tokens",
                          "logical_input_tokens", "logical_output_tokens"):
                row[field] = metrics.get(field)
            row.update(charges.get(key, {"paid_attempts": None, "settled_usd": None,
                                         "unsettled_reserved_usd": None}))
            rows.append(row)
    if len({row["job_key"] for row in rows}) != len(rows):
        raise ValueError("Duplicate planned jobs")
    return rows


def wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    """Board-level 95% binomial interval; no estimate when no board is observed."""
    if total == 0:
        return None, None
    z = 1.959963984540054
    fraction = successes / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / denominator
    return (0.0 if successes == 0 else max(0.0, center - radius),
            1.0 if successes == total else min(1.0, center + radius))


def summaries(rows: list[JSONObject]) -> list[JSONObject]:
    """Report observed rates alongside coverage and full-cohort possible ranges."""
    result = []
    for method in dict.fromkeys(row["method"] for row in rows):
        selected = [row for row in rows if row["method"] == method]
        observed = [row for row in selected if row["solved"] is not None]
        successes = sum(row["solved"] is True for row in observed)
        lower, upper = wilson(successes, len(observed))
        result.append({"method": method, "expected": len(selected), "observed": len(observed),
                       "verified_successes": successes, "denominator": len(observed),
                       "solve_rate": successes / len(observed) if observed else None,
                       "wilson95_low": lower, "wilson95_high": upper,
                       "full_cohort_low": successes / len(selected),
                       "full_cohort_high": (successes + len(selected) - len(observed)) / len(selected)})
    return result


def paired(rows: list[JSONObject]) -> JSONObject:
    """Restrict the main contrast to boards observed under both explicit conditions."""
    pairs: dict[str, dict[str, bool]] = {}
    for row in rows:
        if row["method"] in {"relative_u", "absolute_d"} and row["solved"] is not None:
            pairs.setdefault(row["board_id"], {})[row["method"]] = row["solved"]
    complete = [row for row in pairs.values() if len(row) == 2]
    both = sum(row["relative_u"] and row["absolute_d"] for row in complete)
    relative = sum(row["relative_u"] and not row["absolute_d"] for row in complete)
    absolute = sum(not row["relative_u"] and row["absolute_d"] for row in complete)
    return {"complete_pairs": len(complete), "both": both, "relative_only": relative,
            "absolute_only": absolute, "neither": len(complete) - both - relative - absolute,
            "delta_pp": 100 * (relative - absolute) / len(complete) if complete else None}


def write_csv(path: Path, rows: list[JSONObject]) -> None:
    """Write deterministic UTF-8 tables, encoding missing observations as empty fields."""
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def figure(summary: list[JSONObject], plan: JSONObject) -> go.Figure:
    """Show observed board solve rates; a missing cohort is not a zero-success cohort."""
    observed = [row for row in summary if row["observed"]]
    plot = go.Figure()
    if observed:
        rates = [100 * row["solve_rate"] for row in observed]
        plot.add_trace(go.Scatter(
            x=[row["method"] for row in observed], y=rates, mode="markers+text",
            text=[f'{row["verified_successes"]}/{row["observed"]} observed; {row["expected"]} planned' for row in observed],
            textposition="top center", marker={"size": 12},
            error_y={"type": "data", "symmetric": False,
                     "array": [100 * (row["wilson95_high"] - row["solve_rate"]) for row in observed],
                     "arrayminus": [100 * (row["solve_rate"] - row["wilson95_low"]) for row in observed]},
        ))
    else:
        plot.add_annotation(text="No completed observations. All planned runs remain missing.",
                            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    plot.update_layout(template="plotly_white", width=1000, height=500,
                       title=f'Real 12×12×4 · {plan["node_budget"]} expansion cap · {plan["action_budget"]} action cap',
                       yaxis={"title": "Verified board solve rate (%) · Wilson 95%", "range": [-5, 110]},
                       xaxis={"categoryorder": "array", "categoryarray": plan["methods"]})
    if not observed:
        plot.update_xaxes(visible=False)
        plot.update_yaxes(visible=False)
    return plot


def export(output: Path, *, static: bool = False) -> JSONObject:
    """Read only completed records and the ledger; never load credentials or infer."""
    plan = json.loads((output / "manifest.json").read_text())
    status(plan, output)  # Validate saved identities and completion digests before exporting.
    rows = run_rows(plan, output, costs(plan["freeze_id"]))
    summary, contrast = summaries(rows), paired(rows)
    target = output / "plot_data"
    target.mkdir(parents=True, exist_ok=True)
    write_csv(target / "runs.csv", rows)
    write_csv(target / "solve_rates.csv", summary)
    write_csv(target / "paired_delta.csv", [contrast])
    plot = figure(summary, plan)
    plot.write_html(target / "solve_rate.html", include_plotlyjs=True)
    if static:
        for extension in ("svg", "pdf"):
            plot.write_image(target / f"solve_rate.{extension}")
    result = {"freeze_id": plan["freeze_id"], "expected": len(rows), "methods": summary,
              "paired_relative_minus_absolute": contrast, "source": str(output),
              "note": "Single seed, observed-board denominator; no invented missing outcomes. "
                      "Logical tokens include cache replay. Provider tokens and settled USD describe charged calls. "
                      "Unsettled reservations are not confirmed charges. Calibration is reused and excluded."}
    write_json(target / "summary.json", result)
    (target / "COLUMNS.md").write_text(
        "# Launch tables\n\n`runs.csv` key: freeze_id + board_id + method + seed + node_budget + action_budget.\n"
        "Empty fields are missing observations. `solved` is populated only for completed runs.\n"
        "`node_budget` and `expansions` count expanded queue entries; `action_budget` and `attempted_actions` "
        "count actual simulator moves, including no-ops and failed attempts. Solution length counts path moves.\n"
        "Token counts are tokens; requests are calls; cost columns are USD. Logical metrics include local cache "
        "replay; provider metrics describe physical attempts. Journal replay restores earlier cumulative metrics.\n"
        "`settled_usd` is confirmed ledger usage; `unsettled_reserved_usd` is held headroom, not a charge. "
        "A missing ledger row is NULL. Calibration was paid earlier and is not billed again here.\n"
        "`solve_rates.csv` denominator is completed boards for one method and seed. Wilson intervals use that "
        "denominator. Full-cohort bounds assign every unobserved board failure/success respectively.\n"
        "`paired_delta.csv` uses only complete pairs, relative_u minus absolute_d, in percentage points.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """Offline report command; HTML always, optional SVG/PDF through local Kaleido."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--static", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(export(arguments.output, static=arguments.static), indent=2))


if __name__ == "__main__":
    main()
