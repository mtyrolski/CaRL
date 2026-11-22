"""Compare stored Experience outcomes at common low-level-node budgets."""
import argparse
import csv
from pathlib import Path

from scripts.datasets.gather_k841_solutions import load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imitation-glob", required=True)
    parser.add_argument("--scratch-glob", required=True)
    parser.add_argument("--node-budget", type=int, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if any(b <= 0 for b in args.node_budget):
        parser.error("Budgets must be positive")
    rows = []
    for group, pattern in (("imitation", args.imitation_glob), ("scratch", args.scratch_glob)):
        records, summary = load_records(pattern)
        if summary["failed_files_count"] or summary["counts"]["unknown_items_ignored"]:
            raise ValueError(f"Invalid records in {group}; gather and inspect inputs first")
        for budget in args.node_budget:
            count = sum(
                bool(record.solution.solved)
                and record.search_info.low_level_nodes_visited <= budget
                for _, _, record in records
            )
            rows.append(
                {
                    "type": "budget_rate",
                    "metric": "solved_within_low_level_nodes",
                    "group": group,
                    "budget": budget,
                    "count": count,
                    "total": len(records),
                    "rate": count / len(records) if records else 0.0,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trajectory_stats.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stored trajectory comparison",
        "",
        "| Group | Budget | Solved | Total | Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['group']} | {r['budget']} | {r['count']} | {r['total']} | {r['rate']:.3f} |"
        for r in rows
    )
    (args.output_dir / "trajectory_stats.md").write_text("\n".join(lines) + "\n")
    if not args.no_plots:
        import plotly.express as px

        px.line(rows, x="budget", y="rate", color="group", markers=True).write_html(
            args.output_dir / "trajectory_stats.html"
        )


if __name__ == "__main__":
    main()
