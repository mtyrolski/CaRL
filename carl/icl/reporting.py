"""CSV/Markdown summaries and paired, board-clustered uncertainty estimates."""
import csv
from pathlib import Path
from typing import Final

import numpy as np

from carl.icl.api import write_json
from carl.icl.types import JSONObject

COMPARISON_METHODS: Final = ("relative", "fixed")
MAIN_COMPARISON_METHODS: Final = ("relative_u", "absolute_d")
BOOTSTRAP_REPLICATES: Final = 2000
BOOTSTRAP_SEED: Final = 0
REPORT_HEADER: Final = (
    "# Luna ICL comparison",
    "",
    "Mock outcomes validate plumbing only; they are not Luna benchmark results.",
    "",
    "| Config | Method | Runs | Solved | Mean actions attempted | API errors | API budget stops |",
    "|---|---|---:|---:|---:|---:|---:|",
)


def _write_csv(path: Path, rows: list[JSONObject]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_family(
    family: str, rows: list[JSONObject], methods_pair: tuple[str, str] = COMPARISON_METHODS
) -> JSONObject | None:
    """Return one paired comparison only when each method has a unique condition."""
    group = [row for row in rows if row["family"] == family and row["method"] in methods_pair]
    if any(
        len({row["config"] for row in group if row["method"] == method}) > 1
        for method in methods_pair
    ):
        return None
    methods = {
        method: {(row["instance"], row["seed"]): row for row in group if row["method"] == method}
        for method in methods_pair
    }
    left, right = methods_pair
    if any(len([row for row in group if row["method"] == method]) != len(methods[method]) for method in methods_pair):
        return None
    keys = sorted(methods[left].keys() & methods[right].keys())
    if not keys:
        return None
    boards = sorted({key[0] for key in keys})
    differences = np.array(
        [
            np.mean(
                [
                    float(methods[left][key]["solved"])
                    - float(methods[right][key]["solved"])
                    for key in keys
                    if key[0] == board
                ]
            )
            for board in boards
        ]
    )
    # Resample boards, keeping all search seeds paired within each sampled board.
    bootstrap = (
        np.random.default_rng(BOOTSTRAP_SEED)
        .choice(differences, (BOOTSTRAP_REPLICATES, len(boards)))
        .mean(axis=1)
    )
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "family": family,
        "paired_boards": len(boards),
        "paired_runs": len(keys),
        "method_left": left,
        "method_right": right,
        "comparison": f"{left}_minus_{right}",
        "success_delta": float(differences.mean()),
        f"success_delta_{left}_minus_{right}": float(differences.mean()),
        "unpaired_left": len(methods[left].keys() - methods[right].keys()),
        "unpaired_right": len(methods[right].keys() - methods[left].keys()),
        "paired_api_errors_left": sum(methods[left][key]["finish_reason"] == "api_error" for key in keys),
        "paired_api_errors_right": sum(methods[right][key]["finish_reason"] == "api_error" for key in keys),
        "paired_api_budget_left": sum(methods[left][key]["finish_reason"] == "api_budget" for key in keys),
        "paired_api_budget_right": sum(methods[right][key]["finish_reason"] == "api_budget" for key in keys),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _condition_line(name: str, rows: list[JSONObject]) -> str:
    group = [row for row in rows if row["config"] == name]
    return (
        f"| {name} | {', '.join(sorted({row['method'] for row in group}))} | {len(group)} | "
        f"{sum(row['solved'] for row in group)} | "
        f"{np.mean([row['attempted_actions'] for row in group]):.1f} | "
        f"{sum(row['finish_reason'] == 'api_error' for row in group)} | "
        f"{sum(row['finish_reason'] == 'api_budget' for row in group)} |"
    )


def report(rows: list[JSONObject], directory: Path) -> None:
    """Write raw outcomes and matched comparisons without dropping failed runs."""
    directory.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    _write_csv(directory / "results.csv", rows)
    lines: list[str] = list(REPORT_HEADER)
    lines.extend(
        _condition_line(name, rows) for name in dict.fromkeys(row["config"] for row in rows)
    )
    if any(row["method"] in MAIN_COMPARISON_METHODS for row in rows):
        lines.extend(["", "The primary paired contrast is relative_u − absolute_d. "
                      "fixed, ratio_d_over_ac and normalized_offset are reported as separate conditions above."])
    pairs = []
    for family in dict.fromkeys(row["family"] for row in rows):
        available = {row["method"] for row in rows if row["family"] == family}
        # Never substitute the legacy comparison while a new main contrast is incomplete.
        if available.intersection(MAIN_COMPARISON_METHODS):
            methods = MAIN_COMPARISON_METHODS
        elif "relative" in available:
            methods = COMPARISON_METHODS
        else:
            # A control alone does not identify the intended paired contrast.
            continue
        pair = _paired_family(family, rows, methods)
        if pair is None:
            if available.intersection(methods):
                lines.extend(["", f"{family}: paired comparison ({methods[0]} − {methods[1]}) is incomplete or ambiguous."])
            continue
        pairs.append(pair)
        lines.extend(
            [
                "",
                f"{family}: paired success difference ({methods[0]} − {methods[1]}) "
                f"{pair['success_delta']:+.3f}; board-cluster bootstrap 95% interval "
                f"[{pair['ci95_low']:+.3f}, {pair['ci95_high']:+.3f}] ({pair['paired_boards']} boards). "
                f"Unpaired runs: {pair['unpaired_left']} / {pair['unpaired_right']}. "
                f"Paired API errors: {pair['paired_api_errors_left']} / {pair['paired_api_errors_right']}; "
                f"API budget stops: {pair['paired_api_budget_left']} / {pair['paired_api_budget_right']}.",
            ]
        )
    write_json(directory / "paired_comparison.json", pairs)
    _write_csv(directory / "paired_comparison.csv", pairs)
    (directory / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
