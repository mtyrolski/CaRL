"""Estimate ICL sweep tokens locally; never read credentials or call an API."""
import argparse
import csv
import json
from pathlib import Path
from typing import Final

from carl.icl.api import write_json
from carl.icl.budget_sweep import load_sweep
from carl.icl.config import ROOT
from carl.icl.token_estimate import Assumptions, estimate

DEFAULT_SWEEP: Final[str] = "configs/icl/budget_sweeps/quick.json"
DEFAULT_ASSUMPTIONS: Final[str] = "configs/icl/token_estimates/assumptions.json"


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    """Write reproducible JSON/CSV/Markdown estimates with their assumptions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default=DEFAULT_SWEEP)
    parser.add_argument("--assumptions", default=DEFAULT_ASSUMPTIONS)
    parser.add_argument("--output", default="analysis/luna_icl/token_estimates/quick")
    args = parser.parse_args()
    spec, configs = load_sweep(args.sweep)
    assumptions = Assumptions.model_validate_json(
        _path(args.assumptions).read_text(encoding="utf-8")
    )
    data = estimate(spec, configs, assumptions)
    output = _path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "estimate.json", data)
    rows = data["scenarios"]
    fields = [
        "scenario",
        "calibration_calls",
        "calibration_input",
        "calibration_output",
        "calibration_usd_uncached",
        "search_calls",
        "search_input",
        "search_output",
        "search_usd_uncached",
        "total_input",
        "total_output",
        "estimated_usd_uncached",
        "conditions_over_configured_limits",
    ]
    with (output / "estimate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Offline token scenarios",
        "",
        "Hypothetical workloads, not measured Luna usage or a statistical expectation. "
        "One calibration is included; the scenarios assume no cache savings, direct retrieval "
        "and complete JSON responses. See estimate.json for every assumption and sample.",
        "",
        f"Price snapshot: {assumptions.pricing.model}, {assumptions.pricing.tier}, "
        f"{assumptions.pricing.as_of}: ${assumptions.pricing.input_usd_per_million:g}/M input, "
        f"${assumptions.pricing.output_usd_per_million:g}/M output. "
        f"Source: {assumptions.pricing.source}",
        "",
        "| Scenario | Calibration input | Calibration output | Calibration USD | Search input | Search output | Search USD | Total input | Total output | Total USD | Limit flags |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['calibration_input']:,} | {row['calibration_output']:,} | "
            f"${row['calibration_usd_uncached']:.2f} | {row['search_input']:,} | {row['search_output']:,} | "
            f"${row['search_usd_uncached']:.2f} | {row['total_input']:,} | {row['total_output']:,} | "
            f"${row['estimated_usd_uncached']:.2f} | {', '.join(row['conditions_over_configured_limits']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Scenario assumptions",
            "",
            "| Scenario | Action fraction | Actions per expansion | Calibration fraction | Output multiplier | Policy/gen/value output tokens | Generator attempts | Retry input factor |",
            "|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        values = row["assumptions"]
        lines.append(
            f"| {row['scenario']} | {values['action_fraction']:.0%} | "
            f"{values['actions_per_generator']:g} | {values['calibration_fraction']:.0%} | "
            f"{values['output_multiplier']:g}× | "
            f"{values['policy_output_tokens'] or 'template'}/"
            f"{values['generator_output_tokens'] or 'template'}/"
            f"{values['value_output_tokens'] or 'template'} | "
            f"{values['generator_attempts_per_expansion']:g} | {values['retry_input_factor']:g} |"
        )
    limits = data["logical_limits"]
    lines += [
        "",
        "## Configured logical accounting ceilings",
        "",
        f"- Search across {data['plan']['runs']} runs: {limits['search_input_proxy']:,} locally estimated input tokens and {limits['search_output_tokens']:,} output tokens.",
        f"- Separate calibration: {limits['calibration_input_proxy']:,} locally estimated input tokens, "
        f"{limits['calibration_output_tokens']:,} output tokens, at most {limits['calibration_max_policy_calls']:,} policy calls.",
        "- These are limits in local accounting, not guaranteed limits on provider billing. "
        "Failures, tokenization differences, retries across invocations, tax, regional processing and cache accounting can change the bill.",
        "",
        "## Important assumptions",
        "",
        "- Input counts are o200k_base token counts of serialized request payloads, matching the local guard; provider usage can differ. "
        "Actual test boards and training examples are used; relative labels are synthetic only for prompt sizing.",
        "- Search prompt samples use the initial board as a stand-in policy target; later states, targets and their token counts can differ.",
        "- Model completion lengths, invalid proposals, early solving, primitive fallback, cache reuse, wall time and API failures "
        "are unknown before a live run. Scenarios are chosen assumptions, not probabilities or observed model behavior.",
        "- Unless explicit output-token estimates are provided, sizes count only visible JSON and omit reasoning tokens. "
        "Use measured output tokens for reasoning-enabled components; retry factors are assumptions.",
        "- Price calculations assume every estimated input token is charged as uncached and every estimated output token at "
        "the listed standard short-context rate. There is no cache-write or regional uplift estimate; prices can change.",
        "- Scenario totals that exceed configured local limits are flagged in the JSON/CSV, not clipped or presented as feasible. Per-request output reservations can stop a run before a total limit.",
        "- Calibration is counted once, independently of search. Validation tuning, repeated failed/restarted experiments, embeddings, other families and notebook calls are excluded.",
    ]
    (output / "estimate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(data["scenarios"], indent=2))
    print(f"Estimates: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Estimation error: {error}") from None
