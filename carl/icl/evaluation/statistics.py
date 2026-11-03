"""Board-balanced estimates, observed budget curves and explicit paired comparisons."""
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from carl.icl.evaluation.data import EvaluationConfig, SavedRun
from carl.icl.types import JSONObject


def groups(runs: list[SavedRun]) -> dict[str, list[SavedRun]]:
    """Partition conditions by all settings that must remain matched."""
    result: dict[str, list[SavedRun]] = defaultdict(list)
    for run in runs:
        result[run.cohort].append(run)
    return dict(result)


def conditions(runs: list[SavedRun]) -> dict[str, list[SavedRun]]:
    """Retain configuration names so ablations sharing a method never get pooled."""
    result: dict[str, list[SavedRun]] = defaultdict(list)
    for run in runs:
        result[run.record.config].append(run)
    return dict(result)


def board_means(
    runs: list[SavedRun], metric: Callable[[SavedRun], float]
) -> np.ndarray:
    """Give each board equal weight even when a partial run has missing seeds."""
    values: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        values[run.record.instance].append(metric(run))
    return np.array([np.mean(values[key]) for key in sorted(values)], dtype=float)


def interval(
    values: np.ndarray, config: EvaluationConfig
) -> tuple[float | None, float | None]:
    """Bootstrap independent boards; one observed board cannot provide an interval."""
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(config.bootstrap_seed)
    draws = rng.choice(values, (config.bootstrap_replicates, len(values))).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize(runs: list[SavedRun], config: EvaluationConfig) -> list[JSONObject]:
    """Expose denominators, missing trials and failures alongside success and cost."""
    result: list[JSONObject] = []
    for cohort, matched in groups(runs).items():
        for name, records in conditions(matched).items():
            first = records[0]
            outcomes = board_means(records, lambda run: float(run.record.solved))
            low, high = interval(outcomes, config)
            solved_lengths = [
                run.record.solution_actions
                for run in records
                if run.record.solution_actions is not None
            ]
            expected = first.settings.instances * len(first.settings.repeat_seeds)
            result.append(
                {
                    "cohort": cohort,
                    "label": first.label,
                    "config": name,
                    "method": first.record.method,
                    "family": first.record.family,
                    "provider": first.record.provider,
                    "prompt_profile_recorded": first.detail.get(
                        "_prompt_profile_recorded", False
                    ),
                    "model": first.settings.model,
                    "easy": first.settings.easy,
                    "split": first.record.split,
                    "tuning_status": first.record.tuning_status,
                    "runs": len(records),
                    "expected_runs": expected,
                    "missing_runs": max(0, expected - len(records)),
                    "boards": len(outcomes),
                    "solved": sum(run.record.solved for run in records),
                    "completion": float(outcomes.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "mean_attempted_actions": float(
                        np.mean([run.record.attempted_actions for run in records])
                    ),
                    "mean_solution_actions": float(np.mean(solved_lengths))
                    if solved_lengths
                    else None,
                    "mean_seconds": float(
                        np.mean([run.record.seconds for run in records])
                    ),
                    "api_errors": sum(
                        run.record.finish_reason == "api_error" for run in records
                    ),
                    "logical_requests": sum(
                        run.record.logical_requests for run in records
                    ),
                    "physical_requests": sum(run.record.requests for run in records),
                    "logical_input_tokens": sum(
                        run.record.logical_input_tokens for run in records
                    ),
                    "logical_output_tokens": sum(
                        run.record.logical_output_tokens for run in records
                    ),
                    "physical_input_tokens": sum(
                        run.record.input_tokens for run in records
                    ),
                    "physical_output_tokens": sum(
                        run.record.output_tokens for run in records
                    ),
                }
            )
    return result


def budget_curves(runs: list[SavedRun], config: EvaluationConfig) -> list[JSONObject]:
    """Observed cumulative completion, not counterfactual reruns with smaller budgets."""
    result: list[JSONObject] = []
    for cohort, matched in groups(runs).items():
        cap = min(run.settings.action_budget for run in matched)
        grid = sorted(
            set(np.linspace(0, cap, config.budget_points, dtype=int).tolist())
        )
        grid = sorted(
            set(grid)
            | {run.record.attempted_actions for run in matched if run.record.solved}
        )
        for name, records in conditions(matched).items():
            for budget in grid:
                def completed(run: SavedRun, limit: int = budget) -> float:
                    return float(run.record.solved and run.record.attempted_actions <= limit)

                outcomes = board_means(
                    records,
                    completed,
                )
                low, high = interval(outcomes, config)
                result.append(
                    {
                        "cohort": cohort,
                        "config": name,
                        "budget": budget,
                        "completion": float(outcomes.mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    return result


def _comparison_names(
    runs: list[SavedRun], config: EvaluationConfig
) -> list[tuple[str, str]]:
    available = conditions(runs)
    if config.comparison_configs:
        return [
            pair
            for pair in config.comparison_configs
            if all(name in available for name in pair)
        ]
    names = [
        [
            name
            for name, records in available.items()
            if records[0].record.method == method
        ]
        for method in config.comparison_methods
    ]
    return (
        [(names[0][0], names[1][0])] if all(len(group) == 1 for group in names) else []
    )


def paired(
    runs: list[SavedRun], config: EvaluationConfig
) -> tuple[list[JSONObject], list[JSONObject]]:
    """Pair by board and seed; report incomplete coverage and preserve outcome categories."""
    estimates: list[JSONObject] = []
    points: list[JSONObject] = []
    for cohort, matched in groups(runs).items():
        for left_name, right_name in _comparison_names(matched, config):
            left = {
                (r.record.instance, r.record.seed): r.record
                for r in matched
                if r.record.config == left_name
            }
            right = {
                (r.record.instance, r.record.seed): r.record
                for r in matched
                if r.record.config == right_name
            }
            keys = sorted(left.keys() & right.keys())
            if not keys:
                continue
            by_board: dict[str, list[float]] = defaultdict(list)
            for key in keys:
                first, second = left[key], right[key]
                by_board[key[0]].append(float(first.solved) - float(second.solved))
                points.append(
                    {
                        "cohort": cohort,
                        "left": left_name,
                        "right": right_name,
                        "instance": key[0],
                        "seed": key[1],
                        "left_actions": first.attempted_actions,
                        "right_actions": second.attempted_actions,
                        "left_solved": first.solved,
                        "right_solved": second.solved,
                    }
                )
            differences = np.array([np.mean(values) for values in by_board.values()])
            low, high = interval(differences, config)
            estimates.append(
                {
                    "cohort": cohort,
                    "left": left_name,
                    "right": right_name,
                    "paired_runs": len(keys),
                    "paired_boards": len(by_board),
                    "unpaired_left": len(left.keys() - right.keys()),
                    "unpaired_right": len(right.keys() - left.keys()),
                    "completion_delta": float(differences.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return estimates, points


def attempt_rows(runs: list[SavedRun]) -> list[JSONObject]:
    """Export completed subgoal attempts; interrupted API attempts may have no saved event."""
    return [
        {
            "cohort": run.cohort,
            "config": run.record.config,
            "instance": run.record.instance,
            "seed": run.record.seed,
            "attempt": index,
            "scale": event["scale"],
            "actions": event["attempted_actions"],
            "reached": event["reached"],
            "solved": event["solved"],
        }
        for run in runs
        for index, event in enumerate(run.detail.get("events", []))
    ]
