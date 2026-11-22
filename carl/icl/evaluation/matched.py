"""Validate a planned sweep and compare saved outcomes within each realized budget."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Annotated, Final, Literal, cast

import numpy as np
from pydantic import Field, JsonValue, StringConstraints, model_validator

from carl.icl.api import write_json
from carl.icl.config import Settings
from carl.icl.evaluation.data import EvaluationConfig
from carl.icl.evaluation.statistics import interval
from carl.icl.types import JSONObject

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Seconds = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Count = Annotated[int, Field(strict=True, ge=0)]
Measure = Count | Seconds

TIME_RULE: Final = (
    "Record elapsed runtime separately. Use measured first success time when available, "
    "otherwise completion time for a solved trial. Cap that time at the configured wall horizon. "
    "A recorded trial ending without success receives the full horizon for this restricted time "
    "measure. The observed censor time is the earlier of its recorded stop and the configured horizon. "
    "Missing trials and solved trials with unknown success time are excluded from the time contrast."
)
PAIR_RULE: Final = (
    "Pair within the same realized limits, task, seed, initial snapshot, and frozen policy. "
    "Average repeated pairs within their board or task family and give each observed group equal "
    "weight. Bootstrap whole groups with replacement for pointwise 95% intervals. At least two "
    "groups are needed for an interval. Missing trials are not coded as failures."
)
RESOURCE_METRICS: Final = (
    "attempted_simulator_actions", "executed_plan_actions", "logical_requests", "physical_requests",
)


def _canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


class SweepBudget(Settings):
    """A labeled resolved limit vector, kept intact when deciding whether trials match."""

    id: Name
    limits: dict[str, JsonValue]

    @model_validator(mode="after")
    def check_limits(self) -> "SweepBudget":
        """Validate the caps required for accounting and restricted time."""
        actions = self.limits.get("action_budget")
        wall = self.limits.get("wall_seconds")
        if isinstance(actions, bool) or not isinstance(actions, int) or actions < 1:
            raise ValueError("limits.action_budget must be a positive integer")
        if isinstance(wall, bool) or not isinstance(wall, (float, int)) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("limits.wall_seconds must be a finite positive number")
        _canonical(self.limits)
        return self

    @property
    def horizon(self) -> float:
        """The configured wall horizon common to all modes in this budget."""
        return float(cast(float | int, self.limits["wall_seconds"]))


class SweepInstance(Settings):
    """One scheduled task/seed instance, with an experimental resampling group."""

    id: Name
    group: Name
    seed: Count
    policy_fingerprint: Name
    initial_fingerprint: Name
    task_id: Name | None = None


class SweepTrial(Settings):
    """One observed execution; scheduled trials with no record are represented by absence."""

    budget_id: Name
    instance_id: Name
    mode: Name
    status: Name
    solved: Annotated[bool, Field(strict=True)]
    elapsed_seconds: Seconds | None
    accounting: dict[str, Measure | None]
    artifact: Name
    limits: dict[str, JsonValue]
    cost_usd: Seconds | None = None
    success_seconds: Seconds | None = None
    policy_fingerprint: Name | None = None
    initial_fingerprint: Name | None = None
    task_id: Name | None = None
    seed: Count | None = None
    finish_reason: Name | None = None

    @model_validator(mode="after")
    def check_outcome(self) -> "SweepTrial":
        """Reject contradictory success evidence."""
        if self.finish_reason is not None and self.solved != (self.finish_reason == "solved"):
            raise ValueError("solved and finish_reason disagree")
        if self.status in {"solved", "failed"} and self.solved != (self.status == "solved"):
            raise ValueError("solved and status disagree")
        if self.status in {"missing", "not_started"}:
            raise ValueError("Represent a scheduled trial without a result by omitting the trial")
        if self.success_seconds is not None:
            if not self.solved:
                raise ValueError("Only a solved trial can record success_seconds")
            if self.elapsed_seconds is not None and self.success_seconds > self.elapsed_seconds:
                raise ValueError("success_seconds cannot exceed elapsed_seconds")
        return self

    @property
    def success_time(self) -> float | None:
        """Use elapsed time when the runner records only completion time."""
        if self.solved:
            return self.elapsed_seconds if self.success_seconds is None else self.success_seconds
        return None

    def restricted_time(self, horizon: float) -> float | None:
        """Measure time to success through the common configured wall horizon."""
        if not self.solved:
            return horizon
        success_time = self.success_time
        return min(success_time, horizon) if success_time is not None else None


def _unique_ids(items: Sequence[SweepBudget | SweepInstance], label: str) -> None:
    if len({item.id for item in items}) != len(items):
        raise ValueError(f"Duplicate {label} identifiers")


class MatchedSweep(Settings):
    """Planned cross-product of instances, budgets and modes plus observations on disk."""

    schema_version: Literal[1]
    domain: Name
    execution_mode: Name
    versions: dict[Name, Name] = Field(min_length=1)
    split_fingerprint: Name
    config: dict[str, JsonValue]
    offline_compute_seconds: Seconds | None = None
    adaptation_seconds: Seconds | None = None
    modes: list[Name] = Field(min_length=2)
    budgets: list[SweepBudget] = Field(min_length=1)
    instances: list[SweepInstance] = Field(min_length=1)
    trials: list[SweepTrial]

    @model_validator(mode="after")
    def check_design(self) -> "MatchedSweep":
        """Reject duplicate comparisons and records outside their scheduled contract."""
        _unique_ids(self.budgets, "budget")
        _unique_ids(self.instances, "instance")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("Duplicate sweep modes")
        if len({_canonical(budget.limits) for budget in self.budgets}) != len(self.budgets):
            raise ValueError("Budget identifiers must describe different resolved limits")
        physical_instances = {item.initial_fingerprint for item in self.instances}
        identities = {(item.task_id, item.initial_fingerprint, item.seed) for item in self.instances}
        if len(identities) != len(self.instances):
            raise ValueError("The same task, initial snapshot and seed occur under multiple instance IDs")
        if len({(item.initial_fingerprint, item.group) for item in self.instances}) != len(physical_instances):
            raise ValueError("Repeated snapshots must share their bootstrap group")
        if len({item.policy_fingerprint for item in self.instances}) != 1:
            raise ValueError("A matched sweep must hold the frozen policy fixed")
        limits = {item.id: item.limits for item in self.budgets}
        instances = {item.id: item for item in self.instances}
        seen: set[tuple[str, str, str]] = set()
        for trial in self.trials:
            key = (trial.budget_id, trial.instance_id, trial.mode)
            if trial.budget_id not in limits or trial.instance_id not in instances or trial.mode not in self.modes:
                raise ValueError(f"Trial outside the sweep design: {key}")
            if key in seen:
                raise ValueError(f"Duplicate matched trial: {key}")
            self._check_trial(trial, limits[trial.budget_id], instances[trial.instance_id])
            seen.add(key)
        _canonical(self.config)
        return self

    @staticmethod
    def _check_trial(trial: SweepTrial, limits: dict[str, JsonValue], instance: SweepInstance) -> None:
        if _canonical(trial.limits) != _canonical(limits):
            raise ValueError(f"Actual limits differ from budget {trial.budget_id} for {trial.artifact}")
        for attribute in ("task_id", "seed", "policy_fingerprint", "initial_fingerprint"):
            observed = getattr(trial, attribute)
            if observed is not None and observed != getattr(instance, attribute):
                raise ValueError(f"Mismatched trial {attribute} for {trial.artifact}")
        api = limits.get("api")
        api_limits = api if isinstance(api, dict) else {}
        for count, cap, label in (
            ("attempted_simulator_actions", limits["action_budget"], "Action budget"),
            ("expansions", limits.get("max_expansions"), "Expansion budget"),
            ("logical_requests", api_limits.get("max_requests"), "Logical request budget"),
            ("logical_input_tokens", api_limits.get("max_total_input_tokens"), "Logical input token budget"),
        ):
            observed = trial.accounting.get(count)
            if observed is not None and isinstance(cap, (int, float)) and not isinstance(cap, bool) and observed > cap:
                raise ValueError(f"{label} exceeded for {trial.artifact}")


Pair = tuple[SweepInstance, SweepTrial, SweepTrial]


def _cluster_values(
    pairs: Sequence[Pair], metric: Callable[[SweepTrial, SweepTrial], float | None]
) -> tuple[np.ndarray, int]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for instance, left, right in pairs:
        value = metric(left, right)
        if value is not None:
            clusters[instance.group].append(value)
    values = np.asarray([mean(clusters[group]) for group in sorted(clusters)], dtype=float)
    return values, sum(map(len, clusters.values()))


def _estimate(name: str, values: np.ndarray, config: EvaluationConfig) -> JSONObject:
    low, high = interval(values, config)
    return {
        name: float(values.mean()) if values.size else None,
        f"{name}_ci95_low": low,
        f"{name}_ci95_high": high,
    }


def _trial_row(
    budget: SweepBudget, instance: SweepInstance, mode: str, trial: SweepTrial | None,
) -> JSONObject:
    horizon = budget.horizon
    common = {
        "budget_id": budget.id,
        "instance_id": instance.id,
        "task_id": instance.task_id,
        "group": instance.group,
        "seed": instance.seed,
        "mode": mode,
        "policy_fingerprint": instance.policy_fingerprint,
        "initial_fingerprint": instance.initial_fingerprint,
        "horizon_seconds": horizon,
        "limits": budget.limits,
    }
    if trial is None:
        return {**common, "status": "missing", **dict.fromkeys((
            "solved", "elapsed_seconds", "finish_reason", "accounting", "artifact", "cost_usd", "success_seconds",
            "time_to_success_seconds", "event_within_horizon", "observed_censor_seconds",
            "restricted_time_to_success_seconds", *RESOURCE_METRICS,
        ))}
    success = trial.success_time
    reason = trial.finish_reason or trial.status
    event = success <= horizon if success is not None else (None if trial.solved else False)
    censor = min(trial.elapsed_seconds, horizon) if event is False and trial.elapsed_seconds is not None else None
    return {
        **trial.model_dump(mode="json"),
        **common,
        "finish_reason": reason,
        "time_to_success_seconds": success,
        "event_within_horizon": event,
        "observed_censor_seconds": censor,
        "restricted_time_to_success_seconds": trial.restricted_time(horizon),
        **{key: trial.accounting.get(key) for key in RESOURCE_METRICS},
    }


def _arm_summary(budget: SweepBudget, mode: str, rows: list[JSONObject]) -> JSONObject:
    actual = [r for r in rows if r["status"] != "missing"]
    costs = [r["cost_usd"] for r in actual if r["cost_usd"] is not None]
    successes = [r["time_to_success_seconds"] for r in actual if r["time_to_success_seconds"] is not None]
    return {
        "budget_id": budget.id,
        "mode": mode,
        "limits": budget.limits,
        "planned_trials": len(rows),
        "observed_trials": len(actual),
        "coverage": len(actual) / len(rows),
        "missing_trials": len(rows) - len(actual),
        "missing_instances": [r["instance_id"] for r in rows if r["status"] == "missing"],
        "solved_trials": sum(r["solved"] for r in actual),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in actual)),
        "successes_with_measured_time": len(successes),
        "median_observed_success_seconds": float(np.median(successes)) if successes else None,
        "known_cost_trials": len(costs),
        "observed_cost_usd": sum(costs) if costs else None,
        **{f"total_{metric}": _known_total(actual, metric) for metric in RESOURCE_METRICS},
    }


def _known_total(rows: list[JSONObject], metric: str) -> int | float | None:
    values = [row[metric] for row in rows]
    return sum(values) if values and all(value is not None for value in values) else None


def _contrast(
    budget: SweepBudget, left_mode: str, right_mode: str, instances: list[SweepInstance],
    observations: dict[tuple[str, str, str], SweepTrial], *, config: EvaluationConfig,
) -> JSONObject:
    pairs: list[Pair] = []
    missing_left: list[str] = []
    missing_right: list[str] = []
    missing_both: list[str] = []
    for instance in instances:
        left = observations.get((budget.id, instance.id, left_mode))
        right = observations.get((budget.id, instance.id, right_mode))
        if left is None and right is None:
            missing_both.append(instance.id)
        elif left is None:
            missing_left.append(instance.id)
        elif right is None:
            missing_right.append(instance.id)
        else:
            pairs.append((instance, left, right))

    def time_difference(left: SweepTrial, right: SweepTrial) -> float | None:
        lhs, rhs = left.restricted_time(budget.horizon), right.restricted_time(budget.horizon)
        return lhs - rhs if lhs is not None and rhs is not None else None

    success, _ = _cluster_values(pairs, lambda left, right: float(left.solved) - float(right.solved))
    times, timed_pairs = _cluster_values(pairs, time_difference)
    return {
        "budget_id": budget.id,
        "limits": budget.limits,
        "left": left_mode,
        "right": right_mode,
        "expected_pairs": len(instances),
        "paired_trials": len(pairs),
        "paired_groups": int(success.size),
        "missing_left_only": missing_left,
        "missing_right_only": missing_right,
        "missing_both": missing_both,
        "timed_pairs": timed_pairs,
        "timed_groups": int(times.size),
        **_estimate("success_delta", success, config),
        **_estimate("restricted_time_delta_seconds", times, config),
    }


def matched_tables(sweep: MatchedSweep, config: EvaluationConfig) -> dict[str, list[JSONObject]]:
    """Build separate tables for coverage, paired estimates and every planned trial."""
    observations = {(row.budget_id, row.instance_id, row.mode): row for row in sweep.trials}
    trials: list[JSONObject] = []
    coverage: list[JSONObject] = []
    comparisons: list[JSONObject] = []
    for budget in sweep.budgets:
        for mode in sweep.modes:
            rows = [
                _trial_row(budget, instance, mode, observations.get((budget.id, instance.id, mode)))
                for instance in sweep.instances
            ]
            trials.extend(rows)
            coverage.append(_arm_summary(budget, mode, rows))
        comparisons.extend(
            _contrast(budget, left, right, sweep.instances, observations, config=config)
            for left, right in combinations(sweep.modes, 2)
        )
    return {"coverage": coverage, "comparisons": comparisons, "trials": trials}


def _write_csv(path: Path, rows: list[JSONObject]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: _canonical(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def _metric(value: float | None) -> str:
    return "—" if value is None else f"{value:+.4g}"


def _markdown(sweep: MatchedSweep, tables: dict[str, list[JSONObject]]) -> str:
    lines = ["# Matched sweep evaluation", "", PAIR_RULE, "", TIME_RULE, "", (
        f"Offline compute took {_metric(sweep.offline_compute_seconds)} seconds. "
        f"Adaptation took {_metric(sweep.adaptation_seconds)} seconds."
    ), "", (
        "| Budget | Left | Right | Paired / expected | Groups | Success difference (95% CI) | "
        "Restricted time difference in seconds (95% CI) |"
    ), "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for row in tables["comparisons"]:
        values = []
        for name in ("success_delta", "restricted_time_delta_seconds"):
            low, high = _metric(row[f"{name}_ci95_low"]), _metric(row[f"{name}_ci95_high"])
            values.append(f"{_metric(row[name])} [{low}, {high}]")
        lines.append(
            f"| {row['budget_id']} | {row['left']} | {row['right']} | "
            f"{row['paired_trials']} / {row['expected_pairs']} | {row['paired_groups']} | "
            f"{values[0]} | {values[1]} |"
        )
    lines.extend(("", "## Coverage and cost", "", (
        "| Budget | Mode | Observed / planned | Solved | Simulator attempts | Logical calls | "
        "Physical calls | Observed USD (known trials) |"
    ), "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"))
    for row in tables["coverage"]:
        workload = [row[key] for key in (
            "total_attempted_simulator_actions", "total_logical_requests", "total_physical_requests",
        )]
        attempted, logical, physical = ("—" if value is None else str(value) for value in workload)
        usd = "—" if row["observed_cost_usd"] is None else f"{row['observed_cost_usd']:.4f}"
        lines.append(
            f"| {row['budget_id']} | {row['mode']} | {row['observed_trials']} / {row['planned_trials']} | "
            f"{row['solved_trials']} | {attempted} | {logical} | {physical} | {usd} ({row['known_cost_trials']}) |"
        )
    lines.extend(("", "Details and missing trial IDs are in `comparisons.csv`, `coverage.csv` and `trials.csv`."))
    return "\n".join(lines) + "\n"


def evaluate_matched(source: Path, output: Path, config: EvaluationConfig) -> Path:
    """Validate saved trials, then write a self-contained report and tidy CSVs."""
    raw = source.read_bytes()
    sweep = MatchedSweep.model_validate_json(raw)
    tables = matched_tables(sweep, config)
    report = output / "evaluation.json"
    write_json(report, {
        "schema_version": 1,
        "source": {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest()},
        "design": sweep.model_dump(mode="json", exclude={"trials"}),
        "bootstrap": {"seed": config.bootstrap_seed, "replicates": config.bootstrap_replicates},
        "pairing_method": PAIR_RULE,
        "time_method": TIME_RULE,
        **tables,
    })
    for name, rows in tables.items():
        _write_csv(output / f"{name}.csv", rows)
    (output / "REPORT.md").write_text(_markdown(sweep, tables), encoding="utf-8")
    return report
