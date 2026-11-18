"""Evaluate ICL components against replay-verified offline Sokoban trajectories.

Reference actions are used only to construct targets and offline metrics; they
are never passed to the model as answers. Each invocation has its own bounded
API client and writes partial results after every case.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
from scipy.stats import spearmanr

from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows, to_rows, validate_subgoal
from carl.icl.config import ExperimentConfig, Limits, load_config
from carl.icl.datasets import environment, load_data, verify_solution
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.components import ICLComponents
from carl.icl.search import rollout
from carl.icl.trace import RecordingEnvironment, RecordingPolicy
from carl.icl.types import Board, TrainingData

ACTIONS: Final = ("UP", "DOWN", "LEFT", "RIGHT")


def parser() -> argparse.ArgumentParser:
    """Build the bounded diagnostic command-line interface."""
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", default="configs/icl/experiments/tuned_fixed.json")
    result.add_argument("--component", choices=("policy", "generator", "generated_policy", "value"), required=True)
    result.add_argument("--generator-report", type=Path,
                        help="Generator JSON to evaluate with --component generated_policy")
    result.add_argument("--split", choices=("validation", "test", "component_validation", "scale_validation"),
                        default="validation")
    result.add_argument("--private-data-dir", type=Path,
                        help="Explicit private witness root for campaign validation diagnostics")
    result.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    result.add_argument("--horizons", type=int, nargs="+", default=[4])
    result.add_argument("--positions", choices=("start", "mid"), nargs="+", default=["start", "mid"])
    result.add_argument("--seed", type=int, default=11)
    result.add_argument("--max-requests", type=int, default=100)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path)
    modes = result.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute", action="store_true", help="Use live API; incurs charges")
    modes.add_argument("--mock", action="store_true", help="Offline wiring check only")
    modes.add_argument("--replay", action="store_true", help="Use previously cached live responses")
    return result


def _components(config: ExperimentConfig, api: ExperimentAPI, training: TrainingData, board: Board) -> ICLComponents:
    return ICLComponents(config, api, environment(board), training)


def _policy_case(
    components: ICLComponents, source: Board, target: Board, horizon: int, seed: int
) -> dict[str, Any]:
    policy = RecordingPolicy(components)
    env = RecordingEnvironment(components.env, policy)
    result: dict[str, Any] = {"reached": False, "solved": False}
    try:
        attempt = rollout(env, policy, source, target, cap=horizon, rng=np.random.default_rng(seed))
        result.update(reached=attempt.reached, solved=attempt.solved)
    except (RequestFailed, BudgetExceeded) as error:
        result["error"] = str(error)
        result["budget_exhausted"] = isinstance(error, BudgetExceeded)
    result["steps"] = [
        {"action": ACTIONS[step.action], "probabilities": step.probabilities.tolist(),
         "changed": step.changed, "state": to_rows(step.successor)}
        for step in env.steps
    ]
    result["actions_attempted"] = len(env.steps)
    return result


def _generator_case(components: ICLComponents, source: Board, target: Board, horizon: int) -> dict[str, Any]:
    result: dict[str, Any] = {"valid_count": 0}
    try:
        proposals = components.propose(source, float(horizon))
        result["valid_count"] = len(proposals)
        if proposals:
            proposal = proposals[0]
            result.update(proposal=to_rows(proposal),
                          matches_reference=bool(np.array_equal(proposal, target)))
    except (RequestFailed, BudgetExceeded) as error:
        result["error"] = str(error)
        result["budget_exhausted"] = isinstance(error, BudgetExceeded)
    result["diagnostics"] = dict(components.diagnostics)
    return result


def _value_case(components: ICLComponents, state: Board, remaining: int) -> dict[str, Any]:
    result: dict[str, Any] = {"reference_return": components.config.discount ** remaining,
                              "terminal": remaining == 0}
    try:
        result["value"] = components.value(state)
    except (RequestFailed, BudgetExceeded) as error:
        result["error"] = str(error)
        result["budget_exhausted"] = isinstance(error, BudgetExceeded)
    return result


def _summary(component: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize observed cases; do not silently remove failed requests."""
    result: dict[str, Any] = {"cases": len(rows), "errors": sum("error" in row for row in rows)}
    if component in {"policy", "generated_policy"}:
        result["reached"] = sum(bool(row.get("reached")) for row in rows)
        result["by_horizon"] = {
            str(k): {"cases": sum(row["horizon"] == k for row in rows),
                     "reached": sum(row["horizon"] == k and bool(row.get("reached")) for row in rows)}
            for k in sorted({row["horizon"] for row in rows})
        }
    elif component == "generator":
        result["valid"] = sum(row.get("valid_count", 0) > 0 for row in rows)
        result["matches_reference"] = sum(bool(row.get("matches_reference")) for row in rows)
    else:
        valid = [row for row in rows if "value" in row]
        if valid:
            result["mae_vs_witness_return"] = float(np.mean([
                abs(row["value"] - row["reference_return"]) for row in valid
            ]))
        rank: dict[str, float | None] = {}
        for identifier in sorted({row["instance"] for row in valid}):
            selected = [row for row in valid if row["instance"] == identifier]
            predicted = [row["value"] for row in selected]
            reference = [row["reference_return"] for row in selected]
            correlation = (float(cast(Any, spearmanr(predicted, reference)).statistic)
                           if len(selected) > 1 and len(set(predicted)) > 1 else float("nan"))
            rank[identifier] = correlation if np.isfinite(correlation) else None
        result["spearman_by_trajectory"] = rank
    return result


def run(args: argparse.Namespace) -> Path:
    """Execute a reproducible diagnostic on the selected dataset split."""
    if args.component == "generated_policy":
        return run_generated_policy(args)
    if any(index < 0 for index in args.indices) or not args.horizons or any(k < 1 or k > 16 for k in args.horizons):
        raise ValueError("Indices must be nonnegative and horizons must be in 1..16")
    if len(args.indices) != len(set(args.indices)):
        raise ValueError("Indices must be unique")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{args.component}.json"
    if report_path.exists():
        raise ValueError(f"Output already exists: {report_path}; choose a new directory")
    base = load_config(args.config)
    if args.component == "generator" and base.method != "fixed":
        raise ValueError("Generator trajectory offsets require a fixed-horizon configuration")
    limits = Limits(max_requests=args.max_requests, max_input_tokens=20000,
                    max_total_input_tokens=2_000_000, max_output_tokens=160000,
                    wall_seconds=1800, timeout_seconds=120)
    config = ExperimentConfig.model_validate({**base.model_dump(), "split": args.split,
        "instances": max(args.indices) + 1, "easy": False, "limits": limits.model_dump(),
        **({"cache_dir": str(args.cache_dir.resolve())} if args.cache_dir else {})})
    training, instances = load_data(config)
    if (config.path("data_dir") / "campaign_dataset_manifest.json").exists():
        if config.split not in {"component_validation", "scale_validation"} or args.private_data_dir is None:
            raise ValueError("Campaign diagnostics require a validation split and explicit --private-data-dir")
        witness_path = args.private_data_dir.expanduser().resolve() / config.split / f"{config.family}.json"
        private = json.loads(witness_path.read_text(encoding="utf-8"))["witnesses"]
        witnesses = {item["board_id"]: item["actions"] for item in private}
    else:
        witness_path = config.path("data_dir") / "private_witnesses" / config.family / f"{config.split}.json"
        witnesses = {item["id"]: item["actions"] for item in json.loads(witness_path.read_text(encoding="utf-8"))}
    provider = "openai" if args.execute else "mock" if args.mock else "replay"
    credentials(provider)
    api = ExperimentAPI(config.path("cache_dir"), provider, config.limits)
    data_paths = [config.path("data_dir") / "training.json", witness_path,
                  config.path("data_dir") / config.family / f"{config.split}.json"]
    if config.value_examples == "trajectory":
        data_paths.append(config.path("data_dir") / "value_examples.json")
    report: dict[str, Any] = {
        "version": 1, "component": args.component, "provider": provider,
        "model": config.model, "config": config.model_dump(mode="json"),
        "indices": args.indices, "horizons": args.horizons, "positions": args.positions,
        "seed": args.seed, "witness_note": "Offline solution witness, not necessarily optimal; never provided as a query answer.",
        "data_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in data_paths},
        "cases": [], "status": "running",
    }
    started = time.monotonic()

    def save(status: str) -> None:
        report.update(status=status, metrics=dict(api.metrics), seconds=time.monotonic() - started,
                      summary=_summary(args.component, report["cases"]))
        write_json(report_path, report)

    save("running")
    for index in args.indices:
        item = instances[index]
        actions = witnesses[item["id"]]
        states = verify_solution(from_rows(item["state"]), actions)
        last = len(states) - 1
        if args.component == "value":
            jobs = [(position, 0) for position in sorted({0, last // 4, last // 2, 3 * last // 4, last})]
        else:
            starts = sorted({0 if position == "start" else last // 2 for position in args.positions})
            jobs = [(position, horizon) for position in starts for horizon in args.horizons if position + horizon <= last]
        for position, horizon in jobs:
            state = states[position]
            components = _components(config, api, training, state)
            row: dict[str, Any] = {"instance": item["id"], "index": index, "position": position,
                                   "remaining": last - position, "state": to_rows(state)}
            if args.component == "value":
                row.update(_value_case(components, state, last - position))
            else:
                target = states[position + horizon]
                row.update(horizon=horizon, target=to_rows(target),
                           reference_actions=[ACTIONS[a] for a in actions[position:position + horizon]])
                if args.component == "policy":
                    row.update(_policy_case(components, state, target, horizon, args.seed))
                else:
                    row.update(_generator_case(components, state, target, horizon))
            report["cases"].append(row)
            save("running")
            print(f"{args.component} {item['id']} at {position}, k={horizon}: "
                  f"{row.get('reached', row.get('valid_count', row.get('value', row.get('error'))))}", flush=True)
            if row.get("budget_exhausted"):
                save("budget_exhausted")
                return report_path
    save("completed_with_errors" if any("error" in row for row in report["cases"]) else "completed")
    return report_path


def run_generated_policy(args: argparse.Namespace) -> Path:
    """Test the frozen policy on actual generator outputs, including missing targets."""
    if args.generator_report is None:
        raise ValueError("--generator-report is required for generated_policy")
    generator_path = args.generator_report.expanduser().resolve()
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    if generator.get("component") != "generator" or generator.get("status") != "completed":
        raise ValueError("Expected a completed generator diagnostic")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "generated_policy.json"
    if path.exists():
        raise ValueError(f"Output already exists: {path}; choose a new directory")
    base = ExperimentConfig.model_validate(generator["config"])
    limits = Limits(max_requests=args.max_requests, max_input_tokens=20000,
                    max_total_input_tokens=2_000_000, max_output_tokens=160000,
                    wall_seconds=1800, timeout_seconds=120)
    config = ExperimentConfig.model_validate({**base.model_dump(), "limits": limits.model_dump(),
        **({"cache_dir": str(args.cache_dir.resolve())} if args.cache_dir else {})})
    training, _ = load_data(config)
    provider = "openai" if args.execute else "mock" if args.mock else "replay"
    credentials(provider)
    api = ExperimentAPI(config.path("cache_dir"), provider, config.limits)
    report: dict[str, Any] = {"version": 1, "component": "generated_policy", "provider": provider,
        "model": config.model, "config": config.model_dump(mode="json"),
        "generator_provider": generator["provider"],
        "generator_report": str(generator_path),
        "generator_sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
        "seed": args.seed, "cases": []}
    started = time.monotonic()

    def save(status: str) -> None:
        report.update(status=status, metrics=dict(api.metrics), seconds=time.monotonic() - started,
                      summary=_summary("generated_policy", report["cases"]))
        write_json(path, report)

    save("running")
    for candidate in generator["cases"]:
        row = {"instance": candidate["instance"], "horizon": candidate["horizon"],
               "position": candidate["position"], "state": candidate["state"],
               "reached": False}
        if "proposal" not in candidate:
            row["reason"] = "no_valid_generated_target"
        else:
            source = from_rows(candidate["state"])
            target = from_rows(candidate["proposal"])
            validate_subgoal(source, target)
            row["target"] = candidate["proposal"]
            components = _components(config, api, training, source)
            row.update(_policy_case(components, source, target, candidate["horizon"], args.seed))
        report["cases"].append(row)
        save("running")
        print(f"generated_policy {row['instance']}: {row['reached']}", flush=True)
        if row.get("budget_exhausted"):
            save("budget_exhausted")
            return path
    save("completed_with_errors" if any("error" in row for row in report["cases"]) else "completed")
    return path


def main() -> int:
    """Run a diagnostic with explicit provider selection."""
    args = parser().parse_args()
    path = run(args)
    print(f"Report: {path}")
    return 0 if json.loads(path.read_text(encoding="utf-8"))["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
