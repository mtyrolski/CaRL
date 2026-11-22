"""Plan, run or report the Fig. 1 exact-state Sokoban reachability pilot."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from time import monotonic

from filelock import FileLock, Timeout
from pydantic import TypeAdapter

from carl.icl.api import RequestFailed, write_json
from carl.icl.boards import from_rows
from carl.icl.config import ROOT, load_config
from carl.icl.data_sources.sokoban import environment
from carl.icl.evaluation.full_state_reachability import (
    Tariff, controller, describe, execute_trial, json_object, offsets, runtime_versions,
    saved_trials, trial_path, write_report,
)
from carl.icl.evaluation.reachability_design import ReachabilityCase, ReachabilityModel, ReachabilityStudy, WitnessFile, design, fixture
from carl.icl.full_state_policy import load_settings
from carl.icl.intents.generation.data import file_sha256
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject, TrainingData


def _freeze(path: Path, data: JSONObject) -> None:
    if path.exists():
        if json_object(path) != data:
            raise ValueError(f"Frozen input changed at {path}. Choose a new output directory.")
    else:
        write_json(path, data)


def _design(study: ReachabilityStudy, output: Path) -> tuple[TrainingData, JSONObject]:
    if study.data == "seeded_fixture":
        train, test = fixture(study, "train"), fixture(study, "test")
    else:
        assert study.training is not None and study.test is not None
        train = WitnessFile.model_validate_json(study.training.read_text(encoding="utf-8"))
        test = WitnessFile.model_validate_json(study.test.read_text(encoding="utf-8"))
    training, manifest = design(study, train, test)
    configs = {}
    for model in study.models:
        assert model.config is not None
        settings = load_settings(model.config) if model.kind == "local" else load_config(model.config)
        configs[model.id] = settings.model_dump(mode="json")
    manifest.update(study=study.model_dump(mode="json"), configs=configs, versions=runtime_versions(),
                    code_sha256={str(path): file_sha256(ROOT / path) for path in (
                        Path("carl/icl/evaluation/reachability_design.py"), Path("carl/icl/evaluation/full_state_reachability.py"),
                        Path("carl/icl/llm_components/policy.py"), Path("carl/icl/full_state_policy.py"),
                        Path("carl/icl/search.py"), Path("carl/environment/sokoban/env.py"),
                        Path("scripts/run_full_state_reachability.py"),
                    )})
    _freeze(output / "manifest.json", manifest)
    if study.data == "seeded_fixture":
        for dataset in (train, test):
            _freeze(output / f"{dataset.split}-fixture.json", dataset.model_dump(mode="json"))
    return training, manifest


def _run_model(
    model: ReachabilityModel, study: ReachabilityStudy, training: TrainingData, manifest: JSONObject,
    output: Path, *, live: bool,
) -> None:
    points = [[ReachabilityCase.model_validate(value) for value in manifest["cases"][str(k)]] for k in offsets(study, model)]
    prior = [trial for cases in points for trial in saved_trials(output, model, cases)]
    if any(trial.status == "budget_exhausted" for trial in prior):
        return
    env = environment(from_rows(points[0][0].source))
    started = monotonic()
    active = controller(model, env, training, output, live=live)
    _freeze(output / "policies" / f"{model.id}.json", active.identity)
    setup = output / "controller-setup" / f"{model.id}.json"
    if not setup.exists():
        write_json(setup, {"observed_seconds": monotonic() - started,
                           "offline_compute_seconds": model.offline_compute_seconds, "adaptation_seconds": model.adaptation_seconds})
    if active.api is not None:
        for key in set(active.api.metrics) - {"logical_token_byte_fallback"}:
            active.api.metrics[key] += sum(trial.accounting.get(key, 0) for trial in prior)
        active.api.started -= sum(trial.seconds for trial in prior)
    for cases in points:
        valid = sum(trial.status == "complete" for trial in saved_trials(output, model, cases))
        for case in cases:
            path = trial_path(output, model, case)
            if valid >= study.trials_per_point:
                break
            if path.is_file():
                continue
            trial = execute_trial(case, active, env)
            write_json(path, trial.model_dump(mode="json"))
            if trial.status == "budget_exhausted":
                return
            valid += trial.status == "complete"


def _run(args: argparse.Namespace) -> int:
    if args.report_only:
        with FileLock(args.output / ".run.lock", timeout=0):
            manifest = json_object(args.output / "manifest.json")
            if json_object(args.output / "identity.json")["manifest_sha256"] != contract_hash(manifest):
                raise ValueError("Frozen manifest fingerprint mismatch")
            study = ReachabilityStudy.model_validate(manifest["study"])
            write_report(study, manifest, args.output)
        return 0
    study = ReachabilityStudy.model_validate_json(args.config.read_text(encoding="utf-8"))
    models = [model for model in study.models if args.model in (None, model.id)]
    if not models:
        raise ValueError("Unknown --model")
    tariffs = TypeAdapter(dict[str, Tariff]).validate_json(args.prices.read_text(encoding="utf-8")) if args.prices else {}
    plan = describe(study, tariffs)
    if args.plan_only:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    if args.execute_live and (len(models) != 1 or args.model is None or models[0].kind != "api"):
        raise ValueError("Live requires one explicit API condition selected with --model")
    if any(model.kind == "api" for model in models) and not (args.execute_live or args.prepare_only):
        raise ValueError("Choose an offline model or explicitly enable the selected API condition with --execute-live")
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".run.lock", timeout=0):
        if not (output / "manifest.json").exists() and any(path.name != ".run.lock" for path in output.iterdir()):
            raise ValueError("Use a new output directory or one with a matching frozen manifest")
        training, manifest = _design(study, output)
        write_json(output / "plan.json", plan)
        _freeze(output / "identity.json", {"manifest_sha256": contract_hash(manifest)})
        if args.prepare_only:
            print(output / "manifest.json")
            return 0
        for model in models:
            _run_model(model, study, training, manifest, output, live=args.execute_live)
        write_report(study, manifest, output)
    result = json_object(output / "report.json")
    complete = all(row["status"] == "complete" for row in result["points"] if row["model"] in {m.id for m in models})
    print(output / "report.json")
    return 0 if complete else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Keep planning and offline reporting independent of live credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", help="Select one configured condition by ID")
    parser.add_argument("--prices", type=Path, help="Optional per-model token tariffs for the estimate")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    args = parser.parse_args(argv)
    if not args.report_only and args.config is None:
        parser.error("--config is required when planning or running")
    if not args.plan_only and args.output is None:
        parser.error("--output is required when running or reporting")
    try:
        return _run(args)
    except (ValueError, OSError, ImportError, RuntimeError, RequestFailed, Timeout) as error:
        return parser.exit(1, f"Reachability pilot failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
