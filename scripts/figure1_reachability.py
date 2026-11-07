"""Freeze and measure Figure 1-style CLLP reachability on verified trajectories.

Planning/preparation is offline. Mock results are plumbing checks, never model results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, model_validator

from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows, state_hash, to_rows
from carl.icl.config import ExperimentConfig, load_config
from carl.icl.components import ICLComponents
from carl.icl.datasets import environment, load_data, verify_solution
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.search import rollout

ROOT = Path(__file__).resolve().parents[1]


class MockByteEncoding:
    """Conservative byte count for offline mock plumbing, never live billing."""

    def encode(self, value: str, **_kwargs: object) -> bytes:
        return value.encode("utf-8")


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    data_dir: str
    family: str
    split: Literal["validation", "test", "component_validation", "scale_validation"] = "test"
    trajectories: int = Field(ge=1)
    contexts_per_trajectory: int = Field(default=1, ge=1)
    private_data_dir: str | None = None


class PolicyModelOptions(BaseModel):
    """Explicit per-model response settings; prompts and policy sampling remain shared."""

    model_config = ConfigDict(extra="forbid")
    policy_reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    policy_response_format: Literal["text", "json_schema"] | None = None
    policy_tokens: int | None = Field(default=None, ge=32)


class Study(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment_config: str
    output_dir: str
    cache_dir: str
    models: list[str] = Field(min_length=1)
    model_overrides: dict[str, PolicyModelOptions] = Field(default_factory=dict)
    datasets: list[Dataset] = Field(min_length=1)
    window: int = Field(ge=1)
    targets_per_context: int = Field(ge=1)
    trials: int = Field(ge=1)
    action_cap: int = Field(ge=1)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_study(self) -> Study:
        if len(self.models) != len(set(self.models)) or any(not x.strip() for x in self.models):
            raise ValueError("models must contain unique nonempty model IDs")
        if set(self.model_overrides) - set(self.models):
            raise ValueError("model_overrides keys must be configured model IDs")
        if len({item.name for item in self.datasets}) != len(self.datasets):
            raise ValueError("dataset names must be unique")
        if self.targets_per_context > self.window:
            raise ValueError("targets_per_context cannot exceed window")
        return self


def path(value: str) -> Path:
    item = Path(value).expanduser()
    return item if item.is_absolute() else ROOT / item


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_hash(item: Path) -> str:
    return sha(item.read_bytes())


def stable_seed(*parts: object) -> int:
    return int(sha(json.dumps(parts, separators=(",", ":")).encode())[:16], 16)


def sample_targets(states: list[np.ndarray], position: int, *, window: int, count: int, seed: int) -> list[int]:
    """Uniformly sample offsets of distinct future states, excluding the source."""
    if position < 0 or position >= len(states) or window < 1 or count < 1:
        raise ValueError("Invalid context, window or target count")
    seen = {state_hash(states[position])}
    offsets = []
    for k in range(1, min(window, len(states) - position - 1) + 1):
        digest = state_hash(states[position + k])
        if digest not in seen:
            offsets.append(k)
            seen.add(digest)
    if len(offsets) < count:
        raise ValueError("Not enough distinct future targets for this context")
    return sorted(int(k) for k in np.random.default_rng(seed).choice(offsets, count, replace=False))


def experiment_config(study: Study, dataset: Dataset, model: str) -> ExperimentConfig:
    base = load_config(study.experiment_config)
    if base.locked_policy_fingerprint or base.generator_checkpoint:
        raise ValueError("Use a policy config without a model-locked checkpoint or fingerprint")
    if model not in study.models:
        raise ValueError("Model is not part of this study")
    calibration = base.calibration.model_copy(update={"action_cap": study.action_cap})
    options = study.model_overrides.get(model, PolicyModelOptions()).model_dump(exclude_none=True)
    config = ExperimentConfig.model_validate({**base.model_dump(), **options, "model": model, "method": "direct", "data_dir": dataset.data_dir,
        "cache_dir": study.cache_dir, "family": dataset.family, "split": dataset.split,
        "instances": dataset.trajectories, "easy": False, "calibration": calibration.model_dump()})
    if config.policy_tokens > config.limits.max_output_tokens:
        raise ValueError(f"Policy token cap for {model!r} exceeds the per-trial output limit")
    return config


def witness_file(config: ExperimentConfig, dataset: Dataset) -> Path:
    if (config.path("data_dir") / "campaign_dataset_manifest.json").exists():
        if dataset.split not in {"component_validation", "scale_validation"} or not dataset.private_data_dir:
            raise ValueError("Campaign data require a validation split and explicit private_data_dir")
        return path(dataset.private_data_dir) / dataset.split / f"{dataset.family}.json"
    return config.path("data_dir") / "private_witnesses" / dataset.family / f"{dataset.split}.json"


def load_witnesses(filename: Path) -> dict[str, list[int]]:
    raw = json.loads(filename.read_text(encoding="utf-8"))
    rows = raw["witnesses"] if isinstance(raw, dict) else raw
    result = {row.get("id", row.get("board_id")): row["actions"] for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate witness IDs")
    return result


def freeze(study: Study, provider: str) -> dict:
    """Reconstruct and verify data, sample contexts/targets and bind the complete experiment."""
    files: dict[str, str] = {}
    contexts = []
    resolved = {}
    for dataset in study.datasets:
        config = experiment_config(study, dataset, study.models[0])
        root = config.path("data_dir")
        public_path = root / dataset.family / f"{dataset.split}.json"
        available = len(json.loads(public_path.read_text(encoding="utf-8"))["instances"])
        if available < dataset.trajectories:
            raise ValueError(f"Not enough trajectories in {dataset.name}")
        all_config = ExperimentConfig.model_validate({**config.model_dump(), "instances": available})
        training, all_instances = load_data(all_config)
        selection_rng = np.random.default_rng(stable_seed(study.seed, dataset.name, "trajectories"))
        selected = sorted(int(i) for i in selection_rng.choice(available, dataset.trajectories, replace=False))
        instances = [all_instances[i] for i in selected]
        witness_path = witness_file(config, dataset)
        witnesses = load_witnesses(witness_path)
        paths = [root / "training.json", root / dataset.family / f"{dataset.split}.json", witness_path]
        if config.value_examples == "trajectory":
            paths.append(root / "value_examples.json")
        campaign = root / "campaign_dataset_manifest.json"
        if campaign.exists():
            paths.append(campaign)
        else:
            paths.append(root / dataset.family / "validation.json")
        files.update({str(item.resolve()): file_hash(item) for item in paths})
        resolved[dataset.name] = {model: experiment_config(study, dataset, model).model_dump(mode="json")
                                  for model in study.models}
        for item in instances:
            if item["id"] not in witnesses:
                raise ValueError(f"Missing witness for {item['id']}")
            states = verify_solution(from_rows(item["state"]), witnesses[item["id"]])
            candidates = []
            for position in range(max(0, len(states) - study.window)):
                try:
                    offsets = sample_targets(states, position, window=study.window,
                        count=study.targets_per_context, seed=stable_seed(study.seed, dataset.name, item["id"], position))
                except ValueError:
                    continue
                candidates.append((position, offsets))
            if len(candidates) < dataset.contexts_per_trajectory:
                raise ValueError(f"Insufficient eligible full-window contexts in {item['id']}")
            rng = np.random.default_rng(stable_seed(study.seed, dataset.name, item["id"], "contexts"))
            chosen = rng.choice(len(candidates), dataset.contexts_per_trajectory, replace=False)
            for index in sorted(int(i) for i in chosen):
                position, offsets = candidates[index]
                context_id = f"{dataset.name}:{item['id']}:{position}"
                source = states[position]
                contexts.append({"dataset": dataset.name, "family": dataset.family, "trajectory_id": item["id"],
                    "context_id": context_id, "position": position, "state": to_rows(source),
                    "state_hash": state_hash(source), "targets": [
                        {"target_id": f"{context_id}:{k}", "k": k, "state": to_rows(states[position + k]),
                         "state_hash": state_hash(states[position + k])} for k in offsets]})
        # Ensure the exact training bytes and source versions were loaded above.
        if not training["contexts"]:
            raise ValueError("No training contexts")
    code_paths = sorted((ROOT / "carl/icl").rglob("*.py")) + [ROOT / "scripts/figure1_reachability.py",
        ROOT / "scripts/figure1_report.py", ROOT / "carl/environment/sokoban/env.py", ROOT / "carl/environment/sokoban/core.py"]
    return {"version": 1, "study": study.model_dump(mode="json"), "provider": provider,
        "resolved_config": resolved, "data_sha256": files,
        "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in code_paths},
        "packages": {name: version(name) for name in ("numpy", "openai", "pydantic", "tiktoken")},
        "contexts": contexts,
        "sampling": "uniform trajectories, uniform eligible contexts with a full window; uniform distinct future states without replacement; earliest offset represents repeated state",
        "randomness": "Independent seeded categorical actions conditional on cached policy distributions; provider draws are not independently seeded.",
        "token_accounting": "UTF-8 byte count for mock only; o200k_base estimates for live and replay.",
        "interpretation": "Exact state equality; task terminal before target is not a hit. d is measured, not assumed equal to k."}


def prepare(study: Study, provider: str) -> dict:
    manifest = freeze(study, provider)
    directory = path(study.output_dir)
    target = directory / "manifest.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Freeze differs from existing manifest; choose a new output directory")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Output directory is not empty and has no manifest")
        write_json(target, manifest)
    return manifest


def job_rows(manifest: dict) -> list[dict]:
    study = Study.model_validate(manifest["study"])
    rows = []
    for context in manifest["contexts"]:
        for target in context["targets"]:
            for model in study.models:
                for trial in range(study.trials):
                    # Common random numbers across models; API outputs remain model-specific.
                    seed = stable_seed(study.seed, context["context_id"], target["k"], trial)
                    rows.append({"dataset": context["dataset"], "family": context["family"],
                        "model": model, "context_id": context["context_id"], "target_id": target["target_id"],
                        "trajectory_id": context["trajectory_id"], "position": context["position"], "k": target["k"],
                        "source_hash": context["state_hash"], "target_hash": target["state_hash"],
                        "trial": trial, "seed": seed, "status": "missing"})
    return rows


def job_path(directory: Path, row: dict) -> Path:
    key = sha(json.dumps([row["model"], row["target_id"], row["trial"]], separators=(",", ":")).encode())
    return directory / "trials" / f"{key}.json"


def saved_rows(manifest: dict, directory: Path) -> list[dict]:
    rows = job_rows(manifest)
    for row in rows:
        item = job_path(directory, row)
        if item.exists():
            result = json.loads(item.read_text(encoding="utf-8"))
            if any(result.get(key) != value for key, value in row.items() if key != "status"):
                raise ValueError(f"Trial does not match freeze: {item}")
            row.update(result)
    return rows


def run(study: Study, provider: str, *, resume: bool = False) -> None:
    directory = path(study.output_dir)
    if not (directory / "manifest.json").exists():
        raise ValueError("Run --prepare for this provider before execution")
    expected = freeze(study, provider)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest != expected:
        raise ValueError("Manifest differs from current inputs, provider or code; use a new output directory")
    credentials(provider)
    try:
        with FileLock(str(directory / ".run.lock"), timeout=0):
            _run_locked(study, provider, manifest, directory, resume)
    except Timeout as exc:
        raise RuntimeError("Experiment is already running") from exc


def _run_locked(study: Study, provider: str, manifest: dict, directory: Path, resume: bool) -> None:
    existing = list((directory / "trials").glob("*.json")) if (directory / "trials").exists() else []
    if existing and not resume:
        raise ValueError("Existing trials: pass --resume")
    contexts = {row["context_id"]: row for row in manifest["contexts"]}
    datasets = {row.name: row for row in study.datasets}
    training_cache = {}
    for row in saved_rows(manifest, directory):
        if row["status"] == "complete":
            continue
        if row["status"] == "error":
            history = row.get("previous_errors", [])
            history.append({key: row[key] for key in ("error", "error_type", "metrics") if key in row})
            row["previous_errors"] = history
            row.pop("error", None)
            row.pop("error_type", None)
        dataset = datasets[row["dataset"]]
        config = experiment_config(study, dataset, row["model"])
        if dataset.name not in training_cache:
            training_cache[dataset.name] = load_data(config)[0]
        context = contexts[row["context_id"]]
        target = next(item for item in context["targets"] if item["target_id"] == row["target_id"])
        source_board, target_board = from_rows(context["state"]), from_rows(target["state"])
        encoding = MockByteEncoding() if provider == "mock" else None
        api = ExperimentAPI(config.path("cache_dir"), provider, config.limits, encoding=encoding)
        components = ICLComponents(config, api, environment(source_board), training_cache[dataset.name])
        try:
            attempt = rollout(components.env, components, source_board, target_board, cap=study.action_cap,
                              rng=np.random.default_rng(row["seed"]))
            row.update(status="complete", exact_hit=bool(attempt.reached),
                first_hit=len(attempt.actions) if attempt.reached else None,
                solved_before_target=bool(attempt.solved and not attempt.reached),
                actions_attempted=len(attempt.actions), actions=attempt.actions,
                endpoint_hash=state_hash(attempt.state), metrics=dict(api.metrics),
                policy_fingerprint=components.fingerprint())
        except (RequestFailed, BudgetExceeded) as exc:
            row.update(status="error", error_type=type(exc).__name__, error=str(exc), metrics=dict(api.metrics))
            write_json(job_path(directory, row), row)
            raise
        write_json(job_path(directory, row), row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icl/figure1/example.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("plan", "prepare", "execute", "mock", "replay", "report", "prepare-data", "verify-data"):
        mode.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--data-source", type=Path, default=ROOT / "data/icl/component_study_100_clean")
    parser.add_argument("--data-output", type=Path, default=ROOT / "data/icl/reachability_exploratory_v2")
    parser.add_argument("--data-seed", type=int, default=20261103)
    parser.add_argument("--data-count", type=int, default=100)
    parser.add_argument("--provider", choices=("openai", "mock", "replay"), default="mock",
                        help="Provider to bind on prepare; execution mode must match")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare_data or args.verify_data:
        from carl.icl.reachability_data import prepare_dataset, verify_dataset
        result = (prepare_dataset(args.data_source, args.data_output, seed=args.data_seed, n_main=args.data_count)
                  if args.prepare_data else verify_dataset(args.data_output))
        print(json.dumps({"output": str(args.data_output), "schema": result["schema"],
                          "status": "prepared" if args.prepare_data else "verified",
                          "classification": result["classification"]}, indent=2))
        return
    study = Study.model_validate_json(path(args.config).read_text(encoding="utf-8"))
    if args.plan:
        policy_settings = {}
        for model in study.models:
            config = experiment_config(study, study.datasets[0], model)
            policy_settings[model] = {
                "reasoning_effort": config.policy_reasoning_effort or config.reasoning_effort,
                "response_format": config.policy_response_format or config.response_format,
                "policy_tokens": config.policy_tokens,
            }
        print(json.dumps({"datasets": [d.model_dump() for d in study.datasets], "models": study.models,
            "policy_settings": policy_settings,
            "contexts": sum(d.trajectories * d.contexts_per_trajectory for d in study.datasets),
            "targets_per_context": study.targets_per_context, "trials_per_target": study.trials,
            "action_cap": study.action_cap, "max_policy_calls": sum(d.trajectories * d.contexts_per_trajectory
                for d in study.datasets) * study.targets_per_context * study.trials * len(study.models)
                * study.action_cap * (load_config(study.experiment_config).json_max_retries + 1),
            "note": "Upper bound, not a cost estimate; calls can end early and cached responses are reused."}, indent=2))
    elif args.prepare:
        result = prepare(study, args.provider)
        print(f"Prepared {len(result['contexts'])} contexts for {args.provider} in {study.output_dir}")
    elif args.report:
        from scripts.figure1_report import write_report
        directory = path(study.output_dir)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest != freeze(study, manifest["provider"]):
            raise ValueError("Manifest differs from current inputs or code")
        write_report(saved_rows(manifest, directory), directory, study.trials, provider=manifest["provider"])
    else:
        provider = "openai" if args.execute else "replay" if args.replay else "mock"
        run(study, provider, resume=args.resume)


if __name__ == "__main__":
    main()
