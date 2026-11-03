"""Evaluate Luna components on matched offline trajectories, with resumable cases."""
import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

from pydantic import Field, model_validator

from carl.icl.api import write_json
from carl.icl.boards import from_rows, to_rows
from carl.icl.config import ROOT, ExperimentConfig, Limits, Settings, load_config
from carl.icl.datasets import environment, load_data, verify_solution
from carl.icl.experiment import credentials
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from carl.icl.types import Instance, Provider, TrainingData
from scripts.evaluate_trajectory_components import _generator_case, _policy_case, _value_case

COMPONENTS: Final = ("policy", "generator", "value")
DEFAULT_CONFIG: Final = "configs/icl/component_study/luna_100.json"
CODE_FILES: Final = (
    "scripts/evaluate_component_study.py", "scripts/evaluate_trajectory_components.py",
    "carl/icl/luna.py", "carl/icl/api.py", "carl/icl/experiment_api.py",
    "carl/icl/search.py", "carl/icl/trace.py", "carl/icl/boards.py", "carl/icl/responses.py",
    "carl/icl/config.py", "carl/icl/datasets.py", "carl/icl/types.py",
    "carl/environment/sokoban/env.py", "carl/environment/sokoban/core.py",
)


class StudyConfig(Settings):
    """Explicit, serializable settings for a paired component evaluation."""

    experiment_config: str
    data_dir: str
    output_dir: str
    cache_dir: str
    model: str
    instances: int = Field(ge=1)
    few_shot: list[int] = Field(min_length=1)
    horizon: int = Field(ge=1, le=16)
    value_positions: list[float] = Field(min_length=2)
    seed: int = Field(ge=0)
    monotonic_tolerance: float = Field(ge=0)
    limits: Limits

    @model_validator(mode="after")
    def check_variants(self) -> "StudyConfig":
        """Require ordered, non-duplicated conditions and trajectory positions."""
        if self.few_shot != sorted(set(self.few_shot)) or self.few_shot[0] < 0:
            raise ValueError("few_shot must contain distinct nonnegative counts in increasing order")
        if (self.value_positions != sorted(set(self.value_positions))
                or self.value_positions[0] != 0 or self.value_positions[-1] != 1):
            raise ValueError("value_positions must be distinct, sorted and span 0 to 1")
        return self


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_study(path: str | Path) -> StudyConfig:
    """Load a study file relative to the checkout."""
    return StudyConfig.model_validate_json(_path(path).read_text(encoding="utf-8"))


def experiment_config(study: StudyConfig, shots: int) -> ExperimentConfig:
    """Use exactly one value demonstration per selected training context."""
    base = load_config(study.experiment_config)
    return ExperimentConfig.model_validate({
        **base.model_dump(), "data_dir": study.data_dir, "cache_dir": study.cache_dir,
        "model": study.model, "few_shot": shots, "instances": study.instances,
        "split": "test", "easy": False, "method": "fixed", "value_examples": "initial",
        "limits": study.limits.model_dump(),
    })


def plan(study: StudyConfig) -> dict[str, Any]:
    """Provide a conservative request and completion-token ceiling, before API access."""
    config = experiment_config(study, max(study.few_shot))
    per_variant = {
        "policy": study.instances * study.horizon,
        "generator_and_policy": study.instances * (config.generator_max_attempts + study.horizon),
        "value": study.instances * len(study.value_positions),
    }
    output_ceiling = study.instances * (
        2 * study.horizon * config.policy_tokens
        + config.generator_max_attempts * config.generator_tokens
        + len(study.value_positions) * config.value_tokens
    )
    return {
        "model": study.model, "instances_per_variant": study.instances,
        "few_shot": study.few_shot, "horizon": study.horizon,
        "value_positions": study.value_positions,
        "max_requests_per_variant": per_variant,
        "max_requests_all_variants": sum(per_variant.values()) * len(study.few_shot),
        "max_estimated_input_tokens_all_variants": sum(per_variant.values()) * len(study.few_shot) * study.limits.max_input_tokens,
        "max_output_tokens_all_variants": output_ceiling * len(study.few_shot),
        "note": "Ceilings, not predictions: early stops, cache hits and actual response lengths reduce physical usage."
    }


def manifest(study: StudyConfig, config: ExperimentConfig, provider: Provider) -> dict[str, Any]:
    """Freeze inputs and prompts so resumed cases cannot mix different studies."""
    root = config.path("data_dir")
    files = [root / "training.json", root / config.family / "test.json",
             root / "private_witnesses" / config.family / "test.json"]
    return {"version": 1, "study": study.model_dump(mode="json"), "provider": provider,
            "resolved_config": config.model_dump(mode="json"),
            "data_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
            "code_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in CODE_FILES},
            "package_versions": {name: version(name) for name in ("openai", "numpy", "pydantic", "tiktoken", "scipy")}}


def case_result(
    component: str, study: StudyConfig, config: ExperimentConfig, training: TrainingData,
    *, item: Instance, actions: list[int], provider: Provider, client: Any,
) -> dict[str, Any]:
    """Evaluate one trajectory using the same controller as the actual solver."""
    states = verify_solution(from_rows(item["state"]), actions)
    last = len(states) - 1
    position = min(last // 2, last - study.horizon)
    if position < 0:
        raise ValueError(f"Trajectory {item['id']} is shorter than horizon {study.horizon}")
    source = states[position]
    api = ExperimentAPI(config.path("cache_dir"), provider, config.limits, client=client)
    components = LunaComponents(config, api, environment(source), training)
    row: dict[str, Any] = {
        "component": component, "instance": item["id"], "few_shot": config.few_shot,
        "position": position, "horizon": study.horizon,
    }
    seed = study.seed + int(hashlib.sha256(item["id"].encode()).hexdigest()[:8], 16)
    if component == "policy":
        row.update(_policy_case(components, source, states[position + study.horizon], study.horizon, seed))
    elif component == "generator":
        generated = _generator_case(components, source, states[position + study.horizon], study.horizon)
        row["generator"] = generated
        if "error" in generated:
            row["error"] = generated["error"]
        elif "proposal" in generated:
            rollout = _policy_case(components, source, from_rows(generated["proposal"]), study.horizon, seed)
            row["rollout"] = rollout
            if "error" in rollout:
                row["error"] = rollout["error"]
    elif component == "value":
        values = []
        for fraction in study.value_positions:
            index = round(fraction * last)
            result = _value_case(components, states[index], last - index)
            values.append({"fraction": fraction, "position": index,
                           "state": to_rows(states[index]), **result})
            if result.get("budget_exhausted"):
                break
        row["values"] = values
        if any("error" in value for value in values) or len(values) != len(study.value_positions):
            row["error"] = "One or more value requests failed or were not attempted"
    else:
        raise ValueError(f"Unsupported component: {component}")
    row["metrics"] = dict(api.metrics)
    return row


def run(study: StudyConfig, provider: Provider, *, selected: tuple[str, ...],
        shots: tuple[int, ...], resume: bool, retry_errors: bool) -> Path:
    """Write each case atomically and resume only matching data and configuration."""
    base = experiment_config(study, max(study.few_shot))
    training, instances = load_data(base)
    witness = base.path("data_dir") / "private_witnesses" / base.family / "test.json"
    witnesses = {row["id"]: row["actions"] for row in json.loads(witness.read_text(encoding="utf-8"))}
    if any(item["id"] not in witnesses for item in instances):
        raise ValueError("Missing offline witness for an evaluation instance")
    output = _path(study.output_dir)
    manifest_path = output / "manifest.json"
    expected = manifest(study, base, provider)
    if manifest_path.exists():
        if not resume:
            raise ValueError("Output already exists; pass --resume or choose a different output_dir")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Study manifest differs; choose a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty and has no study manifest")
        write_json(manifest_path, expected)
    credentials(provider)
    client = None
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(timeout=study.limits.timeout_seconds, max_retries=0)
    try:
        for count in shots:
            config = experiment_config(study, count)
            for component in selected:
                for index, item in enumerate(instances):
                    path = output / "cases" / f"shots_{count}" / component / f"{index:04d}.json"
                    previous = None
                    if path.exists():
                        previous = json.loads(path.read_text(encoding="utf-8"))
                        if not retry_errors or "error" not in previous:
                            continue
                    result = case_result(component, study, config, training, item=item,
                                         actions=witnesses[item["id"]], provider=provider, client=client)
                    if previous is not None:
                        result["prior_attempts"] = [*previous.get("prior_attempts", []),
                                                    {"metrics": previous["metrics"], "error": previous["error"]}]
                    write_json(path, result)
                    outcome = result.get("reached", result.get("generator", {}).get("valid_count", len(result.get("values", []))))
                    print(f"{component} shots={count} {index + 1}/{len(instances)}: {outcome}", flush=True)
    finally:
        if client is not None:
            client.close()
    return output


def main() -> int:
    """Plan or run explicitly selected conditions without implicit live calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--component", choices=(*COMPONENTS, "all"), default="all")
    parser.add_argument("--shots", type=int, nargs="+", help="Run a subset of configured variants")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="Retry failed saved cases when resuming")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--execute", action="store_true", help="Use live API; incurs charges")
    modes.add_argument("--mock", action="store_true")
    modes.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    study = load_study(args.config)
    print(json.dumps(plan(study), indent=2), flush=True)
    if args.plan:
        return 0
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    shots = tuple(args.shots if args.shots is not None else study.few_shot)
    if not shots or len(set(shots)) != len(shots) or any(count not in study.few_shot for count in shots):
        parser.error("--shots must select distinct configured few-shot counts")
    provider: Provider = "openai" if args.execute else "mock" if args.mock else "replay"
    selected = COMPONENTS if args.component == "all" else (args.component,)
    output = run(study, provider, selected=selected, shots=shots, resume=args.resume, retry_errors=args.retry_errors)
    print(f"Results: {output}")
    failures = 0
    for count in shots:
        for component in selected:
            for path in (output / "cases" / f"shots_{count}" / component).glob("*.json"):
                failures += "error" in json.loads(path.read_text(encoding="utf-8"))
    if failures:
        print(f"{failures} cases have errors; use --resume --retry-errors to retry them", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Study error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
