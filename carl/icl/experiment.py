"""Reproducible experiment orchestration and paired reporting."""
import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.calibration import calibrate, calibration_id, calibration_result_digest, target_indices
from carl.icl.config import ROOT, ExperimentConfig, load_config
from carl.icl.datasets import environment, load_data
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.experiments.tuning import tune as _tune
from carl.icl.luna import LunaComponents
from carl.icl.search import search
from carl.icl.reporting import report
from carl.icl.types import Board, JSONObject, TrainingData
from carl.utils.aliases import State

__all__ = [
    "calibration_components", "calibration_path", "components_for_run", "credentials", "matrix_configs",
    "output_directory", "preview", "run_calibration", "run_condition", "run_matrix", "tune",
]


def tune(config: ExperimentConfig, provider: str, directory: Path) -> None:
    """Run validation scale selection through the public experiment facade."""
    _tune(config, provider, directory, run_condition=run_condition)


def credentials(provider: str) -> None:
    """Load local credentials only when live inference is explicitly requested."""
    if provider != "openai":
        return
    load_dotenv(ROOT / ".env", override=False)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or any(
        marker in key.lower() for marker in ("replace-with", "your-api", "placeholder")
    ):
        raise ValueError(
            "Set OPENAI_API_KEY in .env (copy .env.example) or in the environment"
        )


def calibration_components(config: ExperimentConfig, provider: str) -> LunaComponents:
    """Bind calibration to training layouts and its independent request budget."""
    training, _ = load_data(config)
    state = from_rows(training["contexts"][0]["state"])
    api = ExperimentAPI(config.path("cache_dir"), provider, config.calibration.limits)
    return LunaComponents(config, api, environment(state), training)


def calibration_path(config: ExperimentConfig, components: LunaComponents) -> Path:
    """Locate labels for this exact frozen policy and provider."""
    return (
        config.path("data_dir")
        / "calibration"
        / calibration_id(config, components)
        / "calibration.json"
    )


def run_calibration(config: ExperimentConfig, provider: str) -> JSONObject:
    """Resume calibration and write separate machine-readable and human cost reports."""
    credentials(provider)
    components = calibration_components(config, provider)
    summary = calibrate(config, components, config.path("data_dir") / "calibration")
    directory = calibration_path(config, components).parent
    fields = ["context", "k", "d", "u", "hits", "success_rate", "weight"]
    with (directory / "calibration.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["examples"])
    (directory / "calibration.md").write_text(
        f"# Calibration ({summary['provider']})\n\nStatus: {summary['status']}. "
        f"Retained {len(summary['examples'])} targets; excluded {summary['excluded_targets']} zero-hit targets "
        f"and {summary['excluded_contexts']} contexts. Completed trials: {summary['trials']}; "
        f"actions in completed trials: {summary['calibration_actions']}.\n\n"
        "Mock labels must not be used for live inference. API and invocation costs are in calibration.json.\n"
    )
    print(
        f"Calibration: {summary['status']}, {len(summary['examples'])} retained targets; {calibration_path(config, components)}",
        flush=True,
    )
    return summary


def components_for_run(
    config: ExperimentConfig, provider: str, training: TrainingData, state: Board
) -> LunaComponents:
    """Validate few-shot and calibration contracts before spending inference budget."""
    api = ExperimentAPI(config.path("cache_dir"), provider, config.limits)
    components = LunaComponents(config, api, environment(state), training)
    if config.generator_checkpoint is None and config.method in {"relative", "relative_u", "absolute", "absolute_d", "ratio_d_over_ac", "normalized_offset"}:
        path = calibration_path(config, components)
        if not path.exists():
            raise ValueError(
                "Run calibrate with the same config/model/provider before relative or absolute search"
            )
        calibrated = json.loads(path.read_text(encoding="utf-8"))
        if calibrated.get("signature") != calibration_id(config, components):
            raise ValueError("Calibration signature does not match the frozen training and policy")
        contexts = {context["id"]: (index, context) for index, context in enumerate(training["contexts"])}
        for example in calibrated.get("examples", []):
            entry = contexts.get(example.get("context"))
            if entry is None:
                raise ValueError("Calibration includes an unknown training context")
            context_index, context = entry
            candidate_index = example.get("target_index")
            eligible, _ = target_indices(config, context, context_index)
            if not isinstance(candidate_index, int) or isinstance(candidate_index, bool) or candidate_index not in eligible:
                raise ValueError("Calibration includes an unselected training target")
            candidate = context["candidates"][candidate_index]
            if (example.get("state") != context["state"] or example.get("subgoal") != candidate["subgoal"]
                    or example.get("k") != candidate["k"] or example.get("actions") != candidate["actions"]):
                raise ValueError("Calibration target differs from its training trajectory")
        if calibrated.get("result_digest") != calibration_result_digest(calibrated):
            raise ValueError("Calibration result labels or weights failed their integrity check")
        components.calibrated = calibrated
    # Validate all scales and few-shot availability BEFORE starting a paid search.
    if config.method in {"relative", "relative_u", "absolute", "absolute_d", "ratio_d_over_ac", "normalized_offset", "fixed"}:
        for scale in config.scales:
            components.generator_examples(state, scale, resolve_embeddings=False)
    return components


def preview(config: ExperimentConfig, provider: str = "replay") -> Path:
    """Write inspectable requests without issuing model or embedding calls."""
    training, instances = load_data(config)
    state = from_rows(instances[0]["state"])
    components = LunaComponents(
        config,
        ExperimentAPI(config.path("cache_dir"), provider, config.limits),
        environment(state),
        training,
    )
    target = from_rows(training["contexts"][0]["candidates"][0]["subgoal"])
    train_state = from_rows(training["contexts"][0]["state"])
    payloads: JSONObject = {
        "policy": components.policy_payload(train_state, target),
        "value": components.value_payload(state),
    }
    note = "Embedding retrieval is not resolved in preview. No requests were sent."
    try:
        components = components_for_run(config, provider, training, state)
        if config.method in {"relative", "relative_u", "absolute", "absolute_d", "ratio_d_over_ac", "normalized_offset", "fixed"}:
            payloads["generator"] = [
                components.generator_payload(state, scale, resolve_embeddings=False)
                for scale in config.scales
            ]
    except ValueError as exc:
        note += f" Generator unavailable: {exc}"
    path = config.path("output_dir") / f"preview_{config.name}.json"
    write_json(
        path, {"config": config.model_dump(), "note": note, "payloads": payloads}
    )
    return path


def _trial_row(
    config: ExperimentConfig, provider: str, seed: int, detail: JSONObject
) -> JSONObject:
    """Build a result row from a durable per-trial record."""
    return {
        "config": config.name,
        "method": config.method,
        "family": config.family,
        "split": config.split,
        "instance": detail["instance"]["id"],
        "seed": seed,
        "provider": provider,
        "tuning_status": config.tuning_status,
        "solved": detail["solved"],
        "finish_reason": detail["finish_reason"],
        "attempted_actions": detail["attempted_actions"],
        "solution_actions": len(detail["action_path"]) if detail["solved"] else None,
        "expansions": detail["expansions"],
        "seconds": detail["seconds"],
        **detail["api"],
    }


def run_condition(
    config: ExperimentConfig,
    provider: str,
    directory: Path,
    *,
    resume: bool = False,
    stop_on_api_error: bool = False,
) -> list[JSONObject]:
    """Execute matched seeds, optionally resuming durable per-trial outcomes."""
    credentials(provider)
    if provider == "mock" and config.selection == "embeddings":
        raise ValueError(
            "Embedding retrieval requires --execute or --replay; mock uses direct examples"
        )
    training, instances = load_data(config)
    # Check calibration and training contracts once, before writing any run outcomes.
    reference = components_for_run(
        config, provider, training, from_rows(instances[0]["state"])
    )
    config_path = directory / "config.json"
    results_path = directory / "results.json"
    if resume and config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config.model_dump(
            mode="json"
        ):
            raise ValueError(f"Cannot resume changed condition: {directory}")
    else:
        write_json(config_path, config.model_dump(mode="json"))
    rows: list[JSONObject] = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if resume and results_path.exists()
        else []
    )
    scheduled = {
        (item["id"], seed) for item in instances for seed in config.repeat_seeds
    }
    indexed = {(row["instance"], row["seed"]): row for row in rows}
    if len(indexed) != len(rows) or not set(indexed).issubset(scheduled):
        raise ValueError(f"Duplicate or unexpected saved trials in {directory}")
    if resume and any(row["provider"] != provider for row in rows):
        raise ValueError(f"Cannot resume results from another provider: {directory}")
    if stop_on_api_error and any(row["finish_reason"] == "api_error" for row in rows):
        raise ValueError(
            f"Saved API error in {directory}; inspect results before starting a new run"
        )
    for seed in config.repeat_seeds:
        for instance in instances:
            key = (instance["id"], seed)
            detail_path = directory / f"{instance['id']}_seed{seed}.json"
            if resume and detail_path.exists():
                detail = json.loads(detail_path.read_text(encoding="utf-8"))
                if (
                    detail.get("instance") != instance
                    or detail.get("provider") != provider
                    or detail.get("frozen_policy") != reference.fingerprint()
                    or detail.get("calibration_signature")
                    != (reference.calibrated or {}).get("signature")
                ):
                    raise ValueError(
                        f"Saved trial has mismatched input or policy: {detail_path}"
                    )
                recovered = _trial_row(config, provider, seed, detail)
                if key in indexed and indexed[key] != recovered:
                    raise ValueError(
                        f"Saved trial conflicts with result row: {detail_path}"
                    )
                if key not in indexed:
                    # Recover if interrupted after writing detail but before results.json.
                    rows.append(recovered)
                    indexed[key] = recovered
                    write_json(results_path, rows)
                if stop_on_api_error and recovered["finish_reason"] == "api_error":
                    raise ValueError(
                        f"Saved API error in {detail_path}; inspect before a new run"
                    )
                continue
            if resume and key in indexed:
                raise ValueError(f"Saved trial detail is missing: {detail_path}")
            state = from_rows(instance["state"])
            components = components_for_run(config, provider, training, state)
            outcome = search(components.env, components, state, config, seed)
            if outcome.solved:
                current: State = state.copy()
                for action in outcome.action_path:
                    current = components.env.next_state(current, action)
                if (
                    not components.env.is_solved(current)
                    or len(outcome.action_path) > outcome.attempted_actions
                ):
                    raise RuntimeError("Search reported a non-replayable solution")
            detail = {
                **asdict(outcome),
                "instance": instance,
                "provider": provider,
                "api": components.api.metrics,
                "diagnostics": components.diagnostics,
                "frozen_policy": components.fingerprint(),
                "calibration_signature": (components.calibrated or {}).get("signature"),
            }
            row = _trial_row(config, provider, seed, detail)
            write_json(detail_path, detail)
            rows.append(row)
            indexed[key] = row
            write_json(results_path, rows)
            print(
                f"{provider} {config.name} {instance['id']} seed={seed}: {outcome.finish_reason}, actions={outcome.attempted_actions}",
                flush=True,
            )
            if stop_on_api_error and outcome.finish_reason == "api_error":
                raise ValueError(f"API error saved in {detail_path}; sweep stopped")
    report(rows, directory)
    return rows


def output_directory(config: ExperimentConfig, provider: str, label: str) -> Path:
    """Choose an isolated directory for a new execution."""
    return config.path("output_dir") / f"{time.time_ns()}_{provider}_{label}"


def matrix_configs(path: Path) -> list[ExperimentConfig]:
    """Load named conditions and enforce equal budgets within each board family."""
    specification = json.loads(path.read_text(encoding="utf-8"))
    configs = [load_config(path.parent / item) for item in specification["configs"]]
    if len({c.name for c in configs}) != len(configs):
        raise ValueError("Matrix config names must be unique")
    for family in {c.family for c in configs}:
        group = [c for c in configs if c.family == family]
        # Only scale/method/label may vary in a matched comparison.
        ignored = {"name", "method", "scales", "fixed_cap"}
        contracts = {
            json.dumps(c.model_dump(exclude=ignored), sort_keys=True) for c in group
        }
        if len(contracts) != 1:
            raise ValueError(
                f"Unmatched model, examples, seeds or budgets in matrix family {family}"
            )
    return configs


def run_matrix(configs: list[ExperimentConfig], provider: str, directory: Path) -> None:
    """Preflight every condition before executing and aggregating the comparison."""
    credentials(provider)
    for config in configs:
        training, instances = load_data(config)
        components_for_run(config, provider, training, from_rows(instances[0]["state"]))
    rows = []
    for config in configs:
        rows.extend(run_condition(config, provider, directory / config.name))
        report(rows, directory)
    if any(row["finish_reason"] == "api_error" for row in rows):
        raise ValueError(
            "Matrix contains API errors; see results.csv and per-instance records"
        )
