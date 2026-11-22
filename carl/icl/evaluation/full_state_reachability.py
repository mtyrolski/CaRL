"""Execute and summarize the exact-state reachability transfer protocol."""

import csv
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from pydantic import Field, model_validator

from carl.environment.sokoban.env import SokobanEnv
from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows
from carl.icl.config import Settings, load_config
from carl.icl.evaluation.component_study import wilson
from carl.icl.evaluation.reachability_design import Action, Positive, ReachabilityCase, ReachabilityModel, ReachabilityStudy, TrainingReceipt
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.full_state_policy import load_policy, load_settings
from carl.icl.llm_components.policy import GoalConditionedPolicy
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.search import rollout
from carl.icl.types import Board, JSONObject, Probabilities, RolloutPolicy, TrainingData
from carl.inference_components.conditional_low_level_policy import ConditionalLowLevelPolicy

TrialReason: TypeAlias = Literal["exact_target", "action_cap", "terminal_before_target", "model_error", "inference_budget"]


class Tariff(Settings):
    """User-supplied per-million-token assumptions for a cost estimate."""

    input_usd: float = Field(ge=0, allow_inf_nan=False)
    output_usd: float = Field(ge=0, allow_inf_nan=False)


class Trial(Settings):
    """One immutable trial outcome, including partial work on an invalid trial."""

    case_sha256: str
    policy_fingerprint: str
    status: Literal["complete", "error", "budget_exhausted"]
    actions: list[Action]
    first_hit: Positive | None
    reason: TrialReason
    seconds: float = Field(ge=0, allow_inf_nan=False)
    accounting: dict[str, float | int]

    @model_validator(mode="after")
    def check_outcome(self) -> "Trial":
        """Keep first hit, completion and failure causes internally consistent."""
        statuses = {"model_error": "error", "inference_budget": "budget_exhausted"}
        if self.status != statuses.get(self.reason, "complete"):
            raise ValueError("Trial status and termination disagree")
        if self.reason == "exact_target":
            if self.first_hit is None or self.first_hit != len(self.actions):
                raise ValueError("A success must stop on the first hit")
        elif self.first_hit is not None:
            raise ValueError("Only a successful trial has a first hit")
        return self


def offsets(study: ReachabilityStudy, model: ReachabilityModel) -> list[int]:
    """Resolve a model-specific explicit subset of the campaign offsets."""
    return model.offsets or study.offsets


def describe(study: ReachabilityStudy, tariffs: dict[str, Tariff]) -> JSONObject:
    """Estimate logical work and uncached inference from the resolved request limits."""
    models = []
    for model in study.models:
        points = offsets(study, model)
        maximum_actions = sum(points) * (study.trials_per_point + study.reserves_per_point)
        row: JSONObject = {"id": model.id, "kind": model.kind, "config": str(model.config), "offsets": points,
                           "planned_valid_trials": len(points) * study.trials_per_point,
                           "max_policy_decisions": maximum_actions, "cost_estimate_usd": 0.0}
        if model.kind != "local":
            config = load_config(str(model.config))
            limits = config.limits
            requests = min(maximum_actions * (1 + config.json_max_retries), limits.max_requests)
            input_cap = min(requests * limits.max_input_tokens, limits.max_total_input_tokens)
            output_cap = min(requests * config.policy_tokens, limits.max_output_tokens)
            row.update(max_logical_requests=requests, estimated_input_token_cap=input_cap, max_output_tokens=output_cap,
                       few_shot=config.few_shot, route=config.role("policy").identity(),
                       can_budget_full_action_envelope=requests >= maximum_actions * (1 + config.json_max_retries))
            if model.kind == "api":
                price = tariffs.get(model.id)
                row["cost_estimate_usd"] = ((input_cap * price.input_usd + output_cap * price.output_usd) / 1e6
                                            if price else None)
        models.append(row)
    return {"protocol": "exact-full-state-reachability-v1", "data": study.data, "trials_per_point": study.trials_per_point,
            "reserves_per_point": study.reserves_per_point, "models": models,
            "estimation": "Uncached, capped request and token estimates at supplied tariffs; billing uses the campaign ledger."}


def runtime_versions() -> dict[str, str]:
    """Record packages that shape the experiment and plotted intervals."""
    return {"python": platform.python_version(), **{p: version(p) for p in ("numpy", "pydantic", "openai", "torch", "matplotlib")}}


@dataclass
class Controller:
    """Common rollout policy, identity, and optional request accounting."""

    policy: RolloutPolicy
    identity: JSONObject
    api: ExperimentAPI | None = None

    def counts(self) -> dict[str, int | float]:
        """Snapshot current inference accounting."""
        return dict(self.api.metrics) if self.api is not None else {}


class ArgmaxPolicy:
    """Adapt local checkpoint logits to the shared first-hit rollout."""

    def __init__(self, local: ConditionalLowLevelPolicy, env: SokobanEnv) -> None:
        self.local, self.env = local, env

    def probabilities(self, state: Board, subgoal: Board) -> Probabilities:
        """Give probability one to the action the local CLLP normally takes."""
        action = self.env.distribution_to_action(self.local.get_action(state, subgoal))
        result = np.zeros(4)
        result[action] = 1.0
        return result


def controller(model: ReachabilityModel, env: SokobanEnv, training: TrainingData, output: Path, *, live: bool) -> Controller:
    """Construct once per condition and route API policy calls through the shared ledger."""
    if model.config is None:
        raise ValueError("Missing controller config")
    if model.kind == "local":
        local, identity = load_policy(load_settings(model.config), env)
        if model.training_receipt is None:
            raise ValueError("Missing local training receipt")
        receipt = TrainingReceipt.model_validate_json(model.training_receipt.read_text(encoding="utf-8"))
        if (receipt.training_split_fingerprint != training.get("provenance", {}).get("train_sha256")
                or receipt.policy_fingerprint != contract_hash(identity)):
            raise ValueError("Training split or checkpoint identity differs from the local training receipt")
        local.construct_network()
        return Controller(ArgmaxPolicy(local, env), identity)
    if model.kind == "api" and not live:
        raise ValueError("API inference requires --execute-live and an explicit --model")
    config = load_config(model.config)
    if config.generator_checkpoint is not None:
        raise ValueError("A policy-only config cannot load a generator checkpoint")
    provider = "mock" if model.kind == "mock" else "openai"
    api = ExperimentAPI(output / "cache", provider, config.limits)
    policy = GoalConditionedPolicy(config, api, env, training)
    return Controller(policy, {"kind": model.kind, "policy": policy.fingerprint(), "few_shot": config.few_shot}, api)


def execute_trial(case: ReachabilityCase, active: Controller, env: SokobanEnv) -> Trial:
    """Run until the first exact hit, terminal state, action cap or technical failure."""
    source, target = from_rows(case.source), from_rows(case.target)
    actions: list[int] = []
    before = active.counts()
    started = time.monotonic()
    reached = False
    status: Literal["complete", "error", "budget_exhausted"] = "complete"
    reason: TrialReason
    try:
        result = rollout(env, active.policy, source, target, cap=case.k, rng=np.random.default_rng(case.seed),
                         on_action=lambda _before, action, _after: actions.append(action))
        reached = result.reached
        reason = "exact_target" if reached else "terminal_before_target" if result.solved else "action_cap"
    except BudgetExceeded:
        status, reason = "budget_exhausted", "inference_budget"
    except RequestFailed:
        status, reason = "error", "model_error"
    counts = {key: value - before.get(key, 0) for key, value in active.counts().items()}
    if "logical_token_byte_fallback" in before:
        counts["logical_token_byte_fallback"] = before["logical_token_byte_fallback"]
    counts["attempted_simulator_actions"] = len(actions)
    return Trial(case_sha256=contract_hash(case.model_dump(mode="json")), policy_fingerprint=contract_hash(active.identity),
                 status=status, reason=reason, actions=actions, first_hit=len(actions) if reached else None,
                 seconds=time.monotonic() - started, accounting=counts)


def trial_path(output: Path, model: ReachabilityModel, case: ReachabilityCase) -> Path:
    """Name a stored attempt by model, offset and canonical layout."""
    return output / "trials" / model.id / str(case.k) / f"{case.layout}.json"


def saved_trials(output: Path, model: ReachabilityModel, cases: Sequence[ReachabilityCase]) -> list[Trial]:
    """Load only manifest-bound records and validate shape, action counts and policy identity."""
    trials = []
    policy_path = output / "policies" / f"{model.id}.json"
    for case in cases:
        path = trial_path(output, model, case)
        if not path.is_file():
            continue
        trial = Trial.model_validate_json(path.read_text(encoding="utf-8"))
        if (trial.case_sha256 != contract_hash(case.model_dump(mode="json")) or len(trial.actions) > case.k
                or trial.policy_fingerprint != contract_hash(json_object(policy_path))):
            raise ValueError(f"Trial does not match the frozen case or policy: {path}")
        if trial.reason == "action_cap" and len(trial.actions) != case.k:
            raise ValueError(f"Action-cap trial stopped before k: {path}")
        trials.append(trial)
    return trials


def json_object(path: Path) -> JSONObject:
    """Read an object artifact without admitting JSON scalars."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def point_rows(study: ReachabilityStudy, manifest: JSONObject, output: Path) -> list[JSONObject]:
    """Compute pointwise Wilson intervals over valid independent layout trials."""
    rows = []
    for model in study.models:
        for k in offsets(study, model):
            cases = [ReachabilityCase.model_validate(value) for value in manifest["cases"][str(k)]]
            trials = saved_trials(output, model, cases)
            valid = [trial for trial in trials if trial.status == "complete"]
            n = len(valid)
            successes = sum(trial.first_hit is not None for trial in valid)
            low, high = wilson(successes, n)
            rows.append({"model": model.id, "execution": model.kind, "k": k, "planned_valid": study.trials_per_point, "valid": n,
                         "attempts": len(trials), "invalid": len(trials) - n, "unattempted": len(cases) - len(trials),
                         "successes": successes, "rate_pct": 100 * successes / n if n else None,
                         "wilson_low_pct": low if n else None, "wilson_high_pct": high if n else None,
                         "status": "complete" if n == study.trials_per_point else "partial" if trials else "missing",
                         "offline_compute_seconds": model.offline_compute_seconds, "adaptation_seconds": model.adaptation_seconds,
                         "inference_seconds": sum(t.seconds for t in trials) if trials else None,
                         "logical_requests": sum(t.accounting.get("logical_requests", 0) for t in trials) if trials else None,
                         "physical_requests": sum(t.accounting.get("physical_requests", 0) for t in trials) if trials else None})
    return rows


def write_report(study: ReachabilityStudy, manifest: JSONObject, output: Path) -> None:
    """Write machine-readable points and a shareable exact reachability plot offline."""
    from matplotlib.figure import Figure  # pylint: disable=import-outside-toplevel

    points = point_rows(study, manifest, output)
    write_json(output / "report.json", {"schema_version": 1, "protocol": "exact-full-state-reachability-v1",
                                      "data_provenance": manifest["data_provenance"], "points": points})
    with (output / "points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    fig = Figure(figsize=(7, 4), layout="constrained")
    ax = fig.subplots()
    for model in study.models:
        rows = sorted((row for row in points if row["model"] == model.id and row["valid"]), key=lambda row: row["k"])
        if rows:
            x, y = [row["k"] for row in rows], [row["rate_pct"] for row in rows]
            label = model.id + " (mock)" if model.kind == "mock" and study.data != "seeded_fixture" else model.id
            line, = ax.plot(x, y, marker="o", label=label)
            ax.fill_between(x, [row["wilson_low_pct"] for row in rows], [row["wilson_high_pct"] for row in rows],
                            color=line.get_color(), alpha=0.18)
    title = "Full-state reachability (mock fixture)" if study.data == "seeded_fixture" else "Full-state reachability"
    ax.set(title=title, xlabel="Demonstration offset k (action cap)", ylabel="Exact target reached (%)", ylim=(0, 101))
    if ax.lines:
        ax.legend()
    fig.savefig(output / "reachability.svg", metadata={"Date": None})
