"""Offline, assumption-driven token estimates for explicit-board solver sweeps."""
import json
import math

import numpy as np
from pydantic import Field, model_validator

from carl.icl.boards import from_rows, to_rows
from carl.icl.budget_sweep import BudgetSweepConfig
from carl.icl.calibration import calibration_id
from carl.icl.config import ExperimentConfig, Settings
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import create_inference_api
from carl.icl.experiment_api import local_token_encoding
from carl.icl.components import ICLComponents
from carl.icl.types import ComponentRole, JSONObject, ModelProvider



class Pricing(Settings):
    """Explicit editable provider pricing, expressed per million tokens."""

    model: str
    provider: ModelProvider = "openai"
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    currency: str = "USD"
    tier: str
    as_of: str
    source: str


class Scenario(Settings):
    """Hypothetical workload, not an empirical statistical forecast."""

    action_fraction: float = Field(gt=0, le=1)
    actions_per_generator: float = Field(gt=0)
    calibration_fraction: float = Field(gt=0, le=1)
    output_multiplier: float = Field(gt=0)
    policy_output_tokens: int | None = Field(default=None, ge=1)
    generator_output_tokens: int | None = Field(default=None, ge=1)
    value_output_tokens: int | None = Field(default=None, ge=1)
    generator_attempts_per_expansion: float = Field(default=1, ge=1)
    retry_input_factor: float = Field(default=1, ge=1)


class Assumptions(Settings):
    """Price snapshot and named, user-editable workload scenarios."""

    pricing: Pricing
    scenarios: dict[str, Scenario]

    @model_validator(mode="after")
    def check_scenarios(self) -> "Assumptions":
        """Require at least one explicit scenario."""
        if not self.scenarios:
            raise ValueError("At least one scenario is required")
        return self


def _tokens(payload: JSONObject) -> int:
    """Match ExperimentAPI's local serialized-payload token estimate."""
    encoding = local_token_encoding()
    return len(
        encoding.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=())
    )


def _sample(config: ExperimentConfig) -> JSONObject:
    """Measure prompts over actual boards; synthesize relative labels only for sizing."""
    if config.selection != "direct":
        raise ValueError(
            "Token estimator currently supports selection=direct only; embeddings require separate pricing"
        )
    if config.num_subgoals != 1:
        raise ValueError("Token estimator currently supports num_subgoals=1 only")
    training, instances = load_data(config)
    state = from_rows(instances[0]["state"])
    api = create_inference_api(config, "mock", preview_only=True)
    components = ICLComponents(config, api, environment(state), training)
    if config.method == "relative":
        # Never use mock calibration as live calibration; labels only size the prompt.
        rows = []
        for context in training["contexts"]:
            for candidate in context["candidates"]:
                rows.append(
                    {
                        **candidate,
                        "state": context["state"],
                        "context": context["id"],
                        "u": 0.5,
                        "d": float(candidate["k"]),
                        "weight": 1.0,
                    }
                )
        components.calibrated = {
            "status": "completed",
            "provider": "mock",
            "policy_fingerprint": components.fingerprint(),
            "examples": rows,
        }
    measured: dict[str, list[int]] = {
        "generator": [],
        "policy": [],
        "value": [],
        "board_output": [],
    }
    for instance in instances:
        board = from_rows(instance["state"])
        # Sizing only: input/target preserve the actual layout; no solution is read.
        generator = components.generator_payload(
            board, config.scales[0], resolve_embeddings=False
        )
        measured["generator"].append(_tokens(generator))
        measured["policy"].append(_tokens(components.policy_payload(board, board)))
        measured["value"].append(_tokens(components.value_payload(board)))
        board_rows = to_rows(board)
        output: JSONObject = ({f"row_{index:02d}": row for index, row in enumerate(board_rows)}
                              if config.generator_layout_schema else {"rows": board_rows})
        measured["board_output"].append(_tokens({"subgoals": [output]}))
    calibration_inputs = []
    for context in training["contexts"]:
        source = from_rows(context["state"])
        for index in context["calibration_indices"]:
            target = from_rows(context["candidates"][index]["subgoal"])
            calibration_inputs.append(
                _tokens(components.policy_payload(source, target))
            )
    samples: JSONObject = {
        key: {"min": min(values), "mean": float(np.mean(values)), "max": max(values)}
        for key, values in measured.items()
    }
    samples["calibration_policy"] = {
        "min": min(calibration_inputs),
        "mean": float(np.mean(calibration_inputs)),
        "max": max(calibration_inputs),
    }
    samples["policy_output"] = _tokens({"probabilities": [0.25, 0.25, 0.25, 0.25]})
    samples["value_output"] = _tokens(
        {"remaining_actions": 30} if config.value_target == "remaining_actions" else {"value": 0.5}
    )
    samples["boards"] = len(instances)
    samples["calibration_signature"] = calibration_id(config, components)
    samples["calibration_targets"] = sum(
        len(context["calibration_indices"]) for context in training["contexts"]
    )
    return samples


def _usd(input_tokens: float, output_tokens: float, pricing: Pricing) -> float:
    return (
        input_tokens * pricing.input_usd_per_million
        + output_tokens * pricing.output_usd_per_million
    ) / 1_000_000


def estimate(
    spec: BudgetSweepConfig, configs: list[ExperimentConfig], assumptions: Assumptions
) -> JSONObject:
    """Estimate explicitly assumed workloads, plus configured logical accounting ceilings."""
    roles: tuple[ComponentRole, ...] = ("policy", "generator", "value")
    models = (config.role(role) for config in configs for role in roles)
    if any((model.provider, model.model) != (assumptions.pricing.provider, assumptions.pricing.model) for model in models):
        raise ValueError("Price model and provider must match all experiment roles for a single-price estimate")
    samples = {config.name: _sample(config) for config in configs}
    if len({sample["calibration_signature"] for sample in samples.values()}) != 1:
        raise ValueError(
            "Conditions require different calibrations; estimate them separately"
        )
    first = configs[0]
    calibration = samples[first.name]
    max_calibration_calls = min(
        first.calibration.limits.max_requests,
        calibration["calibration_targets"]
        * first.calibration.trials
        * first.calibration.action_cap,
    )
    runs = sum(config.instances * len(config.repeat_seeds) for config in configs) * len(
        spec.budgets
    )
    logical_input_cap = sum(
        config.instances
        * len(config.repeat_seeds)
        * len(spec.budgets)
        * config.limits.max_total_input_tokens
        for config in configs
    )
    logical_output_cap = sum(
        config.instances
        * len(config.repeat_seeds)
        * len(spec.budgets)
        * config.limits.max_output_tokens
        for config in configs
    )
    scenarios: list[JSONObject] = []
    for name, assumption in assumptions.scenarios.items():
        calibration_calls = math.ceil(
            max_calibration_calls * assumption.calibration_fraction
        )
        policy_out = min(
            first.policy_tokens,
            assumption.policy_output_tokens
            or math.ceil(calibration["policy_output"] * assumption.output_multiplier),
        )
        cal_input = calibration_calls * calibration["calibration_policy"]["mean"]
        cal_output = calibration_calls * policy_out
        search_input, search_output = 0.0, 0.0
        search_calls = 0
        over_limits = []
        if (
            cal_input > first.calibration.limits.max_total_input_tokens
            or cal_output > first.calibration.limits.max_output_tokens
            or calibration["calibration_policy"]["max"]
            > first.calibration.limits.max_input_tokens
        ):
            over_limits.append("calibration")
        for config in configs:
            sample = samples[config.name]
            repeats = config.instances * len(config.repeat_seeds)
            for budget in spec.budgets:
                policy_calls = math.ceil(budget * assumption.action_fraction)
                expansions = math.ceil(policy_calls / assumption.actions_per_generator)
                gen_calls = math.ceil(expansions * assumption.generator_attempts_per_expansion)
                value_calls = expansions + 1
                request_count = policy_calls + gen_calls + value_calls
                per_input = (
                    policy_calls * sample["policy"]["mean"]
                    + gen_calls * sample["generator"]["mean"] * assumption.retry_input_factor
                    + value_calls * sample["value"]["mean"]
                )
                per_output = (
                    policy_calls
                    * min(
                        config.policy_tokens,
                        math.ceil(
                            assumption.policy_output_tokens
                            or sample["policy_output"] * assumption.output_multiplier
                        ),
                    )
                    + gen_calls
                    * min(
                        config.generator_tokens,
                        math.ceil(
                            assumption.generator_output_tokens
                            or sample["board_output"]["mean"] * assumption.output_multiplier
                        ),
                    )
                    + value_calls
                    * min(
                        config.value_tokens,
                        math.ceil(
                            assumption.value_output_tokens
                            or sample["value_output"] * assumption.output_multiplier
                        ),
                    )
                )
                expansion_overflow = config.max_expansions is not None and expansions > config.max_expansions
                if (
                    request_count > config.limits.max_requests
                    or expansion_overflow
                    or per_input > config.limits.max_total_input_tokens
                    or per_output > config.limits.max_output_tokens
                    or max(
                        sample[role]["max"] for role in ("generator", "policy", "value")
                    )
                    > config.limits.max_input_tokens
                ):
                    over_limits.append(f"{config.name}@{budget}")
                search_calls += repeats * request_count
                search_input += repeats * per_input
                search_output += repeats * per_output
        scenarios.append(
            {
                "scenario": name,
                "assumptions": assumption.model_dump(),
                "calibration_calls": calibration_calls,
                "calibration_input": round(cal_input),
                "calibration_output": round(cal_output),
                "calibration_usd_uncached": round(
                    _usd(cal_input, cal_output, assumptions.pricing), 4
                ),
                "search_calls": search_calls,
                "search_input": round(search_input),
                "search_output": round(search_output),
                "search_usd_uncached": round(
                    _usd(search_input, search_output, assumptions.pricing), 4
                ),
                "total_input": round(cal_input + search_input),
                "total_output": round(cal_output + search_output),
                "estimated_usd_uncached": round(
                    _usd(
                        cal_input + search_input,
                        cal_output + search_output,
                        assumptions.pricing,
                    ),
                    4,
                ),
                "conditions_over_configured_limits": over_limits,
            }
        )
    return {
        "plan": {
            "budgets": spec.budgets,
            "runs": runs,
            "calibration_included_once": True,
        },
        "pricing": assumptions.pricing.model_dump(),
        "samples": samples,
        "scenarios": scenarios,
        "logical_limits": {
            "search_input_proxy": logical_input_cap,
            "search_output_tokens": logical_output_cap,
            "calibration_input_proxy": first.calibration.limits.max_total_input_tokens,
            "calibration_output_tokens": first.calibration.limits.max_output_tokens,
            "calibration_max_policy_calls": max_calibration_calls,
        },
    }
