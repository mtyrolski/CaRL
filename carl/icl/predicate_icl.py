"""Connect checked support to the configured frozen Sokoban components."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from carl.icl.config import ExperimentConfig, PredicateICLSettings, Settings
from carl.icl.experiment import create_inference_api, prepare_credentials
from carl.icl.intent_domain import SokobanObservation
from carl.icl.intent_support import FrozenSupport, load_support
from carl.icl.llm_components.predicate import (
    PREDICATE_GENERATOR_PROMPT, PREDICATE_POLICY_PROMPT, PredicateComponents, create_predicate_components,
)
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject


def predicate_settings(config: ExperimentConfig) -> PredicateICLSettings:
    """Require the opt-in block while leaving legacy experiment profiles intact."""
    if config.predicate_icl is None:
        raise ValueError("predicate experiment needs a predicate_icl config block")
    return config.predicate_icl


def selection_contract(config: ExperimentConfig) -> JSONObject:
    """Bind the validation choice to the actual role prompts and model routes."""
    settings = predicate_settings(config)
    return {
        **settings.model_dump(mode="json"),
        "policy_prompt": settings.policy_prompt or PREDICATE_POLICY_PROMPT,
        "generator_prompt": settings.generator_prompt or PREDICATE_GENERATOR_PROMPT,
        "policy_model": config.role("policy").identity(),
        "generator_model": config.role("generator").identity(),
        "seed": config.seed,
        "policy_tokens": config.policy_tokens,
        "generator_tokens": config.generator_tokens,
        "json_max_retries": config.json_max_retries,
        "action_budget": config.action_budget,
        "limits": config.limits.model_dump(mode="json"),
    }


def load_condition(
    config: ExperimentConfig, data: Path, manifest: Path,
    *, policy_data: Path | None = None, policy_audit: Path | None = None,
) -> FrozenSupport:
    """Read immutable support and check that this is the selected condition."""
    support = load_support(manifest, data, policy_data=policy_data, policy_audit=policy_audit)
    if support.manifest["selection"] != selection_contract(config):
        raise ValueError("config differs from the frozen prompt, scale or support selection")
    return support


def components_for_predicate_run(
    config: ExperimentConfig, support: FrozenSupport, mode: str,
    *, preview_only: bool = False, calibration: bool = False,
) -> PredicateComponents:
    """Build π and G over one budgeted API and separate role supports."""
    settings = predicate_settings(config)
    if not preview_only:
        prepare_credentials(config, mode)
    api = create_inference_api(
        config, mode, limits=config.calibration.limits if calibration else config.limits, preview_only=preview_only,
    )
    return create_predicate_components(
        config, api,
        policy_prompt=settings.policy_prompt or PREDICATE_POLICY_PROMPT,
        generator_prompt=settings.generator_prompt or PREDICATE_GENERATOR_PROMPT,
        policy_examples=support.policy,
        generator_examples=support.generator,
        policy_support_fingerprint=support.policy_fingerprint,
        generator_support_fingerprint=support.generator_fingerprint,
        context_limit=settings.context_limit,
        scale_arm=settings.scale_arm,
    )


class PredicateQuery(Settings):
    """A held-out visible query with split metadata kept outside model inputs."""

    observation: dict[Literal["rows"], list[str]]
    task_goal: str
    scale: float | None = Field(default=None, strict=True)
    split: Literal["validation", "test"]
    task_group: str | None = None

    def visible(self) -> SokobanObservation:
        """Return the domain observation without split, group or source metadata."""
        if set(self.observation) != {"rows"} or not self.task_goal.strip():
            raise ValueError("query needs observation.rows and a nonempty task goal")
        return SokobanObservation(tuple(self.observation["rows"]))

    def check(self, support: FrozenSupport, settings: PredicateICLSettings) -> None:
        """Enforce held-out groups and bind validation to its frozen anchor."""
        observation = self.visible()
        group = self.task_group
        if self.split == "validation":
            anchor = support.manifest["validation"]
            if (contract_hash(list(observation.rows)) != anchor["observation_sha256"]
                    or self.task_goal != anchor["task_goal"]
                    or self.scale != settings.validation_scale
                    or group not in (None, anchor["task_group"])):
                raise ValueError("validation query differs from the frozen observation, task, group or scale")
            group = anchor["task_group"]
        support.assert_query(observation, split=self.split, task_group=group)
