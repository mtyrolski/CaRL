"""Connect checked support to the configured frozen Sokoban components."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from carl.icl.config import ROOT, ExperimentConfig, PredicateICLSettings, Settings
from carl.icl.experiment import create_inference_api, prepare_credentials
from carl.icl.intent_domain import SokobanObservation
from carl.icl.intent_support import FrozenSupport, load_support
from carl.icl.intents.generation.data import file_sha256
from carl.icl.llm_components.predicate import (
    PREDICATE_GENERATOR_PROMPT, PREDICATE_POLICY_PROMPT, PredicateComponents, create_predicate_components,
)
from carl.icl.local_models import LocalRoleModel, bind_local_role
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject


def predicate_settings(config: ExperimentConfig) -> PredicateICLSettings:
    """Require the opt-in block while leaving legacy experiment profiles intact."""
    if config.predicate_icl is None:
        raise ValueError("predicate experiment needs a predicate_icl config block")
    return config.predicate_icl


LocalModels = dict[Literal["policy", "generator"], LocalRoleModel]
PredicateRole = Literal["policy", "generator"]


def local_role_models(config: ExperimentConfig) -> LocalModels:
    """Bind configured local snapshots from paths relative to the checkout."""
    overrides = predicate_settings(config).local_roles
    models: LocalModels = {}
    if overrides is not None:
        for role in ("policy", "generator"):
            settings = getattr(overrides, role)
            if settings is not None:
                paths = {"model": str(ROOT / Path(settings.model).expanduser())}
                if settings.adapter is not None:
                    paths["adapter"] = str(ROOT / Path(settings.adapter).expanduser())
                models[role] = bind_local_role(settings.model_copy(update=paths), role)
    return models


def selection_contract(config: ExperimentConfig, *, local_roles: LocalModels | None = None) -> JSONObject:
    """Bind the validation choice to the actual role prompts and model routes."""
    settings = predicate_settings(config)
    roles = local_role_models(config) if local_roles is None else local_roles
    contract = {
        **settings.model_dump(mode="json"),
        "policy_prompt": settings.policy_prompt or PREDICATE_POLICY_PROMPT,
        "generator_prompt": settings.generator_prompt or PREDICATE_GENERATOR_PROMPT,
        "policy_model": roles["policy"].identity() if "policy" in roles else config.role("policy").identity(),
        "generator_model": roles["generator"].identity() if "generator" in roles else config.role("generator").identity(),
        "seed": config.seed,
        "policy_tokens": config.policy_tokens,
        "generator_tokens": config.generator_tokens,
        "json_max_retries": config.json_max_retries,
        "action_budget": config.action_budget,
        "limits": config.limits.model_dump(mode="json"),
    }
    if settings.planning is not None:
        checkpoint = value_checkpoint(settings)
        contract["planning_contract"] = {
            "discount": config.discount, "max_expansions": config.max_expansions,
            "value_tokens": config.value_tokens,
            "value_model": config.role("value").identity() if checkpoint is None else None,
            "value_checkpoint_sha256": None if checkpoint is None else file_sha256(checkpoint),
        }
    return contract


def value_checkpoint(settings: PredicateICLSettings) -> Path | None:
    """Resolve the fitted heuristic path using the same root as model paths."""
    if settings.planning is None or settings.planning.value_checkpoint is None:
        return None
    return ROOT / Path(settings.planning.value_checkpoint).expanduser()


def load_condition(
    config: ExperimentConfig, data: Path, manifest: Path,
    *, policy_data: Path | None = None, policy_audit: Path | None = None, local_roles: LocalModels | None = None,
) -> FrozenSupport:
    """Read immutable support and check that this is the selected condition."""
    support = load_support(manifest, data, policy_data=policy_data, policy_audit=policy_audit)
    if support.manifest["selection"] != selection_contract(config, local_roles=local_roles):
        raise ValueError("config differs from the frozen prompt, scale or support selection")
    return support


def components_for_predicate_run(
    config: ExperimentConfig, support: FrozenSupport, mode: str,
    *, preview_only: bool = False, calibration: bool = False, local_roles: LocalModels | None = None,
    required_roles: tuple[PredicateRole, ...] | None = None, require_api_value: bool = False,
) -> PredicateComponents:
    """Build π and G with separate support and explicit local or API backends."""
    settings = predicate_settings(config)
    roles = local_role_models(config) if local_roles is None else local_roles
    needed = required_roles if required_roles is not None else ("policy",) if calibration else ("policy", "generator")
    uses_api = require_api_value or any(role not in roles for role in needed)
    if mode == "local":
        if uses_api:
            raise ValueError("local mode requires local weights or a fitted heuristic for each executed role")
    elif not preview_only:
        if not uses_api:
            raise ValueError("use local mode when each executed role uses local weights")
        prepare_credentials(config, mode)
    api = create_inference_api(
        config, "mock" if mode == "local" else mode,
        limits=config.calibration.limits if calibration else config.limits, preview_only=preview_only,
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
        local_roles=roles,
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
