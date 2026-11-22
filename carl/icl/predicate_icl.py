"""Freeze, preview and execute predicate Sokoban conditions."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from carl.icl.config import ROOT, ExperimentConfig, PredicateICLSettings, PredicatePlanningSettings, Settings
from carl.icl.data_sources.sokoban import environment
from carl.icl.experiment import create_inference_api, prepare_credentials
from carl.icl.intent_domain import Intent, SokobanObservation, SokobanSnapshot
from carl.icl.intent_support import FrozenSupport, freeze_support, load_support
from carl.icl.intents.generation.data import file_sha256, parse_json
from carl.icl.llm_components.predicate import (
    PREDICATE_GENERATOR_PROMPT, PREDICATE_POLICY_PROMPT, InvalidProposal, PredicateComponents, PredicatePolicy,
    create_predicate_components,
)
from carl.icl.local_models import LocalRoleModel, bind_local_role
from carl.icl.luna_intent_rollout import RolloutStatus, rollout
from carl.icl.planning.intent_execution import SokobanIntentExecutor
from carl.icl.planning.intent_search import IntentSearchRuntime, IntentSearchSettings, SearchMode, search_intents
from carl.icl.planning.intent_value import FittedIntentValue, load_value
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


class MockResponses(Settings):
    """Fixture text passed through the production API and parsers."""

    policy: str | None = None
    generator: str | None = None
    value: str | None = None

    def check(
        self, components: PredicateComponents, roles: tuple[PredicateRole, ...], *, api_value: bool = False,
    ) -> None:
        """Require one response for each API role in mock mode."""
        for role in roles:
            if (getattr(self, role) is None) != components.is_local(role):
                raise ValueError(f"mock_response must name {role} only when it uses the API backend")
        if api_value and self.value is None:
            raise ValueError("mock_response must include value when using an API value heuristic")


def _preview(
    components: PredicateComponents, query: PredicateQuery, intent: Intent, seed: int,
) -> JSONObject:
    observation = query.visible()
    prompts = {
        "generator": components.generator_payload(observation, query.task_goal, query.scale),
        "policy": components.policy_payload(observation, intent, attempt_seed=seed),
    }
    return {"status": "preview", "prompts": prompts,
            "context": {role: components.budget_counts(payload) for role, payload in prompts.items()}}


def _run(
    components: PredicateComponents, query: PredicateQuery, cap: int, seed: int, mock: MockResponses | None,
) -> JSONObject:
    observation = query.visible()
    proposal_text = None if mock is None else mock.generator
    action_text = None if mock is None else mock.policy
    intent = components.propose(
        observation, query.task_goal, query.scale, mock=None if proposal_text is None else lambda: proposal_text,
    )
    state = SokobanSnapshot(observation.rows).restore()
    policy = components.policy_factory(seed, mock=None if action_text is None else lambda _: action_text)
    result = rollout(environment(state), state, dict(intent), policy, cap=cap, task_goal=query.task_goal)
    return {
        "status": result.status.value, "split": query.split, "scale": query.scale, "intent": intent,
        "actions": result.actions, "first_hit_actions": result.first_hit_actions,
        "task_solved": result.task_solved, "endpoint": result.endpoint, "error": result.error,
        "metrics": components.metrics,
    }


def _plan_settings(config: ExperimentConfig) -> IntentSearchSettings:
    settings = predicate_settings(config)
    plan = settings.planning or PredicatePlanningSettings(rollout_cap=config.calibration.action_cap)
    return IntentSearchSettings(
        scales=plan.scales or (settings.validation_scale,), rollout_cap=plan.rollout_cap,
        max_proposals=plan.max_proposals, trials_per_proposal=plan.trials_per_proposal,
        action_budget=config.action_budget, max_expansions=config.max_expansions,
        wall_seconds=config.limits.wall_seconds, discount=config.discount, seed=config.seed,
    )


def _search(
    components: PredicateComponents, query: PredicateQuery, mode: SearchMode,
    settings: IntentSearchSettings, *, fitted: FittedIntentValue | None, mock: MockResponses | None,
) -> JSONObject:
    action_text = None if mock is None else mock.policy
    proposal_text = None if mock is None else mock.generator
    value_text = None if mock is None else mock.value
    policy_mock = None if action_text is None else lambda _: action_text
    proposal_mock = None if proposal_text is None else lambda: proposal_text
    value_mock = None if value_text is None else lambda: value_text

    def executor_policy(seed: int) -> PredicatePolicy:
        return components.policy_factory(seed, mock=policy_mock)

    def propose(observation: SokobanObservation, goal: str, scale: float | None) -> list[Intent]:
        proposals = []
        for index in range(settings.max_proposals):
            try:
                proposals.append(components.propose(observation, goal, scale, proposal_index=index, mock=proposal_mock))
            except InvalidProposal:
                continue
        return proposals

    def estimate(observation: SokobanObservation, goal: str) -> float:
        if fitted is not None:
            components.record_fitted_value_call()
            return fitted(observation, goal)
        return components.value(observation, goal, mock=value_mock)

    runtime = IntentSearchRuntime(SokobanIntentExecutor(executor_policy), propose, estimate, lambda: components.metrics)
    result = search_intents(SokobanSnapshot(query.visible().rows), query.task_goal, mode=mode, settings=settings, runtime=runtime)
    return {**result.as_dict(), "status": "solved" if result.solved else result.finish_reason, "split": query.split}


def _validate_receipt(path: Path, condition: JSONObject, *, status: str) -> None:
    result = parse_json(path.read_text(encoding="utf-8"))
    if isinstance(result, dict):
        receipt = result.pop("validation_result", None)
        expected = {**condition, "run_fingerprint": contract_hash(result)}
        if result.get("status") == status and result.get("split") == "validation" and receipt == expected:
            return
    raise ValueError("test inference needs a successful validation run for this frozen condition and mode")


@dataclass(frozen=True)
class PredicateCondition:
    """Bind one resolved configuration to checked support and held-out queries."""

    config: ExperimentConfig
    data: Path
    manifest: Path
    policy_data: Path | None = None
    policy_audit: Path | None = None

    def freeze(self, *, support_seed: int | None = None) -> FrozenSupport:
        """Freeze the same checked support for the CLI and experiments."""
        settings = predicate_settings(self.config)
        return freeze_support(
            self.data, self.manifest, seed=self.config.seed if support_seed is None else support_seed,
            policy_shots=settings.policy_shots, generator_shots=settings.generator_shots,
            scale_arm=settings.scale_arm, validation_id=settings.validation_id,
            selection=selection_contract(self.config), policy_data=self.policy_data, policy_audit=self.policy_audit,
        )

    def preview(self, query: PredicateQuery, intent: Intent) -> JSONObject:
        """Report prompt payloads and token counts without model calls."""
        return self._evaluate(query, mode=None, intent=intent)

    def infer(
        self, query: PredicateQuery, *, mode: str, planning_mode: SearchMode | None = None,
        mock: MockResponses | None = None, validation_result: Path | None = None,
    ) -> JSONObject:
        """Run one proposal or a planner against the same frozen support."""
        if (mode == "mock") != (mock is not None):
            raise ValueError("mock responses are required in mock mode and unavailable in replay or live mode")
        return self._evaluate(query, mode=mode, planning_mode=planning_mode, mock=mock, validation_result=validation_result)

    def _evaluate(
        self, query: PredicateQuery, *, mode: str | None, intent: Intent | None = None,
        planning_mode: SearchMode | None = None, mock: MockResponses | None = None, validation_result: Path | None = None,
    ) -> JSONObject:
        config = self.config
        settings = predicate_settings(config)
        local_roles = local_role_models(config)
        support = load_condition(
            config, self.data, self.manifest, policy_data=self.policy_data, policy_audit=self.policy_audit, local_roles=local_roles,
        )
        query.check(support, settings)
        roles: tuple[PredicateRole, ...] = ("policy",) if planning_mode == SearchMode.DIRECT else ("policy", "generator")
        needs_value = planning_mode is not None and planning_mode != SearchMode.DIRECT
        checkpoint = value_checkpoint(settings) if needs_value else None
        api_value = needs_value and checkpoint is None
        fitted = load_value(checkpoint, discount=config.discount) if checkpoint is not None else None
        components = components_for_predicate_run(
            config, support, mode or "replay", preview_only=mode is None, local_roles=local_roles,
            required_roles=roles, require_api_value=api_value,
        )
        task_group = support.manifest["validation"]["task_group"] if query.split == "validation" else query.task_group
        components.assert_local_query(query.visible(), split=query.split, task_group=task_group, roles=roles)
        if fitted is not None:
            fitted.assert_held_out(query.visible(), task_group, split=query.split)
        contracts: dict[str, JSONObject] = {role: components.contract(role) for role in roles}
        if fitted is not None or api_value:
            contracts["value"] = fitted.identity() if fitted is not None else components.contract("value")
        identity: JSONObject = {"components": contracts, "support_fingerprint": support.fingerprint, "mode": mode}
        if intent is not None:
            return {**identity, **_preview(components, query, intent, config.seed)}
        plan = _plan_settings(config) if planning_mode is not None else None
        if plan is not None:
            identity["planning_mode"] = planning_mode
            identity["planning"] = {**asdict(plan), "scales": list(plan.scales)}
        success_status = "solved" if plan is not None else RolloutStatus.ACHIEVED
        anchor = support.manifest["validation"]
        condition = {"condition_fingerprint": contract_hash(identity), "validation_id": settings.validation_id,
                     "query_sha256": contract_hash([anchor, settings.validation_scale])}
        if query.split == "test":
            if validation_result is None:
                raise ValueError("test inference requires --validation-result from a successful validation run")
            _validate_receipt(validation_result, condition, status=success_status)
        if mock is not None:
            mock.check(components, roles, api_value=api_value)
        if planning_mode is not None and plan is not None:
            result = {**identity, **_search(components, query, planning_mode, plan, fitted=fitted, mock=mock)}
        else:
            result = {**identity, **_run(components, query, config.action_budget, config.seed, mock)}
        if query.split == "validation" and result["status"] == success_status:
            result["validation_result"] = {**condition, "run_fingerprint": contract_hash(result)}
        return result
