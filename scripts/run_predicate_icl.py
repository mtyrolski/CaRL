"""Freeze, preview and execute predicate policies and planners in Sokoban."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.config import ExperimentConfig, PredicatePlanningSettings, Settings, load_config
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_domain import Intent, SokobanObservation, SokobanSnapshot, decode_intent
from carl.icl.intent_support import freeze_support
from carl.icl.intents.generation.data import parse_json
from carl.icl.llm_components.predicate import InvalidProposal, PredicateComponents, PredicatePolicy
from carl.icl.luna_intent_rollout import RolloutStatus, rollout
from carl.icl.planning.craftax_experiment import add_arguments as add_craftax_arguments
from carl.icl.planning.craftax_experiment import run as run_craftax
from carl.icl.planning.intent_execution import SokobanIntentExecutor
from carl.icl.planning.intent_search import IntentSearchRuntime, IntentSearchSettings, SearchMode, search_intents
from carl.icl.planning.intent_value import FittedIntentValue, fit_value, load_value
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.predicate_icl import (
    PredicateQuery, PredicateRole, components_for_predicate_run, load_condition, local_role_models,
    predicate_settings, selection_contract, value_checkpoint,
)
from carl.icl.types import JSONObject


class MockResponses(Settings):
    """Explicit fixture text passed through the production API and parsers."""

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


def _read(path: Path) -> object:
    return parse_json(path.read_text(encoding="utf-8"))


def _emit(result: JSONObject, path: Path | None) -> None:
    if path is None:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        write_json(path, result)
        print(path)


def _preview(
    components: PredicateComponents, query: PredicateQuery, intent_path: Path, seed: int,
) -> JSONObject:
    observation = query.visible()
    prompts = {
        "generator": components.generator_payload(observation, query.task_goal, query.scale),
        "policy": components.policy_payload(
            observation, decode_intent(_read(intent_path)), attempt_seed=seed,
        ),
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
                proposals.append(components.propose(
                    observation, goal, scale, proposal_index=index,
                    mock=proposal_mock,
                ))
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
    result = _read(path)
    if isinstance(result, dict):
        receipt = result.pop("validation_result", None)
        expected = {**condition, "run_fingerprint": contract_hash(result)}
        if result.get("status") == status and result.get("split") == "validation" and receipt == expected:
            return
    raise ValueError("test inference needs a successful validation run for this frozen condition and mode")


def _execute(args: argparse.Namespace) -> int:
    if args.command == "craftax":
        return run_craftax(args)
    if args.command == "fit-value":
        fitted = fit_value(args.source, discount=args.discount)
        fitted.save(args.output)
        _emit(fitted.identity(), None)
        return 0
    config = load_config(args.config)
    settings = predicate_settings(config)
    local_roles = local_role_models(config)
    policy_sources = {"policy_data": args.policy_data, "policy_audit": args.policy_audit}
    if args.command == "freeze":
        support = freeze_support(
            args.data, args.manifest, seed=config.seed, policy_shots=settings.policy_shots,
            generator_shots=settings.generator_shots, scale_arm=settings.scale_arm,
            validation_id=settings.validation_id, selection=selection_contract(config, local_roles=local_roles), **policy_sources,
        )
        _emit({"status": "frozen", "support_fingerprint": support.fingerprint}, args.output)
        return 0
    support = load_condition(config, args.data, args.manifest, local_roles=local_roles, **policy_sources)
    query = PredicateQuery.model_validate(_read(args.query))
    query.check(support, settings)
    preview = args.command == "preview"
    mode = SearchMode(args.planning_mode) if args.command == "search" else None
    roles: tuple[PredicateRole, ...] = ("policy",) if mode == SearchMode.DIRECT else ("policy", "generator")
    checkpoint = value_checkpoint(settings) if mode is not None and mode != SearchMode.DIRECT else None
    api_value = mode is not None and mode != SearchMode.DIRECT and checkpoint is None
    fitted = load_value(checkpoint, discount=config.discount) if checkpoint is not None else None
    components = components_for_predicate_run(
        config, support, args.mode or "replay", preview_only=preview, local_roles=local_roles,
        required_roles=roles, require_api_value=api_value,
    )
    task_group = support.manifest["validation"]["task_group"] if query.split == "validation" else query.task_group
    components.assert_local_query(query.visible(), split=query.split, task_group=task_group, roles=roles)
    if fitted is not None:
        fitted.assert_held_out(query.visible(), task_group, split=query.split)
    identity = {"components": {role: components.contract(role) for role in roles},
                "support_fingerprint": support.fingerprint, "mode": args.mode}
    if fitted is not None or api_value:
        identity["components"]["value"] = fitted.identity() if fitted is not None else components.contract("value")
    if preview:
        _emit({**identity, **_preview(components, query, args.intent, config.seed)}, args.output)
        return 0
    plan = _plan_settings(config) if mode is not None else None
    if plan is not None:
        identity.update(planning_mode=mode, planning=asdict(plan))
    success_status = "solved" if plan is not None else RolloutStatus.ACHIEVED
    anchor = support.manifest["validation"]
    condition = {"condition_fingerprint": contract_hash(identity), "validation_id": settings.validation_id,
                 "query_sha256": contract_hash([anchor, settings.validation_scale])}
    if query.split == "test":
        if args.validation_result is None:
            raise ValueError("test inference requires --validation-result from a successful validation run")
        _validate_receipt(args.validation_result, condition, status=success_status)
    mock = None if args.mock_response is None else MockResponses.model_validate(_read(args.mock_response))
    if mock is not None:
        mock.check(components, roles, api_value=api_value)
    if mode is not None and plan is not None:
        result = {**identity, **_search(components, query, mode, plan, fitted=fitted, mock=mock)}
    else:
        result = {**identity, **_run(components, query, config.action_budget, config.seed, mock)}
    if query.split == "validation" and result["status"] == success_status:
        result["validation_result"] = {**condition, "run_fingerprint": contract_hash(result)}
    _emit(result, args.output)
    return 0 if result["status"] == success_status else 2


def main(argv: Sequence[str] | None = None) -> int:
    """Expose explicit inference modes and a keyless preview of both roles."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    add_craftax_arguments(actions.add_parser("craftax", help="Compare four planners on a pinned Craftax task"))
    fit = actions.add_parser("fit-value", help="Fit V from replay-verified training demonstrations")
    fit.add_argument("--source", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--discount", type=float, default=0.99)
    for name in ("freeze", "preview", "run", "search"):
        action = actions.add_parser(name)
        action.add_argument("--config", type=Path, required=True)
        action.add_argument("--data", type=Path, required=True, help="Checked D_A.jsonl or D_C.jsonl for G")
        action.add_argument("--manifest", type=Path, required=True, help="Frozen support manifest")
        action.add_argument("--policy-data", type=Path, help="Separate checked D_A for π")
        action.add_argument("--policy-audit", type=Path, help="Override the adjacent annotation_audit.jsonl for π")
        action.add_argument("--output", type=Path)
        if name != "freeze":
            action.add_argument("--query", type=Path, required=True, help="Visible query JSON and held-out split metadata")
        if name == "preview":
            action.add_argument("--intent", type=Path, required=True, help="Proposed intent for the policy preview")
            action.set_defaults(mode=None)
        if name == "search":
            action.add_argument("--planning-mode", choices=tuple(mode for mode in SearchMode if mode is not SearchMode.PLAN_AND_ACT), required=True)
        if name in ("run", "search"):
            action.add_argument("--mode", choices=("mock", "replay", "openai", "local"), required=True,
                                help="Use local weights for local roles and this mode for any API roles")
            action.add_argument("--mock-response", type=Path, help="Fixture JSON required with --mode mock")
            action.add_argument("--validation-result", type=Path, help="Successful validation output for test inference")
    args = parser.parse_args(argv)
    if args.command in ("run", "search") and (args.mode == "mock") != (args.mock_response is not None):
        parser.error("--mock-response is required in mock mode and unavailable in replay or live mode")
    try:
        return _execute(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        return parser.exit(1, f"Predicate ICL failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
