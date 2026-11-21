"""Freeze checked support, preview prompts and run predicate Sokoban ICL."""

import argparse
import json
from pathlib import Path

from carl.icl.api import write_json
from carl.icl.config import Settings, load_config
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_domain import SokobanSnapshot, decode_intent
from carl.icl.intent_support import freeze_support
from carl.icl.intents.generation.data import parse_json
from carl.icl.llm_components.predicate import PredicateComponents
from carl.icl.luna_intent_rollout import RolloutStatus, rollout
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.predicate_icl import (
    PredicateQuery, components_for_predicate_run, load_condition, local_role_models, predicate_settings, selection_contract,
)
from carl.icl.types import JSONObject


class MockResponses(Settings):
    """Explicit fixture text passed through the production API and parsers."""

    policy: str | None = None
    generator: str | None = None

    def check(self, components: PredicateComponents) -> None:
        """Require one response for each API role in mock mode."""
        for role in ("policy", "generator"):
            if (getattr(self, role) is None) != components.is_local(role):
                raise ValueError(f"mock_response must name {role} only when it uses the API backend")


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


def _validate_receipt(path: Path, condition: JSONObject) -> None:
    result = _read(path)
    if isinstance(result, dict):
        receipt = result.pop("validation_result", None)
        expected = {**condition, "run_fingerprint": contract_hash(result)}
        if result.get("status") == RolloutStatus.ACHIEVED and result.get("split") == "validation" and receipt == expected:
            return
    raise ValueError("test inference needs a successful validation run for this frozen condition and mode")


def _execute(args: argparse.Namespace) -> int:
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
    components = components_for_predicate_run(config, support, args.mode or "replay", preview_only=preview, local_roles=local_roles)
    task_group = support.manifest["validation"]["task_group"] if query.split == "validation" else query.task_group
    components.assert_local_query(query.visible(), split=query.split, task_group=task_group)
    identity = {"components": {"policy": components.contract("policy"), "generator": components.contract("generator")},
                "support_fingerprint": support.fingerprint, "mode": args.mode}
    if preview:
        _emit({**identity, **_preview(components, query, args.intent, config.seed)}, args.output)
        return 0
    anchor = support.manifest["validation"]
    condition = {"condition_fingerprint": contract_hash(identity), "validation_id": settings.validation_id,
                 "query_sha256": contract_hash([anchor, settings.validation_scale])}
    if query.split == "test":
        if args.validation_result is None:
            raise ValueError("test inference requires --validation-result from a successful validation run")
        _validate_receipt(args.validation_result, condition)
    mock = None if args.mock_response is None else MockResponses.model_validate(_read(args.mock_response))
    if mock is not None:
        mock.check(components)
    result = {**identity, **_run(components, query, config.action_budget, config.seed, mock)}
    if query.split == "validation" and result["status"] == RolloutStatus.ACHIEVED:
        result["validation_result"] = {**condition, "run_fingerprint": contract_hash(result)}
    _emit(result, args.output)
    return 0 if result["status"] == RolloutStatus.ACHIEVED else 2


def main() -> int:
    """Expose explicit inference modes and a keyless preview of both roles."""
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "preview", "run"):
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
        if name == "run":
            action.add_argument("--mode", choices=("mock", "replay", "openai", "local"), required=True,
                                help="Use local weights for local roles and this mode for any API roles")
            action.add_argument("--mock-response", type=Path, help="Fixture JSON required with --mode mock")
            action.add_argument("--validation-result", type=Path, help="Successful validation output for test inference")
    args = parser.parse_args()
    if args.command == "run" and (args.mode == "mock") != (args.mock_response is not None):
        parser.error("--mock-response is required in mock mode and unavailable in replay or live mode")
    try:
        return _execute(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        return parser.exit(1, f"Predicate ICL failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
