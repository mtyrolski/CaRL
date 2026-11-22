"""Measure a frozen policy on checked Sokoban intents with offline backends."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carl.icl.config import ExperimentConfig, load_config
from carl.icl.experiment import create_inference_api
from carl.icl.intent_domain import SokobanObservation
from carl.icl.intent_support import task_group_id
from carl.icl.llm_components.predicate import PredicatePolicy
from carl.icl.luna_intent_rollout import Action, ActionAttempt
from carl.icl.policy_relative.measurements import MeasurementSettings, measure
from carl.icl.policy_relative.rollout import PolicyFactory, ReplayIntentPolicy
from carl.icl.predicate_icl import components_for_predicate_run, load_condition, local_role_models

RIGHT = '{"action":"RIGHT"}'


def _read_mapping(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, dict):
        raise ValueError(f"{path} must contain an object mapping names to counts")
    return items


def _memory(path: Path | None) -> tuple[ActionAttempt, ...]:
    if path is None:
        return ()
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or any(
        not isinstance(item, dict) or set(item) != {"action", "no_op"} for item in items
    ):
        raise ValueError("initial memory must be a list of action and no_op records")
    return tuple(ActionAttempt(Action(item["action"]), item["no_op"]) for item in items)


def _mock_right(_seed: int) -> Callable[[str], str]:
    """A deterministic fixture policy that repeatedly chooses RIGHT."""
    return lambda _prompt: RIGHT


def _predicate_policy(
    config: ExperimentConfig, args: argparse.Namespace,
) -> tuple[dict[str, Any], PolicyFactory, Callable[[dict[str, Any]], None]]:
    local_roles = local_role_models(config)
    support = load_condition(
        config, args.support_data or args.data, args.support_manifest,
        policy_data=args.policy_data, policy_audit=args.policy_audit, local_roles=local_roles,
    )
    mode = "local" if args.local is not None else "mock" if args.mock else "replay"
    components = components_for_predicate_run(config, support, mode, calibration=True, local_roles=local_roles)
    controller = {
        "id": f"predicate-{mode}-policy-v1", "deterministic": mode != "replay",
        "predicate_policy": components.contract("policy"),
        "limits": components.api.limits.model_dump(),
        "cache_dir": str(config.path("cache_dir").resolve()),
    }

    def factory(seed: int) -> PredicatePolicy:
        return components.policy_factory(seed, mock=(lambda _: RIGHT) if args.mock else None)

    def check(row: dict[str, Any]) -> None:
        observation = SokobanObservation(tuple(row["start_observation"]))
        group = task_group_id(row)
        support.assert_policy_holdout(observation, task_group=group)
        components.assert_local_query(observation, split="test", task_group=group, roles=("policy",))

    return controller, factory, check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Checked D_A.jsonl from prepare_intent_annotations")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Use the deterministic RIGHT fixture policy")
    mode.add_argument("--replay", type=Path, metavar="CONFIG", help="Use the policy route and cache from this experiment config")
    mode.add_argument("--local", type=Path, metavar="CONFIG", help="Use the pinned local policy in this experiment config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-cap", type=int, default=8)
    parser.add_argument("--trials", type=int, help="Default planned attempts per candidate (mock and local 1, replay 4)")
    parser.add_argument("--attempt-plan", type=Path, help="JSON mapping selected record IDs to planned attempt counts")
    parser.add_argument("--quota", type=Path, help="JSON mapping effect types to per-context candidate caps")
    parser.add_argument("--offset-denominator", type=float, default=8.0)
    parser.add_argument("--initial-memory", type=Path, help="JSON list of initial action/no_op records")
    parser.add_argument("--cache-dir", type=Path, help="Override the query cache for replay")
    parser.add_argument("--predicate-config", type=Path, help="Use the frozen predicate policy config in mock mode")
    parser.add_argument("--support-manifest", type=Path, help="Frozen predicate ICL support manifest")
    parser.add_argument("--support-data", type=Path, help="Dataset used to freeze support, defaults to --data")
    parser.add_argument("--policy-data", type=Path, help="Separate checked D_A used to freeze π support")
    parser.add_argument("--policy-audit", type=Path, help="Override the action audit used to freeze π support")
    args = parser.parse_args()
    try:
        settings = MeasurementSettings(
            seed=args.seed, action_cap=args.action_cap,
            trials=args.trials if args.trials is not None else (4 if args.replay else 1),
            planned_attempts=_read_mapping(args.attempt_plan), quota=_read_mapping(args.quota),
            offset_denominator=args.offset_denominator,
        )
        if args.replay is None and args.cache_dir is not None:
            raise ValueError("--cache-dir requires --replay")
        if args.predicate_config is not None and not args.mock:
            raise ValueError("--predicate-config is for --mock; pass a predicate config directly to --replay")
        config_path = args.predicate_config if args.mock else args.replay or args.local
        config = load_config(config_path) if config_path is not None else None
        if config is not None and args.cache_dir is not None:
            config = ExperimentConfig.model_validate({**config.model_dump(), "cache_dir": str(args.cache_dir)})
        typed = config is not None and config.predicate_icl is not None
        if args.local is not None and not typed:
            raise ValueError("--local requires a predicate config")
        if typed != (args.support_manifest is not None):
            raise ValueError("a predicate config and --support-manifest are required together")
        if not typed and (args.support_data or args.policy_data or args.policy_audit or args.predicate_config):
            raise ValueError("predicate support input requires a predicate config and manifest")
        controller: dict[str, Any]
        factory: PolicyFactory
        check = None
        if config is not None and typed:
            controller, factory, check = _predicate_policy(config, args)
        elif args.mock:
            controller = {"id": "mock-right-v1", "deterministic": True}
            factory = _mock_right
        else:
            assert config is not None
            policy = config.role("policy")
            api = create_inference_api(config, "replay")
            controller = {
                "id": "replay-intent-policy-v1",
                "deterministic": False,
                "transport": policy.identity(),
                "limits": config.limits.model_dump(),
                "output_tokens": config.policy_tokens,
                "cache_dir": str(config.path("cache_dir").resolve()),
            }

            def replay_factory(seed: int) -> ReplayIntentPolicy:
                return ReplayIntentPolicy(api, policy, config.policy_tokens, seed)

            factory = replay_factory
        result = measure(
            args.data, args.output, settings=settings, controller=controller, policy_factory=factory,
            initial_memory=_memory(args.initial_memory), resume=args.resume, candidate_check=check,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Policy measurement failed: {error}\n")
    print(json.dumps({key: result[key] for key in (
        "status", "planned_attempts", "completed_attempts", "outcomes", "labelled_candidates",
    )}))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
