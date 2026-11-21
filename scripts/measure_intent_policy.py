"""Measure a frozen policy on checked Sokoban intents with mock or offline replay."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carl.icl.config import ExperimentConfig, load_config
from carl.icl.experiment import create_inference_api
from carl.icl.luna_intent_rollout import Action, ActionAttempt
from carl.icl.policy_relative.measurements import MeasurementSettings, measure
from carl.icl.policy_relative.rollout import ReplayIntentPolicy


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
    return lambda _prompt: '{"action":"RIGHT"}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Checked D_A.jsonl from prepare_intent_annotations")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Use the deterministic RIGHT fixture policy")
    mode.add_argument("--replay", type=Path, metavar="CONFIG", help="Use the policy route and cache from this experiment config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-cap", type=int, default=8)
    parser.add_argument("--trials", type=int, help="Default planned attempts per candidate (mock 1, replay 4)")
    parser.add_argument("--attempt-plan", type=Path, help="JSON mapping selected record IDs to planned attempt counts")
    parser.add_argument("--quota", type=Path, help="JSON mapping effect types to per-context candidate caps")
    parser.add_argument("--offset-denominator", type=float, default=8.0)
    parser.add_argument("--initial-memory", type=Path, help="JSON list of initial action/no_op records")
    parser.add_argument("--cache-dir", type=Path, help="Override the query cache for replay")
    args = parser.parse_args()
    try:
        settings = MeasurementSettings(
            seed=args.seed, action_cap=args.action_cap,
            trials=args.trials if args.trials is not None else (1 if args.mock else 4),
            planned_attempts=_read_mapping(args.attempt_plan), quota=_read_mapping(args.quota),
            offset_denominator=args.offset_denominator,
        )
        controller: dict[str, Any]
        factory: Callable[[int], Callable[[str], str]]
        if args.mock:
            if args.cache_dir is not None:
                raise ValueError("--cache-dir requires --replay")
            controller = {"id": "mock-right-v1", "deterministic": True}
            factory = _mock_right
        else:
            config = load_config(args.replay)
            if args.cache_dir is not None:
                config = ExperimentConfig.model_validate({**config.model_dump(), "cache_dir": str(args.cache_dir)})
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
            initial_memory=_memory(args.initial_memory), resume=args.resume,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Policy measurement failed: {error}\n")
    print(json.dumps({key: result[key] for key in (
        "status", "planned_attempts", "completed_attempts", "outcomes", "labelled_candidates",
    )}))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
