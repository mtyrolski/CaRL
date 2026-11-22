"""Run matched Craftax planning conditions through the shared search engine."""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from carl.environment.craftax import CraftaxAdapter
from carl.environment.craftax.adapter import CRAFTAX_VERSION, OBSERVATION_VERSION
from carl.icl.api import write_json
from carl.icl.config import ROOT, Limits
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.intents.generation.data import file_sha256
from carl.icl.planning.craftax_execution import CraftaxActionAttempt, CraftaxIntentExecutor
from carl.icl.planning.craftax_tasks import (
    CraftaxIntent, load_tasks, mock_action, mock_proposals, mock_value, render_instruction,
)
from carl.icl.planning.intent_search import IntentSearchRuntime, IntentSearchSettings, SearchMode, search_domain_intents
from carl.icl.planning.plan_and_act import PLAN_AND_ACT_VARIANT, decode_plan, mock_plan
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.responses import text_message
from carl.icl.types import JSONObject


MOCK_MODEL = "craftax-visible-recipes-v1"


@dataclass(frozen=True)
class CraftaxMockRoles:
    """Use the common logical request ledger and typed decoders for offline roles."""

    api: ExperimentAPI
    adapter: CraftaxAdapter
    goal: CraftaxIntent
    fingerprint: str

    def _call(
        self, role: str, operation: str, query: JSONObject, *, output_tokens: int, reply: Callable[[], object],
        attempt_seed: int | None = None,
    ) -> object:
        metadata = {"component": role, "operation": operation, "fingerprint": self.fingerprint}
        if attempt_seed is not None:
            metadata["attempt_seed"] = str(attempt_seed)
        payload = {
            "model": MOCK_MODEL, "input": [text_message("user", json.dumps(query, sort_keys=True, allow_nan=False))],
            "metadata": metadata,
            "max_output_tokens": output_tokens,
            "store": False, "tools": [],
        }
        response = self.api.call("responses", payload, mock=lambda: {
            "output_text": json.dumps(reply(), allow_nan=False), "model": MOCK_MODEL, "usage": {},
        })
        return json.loads(response["output_text"])

    def act(
        self, observation: JSONObject, instruction: str, history: tuple[CraftaxActionAttempt, ...],
        intent: CraftaxIntent | None, *, seed: int,
    ) -> int:
        """Give the same π one current observation, instruction and history."""
        value = self._call("policy", "action", {
            "observation": observation, "instruction": instruction,
            "attempt_history": [asdict(item) for item in history],
        }, output_tokens=64, reply=lambda: {"action": mock_action(observation, self.goal, intent)}, attempt_seed=seed)
        if not isinstance(value, dict) or set(value) != {"action"} or not isinstance(value["action"], str):
            raise ValueError("Expected exactly one named Craftax action")
        return self.adapter.action_id(value["action"])

    def propose(self, observation: JSONObject, goal: str, scale: float | None) -> tuple[CraftaxIntent, ...]:
        """Ask G for a bounded proposal with the requested scale."""
        result = self._call("generator", "intent", {"observation": observation, "task_goal": goal, "scale": scale},
                            output_tokens=384, reply=lambda: _intent_list(mock_proposals(observation, self.goal, scale)))
        return decode_plan(result, max_steps=1)

    def value(self, observation: JSONObject, goal: str) -> float:
        """Estimate continuation from the same visible input as the other roles."""
        result = self._call("value", "value", {"observation": observation, "task_goal": goal},
                            output_tokens=64, reply=lambda: {"value": mock_value(observation, self.goal)})
        if not isinstance(result, dict) or set(result) != {"value"} or type(result["value"]) not in (int, float):
            raise ValueError("Expected one numeric Craftax value")
        return float(result["value"])

    def plan(self, observation: JSONObject, goal: str) -> tuple[CraftaxIntent, ...]:
        """Charge one generator call for each explicit plan or replan."""
        result = self._call("generator", "plan", {"observation": observation, "task_goal": goal},
                            output_tokens=1024, reply=lambda: _intent_list(mock_plan(observation, self.goal)))
        return decode_plan(result)


def _intent_list(intents: tuple[CraftaxIntent, ...]) -> JSONObject:
    return {"intents": [intent.model_dump(mode="json") for intent in intents]}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the matched offline domain entry point to the common CLI."""
    parser.add_argument("--tasks", type=Path, default=ROOT / "configs/icl/craftax/tasks.json")
    parser.add_argument("--task", default="wood-to-stone-pickaxe")
    parser.add_argument("--planning-mode", choices=(*SearchMode, "all"), default="all")
    parser.add_argument("--mode", choices=("mock",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-budget", type=int, default=20)
    parser.add_argument("--rollout-cap", type=int, default=5)
    parser.add_argument("--max-expansions", type=int, default=12)
    parser.add_argument("--max-requests", type=int, default=64)
    parser.add_argument("--wall-seconds", type=float, default=180.0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--map-size", type=int, default=16)
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--controller-seed", type=int, default=73)


def run(args: argparse.Namespace) -> int:
    """Run all modes from one immutable starting state and publish their matched results."""
    task = load_tasks(args.tasks).by_id(args.task)
    if not 0 < args.scale < 1:
        raise ValueError("Craftax relative scale must be strictly between zero and one")
    settings = IntentSearchSettings(
        scales=(args.scale,), action_budget=args.action_budget, rollout_cap=args.rollout_cap,
        max_expansions=args.max_expansions, wall_seconds=args.wall_seconds, seed=args.controller_seed,
    )
    limits = Limits(max_requests=args.max_requests, wall_seconds=args.wall_seconds)
    started = time.monotonic()
    adapter = CraftaxAdapter(max_steps=args.max_steps, map_size=args.map_size)
    initial = adapter.stage(adapter.reset(task.seed), **task.scenario.model_dump())
    warmup_action = adapter.action_id("NOOP")
    adapter.step(initial, warmup_action)
    preparation_seconds = time.monotonic() - started
    goal = render_instruction(task.goal)
    modes = tuple(SearchMode) if args.planning_mode == "all" else (SearchMode(args.planning_mode),)
    policy = {
        "model": MOCK_MODEL, "observation_version": OBSERVATION_VERSION, "actions": adapter.action_names,
        "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in (
            "craftax_tasks.py", "plan_and_act.py", "craftax_experiment.py",
        )},
    }
    fingerprint = contract_hash(policy)
    results = {}
    for mode in modes:
        api = ExperimentAPI(args.output.parent / "craftax-cache" / args.output.stem / mode.value, "mock", limits)
        roles = CraftaxMockRoles(api, adapter, task.goal, fingerprint)
        runtime = IntentSearchRuntime(
            CraftaxIntentExecutor(adapter, task.goal, roles.act), roles.propose, roles.value,
            lambda api=api: api.metrics, plan=roles.plan,
        )
        result = search_domain_intents(initial, goal, mode=mode, settings=settings, runtime=runtime)
        results[mode.value] = result.as_dict()
    artifact = {
        "domain": "craftax", "environment_version": CRAFTAX_VERSION, "observation_version": OBSERVATION_VERSION,
        "execution_mode": "mock", "task_id": task.task_id, "split": task.split, "seed": task.seed,
        "chain_signature": task.chain_signature, "task_manifest_sha256": file_sha256(args.tasks), "task_goal": goal,
        "initial_snapshot": adapter.snapshot_record(initial), "initial_observation": adapter.observe(initial),
        "preparation": {"wall_seconds": preparation_seconds, "attempted_simulator_actions": 1,
                        "warmup_action": warmup_action},
        "settings": asdict(settings), "limits": limits.model_dump(mode="json"),
        "policy": {**policy, "fingerprint": fingerprint}, "plan_and_act_variant": PLAN_AND_ACT_VARIANT,
        "results": results,
    }
    write_json(args.output, artifact)
    print(args.output)
    return 0 if all(result["solved"] for result in results.values()) else 2
