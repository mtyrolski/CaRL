"""Integration-only replay and native tick evidence for the intent pilot."""

import json
import random
from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path
from typing import Any

from carl.environment.craftax import CraftaxAdapter, CraftaxSnapshot
from carl.environment.craftax.adapter import CraftaxTransition
from carl.icl.api import write_json
from carl.icl.config import ROOT
from carl.icl.intent_domain import TASK_GOAL, SokobanSnapshot, decode_intent
from carl.icl.luna_intent_rollout import ActionAttempt, IntentRollout, RolloutStatus
from carl.icl.planning.craftax_execution import CraftaxActionAttempt, CraftaxIntentExecutor, CraftaxState
from carl.icl.planning.craftax_tasks import CraftaxIntent, load_tasks, mock_action, mock_proposals, render_instruction
from carl.icl.planning.intent_execution import SokobanIntentExecutor
from carl.icl.types import JSONObject


SEED = 73
CRAFTAX_OBSERVATION_FIELDS = {
    "version", "tiles", "neighbors", "items", "entities", "inventory", "facing",
    "health", "food", "drink", "energy", "mana", "sleeping", "resting",
}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _report(
    rollout: IntentRollout, decisions: list[JSONObject], work: str,
    *, native: list[JSONObject] | None = None,
) -> JSONObject:
    steps = [{key: value for key, value in step.items() if key not in {"prompt", "raw_response"}}
             for step in rollout.steps]
    executed = sum(step.get("primitive_action_executed", step.get("action") is not None) for step in steps)
    ticks = sum(step.get("native_ticks", int(step.get("action") is not None)) for step in steps)
    automatic = sum(step.get("automatic_ticks", 0) for step in steps)
    _check(ticks == executed + automatic, "Native ticks disagree with primitive and automatic work")
    _check(len(decisions) == rollout.model_calls, "Controller decisions disagree with policy calls")
    return {
        "work": work, "decisions": decisions, "calls_by_role": {"policy": rollout.model_calls, "generator": 0, "value": 0},
        "physical_api_requests": 0, "requested_primitive_actions": len(decisions),
        "executed_primitive_actions": executed, "native_ticks": ticks, "automatic_ticks": automatic,
        "stop_reason": rollout.status, "first_verified_hit_primitive_actions": rollout.first_hit_actions,
        "task_solved": rollout.task_solved, "steps": steps, "native_transitions": native,
    }


class _SokobanPolicy:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.decisions: list[JSONObject] = []

    def __call__(self, prompt: str) -> str:
        query = json.loads(prompt.rsplit("\n\n", 1)[1])
        _check(set(query) == {"state", "state_coordinates", "intent", "task_goal", "attempt_history", "instruction"},
               "Unexpected Sokoban model input")
        _check(all(set(item) == {"action", "no_op"} for item in query["attempt_history"]), "Unexpected Sokoban memory")
        board = query["state"]
        row, col = next((y, x) for y, line in enumerate(board) for x, cell in enumerate(line) if cell in "@+")
        pushes = [action for action, dy, dx in (("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1))
                  if board[row + dy][col + dx] in "$*"]
        draw = self.rng.random()
        previous = query["attempt_history"][-1:]
        action = previous[0]["action"] if (previous and not previous[0]["no_op"]
                                           and previous[0]["action"] in pushes) else pushes[int(draw * len(pushes))]
        self.decisions.append({"model_input": query, "requested_action": action, "policy_draw": draw})
        return json.dumps({"action": action})


def _sokoban() -> JSONObject:
    queries = ROOT / "configs/icl/quickstart/sokoban_test_queries.json"
    board = json.loads(queries.read_text(encoding="utf-8"))["test-two"]
    intent = decode_intent({"effects": [{"predicate": "box_at", "row": 1, "col": 4}], "preserve": []})
    policy = _SokobanPolicy(random.Random(SEED))
    executor = SokobanIntentExecutor(lambda _seed: policy)
    prefix = executor.execute(executor.initial_state(SokobanSnapshot(tuple(board)), ()),
                              intent, TASK_GOAL, seed=SEED, cap=1)
    _check(prefix.status == RolloutStatus.ACTION_LIMIT and len(prefix.state.memory) == 1, "Sokoban checkpoint needs one move")
    rng = policy.rng.getstate()
    cases = {"checkpoint_prefix": _report(prefix.rollout, policy.decisions, "live")}
    endpoints = []
    for label, work in (("live", "live"), ("restored", "speculative")):
        clone = random.Random()
        clone.setstate(rng)
        current = _SokobanPolicy(clone)
        executor = SokobanIntentExecutor(lambda _seed: current)
        state = executor.initial_state(SokobanSnapshot(prefix.state.snapshot.rows),
                                       tuple(ActionAttempt(item.action, item.no_op) for item in prefix.state.memory))
        outcome = executor.execute(state, intent, TASK_GOAL, seed=SEED, cap=2)
        _check(outcome.intent_reached and outcome.rollout.first_hit_actions == 1, "Sokoban replay missed the expected hit")
        _check(current.decisions[0]["model_input"]["attempt_history"] == [asdict(item) for item in prefix.state.memory],
               "Sokoban replay did not restore the controller memory")
        cases[label] = _report(outcome.rollout, current.decisions, work)
        endpoints.append(outcome.state)
    _check(endpoints[0] == endpoints[1] and cases["live"]["decisions"] == cases["restored"]["decisions"]
           and cases["live"]["steps"] == cases["restored"]["steps"], "Sokoban replay drift")
    return {"fixture": "quickstart/test-two", "environment_rng": "deterministic", "controller_seed": SEED,
            "checkpoint_memory": [asdict(item) for item in prefix.state.memory], "same_rng_replay_verified": True,
            "cases": cases}


class _ObservedCraftax(CraftaxAdapter):
    """Keep native transition evidence on the evaluator side."""

    def __init__(self) -> None:
        super().__init__(map_size=16, max_steps=5)
        self.trace: list[JSONObject] = []
        self.coalesce = False

    def _advance(self, snapshot: CraftaxSnapshot, action: int | str) -> CraftaxTransition:
        result = super().step(snapshot, action)
        self.trace.append({
            "requested_action": self.action_id(action), "native_action": result.native_action,
            "observation": result.observation, "reward": result.reward, "terminated": result.terminated,
            "truncated": result.truncated, "snapshot_sha256": result.snapshot.fingerprint,
            "physical_sha256": result.snapshot.physical_fingerprint, "native_ticks": result.native_ticks,
            "automatic_ticks": result.automatic_ticks,
        })
        return result

    def step(self, snapshot: CraftaxSnapshot, action: int | str) -> CraftaxTransition:
        result = self._advance(snapshot, action)
        if self.coalesce:
            # Deliver only the second endpoint to the checker. Both ticks run natively.
            following = self._advance(result.snapshot, "NOOP")
            return replace(following, native_action=result.native_action,
                           native_ticks=result.native_ticks + following.native_ticks,
                           automatic_ticks=result.automatic_ticks + following.automatic_ticks)
        return result


def _craftax_case(
    adapter: _ObservedCraftax, state: CraftaxState, goal: CraftaxIntent, cap: int, work: str,
    *, action: str | None = None,
) -> tuple[CraftaxState, JSONObject]:
    decisions: list[JSONObject] = []

    def policy(
        observation: JSONObject, instruction: str, history: tuple[CraftaxActionAttempt, ...],
        intent: CraftaxIntent | None, *, seed: int,
    ) -> int:
        _check(set(observation) == CRAFTAX_OBSERVATION_FIELDS, "Unexpected Craftax model observation")
        query = {"observation": observation, "instruction": instruction, "attempt_history": [asdict(item) for item in history]}
        chosen = action or mock_action(query["observation"], goal, intent)
        decisions.append({"model_input": query, "requested_action": chosen, "policy_seed": seed})
        return adapter.action_id(chosen)

    executor = CraftaxIntentExecutor(adapter, goal, policy)
    restored = executor.initial_state(state.snapshot, tuple(CraftaxActionAttempt(item.action, item.no_op) for item in state.memory))
    start = len(adapter.trace)
    outcome = executor.execute(restored, goal, render_instruction(goal), seed=SEED, cap=cap)
    _check(decisions[0]["model_input"]["attempt_history"] == [asdict(item) for item in state.memory],
           "Craftax replay did not restore the controller memory")
    result = _report(outcome.rollout, decisions, work, native=adapter.trace[start:])
    _check(result["native_ticks"] == sum(tick["native_ticks"] for tick in adapter.trace[start:]), "Craftax tick mismatch")
    return outcome.state, result


def _future_stream_probe(adapter: _ObservedCraftax, world: CraftaxSnapshot) -> JSONObject:
    grass = adapter.stage(world, inventory={"wood": 2, "sapling": 0}, tiles={"RIGHT": "GRASS"}, facing="RIGHT")
    goal = CraftaxIntent.model_validate({
        "effects": [{"predicate": "inventory_at_least", "item": "sapling", "count": 1}],
        "preserve": [{"predicate": "inventory_at_least", "item": "wood", "count": 1}],
    })
    outcomes: set[int] = set()
    branches = []
    for stream in range(64):
        branch = adapter.fork_future(grass, stream)
        _check(branch.physical_fingerprint == grass.physical_fingerprint and adapter.observe(branch) == adapter.observe(grass),
               "Forking the transition key changed initial physical state or observation")
        endpoint, report = _craftax_case(adapter, CraftaxState(branch), goal, 1, "speculative", action="DO")
        saplings = adapter.inventory(endpoint.snapshot)["sapling"]
        branches.append({"stream": stream, "checkpoint_sha256": branch.fingerprint, "sapling_count": saplings, **report})
        outcomes.add(saplings)
        if len(outcomes) == 2:
            break
    _check(outcomes == {0, 1}, "Independent Craftax environment streams did not demonstrate stochastic divergence")
    _check(len({branch["checkpoint_sha256"] for branch in branches}) == len(branches)
           and all(branch["decisions"] == branches[0]["decisions"] for branch in branches),
           "Craftax independent stream probe must hold policy inputs and decisions fixed")
    return {"initial_physical_sha256": grass.physical_fingerprint, "fixed_policy_seed": SEED,
            "sapling_outcomes": sorted(outcomes), "branches": branches}


def _automatic_probe(adapter: _ObservedCraftax, staged: CraftaxSnapshot, goal: CraftaxIntent) -> JSONObject:
    array = import_module("jax.numpy").asarray
    state = staged._state  # pylint: disable=protected-access
    tired = adapter.stage(replace(staged, _state=state.replace(player_energy=array(0, dtype=state.player_energy.dtype))))
    sleeping, entered = _craftax_case(adapter, CraftaxState(tired), goal, 1, "speculative", action="SLEEP")
    _check(adapter.observe(sleeping.snapshot)["sleeping"], "Native SLEEP did not enter sleep")
    _, observed = _craftax_case(adapter, sleeping, goal, 1, "speculative", action="DO")
    _check(observed["automatic_ticks"] == 1 and observed["executed_primitive_actions"] == 0
           and observed["steps"][0]["preservation_holds"] is True, "Failed to check preservation on an observed automatic tick")
    adapter.coalesce = True
    try:
        _, hidden = _craftax_case(adapter, sleeping, goal, 1, "speculative", action="DO")
    finally:
        adapter.coalesce = False
    _check(hidden["automatic_ticks"] == 2 and hidden["executed_primitive_actions"] == 0
           and hidden["stop_reason"] == RolloutStatus.UNKNOWN and hidden["first_verified_hit_primitive_actions"] is None,
           "Missing automatic tick evidence produced a determinate result")
    return {"enter_sleep": entered, "observed_tick": observed, "withheld_intermediate_tick": hidden}


def _craftax() -> JSONObject:
    task = load_tasks(ROOT / "configs/icl/craftax/tasks.json").by_id("wood-to-stone-pickaxe")
    adapter = _ObservedCraftax()
    world = adapter.reset(task.seed)
    staged = adapter.stage(world, **task.scenario.model_dump())
    prefix, created = _craftax_case(adapter, CraftaxState(staged), task.goal, 1, "live")
    _check(len(prefix.memory) == 1, "Craftax checkpoint needs one recorded primitive")
    endpoint, live = _craftax_case(adapter, prefix, task.goal, 4, "live")
    replay, restored = _craftax_case(adapter, prefix, task.goal, 4, "speculative")
    _check(endpoint.snapshot.fingerprint == replay.snapshot.fingerprint and endpoint.memory == replay.memory
           and all(live[key] == restored[key] for key in live if key != "work"), "Craftax same-key replay drift")
    _check(live["first_verified_hit_primitive_actions"] == 4 and live["task_solved"], "Craftax replay missed the expected hit")
    visible = adapter.observe(prefix.snapshot)
    scale_ignored = mock_proposals(visible, task.goal, 0.2) == mock_proposals(visible, task.goal, 0.8)
    _check(scale_ignored, "Craftax integration mock unexpectedly changed proposals with scale")
    return {
        "fixture": task.task_id, "controller_seed": SEED, "same_rng_replay_verified": True,
        "checkpoint": adapter.snapshot_record(prefix.snapshot), "checkpoint_memory": [asdict(item) for item in prefix.memory],
        "cases": {"checkpoint_prefix": created, "live": live, "restored": restored},
        "independent_future_streams": _future_stream_probe(adapter, world), "automatic_evidence": _automatic_probe(adapter, staged, task.goal),
        "mock_proposals_ignore_scale": scale_ignored,
    }


def run_replay(domain: str, output: Path) -> None:
    """Run a fixture replay without reading campaign training or sealed inputs."""
    result: dict[str, Any] = _sokoban() if domain == "sokoban" else _craftax()
    write_json(output, {"schema": "intent-pilot-replay-v1", "classification": "integration-only", "domain": domain, **result})
