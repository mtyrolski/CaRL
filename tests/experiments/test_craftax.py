"""Craftax snapshots, task contracts and replay across all planning modes."""

import json
import subprocess
import sys
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from carl.environment.craftax import CraftaxAdapter
from carl.icl.config import ROOT
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.planning.active_intent import IntentBudget, IntentSession
from carl.icl.planning.craftax_execution import CraftaxDirect
from carl.icl.planning.craftax_tasks import (
    CraftaxTaskManifest, conjunction, decode_intent, intent_is_admissible, load_tasks, mock_action, mock_proposals,
    render_instruction,
)
from carl.icl.planning.plan_and_act import decode_plan, mock_plan
from carl.icl.types import JSONObject
from scripts import run_predicate_icl


MANIFEST = ROOT / "configs/icl/craftax/tasks.json"


def test_carl_craftax_import_is_lazy() -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-c", (
            "import sys; import carl; from carl.environment.craftax import CraftaxAdapter; "
            "assert not {'jax', 'craftax', 'flax'} & sys.modules.keys()"
        )], check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    if find_spec("craftax") is None:
        with pytest.raises(ImportError, match="Craftax is optional"):
            CraftaxAdapter()


@pytest.fixture(scope="module", name="adapter")
def craftax_adapter() -> CraftaxAdapter:
    pytest.importorskip("craftax", reason="The real Craftax backend is optional")
    return CraftaxAdapter(map_size=16, max_steps=5)


def test_native_craftax_replays_inventory_rewards_and_episode_limit(adapter: CraftaxAdapter) -> None:
    root = adapter.reset(6)
    assert adapter.fingerprint(root) == adapter.fingerprint(adapter.reset(6))
    assert adapter.fingerprint(root) != adapter.fingerprint(adapter.reset(7))
    with pytest.raises(ValueError, match="configuration"):
        CraftaxAdapter(map_size=16, max_steps=6).restore(root)
    staged = adapter.stage(
        root, inventory={"wood": 2}, facing="RIGHT",
        tiles={"UP": "CRAFTING_TABLE", "RIGHT": "TREE", "DOWN": "STONE", "LEFT": "GRASS"},
    )
    assert adapter.fingerprint(root) != adapter.fingerprint(staged)
    assert adapter.inventory(root)["wood"] == 0
    visible = adapter.observe(staged)
    assert len(visible["tiles"]) == 9 and all(len(row) == 11 for row in visible["tiles"])
    assert visible["neighbors"]["RIGHT"] == "TREE" and visible["facing"] == "RIGHT"
    assert json.loads(json.dumps(visible)) == visible
    assert not {"seed", "state", "key", "fingerprint", "timestep", "map"} & visible.keys()
    assert len(adapter.action_names) == 43 and adapter.action_id("DO") == 5
    with pytest.raises(ValueError, match="Craftax action"):
        adapter.action_id(True)

    moves = ("DO", "MAKE_WOOD_PICKAXE", "DOWN", "DO", "MAKE_STONE_PICKAXE")
    path = []
    current = staged
    for move in moves:
        transition = adapter.step(current, move)
        path.append(transition)
        current = transition.snapshot
    assert [step.reward for step in path] == [1.0, 1.0, 0.0, 1.0, 1.0]
    assert [step.observation["inventory"]["pickaxe"] for step in path] == [0, 1, 1, 1, 2]
    assert not any(step.terminated for step in path)
    assert [step.truncated for step in path] == [False, False, False, False, True]

    restored = adapter.restore(path[1].snapshot)
    for action, expected in zip(moves[2:], path[2:], strict=True):
        replayed = adapter.step(restored, adapter.action_id(action))
        restored = replayed.snapshot
        assert adapter.snapshot_record(restored) == adapter.snapshot_record(expected.snapshot)
        assert replayed.observation == expected.observation
        assert replayed.reward == expected.reward
        assert (replayed.terminated, replayed.truncated) == (expected.terminated, expected.truncated)
    with pytest.raises(ValueError, match="finished Craftax snapshot"):
        adapter.step(restored, "NOOP")


def test_native_craftax_death_terminates_before_the_episode_limit(adapter: CraftaxAdapter) -> None:
    array = import_module("jax.numpy").asarray
    root = adapter.reset(6)
    native = root._state  # pylint: disable=protected-access
    exhausted = native.replace(
        player_health=array(1, dtype=native.player_health.dtype),
        player_food=array(0, dtype=native.player_food.dtype),
        player_recover=array(-15, dtype=native.player_recover.dtype),
    )
    snapshot = adapter.stage(replace(root, _state=exhausted))
    transition = adapter.step(snapshot, "NOOP")
    assert transition.observation["health"] == 0
    assert transition.terminated and not transition.truncated


def test_native_active_intent_preserves_checkpoints_and_separate_task_goal(adapter: CraftaxAdapter) -> None:
    snapshot = adapter.stage(adapter.reset(6), inventory={"wood": 2}, facing="RIGHT", tiles={"RIGHT": "TREE"})
    preserve = [{"predicate": "inventory_at_least", "item": "wood", "count": 1}]
    task = decode_intent({"effects": [{"predicate": "inventory_at_least", "item": "wood", "count": 3}], "preserve": preserve})
    assigned = decode_intent({"effects": [{"predicate": "inventory_at_least", "item": "stone", "count": 1}], "preserve": preserve})
    domain = CraftaxDirect(adapter, snapshot, task, task=task)
    session = IntentSession(domain, render_instruction(task), IntentBudget(6, 6, 6, 60))

    def decide(messages: list[dict[str, str]]) -> JSONObject:
        query = json.loads(messages[-1]["content"])
        assert set(query["observation"]) == {"neighbors", "inventory", "facing", "health", "sleeping", "resting"}
        assert query["assigned_intent"] == assigned.model_dump(mode="json")
        assert query["history"] == [] and query["previous_intent"] is None
        return {"response": '{"actions":["DO","NOOP"],"intent":"collect wood"}'}

    speculative = session.attempt("probe", assigned.model_dump(mode="json"), decide, speculative=True)
    assert domain.snapshot == snapshot and not session.history
    live = session.attempt("commit", assigned.model_dump(mode="json"), decide)
    assert speculative["states"] == live["states"]
    assert speculative["continuation"] == live["states"][0]
    assert set(live["states"][0]) == {"state_id", "memory_id", "memory", "observation", "native"}
    assert live["states"][0]["native"] == adapter.snapshot_record(snapshot)
    assert live["continuation"]["memory"]["note"] == "collect wood"
    assert live["status"] == "task_solved" and live["task_solved"] and live["first_hit"] is None
    assert live["decisions"][0]["discarded_suffix"] == ["NOOP"] and session.budget.actions == 2


def test_manifest_rejects_reused_seed_and_threshold_only_chain_variants() -> None:
    manifest = load_tasks(MANIFEST)
    assert len(manifest.by_id("collect-wood").chain) == 1
    assert len(manifest.by_id("wood-to-stone-pickaxe").chain) == 4
    serialized = manifest.model_dump(mode="json")
    overlap = {
        **serialized["tasks"][0], "task_id": "duplicate-seed", "split": "test", "chain": [
            {"effects": [{"predicate": "inventory_at_least", "item": "coal", "count": 1}]},
        ],
    }
    with pytest.raises(ValueError, match="seed 11 crosses"):
        CraftaxTaskManifest.model_validate({**serialized, "tasks": [*serialized["tasks"], overlap]})
    repeated = {
        **overlap, "task_id": "timber-rescaled", "seed": 64, "chain": [
            {"effects": [{"predicate": "inventory_at_least", "item": "wood", "count": count}]}
            for count in (2, 4)
        ],
    }
    with pytest.raises(ValueError, match="dependency chain .* crosses"):
        CraftaxTaskManifest.model_validate({**serialized, "tasks": [*serialized["tasks"], repeated]})


def test_inventory_and_tiers_are_checked_as_a_conjunction() -> None:
    goal = load_tasks(MANIFEST).by_id("wood-to-stone-pickaxe").goal
    assert conjunction(goal.effects, {"wood": 1, "pickaxe": 2}) is True
    assert conjunction(goal.effects, {"wood": 0, "pickaxe": 2}) is False
    assert conjunction(goal.effects, {"wood": 2}) is None
    assert intent_is_admissible(goal, {"wood": 2, "pickaxe": 1})
    assert not intent_is_admissible(goal, {"wood": 0, "pickaxe": 1})
    assert not intent_is_admissible(goal, {"wood": 2, "pickaxe": 2})
    assert "while maintaining" in render_instruction(goal)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        decode_intent('{"effects":[],"effects":[]}')
    with pytest.raises(ValueError):
        decode_intent({"effects": [{"predicate": "tool_tier_at_least", "tool": "pickaxe", "tier": True}]})


def test_plan_and_act_uses_public_goal_and_observation_and_bounds_plan() -> None:
    goal = load_tasks(MANIFEST).by_id("wood-to-stone-pickaxe").goal
    observation = {
        "inventory": {"wood": 2, "stone": 0, "pickaxe": 0},
        "neighbors": {"UP": "CRAFTING_TABLE", "RIGHT": "TREE", "DOWN": "STONE", "LEFT": "GRASS"},
        "facing": "RIGHT",
    }
    plan = mock_plan(observation, goal, max_steps=4)
    assert len(plan) == 4
    assert mock_proposals(observation, goal, None) == plan[:1]
    assert mock_action(observation, goal, None) == mock_action(observation, goal, plan[0]) == "DO"
    assert all(step.preserve == goal.preserve for step in plan)
    serialized = json.dumps({"intents": [step.model_dump(mode="json") for step in plan]})
    assert decode_plan(serialized, max_steps=4) == plan
    with pytest.raises(ValueError, match="at most 3"):
        decode_plan(serialized, max_steps=3)
    with pytest.raises(ValueError, match="exactly one intents list"):
        decode_plan({"actions": ["DO"]})


@pytest.mark.skipif(find_spec("craftax") is None, reason="optional pinned Craftax runtime")
def test_native_craftax_four_modes_replay_one_matched_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "matched.json"
    roles = set()
    call = ExperimentAPI.call

    def record(api: ExperimentAPI, endpoint: str, payload: JSONObject, *args: Any, **kwargs: Any) -> JSONObject:
        query = json.loads(payload["input"][0]["content"][0]["text"])
        private = {"_state", "_key", "seed", "fingerprint", "chain", "snapshot"}
        assert private.isdisjoint(query) and private.isdisjoint(query["observation"])
        roles.add(payload["metadata"]["operation"])
        return call(api, endpoint, payload, *args, **kwargs)

    monkeypatch.setattr(ExperimentAPI, "call", record)
    monkeypatch.setattr(sys, "argv", [
        "run_predicate_icl", "craftax", "--tasks", str(MANIFEST), "--task", "wood-to-stone-pickaxe",
        "--mode", "mock", "--planning-mode", "all", "--output", str(path),
    ])
    assert run_predicate_icl.main() == 0
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data["results"]
    assert roles == {"action", "intent", "value", "plan"}
    assert set(results) == {"direct", "one-step", "full-search", "plan-and-act"}
    assert data["execution_mode"] == "mock" and data["preparation"]["attempted_simulator_actions"] == 1
    assert data["policy"]["fingerprint"] and data["task_manifest_sha256"]
    assert not {"_state", "_key", "seed", "fingerprint"} & data["initial_observation"].keys()
    expected_actions = [5, 11, 4, 5, 12]
    expected_rewards = [1.0, 1.0, 0.0, 1.0, 1.0]
    for result in results.values():
        assert result["solved"] and result["action_path"] == expected_actions
        assert result["nodes"][0]["state"]["snapshot"] == data["initial_snapshot"]
        assert result["endpoint"]["truncated"] and not result["endpoint"]["terminated"]
        assert result["accounting"].keys() == results["direct"]["accounting"].keys()
        assert result["accounting"]["physical_requests"] == 0
        assert result["accounting"]["executed_plan_actions"] == 5
        assert {"_state", "_key"}.isdisjoint(result["endpoint"])
        mapped = {attempt["node"]: attempt for attempt in result["attempts"] if attempt["node"] is not None}
        steps = []
        node = result["selected_node"]
        while node:
            steps.insert(0, mapped[node]["rewards"])
            node = mapped[node]["parent"]
        assert [reward for step in steps for reward in step] == expected_rewards
    assert results["one-step"]["accounting"]["attempted_simulator_actions"] > 5
    assert results["plan-and-act"]["accounting"]["logical_generator_calls"] == 1
    assert len({result["endpoint"]["fingerprint"] for result in results.values()}) == 1
