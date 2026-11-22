"""Craftax task and Plan-and-Act contracts without the optional simulator."""

import json
from pathlib import Path

import pytest

from carl.icl.planning.craftax_tasks import (
    CraftaxTaskManifest, conjunction, decode_intent, intent_is_admissible, load_tasks, mock_action, mock_proposals,
    render_instruction,
)
from carl.icl.planning.plan_and_act import decode_plan, mock_plan


MANIFEST = Path(__file__).resolve().parents[2] / "configs" / "icl" / "craftax" / "tasks.json"


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
