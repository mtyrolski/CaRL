"""Verify the optional real Craftax simulator and its replay boundary."""

import json
import subprocess
import sys
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec

import pytest

from carl.environment.craftax import CraftaxAdapter


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
