"""Collect the pilot's replayable demonstrations independently of annotation."""

import json
from collections.abc import Iterator, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from carl.environment.craftax import CraftaxAdapter, CraftaxSnapshot
from carl.icl.api import write_json
from carl.icl.config import ROOT
from carl.icl.experiments.intent_pilot import SEEDS, SPLITS, _digest
from carl.icl.intent_domain import TASK_GOAL, SokobanSnapshot, replay_fragment
from carl.icl.planning.craftax_tasks import CraftaxTask, conjunction, load_tasks, render_instruction


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _positions(rows: Sequence[str], symbols: str) -> frozenset[tuple[int, int]]:
    return frozenset((y, x) for y, row in enumerate(rows) for x, tile in enumerate(row) if tile in symbols)


def choose_sokoban_segment(frames: Sequence[SokobanSnapshot]) -> tuple[int, tuple[int, ...]]:
    """Select three geometrically distinct endpoints from a single starting observation."""
    boxes = [_positions(frame.rows, "$*") for frame in frames]
    goals = [_positions(frame.rows, "*") for frame in frames]
    best: tuple[int, ...] | None = None
    selection = None
    for start in range(len(frames) - 3):
        seen = set()
        ends = []
        retained = goals[start]
        for end in range(start + 1, min(start + 25, len(frames))):
            retained = retained & goals[end]
            changed_boxes = boxes[end] - boxes[start]
            if changed_boxes and changed_boxes not in seen:
                seen.add(changed_boxes)
                ends.append((end, len(retained), len(goals[end] - goals[start])))
        if len(ends) >= 3:
            preferred = sorted(ends, key=lambda item: (-bool(item[1] and item[2]), -item[1], item[0]))[:3]
            score = (sum(bool(keep and gain) for _, keep, gain in preferred), len(goals[start]), -start)
            if best is None or score > best:
                best = score
                selection = start, tuple(sorted(end for end, _, _ in preferred))
    if selection is None:
        raise ValueError("Sokoban witness has no start with three distinct box-change endpoints")
    return selection


def sokoban_episodes(root: Path) -> Iterator[dict]:
    """Replay the frozen source episodes, keeping full boards and actions private."""
    for split in ("train-support", "validation"):
        for source in _read(root / f"splits/sokoban-{split}.json")["episodes"]:
            demonstration = source["demonstration"]
            frames = replay_fragment(SokobanSnapshot(tuple(demonstration["start_observation"])), demonstration["actions"])
            last = frames[-1].rows
            if _positions(last, "$"):
                raise ValueError(f"Unsolved frozen Sokoban witness {demonstration['source_id']}")
            start, ends = choose_sokoban_segment(frames)
            yield {
                "episode_id": source["episode_id"], "group_id": source["canonical_layout_id"], "pilot_split": split,
                "task_goal": TASK_GOAL, "source": demonstration, "fixture": False,
                "trajectory_provenance": {
                    "kind": "archived_local_solver_witness", "source_files": source["source_files_from_workspace"],
                    "source_seed": source["source_seed"],
                },
                "observations": [frame.rows for frame in frames], "actions": demonstration["actions"],
                "snapshots": [{"observation_index": i, "memory_prefix_actions": i, "sha256": _digest(frame.rows)}
                              for i, frame in enumerate(frames)],
                "facts": [{"occupied_goals": sorted(_positions(frame.rows, "*")), "boxes": sorted(_positions(frame.rows, "$*"))}
                          for frame in frames],
                "start": start, "ends": ends, "replay_verified": True,
            }


def _requirement(item: str, amount: int) -> dict:
    if item in {"pickaxe", "sword"}:
        return {"predicate": "tool_tier_at_least", "tool": item, "tier": amount}
    return {"predicate": "inventory_at_least", "item": item, "count": amount}


def _task(identifier: str, seed: int, inventory: dict, tiles: dict, facing: str, milestones: list[tuple[str, int]]) -> CraftaxTask:
    target: dict[str, int] = dict(milestones)
    # All gather, craft and survival constraints are explicit in the public goal.
    retain = {"wood": 1} if inventory.get("wood", 0) > 0 and "wood" not in target else {}
    return CraftaxTask.model_validate({
        "task_id": identifier, "seed": seed, "split": "train",
        "scenario": {"inventory": inventory, "tiles": tiles, "facing": facing},
        "chain": [{"effects": [_requirement(item, amount)]} for item, amount in milestones],
        "goal": {"effects": [_requirement(item, amount) for item, amount in target.items()],
                 "preserve": [_requirement(item, amount) for item, amount in retain.items()]},
    })


def _craftax_plans(root: Path) -> list[dict]:
    # Read only grouping identifiers from withheld cohorts, never use their observations or instructions.
    reserved_chains = set()
    used_worlds = set()
    for split in SPLITS:
        path = root / ("sealed" if split == "sealed-test" else "splits") / f"craftax-{split}.json"
        for row in _read(path)["episodes"]:
            used_worlds.add(row["world_id"])
            if split != "train-support":
                reserved_chains.add(row["chain_id"])
    fixture = load_tasks(ROOT / "configs/icl/craftax/tasks.json")
    reserved_chains.add(_digest(fixture.by_id("wood-to-stone-pickaxe").chain_signature))
    fixture_seeds = {task.seed for task in fixture.tasks}

    def seed_for(identifier: str) -> int:
        return int(_digest([SEEDS["craftax_world"], "annotations-v1", identifier])[:8], 16)

    plans = []
    gathers = (
        ("wood", "TREE", {"wood": 1, "stone": 1}, ("RIGHT", "DOWN", "LEFT")),
        ("stone", "STONE", {"pickaxe": 1, "wood": 2}, ("DOWN", "LEFT", "UP")),
        ("coal", "COAL", {"pickaxe": 2, "wood": 2, "coal": 1}, ("LEFT", "UP", "RIGHT")),
        ("iron", "IRON", {"pickaxe": 2, "wood": 2}, ("UP", "RIGHT", "DOWN")),
    )
    for item, block, inventory, directions in gathers:
        identifier = f"cx-stage03-gather-{item}"
        start = inventory.get(item, 0)
        task = _task(identifier, seed_for(identifier), inventory, dict.fromkeys(directions, block),
                     directions[0], [(item, start + gain) for gain in (1, 2, 3)])
        actions = ["DO", directions[1], "DO", directions[2], "DO"]
        plans.append({"task": task.model_dump(mode="json"), "actions": actions, "ends": [1, 3, 5]})
    crafts = (
        (("WOOD_PICKAXE", "WOOD_SWORD", "STONE_SWORD"), (("pickaxe", 1), ("sword", 1), ("sword", 2)), "RIGHT"),
        (("WOOD_SWORD", "WOOD_PICKAXE", "STONE_PICKAXE"), (("sword", 1), ("pickaxe", 1), ("pickaxe", 2)), "DOWN"),
        (("STONE_PICKAXE", "WOOD_SWORD", "STONE_SWORD"), (("pickaxe", 2), ("sword", 1), ("sword", 2)), "LEFT"),
        (("STONE_SWORD", "WOOD_PICKAXE", "STONE_PICKAXE"), (("sword", 2), ("pickaxe", 1), ("pickaxe", 2)), "UP"),
    )
    for index, (actions, milestones, direction) in enumerate(crafts):
        identifier = f"cx-stage03-craft-{index + 1}"
        task = _task(identifier, seed_for(identifier), {"wood": 5 + index, "stone": 2},
                     {direction: "CRAFTING_TABLE"}, direction, list(milestones))
        plans.append({"task": task.model_dump(mode="json"), "actions": [f"MAKE_{name}" for name in actions], "ends": [1, 2, 3]})
    chain_ids = set()
    for plan in plans:
        task = CraftaxTask.model_validate(plan["task"])
        chain = task.chain_signature
        world_id = _digest(["craftax-1.1.0-symbolic", 16, task.seed])
        subchains = {_digest(part) for size in range(1, len(chain) + 1) for part in combinations(chain, size)}
        if task.seed in fixture_seeds or world_id in used_worlds or subchains & reserved_chains:
            raise ValueError("New support world or chain overlaps a withheld group")
        plan.update(world_id=world_id, chain_id=_digest(chain), pilot_split="train-support")
        used_worlds.add(world_id)
        chain_ids.add(plan["chain_id"])
    if len(chain_ids) != 8:
        raise ValueError("The support extension needs eight independent chains")

    known_validation = _read(root / "splits/craftax-validation.json")["episodes"]
    first_per_chain = {}
    for row in known_validation:
        first_per_chain.setdefault(row["chain_id"], row)
    # Use one existing world per validation chain. Facing alone adds no inventory effect.
    for row in first_per_chain.values():
        goal = row["task"]["task_id"]
        to_stone = next(direction for direction, tile in row["task"]["scenario"]["tiles"].items() if tile == "STONE")
        if goal.startswith("cx-val-01"):
            actions, ends = ["MAKE_WOOD_SWORD", to_stone, "DO"], [1, 2, 3]
        elif goal.startswith("cx-val-02"):
            actions, ends = ["DO", to_stone, "DO", "MAKE_STONE_PICKAXE"], [1, 3, 4]
        else:
            raise ValueError("Unknown frozen Craftax validation group")
        plans.append({"task": row["task"], "world_id": row["world_id"], "chain_id": row["chain_id"],
                      "actions": actions, "ends": ends, "pilot_split": "validation"})
    return plans


def _advance(adapter: CraftaxAdapter, root: CraftaxSnapshot, actions: Sequence[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Capture native facts, memory references and complete state fingerprints at each step."""
    snapshot = adapter.restore(root)
    observations = [adapter.observe(snapshot)]
    states = [{**adapter.snapshot_record(snapshot), "physical_sha256": snapshot.physical_fingerprint,
               "memory_prefix_actions": 0, "elapsed_primitive_actions": 0}]
    steps = []
    for action in actions:
        step = adapter.step(snapshot, action)
        snapshot = step.snapshot
        observations.append(step.observation)
        states.append({**adapter.snapshot_record(snapshot), "physical_sha256": snapshot.physical_fingerprint,
                       "memory_prefix_actions": len(steps) + 1,
                       "elapsed_primitive_actions": states[-1]["elapsed_primitive_actions"] + int(step.primitive_action_executed)})
        steps.append({
            "requested_action": action, "native_action": adapter.action_names[step.native_action],
            "executed_primitive": step.primitive_action_executed, "native_ticks": step.native_ticks,
            "automatic_ticks": step.automatic_ticks, "reward": step.reward,
            "terminated": step.terminated, "truncated": step.truncated, "no_op": step.no_op,
        })
    return observations, states, steps


def craftax_episodes(root: Path, private: Path) -> Iterator[dict]:
    """Freeze the support supplement, collect native trajectories, and check same-key replay."""
    plans = _craftax_plans(root)
    roster = private / "craftax-collection-plan.json"
    payload = {"map_size": 16, "max_steps": 32, "plans": plans}
    if roster.exists() and _read(roster) != payload:
        raise ValueError("The existing Craftax collection plan differs from the frozen plan")
    if not roster.exists():
        write_json(roster, payload)
    adapter = CraftaxAdapter(map_size=16, max_steps=32)
    seen_worlds = set()
    for plan in plans:
        task = CraftaxTask.model_validate(plan["task"])
        world = adapter.reset(task.seed)
        if world.physical_fingerprint in seen_worlds:
            raise ValueError("Craftax collection produced a duplicate native world")
        seen_worlds.add(world.physical_fingerprint)
        snapshot = adapter.stage(world, **task.scenario.model_dump())
        trace = _advance(adapter, snapshot, plan["actions"])
        if trace != _advance(adapter, snapshot, plan["actions"]):
            raise ValueError(f"Craftax native checkpoint could not be replayed for {task.task_id}")
        observations, states, steps = trace
        facts = [observation["inventory"] for observation in observations]
        if conjunction(task.goal.effects, facts[-1]) is not True:
            raise ValueError(f"Craftax scripted trajectory did not reach its public goal for {task.task_id}")
        if any(conjunction(task.goal.preserve, inventory) is not True for inventory in facts):
            raise ValueError(f"Craftax scripted trajectory lost a required resource for {task.task_id}")
        yield {
            "episode_id": task.task_id, "group_id": plan["chain_id"], "world_id": plan["world_id"],
            "world_physical_sha256": world.physical_fingerprint, "pilot_split": plan["pilot_split"],
            "task_goal": render_instruction(task.goal), "native_task": plan["task"], "observations": observations,
            "snapshots": states, "steps": steps, "facts": facts, "actions": plan["actions"],
            "start": 0, "ends": plan["ends"], "fixture": False, "replay_verified": True,
            "trajectory_provenance": {"kind": "scripted_native_craftax", "seed": task.seed, "controller_weights": None},
        }
