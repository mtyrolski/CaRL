"""Typed Craftax effects, partitioned task definitions and the offline recipe policy."""

import json
from collections.abc import Mapping, Sequence
from itertools import groupby
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

from pydantic import ConfigDict, Field, model_validator

from carl.icl.config import Settings


Resource = Literal[
    "wood", "stone", "coal", "iron", "diamond", "sapling", "bow", "arrows", "torches",
    "ruby", "sapphire", "books",
]
Tool = Literal["pickaxe", "sword"]
Direction = Literal["UP", "DOWN", "LEFT", "RIGHT"]
InventoryItem = Resource | Tool
Split = Literal["train", "validation", "test"]
Counts: TypeAlias = Mapping[str, int]


class InventoryThreshold(Settings):
    """A minimum visible count for a scalar inventory item."""

    model_config = ConfigDict(frozen=True)
    predicate: Literal["inventory_at_least"] = "inventory_at_least"
    item: Resource
    count: int = Field(ge=1, strict=True)


class ToolThreshold(Settings):
    """A minimum ordinal tool tier in full Craftax."""

    model_config = ConfigDict(frozen=True)
    predicate: Literal["tool_tier_at_least"] = "tool_tier_at_least"
    tool: Tool
    tier: int = Field(ge=1, le=4, strict=True)


CraftaxPredicate = Annotated[InventoryThreshold | ToolThreshold, Field(discriminator="predicate")]


class CraftaxIntent(Settings):
    """A simultaneous goal conjunction and invariants for one rollout."""

    model_config = ConfigDict(frozen=True)
    effects: tuple[CraftaxPredicate, ...] = Field(min_length=1)
    preserve: tuple[CraftaxPredicate, ...] = ()
    instruction: None = None


class CraftaxScenario(Settings):
    """Explicit controlled changes applied once to a seeded native reset."""

    inventory: dict[InventoryItem, Annotated[int, Field(ge=0, strict=True)]] = Field(default_factory=dict)
    tiles: dict[Direction, str] = Field(default_factory=dict)
    facing: Direction | None = None


class CraftaxTask(Settings):
    """A task goal and its independent reference dependency chain."""

    task_id: str = Field(min_length=1)
    split: Split
    seed: int = Field(ge=0, le=2**32 - 1, strict=True)
    scenario: CraftaxScenario = Field(default_factory=CraftaxScenario)
    chain: tuple[CraftaxIntent, ...] = Field(min_length=1)
    goal: CraftaxIntent

    @property
    def chain_signature(self) -> tuple[tuple[str, ...], ...]:
        """Identify an ordered composition independently of seed, IDs and inventory thresholds."""
        return tuple(signature for signature, _ in groupby(_stage_signature(stage) for stage in self.chain))


class CraftaxTaskManifest(Settings):
    """Tasks separated by both seed and semantic chain, before evaluation."""

    version: Literal["craftax-tasks-v1"] = "craftax-tasks-v1"
    tasks: tuple[CraftaxTask, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def disjoint_splits(self) -> "CraftaxTaskManifest":
        """Reject duplicate IDs, missing splits and shared seeds or chain identities."""
        ids: set[str] = set()
        seed_splits: dict[int, Split] = {}
        chain_splits: dict[tuple[tuple[str, ...], ...], Split] = {}
        for task in self.tasks:
            if task.task_id in ids:
                raise ValueError(f"duplicate Craftax task {task.task_id}")
            ids.add(task.task_id)
            if seed_splits.setdefault(task.seed, task.split) != task.split:
                raise ValueError(f"Craftax seed {task.seed} crosses dataset splits")
            if chain_splits.setdefault(task.chain_signature, task.split) != task.split:
                raise ValueError(f"Craftax dependency chain for {task.task_id} crosses dataset splits")
        if set(seed_splits.values()) != {"train", "validation", "test"}:
            raise ValueError("Craftax manifest needs train, validation and test tasks")
        return self

    def by_id(self, task_id: str) -> CraftaxTask:
        """Look up a unique task from the checked manifest."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise ValueError(f"Unknown Craftax task {task_id}")


def _stage_signature(intent: CraftaxIntent) -> tuple[str, ...]:
    items = {f"item:{p.item}" for p in intent.effects if isinstance(p, InventoryThreshold)}
    tools: dict[str, int] = {}
    for predicate in intent.effects:
        if isinstance(predicate, ToolThreshold):
            tools[predicate.tool] = max(predicate.tier, tools.get(predicate.tool, 0))
    return tuple(sorted(items | {f"tool:{tool}:{tier}" for tool, tier in tools.items()}))


def load_tasks(path: Path) -> CraftaxTaskManifest:
    """Load task definitions and verify partition isolation."""
    return CraftaxTaskManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, entry in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key}")
        value[key] = entry
    return value


def decode_intent(value: object) -> CraftaxIntent:
    """Decode a typed intent and reject duplicate keys or free-text instructions."""
    if isinstance(value, str):
        value = json.loads(value, object_pairs_hook=_unique_object)
    return CraftaxIntent.model_validate(value)


def predicate_holds(predicate: CraftaxPredicate, inventory: Counts) -> bool | None:
    """Return unknown when the stated field has no known count."""
    if isinstance(predicate, ToolThreshold):
        observed = inventory.get(predicate.tool)
        target = predicate.tier
    else:
        observed = inventory.get(predicate.item)
        target = predicate.count
    return None if observed is None else observed >= target


def conjunction(predicates: Sequence[CraftaxPredicate], inventory: Counts) -> bool | None:
    """Check all predicates using false-dominant three-valued conjunction."""
    results = [predicate_holds(predicate, inventory) for predicate in predicates]
    if False in results:
        return False
    return None if None in results else True


def intent_is_admissible(intent: CraftaxIntent, inventory: Counts) -> bool:
    """Require a known unmet effect conjunction and verified initial invariants."""
    effects = [predicate_holds(predicate, inventory) for predicate in intent.effects]
    return False in effects and None not in effects and conjunction(intent.preserve, inventory) is True


def render_instruction(intent: CraftaxIntent) -> str:
    """Describe exactly the typed thresholds without an executable free-text field."""
    def clause(predicate: CraftaxPredicate) -> str:
        if isinstance(predicate, ToolThreshold):
            return f"{predicate.tool} tier at least {predicate.tier}"
        return f"at least {predicate.count} {predicate.item}"

    aim = "Reach " + " and ".join(map(clause, intent.effects))
    if intent.preserve:
        aim += " while maintaining " + " and ".join(map(clause, intent.preserve))
    return aim + "."


def _visible_inventory(observation: Mapping[str, object]) -> dict[str, int]:
    inventory = observation.get("inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("Craftax recipe policy requires the visible inventory")
    return {
        key: value for key, value in inventory.items()
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
    }


def recipe_objectives(observation: Mapping[str, object], goal: CraftaxIntent) -> tuple[CraftaxIntent, ...]:
    """Derive remaining wood, stone and pickaxe objectives from public observations and a goal."""
    visible = _visible_inventory(observation)
    if not intent_is_admissible(goal, visible):
        return ()
    stock = visible.copy()
    steps: list[CraftaxIntent] = []

    def append(predicate: CraftaxPredicate) -> None:
        if predicate_holds(predicate, stock) is False:
            steps.append(CraftaxIntent(effects=(predicate,), preserve=goal.preserve))
            if isinstance(predicate, ToolThreshold):
                stock[predicate.tool] = predicate.tier
            else:
                stock[predicate.item] = predicate.count

    def requested(item: str, predicates: Sequence[CraftaxPredicate]) -> int:
        return max((
            p.count for p in predicates if isinstance(p, InventoryThreshold) and p.item == item
        ), default=0)

    if any(item not in stock for item in ("wood", "stone", "pickaxe")):
        return ()
    current_tier = stock["pickaxe"]
    target_tier = max((
        p.tier for p in goal.effects if isinstance(p, ToolThreshold) and p.tool == "pickaxe"
    ), default=0)
    if target_tier > 2 and current_tier < target_tier:
        # This deterministic mock covers only the two recipes in the controlled tasks.
        return ()
    stone_upgrade = current_tier < 2 <= target_tier
    stone_needed = max(requested("stone", goal.effects), requested("stone", goal.preserve)) + int(stone_upgrade)
    wooden_pickaxe = current_tier == 0 and (target_tier > 0 or stone_needed > stock["stone"])
    wood_needed = max(requested("wood", goal.effects), requested("wood", goal.preserve))
    wood_needed += int(wooden_pickaxe) + int(stone_upgrade)

    if wood_needed > stock["wood"]:
        append(InventoryThreshold(item="wood", count=wood_needed))
    if wooden_pickaxe:
        append(ToolThreshold(tool="pickaxe", tier=1))
        stock["wood"] -= 1
    if stone_needed > stock["stone"]:
        append(InventoryThreshold(item="stone", count=stone_needed))
    if stone_upgrade:
        append(ToolThreshold(tool="pickaxe", tier=2))
        stock["wood"] -= 1
        stock["stone"] -= 1
    if conjunction(goal.effects, stock) is False:
        steps.append(goal)
    return tuple(steps)


def mock_proposals(
    observation: Mapping[str, object], goal: CraftaxIntent, _scale: float | None,
) -> tuple[CraftaxIntent, ...]:
    """Propose the next recipe objective without accessing the task's reference chain."""
    return recipe_objectives(observation, goal)[:1]


def mock_value(observation: Mapping[str, object], goal: CraftaxIntent) -> float:
    """Return a deterministic recipe-distance heuristic for the pilot."""
    if conjunction(goal.effects, _visible_inventory(observation)) is True:
        return 1.0
    steps = recipe_objectives(observation, goal)
    return 1.0 / (1 + len(steps)) if steps else 0.0


def _craft_action(
    predicate: ToolThreshold, inventory: Counts, neighbors: Mapping[str, object],
) -> str:
    if (predicate.tool != "pickaxe" or "CRAFTING_TABLE" not in neighbors.values()
            or inventory.get("wood", 0) < 1):
        return "NOOP"
    if predicate.tier == 1:
        return "MAKE_WOOD_PICKAXE"
    return "MAKE_STONE_PICKAXE" if predicate.tier == 2 and inventory.get("stone", 0) >= 1 else "NOOP"


def mock_action(
    observation: Mapping[str, object], goal: CraftaxIntent, intent: CraftaxIntent | None,
) -> str:
    """Execute an objective using visible inventory, facing and neighboring terrain."""
    objectives = recipe_objectives(observation, intent or goal)
    if not objectives:
        return "NOOP"
    unmet = objectives[0].effects[0]
    neighbors = observation.get("neighbors")
    if not isinstance(neighbors, Mapping):
        return "NOOP"
    if isinstance(unmet, ToolThreshold):
        return _craft_action(unmet, _visible_inventory(observation), neighbors)
    material = {"wood": {"TREE", "FIRE_TREE", "ICE_SHRUB"}, "stone": {"STONE"}}.get(unmet.item, set())
    facing = observation.get("facing")
    if isinstance(facing, str) and neighbors.get(facing) in material:
        return "DO"
    return next((direction for direction in ("UP", "DOWN", "LEFT", "RIGHT")
                 if neighbors.get(direction) in material), "NOOP")
