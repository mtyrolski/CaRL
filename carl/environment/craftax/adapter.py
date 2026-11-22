"""Pure snapshot transitions for the pinned, non-auto-reset Craftax environment."""

import hashlib
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from importlib import import_module, metadata
from typing import Any, Final

import numpy as np


CRAFTAX_VERSION: Final = "1.1.0"
JAX_VERSION: Final = "0.4.23"
OBSERVATION_VERSION: Final = "craftax-symbolic-visible-v1"
_DIRECTIONS: Final = {"LEFT": (0, -1), "RIGHT": (0, 1), "UP": (-1, 0), "DOWN": (1, 0)}
_INVENTORY_FIELDS: Final = (
    "wood", "stone", "coal", "iron", "diamond", "sapling", "pickaxe", "sword", "bow", "arrows",
    "torches", "ruby", "sapphire", "books",
)


@dataclass(frozen=True, eq=False)
class CraftaxSnapshot:
    """An immutable native Flax state and the PRNG for its next transition."""

    seed: int
    config: tuple[int, int]
    terminated: bool
    truncated: bool
    _state: Any = field(repr=False)
    _key: Any = field(repr=False)
    _symbolic: Any = field(repr=False)

    @cached_property
    def fingerprint(self) -> str:
        """Identify complete simulator state, including hidden timers and RNG."""
        return self._digest(self._state, (self._key,))

    @cached_property
    def physical_fingerprint(self) -> str:
        """Identify the native state independently of its current and next RNG keys."""
        return self._digest(self._state.replace(state_rng=None), ())

    def _digest(self, state: Any, keys: tuple[Any, ...]) -> str:
        jax = import_module("jax")
        leaves, tree = jax.tree_util.tree_flatten(state)
        h = hashlib.sha256()
        h.update(f"craftax-{CRAFTAX_VERSION}:{self.config}:{self.terminated}:{self.truncated}:{tree}".encode())
        for leaf in [*leaves, *keys]:
            array = np.asarray(leaf)
            h.update(f"{array.shape}:{array.dtype.str}:".encode())
            h.update(array.tobytes(order="C"))
        return h.hexdigest()


@dataclass(frozen=True)
class CraftaxTransition:
    """A native action result with separate episode termination and truncation."""

    snapshot: CraftaxSnapshot
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    no_op: bool
    native_ticks: int
    automatic_ticks: int
    native_action: int

    @property
    def primitive_action_executed(self) -> bool:
        """Count an explicit primitive only when it reaches the native action handler."""
        return self.native_ticks > self.automatic_ticks


class CraftaxAdapter:
    """Use the official 43-action symbolic environment with explicit CPU snapshots."""

    def __init__(self, *, max_steps: int = 64, map_size: int = 16) -> None:
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(map_size, int) or isinstance(map_size, bool) or map_size < 16 or map_size % 16:
            raise ValueError("map_size must be a positive multiple of 16")
        try:
            for distribution, version in (("craftax", CRAFTAX_VERSION), ("jax", JAX_VERSION), ("jaxlib", JAX_VERSION)):
                if metadata.version(distribution) != version:
                    raise ImportError(f"Craftax adapter requires {distribution}=={version}")
            self._jax = import_module("jax")
            self._cpu = self._jax.devices("cpu")[0]
            with self._jax.default_device(self._cpu):
                constants = import_module("craftax.constants")
                state_module = import_module("craftax.craftax_state")
                env_module = import_module("craftax.envs.craftax_symbolic_env")
                self._has_beaten_boss = import_module("craftax.util.game_logic_utils").has_beaten_boss
        except (metadata.PackageNotFoundError, ModuleNotFoundError) as error:
            raise ImportError("Craftax is optional. Install requirements/craftax-1.1.0.txt in a domain environment.") from error

        self._config = (map_size, max_steps)
        self._static = state_module.StaticEnvParams(map_size=(map_size, map_size))
        self._env = env_module.CraftaxSymbolicEnv(self._static)
        self._params = self._env.default_params.replace(max_timesteps=max_steps)
        self._render = self._jax.jit(self._env.get_obs)
        self._actions = constants.Action
        self._block = constants.BlockType
        self._item = constants.ItemType
        self._size = tuple(constants.OBS_DIM)
        self._map_shape = tuple(self._env.get_map_obs_shape())

    @property
    def action_names(self) -> tuple[str, ...]:
        """Return canonical action names in native discrete-action order."""
        return tuple(self._actions(index).name for index in range(self._env.num_actions))

    def action_id(self, action: str | int) -> int:
        """Map an action name or native index without silently coercing inputs."""
        try:
            if isinstance(action, str):
                return int(self._actions[action].value)
            if isinstance(action, int) and not isinstance(action, bool):
                return int(self._actions(action).value)
        except (KeyError, ValueError) as error:
            raise ValueError(f"Unknown Craftax action: {action!r}") from error
        raise ValueError("Craftax action must be a canonical name or integer index")

    def reset(self, seed: int) -> CraftaxSnapshot:
        """Create a deterministic native world and a distinct transition RNG stream."""
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
            raise ValueError("Craftax seed must be an unsigned 32-bit integer")
        with self._jax.default_device(self._cpu):
            key, reset_key = self._jax.random.split(self._jax.random.PRNGKey(seed))
            symbolic, state = self._env.reset(reset_key, self._params)
        return CraftaxSnapshot(seed, self._config, False, False, state, key, symbolic)

    def restore(self, snapshot: CraftaxSnapshot) -> CraftaxSnapshot:
        """Restore a compatible snapshot without mutating any existing branch."""
        if not isinstance(snapshot, CraftaxSnapshot) or snapshot.config != self._config:
            raise ValueError("Snapshot does not match this Craftax world configuration")
        return snapshot

    def fork_future(self, snapshot: CraftaxSnapshot, stream: int) -> CraftaxSnapshot:
        """Derive a separate future transition stream on the same physical state."""
        if not isinstance(stream, int) or isinstance(stream, bool) or not 0 <= stream < 2**32:
            raise ValueError("Craftax stream must be an unsigned 32-bit integer")
        snapshot = self.restore(snapshot)
        with self._jax.default_device(self._cpu):
            return replace(snapshot, _key=self._jax.random.fold_in(snapshot._key, stream))

    def fingerprint(self, snapshot: CraftaxSnapshot) -> str:
        """Return the search identity of the full state and transition stream."""
        return self.restore(snapshot).fingerprint

    def snapshot_record(self, snapshot: CraftaxSnapshot) -> dict[str, Any]:
        """Describe a replay checkpoint without disclosing its native state."""
        snapshot = self.restore(snapshot)
        return {
            "fingerprint": snapshot.fingerprint, "seed": snapshot.seed, "timestep": int(snapshot._state.timestep),
            "terminated": snapshot.terminated, "truncated": snapshot.truncated,
        }

    def inventory(self, snapshot: CraftaxSnapshot) -> dict[str, int]:
        """Read authoritative scalar inventory and tool tiers for predicate checks."""
        native = self.restore(snapshot)._state.inventory
        return {name: int(getattr(native, name)) for name in _INVENTORY_FIELDS}

    def loop_key(self, snapshot: CraftaxSnapshot) -> Hashable:
        """Identify repeated positions and inventories independently of the clock and RNG."""
        state = self.restore(snapshot)._state
        return (
            int(state.player_level), tuple(map(int, state.player_position)), int(state.player_direction),
            tuple(sorted(self.inventory(snapshot).items())),
        )

    def observe(self, snapshot: CraftaxSnapshot) -> dict[str, Any]:
        """Return model-visible data only, including native darkness and view crop."""
        snapshot = self.restore(snapshot)
        native = snapshot._state
        flat = np.asarray(snapshot._symbolic)
        features = flat[: int(np.prod(self._map_shape))].reshape(self._map_shape)
        block_count = len(self._block)
        item_count = len(self._item)
        center = (self._size[0] // 2, self._size[1] // 2)
        tiles = [[
            self._block(int(cell[:block_count].argmax())).name if cell[-1] else "UNKNOWN"
            for cell in row
        ] for row in features]
        visible_items = np.argwhere(features[:, :, block_count + 1: block_count + item_count])
        items = [[int(row), int(col), self._item(int(item) + 1).name] for row, col, item in visible_items]
        entities = [[int(row), int(col), int(channel)] for row, col, channel in np.argwhere(
            features[:, :, block_count + item_count: -1],
        )]
        return {
            "version": OBSERVATION_VERSION,
            "tiles": tiles,
            "neighbors": {direction: tiles[center[0] + row][center[1] + col] for direction, (row, col) in _DIRECTIONS.items()},
            "items": items, "entities": entities, "inventory": self.inventory(snapshot),
            "facing": self._actions(int(native.player_direction)).name,
            "health": float(native.player_health), "food": int(native.player_food), "drink": int(native.player_drink),
            "energy": int(native.player_energy), "mana": int(native.player_mana),
            "sleeping": bool(native.is_sleeping), "resting": bool(native.is_resting),
        }

    def stage(
        self, snapshot: CraftaxSnapshot, *, inventory: Mapping[str, int] | None = None,
        tiles: Mapping[str, str] | None = None, facing: str | None = None,
    ) -> CraftaxSnapshot:
        """Prepare an explicitly staged starting scenario on native state."""
        snapshot = self.restore(snapshot)
        state = snapshot._state
        if int(state.timestep) or snapshot.terminated or snapshot.truncated:
            raise ValueError("Craftax scenarios can only stage an initial state")
        overrides = dict(inventory or {})
        for name, value in overrides.items():
            maximum = 4 if name in {"pickaxe", "sword"} else 1 if name == "bow" else 99
            if name not in _INVENTORY_FIELDS or not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
                raise ValueError(f"Invalid staged Craftax inventory field: {name!r}")
        if facing is not None and facing not in _DIRECTIONS:
            raise ValueError("Staged facing must be a direction")
        level = int(state.player_level)
        row, col = map(int, state.player_position)
        world, items = state.map, state.item_map
        with self._jax.default_device(self._cpu):
            for direction, tile in (tiles or {}).items():
                if direction not in _DIRECTIONS or tile not in self._block.__members__ or tile in {"INVALID", "OUT_OF_BOUNDS"}:
                    raise ValueError(f"Invalid staged Craftax tile: {direction!r} → {tile!r}")
                d_row, d_col = _DIRECTIONS[direction]
                cell = (level, row + d_row, col + d_col)
                world = world.at[cell].set(self._block[tile].value)
                items = items.at[cell].set(self._item.NONE.value)
            inventory_fields = {
                name: self._jax.numpy.asarray(value, dtype=getattr(state.inventory, name).dtype)
                for name, value in overrides.items()
            }
            state = state.replace(
                inventory=state.inventory.replace(**inventory_fields),
                map=world, item_map=items,
                player_direction=state.player_direction if facing is None else self._jax.numpy.asarray(
                    self.action_id(facing), dtype=state.player_direction.dtype,
                ),
            )
            symbolic = self._render(state)
        return CraftaxSnapshot(snapshot.seed, self._config, False, False, state, snapshot._key, symbolic)

    def step(self, snapshot: CraftaxSnapshot, action: str | int) -> CraftaxTransition:
        """Apply one native action and report no-op relative to the visible projection."""
        snapshot = self.restore(snapshot)
        if snapshot.terminated or snapshot.truncated:
            raise ValueError("A finished Craftax snapshot cannot advance")
        index = self.action_id(action)
        # Craftax 1.1.0 replaces the request with NOOP while sleeping or resting.
        automatic = bool(snapshot._state.is_sleeping) or bool(snapshot._state.is_resting)
        with self._jax.default_device(self._cpu):
            key, step_key = self._jax.random.split(snapshot._key)
            symbolic, state, reward, done, _ = self._env.step(step_key, snapshot._state, index, self._params)
            terminated = bool(state.player_health <= 0) or bool(self._has_beaten_boss(state, self._static))
        truncated = bool(done) and not terminated
        next_snapshot = CraftaxSnapshot(snapshot.seed, self._config, terminated, truncated, state, key, symbolic)
        observation = self.observe(next_snapshot)
        ticks = int(state.timestep) - int(snapshot._state.timestep)
        return CraftaxTransition(
            next_snapshot, observation, float(reward), terminated, truncated, observation == self.observe(snapshot),
            ticks, ticks if automatic else max(0, ticks - 1), self.action_id("NOOP") if automatic else index,
        )
