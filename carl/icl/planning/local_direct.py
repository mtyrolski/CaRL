"""Run one frozen local controller with interruptible primitive batches."""

import json
import time
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from carl.environment.craftax import CraftaxAdapter, CraftaxSnapshot
from carl.icl.boards import to_rows
from carl.icl.config import LocalModelSettings
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_domain import Action, Intent, SokobanSnapshot, predicate_truth
from carl.icl.intents.generation.data import parse_json
from carl.icl.local_models import LocalRoleModel, bind_local_role
from carl.icl.planning.craftax_tasks import CraftaxIntent, conjunction
from carl.icl.types import JSONObject


SYSTEM = (
    'Choose actions for the current task using the observation and your last outcomes. '
    'Reply with one JSON object of the form {"actions":["UP","RIGHT"],"intent":"short optional goal"}. '
    "Use one to four action names from the supplied vocabulary. Stop the batch at a natural change in the task. "
    "The environment can interrupt a batch. Use the new observation and executed action history after an interrupt."
)
SOKOBAN_RULES = (
    "Rows increase DOWN and columns increase RIGHT. # is a wall, $ a box, * a box on a goal, . an empty goal, "
    "@ the player and + the player on a goal. Move into an adjacent box to push it one square if the far square is free."
)
CRAFTAX_RULES = (
    "UP, DOWN, LEFT and RIGHT set facing. DO harvests a nearby tree or mines stone in the facing direction. "
    "MAKE_WOOD_SWORD crafts a wooden sword with one wood while adjacent to a crafting table. "
    "MAKE_WOOD_PICKAXE crafts a wooden pickaxe with one wood at a table. Read the inventory and nearby terrain."
)


@dataclass(frozen=True)
class DirectLimits:
    """The same limits apply to both domains and both measured controllers."""

    decisions: int = 3
    primitive_actions: int = 6
    native_ticks: int = 8
    history: int = 6
    output_tokens: int = 96
    context_limit: int = 1024
    wall_seconds: float = 60.0

    def __post_init__(self) -> None:
        if min(self.decisions, self.primitive_actions, self.native_ticks, self.history,
               self.output_tokens, self.context_limit, self.wall_seconds) <= 0:
            raise ValueError("Direct limits must be positive")


@dataclass(frozen=True)
class DirectStep:
    native_action: str
    executed: bool
    native_ticks: int
    automatic_ticks: int
    no_op: bool
    progress_changed: bool
    terminal: bool


class DirectDomain(Protocol):
    """Separate native evaluation from the current public view shown to the controller."""

    rules: str
    action_names: tuple[str, ...]

    def observe(self) -> JSONObject: ...
    def check(self) -> tuple[bool | None, bool | None]: ...
    def loop_key(self) -> Hashable: ...
    def step(self, action: str) -> DirectStep: ...


def _all(values: list[bool | None]) -> bool | None:
    if False in values:
        return False
    return None if None in values else True


class SokobanDirect:
    rules = SOKOBAN_RULES
    action_names = tuple(action.name for action in Action)

    def __init__(self, snapshot: SokobanSnapshot, intent: Intent) -> None:
        self.snapshot = snapshot
        self.intent = intent
        self.env = environment(snapshot.restore())

    def observe(self) -> JSONObject:
        rows = self.snapshot.rows
        y, x = next((row, col) for row, line in enumerate(rows) for col, tile in enumerate(line) if tile in "@+")
        neighbors = {}
        for name, dy, dx in (("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1)):
            row, col = y + dy, x + dx
            neighbors[name] = rows[row][col] if 0 <= row < len(rows) and 0 <= col < len(rows[0]) else "#"
        return {"rows": list(rows), "player": {"row": y, "col": x}, "adjacent_tiles": neighbors}

    def check(self) -> tuple[bool | None, bool | None]:
        visible = self.snapshot.observation
        return (_all([predicate_truth(p, visible) for p in self.intent["effects"]]),
                _all([predicate_truth(p, visible) for p in self.intent["preserve"]]))

    def loop_key(self) -> tuple[str, ...]:
        return self.snapshot.rows

    @staticmethod
    def _boxes(rows: tuple[str, ...]) -> set[tuple[int, int]]:
        return {(y, x) for y, line in enumerate(rows) for x, tile in enumerate(line) if tile in "$*"}

    def step(self, action: str) -> DirectStep:
        before = self.snapshot.rows
        board = self.env.next_state(self.snapshot.restore(), int(Action[action]))
        self.snapshot = SokobanSnapshot(tuple(to_rows(board)))
        after = self.snapshot.rows
        return DirectStep(action, True, 1, 0, before == after, self._boxes(before) != self._boxes(after),
                          self.env.is_solved(board))


class CraftaxDirect:
    rules = CRAFTAX_RULES

    def __init__(self, adapter: CraftaxAdapter, snapshot: CraftaxSnapshot, intent: CraftaxIntent) -> None:
        self.adapter = adapter
        self.snapshot = adapter.restore(snapshot)
        self.intent = intent
        self.action_names = adapter.action_names

    def observe(self) -> JSONObject:
        visible = self.adapter.observe(self.snapshot)
        return {key: visible[key] for key in ("neighbors", "inventory", "facing", "health", "sleeping", "resting")}

    def check(self) -> tuple[bool | None, bool | None]:
        facts = self.adapter.inventory(self.snapshot)
        return conjunction(self.intent.effects, facts), conjunction(self.intent.preserve, facts)

    def loop_key(self) -> Hashable:
        state = self.snapshot._state  # pylint: disable=protected-access
        # The clock and RNG advance even when the controller returns to the same place.
        return (int(state.player_level), tuple(map(int, state.player_position)), int(state.player_direction),
                tuple(sorted(self.adapter.inventory(self.snapshot).items())))

    def step(self, action: str) -> DirectStep:
        before = self.observe()
        outcome = self.adapter.step(self.snapshot, action)
        self.snapshot = outcome.snapshot
        changed = any(before[key] != outcome.observation[key] for key in ("inventory", "health", "sleeping", "resting"))
        return DirectStep(
            self.action_names[outcome.native_action], outcome.primitive_action_executed,
            outcome.native_ticks, outcome.automatic_ticks, outcome.no_op or action == "NOOP", changed,
            outcome.terminated or outcome.truncated,
        )


def parse_batch(text: str, action_names: tuple[str, ...]) -> tuple[list[str], str | None]:
    """Parse one bounded JSON batch. Accept an outer JSON fence and named actions."""
    value = text.strip()
    if value.startswith("```json\n") and value.endswith("```"):
        value = value[len("```json\n"):-3].strip()
    payload = parse_json(value)
    if not isinstance(payload, dict) or set(payload) - {"actions", "intent"}:
        raise ValueError("Expected one JSON object with actions and optional intent")
    actions = payload.get("actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= 4:
        raise ValueError("Expected one to four actions")
    names = []
    for action in actions:
        name = action["name"] if isinstance(action, dict) and set(action) == {"name"} else action
        if not isinstance(name, str) or name.upper() not in action_names:
            raise ValueError("Action outside the supplied vocabulary")
        names.append(name.upper())
    intent = payload.get("intent")
    if intent is not None and not isinstance(intent, str):
        raise ValueError("Intent must be short text")
    return names, intent.strip()[:160] if intent else None


def load_direct_controller(path: Path) -> tuple[LocalRoleModel, JSONObject]:
    """Pin an unadapted pipeline without weakening the Sokoban adapter checks."""
    import torch

    config = json.loads(path.read_text(encoding="utf-8"))
    settings = LocalModelSettings.model_validate(config["local"])
    if settings.adapter is not None:
        raise ValueError("The cross-domain direct protocol requires base weights")
    threads = int(config["threads"])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(73)
    started = time.perf_counter()
    role = bind_local_role(settings, "policy")
    return role, {
        "model_id": config["model_id"], "requested_model_id": config.get("requested_model_id", config["model_id"]),
        "threads": torch.get_num_threads(), "seed": 73, "identity": role.identity(),
        "binding_seconds": time.perf_counter() - started,
    }


def run_direct(
    role: LocalRoleModel, domain: DirectDomain, public_task: JSONObject, limits: DirectLimits,
) -> JSONObject:
    """Record the exact model call, applied actions and interrupted suffix for each decision."""
    if domain.check() != (False, True):
        raise ValueError("Direct validation must begin with an unmet task and verified invariants")
    memory: list[JSONObject] = []
    decisions: list[JSONObject] = []
    own_intent = None
    feedback = None
    visited = {domain.loop_key()}
    executed = native_ticks = automatic_ticks = no_ops = loops = errors = requested = 0
    stop = "decision_limit"
    started = time.perf_counter()
    for _ in range(limits.decisions):
        if time.perf_counter() - started >= limits.wall_seconds:
            stop = "wall_limit"
            break
        query = {
            "observation": domain.observe(), "public_task": public_task, "action_names": domain.action_names,
            "history": memory[-limits.history:], "previous_intent": own_intent, "feedback": feedback,
        }
        messages = [{"role": "system", "content": SYSTEM + "\n" + domain.rules},
                    {"role": "user", "content": json.dumps(query, separators=(",", ":"), ensure_ascii=False)}]
        payload: JSONObject = {}
        decision: JSONObject = {"query": query, "payload": payload, "steps": []}
        decisions.append(decision)
        call_start = time.perf_counter()
        try:
            raw, inputs, outputs = role.complete_chat(
                messages, output_tokens=limits.output_tokens, context_limit=limits.context_limit, on_payload=payload.update,
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors += 1
            stop = "runtime_error"
            decision.update(error=str(error), latency_seconds=time.perf_counter() - call_start)
            break
        decision.update(response=raw, input_tokens=inputs, output_tokens=outputs, latency_seconds=time.perf_counter() - call_start)
        try:
            actions, own_intent = parse_batch(raw, domain.action_names)
        except (TypeError, ValueError) as error:
            errors += 1
            feedback = "Previous response had invalid action JSON. Use the requested format and names."
            decision.update(error=str(error), interrupt="invalid_response")
            continue
        feedback = None
        requested += len(actions)
        decision["actions"] = actions
        decision["own_intent"] = own_intent
        for index, action in enumerate(actions):
            if time.perf_counter() - started >= limits.wall_seconds:
                stop = "wall_limit"
                decision.update(interrupt=stop, discarded_suffix=actions[index:], observation_after_interrupt=domain.observe())
                break
            step = domain.step(action)
            executed += int(step.executed)
            native_ticks += step.native_ticks
            automatic_ticks += step.automatic_ticks
            no_ops += int(step.no_op and step.executed)
            key = domain.loop_key()
            repeated = key in visited and not step.no_op and step.executed
            loops += int(repeated)
            visited.add(key)
            hits, preserves = domain.check()
            memory.append({"action": action, "applied": step.native_action, "no_op": step.no_op, "executed": step.executed})
            decision["steps"].append({
                **memory[-1], "native_ticks": step.native_ticks, "automatic_ticks": step.automatic_ticks,
                "effect_holds": hits, "preservation_holds": preserves, "revisited_state": repeated,
            })
            terminal = (
                "preservation_violated" if preserves is False else "unknown" if preserves is None or step.native_ticks != 1
                else "success" if hits is True else "episode_ended" if step.terminal
                else "action_limit" if executed >= limits.primitive_actions
                else "tick_limit" if native_ticks >= limits.native_ticks else "loop_limit" if loops >= 2 else None
            )
            interrupt = terminal or ("automatic_tick" if step.automatic_ticks else "no_op" if step.no_op
                                     else "revisited_state" if repeated else "changed_task_observation" if step.progress_changed else None)
            if interrupt is not None:
                decision.update(interrupt=interrupt, discarded_suffix=actions[index + 1:],
                                observation_after_interrupt=domain.observe())
                if terminal is not None:
                    stop = terminal
                break
        else:
            decision.update(interrupt="batch_finished", discarded_suffix=[])
        if stop != "decision_limit":
            break
    return {
        "success": stop == "success", "stop": stop, "errors": errors, "loops": loops, "executed_no_ops": no_ops,
        "requested_primitives": requested, "executed_primitives": executed, "native_ticks": native_ticks,
        "automatic_ticks": automatic_ticks, "elapsed_seconds": time.perf_counter() - started,
        "input_tokens": sum(d.get("input_tokens", 0) for d in decisions),
        "output_tokens": sum(d.get("output_tokens", 0) for d in decisions),
        "decisions": decisions, "role_calls": {"policy": len(decisions), "generator": 0, "value": 0},
        "first_verified_hit_primitives": executed if stop == "success" else None,
    }
