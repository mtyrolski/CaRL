"""Execute one assigned intent with native checkpoints and shared budgets."""

import json
import time
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from typing import TypeAlias

from carl.icl.api import BudgetExceeded
from carl.icl.intent_domain import annotation_digest, annotation_json
from carl.icl.intent_domain import decode_intent as sokoban_intent
from carl.icl.intent_domain import render_instruction as sokoban_instruction
from carl.icl.planning.craftax_tasks import CraftaxIntent, conjunction
from carl.icl.planning.craftax_tasks import render_instruction as craftax_instruction
from carl.icl.planning.local_direct import SYSTEM, CraftaxDirect, SokobanDirect, parse_batch
from carl.icl.types import JSONObject


ActiveDomain: TypeAlias = SokobanDirect | CraftaxDirect
Decide: TypeAlias = Callable[[list[dict[str, str]]], JSONObject]
INTENT_CAP = 12
HISTORY_CAP = 6
SYSTEM_PROMPT = SYSTEM + (
    " Work on the assigned intent until the environment stops it. Your optional intent text is a note. "
    "If your last move was blocked, select another legal action that advances the assigned effects."
)


def digest(value: object) -> str:
    return annotation_digest(annotation_json(value).encode())


@dataclass
class IntentBudget:
    """Shared session limits that also charge unsuccessful speculative work."""

    max_actions: int
    max_calls: int
    max_ticks: int
    wall_seconds: float
    actions: int = 0
    calls: int = 0
    ticks: int = 0
    started: float = field(default_factory=time.monotonic, init=False)

    def remaining(self) -> JSONObject:
        return {"actions": self.max_actions - self.actions, "calls": self.max_calls - self.calls,
                "ticks": self.max_ticks - self.ticks}

    def cutoff(self) -> str | None:
        if time.monotonic() - self.started >= self.wall_seconds:
            return "wall"
        for key in ("actions", "ticks", "calls"):
            if self.remaining()[key] <= 0:
                return key
        return None


def _assign(domain: ActiveDomain, requested_h: JSONObject) -> tuple[JSONObject, str]:
    if isinstance(domain, SokobanDirect):
        domain.intent = sokoban_intent(requested_h)
        return dict(domain.intent), sokoban_instruction(domain.intent)
    domain.intent = CraftaxIntent.model_validate(requested_h)
    return domain.intent.model_dump(mode="json"), craftax_instruction(domain.intent)


def _goal(domain: ActiveDomain, task: CraftaxIntent | None) -> tuple[bool, bool]:
    if isinstance(domain, SokobanDirect):
        return bool(domain.env.is_solved(domain.snapshot.restore())), False
    snapshot = domain.snapshot
    facts = domain.adapter.inventory(snapshot)
    solved = task is not None and conjunction((*task.effects, *task.preserve), facts) is True
    return solved, snapshot.terminated or snapshot.truncated


def _state(domain: ActiveDomain, history: list[JSONObject], note: str | None, feedback: str | None) -> JSONObject:
    memory = {"history": history[-HISTORY_CAP:], "note": note, "feedback": feedback}
    if isinstance(domain, SokobanDirect):
        rows = list(domain.snapshot.rows)
        return {"state_id": digest(rows), "memory_id": digest(memory), "memory": memory, "observation": rows}
    snapshot = domain.snapshot
    return {
        "state_id": snapshot.fingerprint, "memory_id": digest(memory), "memory": memory,
        "observation": domain.adapter.observe(snapshot), "native": domain.adapter.snapshot_record(snapshot),
    }


def _stop(hits: bool | None, preserves: bool | None, solved: bool, ended: bool, *, observed: bool) -> str | None:
    if preserves is False:
        return "preservation_failed"
    if preserves is None or not observed:
        return "preservation_unknown"
    if hits is True:
        return "intent_reached"
    if solved:
        return "task_solved"
    return "episode_ended" if ended else None


@dataclass
class IntentSession:
    """Own the live state and bounded policy memory between assigned intents."""

    domain: ActiveDomain
    public_task: str
    budget: IntentBudget
    craftax_task: CraftaxIntent | None = None
    history: list[JSONObject] = field(default_factory=list)
    note: str | None = None
    feedback: str | None = None

    def attempt(
        self, attempt_id: str, requested_h: JSONObject, decide: Decide, *,
        speculative: bool = False, cap: int = INTENT_CAP,
    ) -> JSONObject:
        """Return the evidence endpoint and the actual continuation of the session."""
        if cap <= 0:
            raise ValueError("An intent cap must be positive")
        domain = copy(self.domain) if speculative else self.domain
        h, instruction = _assign(domain, requested_h)
        history, note, feedback = list(self.history), self.note, self.feedback
        initial = _state(domain, history, note, feedback)
        states = [initial]
        steps: list[JSONObject] = []
        decisions: list[JSONObject] = []
        solved, ended = _goal(domain, self.craftax_task)
        hits, preserves = domain.check()
        status = _stop(hits, preserves, solved, ended, observed=True)
        initial_check = {"effects": hits, "preservation": preserves, "task_solved": solved, "episode_ended": ended}
        cutoff = None
        primitives = 0
        budget_before = self.budget.remaining()
        started = time.monotonic()

        while status is None:
            cutoff = self.budget.cutoff()
            if cutoff is not None:
                status = "budget_cutoff"
                break
            query = {
                "observation": domain.observe(), "public_task": self.public_task,
                "assigned_intent": h, "instruction": instruction, "action_names": domain.action_names,
                "history": history[-HISTORY_CAP:], "previous_intent": note, "feedback": feedback,
            }
            messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n" + domain.rules},
                        {"role": "user", "content": json.dumps(query, separators=(",", ":"), ensure_ascii=False)}]
            self.budget.calls += 1
            decision: JSONObject = {"query": query, "state_id": states[-1]["state_id"], "executed": []}
            decisions.append(decision)
            try:
                decision.update(decide(messages))
            except BudgetExceeded as error:
                status, cutoff = "budget_cutoff", "model_context"
                decision["error"] = str(error)
                break
            except (OSError, RuntimeError, ValueError) as error:
                status = "controller_error"
                decision["error"] = str(error)
                break
            try:
                actions, note = parse_batch(decision["response"], domain.action_names)
            except (KeyError, TypeError, ValueError) as error:
                feedback = "The previous batch was invalid. Return one to four allowed actions."
                decision["error"] = str(error)
                continue

            feedback = None
            decision["requested"] = actions
            decision["discarded_suffix"] = []
            for index, action in enumerate(actions):
                cutoff = self.budget.cutoff()
                # The call reservation includes its admitted batch.
                if cutoff is not None and cutoff != "calls":
                    status = "budget_cutoff"
                    decision["discarded_suffix"] = actions[index:]
                    break
                step = domain.step(action)
                primitives += int(step.executed)
                self.budget.actions += int(step.executed)
                self.budget.ticks += step.native_ticks
                hits, preserves = domain.check()
                solved, ended = _goal(domain, self.craftax_task)
                entry = {"requested": action, "applied": step.native_action, "executed": step.executed, "no_op": step.no_op}
                history.append(entry)
                history = history[-HISTORY_CAP:]
                state = _state(domain, history, note, feedback)
                states.append(state)
                decision["executed"].append(len(steps))
                steps.append({
                    **entry, "native_ticks": step.native_ticks, "automatic_ticks": step.automatic_ticks,
                    "effects": hits, "preservation": preserves, "task_solved": solved, "episode_ended": ended,
                    "executed_primitives": primitives, "state_id": state["state_id"], "memory_id": state["memory_id"],
                })
                status = _stop(hits, preserves, solved, ended, observed=step.native_ticks == 1)
                if status is None and primitives == cap:
                    status = "intent_timeout"
                if status is not None or step.progress_changed or step.no_op or step.automatic_ticks:
                    decision["interrupt"] = status or "observation_changed"
                    decision["discarded_suffix"] = actions[index + 1:]
                    break

        endpoint = _state(domain, history, note, feedback)
        if not speculative:
            self.history, self.note, self.feedback = history, note, feedback
        return {
            "attempt_id": attempt_id, "requested_h": h, "instruction": instruction, "status": status,
            "budget_reason": cutoff if status == "budget_cutoff" else None, "intent_cap": cap,
            "mode": "speculative" if speculative else "live", "initial_check": initial_check,
            "first_hit": primitives if status == "intent_reached" else None,
            "task_solved": solved, "episode_ended": ended, "executed_primitives": primitives,
            "budget_before": budget_before, "budget_after": self.budget.remaining(),
            "states": states, "steps": steps, "decisions": decisions,
            "continuation": initial if speculative else endpoint,
            "elapsed_seconds": time.monotonic() - started,
        }
