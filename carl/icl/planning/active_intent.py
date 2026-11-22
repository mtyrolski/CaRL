"""Execute one assigned intent with native checkpoints and shared budgets."""

import json
import time
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, TypeAlias

from carl.icl.api import BudgetExceeded
from carl.icl.intent_domain import annotation_digest, annotation_json
from carl.icl.planning.local_direct import SYSTEM, DirectDomain, TaskStatus, parse_batch
from carl.icl.types import JSONObject, JSONValue


Decide: TypeAlias = Callable[[list[dict[str, str]]], JSONObject]
INTENT_CAP = 12
HISTORY_CAP = 6
SYSTEM_PROMPT = SYSTEM + (
    " Work on the assigned intent until the environment stops it. Your optional intent text is a note. "
    "If your last move was blocked, select another legal action that advances the assigned effects."
)


class IntentStatus(StrEnum):
    """Checkpoint and terminal statuses for an assigned intent."""

    RUNNING = "running"
    REACHED = "intent_reached"
    TIMEOUT = "intent_timeout"
    PRESERVATION_FAILED = "preservation_failed"
    PRESERVATION_UNKNOWN = "preservation_unknown"
    TASK_SOLVED = "task_solved"
    EPISODE_ENDED = "episode_ended"
    BUDGET_CUTOFF = "budget_cutoff"
    CONTROLLER_ERROR = "controller_error"
    INVALID_RESPONSE = "invalid_response"


class ActiveDomain(DirectDomain, Protocol):
    """An assigned-intent adapter that can be copied to run a speculative branch."""

    def assign_intent(self, requested: JSONObject) -> tuple[JSONObject, str]: ...
    def task_status(self) -> TaskStatus: ...
    def checkpoint(self) -> tuple[str, JSONObject]: ...


def digest(value: JSONValue) -> str:
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


def _state(domain: ActiveDomain, history: list[JSONObject], note: str | None, feedback: str | None) -> JSONObject:
    memory = {"history": history[-HISTORY_CAP:], "note": note, "feedback": feedback}
    state_id, checkpoint = domain.checkpoint()
    return {"state_id": state_id, "memory_id": digest(memory), "memory": memory, **checkpoint}


def _stop(hits: bool | None, preserves: bool | None, solved: bool, ended: bool, *, observed: bool) -> IntentStatus | None:
    if preserves is False:
        return IntentStatus.PRESERVATION_FAILED
    if preserves is None or not observed:
        return IntentStatus.PRESERVATION_UNKNOWN
    if hits is True:
        return IntentStatus.REACHED
    if solved:
        return IntentStatus.TASK_SOLVED
    return IntentStatus.EPISODE_ENDED if ended else None


@dataclass
class IntentSession:
    """Own the live state and bounded policy memory between assigned intents."""

    domain: ActiveDomain
    public_task: str
    budget: IntentBudget
    history: list[JSONObject] = field(default_factory=list)
    note: str | None = None
    feedback: str | None = None

    def attempt(
        self, attempt_id: str, requested_h: JSONObject, decide: Decide, *,
        speculative: bool = False, cap: int = INTENT_CAP,
        stop_on_invalid: bool = False, checkpoint: Callable[[JSONObject], None] | None = None,
    ) -> JSONObject:
        """Return the evidence endpoint and the actual continuation of the session."""
        if cap <= 0:
            raise ValueError("An intent cap must be positive")
        domain = copy(self.domain) if speculative else self.domain
        h, instruction = domain.assign_intent(requested_h)
        history, note, feedback = list(self.history), self.note, self.feedback
        initial = _state(domain, history, note, feedback)
        states = [initial]
        steps: list[JSONObject] = []
        decisions: list[JSONObject] = []
        solved, ended = domain.task_status()
        hits, preserves = domain.check()
        status = _stop(hits, preserves, solved, ended, observed=True)
        initial_check = {"effects": hits, "preservation": preserves, "task_solved": solved, "episode_ended": ended}
        cutoff = None
        primitives = 0
        budget_before = self.budget.remaining()
        started = time.monotonic()

        def save_progress() -> None:
            if checkpoint is not None:
                checkpoint({
                    "attempt_id": attempt_id, "requested_h": h, "status": IntentStatus.RUNNING.value,
                    "executed_primitives": primitives, "initial_check": initial_check,
                    "states": states, "steps": steps, "decisions": decisions,
                    "budget_after": self.budget.remaining(), "elapsed_seconds": time.monotonic() - started,
                })

        while status is None:
            cutoff = self.budget.cutoff()
            if cutoff is not None:
                status = IntentStatus.BUDGET_CUTOFF
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
            save_progress()
            try:
                decision.update(decide(messages))
            except BudgetExceeded as error:
                status, cutoff = IntentStatus.BUDGET_CUTOFF, "model_context"
                decision["error"] = str(error)
                break
            except (OSError, RuntimeError, ValueError) as error:
                status = IntentStatus.CONTROLLER_ERROR
                decision["error"] = str(error)
                break
            try:
                actions, note = parse_batch(decision["response"], domain.action_names)
            except (KeyError, TypeError, ValueError) as error:
                feedback = "The previous batch was invalid. Return one to four allowed actions."
                decision["error"] = str(error)
                if stop_on_invalid:
                    status = IntentStatus.INVALID_RESPONSE
                    break
                continue

            feedback = None
            decision["requested"] = actions
            decision["discarded_suffix"] = []
            for index, action in enumerate(actions):
                cutoff = self.budget.cutoff()
                # The call reservation includes its admitted batch.
                if cutoff is not None and cutoff != "calls":
                    status = IntentStatus.BUDGET_CUTOFF
                    decision["discarded_suffix"] = actions[index:]
                    break
                step = domain.step(action)
                primitives += int(step.executed)
                self.budget.actions += int(step.executed)
                self.budget.ticks += step.native_ticks
                hits, preserves = domain.check()
                solved, ended = domain.task_status()
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
                    status = IntentStatus.TIMEOUT
                if status is not None or step.progress_changed or step.no_op or step.automatic_ticks:
                    decision["interrupt"] = status.value if status is not None else "observation_changed"
                    decision["discarded_suffix"] = actions[index + 1:]
                    save_progress()
                    break
                save_progress()

        endpoint = _state(domain, history, note, feedback)
        if not speculative:
            self.history, self.note, self.feedback = history, note, feedback
        return {
            "attempt_id": attempt_id, "requested_h": h, "instruction": instruction, "status": status.value,
            "budget_reason": cutoff if status is IntentStatus.BUDGET_CUTOFF else None, "intent_cap": cap,
            "mode": "speculative" if speculative else "live", "initial_check": initial_check,
            "first_hit": primitives if status is IntentStatus.REACHED else None,
            "task_solved": solved, "episode_ended": ended, "executed_primitives": primitives,
            "budget_before": budget_before, "budget_after": self.budget.remaining(),
            "states": states, "steps": steps, "decisions": decisions,
            "continuation": initial if speculative else endpoint,
            "elapsed_seconds": time.monotonic() - started,
        }
