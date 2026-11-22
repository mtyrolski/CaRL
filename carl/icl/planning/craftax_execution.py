"""Execute and check Craftax intents with private native snapshots."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from carl.environment.craftax import CraftaxAdapter, CraftaxSnapshot
from carl.icl.api import BudgetExceeded, RequestFailed
from carl.icl.luna_intent_rollout import IntentRollout, RolloutStatus
from carl.icl.planning.craftax_tasks import CraftaxIntent, conjunction, intent_is_admissible, render_instruction
from carl.icl.planning.intent_execution import DomainExecutionOutcome, DomainExecutionState
from carl.icl.types import JSONObject


@dataclass(frozen=True)
class CraftaxActionAttempt:
    """One observed decision and its outcome for the next controller call."""

    action: int
    no_op: bool


CraftaxState = DomainExecutionState[CraftaxSnapshot, CraftaxActionAttempt]
CraftaxOutcome = DomainExecutionOutcome[CraftaxSnapshot, CraftaxActionAttempt]


class Controller(Protocol):
    """A primitive decision from the visible input and a deterministic seed."""

    def __call__(
        self, observation: JSONObject, instruction: str, history: tuple[CraftaxActionAttempt, ...],
        intent: CraftaxIntent | None, *, seed: int,
    ) -> int: ...


@dataclass(frozen=True)
class CraftaxIntentExecutor:
    """Restore each trial, observe it and check predicates against authoritative facts."""

    adapter: CraftaxAdapter
    goal: CraftaxIntent
    controller: Controller

    def initial_state(self, snapshot: CraftaxSnapshot, memory: Sequence[CraftaxActionAttempt]) -> CraftaxState:
        """Validate primitive history once at the planning boundary."""
        for item in memory:
            if not isinstance(item, CraftaxActionAttempt) or not isinstance(item.no_op, bool):
                raise ValueError("Craftax memory requires typed primitive action attempts")
            self.adapter.action_id(item.action)
        return DomainExecutionState(self.adapter.restore(snapshot), tuple(memory))

    def observation(self, state: CraftaxState) -> JSONObject:
        """Extract the symbolic view separately from the simulator state."""
        return self.adapter.observe(state.snapshot)

    def task_solved(self, state: CraftaxState) -> bool:
        """Check the task goal on authoritative inventory."""
        inventory = self.adapter.inventory(state.snapshot)
        return conjunction(self.goal.effects, inventory) is True and conjunction(self.goal.preserve, inventory) is True

    def terminal(self, state: CraftaxState) -> bool:
        """Observe native termination and time limits."""
        return state.snapshot.terminated or state.snapshot.truncated

    def validate_intent(self, observation: JSONObject, intent: CraftaxIntent) -> CraftaxIntent:
        """Check admissibility using the current model-visible inventory."""
        if not isinstance(intent, CraftaxIntent) or not intent_is_admissible(intent, observation["inventory"]):
            raise ValueError("Craftax intent needs verified invariants and an unmet effect conjunction")
        return intent

    def state_key(self, state: CraftaxState) -> tuple[str, tuple[CraftaxActionAttempt, ...]]:
        """Include the transition RNG and the complete controller-visible history."""
        return self.adapter.fingerprint(state.snapshot), state.memory

    def snapshot_record(self, snapshot: CraftaxSnapshot) -> JSONObject:
        """Publish a replay identity rather than exposing native state arrays."""
        return self.adapter.snapshot_record(snapshot)

    def _check(self, snapshot: CraftaxSnapshot, intent: CraftaxIntent | None) -> tuple[bool, bool | None, bool | None]:
        facts = self.adapter.inventory(snapshot)
        solved = conjunction(self.goal.effects, facts) is True
        effects = False if intent is None else conjunction(intent.effects, facts)
        constraints = self.goal.preserve if intent is None else self.goal.preserve + intent.preserve
        return solved, effects, conjunction(constraints, facts)

    def execute(
        self, state: CraftaxState, intent: CraftaxIntent | None, goal: str, *, seed: int, cap: int,
        before_action: Callable[[], None] | None = None,
    ) -> CraftaxOutcome:
        """Return the last confirmed branch with action, check and reward traces."""
        snapshot = self.adapter.restore(state.snapshot)
        history = list(state.memory)
        actions: list[int] = []
        rewards: list[float] = []
        steps: list[JSONObject] = []
        calls = 0
        error = None
        solved, effects, preserve = self._check(snapshot, intent)
        status = _termination(snapshot, solved, effects, preserve) or RolloutStatus.ACTION_LIMIT
        instruction = goal if intent is None else render_instruction(intent)
        for _ in range(cap):
            if status != RolloutStatus.ACTION_LIMIT:
                break
            try:
                calls += 1
                action = self.adapter.action_id(self.controller(
                    self.adapter.observe(snapshot), instruction, tuple(history), intent, seed=seed,
                ))
                if before_action is not None:
                    before_action()
            except BudgetExceeded as exc:
                status, error = RolloutStatus.BUDGET_EXHAUSTED, str(exc)
                break
            except RequestFailed as exc:
                status, error = RolloutStatus.API_ERROR, str(exc)
                break
            except ValueError as exc:
                status, error = RolloutStatus.INVALID_RESPONSE, str(exc)
                break
            try:
                transition = self.adapter.step(snapshot, action)
            except (ValueError, RuntimeError) as exc:
                status, error = RolloutStatus.SIMULATOR_ERROR, str(exc)
                break
            snapshot = transition.snapshot
            history.append(CraftaxActionAttempt(action, transition.no_op))
            actions.append(action)
            rewards.append(transition.reward)
            solved, effects, preserve = self._check(snapshot, intent)
            steps.append({"action": action, "no_op": transition.no_op, "preservation_holds": preserve,
                          "intent_holds": effects, "task_solved": solved, "reward": transition.reward})
            status = _termination(snapshot, solved, effects, preserve) or RolloutStatus.ACTION_LIMIT
        return DomainExecutionOutcome(
            DomainExecutionState(snapshot, tuple(history)),
            IntentRollout(status, status == RolloutStatus.ACHIEVED, solved, actions, calls, [], steps, error),
            tuple(rewards),
        )


def _termination(
    snapshot: CraftaxSnapshot, solved: bool, effects: bool | None, preserve: bool | None,
) -> RolloutStatus | None:
    if preserve is False:
        return RolloutStatus.PRESERVATION_VIOLATED
    if preserve is None:
        return RolloutStatus.UNKNOWN
    if effects is True:
        return RolloutStatus.ACHIEVED
    if solved:
        return RolloutStatus.TASK_SOLVED if effects is False else RolloutStatus.UNKNOWN
    if snapshot.terminated or snapshot.truncated:
        return RolloutStatus.EPISODE_TERMINATED if snapshot.terminated else RolloutStatus.EPISODE_TRUNCATED
    return None
