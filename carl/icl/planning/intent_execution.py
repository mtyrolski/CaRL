"""Restore a Sokoban search node and execute an instruction against its simulator."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_domain import Intent, SokobanObservation, SokobanSnapshot
from carl.icl.luna_intent_rollout import (
    Action, ActionAttempt, IntentRollout, ObservationPolicy, RolloutStatus,
    TaskObservationPolicy, observe_state, rollout,
)
from carl.icl.types import Board


PolicyFactory: TypeAlias = Callable[[int], Callable[[str], str] | ObservationPolicy | TaskObservationPolicy]


@dataclass(frozen=True)
class ExecutionState:
    """The dependencies of future execution and deduplication in the planner."""

    snapshot: SokobanSnapshot
    memory: tuple[ActionAttempt, ...] = ()

    def __post_init__(self) -> None:
        memory = tuple(self.memory)
        if any(not isinstance(item, ActionAttempt) or not isinstance(item.action, Action)
               or not isinstance(item.no_op, bool) for item in memory):
            raise ValueError("Controller memory requires typed action attempts")
        object.__setattr__(self, "memory", memory)


@dataclass(frozen=True)
class ExecutionOutcome:
    """The last confirmed full snapshot and the observation-only rollout trace."""

    state: ExecutionState
    rollout: IntentRollout

    @property
    def status(self) -> RolloutStatus:
        """The reason the shared executor stopped."""
        return self.rollout.status

    @property
    def intent_reached(self) -> bool:
        """The intent effects were reached with verified preservation."""
        return self.rollout.achieved

    @property
    def task_solved(self) -> bool:
        """The simulator reached the task goal independently of intent checks."""
        return self.rollout.task_solved

    @property
    def actions(self) -> tuple[int, ...]:
        """Actions applied to the confirmed endpoint, in replay order."""
        return tuple(self.rollout.actions)

    @property
    def error(self) -> str | None:
        """The execution failure, when one was reported."""
        return self.rollout.error


@dataclass(frozen=True)
class SokobanIntentExecutor:
    """The planner's domain adapter over the shared calibration rollout."""

    policy_factory: PolicyFactory
    observe: Callable[[Board], SokobanObservation] | None = None

    def observation(self, state: ExecutionState) -> SokobanObservation:
        """Return the same visible observation used at every controller step."""
        return observe_state(state.snapshot.restore(), self.observe) if self.observe else state.snapshot.observation

    def task_solved(self, state: ExecutionState) -> bool:
        """Evaluate the task goal in the simulator's complete state."""
        board = state.snapshot.restore()
        return environment(board).is_solved(board)

    def execute(
        self, state: ExecutionState, intent: Intent | None, goal: str, *, seed: int, cap: int,
        before_action: Callable[[], None] | None = None,
    ) -> ExecutionOutcome:
        """Attempt one instruction with fresh state, controller and parent memory."""
        board = state.snapshot.restore()
        endpoint = board
        memory = state.memory

        def record(current: Board, history: tuple[ActionAttempt, ...]) -> None:
            nonlocal endpoint, memory
            endpoint, memory = current, history

        result = rollout(
            environment(board), board, dict(intent) if intent is not None else None,
            self.policy_factory(seed), task_goal=goal, cap=cap, initial_memory=memory,
            observe=self.observe, measured=True, before_action=before_action, on_state=record,
        )
        return ExecutionOutcome(ExecutionState(SokobanSnapshot(tuple(to_rows(endpoint))), memory), result)
