"""Restore a Sokoban search node and execute an instruction against its simulator."""

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar

from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_domain import Intent, SokobanObservation, SokobanSnapshot, decode_intent, validate_intent_proposal
from carl.icl.luna_intent_rollout import (
    Action, ActionAttempt, IntentRollout, ObservationPolicy, RolloutStatus,
    TaskObservationPolicy, observe_state, rollout,
)
from carl.icl.types import Board, JSONObject


PolicyFactory: TypeAlias = Callable[[int], Callable[[str], str] | ObservationPolicy | TaskObservationPolicy]
SnapshotT = TypeVar("SnapshotT")
ObservationT = TypeVar("ObservationT")
IntentT = TypeVar("IntentT")
MemoryT = TypeVar("MemoryT")


@dataclass(frozen=True)
class DomainExecutionState(Generic[SnapshotT, MemoryT]):
    """The state restored by a branch, with immutable controller memory."""

    snapshot: SnapshotT
    memory: tuple[MemoryT, ...] = ()


@dataclass(frozen=True)
class ExecutionState(DomainExecutionState[SokobanSnapshot, ActionAttempt]):
    """The Sokoban snapshot and checked primitive action history."""

    def __post_init__(self) -> None:
        memory = tuple(self.memory)
        if any(not isinstance(item, ActionAttempt) or not isinstance(item.action, Action)
               or not isinstance(item.no_op, bool) for item in memory):
            raise ValueError("Controller memory requires typed action attempts")
        object.__setattr__(self, "memory", memory)


@dataclass(frozen=True)
class DomainExecutionOutcome(Generic[SnapshotT, MemoryT]):
    """The last confirmed snapshot, public rollout and native environment rewards."""

    state: DomainExecutionState[SnapshotT, MemoryT]
    rollout: IntentRollout
    rewards: tuple[float, ...] = ()

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
class ExecutionOutcome(DomainExecutionOutcome[SokobanSnapshot, ActionAttempt]):
    """An executed Sokoban branch."""


class DomainExecutor(Protocol[SnapshotT, ObservationT, IntentT, MemoryT]):
    """Keep simulator state and replay keys outside the model-visible interface."""

    def initial_state(self, snapshot: SnapshotT, memory: Sequence[MemoryT]) -> DomainExecutionState[SnapshotT, MemoryT]: ...
    def observation(self, state: DomainExecutionState[SnapshotT, MemoryT]) -> ObservationT: ...
    def task_solved(self, state: DomainExecutionState[SnapshotT, MemoryT]) -> bool: ...
    def terminal(self, state: DomainExecutionState[SnapshotT, MemoryT]) -> bool: ...
    def validate_intent(self, observation: ObservationT, intent: IntentT) -> IntentT: ...
    def state_key(self, state: DomainExecutionState[SnapshotT, MemoryT]) -> Hashable: ...
    def snapshot_record(self, snapshot: SnapshotT) -> JSONObject: ...
    def execute(
        self, state: DomainExecutionState[SnapshotT, MemoryT], intent: IntentT | None, goal: str, *,
        seed: int, cap: int, before_action: Callable[[], None] | None = None,
    ) -> DomainExecutionOutcome[SnapshotT, MemoryT]: ...


@dataclass(frozen=True)
class SokobanIntentExecutor:
    """The planner's domain adapter over the shared calibration rollout."""

    policy_factory: PolicyFactory
    observe: Callable[[Board], SokobanObservation] | None = None

    def initial_state(self, snapshot: SokobanSnapshot, memory: Sequence[ActionAttempt]) -> ExecutionState:
        """Validate the starting history once per search."""
        return ExecutionState(snapshot, tuple(memory))

    def observation(self, state: DomainExecutionState[SokobanSnapshot, ActionAttempt]) -> SokobanObservation:
        """Return the same visible observation used at every controller step."""
        return observe_state(state.snapshot.restore(), self.observe) if self.observe else state.snapshot.observation

    def task_solved(self, state: DomainExecutionState[SokobanSnapshot, ActionAttempt]) -> bool:
        """Evaluate the task goal in the simulator's complete state."""
        board = state.snapshot.restore()
        return environment(board).is_solved(board)

    def terminal(self, state: DomainExecutionState[SokobanSnapshot, ActionAttempt]) -> bool:
        """Sokoban only terminates on task completion."""
        return self.task_solved(state)

    def validate_intent(self, observation: SokobanObservation, intent: Intent) -> Intent:
        """Require a legal, unachieved intent on the visible board."""
        checked = decode_intent(intent)
        if problems := validate_intent_proposal(checked, observation):
            raise ValueError(", ".join(problems))
        return checked

    def state_key(self, state: DomainExecutionState[SokobanSnapshot, ActionAttempt]) -> Hashable:
        """An oriented board and all controller history determine future execution."""
        return state.snapshot, state.memory

    def snapshot_record(self, snapshot: SokobanSnapshot) -> JSONObject:
        """Preserve the established public board format."""
        return {"rows": snapshot.rows}

    def execute(
        self, state: DomainExecutionState[SokobanSnapshot, ActionAttempt], intent: Intent | None, goal: str, *, seed: int, cap: int,
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
