"""Budgeted planning over checked intents with separately restored domain state."""

import heapq
import math
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import StrEnum
from itertools import islice
from typing import Any, Generic

from pydantic import BaseModel

from carl.icl.api import BudgetExceeded
from carl.icl.intent_domain import TASK_GOAL, Intent, SokobanObservation, SokobanSnapshot
from carl.icl.luna_intent_rollout import ActionAttempt, RolloutStatus
from carl.icl.planning.intent_execution import (
    DomainExecutionOutcome, DomainExecutionState, DomainExecutor, IntentT, MemoryT, ObservationT, SnapshotT,
)
from carl.icl.planning.search_types import FinishReason
from carl.icl.types import JSONObject

Scale = float | None
Metrics = Mapping[str, int | float]
IntentFinishReason = FinishReason | RolloutStatus
TERMINAL_FAILURES = (RolloutStatus.EPISODE_TERMINATED, RolloutStatus.EPISODE_TRUNCATED)


class SearchMode(StrEnum):
    """Predicate control, exploration and explicit plan execution conditions."""

    DIRECT = "direct"
    ONE_STEP = "one-step"
    FULL_SEARCH = "full-search"
    PLAN_AND_ACT = "plan-and-act"


class Phase(StrEnum):
    DIRECT = "direct"
    PROBE = "probe"
    COMMIT = "commit"
    SEARCH = "search"
    PLAN = "plan"


class AttemptDisposition(StrEnum):
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"
    COMMITTED = "committed"
    COMMITTED_INCOMPLETE = "committed_incomplete"
    SCORED = "scored"
    DUPLICATE = "duplicate"
    REOPENED = "reopened"
    ACCEPTED = "accepted"


@dataclass(frozen=True)
class IntentSearchSettings:
    """One shared trial budget and bounded one-step selection settings."""

    scales: tuple[Scale, ...] = (None,)
    action_budget: int = 128
    rollout_cap: int = 16
    max_expansions: int | None = None
    wall_seconds: float = 180.0
    max_proposals: int = 1
    trials_per_proposal: int = 1
    discount: float = 0.99
    seed: int = 0

    def __post_init__(self) -> None:
        def check_integer(name: str, value: int, minimum: int) -> None:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

        check_integer("action_budget", self.action_budget, 0)
        check_integer("rollout_cap", self.rollout_cap, 1)
        check_integer("max_proposals", self.max_proposals, 1)
        check_integer("trials_per_proposal", self.trials_per_proposal, 1)
        check_integer("seed", self.seed, 0)
        if self.max_expansions is not None:
            check_integer("max_expansions", self.max_expansions, 0)
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be finite and positive")
        if not math.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        scales = tuple(self.scales)
        if not scales or len(set(scales)) != len(scales):
            raise ValueError("scales must be nonempty and distinct")
        if scales != (None,) and any(
            scale is None or isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0 for scale in scales
        ):
            raise ValueError("use positive finite scales or a single null scale")
        object.__setattr__(self, "scales", scales)


@dataclass(frozen=True)
class IntentSearchRuntime(Generic[SnapshotT, ObservationT, IntentT, MemoryT]):
    """Bind the same checked executor and inference accounting across modes."""

    executor: DomainExecutor[SnapshotT, ObservationT, IntentT, MemoryT]
    propose: Callable[[ObservationT, str, Scale], Sequence[IntentT]]
    value: Callable[[ObservationT, str], float]
    metrics: Callable[[], Metrics]
    plan: Callable[[ObservationT, str], Sequence[IntentT]] | None = None


@dataclass
class IntentNode(Generic[SnapshotT, MemoryT]):
    """A checked endpoint with its parent and executed segment."""

    node: int
    parent: int | None
    state: DomainExecutionState[SnapshotT, MemoryT]
    actions: tuple[int, ...]
    path_actions: int
    intent_reached: bool
    task_solved: bool
    value: float | None = None


@dataclass(frozen=True, order=True)
class _QueueEntry:
    negative_scale: float
    negative_value: float
    serial: int
    node: int = field(compare=False)
    scale: Scale = field(compare=False)


@dataclass(frozen=True)
class IntentExpansion:
    """One actual queue pop, including its scale and tie order."""

    node: int
    scale: Scale
    value: float
    serial: int
    queue_after_pop: int
    attempted_simulator_actions: int


@dataclass
class IntentSearchAttempt(Generic[SnapshotT, IntentT]):
    """One actual trial or an invalid proposal, with original outcome flags."""

    parent: int
    phase: str
    scale: Scale
    intent: IntentT | None
    status: str
    actions: tuple[int, ...]
    endpoint: SnapshotT
    intent_reached: bool
    task_solved: bool
    error: str | None = None
    node: int | None = None
    score: float | None = None
    disposition: str = AttemptDisposition.REJECTED.value
    rewards: tuple[float, ...] = ()


@dataclass
class IntentSearchResult(Generic[SnapshotT, IntentT, MemoryT]):
    """A replayable selected path, actual traces and a common accounting schema."""

    mode: SearchMode
    solved: bool
    finish_reason: str
    action_path: list[int]
    endpoint: SnapshotT
    task_solved: bool
    intent_reached: bool
    selected_node: int
    nodes: list[IntentNode[SnapshotT, MemoryT]]
    queue_pops: list[IntentExpansion]
    attempts: list[IntentSearchAttempt[SnapshotT, IntentT]]
    accounting: dict[str, int | float]
    error: str | None
    snapshot_record: Callable[[SnapshotT], JSONObject] = field(repr=False)
    planning_calls: list[JSONObject] = field(default_factory=list)

    def as_dict(self) -> JSONObject:
        """Return the complete result without raw model prompts or responses."""
        result = _record(self, {"endpoint", "nodes", "attempts", "queue_pops", "snapshot_record", "planning_calls"})
        result["endpoint"] = self.snapshot_record(self.endpoint)
        result["queue_pops"] = [asdict(entry) for entry in self.queue_pops]
        result["nodes"] = [{
            **_record(node, {"state"}),
            "state": {"snapshot": self.snapshot_record(node.state.snapshot),
                      "memory": [_public_value(item) for item in node.state.memory]},
        } for node in self.nodes]
        result["attempts"] = [{
            **_record(attempt, {"intent", "endpoint", "rewards"}),
            "intent": _public_value(attempt.intent), "endpoint": self.snapshot_record(attempt.endpoint),
            **({"rewards": list(attempt.rewards)} if attempt.rewards else {}),
        } for attempt in self.attempts]
        if self.planning_calls:
            result["planning_calls"] = self.planning_calls
        return result


def _record(value: Any, excluded: set[str]) -> JSONObject:
    return {item.name: getattr(value, item.name) for item in fields(value) if item.name not in excluded}


def _public_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


class _Stop(BudgetExceeded):
    def __init__(self, reason: FinishReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


def _priority(scale: Scale) -> float:
    return 0.0 if scale is None else scale


def _accepted(outcome: DomainExecutionOutcome[SnapshotT, MemoryT]) -> bool:
    """Require checked preservation and a determinate effect or task outcome."""
    return outcome.status in (RolloutStatus.ACHIEVED, RolloutStatus.TASK_SOLVED) and (
        outcome.intent_reached or outcome.task_solved
    )


def _confirmed_prefix(outcome: DomainExecutionOutcome[SnapshotT, MemoryT]) -> bool:
    """Retain applied one-step actions before an execution interruption."""
    return bool(outcome.actions) and outcome.status in (
        RolloutStatus.API_ERROR, RolloutStatus.OUTPUT_LIMIT, RolloutStatus.SIMULATOR_ERROR,
        RolloutStatus.BUDGET_EXHAUSTED, RolloutStatus.INVALID_RESPONSE,
        *TERMINAL_FAILURES,
    ) and all(
        step["preservation_holds"] is True for step in outcome.rollout.steps if step["action"] is not None
    )


class _Search(Generic[SnapshotT, ObservationT, IntentT, MemoryT]):
    def __init__(
        self, settings: IntentSearchSettings, runtime: IntentSearchRuntime[SnapshotT, ObservationT, IntentT, MemoryT], goal: str,
    ) -> None:
        self.settings = settings
        self.runtime = runtime
        self.goal = goal
        self.started = time.monotonic()
        self.initial_metrics = dict(runtime.metrics())
        self.scales = sorted(settings.scales, key=_priority, reverse=True)
        self.actions = 0
        self.no_ops = 0
        self.expansions = 0
        self.proposal_queries = 0
        self.value_evaluations = 0
        self.nodes: list[IntentNode[SnapshotT, MemoryT]] = []
        self.queue_pops: list[IntentExpansion] = []
        self.attempts: list[IntentSearchAttempt[SnapshotT, IntentT]] = []
        self.frontier: list[_QueueEntry] = []
        self.best: dict[Hashable, int] = {}
        self.planning_calls: list[JSONObject] = []
        self.serial = 0
        self.selected = 0

    @property
    def remaining(self) -> int:
        return self.settings.action_budget - self.actions

    def _guard(self) -> None:
        if time.monotonic() - self.started >= self.settings.wall_seconds:
            raise _Stop(FinishReason.WALL_BUDGET)
        if self.remaining <= 0:
            raise _Stop(FinishReason.ACTION_BUDGET)

    def _expand(self) -> None:
        self._guard()
        if self.settings.max_expansions is not None and self.expansions >= self.settings.max_expansions:
            raise _Stop(FinishReason.EXPANSION_BUDGET)
        self.expansions += 1

    def _before_action(self) -> None:
        self._guard()
        self.actions += 1

    def _score(self, state: DomainExecutionState[SnapshotT, MemoryT]) -> float:
        self._guard()
        self.value_evaluations += 1
        estimate = self.runtime.value(self.runtime.executor.observation(state), self.goal)
        if isinstance(estimate, bool) or not math.isfinite(estimate) or not 0 <= estimate <= 1:
            raise ValueError("V must return a finite discounted goal return in [0, 1]")
        return float(estimate)

    def _terminal_reason(self, state: DomainExecutionState[SnapshotT, MemoryT]) -> RolloutStatus:
        record = self.runtime.executor.snapshot_record(state.snapshot)
        return RolloutStatus.EPISODE_TRUNCATED if record.get("truncated") is True else RolloutStatus.EPISODE_TERMINATED

    def _invalid(
        self, parent: IntentNode[SnapshotT, MemoryT], scale: Scale, phase: Phase, error: ValueError,
    ) -> None:
        self.attempts.append(IntentSearchAttempt(
            parent.node, phase.value, scale, None, "invalid_proposal", (), parent.state.snapshot, False, False, str(error),
        ))

    def _proposals(self, parent: IntentNode[SnapshotT, MemoryT], scale: Scale, phase: Phase) -> list[IntentT]:
        self._guard()
        observation = self.runtime.executor.observation(parent.state)
        self.proposal_queries += 1
        try:
            proposals = self.runtime.propose(observation, self.goal, scale)
        except ValueError as exc:
            self._invalid(parent, scale, phase, exc)
            return []
        checked = []
        for proposal in islice(proposals, self.settings.max_proposals):
            try:
                intent = self.runtime.executor.validate_intent(observation, proposal)
            except ValueError as exc:
                self._invalid(parent, scale, phase, exc)
            else:
                checked.append(intent)
        return checked

    def _trial(
        self, parent: IntentNode[SnapshotT, MemoryT], intent: IntentT | None, scale: Scale, phase: Phase, *, trial: int = 0,
    ) -> tuple[DomainExecutionOutcome[SnapshotT, MemoryT], IntentSearchAttempt[SnapshotT, IntentT]]:
        self._guard()
        cap = self.remaining if intent is None else min(self.remaining, self.settings.rollout_cap)
        outcome = self.runtime.executor.execute(
            parent.state, intent, self.goal, seed=self.settings.seed + trial, cap=cap, before_action=self._before_action,
        )
        self.no_ops += sum(
            step.get("simulator_no_op", step.get("no_op")) is True for step in outcome.rollout.steps
        )
        record = IntentSearchAttempt(
            parent.node, phase.value, scale, intent, outcome.status.value, tuple(outcome.actions), outcome.state.snapshot,
            outcome.intent_reached, outcome.task_solved, outcome.rollout.error,
            rewards=outcome.rewards,
        )
        self.attempts.append(record)
        return outcome, record

    def _record_node(
        self, parent: IntentNode[SnapshotT, MemoryT], outcome: DomainExecutionOutcome[SnapshotT, MemoryT],
        record: IntentSearchAttempt[SnapshotT, IntentT],
    ) -> IntentNode[SnapshotT, MemoryT]:
        child = IntentNode(
            len(self.nodes), parent.node, outcome.state, tuple(outcome.actions), parent.path_actions + len(outcome.actions),
            outcome.intent_reached, outcome.task_solved and _accepted(outcome),
        )
        record.node = child.node
        self.nodes.append(child)
        self.selected = child.node
        return child

    def _interrupted(
        self, outcome: DomainExecutionOutcome[SnapshotT, MemoryT], record: IntentSearchAttempt[SnapshotT, IntentT],
    ) -> IntentFinishReason | None:
        if outcome.status is RolloutStatus.BUDGET_EXHAUSTED:
            record.disposition = AttemptDisposition.INTERRUPTED.value
            self._guard()
            return FinishReason.API_BUDGET
        if outcome.status in (RolloutStatus.API_ERROR, RolloutStatus.OUTPUT_LIMIT):
            record.disposition = AttemptDisposition.INTERRUPTED.value
            return FinishReason.API_ERROR
        if outcome.status is RolloutStatus.SIMULATOR_ERROR:
            record.disposition = AttemptDisposition.INTERRUPTED.value
            return outcome.status
        return None

    def direct(self, root: IntentNode[SnapshotT, MemoryT]) -> tuple[IntentFinishReason, str | None]:
        outcome, record = self._trial(root, None, None, Phase.DIRECT)
        if outcome.actions or _accepted(outcome):
            self._record_node(root, outcome, record)
            record.disposition = AttemptDisposition.COMMITTED.value
        if _accepted(outcome):
            return FinishReason.SOLVED, None
        return self._interrupted(outcome, record) or (
            FinishReason.ACTION_BUDGET if self.remaining == 0 else outcome.status
        ), record.error

    def _trial_score(
        self, outcome: DomainExecutionOutcome[SnapshotT, MemoryT], record: IntentSearchAttempt[SnapshotT, IntentT],
    ) -> float:
        exponent = self.settings.discount ** len(outcome.actions)
        # The reward is one at primitive time τ if g is attained; terminal V is zero.
        if outcome.task_solved:
            score = exponent
        elif self.runtime.executor.terminal(outcome.state):
            score = 0.0
        else:
            score = exponent * self._score(outcome.state)
        record.disposition = AttemptDisposition.SCORED.value
        record.score = score
        return score

    def one_step(self, parent: IntentNode[SnapshotT, MemoryT]) -> tuple[IntentFinishReason, str | None]:
        while True:
            self._expand()
            choice: tuple[Scale, IntentT] | None = None
            best_score = -math.inf
            for scale in self.scales:
                for candidate in self._proposals(parent, scale, Phase.PROBE):
                    trials = []
                    for trial in range(self.settings.trials_per_proposal):
                        outcome, record = self._trial(parent, candidate, scale, Phase.PROBE, trial=trial)
                        if reason := self._interrupted(outcome, record):
                            return reason, record.error
                        trials.append((outcome, record))
                    if any(not _accepted(outcome) and outcome.status not in (RolloutStatus.ACTION_LIMIT, *TERMINAL_FAILURES)
                           for outcome, _ in trials):
                        continue
                    score = math.fsum(self._trial_score(outcome, record) for outcome, record in trials) / len(trials)
                    if score > best_score:
                        choice, best_score = (scale, candidate), score
            self._guard()
            if choice is None:
                return FinishReason.NO_PROPOSALS, None
            scale, candidate = choice
            outcome, record = self._trial(parent, candidate, scale, Phase.COMMIT, trial=self.settings.trials_per_proposal)
            if not _accepted(outcome) and outcome.status is not RolloutStatus.ACTION_LIMIT:
                if _confirmed_prefix(outcome):
                    self._record_node(parent, outcome, record)
                    record.disposition = AttemptDisposition.COMMITTED_INCOMPLETE.value
                reason = self._interrupted(outcome, record)
                if outcome.status in TERMINAL_FAILURES:
                    return outcome.status, record.error
                return reason or (FinishReason.ACTION_BUDGET if self.remaining == 0 else FinishReason.INTENT_FAILED), record.error
            disposition = AttemptDisposition.COMMITTED if _accepted(outcome) else AttemptDisposition.COMMITTED_INCOMPLETE
            record.disposition = disposition.value
            parent = self._record_node(parent, outcome, record)
            if parent.task_solved:
                return FinishReason.SOLVED, None
            if self.runtime.executor.terminal(parent.state):
                return self._terminal_reason(parent.state), None

    def _enqueue(self, node: IntentNode[SnapshotT, MemoryT]) -> None:
        node.value = self._score(node.state)
        self.best[self.runtime.executor.state_key(node.state)] = node.node
        for scale in self.scales:
            heapq.heappush(self.frontier, _QueueEntry(-_priority(scale), -node.value, self.serial, node.node, scale))
            self.serial += 1

    def _admit(
        self, parent: IntentNode[SnapshotT, MemoryT], outcome: DomainExecutionOutcome[SnapshotT, MemoryT],
        record: IntentSearchAttempt[SnapshotT, IntentT],
    ) -> bool:
        if not _accepted(outcome):
            return False
        key = self.runtime.executor.state_key(outcome.state)
        prior = self.best.get(key)
        if prior is not None and self.nodes[prior].path_actions <= parent.path_actions + len(outcome.actions):
            record.disposition = AttemptDisposition.DUPLICATE.value
            return False
        child = self._record_node(parent, outcome, record)
        disposition = AttemptDisposition.REOPENED if prior is not None else AttemptDisposition.ACCEPTED
        record.disposition = disposition.value
        if child.task_solved:
            return True
        if not self.runtime.executor.terminal(child.state):
            self._enqueue(child)
        return False

    def full_search(self, root: IntentNode[SnapshotT, MemoryT]) -> tuple[IntentFinishReason, str | None]:
        if self.settings.max_expansions == 0:
            self._guard()
            return FinishReason.EXPANSION_BUDGET, None
        self._enqueue(root)
        while self.frontier:
            entry = heapq.heappop(self.frontier)
            parent = self.nodes[entry.node]
            if self.best[self.runtime.executor.state_key(parent.state)] != parent.node:
                continue
            self._expand()
            self.queue_pops.append(IntentExpansion(
                parent.node, entry.scale, -entry.negative_value, entry.serial, len(self.frontier), self.actions,
            ))
            for candidate in self._proposals(parent, entry.scale, Phase.SEARCH):
                for trial in range(self.settings.trials_per_proposal):
                    outcome, record = self._trial(parent, candidate, entry.scale, Phase.SEARCH, trial=trial)
                    if reason := self._interrupted(outcome, record):
                        return reason, record.error
                    if self._admit(parent, outcome, record):
                        return FinishReason.SOLVED, None
            self._guard()
        return FinishReason.FRONTIER_EXHAUSTED, None

    def plan_and_act(self, parent: IntentNode[SnapshotT, MemoryT]) -> tuple[IntentFinishReason, str | None]:
        """Request a high-level plan, execute it directly and replan after a safe cutoff."""
        plan = self.runtime.plan
        if plan is None:
            raise ValueError("plan-and-act requires an explicit planning role")
        while True:
            self._expand()
            row: JSONObject = {"parent": parent.node, "replan": bool(self.planning_calls), "intents": []}
            self.planning_calls.append(row)
            candidates = list(islice(plan(self.runtime.executor.observation(parent.state), self.goal), self.remaining))
            row["intents"] = [_public_value(candidate) for candidate in candidates]
            start = self.actions
            for candidate in candidates:
                try:
                    candidate = self.runtime.executor.validate_intent(self.runtime.executor.observation(parent.state), candidate)
                except ValueError as exc:
                    self._invalid(parent, None, Phase.PLAN, exc)
                    break
                outcome, record = self._trial(parent, candidate, None, Phase.PLAN)
                accepted = _accepted(outcome)
                if accepted or (outcome.actions and outcome.status is RolloutStatus.ACTION_LIMIT) or _confirmed_prefix(outcome):
                    parent = self._record_node(parent, outcome, record)
                    disposition = AttemptDisposition.COMMITTED if accepted else AttemptDisposition.COMMITTED_INCOMPLETE
                    record.disposition = disposition.value
                if parent.task_solved:
                    return FinishReason.SOLVED, None
                if reason := self._interrupted(outcome, record):
                    return reason, record.error
                if self.runtime.executor.terminal(parent.state):
                    return self._terminal_reason(parent.state), None
                if outcome.status not in (RolloutStatus.ACHIEVED, RolloutStatus.ACTION_LIMIT):
                    return FinishReason.INTENT_FAILED, record.error
                if not accepted:
                    break
            if self.actions == start:
                return FinishReason.NO_PROPOSALS, None

    def _path(self) -> list[int]:
        segment_paths = []
        current = self.nodes[self.selected]
        while current.parent is not None:
            segment_paths.append(current.actions)
            current = self.nodes[current.parent]
        return [action for segment in reversed(segment_paths) for action in segment]

    def _accounting(self, path: list[int]) -> dict[str, int | float]:
        current = self.runtime.metrics()
        metrics = {key: value - self.initial_metrics.get(key, 0) for key, value in current.items()}
        for key in (
            "requests", "logical_requests", "physical_requests", "response_requests", "chat_completion_requests",
            "embedding_requests", "cache_hits", "input_tokens", "output_tokens", "cached_input_tokens",
            "cache_write_tokens", "logical_input_tokens", "logical_output_tokens", "logical_token_byte_fallback",
            "api_seconds", "policy_calls", "generator_calls", "value_calls", "local_requests", "local_policy_calls",
            "local_generator_calls", "local_value_calls", "local_input_tokens", "local_output_tokens",
            "fitted_value_calls",
        ):
            metrics.setdefault(key, 0)
        if "logical_token_byte_fallback" in current:
            metrics["logical_token_byte_fallback"] = current["logical_token_byte_fallback"]
        return {
            **metrics,
            "attempted_simulator_actions": self.actions,
            "attempted_no_op_actions": self.no_ops,
            "executed_plan_actions": len(path),
            "expansions": self.expansions,
            "queue_expansions": len(self.queue_pops),
            "proposal_queries": self.proposal_queries,
            "plan_queries": len(self.planning_calls),
            "replans": max(0, len(self.planning_calls) - 1),
            "value_evaluations": self.value_evaluations,
            "logical_api_calls": metrics["logical_requests"],
            "logical_model_calls": metrics["logical_requests"] + metrics["local_requests"] + metrics["fitted_value_calls"],
            "logical_policy_calls": metrics["policy_calls"] + metrics["local_policy_calls"],
            "logical_generator_calls": metrics["generator_calls"] + metrics["local_generator_calls"],
            "logical_value_calls": metrics["value_calls"] + metrics["local_value_calls"] + metrics["fitted_value_calls"],
            "wall_seconds": time.monotonic() - self.started,
        }

    def run(
        self, mode: SearchMode, initial: DomainExecutionState[SnapshotT, MemoryT],
    ) -> IntentSearchResult[SnapshotT, IntentT, MemoryT]:
        root = IntentNode(0, None, initial, (), 0, False, False)
        self.nodes.append(root)
        reason: IntentFinishReason
        try:
            root.task_solved = self.runtime.executor.task_solved(initial)
            operation = {SearchMode.DIRECT: self.direct, SearchMode.ONE_STEP: self.one_step,
                         SearchMode.FULL_SEARCH: self.full_search, SearchMode.PLAN_AND_ACT: self.plan_and_act}
            if root.task_solved:
                reason, error = FinishReason.SOLVED, None
            elif self.runtime.executor.terminal(initial):
                reason, error = self._terminal_reason(initial), None
            else:
                reason, error = operation[mode](root)
        except _Stop as exc:
            reason, error = exc.reason, None
        except BudgetExceeded as exc:
            reason, error = FinishReason.API_BUDGET, str(exc)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            reason, error = FinishReason.API_ERROR, f"{type(exc).__name__}: {exc}"
        selected = self.nodes[self.selected]
        path = self._path()
        return IntentSearchResult(
            mode, selected.task_solved, reason.value, path, selected.state.snapshot, selected.task_solved,
            selected.intent_reached, selected.node, self.nodes, self.queue_pops, self.attempts, self._accounting(path), error,
            self.runtime.executor.snapshot_record, self.planning_calls,
        )


def search_intents(
    initial: SokobanSnapshot, task_goal: str, *, mode: SearchMode | str, settings: IntentSearchSettings,
    runtime: IntentSearchRuntime[SokobanSnapshot, SokobanObservation, Intent, ActionAttempt],
    initial_memory: Sequence[ActionAttempt] = (),
) -> IntentSearchResult[SokobanSnapshot, Intent, ActionAttempt]:
    """Plan from a fresh initial snapshot using one shared budget per run."""
    if task_goal != TASK_GOAL:
        raise ValueError("The Sokoban planner supports the all-boxes-on-goals task")
    return search_domain_intents(initial, task_goal, mode=mode, settings=settings, runtime=runtime, initial_memory=initial_memory)


def search_domain_intents(
    initial: SnapshotT, task_goal: str, *, mode: SearchMode | str, settings: IntentSearchSettings,
    runtime: IntentSearchRuntime[SnapshotT, ObservationT, IntentT, MemoryT], initial_memory: Sequence[MemoryT] = (),
) -> IntentSearchResult[SnapshotT, IntentT, MemoryT]:
    """Run the shared search against a domain's typed snapshot and checks."""
    if not isinstance(task_goal, str) or not task_goal.strip():
        raise ValueError("Task goal must be nonempty")
    initial_state = runtime.executor.initial_state(initial, initial_memory)
    return _Search(settings, runtime, task_goal).run(SearchMode(mode), initial_state)
