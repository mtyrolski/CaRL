"""Execute direct, one-step and best-first planning over checked Sokoban intents."""

import heapq
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import islice
from typing import Literal

from carl.icl.api import BudgetExceeded
from carl.icl.intent_domain import (
    TASK_GOAL, Intent, SokobanObservation, SokobanSnapshot, decode_intent, validate_intent_proposal,
)
from carl.icl.luna_intent_rollout import ActionAttempt, RolloutStatus
from carl.icl.planning.intent_execution import ExecutionOutcome, ExecutionState, SokobanIntentExecutor
from carl.icl.types import JSONObject

Scale = float | None
Phase = Literal["direct", "probe", "commit", "search"]
Metrics = Mapping[str, int | float]


class SearchMode(StrEnum):
    """The three predicate control and planning conditions."""

    DIRECT = "direct"
    ONE_STEP = "one-step"
    FULL_SEARCH = "full-search"


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
class IntentSearchRuntime:
    """Bind the same checked executor and inference accounting across modes."""

    executor: SokobanIntentExecutor
    propose: Callable[[SokobanObservation, str, Scale], Sequence[Intent]]
    value: Callable[[SokobanObservation, str], float]
    metrics: Callable[[], Metrics]


@dataclass
class IntentNode:
    """A checked endpoint with its parent and executed segment."""

    node: int
    parent: int | None
    state: ExecutionState
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
class IntentSearchAttempt:
    """One actual trial or an invalid proposal, with original outcome flags."""

    parent: int
    phase: Phase
    scale: Scale
    intent: Intent | None
    status: str
    actions: tuple[int, ...]
    endpoint: SokobanSnapshot
    intent_reached: bool
    task_solved: bool
    error: str | None = None
    node: int | None = None
    score: float | None = None
    disposition: str = "rejected"


@dataclass
class IntentSearchResult:
    """A replayable selected path, actual traces and a common accounting schema."""

    mode: SearchMode
    solved: bool
    finish_reason: str
    action_path: list[int]
    endpoint: SokobanSnapshot
    task_solved: bool
    intent_reached: bool
    selected_node: int
    nodes: list[IntentNode]
    queue_pops: list[IntentExpansion]
    attempts: list[IntentSearchAttempt]
    accounting: dict[str, int | float]
    error: str | None

    def as_dict(self) -> JSONObject:
        """Return the complete result without raw model prompts or responses."""
        return asdict(self)


class _Stop(BudgetExceeded):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _priority(scale: Scale) -> float:
    return 0.0 if scale is None else scale


def _state_key(state: ExecutionState) -> tuple[SokobanSnapshot, tuple[ActionAttempt, ...]]:
    return state.snapshot, state.memory


def _accepted(outcome: ExecutionOutcome) -> bool:
    """Require checked preservation and a determinate effect or task outcome."""
    return outcome.status in (RolloutStatus.ACHIEVED, RolloutStatus.TASK_SOLVED) and (
        outcome.intent_reached or outcome.task_solved
    )


def _confirmed_prefix(outcome: ExecutionOutcome) -> bool:
    """Retain applied one-step actions before an execution interruption."""
    return bool(outcome.actions) and outcome.status in (
        RolloutStatus.API_ERROR, RolloutStatus.OUTPUT_LIMIT, RolloutStatus.SIMULATOR_ERROR,
        RolloutStatus.BUDGET_EXHAUSTED, RolloutStatus.INVALID_RESPONSE,
    ) and all(
        step["preservation_holds"] is True for step in outcome.rollout.steps if step["action"] is not None
    )


class _Search:
    def __init__(self, settings: IntentSearchSettings, runtime: IntentSearchRuntime, goal: str) -> None:
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
        self.nodes: list[IntentNode] = []
        self.queue_pops: list[IntentExpansion] = []
        self.attempts: list[IntentSearchAttempt] = []
        self.frontier: list[_QueueEntry] = []
        self.best: dict[tuple[SokobanSnapshot, tuple[ActionAttempt, ...]], int] = {}
        self.serial = 0
        self.selected = 0

    @property
    def remaining(self) -> int:
        return self.settings.action_budget - self.actions

    def _guard(self) -> None:
        if time.monotonic() - self.started >= self.settings.wall_seconds:
            raise _Stop("wall_budget")
        if self.remaining <= 0:
            raise _Stop("action_budget")

    def _expand(self) -> None:
        self._guard()
        if self.settings.max_expansions is not None and self.expansions >= self.settings.max_expansions:
            raise _Stop("expansion_budget")
        self.expansions += 1

    def _before_action(self) -> None:
        self._guard()
        self.actions += 1

    def _score(self, state: ExecutionState) -> float:
        self._guard()
        self.value_evaluations += 1
        estimate = self.runtime.value(self.runtime.executor.observation(state), self.goal)
        if isinstance(estimate, bool) or not math.isfinite(estimate) or not 0 <= estimate <= 1:
            raise ValueError("V must return a finite discounted goal return in [0, 1]")
        return float(estimate)

    def _invalid(
        self, parent: IntentNode, scale: Scale, phase: Phase, error: ValueError,
    ) -> None:
        self.attempts.append(IntentSearchAttempt(
            parent.node, phase, scale, None, "invalid_proposal", (), parent.state.snapshot, False, False, str(error),
        ))

    def _proposals(self, parent: IntentNode, scale: Scale, phase: Phase) -> list[Intent]:
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
                intent = decode_intent(proposal)
                if problems := validate_intent_proposal(intent, observation):
                    raise ValueError(", ".join(problems))
            except ValueError as exc:
                self._invalid(parent, scale, phase, exc)
            else:
                checked.append(intent)
        return checked

    def _trial(
        self, parent: IntentNode, intent: Intent | None, scale: Scale, phase: Phase, *, trial: int = 0,
    ) -> tuple[ExecutionOutcome, IntentSearchAttempt]:
        self._guard()
        cap = self.remaining if intent is None else min(self.remaining, self.settings.rollout_cap)
        outcome = self.runtime.executor.execute(
            parent.state, intent, self.goal, seed=self.settings.seed + trial, cap=cap, before_action=self._before_action,
        )
        self.no_ops += sum(
            step.get("simulator_no_op", step.get("no_op")) is True for step in outcome.rollout.steps
        )
        record = IntentSearchAttempt(
            parent.node, phase, scale, intent, outcome.status.value, tuple(outcome.actions), outcome.state.snapshot,
            outcome.intent_reached, outcome.task_solved, outcome.rollout.error,
        )
        self.attempts.append(record)
        return outcome, record

    def _record_node(self, parent: IntentNode, outcome: ExecutionOutcome, record: IntentSearchAttempt) -> IntentNode:
        child = IntentNode(
            len(self.nodes), parent.node, outcome.state, tuple(outcome.actions), parent.path_actions + len(outcome.actions),
            outcome.intent_reached, outcome.task_solved and _accepted(outcome),
        )
        record.node = child.node
        self.nodes.append(child)
        self.selected = child.node
        return child

    def _interrupted(self, outcome: ExecutionOutcome, record: IntentSearchAttempt) -> str | None:
        if outcome.status is RolloutStatus.BUDGET_EXHAUSTED:
            record.disposition = "interrupted"
            self._guard()
            return "api_budget"
        if outcome.status in (RolloutStatus.API_ERROR, RolloutStatus.OUTPUT_LIMIT):
            record.disposition = "interrupted"
            return "api_error"
        if outcome.status is RolloutStatus.SIMULATOR_ERROR:
            record.disposition = "interrupted"
            return "simulator_error"
        return None

    def direct(self, root: IntentNode) -> tuple[str, str | None]:
        outcome, record = self._trial(root, None, None, "direct")
        if outcome.actions or _accepted(outcome):
            self._record_node(root, outcome, record)
            record.disposition = "committed"
        if _accepted(outcome):
            return "solved", None
        return self._interrupted(outcome, record) or (
            "action_budget" if self.remaining == 0 else outcome.status.value
        ), record.error

    def _trial_score(self, outcome: ExecutionOutcome, record: IntentSearchAttempt) -> float:
        exponent = self.settings.discount ** len(outcome.actions)
        # The reward is one at primitive time τ if g is attained; terminal V is zero.
        score = exponent if outcome.task_solved else exponent * self._score(outcome.state)
        record.disposition = "scored"
        record.score = score
        return score

    def one_step(self, parent: IntentNode) -> tuple[str, str | None]:
        while True:
            self._expand()
            choice: tuple[Scale, Intent] | None = None
            best_score = -math.inf
            for scale in self.scales:
                for candidate in self._proposals(parent, scale, "probe"):
                    trials = []
                    for trial in range(self.settings.trials_per_proposal):
                        outcome, record = self._trial(parent, candidate, scale, "probe", trial=trial)
                        if reason := self._interrupted(outcome, record):
                            return reason, record.error
                        trials.append((outcome, record))
                    if any(not _accepted(outcome) and outcome.status is not RolloutStatus.ACTION_LIMIT
                           for outcome, _ in trials):
                        continue
                    score = math.fsum(self._trial_score(outcome, record) for outcome, record in trials) / len(trials)
                    if score > best_score:
                        choice, best_score = (scale, candidate), score
            self._guard()
            if choice is None:
                return "no_proposals", None
            scale, candidate = choice
            outcome, record = self._trial(parent, candidate, scale, "commit", trial=self.settings.trials_per_proposal)
            if not _accepted(outcome) and outcome.status is not RolloutStatus.ACTION_LIMIT:
                if _confirmed_prefix(outcome):
                    self._record_node(parent, outcome, record)
                    record.disposition = "committed_incomplete"
                reason = self._interrupted(outcome, record)
                return reason or ("action_budget" if self.remaining == 0 else "intent_failed"), record.error
            record.disposition = "committed" if _accepted(outcome) else "committed_incomplete"
            parent = self._record_node(parent, outcome, record)
            if parent.task_solved:
                return "solved", None

    def _enqueue(self, node: IntentNode) -> None:
        node.value = self._score(node.state)
        self.best[_state_key(node.state)] = node.node
        for scale in self.scales:
            heapq.heappush(self.frontier, _QueueEntry(-_priority(scale), -node.value, self.serial, node.node, scale))
            self.serial += 1

    def _admit(self, parent: IntentNode, outcome: ExecutionOutcome, record: IntentSearchAttempt) -> bool:
        if not _accepted(outcome):
            return False
        key = _state_key(outcome.state)
        prior = self.best.get(key)
        if prior is not None and self.nodes[prior].path_actions <= parent.path_actions + len(outcome.actions):
            record.disposition = "duplicate"
            return False
        child = self._record_node(parent, outcome, record)
        record.disposition = "reopened" if prior is not None else "accepted"
        if child.task_solved:
            return True
        self._enqueue(child)
        return False

    def full_search(self, root: IntentNode) -> tuple[str, str | None]:
        if self.settings.max_expansions == 0:
            self._guard()
            return "expansion_budget", None
        self._enqueue(root)
        while self.frontier:
            entry = heapq.heappop(self.frontier)
            parent = self.nodes[entry.node]
            if self.best[_state_key(parent.state)] != parent.node:
                continue
            self._expand()
            self.queue_pops.append(IntentExpansion(
                parent.node, entry.scale, -entry.negative_value, entry.serial, len(self.frontier), self.actions,
            ))
            for candidate in self._proposals(parent, entry.scale, "search"):
                for trial in range(self.settings.trials_per_proposal):
                    outcome, record = self._trial(parent, candidate, entry.scale, "search", trial=trial)
                    if reason := self._interrupted(outcome, record):
                        return reason, record.error
                    if self._admit(parent, outcome, record):
                        return "solved", None
            self._guard()
        return "frontier_exhausted", None

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
            "value_evaluations": self.value_evaluations,
            "logical_api_calls": metrics["logical_requests"],
            "logical_model_calls": metrics["logical_requests"] + metrics["local_requests"] + metrics["fitted_value_calls"],
            "logical_policy_calls": metrics["policy_calls"] + metrics["local_policy_calls"],
            "logical_generator_calls": metrics["generator_calls"] + metrics["local_generator_calls"],
            "logical_value_calls": metrics["value_calls"] + metrics["local_value_calls"] + metrics["fitted_value_calls"],
            "wall_seconds": time.monotonic() - self.started,
        }

    def run(self, mode: SearchMode, initial: ExecutionState) -> IntentSearchResult:
        root = IntentNode(0, None, initial, (), 0, False, False)
        self.nodes.append(root)
        try:
            root.task_solved = self.runtime.executor.task_solved(initial)
            operation = {SearchMode.DIRECT: self.direct, SearchMode.ONE_STEP: self.one_step, SearchMode.FULL_SEARCH: self.full_search}
            reason, error = ("solved", None) if root.task_solved else operation[mode](root)
        except _Stop as exc:
            reason, error = exc.reason, None
        except BudgetExceeded as exc:
            reason, error = "api_budget", str(exc)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            reason, error = "api_error", f"{type(exc).__name__}: {exc}"
        selected = self.nodes[self.selected]
        path = self._path()
        return IntentSearchResult(
            mode, selected.task_solved, reason, path, selected.state.snapshot, selected.task_solved,
            selected.intent_reached, selected.node, self.nodes, self.queue_pops, self.attempts, self._accounting(path), error,
        )


def search_intents(
    initial: SokobanSnapshot, task_goal: str, *, mode: SearchMode | str, settings: IntentSearchSettings,
    runtime: IntentSearchRuntime, initial_memory: Sequence[ActionAttempt] = (),
) -> IntentSearchResult:
    """Plan from a fresh initial snapshot using one shared budget per run."""
    if task_goal != TASK_GOAL:
        raise ValueError("The Sokoban planner supports the all-boxes-on-goals task")
    initial_state = ExecutionState(initial, tuple(initial_memory))
    return _Search(settings, runtime, task_goal).run(SearchMode(mode), initial_state)
