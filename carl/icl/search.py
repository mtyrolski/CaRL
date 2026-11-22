"""Explicit-state search with bounded execution and lexicographic priorities."""
import heapq
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Final

import numpy as np

from carl.icl.api import BudgetExceeded, RequestFailed
from carl.icl.boards import to_rows
from carl.icl.config import ExperimentConfig
from carl.icl.planning.search_types import (
    Attempt,
    FinishReason,
    QueueEntry,
    SearchObserver,
    SearchPath,
    SearchResult,
    SearchTransition,
    TransitionKind,
    TransitionStatus,
)
from carl.icl.types import (
    Board,
    BoardEnvironment,
    RolloutPolicy,
    SearchComponents,
    SearchEvent,
)

ACTION_COUNT: Final = 4
PRIMITIVE_SCALE: Final = 0.0

__all__ = [
    "ACTION_COUNT", "PRIMITIVE_SCALE", "Attempt", "FinishReason", "Frontier", "QueueEntry",
    "SearchObserver", "SearchPath", "SearchResult", "SearchSession", "SearchTransition",
    "TransitionKind", "TransitionStatus", "rollout", "search",
]


def rollout(
    env: BoardEnvironment,
    components: RolloutPolicy,
    state: Board,
    target: Board,
    *,
    cap: int,
    rng: np.random.Generator,
    on_step: Callable[[], None] | None = None,
    on_action: Callable[[Board, int, Board], None] | None = None,
) -> Attempt:
    """Sample the frozen policy until an exact hit, task completion or the cap."""
    actions: list[int] = []
    current = state.copy()
    if np.array_equal(current, target) or env.is_solved(current):
        return Attempt(current, actions, np.array_equal(current, target), env.is_solved(current))
    for _ in range(cap):
        action = int(rng.choice(ACTION_COUNT, p=components.probabilities(current, target)))
        previous = current
        current = np.asarray(env.next_state(current, action))
        actions.append(action)
        if on_step is not None:
            on_step()
        if on_action is not None:
            on_action(previous, action, current)
        reached, solved = np.array_equal(current, target), env.is_solved(current)
        if reached or solved:
            return Attempt(current, actions, reached, solved)
    return Attempt(current, actions, False, False)


class Frontier:
    """Keep shortest discovered plans and discard obsolete heap entries."""

    def __init__(self, levels: list[float], components: SearchComponents) -> None:
        self.levels = levels
        self.components = components
        self.queue: list[QueueEntry] = []
        self.best_cost: dict[bytes, int] = {}
        self.serial = 0

    def push(self, path: SearchPath) -> bool:
        """Insert an improvement; return whether an existing state was reopened."""
        old_cost = self.best_cost.get(path.key)
        if old_cost is not None and old_cost <= len(path.actions):
            return False
        value = self.components.value(path.state)
        self.best_cost[path.key] = len(path.actions)
        for scale in self.levels:
            self.serial += 1
            heapq.heappush(self.queue, QueueEntry(-scale, -value, self.serial, path))
        return old_cost is not None

    def pop(self) -> QueueEntry | None:
        """Return the highest-priority current entry, skipping stale plans."""
        while self.queue:
            entry = heapq.heappop(self.queue)
            if len(entry.path.actions) == self.best_cost[entry.path.key]:
                return entry
        return None


def _task_target(initial: Board) -> Board:
    """Construct a solved target for the direct goal-conditioned control."""
    target = initial.argmax(-1).copy()
    boxes = np.isin(target, [3, 4])
    target[boxes] = np.where(target[boxes] == 3, 2, 1)
    target[np.isin(target, [2, 6])] = 3
    player = tuple(np.argwhere(np.isin(initial.argmax(-1), [5, 6]))[0])
    if target[player] == 3:
        player = tuple(np.argwhere(target == 1)[0])
    target[np.isin(target, [5, 6])] = 1
    target[player] = 5
    result: Board = np.eye(7, dtype=np.float32)[target]
    return result


class SearchSession:
    """Own one run's mutable frontier, random stream and accounting."""

    def __init__(
        self,
        env: BoardEnvironment,
        components: SearchComponents,
        config: ExperimentConfig,
        seed: int,
        *,
        observer: SearchObserver | None = None,
    ) -> None:
        self.env, self.components, self.config = env, components, config
        self.observer = observer
        self.result = SearchResult()
        self.rng = np.random.default_rng(seed)
        levels = [] if config.method == "primitive" else sorted(config.scales, reverse=True)
        self.frontier = Frontier(levels + [PRIMITIVE_SCALE], components)

    @property
    def remaining_actions(self) -> int:
        """Return the budget available to the next execution attempt."""
        return self.config.action_budget - self.result.attempted_actions

    def _accept_solution(self, path: SearchPath) -> bool:
        if not self.env.is_solved(path.state):
            return False
        self.result.solved = True
        self.result.finish_reason = FinishReason.SOLVED.value
        self.result.action_path = path.actions
        self.result.subgoal_path = [to_rows(state) for state in path.endpoints]
        return True

    def _insert(self, path: SearchPath) -> TransitionStatus:
        old_cost = self.frontier.best_cost.get(path.key)
        reopened = self.frontier.push(path)
        self.result.reopened += int(reopened)
        if old_cost is not None and old_cost <= len(path.actions):
            return TransitionStatus.DUPLICATE
        return TransitionStatus.REOPENED if reopened else TransitionStatus.NEW

    def _observe(self, transition: SearchTransition) -> None:
        if self.observer is not None:
            self.observer.on_transition(transition)

    def _admit(self, transition: SearchTransition) -> None:
        try:
            status = self._insert(transition.destination)
        except (BudgetExceeded, RequestFailed):
            self._observe(replace(transition, status=TransitionStatus.INTERRUPTED.value))
            raise
        self._observe(replace(transition, status=status.value))

    def _primitive(self, path: SearchPath) -> None:
        for action in range(ACTION_COUNT):
            if self.remaining_actions <= 0:
                return
            self.result.count_step()
            successor = np.asarray(self.env.next_state(path.state, action))
            if np.array_equal(successor, path.state):
                if self.observer is not None:
                    self._observe(SearchTransition(
                        path, path.extend(successor, [action]),
                        TransitionKind.PRIMITIVE.value, TransitionStatus.NO_OP.value, 0.0, (action,),
                    ))
                continue
            self.result.primitive_successors += 1
            reached = path.extend(successor, [action])
            if self._accept_solution(reached):
                self._observe(SearchTransition(
                    path, reached, TransitionKind.PRIMITIVE.value, TransitionStatus.SOLVED.value, 0.0, (action,), solved=True,
                ))
                return
            self._admit(SearchTransition(path, reached, TransitionKind.PRIMITIVE.value, TransitionStatus.NEW.value, 0.0, (action,)))

    def _attempt(
        self, path: SearchPath, target: Board, cap: int, *, scale: float, kind: TransitionKind
    ) -> Attempt:
        actions: list[int] = []
        endpoint = path.state

        def record_action(_before: Board, action: int, after: Board) -> None:
            nonlocal endpoint
            actions.append(action)
            endpoint = after.copy()

        try:
            return rollout(
                self.env,
                self.components,
                path.state,
                target,
                cap=min(cap, self.remaining_actions),
                rng=self.rng,
                on_step=self.result.count_step,
                on_action=record_action if self.observer is not None else None,
            )
        except (BudgetExceeded, RequestFailed):
            if self.observer is not None:
                self._observe(SearchTransition(
                    path, path.extend(endpoint, actions), kind.value, TransitionStatus.INTERRUPTED.value, scale, tuple(actions), proposal=target,
                ))
            raise

    def _hierarchical(self, path: SearchPath, scale: float) -> None:
        cap = self.config.calibration.action_cap
        if self.config.method == "fixed" and self.config.fixed_cap == "horizon":
            cap = min(cap, int(scale))
        for proposal in self.components.propose(path.state, scale):
            if self.remaining_actions <= 0:
                return
            attempt = self._attempt(path, proposal, cap, scale=scale, kind=TransitionKind.SUBGOAL)
            self.result.events.append(
                SearchEvent(
                    scale=scale,
                    attempted_actions=len(attempt.actions),
                    reached=attempt.reached,
                    solved=attempt.solved,
                    source=to_rows(path.state),
                    proposal=to_rows(proposal),
                    endpoint=to_rows(attempt.state),
                )
            )
            reached = path.extend(attempt.state, attempt.actions)
            if self._accept_solution(reached):
                self._observe(SearchTransition(
                    path, reached, TransitionKind.SUBGOAL.value, TransitionStatus.SOLVED.value, scale, tuple(attempt.actions), proposal, attempt.reached, True,
                ))
                return
            if attempt.reached:
                self.result.accepted += 1
                self._admit(SearchTransition(
                    path, reached, TransitionKind.SUBGOAL.value, TransitionStatus.NEW.value, scale, tuple(attempt.actions), proposal, True, False,
                ))
            else:
                self.result.rejected += 1
                self._observe(SearchTransition(
                    path, reached, TransitionKind.SUBGOAL.value, TransitionStatus.REJECTED.value, scale, tuple(attempt.actions), proposal, False, False,
                ))

    def _expand(self, initial: SearchPath, started: float) -> None:
        self._insert(initial)
        while self.frontier.queue and self.remaining_actions > 0:
            if time.perf_counter() - started >= self.config.limits.wall_seconds:
                raise BudgetExceeded("Search wall-time budget exhausted")
            if self.config.max_expansions is not None and self.result.expansions >= self.config.max_expansions:
                self.result.finish_reason = FinishReason.EXPANSION_BUDGET.value
                return
            entry = self.frontier.pop()
            if entry is None:
                return
            self.result.expansions += 1
            if self.observer is not None:
                self.observer.on_expansion(entry, self.result.expansions, self.result.attempted_actions, len(self.frontier.queue))
            if entry.negative_scale == PRIMITIVE_SCALE:
                self._primitive(entry.path)
            else:
                self._hierarchical(entry.path, -entry.negative_scale)
            if self.result.solved:
                return

    def run(self, initial: Board) -> SearchResult:
        """Execute a bounded search and retain partial accounting on API failure."""
        started = time.perf_counter()
        root = SearchPath(initial, [], [initial])
        if self.observer is not None:
            self.observer.on_start(root)
        try:
            if not self._accept_solution(root):
                if self.config.method == "direct":
                    target = _task_target(initial)
                    attempt = self._attempt(
                        root, target, self.config.action_budget, scale=0.0, kind=TransitionKind.DIRECT,
                    )
                    endpoint = root.extend(attempt.state, attempt.actions)
                    solved = self._accept_solution(endpoint)
                    status = TransitionStatus.SOLVED if solved else TransitionStatus.REJECTED
                    self._observe(SearchTransition(
                        root, endpoint, TransitionKind.DIRECT.value, status.value, 0.0, tuple(attempt.actions), target, attempt.reached, solved,
                    ))
                    if not solved:
                        self.result.finish_reason = FinishReason.ACTION_BUDGET.value
                else:
                    self._expand(root, started)
                if not self.result.solved and self.remaining_actions <= 0:
                    self.result.finish_reason = FinishReason.ACTION_BUDGET.value
        except (BudgetExceeded, RequestFailed) as error:
            reason = FinishReason.API_BUDGET if isinstance(error, BudgetExceeded) else FinishReason.API_ERROR
            self.result.finish_reason = reason.value
            self.result.error = str(error)
        finally:
            self.result.seconds = time.perf_counter() - started
            if self.observer is not None:
                self.observer.on_finish(self.result)
        return self.result


def search(
    env: BoardEnvironment,
    components: SearchComponents,
    initial: Board,
    config: ExperimentConfig,
    seed: int,
    *,
    observer: SearchObserver | None = None,
) -> SearchResult:
    """Run the shared search used by relative and fixed-horizon conditions."""
    return SearchSession(env, components, config, seed, observer=observer).run(initial)
