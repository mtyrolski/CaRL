"""Search with textual subgoals and a model judge, without a hidden reachability oracle."""
import heapq
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Final, cast

import numpy as np

from carl.icl.api import BudgetExceeded, RequestFailed
from carl.icl.boards import from_rows, to_rows
from carl.icl.types import Board, BoardEnvironment, BoardRows
from scripts.verbal_icl.types import Attempt, Components, Result, Step

ACTION_COUNT: Final = 4


class RolloutFailure(Exception):
    """Preserve actions taken before an API failure."""

    def __init__(self, attempt: Attempt, cause: BudgetExceeded | RequestFailed) -> None:
        super().__init__(str(cause))
        self.attempt, self.cause = attempt, cause


def verify_trace(env: BoardEnvironment, initial: Board, events: list[dict[str, object]], actions: int) -> None:
    """Replay recorded solver actions; never reclassify a model judgment."""
    seen = {initial.argmax(-1).tobytes()}
    if len(events) != actions:
        raise ValueError('Trace action count disagrees with the result')
    for event in events:
        before = from_rows(cast(BoardRows, event['before']))
        expected = from_rows(cast(BoardRows, event['after']))
        action = event['action']
        if (before.argmax(-1).tobytes() not in seen or not isinstance(action, int)
                or isinstance(action, bool) or action not in range(ACTION_COUNT) or env.is_solved(before)):
            raise ValueError('Trace has an invalid source or action')
        actual = np.asarray(env.next_state(before, action))
        if not np.array_equal(actual, expected) or bool(env.is_solved(actual)) != event['solved']:
            raise ValueError('Trace endpoint or solved flag does not replay')
        seen.add(actual.argmax(-1).tobytes())


def rollout(env: BoardEnvironment, components: Components, source: Board, goal: str, cap: int,
            on_action: Callable[[Board, int, Board, bool], None] | None = None) -> Attempt:
    """Execute at most cap actions; only the judge decides whether the text goal was reached."""
    state = source.copy()
    actions: list[int] = []
    steps: list[Step] = []
    if env.is_solved(state):
        return Attempt(state, actions, steps, False, True)
    for _ in range(cap):
        try:
            action = components.action(state, goal, source)
            if action not in range(ACTION_COUNT):
                raise RequestFailed('Controller returned an invalid action')
            before = state
            state = np.asarray(env.next_state(state, action))
            actions.append(action)
            solved = bool(env.is_solved(state))
            if on_action is not None:
                on_action(before, action, state, solved)
            try:
                judgment = components.judge(state, goal, source)
            except (BudgetExceeded, RequestFailed):
                steps.append(Step(to_rows(before), action, to_rows(state), None, solved))
                raise
            steps.append(Step(to_rows(before), action, to_rows(state), judgment, solved))
            if solved or judgment.reached:
                return Attempt(state, actions, steps, judgment.reached, solved)
        except (BudgetExceeded, RequestFailed) as error:
            raise RolloutFailure(Attempt(state, actions, steps, False, bool(env.is_solved(state)), True), error) from error
    return Attempt(state, actions, steps, False, False)


@dataclass(order=True)
class Entry:
    """Scale takes priority over value, then insertion order."""

    negative_scale: float
    negative_value: float
    serial: int
    state: Board = field(compare=False)
    path: list[int] = field(compare=False)


def search(env: BoardEnvironment, components: Components, initial: Board, scales: tuple[float, float],
           action_budget: int, action_cap: int, max_expansions: int | None,
           trace: Callable[[dict[str, object]], None] | None = None) -> Result:
    """Run bounded best-first search. Primitive fallback executes four visible actions."""
    if (not 0 < scales[1] < scales[0] < 1 or min(action_budget, action_cap) < 1
            or (max_expansions is not None and max_expansions < 1)):
        raise ValueError('Expected descending relative scales in (0,1) and positive budgets')
    result = Result()
    heap: list[Entry] = []
    best: dict[bytes, int] = {}
    serial = 0

    def trace_action(before: Board, action: int, after: Board, solved: bool) -> None:
        if trace is not None:
            trace({'event': 'action', 'before': to_rows(before), 'action': action,
                   'after': to_rows(after), 'solved': solved})

    def add(state: Board, path: list[int]) -> None:
        nonlocal serial
        key = state.argmax(-1).tobytes()
        old = best.get(key)
        if old is not None and old <= len(path):
            return
        value = components.value(state)
        best[key] = len(path)
        result.reopened += int(old is not None)
        for scale in (*scales, 0.0):
            serial += 1
            heapq.heappush(heap, Entry(-scale, -value, serial, state.copy(), path.copy()))

    def solve(state: Board, path: list[int]) -> bool:
        if not env.is_solved(state):
            return False
        result.solved, result.finish_reason, result.action_path = True, 'solved', path
        return True

    try:
        if solve(initial, []):
            return result
        add(initial, [])
        while heap and result.attempted_actions < action_budget:
            if max_expansions is not None and result.expansions >= max_expansions:
                result.finish_reason = 'expansion_budget'
                return result
            entry = heapq.heappop(heap)
            if len(entry.path) != best[entry.state.argmax(-1).tobytes()]:
                continue
            result.expansions += 1
            scale = -entry.negative_scale
            if scale == 0:
                for action in range(ACTION_COUNT):
                    if result.attempted_actions == action_budget:
                        break
                    endpoint = np.asarray(env.next_state(entry.state, action))
                    result.attempted_actions += 1
                    result.primitive_actions += 1
                    trace_action(entry.state, action, endpoint, bool(env.is_solved(endpoint)))
                    path = [*entry.path, action]
                    result.events.append({'kind': 'primitive', 'source': to_rows(entry.state),
                                          'endpoint': to_rows(endpoint), 'actions': [action], 'scale': 0.0})
                    if solve(endpoint, path):
                        return result
                    if not np.array_equal(endpoint, entry.state):
                        add(endpoint, path)
                continue
            for goal in components.propose(entry.state, scale):
                if result.attempted_actions == action_budget:
                    break
                failure: RolloutFailure | None = None
                try:
                    attempt = rollout(env, components, entry.state, goal,
                                      min(action_cap, action_budget - result.attempted_actions), trace_action)
                except RolloutFailure as error:
                    attempt, failure = error.attempt, error
                result.attempted_actions += len(attempt.actions)
                result.events.append({'kind': 'verbal', 'source': to_rows(entry.state), 'goal': goal,
                                      'endpoint': to_rows(attempt.endpoint), 'scale': scale,
                                      'actions': attempt.actions, 'steps': [asdict(step) for step in attempt.steps],
                                      'judged_reached': attempt.judged_reached, 'solved': attempt.solved,
                                      'interrupted': attempt.interrupted})
                path = [*entry.path, *attempt.actions]
                if attempt.solved and solve(attempt.endpoint, path):
                    result.error = str(failure.cause) if failure is not None else None
                    return result
                if failure is not None:
                    raise failure.cause
                if attempt.judged_reached:
                    result.accepted += 1
                    add(attempt.endpoint, path)
                else:
                    result.rejected += 1
        if result.attempted_actions == action_budget:
            result.finish_reason = 'action_budget'
    except (BudgetExceeded, RequestFailed) as error:
        result.finish_reason = 'api_budget' if isinstance(error, BudgetExceeded) else 'api_error'
        result.error = str(error)
    return result
