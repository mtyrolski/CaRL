"""Execute structured Sokoban intents with a one-action-at-a-time policy.

The policy receives the current board, the task and the predicate conjunction;
demonstration endpoints and offsets are never inputs. The controller prompt is
fixed across calibration and evaluation. Predicate checks use the real states
returned by the environment, not a language-model judge.
"""
import json
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np

from carl.icl.boards import to_rows
from carl.icl.intent_data import predicate_value
from carl.icl.intent_data import validate_prediction
from carl.icl.intent_domain import decode_intent
from carl.icl.luna_components.formatting import board_coordinates
from carl.icl.types import Board
from carl.icl.types import BoardEnvironment

POLICY_PROMPT: Final = """You control a Sokoban player. Choose exactly one primitive action that helps fulfil the given intent while keeping every preservation condition true.
The task is to put all boxes on goals. The board uses # wall, - floor, . goal, $ box, * box on goal, @ player, + player on goal. \
A player can push one box into an empty square, but cannot pull or push two boxes. A move into a wall or blocked box has no effect.
Coordinates are zero-based [row, column], rows increase DOWN and columns increase RIGHT. box_at(row,col) requires a box on that cell; \
player_at(row,col) requires the player there; boxes_on_goals_at_least(count) requires at least that many boxes on goals. \
Every predicate in effects must hold simultaneously. Every predicate in preserve must remain true throughout.
attempt_history lists earlier actions in this same attempt in order; no_op means an action did not change the board. Use this history to avoid repeating blocked moves unnecessarily.
Respond with exactly one JSON object, with no other text: {"action":"UP"}, {"action":"DOWN"}, {"action":"LEFT"}, or {"action":"RIGHT"}."""


class Action(StrEnum):
    """Primitive actions in the order used by the Sokoban environment."""

    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass(frozen=True)
class ActionAttempt:
    """An action already tried in the current attempt and whether it changed the board."""

    action: Action
    no_op: bool


class RolloutStatus(StrEnum):
    """Mutually exclusive termination reasons for an intent attempt."""

    ACHIEVED = "achieved"
    PRESERVATION_VIOLATED = "preservation_violated"
    TASK_SOLVED = "task_solved"
    ACTION_LIMIT = "action_limit"
    INVALID_INTENT = "invalid_intent"
    INVALID_RESPONSE = "invalid_response"
    OUTPUT_LIMIT = "output_limit"
    API_ERROR = "api_error"


class OutputLimitError(RuntimeError):
    """A policy response was cut off before completing its action."""


@dataclass
class IntentRollout:
    """Serializable attempt, including unsuccessful and interrupted decisions."""

    status: RolloutStatus
    achieved: bool
    task_solved: bool
    actions: list[int]
    model_calls: int
    endpoint: list[str]
    steps: list[dict[str, Any]]
    error: str | None = None


def build_prompt(state: Board, intent: dict[str, Any], task_goal: str,
                 memory: Sequence[ActionAttempt] = ()) -> str:
    """Render the current state and previous actions from this attempt only."""
    if any(not isinstance(item, ActionAttempt) or not isinstance(item.action, Action) or not isinstance(item.no_op, bool)
           for item in memory):
        raise ValueError("Memory must contain action attempts with a boolean no-op flag")
    rows = to_rows(state)
    checked = decode_intent({"effects": intent["effects"], "preserve": intent["preserve"], "instruction": None})
    question = {
        "state": rows,
        "state_coordinates": board_coordinates(rows),
        "intent": {"effects": checked["effects"], "preserve": checked["preserve"]},
        "task_goal": task_goal,
        "attempt_history": [{"action": item.action.value, "no_op": item.no_op} for item in memory],
    }
    return f"{POLICY_PROMPT}\n\n{json.dumps(question, sort_keys=True, separators=(',', ':'))}"


def _action(raw_response: str) -> int:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    response = json.loads(raw_response, object_pairs_hook=unique_object)
    if not isinstance(response, dict) or set(response) != {"action"}:
        raise ValueError("Expected an object containing only action")
    try:
        action = Action(response["action"])
    except (TypeError, ValueError):
        raise ValueError("Action must be UP, DOWN, LEFT or RIGHT") from None
    return list(Action).index(action)


def rollout(
    env: BoardEnvironment,
    state: Board,
    intent: dict[str, Any],
    decide: Callable[[str], str],
    *,
    cap: int,
    task_goal: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> IntentRollout:
    """Attempt an intent until first satisfaction, violation, terminal or cap.

    ``decide`` makes one policy call and returns its raw output text. The caller
    owns the transport, output-token cap, request timeout, usage and cost logs.
    A failed or malformed response stops the attempt without fabricating an
    action. Every valid attempted action, including a no-op, counts towards cap.
    ``on_event`` receives request, response/error and executed-action events.
    """
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError("cap must be a positive integer")
    if not isinstance(task_goal, str) or not task_goal.strip():
        raise ValueError("task_goal must be a nonempty string")
    current = state.copy()
    rows = to_rows(current)
    errors = validate_prediction(intent, rows)
    solved = bool(env.is_solved(current))
    if errors:
        return IntentRollout(RolloutStatus.INVALID_INTENT, False, solved, [], 0, rows, [], "; ".join(errors))
    if solved:
        return IntentRollout(RolloutStatus.TASK_SOLVED, False, True, [], 0, rows, [])

    actions: list[int] = []
    steps: list[dict[str, Any]] = []
    memory: list[ActionAttempt] = []

    def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    def finish(status: RolloutStatus, error: str | None = None) -> IntentRollout:
        return IntentRollout(status, status == RolloutStatus.ACHIEVED, solved, actions, len(steps), rows, steps, error)

    for index in range(cap):
        prompt = build_prompt(current, intent, task_goal, memory)
        step: dict[str, Any] = {
            "index": index, "before": rows, "prompt": prompt, "raw_response": None,
            "action": None, "after": None, "no_op": None, "intent_holds": None,
            "preservation_holds": None, "task_solved": None, "error": None,
        }
        steps.append(step)
        emit({"type": "request", "index": index, "before": rows, "prompt": prompt})
        try:
            raw = decide(prompt)
        except OutputLimitError as exc:
            error = f"{type(exc).__name__}: {exc}"
            step["error"] = error
            emit({"type": "output_limit", "index": index, "error": error})
            return finish(RolloutStatus.OUTPUT_LIMIT, error)
        except Exception as exc:  # Preserve earlier actions when a transport fails.
            error = f"{type(exc).__name__}: {exc}"
            step["error"] = error
            emit({"type": "error", "index": index, "error": error})
            return finish(RolloutStatus.API_ERROR, error)
        step["raw_response"] = raw
        emit({"type": "response", "index": index, "raw_response": raw})
        try:
            action = _action(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            step["error"] = error
            return finish(RolloutStatus.INVALID_RESPONSE, error)

        previous = current
        current = np.asarray(env.next_state(current, action))
        rows = to_rows(current)
        actions.append(action)
        preserved = all(predicate_value(predicate, rows) for predicate in intent["preserve"])
        holds = all(predicate_value(predicate, rows) for predicate in intent["effects"])
        solved = bool(env.is_solved(current))
        step.update(action=action, after=rows, no_op=bool(np.array_equal(previous, current)),
                    intent_holds=holds, preservation_holds=preserved, task_solved=solved)
        memory.append(ActionAttempt(list(Action)[action], step["no_op"]))
        emit({"type": "action", **{key: value for key, value in step.items() if key != "prompt"}})
        if not preserved:
            return finish(RolloutStatus.PRESERVATION_VIOLATED)
        if holds:
            return finish(RolloutStatus.ACHIEVED)
        if solved:
            return finish(RolloutStatus.TASK_SOLVED)
    return finish(RolloutStatus.ACTION_LIMIT)
