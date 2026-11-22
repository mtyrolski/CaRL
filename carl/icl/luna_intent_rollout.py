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
from typing import Any, Final, Protocol, runtime_checkable

import numpy as np

from carl.icl.api import BudgetExceeded
from carl.icl.boards import to_rows
from carl.icl.intent_data import validate_prediction
from carl.icl.intent_domain import Intent, Predicate, SokobanObservation, UNKNOWN_TILE
from carl.icl.intent_domain import decode_intent, predicate_truth, render_instruction
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

MEASUREMENT_PROMPT_VERSION: Final = "sokoban-measurement-prompt-v1"
ACTION_PARSER_VERSION: Final = "sokoban-one-action-json-v1"
MEASUREMENT_PROMPT: Final = POLICY_PROMPT + (
    "\nThe instruction describes the same intent as the predicates. "
    "History starts with the restored controller memory. ? hides a cell, "
    "and no_op means that the visible board did not change."
)


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


@runtime_checkable
class ObservationPolicy(Protocol):
    """A controller that consumes the visible observation and its own action history."""

    def render_action(
        self, observation: SokobanObservation, intent: Intent, memory: Sequence[ActionAttempt],
    ) -> str:
        """Render the action request for the rollout trace without calling the model."""
        ...

    def choose_action(
        self, observation: SokobanObservation, intent: Intent, memory: Sequence[ActionAttempt],
    ) -> str:
        """Return the raw response for the shared action parser."""
        ...


@runtime_checkable
class TaskObservationPolicy(Protocol):
    """The same observation policy with the task goal as its instruction."""

    def render_task_action(
        self, observation: SokobanObservation, task_goal: str, memory: Sequence[ActionAttempt],
    ) -> str:
        """Render a direct decision request without sending it."""
        ...

    def choose_task_action(
        self, observation: SokobanObservation, task_goal: str, memory: Sequence[ActionAttempt],
    ) -> str:
        """Return a direct response for the same action parser."""
        ...


class RolloutStatus(StrEnum):
    """Mutually exclusive termination reasons for an intent attempt."""

    ACHIEVED = "achieved"
    PRESERVATION_VIOLATED = "preservation_violated"
    TASK_SOLVED = "task_solved"
    ACTION_LIMIT = "action_limit"
    INVALID_INTENT = "invalid_intent"
    INVALID_RESPONSE = "invalid_response"
    OUTPUT_LIMIT = "output_limit"
    EPISODE_TERMINATED = "episode_terminated"
    EPISODE_TRUNCATED = "episode_truncated"
    API_ERROR = "api_error"
    SIMULATOR_ERROR = "simulator_error"
    UNKNOWN = "unknown"
    BUDGET_EXHAUSTED = "budget_exhausted"


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

    @property
    def first_hit_actions(self) -> int | None:
        """Report the first confirmed hit, counted from the restored snapshot."""
        return len(self.actions) if self.achieved else None


def build_prompt(state: Board, intent: dict[str, Any], task_goal: str,
                 memory: Sequence[ActionAttempt] = (), *, measured: bool = False) -> str:
    """Render the current state and previous actions from this attempt only."""
    observation = SokobanObservation(tuple(to_rows(state)))
    checked = decode_intent({"effects": intent["effects"], "preserve": intent["preserve"], "instruction": None})
    return _render_prompt(observation, checked, task_goal, memory, measured=measured)


def _render_prompt(
    observation: SokobanObservation, intent: Intent, task_goal: str,
    memory: Sequence[ActionAttempt], *, measured: bool,
) -> str:
    if any(not isinstance(item, ActionAttempt) or not isinstance(item.action, Action) or not isinstance(item.no_op, bool)
           for item in memory):
        raise ValueError("Memory must contain action attempts with a boolean no-op flag")
    rows = list(observation.rows)
    question: dict[str, Any] = {
        "state": rows,
        "state_coordinates": None if observation.partial else board_coordinates(rows),
        "intent": {"effects": intent["effects"], "preserve": intent["preserve"]},
        "task_goal": task_goal,
        "attempt_history": [{"action": item.action.value, "no_op": item.no_op} for item in memory],
    }
    if measured:
        question["instruction"] = render_instruction(intent)
    prompt = MEASUREMENT_PROMPT if measured else POLICY_PROMPT
    return f"{prompt}\n\n{json.dumps(question, sort_keys=True, separators=(',', ':'))}"


def observe_state(
    state: Board, observe: Callable[[Board], SokobanObservation] | None,
) -> SokobanObservation:
    """Observe a copy of the state, allowing only faithful masking of cells."""
    actual = tuple(to_rows(state))
    if observe is None:
        return SokobanObservation(actual)
    observation = observe(state.copy())
    if len(observation.rows) != len(actual) or len(observation.rows[0]) != len(actual[0]):
        raise ValueError("Observed board must have the simulator's dimensions")
    if any(
        visible not in (UNKNOWN_TILE, tile)
        for original, masked in zip(actual, observation.rows, strict=True)
        for tile, visible in zip(original, masked, strict=True)
    ):
        raise ValueError("Observation must preserve every visible simulator tile")
    return observation


def _holds(predicates: Sequence[Predicate], observation: SokobanObservation) -> bool | None:
    values = [predicate_truth(predicate, observation) for predicate in predicates]
    if False in values:
        return False
    return None if None in values else True


def _unknown_initial(intent: Intent, observation: SokobanObservation) -> bool:
    if not observation.partial:
        return False
    effects = [predicate_truth(predicate, observation) for predicate in intent["effects"]]
    preserve = _holds(intent["preserve"], observation)
    return True not in effects and preserve is not False and (None in effects or preserve is None)


def _initial_intent(
    intent: dict[str, Any], observation: SokobanObservation, state: Board,
) -> tuple[Intent | None, RolloutStatus | None, str | None]:
    try:
        checked = decode_intent({**intent, "instruction": None})
        if _unknown_initial(checked, observation):
            return None, RolloutStatus.UNKNOWN, None
    except (TypeError, ValueError, IndexError) as exc:
        return None, RolloutStatus.INVALID_INTENT, str(exc)
    errors = validate_prediction(intent, to_rows(state))
    if errors:
        return None, RolloutStatus.INVALID_INTENT, "; ".join(errors)
    return checked, None, None


def _policy_failure(exc: Exception) -> tuple[RolloutStatus, str, str]:
    if isinstance(exc, BudgetExceeded):
        return RolloutStatus.BUDGET_EXHAUSTED, "budget_exhausted", str(exc)
    error = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, OutputLimitError):
        return RolloutStatus.OUTPUT_LIMIT, "output_limit", error
    return RolloutStatus.API_ERROR, "error", error


def _termination(preserved: bool | None, holds: bool | None, solved: bool) -> RolloutStatus | None:
    if preserved is False:
        return RolloutStatus.PRESERVATION_VIOLATED
    if preserved is None:
        return RolloutStatus.UNKNOWN
    if holds is True:
        return RolloutStatus.ACHIEVED
    if solved:
        return RolloutStatus.UNKNOWN if holds is None else RolloutStatus.TASK_SOLVED
    return None


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
    intent: dict[str, Any] | None,
    decide: Callable[[str], str] | ObservationPolicy | TaskObservationPolicy,
    *,
    cap: int,
    task_goal: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    initial_memory: Sequence[ActionAttempt] = (),
    observe: Callable[[Board], SokobanObservation] | None = None,
    measured: bool = False,
    before_action: Callable[[], None] | None = None,
    on_state: Callable[[Board, tuple[ActionAttempt, ...]], None] | None = None,
) -> IntentRollout:
    """Attempt an intent until first satisfaction, violation, terminal or cap.

    ``decide`` makes one policy call and returns its raw output text. The caller
    owns the transport, output-token cap, request timeout, usage and cost logs.
    A failed or malformed response stops the attempt without fabricating an
    action. Every valid attempted action, including a no-op, counts towards cap.
    ``on_event`` receives request, response/error and executed-action events.
    ``initial_memory`` is copied on every attempt. An observation may mask cells.
    ``before_action`` is called once before each simulator transition. The
    trusted ``on_state`` hook receives a private successor and immutable memory.
    A missing intent gives the task goal directly to a typed observation policy.
    """
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError("cap must be a positive integer")
    if not isinstance(task_goal, str) or not task_goal.strip():
        raise ValueError("task_goal must be a nonempty string")
    current = state.copy()
    observation = observe_state(current, observe)
    rows = list(observation.rows)
    solved = bool(env.is_solved(current))
    checked = None
    if intent is not None:
        checked, status, error = _initial_intent(intent, observation, current)
        if status is not None:
            return IntentRollout(status, False, solved, [], 0, rows, [], error)
    if solved:
        return IntentRollout(RolloutStatus.TASK_SOLVED, False, True, [], 0, rows, [])
    if checked is None and not isinstance(decide, TaskObservationPolicy):
        raise ValueError("Direct execution requires a policy with task-goal instructions")

    actions: list[int] = []
    steps: list[dict[str, Any]] = []
    memory = list(initial_memory)
    status = None
    error = None

    def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    def finish(status: RolloutStatus, error: str | None = None) -> IntentRollout:
        return IntentRollout(status, status == RolloutStatus.ACHIEVED, solved, actions, len(steps), rows, steps, error)

    holds: bool | None = False
    for index in range(cap):
        history = tuple(memory)
        try:
            if checked is None and isinstance(decide, TaskObservationPolicy):
                prompt = decide.render_task_action(observation, task_goal, history)
            elif checked is not None and isinstance(decide, ObservationPolicy):
                prompt = decide.render_action(observation, checked, history)
            else:
                assert checked is not None and callable(decide)
                prompt = _render_prompt(observation, checked, task_goal, history, measured=measured)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            status, event, error = _policy_failure(exc)
            emit({"type": event, "index": index, "error": error})
            break
        step: dict[str, Any] = {
            "index": index, "before": rows, "prompt": prompt, "raw_response": None,
            "action": None, "after": None, "no_op": None, "intent_holds": None,
            "preservation_holds": None, "task_solved": None, "error": None,
        }
        steps.append(step)
        emit({"type": "request", "index": index, "before": rows, "prompt": prompt})
        try:
            if checked is None and isinstance(decide, TaskObservationPolicy):
                raw = decide.choose_task_action(observation, task_goal, history)
            elif checked is not None and isinstance(decide, ObservationPolicy):
                raw = decide.choose_action(observation, checked, history)
            else:
                assert callable(decide)
                raw = decide(prompt)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            status, event, error = _policy_failure(exc)
            step["error"] = error
            emit({"type": event, "index": index, "error": error})
            break
        step["raw_response"] = raw
        emit({"type": "response", "index": index, "raw_response": raw})
        try:
            action = _action(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            step["error"] = error
            status = RolloutStatus.INVALID_RESPONSE
            break

        if before_action is not None:
            try:
                before_action()
            except BudgetExceeded as exc:
                error = str(exc)
                step["error"] = error
                status = RolloutStatus.BUDGET_EXHAUSTED
                break
        try:
            next_state = np.asarray(env.next_state(current.copy(), action))
            next_observation = observe_state(next_state, observe)
            next_solved = bool(env.is_solved(next_state))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            error = f"{type(exc).__name__}: {exc}"
            step["error"] = error
            emit({"type": "simulator_error", "index": index, "error": error})
            status = RolloutStatus.SIMULATOR_ERROR
            break
        if on_state is not None:
            step["simulator_no_op"] = np.array_equal(current, next_state)
        previous, observation = observation, next_observation
        current, solved = next_state, next_solved
        rows = list(observation.rows)
        actions.append(action)
        preserved = True if checked is None else _holds(checked["preserve"], observation)
        holds = False if checked is None else _holds(checked["effects"], observation)
        step.update(action=action, after=rows, no_op=previous == observation,
                    intent_holds=holds, preservation_holds=preserved, task_solved=solved)
        memory.append(ActionAttempt(list(Action)[action], step["no_op"]))
        if on_state is not None:
            on_state(current.copy(), tuple(memory))
        emit({"type": "action", **{key: value for key, value in step.items() if key != "prompt"}})
        status = _termination(preserved, holds, solved)
        if status is not None:
            break
    return finish(status or (RolloutStatus.UNKNOWN if holds is None else RolloutStatus.ACTION_LIMIT), error)
