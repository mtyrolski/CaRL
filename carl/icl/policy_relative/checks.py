"""Validation for measured policy-relative candidate pools."""

import json
import math
from typing import Any, cast

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.calibration import weighted_percentiles
from carl.icl.intent_data import validate_prediction
from carl.icl.policy_relative.labels import MEASUREMENT_VERSION
from carl.icl.policy_relative.labels import label_candidates
from carl.icl.policy_relative.types import AttemptStatus
from carl.icl.policy_relative.types import Candidate
from carl.icl.policy_relative.types import Context


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return cast(int, value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return cast(int, value)


def _digest_string(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _equal(actual: Any, expected: float, name: str) -> float:
    value = _number(actual, name)
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{name} disagrees with the measured attempts")
    return value


def _identity(record: dict[str, Any]) -> tuple[str, str, str]:
    values = tuple(
        record.get(name) for name in ("record_id", "context_id", "layout_id")
    )
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("record_id, context_id and layout_id must be nonempty strings")
    board, goal = record.get("start_observation"), record.get("task_goal")
    if (
        not isinstance(board, list)
        or not board
        or not all(isinstance(row, str) for row in board)
    ):
        raise ValueError("start_observation must contain board rows")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("task_goal must be a nonempty string")
    try:
        actual_layout = canonical_layout_id(from_rows(board))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid start board: {error}") from error
    if actual_layout != values[2]:
        raise ValueError("layout_id differs from the canonical wall/goal layout")
    return values  # type: ignore[return-value]


def _intent_key(intent: dict[str, Any]) -> str:
    return json.dumps(
        {
            key: sorted(json.dumps(item, sort_keys=True) for item in intent[key])
            for key in ("effects", "preserve")
        },
        sort_keys=True,
    )


def _check_versioned_attempt(
    attempt: dict[str, Any], index: int, status: AttemptStatus, count: int,
) -> None:
    if _nonnegative_int(attempt.get("index"), "attempt index") != index:
        raise ValueError("attempt indices must cover the frozen plan in order")
    _nonnegative_int(attempt.get("seed"), "attempt seed")
    if "first_hit_actions" not in attempt:
        raise ValueError("a measured attempt must record its first hit or null")
    first_hit = attempt["first_hit_actions"]
    if status == AttemptStatus.ACHIEVED:
        if _positive_int(first_hit, "first_hit_actions") != count:
            raise ValueError("first_hit_actions disagrees with the successful duration")
    elif first_hit is not None:
        raise ValueError("an unsuccessful attempt cannot record a first hit")
    solved = attempt.get("task_solved")
    if not isinstance(solved, bool):
        raise ValueError("task_solved must be a boolean")
    if (status == AttemptStatus.TASK_SOLVED and not solved
            or status in (AttemptStatus.ACTION_LIMIT, AttemptStatus.INVALID_RESPONSE) and solved):
        raise ValueError("task_solved disagrees with the terminal event")
    _nonnegative_int(attempt.get("model_calls"), "model_calls")
    metrics = attempt.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("attempt must record its metrics")
    if any(_number(value, name) < 0 for name, value in metrics.items()):
        raise ValueError("attempt metrics must be nonnegative")


def _attempts(
    measurement: dict[str, Any], cap: int, *, versioned: bool,
) -> tuple[int, list[int]]:
    attempts = measurement.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("attempts must be a nonempty list of completed attempts")
    if versioned and len(attempts) != _positive_int(measurement.get("planned_attempts"), "planned_attempts"):
        raise ValueError("complete measurements must contain every planned attempt")
    if versioned and "completed_attempts" in measurement:
        completed = _positive_int(measurement["completed_attempts"], "completed_attempts")
        if completed != len(attempts):
            raise ValueError("completed_attempts disagrees with the recorded attempts")
    successful = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise ValueError("each attempt must be an object")
        try:
            status = AttemptStatus(cast(str, attempt.get("status")))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "incomplete, infrastructure-error or unknown attempt status"
            ) from error
        if status == AttemptStatus.UNKNOWN and not versioned:
            raise ValueError("unknown attempt status requires a versioned measurement")
        count = attempt.get("primitive_actions_attempted")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= cap
        ):
            raise ValueError(
                "primitive_actions_attempted must be an integer within the cap"
            )
        if status == AttemptStatus.ACHIEVED:
            if count == 0:
                raise ValueError(
                    "a non-trivial intent cannot be achieved in zero actions"
                )
            successful.append(count)
        if status == AttemptStatus.ACTION_LIMIT and count != cap:
            raise ValueError("an action-limit attempt must have used the complete cap")
        if versioned:
            _check_versioned_attempt(attempt, index, status, count)
    return len(attempts), successful


def _check_scale_labels(observed: list[tuple[dict[str, Any], float, float | None]]) -> None:
    denominators = {
        _number(row["policy_measurements"].get("offset_denominator"), "offset_denominator")
        for row, _, _ in observed
    }
    if len(denominators) != 1:
        raise ValueError("offset denominator must match within a frozen context")
    labels = label_candidates(
        [duration for _, _, duration in observed],
        [probability for _, probability, _ in observed],
        [_positive_int(row.get("demonstration_offset"), "demonstration_offset") for row, _, _ in observed],
        offset_denominator=next(iter(denominators)),
    )
    for (row, _, _), expected in zip(observed, labels, strict=True):
        actual = row["policy_measurements"].get("scale_labels")
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("versioned measurement must contain every scale label")
        for name, value in expected.items():
            if value is not None:
                assert isinstance(value, (float, int))
                _equal(actual[name], value, name)
            elif actual[name] is not None:
                raise ValueError(f"{name} must be null for this candidate")


def _checked_context(
    rows: list[dict[str, Any]]
) -> tuple[Context | None, tuple[str, int, int | None]]:
    first = rows[0]
    first_id, context_id, layout_id = _identity(first)
    del first_id
    seen_intents: set[str] = set()
    observed: list[tuple[dict[str, Any], float, float | None]] = []
    versioned = first["policy_measurements"].get("schema_version") == MEASUREMENT_VERSION
    signature: tuple[str, int, int | None] | None = None
    for row in rows:
        _, row_context, row_layout = _identity(row)
        if (
            row_context != context_id
            or row_layout != layout_id
            or row["split"] != first["split"]
            or row["start_observation"] != first["start_observation"]
            or row["task_goal"] != first["task_goal"]
        ):
            raise ValueError(
                "all candidates in a context must share split, board, layout and task"
            )
        validation = row.get("validation")
        if not isinstance(validation, dict) or validation.get("accepted") is not True:
            raise ValueError("all calibration intents must be checked and accepted")
        intent = row.get("intent")
        errors = validate_prediction(intent, row["start_observation"])
        if errors or not isinstance(intent, dict):
            raise ValueError(f"invalid or already satisfied intent: {errors}")
        key = _intent_key(intent)
        if key in seen_intents:
            raise ValueError("duplicate candidate intent within a context")
        seen_intents.add(key)
        measurement = row.get("policy_measurements")
        if not isinstance(measurement, dict) or measurement.get("complete") is not True:
            raise ValueError("all context measurements must be complete")
        if _positive_int(
            measurement.get("expected_candidates"), "expected_candidates"
        ) != len(rows):
            raise ValueError("context does not contain every expected candidate")
        controller = _digest_string(
            measurement.get("controller_hash"), "controller_hash"
        )
        cap = _positive_int(measurement.get("action_cap"), "action_cap")
        attempts, successful = _attempts(measurement, cap, versioned=versioned)
        current = controller, cap, None if versioned else attempts
        if signature is not None and signature != current:
            raise ValueError("controller, action cap and attempt count must match")
        signature = current
        success_fraction = _equal(
            measurement.get("success_fraction"),
            len(successful) / attempts,
            "success_fraction",
        )
        if successful:
            duration = _equal(
                measurement.get("mean_success_actions"),
                sum(successful) / len(successful),
                "mean_success_actions",
            )
        else:
            duration = None
            if any(
                measurement.get(field) is not None
                for field in ("mean_success_actions", "u", "weight")
            ):
                raise ValueError(
                    "a candidate without successes must not have duration, u or weight"
                )
        observed.append((row, success_fraction, duration))

    assert signature is not None
    if versioned:
        _check_scale_labels(observed)
    reached = [
        (row, fraction, duration)
        for row, fraction, duration in observed
        if duration is not None
    ]
    if not reached:
        return None, signature
    weights, midpoints = weighted_percentiles(
        [duration for _, _, duration in reached],
        [fraction for _, fraction, _ in reached],
    )
    candidates = []
    for (row, fraction, duration), mass, midpoint in zip(reached, weights, midpoints, strict=True):
        assert duration is not None
        weight, relative_scale = float(mass), float(midpoint)
        measurement = row["policy_measurements"]
        _equal(measurement.get("weight"), weight, "weight")
        _equal(measurement.get("u"), relative_scale, "u")
        candidates.append(Candidate(row, fraction, duration, relative_scale, weight))
    return (
        Context(
            context_id,
            layout_id,
            tuple(sorted(candidates, key=lambda item: item.record["record_id"])),
        ),
        signature,
    )
