#!/usr/bin/env python3
"""Assess frozen generator cases and saved Luna rollouts without model calls.

The plan has one row per requested context, arm and scale, including invalid
and unavailable predictions. Rollouts join by the plan's unique record_id.
Missing and infrastructure-failed cases remain visible but are not failures.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_data import predicate_value
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.policy_relative_lora import Arm
from carl.icl.policy_relative_lora import PredictionStatus
from carl.icl.policy_relative_lora import assess_output

plt.switch_backend("Agg")


class CaseStatus(StrEnum):
    """Whether a planned case is available to the end-to-end analysis."""

    COMPLETE = "complete"
    INVALID_PREDICTION = "invalid_prediction"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    NOT_ATTEMPTED = "not_attempted"
    INCOMPLETE = "incomplete_attempts"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{number}")
            rows.append(value)
    return rows


def read_rollouts(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep complete saved lines if an interrupted final write is incomplete."""
    if not path.exists():
        return [], {"present": False, "sha256": None, "interrupted_tail_sha256": None, "interrupted_tail_bytes": 0}
    if not path.is_file():
        raise ValueError("Rollouts path is not a file")
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    tail = b""
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                tail = line
                break
            raise ValueError(f"Invalid saved rollout at line {index + 1}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a saved rollout object at line {index + 1}")
        rows.append(value)
    return rows, {
        "present": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "interrupted_tail_sha256": hashlib.sha256(tail).hexdigest() if tail else None,
        "interrupted_tail_bytes": len(tail)
    }


def _text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _scale(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value,
                                                 (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("requested_scale must be null or a finite number in [0, 1]")
    return float(value)


def _hash(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("controller_hash must be a lowercase SHA-256 digest")
    return value


def validate_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check unique cases, source identity and prediction-status consistency."""
    if not rows:
        raise ValueError("The frozen plan is empty")
    record_ids: set[str] = set()
    cases: set[tuple[str, str, float | None]] = set()
    sources: dict[str, tuple[str, str, str, tuple[str, ...]]] = {}
    contexts: dict[str, tuple[str, str, tuple[str, ...]]] = {}
    grouped: dict[tuple[str, float | None], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    requested_inputs: dict[tuple[str, float | None], float | None] = {}
    checked = []
    for row in rows:
        record_id, source_id = _text(row, "record_id"), _text(row, "source_record_id")
        context_id, layout_id, family = (_text(row, key) for key in ("context_id", "layout_id", "family"))
        arm_value = _text(row, "arm")
        arm = "R" if arm_value == "R" else Arm(arm_value).value
        scale = _scale(row.get("requested_scale"))
        if arm == Arm.SCALED and scale is None:
            raise ValueError("The scaled arm requires an explicit requested scale")
        if arm in {"B", "D", "N", "T", "U", "R"}:
            if "requested_input" not in row or scale is None:
                raise ValueError("Matrix cases require a declared input and pairing tier")
            requested = row["requested_input"]
            if arm in {"B", "D", "N", "R"}:
                if requested is not None:
                    raise ValueError("Unscaled arms must not receive a requested scale")
            elif (isinstance(requested, bool) or not isinstance(requested, (int, float))
                  or not math.isfinite(requested)
                  or (arm == "T" and requested <= 0)
                  or (arm == "U" and not 0 <= requested <= 1)):
                raise ValueError("T requires a positive duration and U a relative scale in [0, 1]")
            input_key = (arm, scale)
            if input_key in requested_inputs and requested_inputs[input_key] != requested:
                raise ValueError("Requested input changes within an arm and pairing tier")
            requested_inputs[input_key] = requested
        status = PredictionStatus(row.get("prediction_status"))
        if record_id in record_ids or (source_id, arm, scale) in cases:
            raise ValueError("Duplicate planned record or source/arm/scale")
        record_ids.add(record_id)
        cases.add((source_id, arm, scale))
        board = row.get("start_observation")
        if not isinstance(board, list) or canonical_layout_id(from_rows(board)) != layout_id:
            raise ValueError("layout_id differs from the canonical start board")
        identity = (context_id, layout_id, family, tuple(board))
        if source_id in sources and sources[source_id] != identity:
            raise ValueError("A source record changes context, board or family across arms")
        sources[source_id] = identity
        context_identity = (layout_id, family, tuple(board))
        if context_id in contexts and contexts[context_id] != context_identity:
            raise ValueError("A context ID refers to different boards")
        contexts[context_id] = context_identity
        errors = row.get("validation_errors")
        if not isinstance(errors, list) or not all(isinstance(error, str) for error in errors):
            raise ValueError("validation_errors must be a list of strings")
        parsed, json_valid = row.get("parsed"), row.get("json_valid")
        if status in {PredictionStatus.VALID, PredictionStatus.INVALID_INTENT, PredictionStatus.INVALID_JSON}:
            raw = row.get("raw_output")
            if not isinstance(raw, str):
                raise ValueError("An assessed prediction needs its raw output")
            # The current D/N/T/U/R study permits one enclosing Markdown
            # fence. The legacy unscaled/scaled comparison stays strict.
            actual = assess_output(raw, board, normalize=arm not in (Arm.UNSCALED, Arm.SCALED))
            if (actual["status"] != status or actual["json_valid"] is not json_valid
                    or json.dumps(actual["parsed"], sort_keys=True) != json.dumps(parsed, sort_keys=True)
                    or bool(actual["validation_errors"]) != bool(errors)):
                raise ValueError("Saved prediction status or parse disagrees with the current-board checker")
        elif parsed is not None or json_valid is not None or row.get("raw_output") is not None:
            raise ValueError("Unavailable predictions cannot claim a parsed result")
        grouped[(family, scale)][arm].add(source_id)
        checked.append({**row, "arm": arm, "requested_scale": scale, "prediction_status": status.value})
    all_arms = set().union(*(set(arms) for arms in grouped.values()))
    if len(all_arms) < 2:
        raise ValueError("The frozen plan must include both arms, with missing predictions explicit")
    for arms in grouped.values():
        if set(arms) != all_arms:
            raise ValueError("The frozen plan must include both arms, with missing predictions explicit")
        populations = list(arms.values())
        if len(populations) > 1 and any(population != populations[0] for population in populations[1:]):
            raise ValueError("Arms must have the same planned contexts within each family and scale")
    for arm in ("T", "U"):
        values = sorted((tier, value) for (candidate, tier), value in requested_inputs.items()
                        if candidate == arm and tier is not None and value is not None)
        if values and (len(values) != 2 or values[0][1] >= values[1][1]):
            raise ValueError("Scaled-arm inputs must be distinct and increase with the tier")
    return checked


def validate_rollout(row: dict[str, Any], plan: dict[str, Any], controller_hash: str, action_cap: int,
                     expected_attempts: int) -> dict[str, Any]:
    """Replay saved actions and reject inconsistent first-hit or terminal statuses."""
    if row.get("context_id") != plan["context_id"] or row.get("controller_hash") != controller_hash:
        raise ValueError("Rollout context or controller hash differs from the frozen plan")
    attempt = _integer(row.get("attempt"), "attempt", 1)
    if attempt > expected_attempts:
        raise ValueError("Rollout attempt exceeds the frozen attempt count")
    result = row.get("result")
    if not isinstance(result, dict):
        raise ValueError("Rollout result must be an object")
    status = RolloutStatus(result.get("status"))
    if status == RolloutStatus.INVALID_INTENT:
        raise ValueError("Controller rejected an intent marked valid by the same checker")
    actions, steps = result.get("actions"), result.get("steps")
    calls = _integer(result.get("model_calls"), "model_calls")
    if not isinstance(actions, list) or len(actions) > action_cap:
        raise ValueError("Invalid action list or action cap exceeded")
    if not isinstance(steps, list) or len(steps) != calls:
        raise ValueError("Saved steps do not match the model-call count")
    extra_call = status in {RolloutStatus.API_ERROR, RolloutStatus.OUTPUT_LIMIT, RolloutStatus.INVALID_RESPONSE}
    if calls != len(actions) + int(extra_call):
        raise ValueError("Action and model-call counts disagree with the rollout status")
    duration = row.get("duration_seconds")
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                                 or not math.isfinite(duration) or duration < 0):
        raise ValueError("duration_seconds must be non-negative or null")

    initial = from_rows(plan["start_observation"])
    env = environment(initial)
    current = initial
    board = to_rows(current)
    initial_on_goals = sum(line.count("*") for line in board)
    intent = plan["parsed"]
    terminal = RolloutStatus.TASK_SOLVED if env.is_solved(current) else None
    for index, action in enumerate(actions):
        if terminal is not None or _integer(action, "action") not in range(4):
            raise ValueError("Invalid action or an action after a stopping condition")
        step = steps[index]
        if not isinstance(step, dict) or step.get("index") != index or step.get("before") != board:
            raise ValueError("Step index or start board differs from replay")
        following = np.asarray(env.next_state(current, action))
        after = to_rows(following)
        preserved = all(predicate_value(predicate, after) for predicate in intent["preserve"])
        holds = all(predicate_value(predicate, after) for predicate in intent["effects"])
        solved = bool(env.is_solved(following))
        if (step.get("action") != action or step.get("after") != after
                or step.get("no_op") is not bool(np.array_equal(current, following))
                or step.get("preservation_holds") is not preserved or step.get("intent_holds") is not holds
                or step.get("task_solved") is not solved):
            raise ValueError("Saved step differs from the simulator or predicate checks")
        terminal = (RolloutStatus.PRESERVATION_VIOLATED if not preserved else
                    RolloutStatus.ACHIEVED if holds else RolloutStatus.TASK_SOLVED if solved else None)
        current, board = following, after
    if extra_call:
        step = steps[-1]
        if (terminal is not None or len(actions) >= action_cap or not isinstance(step, dict)
                or step.get("index") != len(actions) or step.get("before") != board or step.get("action") is not None
                or step.get("after") is not None):
            raise ValueError("Failed model call is inconsistent with the replay")
        expected = status
    else:
        expected = terminal if terminal is not None else RolloutStatus.ACTION_LIMIT if len(
            actions) == action_cap else None
    if (status != expected or result.get("endpoint") != board
            or result.get("achieved") is not (status == RolloutStatus.ACHIEVED)
            or result.get("task_solved") is not bool(env.is_solved(current))):
        raise ValueError("Rollout status, endpoint or completion flags disagree with replay")
    return {
        "attempt": attempt,
        "status": status.value,
        "actions": len(actions),
        "no_ops": sum(step["no_op"] for step in steps[:len(actions)]),
        "model_calls": calls,
        "duration_seconds": duration,
        "task_solved": result["task_solved"],
        "initial_boxes_on_goals": initial_on_goals,
        "final_boxes_on_goals": sum(line.count("*") for line in board),
        "delta_boxes_on_goals": sum(line.count("*") for line in board) - initial_on_goals
    }


def prepare_cases(plan: list[dict[str, Any]], rollouts: list[dict[str, Any]], *, controller_hash: str, action_cap: int,
                  expected_attempts: int) -> list[dict[str, Any]]:
    _hash(controller_hash)
    _integer(action_cap, "action_cap", 1)
    _integer(expected_attempts, "expected_attempts", 1)
    plan = validate_plan(plan)
    by_id = {row["record_id"]: row for row in plan}
    attempts: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for rollout in rollouts:
        record_id = _text(rollout, "record_id")
        if record_id not in by_id:
            raise ValueError("Rollout is not in the frozen plan")
        source = by_id[record_id]
        if source["prediction_status"] != PredictionStatus.VALID:
            raise ValueError("Only a valid prediction may have a rollout")
        item = validate_rollout(rollout, source, controller_hash, action_cap, expected_attempts)
        if item["attempt"] in attempts[record_id]:
            raise ValueError("Duplicate physical attempt")
        attempts[record_id][item["attempt"]] = item
    cases = []
    for row in plan:
        status = PredictionStatus(row["prediction_status"])
        observed = list(attempts[row["record_id"]].values())
        if status in {PredictionStatus.INVALID_JSON, PredictionStatus.INVALID_INTENT}:
            case_status = CaseStatus.INVALID_PREDICTION
        elif status == PredictionStatus.INFRASTRUCTURE_ERROR or any(
                a["status"] == RolloutStatus.API_ERROR for a in observed):
            case_status = CaseStatus.INFRASTRUCTURE_ERROR
        elif any(a["status"] == RolloutStatus.OUTPUT_LIMIT for a in observed):
            case_status = CaseStatus.INCOMPLETE
        elif status == PredictionStatus.NOT_ATTEMPTED or not observed:
            case_status = CaseStatus.NOT_ATTEMPTED
        elif len(observed) < expected_attempts:
            case_status = CaseStatus.INCOMPLETE
        else:
            case_status = CaseStatus.COMPLETE
        cases.append({
            key: row[key] for key in ("record_id", "source_record_id", "context_id", "layout_id", "family", "arm",
                                     "requested_scale", "prediction_status")
        } | {
            "requested_input":
                row.get("requested_input"),
            "local_rule_work":
                row.get("local_rule_work"),
            "case_status":
                case_status.value,
            "expected_attempts":
                expected_attempts,
            "attempts":
                observed,
            "effect_predicate_types":
                sorted({predicate["predicate"]
                        for predicate in row["parsed"]["effects"]}) if status == PredictionStatus.VALID else None,
            "effect_predicate_count":
                len(row["parsed"]["effects"]) if status == PredictionStatus.VALID else None,
            "preserve_predicate_count":
                len(row["parsed"]["preserve"]) if status == PredictionStatus.VALID else None,
        })
    return cases


def _observed_score(case: dict[str, Any], budget: int) -> float | None:
    if case["case_status"] == CaseStatus.INVALID_PREDICTION:
        return 0.0
    if case["case_status"] != CaseStatus.COMPLETE:
        return None
    attempts = case["attempts"]
    return sum(
        item["status"] == RolloutStatus.ACHIEVED and item["actions"] <= budget for item in attempts) / len(attempts)


def _observed_cost(case: dict[str, Any], field: str) -> float | None:
    if case["case_status"] == CaseStatus.INVALID_PREDICTION:
        return 0.0
    if case["case_status"] != CaseStatus.COMPLETE:
        return None
    return sum(item[field] for item in case["attempts"]) / len(case["attempts"])


def _observed_goal_delta(case: dict[str, Any]) -> float | None:
    """A goal change exists only for actual, completely observed rollouts."""
    if case["case_status"] != CaseStatus.COMPLETE:
        return None
    return sum(item["delta_boxes_on_goals"] for item in case["attempts"]) / len(case["attempts"])


def _mean_goal_delta(attempts: list[dict[str, Any]]) -> float | None:
    return sum(item["delta_boxes_on_goals"] for item in attempts) / len(attempts) if attempts else None


def _board_estimate(values: dict[str, list[float]], *, seed: int,
                    samples: int) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    board_values = np.array([sum(values[board]) / len(values[board]) for board in sorted(values)])
    mean = float(board_values.mean())
    if len(board_values) < 5:
        return mean, None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(board_values), size=(samples, len(board_values)))
    low, high = np.percentile(board_values[draws].mean(axis=1), [2.5, 97.5])
    return mean, float(low), float(high)


def summarise(cases: list[dict[str, Any]],
              *,
              action_cap: int,
              bootstrap_seed: int = 20261114,
              bootstrap_samples: int = 2000) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    _integer(action_cap, "action_cap", 1)
    _integer(bootstrap_seed, "bootstrap_seed")
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    groups: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[(case["family"], case["arm"], case["requested_scale"])].append(case)
    summaries, curves = [], []
    for family, arm, scale in sorted(groups, key=lambda key: (key[0], key[1], -1 if key[2] is None else key[2])):
        rows = groups[(family, arm, scale)]
        statuses = Counter(row["case_status"] for row in rows)
        prediction_statuses = Counter(row["prediction_status"] for row in rows)
        attempts = [item for row in rows for item in row["attempts"]]
        complete_attempts = [
            item for row in rows if row["case_status"] == CaseStatus.COMPLETE for item in row["attempts"]
        ]
        successful = [item for item in complete_attempts if item["status"] == RolloutStatus.ACHIEVED]
        observed_attempts = [item for item in attempts
                             if item["status"] not in {RolloutStatus.API_ERROR, RolloutStatus.OUTPUT_LIMIT}]
        saved_successful = [item for item in observed_attempts if item["status"] == RolloutStatus.ACHIEVED]
        infrastructure_attempts = [item for item in attempts if item["status"] == RolloutStatus.API_ERROR]
        output_limit_attempts = [item for item in attempts if item["status"] == RolloutStatus.OUTPUT_LIMIT]
        valid_intents = [row for row in rows if row["prediction_status"] == PredictionStatus.VALID]
        intent_types = Counter(" & ".join(row["effect_predicate_types"]) for row in valid_intents)
        effect_sizes = Counter(str(row["effect_predicate_count"]) for row in valid_intents)
        preserve_sizes = Counter(str(row["preserve_predicate_count"]) for row in valid_intents)
        goal_increases = sum(item["delta_boxes_on_goals"] > 0 for item in observed_attempts)
        successful_goal_increases = sum(item["delta_boxes_on_goals"] > 0 for item in saved_successful)
        scores = [_observed_score(row, action_cap) for row in rows]
        evaluated = [score for score in scores if score is not None]
        board_scores: dict[str, list[float]] = defaultdict(list)
        for row, score in zip(rows, scores, strict=True):
            if score is not None:
                board_scores[row["layout_id"]].append(score)
        mean, low, high = _board_estimate(board_scores, seed=bootstrap_seed, samples=bootstrap_samples)
        board_action_costs: dict[str, list[float]] = defaultdict(list)
        board_call_costs: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            actions, calls = _observed_cost(row, "actions"), _observed_cost(row, "model_calls")
            if actions is not None and calls is not None:
                board_action_costs[row["layout_id"]].append(actions)
                board_call_costs[row["layout_id"]].append(calls)
        action_mean, action_low, action_high = _board_estimate(board_action_costs,
                                                               seed=bootstrap_seed,
                                                               samples=bootstrap_samples)
        call_mean, call_low, call_high = _board_estimate(board_call_costs,
                                                         seed=bootstrap_seed,
                                                         samples=bootstrap_samples)
        board_goal_deltas: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            delta = _observed_goal_delta(row)
            if delta is not None:
                board_goal_deltas[row["layout_id"]].append(delta)
        goal_mean, goal_low, goal_high = _board_estimate(board_goal_deltas,
                                                         seed=bootstrap_seed,
                                                         samples=bootstrap_samples)
        durations = [item["duration_seconds"] for item in attempts if item["duration_seconds"] is not None]
        summaries.append({
            "family":
                family,
            "arm":
                arm,
            "requested_scale":
                scale,
            "requested_input":
                rows[0]["requested_input"],
            "planned_cases":
                len(rows),
            "planned_boards":
                len({row["layout_id"] for row in rows}),
            "assessed_cases":
                len(evaluated),
            "assessed_boards":
                len(board_scores),
            "valid_predictions":
                prediction_statuses[PredictionStatus.VALID],
            "intent_type_counts":
                dict(sorted(intent_types.items())),
            "effect_predicate_count_distribution":
                dict(sorted(effect_sizes.items())),
            "preserve_predicate_count_distribution":
                dict(sorted(preserve_sizes.items())),
            "invalid_json":
                prediction_statuses[PredictionStatus.INVALID_JSON],
            "invalid_intent":
                prediction_statuses[PredictionStatus.INVALID_INTENT],
            "prediction_infrastructure_errors":
                prediction_statuses[PredictionStatus.INFRASTRUCTURE_ERROR],
            "prediction_not_attempted":
                prediction_statuses[PredictionStatus.NOT_ATTEMPTED],
            "complete_policy_cases":
                statuses[CaseStatus.COMPLETE],
            "infrastructure_cases":
                statuses[CaseStatus.INFRASTRUCTURE_ERROR],
            "rollout_infrastructure_cases":
                sum(row["prediction_status"] == PredictionStatus.VALID
                    and row["case_status"] == CaseStatus.INFRASTRUCTURE_ERROR for row in rows),
            "not_attempted_cases":
                statuses[CaseStatus.NOT_ATTEMPTED],
            "rollout_not_started_cases":
                sum(row["prediction_status"] == PredictionStatus.VALID and not row["attempts"] for row in rows),
            "incomplete_cases":
                statuses[CaseStatus.INCOMPLETE],
            "eligible_policy_attempts":
                sum(row["expected_attempts"] for row in rows if row["prediction_status"] == PredictionStatus.VALID),
            "saved_attempts":
                len(attempts),
            "complete_attempts":
                len(complete_attempts),
            "saved_successful_attempts":
                sum(a["status"] == RolloutStatus.ACHIEVED for a in attempts),
            "successful_attempts":
                len(successful),
            "conditional_achievement_given_valid_completed":
                len(successful) / len(complete_attempts) if complete_attempts else None,
            "all_saved_primitive_actions":
                sum(a["actions"] for a in attempts),
            "all_saved_no_ops":
                sum(a["no_ops"] for a in attempts),
            "all_saved_model_calls":
                sum(a["model_calls"] for a in attempts),
            "complete_attempts_no_ops":
                sum(a["no_ops"] for a in complete_attempts),
            "infrastructure_attempts":
                len(infrastructure_attempts),
            "output_limit_attempts":
                len(output_limit_attempts),
            "infrastructure_primitive_actions":
                sum(a["actions"] for a in infrastructure_attempts),
            "infrastructure_no_ops":
                sum(a["no_ops"] for a in infrastructure_attempts),
            "infrastructure_model_calls":
                sum(a["model_calls"] for a in infrastructure_attempts),
            "complete_task_solved_attempts":
                sum(a["task_solved"] for a in complete_attempts),
            "observed_endpoint_attempts":
                len(observed_attempts),
            "observed_endpoint_goal_increases":
                goal_increases,
            "observed_endpoint_goal_increase_fraction":
                goal_increases / len(observed_attempts) if observed_attempts else None,
            "observed_endpoint_goal_unchanged":
                sum(a["delta_boxes_on_goals"] == 0 for a in observed_attempts),
            "observed_endpoint_goal_decreases":
                sum(a["delta_boxes_on_goals"] < 0 for a in observed_attempts),
            "observed_endpoint_goal_delta_mean":
                _mean_goal_delta(observed_attempts),
            "successful_endpoint_attempts":
                len(saved_successful),
            "successful_endpoint_goal_increases":
                successful_goal_increases,
            "successful_endpoint_goal_increase_fraction":
                successful_goal_increases / len(saved_successful) if saved_successful else None,
            "successful_endpoint_goal_unchanged":
                sum(a["delta_boxes_on_goals"] == 0 for a in saved_successful),
            "successful_endpoint_goal_decreases":
                sum(a["delta_boxes_on_goals"] < 0 for a in saved_successful),
            "successful_endpoint_goal_delta_mean":
                _mean_goal_delta(saved_successful),
            "infrastructure_endpoint_goal_increases":
                sum(a["delta_boxes_on_goals"] > 0 for a in infrastructure_attempts),
            "infrastructure_endpoint_goal_unchanged":
                sum(a["delta_boxes_on_goals"] == 0 for a in infrastructure_attempts),
            "infrastructure_endpoint_goal_decreases":
                sum(a["delta_boxes_on_goals"] < 0 for a in infrastructure_attempts),
            "infrastructure_endpoint_goal_delta_mean":
                _mean_goal_delta(infrastructure_attempts),
            "known_duration_seconds":
                sum(durations) if durations else None,
            "duration_missing_attempts":
                len(attempts) - len(durations),
            "successful_actions_mean":
                sum(a["actions"] for a in successful) / len(successful) if successful else None,
            "pooled_assessed_fraction":
                sum(evaluated) / len(evaluated) if evaluated else None,
            "assessed_fraction_of_planned":
                len(evaluated) / len(rows),
            "observed_success_equivalent":
                sum(evaluated),
            "planned_fraction_lower_bound":
                sum(evaluated) / len(rows),
            "planned_fraction_upper_bound": (sum(evaluated) + len(rows) - len(evaluated)) / len(rows),
            "board_mean":
                mean,
            "bootstrap_low":
                low,
            "bootstrap_high":
                high,
            "board_mean_actions":
                action_mean,
            "actions_bootstrap_low":
                action_low,
            "actions_bootstrap_high":
                action_high,
            "board_mean_model_calls":
                call_mean,
            "calls_bootstrap_low":
                call_low,
            "calls_bootstrap_high":
                call_high,
            "goal_delta_observed_boards":
                len(board_goal_deltas),
            "board_mean_goal_delta":
                goal_mean,
            "goal_delta_bootstrap_low":
                goal_low,
            "goal_delta_bootstrap_high":
                goal_high
        })
        for budget in range(action_cap + 1):
            board_scores: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                score = _observed_score(row, budget)
                if score is not None:
                    board_scores[row["layout_id"]].append(score)
            mean, low, high = _board_estimate(board_scores, seed=bootstrap_seed, samples=bootstrap_samples)
            curves.append({
                "family": family,
                "arm": arm,
                "requested_scale": scale,
                "requested_input": rows[0]["requested_input"],
                "budget": budget,
                "planned_cases": len(rows),
                "assessed_cases": sum(len(v) for v in board_scores.values()),
                "assessed_boards": len(board_scores),
                "board_mean": mean,
                "bootstrap_low": low,
                "bootstrap_high": high
            })
    paired = []
    for family, scale in sorted({(key[0], key[2]) for key in groups},
                                key=lambda key: (key[0], -1 if key[1] is None else key[1])):
        arms = sorted((key[1] for key in groups if key[0] == family and key[2] == scale),
                      key=lambda arm: (arm != Arm.UNSCALED, arm))
        for left, right in itertools.combinations(arms, 2):
            lhs = {row["source_record_id"]: row for row in groups[(family, left, scale)]}
            rhs = {row["source_record_id"]: row for row in groups[(family, right, scale)]}
            differences: dict[str, list[float]] = defaultdict(list)
            action_differences: dict[str, list[float]] = defaultdict(list)
            call_differences: dict[str, list[float]] = defaultdict(list)
            goal_differences: dict[str, list[float]] = defaultdict(list)
            available = Counter()
            for source_id in sorted(lhs.keys() | rhs.keys()):
                lrow, rrow = lhs.get(source_id), rhs.get(source_id)
                lscore = _observed_score(lrow, action_cap) if lrow else None
                rscore = _observed_score(rrow, action_cap) if rrow else None
                if lscore is not None and rscore is not None:
                    assert lrow is not None and rrow is not None
                    if lrow["layout_id"] != rrow["layout_id"]:
                        raise ValueError("Paired cases have different layouts")
                    differences[lrow["layout_id"]].append(rscore - lscore)
                    left_actions, right_actions = _observed_cost(lrow, "actions"), _observed_cost(rrow, "actions")
                    left_calls, right_calls = _observed_cost(lrow, "model_calls"), _observed_cost(rrow, "model_calls")
                    assert left_actions is not None and right_actions is not None
                    assert left_calls is not None and right_calls is not None
                    action_differences[lrow["layout_id"]].append(right_actions - left_actions)
                    call_differences[lrow["layout_id"]].append(right_calls - left_calls)
                    left_goal, right_goal = _observed_goal_delta(lrow), _observed_goal_delta(rrow)
                    if left_goal is not None and right_goal is not None:
                        goal_differences[lrow["layout_id"]].append(right_goal - left_goal)
                    available["paired"] += 1
                elif lscore is not None:
                    available["left_only"] += 1
                elif rscore is not None:
                    available["right_only"] += 1
                else:
                    available["neither"] += 1
            mean, low, high = _board_estimate(differences, seed=bootstrap_seed, samples=bootstrap_samples)
            action_mean, action_low, action_high = _board_estimate(action_differences,
                                                                   seed=bootstrap_seed,
                                                                   samples=bootstrap_samples)
            call_mean, call_low, call_high = _board_estimate(call_differences,
                                                             seed=bootstrap_seed,
                                                             samples=bootstrap_samples)
            goal_mean, goal_low, goal_high = _board_estimate(goal_differences,
                                                             seed=bootstrap_seed,
                                                             samples=bootstrap_samples)
            paired.append({
                "family": family,
                "requested_scale": scale,
                "left_arm": left,
                "right_arm": right,
                "planned_contexts": len(lhs.keys() | rhs.keys()),
                "paired_contexts": available["paired"],
                "paired_boards": len(differences),
                "left_only": available["left_only"],
                "right_only": available["right_only"],
                "neither": available["neither"],
                "right_minus_left_board_mean": mean,
                "bootstrap_low": low,
                "bootstrap_high": high,
                "right_minus_left_actions_board_mean": action_mean,
                "actions_bootstrap_low": action_low,
                "actions_bootstrap_high": action_high,
                "right_minus_left_calls_board_mean": call_mean,
                "calls_bootstrap_low": call_low,
                "calls_bootstrap_high": call_high,
                "goal_delta_paired_contexts": sum(len(values) for values in goal_differences.values()),
                "goal_delta_paired_boards": len(goal_differences),
                "right_minus_left_goal_delta_board_mean": goal_mean,
                "goal_delta_bootstrap_low": goal_low,
                "goal_delta_bootstrap_high": goal_high
            })
    return summaries, curves, paired


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def write_results(output: Path, cases: list[dict[str, Any]], summary: list[dict[str, Any]],
                  curves: list[dict[str, Any]], paired: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    """Save processed values and an unsmoothed, empirical budget figure."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in cases:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (output / "summary.json").write_text(
        json.dumps({
            "metadata": metadata,
            "summary": summary,
            "paired": paired
        }, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    _write_csv(output / "summary.csv", summary, list(summary[0]))
    _write_csv(output / "budget_curves.csv", curves, list(curves[0]))
    distribution_fields = ["family", "arm", "requested_scale", "distribution", "value", "count", "valid_intents"]
    distributions = []
    for item in summary:
        for name in ("intent_type_counts", "effect_predicate_count_distribution",
                     "preserve_predicate_count_distribution"):
            for value, count in item[name].items():
                distributions.append({
                    "family": item["family"],
                    "arm": item["arm"],
                    "requested_scale": item["requested_scale"],
                    "distribution": name,
                    "value": value,
                    "count": count,
                    "valid_intents": item["valid_predictions"]
                })
    _write_csv(output / "intent_distribution.csv", distributions, distribution_fields)
    paired_fields = [
        "family", "requested_scale", "left_arm", "right_arm", "planned_contexts", "paired_contexts", "paired_boards",
        "left_only", "right_only", "neither", "right_minus_left_board_mean", "bootstrap_low", "bootstrap_high",
        "right_minus_left_actions_board_mean", "actions_bootstrap_low", "actions_bootstrap_high",
        "right_minus_left_calls_board_mean", "calls_bootstrap_low", "calls_bootstrap_high",
        "goal_delta_paired_contexts", "goal_delta_paired_boards", "right_minus_left_goal_delta_board_mean",
        "goal_delta_bootstrap_low", "goal_delta_bootstrap_high"
    ]
    _write_csv(output / "paired.csv", paired, paired_fields)
    families = sorted({row["family"] for row in curves})
    with plt.rc_context({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4), squeeze=False, sharey=True)
        for axis, family in zip(axes[0], families, strict=True):
            family_rows = [row for row in curves if row["family"] == family]
            keys = sorted({(row["arm"], row["requested_scale"]) for row in family_rows},
                          key=lambda key: (key[0], -1 if key[1] is None else key[1]))
            legend_items = []
            for arm, scale in keys:
                matching = [row for row in family_rows if row["arm"] == arm and row["requested_scale"] == scale]
                points = [row for row in matching if row["board_mean"] is not None]
                if arm in {"B", "D", "N", "T", "U", "R"}:
                    requested = matching[0]["requested_input"]
                    if arm == "T":
                        description = f"T, d={requested:g} (tier {scale:g})"
                    elif arm == "U":
                        description = f"U, u={requested:g} (tier {scale:g})"
                    else:
                        description = f"{arm}, tier={scale:g}"
                else:
                    description = (f"{arm}, paired u={scale:g}" if arm == Arm.UNSCALED else f"{arm}, u={scale:g}")
                if not points:
                    label = f"{description} (0/{matching[-1]['planned_cases']}; unavailable)"
                    legend_items.append(Line2D([], [], color="0.55", marker="x", linestyle="none", label=label))
                    continue
                label = f"{description} ({points[-1]['assessed_cases']}/{points[-1]['planned_cases']})"
                line, = axis.step([row["budget"] for row in points], [row["board_mean"] for row in points],
                                  where="post",
                                  label=label)
                legend_items.append(line)
                if all(row["bootstrap_low"] is not None and row["bootstrap_high"] is not None for row in points):
                    axis.fill_between([row["budget"] for row in points], [row["bootstrap_low"] for row in points],
                                      [row["bootstrap_high"] for row in points],
                                      color=line.get_color(),
                                      alpha=0.12,
                                      linewidth=0,
                                      step="post")
            axis.set_title(family)
            axis.set_xlabel("Primitive-action budget")
            axis.set_ylim(0, 1.02)
            if legend_items:
                axis.legend(handles=legend_items, fontsize="small")
            if not axis.lines:
                axis.text(0.5, 0.5, "No assessed cases", transform=axis.transAxes, ha="center")
        axes[0][0].set_ylabel("Board-mean intent achievement")
        fig.suptitle("Assessed cases / planned cases shown in legend", fontsize="small")
        fig.tight_layout()
        for extension in ("pdf", "svg", "png"):
            fig.savefig(output / f"intent_achievement.{extension}", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-hash", required=True)
    parser.add_argument("--action-cap", type=int, required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--bootstrap-seed", type=int, default=20261114)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    rollouts, rollout_file = read_rollouts(args.rollouts)
    cases = prepare_cases(_read_jsonl(args.plan),
                          rollouts,
                          controller_hash=args.controller_hash,
                          action_cap=args.action_cap,
                          expected_attempts=args.attempts)
    summary, curves, paired = summarise(cases,
                                        action_cap=args.action_cap,
                                        bootstrap_seed=args.bootstrap_seed,
                                        bootstrap_samples=args.bootstrap_samples)
    metadata = {
        "plan_sha256": file_sha256(args.plan),
        "rollouts_file": rollout_file,
        "controller_hash": args.controller_hash,
        "action_cap": args.action_cap,
        "attempts": args.attempts,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "uncertainty": "Paired or unpaired board bootstrap; fewer than five boards is descriptive only.",
        "scale": "Matrix tiers are pairing labels: B/D/N/R receive no scale, T receives the recorded absolute "
                 "duration and U the recorded relative scale. Legacy unscaled/scaled plans can reuse one unscaled "
                 "prediction across scales; consult the plan manifest.",
        "conditional_achievement": "Successful divided by complete attempts on valid predictions; "
                                   "incomplete and infrastructure-failed cases are excluded.",
        "intent_distributions":
            "Count valid predictions by conjunction of effect predicate types and by the "
            "numbers of effect and preservation predicates; invalid or missing outputs are excluded.",
        "goal_delta": "Final boxes on goals minus initial boxes on goals, measured by simulator replay. "
                      "Saved endpoints excluding API errors and output limits, and achieved endpoints, are "
                      "reported separately. Infrastructure-failed partial endpoints are separate; "
                      "a positive change does not establish "
                      "complete puzzle success or long-term usefulness.",
        "missing": "Unavailable or incomplete cases are excluded from measured fractions, not counted as failures. "
                   "Planned-fraction bounds assign all unknown cases success 0 or 1; they are bounds, not estimates.",
        "costs": "Primitive actions and model calls include saved partial infrastructure-failed attempts. "
                 "Board and paired mean costs include invalid predictions at zero controller cost and exclude "
                 "unavailable cases. "
                 "An interrupted unsaved attempt can incur extra work; consult events and the budget ledger. "
                 "Provider billing is not inferred."
    }
    write_results(args.output, cases, summary, curves, paired, metadata)
    print(json.dumps({"planned_cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
