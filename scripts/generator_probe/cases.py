"""Capture model proposals and execute only the production CLLP rollout."""

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np

from carl.icl.api import BudgetExceeded, RequestFailed, request_cache_digest, write_json
from carl.icl.boards import from_rows, state_hash, to_rows, validate_subgoal
from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.components import ICLComponents
from carl.icl.search import rollout
from carl.icl.trace import RecordingEnvironment, RecordingPolicy
from carl.icl.types import Board, Instance, JSONObject, MockResponse, Provider, TrainingData

TAG: Final = "generator_probe_v2"
ACTIONS: Final = ("UP", "DOWN", "LEFT", "RIGHT")


class CapturedAPI(ExperimentAPI):
    """Keep every raw response, including rejected proposals and JSON retries."""

    def __init__(self, config: ExperimentConfig, provider: Provider, tag: str) -> None:
        super().__init__(config.path("cache_dir"), provider, config.limits)
        self.budget_tag = f"{TAG}:{tag}"
        self.calls: list[JSONObject] = []

    def call(
        self, endpoint: str, payload: JSONObject, mock: MockResponse | None = None
    ) -> JSONObject:
        record: JSONObject = {"endpoint": endpoint, "payload": copy.deepcopy(payload)}
        self.calls.append(record)
        try:
            response = super().call(endpoint, payload, mock)
        except (RequestFailed, BudgetExceeded) as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            cached = self.cache_dir / f"{request_cache_digest({'endpoint': endpoint, 'payload': payload})}.json"
            if cached.exists():
                record["response"] = json.loads(cached.read_text())
            raise
        record["response"] = response
        return response


def classify(calls: list[JSONObject], source: Board, layout_schema: bool) -> list[JSONObject]:
    """Classify before filtering; no inference or simulator is used here."""
    observations: list[JSONObject] = []
    for call in calls:
        row: JSONObject = {"parse_ok": False, "empty": None, "candidates": []}
        observations.append(row)
        try:
            decoded = json.loads(call.get("response", {}).get("output_text", ""))
            items = decoded["subgoals"]
            if not isinstance(items, list):
                raise ValueError("Expected subgoals array")
        except (ValueError, TypeError, KeyError):
            continue
        row.update(parse_ok=True, empty=not items)
        seen: set[str] = set()
        for item in items:
            candidate: JSONObject = {"static_valid": False, "source_copy": None, "duplicate": None}
            row["candidates"].append(candidate)
            try:
                rows = ([item[f"row_{index:02d}"] for index in range(source.shape[0])]
                        if layout_schema else item["rows"])
                board = from_rows(rows)
                validate_subgoal(source, board)
                digest = state_hash(board)
                candidate.update(static_valid=True, source_copy=np.array_equal(source, board),
                                 duplicate=digest in seen, state_hash=digest)
                seen.add(digest)
            except (ValueError, TypeError, KeyError) as error:
                candidate["reason"] = str(error)
    return observations


def execute_target(
    components: ICLComponents, source: Board, target: Board, seed: int,
    on_progress: Callable[[JSONObject], None] | None = None,
) -> JSONObject:
    """Use the production controller and independently replay its recorded actions."""
    policy = RecordingPolicy(components)
    recorded = RecordingEnvironment(components.env, policy)
    result: JSONObject = {"exact_hit": False, "task_solved": False, "first_hit": None, "error": None}
    progress: list[JSONObject] = []

    def persist_action(_before: Board, action: int, after: Board) -> None:
        step = recorded.steps[-1]
        progress.append({"action_index": action, "probabilities": step.probabilities.tolist(),
                         "endpoint": to_rows(after), "no_op": not step.changed})
        if on_progress is not None:
            on_progress({"steps": progress, "actions": len(progress), "actual_endpoint": to_rows(after)})

    try:
        attempt = rollout(recorded, policy, source, target, cap=components.config.calibration.action_cap,
                          rng=np.random.default_rng(seed), on_action=persist_action)
        result.update(exact_hit=attempt.reached, task_solved=attempt.solved,
                      first_hit=len(attempt.actions) if attempt.reached else None)
    except (RequestFailed, BudgetExceeded) as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    replay = source.copy()
    replay_env = environment(source)
    seen = {state_hash(source)}
    steps: list[JSONObject] = []
    for step in recorded.steps:
        replay = np.asarray(replay_env.next_state(replay, step.action), dtype=np.float32)
        if not np.array_equal(replay, step.successor):
            raise ValueError("Recorded CLLP transition failed replay")
        digest = state_hash(replay)
        steps.append({"action": ACTIONS[step.action], "action_index": step.action,
                      "probabilities": step.probabilities.tolist(), "no_op": not step.changed,
                      "repeated_state": digest in seen, "endpoint": to_rows(replay)})
        seen.add(digest)
    if result["error"] is None and bool(np.array_equal(replay, target)) != result["exact_hit"]:
        raise ValueError("Exact-hit flag disagrees with replay")
    result.update(actions=len(steps), steps=steps, actual_endpoint=to_rows(replay), replay_verified=True)
    return result


def run_case(
    config: ExperimentConfig, training: TrainingData, calibration: JSONObject,
    instance: Instance, variant: str, scale: float, provider: Provider, destination: Path | None = None,
) -> JSONObject:
    """Count all requests in the denominator, including copies and empty answers."""
    source = from_rows(instance["state"])
    identifier = f"{instance['id']}:{scale}:{variant}"
    api = CapturedAPI(config, provider, identifier)
    components = ICLComponents(config, api, environment(source), training, calibrated=calibration)
    seed = int(hashlib.sha256(f"11:{instance['id']}:{scale}".encode()).hexdigest()[:8], 16)
    row: JSONObject = {"board_id": instance["id"], "family": config.family, "split": config.split,
                       "variant": variant, "scale": scale, "sampling_seed": seed, "provider": provider,
                       "config": config.model_dump(mode="json"), "source": instance["state"],
                       "error": None, "complete": False, "reached_any": False, "proposals": []}
    if destination is not None:
        write_json(destination, row)
    try:
        proposals = components.propose(source, scale)
    except (RequestFailed, BudgetExceeded) as error:
        proposals = []
        row["error"] = {"type": type(error).__name__, "message": str(error)}
    row["generator_calls"] = copy.deepcopy(api.calls)
    row["classification"] = classify(api.calls, source, config.generator_layout_schema)
    row["generator_metrics"] = dict(api.metrics)
    row["diagnostics"] = dict(components.diagnostics)
    if destination is not None:
        write_json(destination, row)
    for target in proposals:
        if np.array_equal(target, source):
            raise ValueError("Production generator accepted an unchanged board")
        proposal: JSONObject = {"target": to_rows(target)}
        row["proposals"].append(proposal)

        def persist(progress: JSONObject) -> None:
            proposal["progress"] = progress
            if destination is not None:
                write_json(destination, row)

        execution = execute_target(components, source, target, seed, persist)
        proposal["rollout"] = execution
        row["reached_any"] |= execution["exact_hit"]
        if execution["error"] is not None:
            row["error"] = execution["error"]
            break
    row["all_calls"] = api.calls
    row["metrics"] = dict(api.metrics)
    row["complete"] = row["error"] is None
    if destination is not None:
        write_json(destination, row)
    return row
