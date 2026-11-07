"""Measure exact-state generation and task progress on fixed validation sources."""

import copy
import hashlib
from pathlib import Path

from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows, to_rows
from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.components import ICLComponents
from carl.icl.types import JSONObject, Provider, TrainingData
from scripts.generator_probe.cases import CapturedAPI, classify, execute_target
from scripts.goal_probe.components import MechanicsExample, TaskProgressComponents
from scripts.goal_probe.data import Query, goals


class GoalAPI(CapturedAPI):
    """Use the shared cache and ledger while keeping this study's costs separate."""

    def __init__(self, config: ExperimentConfig, provider: Provider, tag: str) -> None:
        super().__init__(config, provider, tag)
        self.budget_tag = f"goal_progress_probe:{tag}"


def run_case(
    config: ExperimentConfig, training: TrainingData, query: Query,
    demonstrations: list[MechanicsExample], variant: str, provider: Provider, destination: Path,
) -> JSONObject:
    """Persist before calls and after every physical action; never supply a query witness."""
    state = from_rows(query["state"])
    api = GoalAPI(config, provider, f"explicit:{variant}:{query['id']}")
    components = (TaskProgressComponents(config, api, environment(state), training, demonstrations)
                  if variant == "goal_examples" else ICLComponents(config, api, environment(state), training))
    seed = int(hashlib.sha256(f"11:{query['id']}:0.85".encode()).hexdigest()[:8], 16)
    row: JSONObject = {"query_id": query["id"], "board_id": query["provenance"]["board_id"],
                       "source": query["state"], "provenance": query["provenance"],
                       "variant": variant, "u": 0.85, "seed": seed, "provider": provider,
                       "config": config.model_dump(mode="json"), "complete": False,
                       "error": None, "proposals": [], "goals_before": goals(query["state"])}
    write_json(destination, row)
    try:
        try:
            proposals = components.propose(state, 0.85)
        except (RequestFailed, BudgetExceeded) as error:
            proposals = []
            row["error"] = {"type": type(error).__name__, "message": str(error)}
        row["generator_calls"] = copy.deepcopy(api.calls)
        row["classification"] = classify(api.calls, state, config.generator_layout_schema)
        row["diagnostics"] = dict(components.diagnostics)
        write_json(destination, row)
        for target in proposals:
            proposal: JSONObject = {"target": to_rows(target), "target_goals": goals(to_rows(target))}
            row["proposals"].append(proposal)

            def persist(progress: JSONObject) -> None:
                proposal["progress"] = progress
                write_json(destination, row)

            execution = execute_target(components, state, target, seed, persist)
            proposal["rollout"] = execution
            proposal["endpoint_goals"] = goals(execution["actual_endpoint"])
            if execution["error"] is not None:
                row["error"] = execution["error"]
                break
        row["reached_any"] = any(item["rollout"]["exact_hit"] for item in row["proposals"])
        row["reached_goal_progress"] = any(
            item["rollout"]["exact_hit"] and item["endpoint_goals"] > row["goals_before"]
            for item in row["proposals"])
        row["complete"] = row["error"] is None
        return row
    finally:
        row["calls"] = api.calls
        row["metrics"] = dict(api.metrics)
        write_json(destination, row)
        if api.client is not None:
            api.client.close()
