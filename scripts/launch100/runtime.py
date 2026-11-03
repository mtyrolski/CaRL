"""Run paired jobs through the production search with durable response checkpoints."""

import csv
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash, to_rows
from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.experiment import credentials
from carl.icl.search import search
from carl.icl.search_tree import SearchTreeRecorder
from carl.icl.types import Board, Instance, JSONObject
from carl.utils.aliases import State
from scripts.launch100.journal import EventJournal, JournalAPI
from scripts.launch100.plan import Bundle, METHODS, component, digest, ledger_status, sha256


def job_key(plan: JSONObject, instance: Mapping[str, object], method: str) -> str:
    """Bind results to the freeze, board, condition, seed and both budgets."""
    return digest({"freeze": plan["freeze_id"], "board": instance["id"], "method": method,
                   "seed": plan["seed"], "nodes": plan["node_budget"], "actions": plan["action_budget"]})


class RecordedEnvironment:
    """Observe only simulator actions actually requested by the production search."""

    def __init__(self, initial: Board, journal: EventJournal) -> None:
        self.delegate = environment(initial)
        self.journal = journal

    def next_state(self, state: State, action: int) -> State:
        after = self.delegate.next_state(state, action)
        self.journal({"before": to_rows(state), "action": action, "after": to_rows(after),
                      "solved": bool(self.delegate.is_solved(after))})
        return after

    def is_solved(self, state: State) -> bool:
        return self.delegate.is_solved(state)


def run_job(bundle: Bundle, plan: JSONObject, instance: Instance, method: str, directory: Path) -> JSONObject:
    """Rebuild the frontier deterministically from saved responses before continuing."""
    config = ExperimentConfig.model_validate(plan["configs"][method])
    key = job_key(plan, instance, method)
    os.environ["CARL_CAMPAIGN_STAGE"] = METHODS[method][3]
    api = JournalAPI(config, directory / "responses", f"launch100:{plan['freeze_id']}:{key}")
    journal = EventJournal(directory / "actions.jsonl")
    initial = from_rows(instance["state"])
    tree = None
    model = None
    try:
        if method == "verbal_relative":
            from scripts.verbal_icl.launch import load_bundle, run_case

            verbal = load_bundle(bundle.path / "verbal.json")
            outcome = run_case(verbal, api, initial, action_budget=plan["action_budget"],
                               node_budget=plan["node_budget"], trace=journal)
        else:
            env = RecordedEnvironment(initial, journal)
            model = component(bundle, config, api, instance)
            model.env = env
            tree = SearchTreeRecorder()
            outcome = search(env, model, initial, config, plan["seed"], observer=tree)
        api.verify_consumed()
        journal.verify_consumed()
        from scripts.verbal_icl.search import verify_trace

        events = [json.loads(line) for line in journal.path.read_text().splitlines()] if journal.path.exists() else []
        verify_trace(environment(initial), initial, events, outcome.attempted_actions)
        if outcome.attempted_actions > config.action_budget or outcome.expansions > config.max_expansions:
            raise ValueError("Search exceeded its declared budget")
        if outcome.solved:
            replay, replay_env = initial, environment(initial)
            for action in outcome.action_path:
                replay = np.asarray(replay_env.next_state(replay, action))
            if not replay_env.is_solved(replay):
                raise ValueError("Claimed solution does not replay")
        shared_stop = outcome.finish_reason == "api_budget" and any(
            word in (outcome.error or "").lower() for word in ("campaign", "stage", "pricing"))
        result = {"job_key": key, "freeze_id": plan["freeze_id"], "board_id": instance["id"], "method": method,
                  "initial_state_hash": state_hash(initial),
                  "seed": plan["seed"], "node_budget": config.max_expansions, "action_budget": config.action_budget,
                  "complete": outcome.finish_reason != "api_error" and not shared_stop, "outcome": asdict(outcome),
                  "stop_category": "shared_budget" if shared_stop else
                                   "model_resource_cap" if outcome.finish_reason == "api_budget" else outcome.finish_reason,
                  "metrics": api.metrics, "response_replay_calls_this_invocation": api.replayed_calls,
                  "reconciled_pending_calls": api.reconciled_pending_calls,
                  "physical_accounting": "Shared ledger is authoritative; replayed pending calls can undercount API physical counters",
                  "diagnostics": model.diagnostics if model is not None else {},
                  "solution_replay_verified": bool(outcome.solved), "all_actions_replay_verified": True,
                  "trace_sha256": sha256(journal.path) if journal.path.exists() else None}
        if tree is not None:
            write_json(directory / "tree.json", {"nodes": [asdict(row) for row in tree.nodes],
                                                  "expansions": [asdict(row) for row in tree.expansions]})
        result["case_digest"] = digest(result)
        write_json(directory / "case.json", result)
        return result
    finally:
        if api.client is not None:
            api.client.close()


def verify_saved(plan: JSONObject, instance: Instance, method: str, directory: Path) -> JSONObject:
    """Verify completed checkpoints and replay actual actions before skipping work."""
    from scripts.verbal_icl.search import verify_trace

    saved = json.loads((directory / "case.json").read_text())
    expected = digest({key: value for key, value in saved.items() if key != "case_digest"})
    initial = from_rows(instance["state"])
    if (saved.get("case_digest") != expected or saved.get("job_key") != job_key(plan, instance, method)
            or saved.get("initial_state_hash") != state_hash(initial)):
        raise ValueError("Saved case identity or content differs")
    trace = directory / "actions.jsonl"
    if saved["trace_sha256"] != (sha256(trace) if trace.exists() else None):
        raise ValueError("Saved action trace differs")
    events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    verify_trace(environment(initial), initial, events, saved["outcome"]["attempted_actions"])
    if saved["outcome"]["solved"]:
        current, env = initial, environment(initial)
        for action in saved["outcome"]["action_path"]:
            current = np.asarray(env.next_state(current, action))
        if not env.is_solved(current):
            raise ValueError("Saved solution does not replay")
    return saved


def status(plan: JSONObject, output: Path, *, write_csv: bool = False) -> JSONObject:
    """Include every scheduled job, including missing and interrupted cases."""
    rows = []
    for board in plan["cohort"]:
        for method in plan["methods"]:
            key = job_key(plan, board, method)
            directory = output / "jobs" / key
            record = json.loads((directory / "case.json").read_text()) if (directory / "case.json").exists() else None
            if record is not None and (record.get("job_key") != key or record.get("freeze_id") != plan["freeze_id"]):
                raise ValueError("Saved job identity differs from launch manifest")
            if record is not None and record.get("case_digest") != digest({k: v for k, v in record.items() if k != "case_digest"}):
                raise ValueError("Saved case content differs")
            outcome = record["outcome"] if record else {}
            rows.append({"board_id": board["id"], "method": method, "job_key": key,
                         "status": ("complete" if record["complete"] else "blocked") if record else
                                   "interrupted" if directory.exists() else "missing",
                         "solved": outcome.get("solved"), "finish_reason": outcome.get("finish_reason"),
                         "expansions": outcome.get("expansions"), "actions": outcome.get("attempted_actions")})
    if write_csv:
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "status.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / "status.csv")
    return {"freeze_id": plan["freeze_id"], "expected": len(rows),
            "counts": {kind: sum(row["status"] == kind for row in rows)
                       for kind in ("complete", "blocked", "interrupted", "missing")},
            "solved": sum(row["solved"] is True for row in rows), "ledger": ledger_status()}


def execute(bundle: Bundle, plan: JSONObject, output: Path, usd_cap: float, *, resume: bool) -> None:
    """Require an explicit absolute campaign ceiling; all stages share the existing ledger."""
    if not math.isfinite(usd_cap) or not 0 < usd_cap <= 40:
        raise ValueError("--usd-cap must be finite and in (0,40], the absolute shared campaign ceiling")
    if not ledger_status()["ledger_present"]:
        raise ValueError("Existing shared campaign ledger is missing; restore its history before execution")
    previous = os.environ.get("CARL_CAMPAIGN_MAX_USD")
    os.environ["CARL_CAMPAIGN_MAX_USD"] = str(min(usd_cap, float(previous))) if previous else str(usd_cap)
    before = status(plan, output)
    if not resume and any(before["counts"][name] for name in ("complete", "blocked", "interrupted")):
        raise ValueError("Existing jobs require --resume")
    credentials("openai")
    for instance in bundle.instances:
        for method in plan["methods"]:
            directory = output / "jobs" / job_key(plan, instance, method)
            path = directory / "case.json"
            if path.exists():
                saved = verify_saved(plan, instance, method, directory)
                if not saved["complete"]:
                    raise ValueError(f"Inspect blocked case before any further execution: {path}")
                continue
            result = run_job(bundle, plan, instance, method, directory)
            status(plan, output, write_csv=True)
            print(json.dumps({"board": instance["id"], "method": method,
                              "complete": result["complete"], **{name: result["outcome"][name] for name in
                              ("solved", "finish_reason", "expansions", "attempted_actions")}}), flush=True)
            if not result["complete"]:
                return
