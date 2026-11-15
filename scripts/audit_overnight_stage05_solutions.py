#!/usr/bin/env python3
"""Independently replay saved complete-task solutions and charged transitions.

All simulator calls in this script are *offline audit work*, recorded separately
from the methods' original search budgets. No model is loaded or queried.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl
from scripts.run_overnight_stage05_policy import validate_board

AUDIT_TRANSITIONS = 0


class IncompleteTrace(Exception):
    """An infrastructure interruption left a trace that cannot be fully audited."""


def replay_actions(start: np.ndarray, actions: list[int], *, hashes: list[str] | None = None,
                   simulator: Any | None = None) -> tuple[np.ndarray, int]:
    """Replay with the environment configured for the actual 4 or 10 boxes."""
    global AUDIT_TRANSITIONS
    if hashes is not None and (len(hashes) != len(actions) + 1 or hashes[0] != state_hash(start)):
        raise ValueError("recorded trace starts from an inconsistent state")
    env = simulator if simulator is not None else environment(start)
    state = start.copy()
    noops = 0
    for index, action in enumerate(actions, 1):
        if not isinstance(action, int) or isinstance(action, bool) or action not in range(4):
            raise ValueError("invalid primitive action in saved trace")
        if env.is_solved(state):
            raise ValueError("trace executes an action after task success")
        AUDIT_TRANSITIONS += 1
        successor = np.asarray(env.next_state(state, action))
        validate_subgoal(state, successor)
        noops += int(np.array_equal(state, successor))
        state = successor
        if hashes is not None and state_hash(state) != hashes[index]:
            raise ValueError(f"recorded transition {index} does not match environment replay")
    return state, noops


def audit_policy(start: np.ndarray, job: dict[str, Any]) -> dict[str, int]:
    """Replay the complete actual policy rollout; no target is supplied."""
    events = job.get("trace", [])
    if any(not event.get("completed", True) for event in events):
        if job.get("status") == "infra_error":
            raise IncompleteTrace("policy simulator transition was interrupted")
        raise ValueError("a non-infrastructure policy job has an incomplete simulator transition")
    actions = [event["action"] for event in events]
    hashes = [state_hash(start)]
    for index, event in enumerate(events, 1):
        if event["transition"] != index or event["before_hash"] != hashes[-1]:
            raise ValueError("policy trace order or source hash mismatch")
        hashes.append(event["after_hash"])
    env = environment(start)
    end, noops = replay_actions(start, actions, hashes=hashes, simulator=env)
    if (job["transitions"] != len(actions) or job["noop_actions"] != noops
            or noops != sum(event["no_op"] is True for event in events)):
        raise ValueError("policy transition or no-op count differs from the trace")
    if job.get("final_state_hash") is not None and job["final_state_hash"] != state_hash(end):
        raise ValueError("policy final-state hash differs from replay")
    if job.get("final_state") is not None and job["final_state"] != to_rows(end):
        raise ValueError("policy final board differs from replay")
    solved = env.is_solved(end)
    if job["solved"] is not None and bool(job["solved"]) != solved:
        raise ValueError("policy goal claim differs from replay")
    if solved and (job.get("solution_length") != len(actions) or job.get("solve_at_transition") != len(actions)):
        raise ValueError("policy solution cost differs from the replayed actions")
    return {"recorded_transitions_replayed": len(actions), "solution_transitions_replayed": 0,
            "solutions_verified": int(solved), "noops_verified": noops}


def audit_search(start: np.ndarray, job: dict[str, Any]) -> dict[str, int]:
    """Replay every attempted full-state controller edge and the returned plan."""
    trace = job.get("trace", [])
    env = environment(start)
    sources = {state_hash(start): to_rows(start)}
    for event in trace:
        if event.get("event") == "generation":
            rows = event["source_rows"]
            if state_hash(from_rows(rows)) != event["source_hash"]:
                raise ValueError("generator source rows do not match their state hash")
            validate_subgoal(start, from_rows(rows))
            sources[event["source_hash"]] = rows
    transitions = noops = 0
    attempted = hits = 0
    solved_during_attempt = False
    for event in trace:
        if event.get("event") != "candidate" or "start_transitions" not in event:
            continue
        actions = event.get("actions", [])
        # A candidate not attempted has a counter start but no controller call.
        if "end_transitions" not in event:
            if "noop_actions" in event and job.get("status") == "infra_error":
                raise IncompleteTrace("controller attempt was interrupted")
            if actions or "noop_actions" in event:
                raise ValueError("unfinished candidate has unaccounted primitive actions")
            continue
        source = from_rows(sources[event["source_hash"]])
        if event["start_transitions"] != transitions:
            raise ValueError("controller start counter differs from prior actions")
        reached, count = replay_actions(source, actions, hashes=event.get("state_hashes"), simulator=env)
        attempted += 1
        transitions += len(actions)
        noops += count
        if (event["end_transitions"] != transitions or event.get("noop_actions") != count
                or event.get("reached_rows") != to_rows(reached)):
            raise ValueError("controller transition counter, no-ops or reached board differs")
        task_solved = env.is_solved(reached)
        if bool(event.get("task_solved")) != task_solved:
            raise ValueError("candidate task-goal claim differs from replay")
        solved_during_attempt |= task_solved
        target = from_rows(event["target_rows"])
        validate_subgoal(source, target)
        hit = bool(np.array_equal(reached, target))
        if bool(event.get("hit")) != hit:
            raise ValueError("candidate target-hit claim differs from replay")
        hits += int(hit)
    if (job["transitions"] != transitions or job["noop_actions"] != noops
            or job["controller_attempts"] != attempted or job["controller_hits"] != hits):
        raise ValueError("search aggregate controller counters differ from replay")
    if job.get("solved") is not None and bool(job["solved"]) != solved_during_attempt:
        raise ValueError("search solution status differs from the attempted edges")
    if job.get("status") != "infra_error":
        generations = [event for event in trace if event.get("event") == "generation"]
        candidates = [event for event in trace if event.get("event") == "candidate"]
        valid_candidates = sum("target_rows" in event and event.get("reason") != "invalid_target" for event in candidates)
        value_calls = sum(event.get("reason") == "target_reached" for event in candidates)
        if (job["candidate_generation_calls"] != len(generations)
                or job["candidates_proposed"] != sum(len(event["output_lengths"]) for event in generations)
                or job["candidate_tokens"] != sum(sum(event["output_lengths"]) for event in generations)
                or job["candidates_assessed"] != len(candidates) or job["candidates_valid"] != valid_candidates
                or job["value_calls"] != value_calls or job["policy_calls"] != transitions):
            raise ValueError("search model-call or candidate counters differ from the trace")
    solution_count = 0
    if job.get("solved") is True:
        actions = job.get("solution_actions")
        if not isinstance(actions, list) or len(actions) != job.get("solution_length"):
            raise ValueError("missing reconstructed action sequence for a claimed solution")
        end, _ = replay_actions(start, actions, simulator=env)
        if not env.is_solved(end):
            raise ValueError("reconstructed full plan does not solve its original board")
        if job["solve_at_transition"] < len(actions):
            raise ValueError("solution cost exceeds all charged transitions")
        solution_count = len(actions)
    return {"recorded_transitions_replayed": transitions, "solution_transitions_replayed": solution_count,
            "solutions_verified": int(job.get("solved") is True), "noops_verified": noops}


def audit_primitive(start: np.ndarray, job: dict[str, Any]) -> dict[str, int]:
    """Replay every primitive search edge, including discarded moves and no-ops."""
    events = job.get("trace", [])
    env = environment(start)
    if any(not event.get("completed", True) for event in events):
        if job.get("status") == "infra_error":
            raise IncompleteTrace("primitive simulator transition was interrupted")
        raise ValueError("a non-infrastructure primitive job has an incomplete transition")
    noops = solved_events = 0
    for index, event in enumerate(events, 1):
        if event.get("event") != "primitive" or event.get("transition") != index:
            raise ValueError("primitive trace order or event type mismatch")
        source = from_rows(event["source_rows"])
        validate_subgoal(start, source)
        if state_hash(source) != event["source_hash"]:
            raise ValueError("primitive source differs from its recorded hash")
        reached, count = replay_actions(source, [event["action"]],
                                        hashes=[event["source_hash"], event["target_hash"]], simulator=env)
        if (event["reached_rows"] != to_rows(reached) or bool(event["no_op"]) != bool(count)
                or bool(event["solved"]) != env.is_solved(reached)):
            raise ValueError("primitive reached state, no-op or goal differs from replay")
        noops += count
        solved_events += int(event["solved"])
    if job["transitions"] != len(events) or job["noop_actions"] != noops:
        raise ValueError("primitive transition or no-op count differs from its trace")
    if job.get("solved") is not None and bool(job["solved"]) != bool(solved_events):
        raise ValueError("primitive solution status differs from its trace")
    if job.get("status") != "infra_error":
        if (job["value_calls"] != sum("value" in event for event in events)
                or job["value_input_tokens"] != 146 * job["value_calls"]
                or job["policy_calls"] != 0 or job["candidate_generation_calls"] != 0):
            raise ValueError("primitive model-call accounting differs from its trace")
    solution_count = 0
    if job.get("solved") is True:
        actions = job.get("solution_actions")
        if not isinstance(actions, list) or len(actions) != job.get("solution_length"):
            raise ValueError("missing primitive solution action sequence")
        end, _ = replay_actions(start, actions, simulator=env)
        if not env.is_solved(end):
            raise ValueError("primitive action sequence does not solve its original board")
        if job["solve_at_transition"] < len(actions):
            raise ValueError("primitive solution length exceeds all charged transitions")
        solution_count = len(actions)
    return {"recorded_transitions_replayed": len(events), "solution_transitions_replayed": solution_count,
            "solutions_verified": int(job.get("solved") is True), "noops_verified": noops}


def audit_job(board: dict[str, Any], job: dict[str, Any]) -> dict[str, int]:
    for key in ("layout_id", "family", "source_family", "split", "board_rank"):
        if job.get(key) != board[key]:
            raise ValueError(f"job and frozen board disagree on {key}")
    start = validate_board(board)
    if job.get("initial_state_hash") is not None and job["initial_state_hash"] != state_hash(start):
        raise ValueError("job start hash does not match the frozen full puzzle")
    if job.get("transitions") is None:
        if job.get("status") != "infra_error":
            raise ValueError("only an infrastructure failure may omit transition count")
        raise IncompleteTrace("infrastructure failure left no auditable transition count")
    if job["method"] == "direct_policy":
        return audit_policy(start, job)
    if job["method"] == "primitive_astar":
        return audit_primitive(start, job)
    return audit_search(start, job)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true", help="allow an explicitly partial development audit")
    args = parser.parse_args()
    boards = [row for row in read_jsonl(args.boards) if row["split"] == args.split]
    board_by_id = {row["layout_id"]: row for row in boards}
    if not boards or len(board_by_id) != len(boards):
        raise ValueError("frozen board list is empty or has duplicate layouts")
    jobs = {}
    for root in args.shards:
        for path in sorted(root.rglob("*.json")):
            job = json.loads(path.read_text())
            key = (job["layout_id"], job["method"])
            if key in jobs:
                raise ValueError(f"duplicate shard {key}")
            jobs[key] = (path, job)
    expected = {(layout, method) for layout in board_by_id for method in args.methods}
    extra = set(jobs) - expected
    if extra:
        raise ValueError(f"unexpected shards: {sorted(extra)}")
    missing = sorted(expected - set(jobs))
    errors = []
    incomplete = []
    totals: Counter[str] = Counter()
    by_method: dict[str, Counter[str]] = {method: Counter() for method in args.methods}
    for key, (path, job) in sorted(jobs.items()):
        try:
            metrics = audit_job(board_by_id[key[0]], job)
            totals.update(metrics)
            by_method[key[1]].update(metrics)
            by_method[key[1]]["jobs_audited"] += 1
        except IncompleteTrace as error:
            incomplete.append({"shard": str(path), "reason": str(error)})
        except Exception as error:
            errors.append({"shard": str(path), "error": f"{type(error).__name__}: {error}"})
    state = "fail" if errors else "partial" if missing or incomplete else "pass"
    result = {"state": state,
              "frozen_boards": len(boards), "methods": args.methods, "expected_jobs": len(expected),
              "present_jobs": len(jobs), "missing_jobs": [list(key) for key in missing],
              "incomplete_traces": incomplete, "errors": errors,
              "totals": dict(totals), "by_method": {method: dict(value) for method, value in by_method.items()},
              "offline_audit_transitions": AUDIT_TRANSITIONS,
              "successfully_audited_transitions": totals["recorded_transitions_replayed"] + totals["solution_transitions_replayed"],
              "boards_sha256": file_sha(args.boards), "auditor_sha256": file_sha(Path(__file__)),
              "scope": "Offline real-environment replay; audit transitions are separate from the original method budgets."}
    atomic_json(args.output, result)
    print(json.dumps(result), flush=True)
    if errors or ((missing or incomplete) and not args.allow_missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
