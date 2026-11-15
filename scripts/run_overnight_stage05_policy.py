#!/usr/bin/env python3
"""Run the frozen, memoryless Sokoban policy from full puzzle starts on CPU.

Each completed board is written atomically. The policy receives only the
current full state. A repeated complete state proves an argmax-policy cycle:
the environment and the evaluation-mode policy are both deterministic.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from filelock import FileLock
from transformers import BertForSequenceClassification

from carl.environment.training_goal import TrainingGoal
from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import tiles
from carl.icl.boards import to_rows
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from carl.inference_components.policy import TransformerPolicy
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "rl-data/validation/sokoban/components/full_data/policy/checkpoint-94820"
FAMILY_BOXES = {"12x12x4": 4, "12x12x10": 10}
WORKER: dict[str, Any] = {}


class JobStatus(StrEnum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    INFRA_ERROR = "infra_error"


def validate_board(board: dict[str, Any]) -> np.ndarray:
    """Validate a full initial board without reading any demonstration future."""
    family = board["family"]
    if family not in FAMILY_BOXES:
        raise ValueError("unknown Sokoban family")
    state = from_rows(board["initial_state"])
    if state.shape != (12, 12, 7):
        raise ValueError("expected a complete 12x12 board")
    cells = tiles(state)
    if (int(np.isin(cells, (3, 4)).sum()) != FAMILY_BOXES[family]
            or int(np.isin(cells, (2, 3, 6)).sum()) != FAMILY_BOXES[family]
            or int(np.isin(cells, (5, 6)).sum()) != 1):
        raise ValueError("board boxes, goals or player count does not match its family")
    if canonical_layout_id(state) != board["layout_id"]:
        raise ValueError("canonical layout differs from the frozen board")
    if "initial_state_sha256" in board and state_hash(state) != board["initial_state_sha256"]:
        raise ValueError("initial state differs from the frozen board hash")
    validate_subgoal(state, state)
    if environment(state).is_solved(state):
        raise ValueError("a full-puzzle start may not already be solved")
    return state


def run_board(board: dict[str, Any], policy: Any, *, budget: int, wall_seconds: float) -> dict[str, Any]:
    """Run actual primitive transitions with an independently restored simulator."""
    if budget < 1 or wall_seconds <= 0:
        raise ValueError("budget and per-board wall limit must be positive")
    started = time.monotonic()
    state = validate_board(board)
    env = environment(state)
    policy.env = env
    encoded, _ = env.tokenizer.x_y_tokenizer(x=state, y=0, training_goal=TrainingGoal.POLICY)
    token_length = int(encoded.numel())
    if token_length != 146:
        raise ValueError("the current-board policy must receive 146 tokens")
    first_hash = state_hash(state)
    seen = {first_hash: 0}
    trace: list[dict[str, Any]] = []
    calls = noops = 0
    solved: bool | None = False
    status = JobStatus.COMPLETED
    reason: str | None = None
    cycle_first: int | None = None
    cycle_length: int | None = None
    solve_seconds: float | None = None
    baseline_swap = psutil.swap_memory().used
    while len(trace) < budget:
        if time.monotonic() - started >= wall_seconds:
            status, reason = JobStatus.TIMEOUT, "wall_timeout"
            break
        if psutil.virtual_memory().percent >= 80 or psutil.swap_memory().used - baseline_swap >= 128 * 1024 * 1024:
            status, solved, reason = JobStatus.INFRA_ERROR, None, "memory_or_swap_guard"
            break
        calls += 1
        action_started = time.monotonic()
        try:
            proposals = policy.get_actions(state)
            if len(proposals) != 1:
                raise ValueError("greedy direct policy must return exactly one action")
            action = int(proposals[0].action)
            if action not in range(4):
                raise ValueError("policy returned an invalid primitive action")
        except Exception as error:
            status, solved, reason = JobStatus.INFRA_ERROR, None, f"{type(error).__name__}: {error}"
            break
        if time.monotonic() - started >= wall_seconds:
            status, reason = JobStatus.TIMEOUT, "wall_timeout"
            break
        before_hash = state_hash(state)
        event = {"transition": len(trace) + 1, "action": action, "before_hash": before_hash,
                 "after_hash": None, "no_op": None, "solved": None, "completed": False,
                 "boxes_on_goals": None, "wall_seconds": None, "action_wall_seconds": None}
        trace.append(event)
        try:
            next_state = np.asarray(env.next_state(state, action))
            validate_subgoal(state, next_state)
        except Exception as error:
            event["wall_seconds"] = time.monotonic() - started
            event["action_wall_seconds"] = time.monotonic() - action_started
            status, solved, reason = JobStatus.INFRA_ERROR, None, f"{type(error).__name__}: {error}"
            break
        after_hash = state_hash(next_state)
        no_op = after_hash == before_hash
        noops += int(no_op)
        state = next_state
        solved = bool(env.is_solved(state))
        elapsed = time.monotonic() - started
        event.update({"after_hash": after_hash, "no_op": no_op, "solved": solved, "completed": True,
                      "boxes_on_goals": env.num_boxes_on_target(state), "wall_seconds": elapsed,
                      "action_wall_seconds": time.monotonic() - action_started})
        if solved:
            reason, solve_seconds = None, elapsed
            break
        if after_hash in seen:
            cycle_first = seen[after_hash]
            cycle_length = len(trace) - cycle_first
            reason = "deterministic_cycle"
            break
        seen[after_hash] = len(trace)
    else:
        reason = "transition_budget_exhausted"
    return {**{key: board[key] for key in ("layout_id", "family", "source_family", "split", "board_rank", "trajectory_id")},
            "method": "direct_policy", "status": status, "solved": solved, "failure_reason": reason,
            "initial_state": board["initial_state"], "initial_state_hash": first_hash,
            "transitions": len(trace), "solve_at_transition": len(trace) if solved else None,
            "solution_length": len(trace) if solved else None, "solution_verified": True if solved else None,
            "verification_kind": "direct_simulation_goal_check" if solved else None,
            "final_state": to_rows(state), "final_state_hash": state_hash(state), "wall_seconds": time.monotonic() - started,
            "solve_wall_seconds": solve_seconds, "policy_calls": calls, "policy_input_tokens": calls * token_length,
            "candidate_generation_calls": 0, "candidate_tokens": 0, "candidate_input_tokens": 0,
            "value_calls": 0, "value_input_tokens": 0, "high_level_expansions": 0,
            "candidates_proposed": 0, "candidates_assessed": 0, "candidates_valid": 0,
            "controller_attempts": 0, "controller_hits": 0,
            "noop_actions": noops, "cycle_first_seen_transition": cycle_first, "cycle_length": cycle_length,
            "transition_budget": budget, "wall_budget_seconds": wall_seconds, "trace": trace}


def worker_init(threads: int, checkpoint: str) -> None:
    """Construct one resident frozen CPU model per bounded worker process."""
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = BertForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).cpu().eval()
    if model.config.num_labels != 4 or model.config.max_position_embeddings < 146:
        raise ValueError("checkpoint is not a compatible four-action full-state policy")
    model.requires_grad_(False)
    WORKER.clear()
    WORKER["model"] = model
    WORKER["checkpoint"] = checkpoint


def worker_job(board: dict[str, Any], output: str, budget: int, wall_seconds: float, contract_sha: str) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(output) / "shards" / "direct_policy" / f"{board['layout_id']}.json"
    try:
        state = validate_board(board)
        env = environment(state)
        policy = TransformerPolicy(BertForSequenceClassification.from_pretrained, env, WORKER["checkpoint"], n_actions=1)
        policy.policy = WORKER["model"]
        result = run_board(board, policy, budget=budget, wall_seconds=wall_seconds)
    except Exception as error:
        result = {**{key: board[key] for key in ("layout_id", "family", "source_family", "split", "board_rank", "trajectory_id")},
                  "method": "direct_policy", "status": JobStatus.INFRA_ERROR, "solved": None,
                  "failure_reason": f"{type(error).__name__}: {error}", "wall_seconds": time.monotonic() - started,
                  "transitions": None, "solve_at_transition": None, "solution_length": None,
                  "solution_verified": None, "verification_kind": None, "solve_wall_seconds": None,
                  "policy_calls": None, "policy_input_tokens": None, "noop_actions": None,
                  "candidate_generation_calls": 0, "candidate_tokens": 0, "candidate_input_tokens": 0,
                  "value_calls": 0, "value_input_tokens": 0,
                  "high_level_expansions": 0, "candidates_proposed": 0, "candidates_assessed": 0, "candidates_valid": 0,
                  "controller_attempts": 0, "controller_hits": 0,
                  "transition_budget": budget, "wall_budget_seconds": wall_seconds, "trace": []}
    result["contract_sha256"] = contract_sha
    atomic_json(path, result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.threads < 1 or args.budget < 1 or args.wall_seconds <= 0:
        raise ValueError("workers, threads, budget and wall limit must be positive")
    affinity = len(os.sched_getaffinity(0))
    if args.workers * args.threads > affinity:
        raise ValueError("workers times threads exceeds CPU affinity")
    boards = read_jsonl(args.boards)
    selected = [row for row in boards if row["split"] == args.split]
    if not selected or len({row["layout_id"] for row in selected}) != len(selected):
        raise ValueError("expected a nonempty unique, frozen board list")
    for row in selected:
        validate_board(row)
    checkpoint = args.checkpoint.resolve()
    needed = [checkpoint / "config.json", checkpoint / "pytorch_model.bin"]
    if not all(path.is_file() for path in needed):
        raise ValueError("local ordinary-policy checkpoint is unavailable")
    contract = {"schema_version": 1, "method": "direct_policy", "split": args.split,
                "board_list_sha256": file_sha(args.boards), "board_ids": [row["layout_id"] for row in selected],
                "checkpoint": str(checkpoint), "checkpoint_sha256": {p.name: file_sha(p) for p in needed},
                "runner_sha256": file_sha(Path(__file__)), "transition_budget": args.budget,
                "wall_budget_seconds": args.wall_seconds, "workers": args.workers, "threads": args.threads,
                "input_contract": "current full board only, 146 tokens; no target or history",
                "action_selection": "CPU float32, frozen eval-mode network, greedy argmax, no action mask",
                "cycle_rule": "stop on first repeated full state; no artificial future transitions",
                "model_load_time": "one-time per worker, excluded from per-board wall times",
                "environment": "real Sokoban simulator with box count inferred from each full start"}
    sha = digest(contract)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        contract_path = args.output / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("output already belongs to a different frozen policy contract")
        atomic_json(contract_path, contract)
        done = {}
        for row in selected:
            path = args.output / "shards" / "direct_policy" / f"{row['layout_id']}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get("contract_sha256") != sha or saved.get("layout_id") != row["layout_id"]:
                    raise ValueError("saved policy shard differs from this frozen contract")
                done[row["layout_id"]] = saved
        started = time.monotonic()
        atomic_json(args.output / "status.json", {"state": "running", "planned": len(selected), "completed": len(done)})
        pending = [row for row in selected if row["layout_id"] not in done]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                initializer=worker_init, initargs=(args.threads, str(checkpoint))) as pool:
            futures = {pool.submit(worker_job, row, str(args.output), args.budget, args.wall_seconds, sha): row
                       for row in pending}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                done[result["layout_id"]] = result
                atomic_json(args.output / "status.json", {"state": "running", "planned": len(selected),
                            "completed": len(done), "elapsed_seconds": time.monotonic() - started})
        counts = {str(status): sum(row["status"] == status for row in done.values()) for status in JobStatus}
        result = {"state": "complete" if len(done) == len(selected) else "partial", "planned": len(selected),
                  "completed": len(done), "statuses": counts, "elapsed_seconds": time.monotonic() - started,
                  "contract_sha256": sha}
        atomic_json(args.output / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("development", "evaluation"))
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--wall-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
