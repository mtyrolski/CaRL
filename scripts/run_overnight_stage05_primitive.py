#!/usr/bin/env python3
"""Secondary primitive-action search with CaRL's shared frozen value network.

Every attempted simulator step, including no-ops and duplicate states, is
charged. This is a weighted A*-style baseline; no optimality claim is made.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from filelock import FileLock
from transformers import BertForSequenceClassification

from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from carl.inference_components.value import TransformerValue
from carl.planners.greedy import AstarPlanner
from carl.solver.nodes import SearchTreeNode
from carl.solver.nodes import get_solving_path_data
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl
from scripts.run_overnight_stage05_policy import validate_board

ROOT = Path(__file__).resolve().parents[1]
VALUE = ROOT / "rl-data/validation/sokoban/components/full_data/value/checkpoint-1343100"
WORKER: dict[str, Any] = {}


def search(board: dict[str, Any], value: Any, *, budget: int, wall_seconds: float) -> dict[str, Any]:
    """Expand primitive actions with a shared value model and exact accounting."""
    if budget < 1 or wall_seconds <= 0:
        raise ValueError("transition and wall budgets must be positive")
    started = time.monotonic()
    initial = validate_board(board)
    env = environment(initial)
    value.env = env
    planner = AstarPlanner(initial, value_weight=150.0, depth_weight=1.0)
    result: dict[str, Any] = {**{key: board[key] for key in ("layout_id", "family", "source_family", "split", "board_rank", "trajectory_id")},
        "method": "primitive_astar", "status": "completed", "solved": False, "initial_state_hash": state_hash(initial),
        "transitions": 0, "solve_at_transition": None, "solution_length": None, "solution_actions": None,
        "solution_verified": None, "verification_kind": None, "solve_wall_seconds": None,
        "policy_calls": 0, "policy_input_tokens": 0, "value_calls": 0, "value_input_tokens": 0,
        "candidate_generation_calls": 0, "candidate_tokens": 0, "candidate_input_tokens": 0,
        "high_level_expansions": 0, "candidates_proposed": 0, "candidates_assessed": 0, "candidates_valid": 0,
        "controller_attempts": 0, "controller_hits": 0, "noop_actions": 0, "failure_reason": None,
        "transition_budget": budget, "wall_budget_seconds": wall_seconds, "trace": []}
    starting_swap = psutil.swap_memory().used
    try:
        while result["transitions"] < budget:
            if time.monotonic() - started >= wall_seconds:
                result.update(status="timeout", failure_reason="wall_timeout")
                break
            if psutil.virtual_memory().percent >= 80 or psutil.swap_memory().used - starting_swap >= 128 * 1024 * 1024:
                result.update(status="infra_error", solved=None, failure_reason="memory_or_swap_guard")
                break
            node = planner.get()
            if node is None:
                result["failure_reason"] = "queue_exhausted"
                break
            result["high_level_expansions"] += 1
            for action in range(4):
                if time.monotonic() - started >= wall_seconds:
                    result.update(status="timeout", failure_reason="wall_timeout")
                    break
                if result["transitions"] >= budget:
                    result["failure_reason"] = "transition_budget_exhausted"
                    break
                source = node.state
                event: dict[str, Any] = {"event": "primitive", "transition": result["transitions"] + 1,
                    "expansion": result["high_level_expansions"],
                    "source_hash": state_hash(source), "source_rows": to_rows(source), "action": action,
                    "target_hash": None, "reached_rows": None, "no_op": None, "solved": None,
                    "completed": False, "elapsed_seconds": None}
                result["trace"].append(event)
                result["transitions"] += 1
                successor = np.asarray(env.next_state(source, action))
                validate_subgoal(source, successor)
                same = bool(np.array_equal(source, successor))
                solved = bool(env.is_solved(successor))
                result["noop_actions"] += int(same)
                event.update(target_hash=state_hash(successor), reached_rows=to_rows(successor),
                             no_op=same, solved=solved, completed=True, elapsed_seconds=time.monotonic() - started)
                if solved:
                    leaf = SearchTreeNode(successor, 0.0, [action], node)
                    _, actions, _, _, _ = get_solving_path_data(leaf, include_state_path=False)
                    result.update(solved=True, solve_at_transition=result["transitions"], solution_length=len(actions),
                                  solution_actions=actions, solution_verified=True,
                                  verification_kind="executed_edges_and_terminal_goal_check",
                                  solve_wall_seconds=time.monotonic() - started)
                    break
                if planner.is_seen(successor):
                    continue
                result["value_calls"] += 1
                result["value_input_tokens"] += 146
                score = float(value.get_value(successor))
                if not math.isfinite(score):
                    raise ValueError("nonfinite value prediction")
                event["value"] = score
                planner.add(SearchTreeNode(successor, score, [action], node))
            if result["solved"] or result["status"] != "completed" or result["failure_reason"] is not None:
                break
        if result["failure_reason"] is None and not result["solved"]:
            result["failure_reason"] = "transition_budget_exhausted"
    except Exception as error:
        result.update(status="infra_error", solved=None, failure_reason=f"{type(error).__name__}: {error}")
    result["wall_seconds"] = time.monotonic() - started
    return result


def worker_init(threads: int, checkpoint: str) -> None:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    model = BertForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).cpu().eval()
    if model.config.num_labels != 1 or model.config.max_position_embeddings < 146:
        raise ValueError("checkpoint is not the compatible scalar board-value model")
    model.requires_grad_(False)
    WORKER.update(model=model, checkpoint=checkpoint)


def worker_job(board: dict[str, Any], output: str, budget: int, wall_seconds: float, contract_sha: str) -> dict[str, Any]:
    started = time.monotonic()
    path = Path(output) / "shards" / "primitive_astar" / f"{board['layout_id']}.json"
    try:
        env = environment(validate_board(board))
        value = TransformerValue(BertForSequenceClassification.from_pretrained, WORKER["checkpoint"], env,
                                 type_of_evaluation="regression", noise_variance=0.0)
        value.device = torch.device("cpu")
        value.value_network = WORKER["model"]
        result = search(board, value, budget=budget, wall_seconds=wall_seconds)
    except Exception as error:
        result = {**{key: board[key] for key in ("layout_id", "family", "source_family", "split", "board_rank", "trajectory_id")},
                  "method": "primitive_astar", "status": "infra_error", "solved": None,
                  "failure_reason": f"{type(error).__name__}: {error}", "wall_seconds": time.monotonic() - started,
                  "transitions": None, "solve_at_transition": None, "solution_length": None, "solution_actions": None,
                  "solution_verified": None, "solve_wall_seconds": None, "trace": []}
    result["contract_sha256"] = contract_sha
    atomic_json(path, result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.threads < 1 or args.budget < 1 or args.wall_seconds <= 0:
        raise ValueError("workers, threads, budget and wall limit must be positive")
    if args.workers * args.threads > len(os.sched_getaffinity(0)):
        raise ValueError("workers times threads exceeds CPU affinity")
    selected = [row for row in read_jsonl(args.boards) if row["split"] == args.split]
    if args.limit_per_family is not None:
        if args.limit_per_family < 1:
            raise ValueError("limit-per-family must be positive")
        selected = [row for row in selected if row["board_rank"] < args.limit_per_family]
    if not selected or len({row["layout_id"] for row in selected}) != len(selected):
        raise ValueError("expected a nonempty unique, frozen board list")
    for row in selected:
        validate_board(row)
    checkpoint = args.checkpoint.resolve()
    needed = [checkpoint / "config.json", checkpoint / "pytorch_model.bin"]
    if not all(path.is_file() for path in needed):
        raise ValueError("local scalar value checkpoint is unavailable")
    contract = {"schema_version": 1, "method": "primitive_astar", "split": args.split,
        "board_list_sha256": file_sha(args.boards), "board_ids": [row["layout_id"] for row in selected],
        "checkpoint": str(checkpoint), "checkpoint_sha256": {p.name: file_sha(p) for p in needed},
        "runner_sha256": file_sha(Path(__file__)), "transition_budget": args.budget,
        "wall_budget_seconds": args.wall_seconds, "workers": args.workers, "threads": args.threads,
        "priority": "CaRL AstarPlanner: -value*150 + depth; value=-predicted_distance",
        "cost": "each actual env.next_state call, including noops and duplicate states",
        "model_load_time": "one-time per worker, excluded from per-board wall times",
        "scope": "secondary weighted primitive-action search, no optimality guarantee"}
    sha = digest(contract)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        path = args.output / "contract.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError("output already belongs to a different frozen primitive-search contract")
        atomic_json(path, contract)
        done = {}
        for row in selected:
            path = args.output / "shards" / "primitive_astar" / f"{row['layout_id']}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get("contract_sha256") != sha or saved.get("layout_id") != row["layout_id"]:
                    raise ValueError("saved primitive-search shard differs from this contract")
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
        counts = {status: sum(row["status"] == status for row in done.values())
                  for status in ("completed", "timeout", "infra_error")}
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
    parser.add_argument("--checkpoint", type=Path, default=VALUE)
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--wall-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-per-family", type=int)
    print(json.dumps(run(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()
