#!/usr/bin/env python3
"""Run full-state AdaSubS or fixed-k search under an exact transition budget.

The learned generators predict whole boards, and the CLLP sees each complete
target.  This runner is not an intent-conditioned controller experiment.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import threading
import time
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from transformers import BartForConditionalGeneration
from transformers import BertForSequenceClassification

from carl.environment.training_goal import TrainingGoal
from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import tiles
from carl.icl.boards import to_rows
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from carl.inference_components.conditional_low_level_policy import TransformerConditionalLowLevelPolicy
from carl.inference_components.subgoal_generator import TransformerSubgoalGenerator
from carl.inference_components.value import TransformerValue
from carl.planners.adasubs import AdasubsPlanner
from carl.solver.nodes import SearchTreeNode
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / "rl-data/validation/sokoban/components/full_data"
GENERATORS = {
    1: COMPONENTS / "generator/border/1/checkpoint-151712",
    4: COMPONENTS / "generator/border/4/checkpoint-75856",
    8: COMPONENTS / "generator/border/8/checkpoint-75856",
}
CLLP = COMPONENTS / "cllp/8/checkpoint-167585"
VALUE = COMPONENTS / "value/checkpoint-1343100"
BOXES = {"12x12x4": 4, "12x12x10": 10}
WORKER: dict[str, Any] = {}


class SearchMethod(StrEnum):
    ADASUBS = "adasubs"
    KSUBS = "ksubs"


class ResourceMonitor:
    """Sample host pressure and approximate aggregate RSS of this job's process tree."""

    def __init__(self, started: float) -> None:
        self.started = started
        self.peaks: dict[str, float | int] = {"peak_host_memory_percent": 0.0,
            "peak_swap_used_bytes": 0, "peak_process_tree_rss_bytes": 0}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process()
        while not self.stop_event.is_set():
            try:
                members = [process, *process.children(recursive=True)]
                rss = 0
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except psutil.Error:
                        pass
                self.peaks["peak_process_tree_rss_bytes"] = max(self.peaks["peak_process_tree_rss_bytes"], rss)
                self.peaks["peak_host_memory_percent"] = max(self.peaks["peak_host_memory_percent"],
                                                              psutil.virtual_memory().percent)
                self.peaks["peak_swap_used_bytes"] = max(self.peaks["peak_swap_used_bytes"], psutil.swap_memory().used)
            except psutil.Error:
                pass
            self.stop_event.wait(0.5)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def snapshot(self) -> dict[str, Any]:
        return {**self.peaks, "elapsed_seconds": time.monotonic() - self.started,
                "rss_note": "sum of process RSS, may count shared pages more than once"}


def checkpoint_hashes(paths: list[Path]) -> dict[str, dict[str, str]]:
    """Hash exactly the configuration and locally loaded model weights."""
    return {str(path): {name: file_sha(path / name) for name in ("config.json", "pytorch_model.bin")}
            for path in paths}


def valid_start(board: dict[str, Any]) -> tuple[np.ndarray, Any]:
    state = from_rows(board["initial_state"])
    if board["family"] not in BOXES or state.shape != (12, 12, 7):
        raise ValueError("unsupported board family or dimensions")
    values = tiles(state)
    boxes = int(np.isin(values, [3, 4]).sum())
    goals = int(np.isin(values, [2, 3, 6]).sum())
    if boxes != BOXES[board["family"]] or goals != boxes:
        raise ValueError("incorrect box or goal count")
    validate_subgoal(state, state)
    if canonical_layout_id(state) != board["layout_id"]:
        raise ValueError("board does not match its canonical layout")
    env = environment(state)
    if env.num_boxes != boxes or env.is_solved(state):
        raise ValueError("unexpected environment box count or already solved start")
    return state, env


def initialize_worker(config: dict[str, Any]) -> None:
    """One persistent model bank per worker; all inference remains on CPU."""
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    np.random.seed(config["seed"])
    # Components use a fresh environment with the actual box count for each job.
    dummy = np.zeros((12, 12, 7), dtype=np.float32)
    dummy[:, :, 0] = 1
    dummy[1, 1, 0] = dummy[1, 2, 0] = 0
    dummy[1, 1, 5], dummy[1, 2, 4] = 1, 1
    env = environment(dummy)
    generators = {}
    for k in config["ks"]:
        component = TransformerSubgoalGenerator(BartForConditionalGeneration.from_pretrained,
            str(GENERATORS[k]), env, config["generation"])
        component.sub_generator = BartForConditionalGeneration.from_pretrained(
            str(GENERATORS[k]), local_files_only=True).to("cpu").eval()
        generators[k] = component
    cllp = TransformerConditionalLowLevelPolicy(BertForSequenceClassification.from_pretrained, str(CLLP), env)
    cllp.cllp = BertForSequenceClassification.from_pretrained(str(CLLP), local_files_only=True).to("cpu").eval()
    value = TransformerValue(BertForSequenceClassification.from_pretrained, str(VALUE), env,
                             type_of_evaluation="regression", noise_variance=0.0)
    value.device = torch.device("cpu")
    value.value_network = BertForSequenceClassification.from_pretrained(str(VALUE), local_files_only=True).to("cpu").eval()
    WORKER.update({"generators": generators, "cllp": cllp, "value": value, "config": config})


def generate(component: TransformerSubgoalGenerator, state: np.ndarray, params: dict[str, Any]) -> tuple[list[tuple[Any, list[int]]], int]:
    """Use CaRL's generator tokenizer/model while retaining malformed outputs."""
    tokens, _ = component.env.tokenizer.x_y_tokenizer(x=state, y=state, training_goal=TrainingGoal.GENERATOR)
    with torch.inference_mode():
        sequences = component.get_network().generate(
            tokens.to("cpu"), max_new_tokens=params["max_new_tokens"],
            num_beams=params["num_beams"], num_return_sequences=params["num_return_sequences"], do_sample=False,
        ).tolist()
    return [(component.env.tokenizer.board_detokenizer(sequence), sequence) for sequence in sequences], int(tokens.numel())


def solution_actions(node: SearchTreeNode) -> list[int]:
    chunks: list[list[int]] = []
    current = node
    while current.parent_node is not None:
        if current.low_level_path:
            chunks.append(list(current.low_level_path))
        current = current.parent_node
    return [action for chunk in reversed(chunks) for action in chunk]


def verify_solution_chain(node: SearchTreeNode, trace: list[dict[str, Any]]) -> bool:
    """Verify every plan edge against transitions already executed by search.

    This does not make uncharged simulator calls. A separate offline replay can
    independently check the complete reconstructed action sequence.
    """
    current = node
    while current.parent_node is not None:
        parent = current.parent_node
        actions = list(current.low_level_path or [])
        if not actions:
            if state_hash(parent.state) != state_hash(current.state):
                return False
        elif not any(event.get("event") == "candidate" and event.get("source_hash") == state_hash(parent.state)
                     and event.get("actions") == actions and event.get("state_hashes", [None])[-1] == state_hash(current.state)
                     and event.get("reason") in ("target_reached", "task_solved") for event in trace):
            return False
        current = parent
    return True


def search(board: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    starting_swap = psutil.swap_memory().used
    identity = {key: board[key] for key in ("layout_id", "family", "source_family", "split", "board_rank")}
    result: dict[str, Any] = {**identity, "method": config["method"], "status": "completed", "solved": False,
        "transitions": 0, "solve_at_transition": None, "solution_length": None, "solution_actions": None,
        "solution_verified": None, "solution_verification_method": None,
        "wall_seconds": None, "solve_wall_seconds": None, "policy_calls": 0, "policy_input_tokens": 0,
        "value_calls": 0, "value_input_tokens": 0,
        "candidate_generation_calls": 0, "candidate_tokens": 0, "candidate_input_tokens": 0,
        "high_level_expansions": 0, "candidates_proposed": 0, "candidates_assessed": 0, "candidates_valid": 0,
        "controller_attempts": 0, "controller_attempts_completed": 0, "controller_hits": 0,
        "controller_attempts_missed": 0, "controller_cutoffs": 0,
        "controller_task_solved_before_hit": 0, "noop_actions": 0,
        "generation_seconds": 0.0, "controller_seconds": 0.0, "value_seconds": 0.0,
        "failure_reason": None, "target_rejections": {}, "candidate_outcomes": {}, "trace": []}
    try:
        initial, env = valid_start(board)
        policy_input, _ = env.tokenizer.x_y_tokenizer(x=(initial, initial), y=0, training_goal=TrainingGoal.CLLP)
        value_input, _ = env.tokenizer.x_y_tokenizer(x=initial, y=0, training_goal=TrainingGoal.VALUE)
        policy_token_count, value_token_count = int(policy_input.numel()), int(value_input.numel())
        for generator in WORKER["generators"].values():
            generator.env = env
        WORKER["cllp"].env = env
        WORKER["value"].env = env
        planner = AdasubsPlanner(initial, config["ks"])
        result["initial_state_hash"] = state_hash(initial)
        stop_mono, stop_epoch = config["stop_monotonic"], config["stop_epoch"]

        def expired() -> bool:
            return (time.monotonic() - started >= config["board_wall_seconds"]
                    or time.monotonic() >= stop_mono or time.time() >= stop_epoch)

        def pressure() -> bool:
            return (psutil.virtual_memory().percent >= 80
                    or psutil.swap_memory().used - starting_swap >= 128 * 1024 * 1024)

        while result["transitions"] < config["budget"] and not result["solved"]:
            if expired():
                result.update(status="timeout", failure_reason="wall_clock_limit")
                break
            if pressure():
                result.update(status="infra_error", solved=None, failure_reason="memory_or_swap_guard")
                break
            if result["high_level_expansions"] >= config["max_expansions"]:
                result["failure_reason"] = "expansion_safety_limit"
                break
            node = planner.get()
            if node is None:
                result["failure_reason"] = "queue_exhausted"
                break
            k = node.next_expand_with_k_generator
            if k not in config["ks"]:
                raise ValueError("planner selected an unconfigured generator")
            result["high_level_expansions"] += 1
            tick = time.monotonic()
            candidates, input_tokens = generate(WORKER["generators"][k], node.state, config["generation"])
            duration = time.monotonic() - tick
            result["generation_seconds"] += duration
            result["candidate_generation_calls"] += 1
            result["candidate_input_tokens"] += input_tokens
            result["candidate_tokens"] += sum(len(sequence) for _, sequence in candidates)
            result["candidates_proposed"] += len(candidates)
            result["trace"].append({"event": "generation", "k": k, "source_hash": state_hash(node.state),
                "source_rows": to_rows(node.state), "elapsed_seconds": time.monotonic() - started,
                "transitions": result["transitions"], "seconds": duration, "output_lengths": [len(x[1]) for x in candidates]})
            prepared: list[tuple[np.ndarray, dict[str, Any]]] = []
            for rank, (target, sequence) in enumerate(candidates):
                event: dict[str, Any] = {"event": "candidate", "k": k, "rank": rank,
                    "source_hash": state_hash(node.state), "token_count": len(sequence), "actions": [],
                    "state_hashes": [state_hash(node.state)]}
                result["trace"].append(event)
                result["candidates_assessed"] += 1
                if target is None:
                    event["reason"] = "undecodable_board"
                    continue
                try:
                    validate_subgoal(node.state, target)
                except ValueError as exc:
                    event.update(reason="invalid_target", detail=str(exc), target_rows=to_rows(target))
                    continue
                result["candidates_valid"] += 1
                event.update(target_hash=state_hash(target), target_rows=to_rows(target))
                prepared.append((target, event))
            for target, event in prepared:
                event["start_transitions"] = result["transitions"]
                if result["solved"]:
                    event["reason"] = "not_attempted_task_solved"
                    continue
                if pressure():
                    result.update(status="infra_error", solved=None, failure_reason="memory_or_swap_guard")
                    event["reason"] = "not_attempted_memory_guard"
                    continue
                if expired():
                    result.update(status="timeout", failure_reason="wall_clock_limit")
                    event["reason"] = "not_attempted_wall_limit"
                    continue
                if result["transitions"] >= config["budget"]:
                    result["failure_reason"] = "transition_budget"
                    event["reason"] = "not_attempted_transition_budget"
                    continue
                if planner.is_seen(target):
                    event["reason"] = "previously_seen_target"
                    continue
                result["controller_attempts"] += 1
                current = node.state.copy()
                event["noop_actions"] = 0
                tick = time.monotonic()
                hit = solved = False
                for _ in range(k):
                    if expired() or result["transitions"] >= config["budget"]:
                        break
                    logits = WORKER["cllp"].get_action(current, target)
                    result["policy_calls"] += 1
                    result["policy_input_tokens"] += policy_token_count
                    action = env.distribution_to_action(logits)
                    successor = np.asarray(env.next_state(current, action))
                    result["transitions"] += 1
                    if np.array_equal(current, successor):
                        result["noop_actions"] += 1
                        event["noop_actions"] += 1
                    current = successor
                    event["actions"].append(action)
                    event["state_hashes"].append(state_hash(current))
                    hit = np.array_equal(current, target)
                    solved = env.is_solved(current)
                    if hit or solved:
                        break
                result["controller_seconds"] += time.monotonic() - tick
                event.update(hit=bool(hit), task_solved=bool(solved), reached_rows=to_rows(current),
                             end_transitions=result["transitions"], elapsed_seconds=time.monotonic() - started)
                if hit:
                    result["controller_hits"] += 1
                    result["controller_attempts_completed"] += 1
                elif solved:
                    result["controller_task_solved_before_hit"] += 1
                elif len(event["actions"]) == k:
                    result["controller_attempts_completed"] += 1
                    result["controller_attempts_missed"] += 1
                else:
                    result["controller_cutoffs"] += 1
                if solved:
                    found = SearchTreeNode(current, 0.0, event["actions"], node)
                    actions = solution_actions(found)
                    event["reason"] = "task_solved"
                    verified = verify_solution_chain(found, result["trace"]) and env.is_solved(current)
                    if not verified:
                        raise ValueError("reconstructed solution does not match executed search edges")
                    result.update(solved=True, solve_at_transition=result["transitions"],
                                  solution_length=len(actions), solution_actions=actions,
                                  solution_verified=True,
                                  solution_verification_method="executed edge hashes and terminal goal; no extra simulator calls",
                                  solve_wall_seconds=time.monotonic() - started)
                elif hit:
                    tick = time.monotonic()
                    value = float(WORKER["value"].get_value(current))
                    result["value_seconds"] += time.monotonic() - tick
                    result["value_calls"] += 1
                    result["value_input_tokens"] += value_token_count
                    if not math.isfinite(value):
                        raise ValueError("nonfinite value model output")
                    event.update(reason="target_reached", value=value)
                    planner.add(SearchTreeNode(current, value, event["actions"], node))
                else:
                    if len(event["actions"]) == k:
                        event["reason"] = "controller_target_not_reached"
                    elif result["transitions"] >= config["budget"]:
                        event["reason"] = "controller_cutoff_transition_budget"
                    elif expired():
                        event["reason"] = "controller_cutoff_wall_limit"
                    else:
                        raise ValueError("controller attempt ended early without a cutoff")
            if result["status"] in ("timeout", "infra_error") or result["transitions"] >= config["budget"]:
                break
        if not result["solved"] and result["failure_reason"] is None:
            result["failure_reason"] = "transition_budget" if result["transitions"] >= config["budget"] else "stopped"
        outcomes = Counter(event.get("reason") for event in result["trace"] if event["event"] == "candidate")
        result["candidate_outcomes"] = dict(outcomes)
        result["target_rejections"] = {key: value for key, value in outcomes.items()
                                       if key in ("undecodable_board", "invalid_target", "previously_seen_target",
                                                  "controller_target_not_reached")}
    except Exception as exc:
        result.update(status="infra_error", solved=None, failure_reason=f"{type(exc).__name__}: {exc}")
    result["wall_seconds"] = time.monotonic() - started
    return result


def worker_job(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
    board, output = item
    result = search(board, WORKER["config"])
    atomic_json(Path(output), result)
    return {key: result[key] for key in ("layout_id", "family", "method", "status", "solved", "transitions")}


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--method", choices=tuple(SearchMethod), required=True)
    parser.add_argument("--fixed-k", type=int, choices=(1, 4, 8), default=8)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--board-wall-seconds", type=float, required=True)
    parser.add_argument("--max-expansions", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-return-sequences", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=145)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--stop-epoch", type=float, required=True)
    parser.add_argument("--stop-monotonic", type=float, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.budget < 1 or args.board_wall_seconds <= 0 or args.max_expansions < 1 or args.workers < 1
            or args.threads < 1 or args.num_return_sequences < 1 or args.num_beams < args.num_return_sequences):
        parser.error("invalid budget, worker count, expansion limit, or decoding settings")
    if args.workers * args.threads > len(os.sched_getaffinity(0)):
        parser.error("worker and thread settings exceed available CPU affinity")
    ks = [8, 4, 1] if args.method == SearchMethod.ADASUBS else [args.fixed_k]
    generation = {"max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams,
                  "num_return_sequences": args.num_return_sequences, "do_sample": False}
    config = {"method": args.method, "ks": ks, "generation": generation, "budget": args.budget,
              "board_wall_seconds": args.board_wall_seconds, "max_expansions": args.max_expansions,
              "threads": args.threads, "seed": args.seed, "stop_epoch": args.stop_epoch,
              "stop_monotonic": args.stop_monotonic}
    boards = [board for board in read_jsonl(args.boards) if board["split"] == args.split]
    if not boards or len({board["layout_id"] for board in boards}) != len(boards):
        raise ValueError("empty board list or duplicate layouts within split")
    for board in boards:
        valid_start(board)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = [*(GENERATORS[k] for k in ks), CLLP, VALUE]
    contract = {"schema_version": 1, "config": config, "workers": args.workers,
                "source_sha256": file_sha(Path(__file__).resolve()),
                "input": str(args.boards.resolve()), "input_sha256": file_sha(args.boards),
                "boards": [{key: row[key] for key in ("layout_id", "family", "source_family", "split", "board_rank")}
                           for row in boards], "checkpoint_sha256": checkpoint_hashes(checkpoints),
                "value_convention": "TransformerValue returns negative predicted distance; higher value first",
                "controller_contract": "CLLP-8 receives current full state and proposed full target state",
                "transition_accounting": "every env.next_state in all attempted CLLP rollouts, including no-ops; generator makes no simulator calls",
                "token_accounting": "input token positions measured from CaRL tokenizers; generated token positions include special/padding tokens", "generator_simulator_transitions": 0,
                "controller_accounting": "attempts started include hits, full-horizon misses, cutoffs, and task completion before target; completed means hits plus full-horizon misses",
                "limit_note": "wall cutoff checked between neural calls; a single call may overrun its per-board limit"}
    cp = args.output / "contract.json"
    if cp.exists():
        if not args.resume or json.loads(cp.read_text()) != contract:
            raise ValueError("existing output requires --resume with exactly the same contract")
    else:
        atomic_json(cp, contract)
    shards = args.output / "shards"
    shards.mkdir(exist_ok=True)
    pending = [(board, str(shards / f"{board['layout_id']}__{args.method}.json")) for board in boards
               if not (shards / f"{board['layout_id']}__{args.method}.json").is_file()]
    monitor = ResourceMonitor(started)
    monitor.start()
    atomic_json(args.output / "status.json", {"state": "running", "planned": len(boards),
        "completed": len(boards) - len(pending), "method": args.method, **monitor.snapshot()})
    try:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                initializer=initialize_worker, initargs=(config,)) as pool:
            futures = [pool.submit(worker_job, job) for job in pending]
            for future in concurrent.futures.as_completed(futures):
                item = future.result()
                complete = len(list(shards.glob("*.json")))
                atomic_json(args.output / "status.json", {"state": "running", "planned": len(boards),
                    "completed": complete, "method": args.method, "latest": item, **monitor.snapshot()})
    except Exception as exc:
        atomic_json(args.output / "status.json", {"state": "failed", "planned": len(boards),
            "completed": len(list(shards.glob("*.json"))), "method": args.method,
            "error": f"{type(exc).__name__}: {exc}", **monitor.snapshot()})
        raise
    finally:
        monitor.stop()
    complete = len(list(shards.glob("*.json")))
    atomic_json(args.output / "status.json", {"state": "complete" if complete == len(boards) else "partial",
        "planned": len(boards), "completed": complete, "method": args.method, **monitor.snapshot()})


if __name__ == "__main__":
    main()
