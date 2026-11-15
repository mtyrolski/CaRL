#!/usr/bin/env python3
"""Run frozen CaRL policies on pre-split Sokoban pairs using local CPUs.

CLLP checkpoints receive the current and complete target state. The ordinary
policy receives only the current state and is an unconditioned baseline. Each
board/model shard is written atomically; missing shards are not failures.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import platform
import resource
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
from carl.icl.boards import to_rows
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from scripts.figure1_cllp_matched import Pair
from scripts.figure1_cllp_matched import model_predictor
from scripts.figure1_cllp_matched import rollout_pairs
from scripts.overnight_cpu_memo import MemoPredictor

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "rl-data/validation/sokoban/components/full_data"
OFFSETS = (1, 2, 4, 8, 16, 32)
CAP = 32


class ModelId(StrEnum):
    CLLP_1 = "cllp1"
    CLLP_4 = "cllp4"
    CLLP_8 = "cllp8"
    CLLP_16 = "cllp16"
    CLLP_32 = "cllp32"
    POLICY = "policy94820"


class RunState(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


CHECKPOINTS = {
    ModelId.CLLP_1: BASE / "cllp/1/checkpoint-149248",
    ModelId.CLLP_4: BASE / "cllp/4/checkpoint-587940",
    ModelId.CLLP_8: BASE / "cllp/8/checkpoint-167585",
    ModelId.CLLP_16: BASE / "cllp/16/checkpoint-37224",
    ModelId.CLLP_32: BASE / "cllp/32/checkpoint-20045",
    ModelId.POLICY: BASE / "policy/checkpoint-94820",
}
DEFAULT_MODELS = (ModelId.CLLP_8, ModelId.CLLP_32, ModelId.CLLP_4,
                  ModelId.CLLP_1, ModelId.CLLP_16, ModelId.POLICY)
FAMILY_BOXES = {"12x12x4": 4, "12x12x10": 10}
WORKER: dict[str, Any] = {}


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        result = [json.loads(line) for line in stream if line.strip()]
    if not all(isinstance(row, dict) for row in result):
        raise ValueError("Expected JSON objects in pair file")
    return result


def select_groups(rows: list[dict[str, Any]], split: str, max_boards: int | None,
                  *, max_contexts: int = 1) -> list[dict[str, Any]]:
    """Select a deterministic prefix per family, keeping every pair per board."""
    if (max_boards is not None and max_boards < 1) or max_contexts not in (1, 2):
        raise ValueError("max-boards must be positive and max-contexts must be 1 or 2")
    ids: set[str] = set()
    split_by_layout: dict[str, str] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        required = {"pair_id", "layout_id", "family", "source_family", "split", "trajectory_id", "i", "k",
                    "board_rank", "context_rank", "context_id",
                    "source", "target", "source_hash", "target_hash"}
        if not required <= row.keys():
            raise ValueError(f"Pair is missing keys: {sorted(required - row.keys())}")
        if row["pair_id"] in ids:
            raise ValueError("Duplicate pair_id")
        ids.add(row["pair_id"])
        if row["family"] not in FAMILY_BOXES or row["split"] not in {"development", "evaluation"}:
            raise ValueError("Unknown family or split")
        if row["layout_id"] in split_by_layout and split_by_layout[row["layout_id"]] != row["split"]:
            raise ValueError("A canonical layout occurs in more than one split")
        split_by_layout[row["layout_id"]] = row["split"]
        if (not isinstance(row["board_rank"], int) or row["board_rank"] < 0
                or row["context_rank"] not in (0, 1)):
            raise ValueError("Invalid frozen board/context rank")
        if (row["split"] == split and row["context_rank"] < max_contexts
                and (max_boards is None or row["board_rank"] < max_boards)):
            # Teacher actions remain in the dataset; never send them to workers.
            visible = {key: row[key] for key in required}
            grouped.setdefault((row["family"], row["layout_id"]), []).append(visible)
    selected: list[dict[str, Any]] = []
    for family in FAMILY_BOXES:
        boards = [(layout, records) for (fam, layout), records in grouped.items() if fam == family]
        if not boards:
            raise ValueError(f"No {split} boards for {family}")
        boards.sort(key=lambda item: item[1][0]["board_rank"])
        board_ranks = [records[0]["board_rank"] for _, records in boards]
        if len(set(board_ranks)) != len(board_ranks):
            raise ValueError("A board rank identifies more than one layout")
        for layout, records in boards:
            sources = {row["source_family"] for row in records}
            trajectories = {row["trajectory_id"] for row in records}
            ranks = {row["board_rank"] for row in records}
            if len(sources) != 1 or len(trajectories) != 1 or len(ranks) != 1:
                raise ValueError("A board shard must have one source family and trajectory")
            for rank in {row["context_rank"] for row in records}:
                context = [row for row in records if row["context_rank"] == rank]
                if (len({row["context_id"] for row in context}) != 1
                        or len({row["source_hash"] for row in context}) != 1
                        or len({row["k"] for row in context}) != len(context)):
                    raise ValueError("Inconsistent context or duplicate offset")
            records.sort(key=lambda row: (row["context_rank"], row["k"]))
            selected.append({"family": family, "layout_id": layout, "split": split,
                             "source_family": next(iter(sources)), "trajectory_id": next(iter(trajectories)),
                             "board_rank": records[0]["board_rank"],
                             "records": records})
    # Dispatch paired methods on both families from the earliest rank onward,
    # so a cutoff leaves measurable coverage in each family.
    family_order = {family: index for index, family in enumerate(FAMILY_BOXES)}
    selected.sort(key=lambda group: (group["board_rank"], family_order[group["family"]]))
    return selected


def materialise_pairs(group: dict[str, Any]) -> list[Pair]:
    """Validate dimensions, identity, targets and hashes before any inference."""
    pairs: list[Pair] = []
    expected_boxes = FAMILY_BOXES[group["family"]]
    env = None
    for row in group["records"]:
        source, target = from_rows(row["source"]), from_rows(row["target"])
        if source.shape != (12, 12, 7) or target.shape != source.shape:
            raise ValueError("Expected 12x12 full boards")
        for board in (source, target):
            cells = board.argmax(-1)
            if (int(np.isin(cells, (3, 4)).sum()) != expected_boxes
                    or int(np.isin(cells, (2, 3, 6)).sum()) != expected_boxes):
                raise ValueError("Wrong box or goal count")
        if (canonical_layout_id(source) != group["layout_id"] or state_hash(source) != row["source_hash"]
                or state_hash(target) != row["target_hash"]):
            raise ValueError("Layout or state hash mismatch")
        validate_subgoal(source, target)
        if env is None:
            env = environment(source)
        if np.array_equal(source, target) or env.is_solved(source):
            raise ValueError("Source is solved or target is already satisfied")
        if row["k"] not in OFFSETS or not isinstance(row["i"], int) or row["i"] < 0:
            raise ValueError("Invalid trajectory offset/index")
        pairs.append(Pair(row["pair_id"], row["layout_id"], row["trajectory_id"], row["context_rank"],
                          row["i"], row["k"], row["source_hash"], row["target_hash"], source, target))
    return pairs


def policy_predictor(model: Any, tokenizer: Any, batch_size: int):
    """Use only the current board for the ordinary, unconditioned policy."""
    def predict(states: list[np.ndarray], targets: list[np.ndarray]) -> list[int]:
        if len(states) != len(targets):
            raise ValueError("Mismatched rollout batch")
        actions = []
        for start in range(0, len(states), batch_size):
            tokens = torch.cat([tokenizer.x_y_tokenizer(state, 0, TrainingGoal.POLICY)[0]
                                for state in states[start:start + batch_size]])
            if tokens.shape[1] != 146 or tokens.shape[1] > model.config.max_position_embeddings:
                raise ValueError("Unsupported ordinary policy input length")
            with torch.inference_mode():
                logits = model(tokens).logits
            if logits.shape != (len(tokens), 4) or not torch.isfinite(logits).all():
                raise ValueError("Invalid ordinary policy logits")
            actions.extend(int(value) for value in logits.argmax(-1).tolist())
        return actions
    return predict


def cycle_info(hashes: list[str]) -> tuple[bool, int | None, int | None]:
    """Report the first return after a changed state; count no-ops separately."""
    first: dict[str, int] = {}
    for step, value in enumerate(hashes):
        if value in first:
            begin = first[value]
            if any(other != value for other in hashes[begin + 1:step]):
                return True, step - begin, step
            continue
        first[value] = step
    return False, None, None


def decorate_trial(pair: Pair, row: dict[str, Any], group: dict[str, Any], model_id: ModelId,
                   env: Any | None = None) -> dict[str, Any]:
    """Independently replay a completed rollout, including simultaneous goal/hit."""
    expected = {"model": str(model_id), "pair_id": pair.pair_id, "layout_id": pair.layout_id,
                "trajectory_id": pair.trajectory_id, "rank": pair.rank, "i": pair.position,
                "k": pair.k, "source_hash": pair.source_hash, "target_hash": pair.target_hash,
                "action_cap": CAP}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Trial identity differs from the frozen pair")
    actions = row.get("actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= CAP:
        raise ValueError("Invalid number of primitive actions")
    env = env or environment(pair.source)
    state = pair.source.copy()
    hashes = [state_hash(state)]
    no_ops = 0
    stop = None
    for step, action in enumerate(actions, 1):
        if stop is not None or not isinstance(action, int) or isinstance(action, bool) or action not in range(4):
            raise ValueError("Invalid action or action after stopping")
        before = state
        state = np.asarray(env.next_state(before, action))
        hashes.append(state_hash(state))
        no_ops += int(np.array_equal(before, state))
        stop = ("exact_target" if np.array_equal(state, pair.target) else
                "task_terminal" if env.done else "action_limit" if step == CAP else None)
    hit = stop == "exact_target"
    if (stop is None or row.get("state_hashes") != hashes or row.get("no_op_count") != no_ops
            or row.get("attempted_actions") != len(actions) or row.get("stop_reason") != stop
            or row.get("exact_hit") is not hit or row.get("first_hit") != (len(actions) if hit else None)):
        raise ValueError("Trial differs from its independent replay")
    cells = state.argmax(-1)
    players = np.argwhere(np.isin(cells, (5, 6)))
    if len(players) != 1:
        raise ValueError("Expected exactly one player")
    cycle, cycle_length, cycle_first_seen = cycle_info(row["state_hashes"])
    original = next(item for item in group["records"] if item["pair_id"] == pair.pair_id)
    return {**row, "status": RunState.COMPLETE, "model_id": model_id,
            "family": group["family"], "source_family": group["source_family"], "split": group["split"],
            "context_id": original["context_id"], "context_rank": original["context_rank"],
            "board_rank": original["board_rank"], "task_solved": bool(env.is_solved(state)),
            "final_state": to_rows(state), "final_player": players[0].tolist(),
            "cycle_detected": cycle, "cycle_length": cycle_length, "cycle_first_repeat_step": cycle_first_seen}


def worker_init(threads: int, expected_hashes: dict[str, str]) -> None:
    """Spawned workers keep loaded models and never access API/network/GPU."""
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    WORKER.clear()
    WORKER.update(models={}, threads=threads, expected_hashes=expected_hashes)


def shard_path(output: Path, group: dict[str, Any], model_id: ModelId) -> Path:
    return output / "shards" / group["split"] / group["family"] / str(model_id) / f"{group['layout_id']}.json"


def evaluate_shard(group: dict[str, Any], model_value: str, output: str, contract_sha: str, batch_size: int,
                   use_memo: bool, memo_max_entries: int) -> dict:
    model_id = ModelId(model_value)
    checkpoint = CHECKPOINTS[model_id]
    model = WORKER["models"].get(model_id)
    if model is None:
        for asset in (checkpoint / "config.json", checkpoint / "pytorch_model.bin"):
            if file_sha(asset) != WORKER["expected_hashes"].get(str(asset)):
                raise ValueError("Checkpoint changed after the run contract was frozen")
        model = BertForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, use_safetensors=False).to("cpu").float().eval()
        model.requires_grad_(False)
        WORKER["models"][model_id] = model
    pairs = materialise_pairs(group)
    env = environment(pairs[0].source)
    tokenizer = env.tokenizer
    predict = (policy_predictor(model, tokenizer, batch_size) if model_id == ModelId.POLICY else
               model_predictor(model, tokenizer, torch.device("cpu"), batch_size))
    memo = None
    if use_memo:
        memo = MemoPredictor(predict, conditioned=model_id != ModelId.POLICY,
                             batch_size=batch_size, max_entries=memo_max_entries)
        predict = memo
    # next_state restores the complete supplied state on every call, so these
    # sequential steps can share one environment without sharing episode state.
    raw = rollout_pairs(pairs, predict, str(model_id), cap=CAP, make_environment=lambda _state: env)
    by_id = {pair.pair_id: pair for pair in pairs}
    trials = [decorate_trial(by_id[row["pair_id"]], row, group, model_id, env) for row in raw]
    if len(trials) != len(pairs):
        raise ValueError("Missing trial in completed shard")
    shard = {"schema_version": 1, "status": RunState.COMPLETE, "model_id": model_id,
             "family": group["family"], "source_family": group["source_family"], "split": group["split"],
             "layout_id": group["layout_id"], "planned_pair_ids": [pair.pair_id for pair in pairs],
             "contract_sha256": contract_sha, "trials": trials}
    memo_stats = memo.stats.to_dict() if memo is not None else None
    if memo_stats is not None:
        if (memo_stats["requests"] != sum(row["attempted_actions"] for row in trials)
                or memo_stats["unique_decisions"] + memo_stats["previous_cache_hits"]
                + memo_stats["same_batch_hits"] != memo_stats["requests"]):
            raise ValueError("Memo statistics do not match executed policy decisions")
        shard["memo_stats"] = memo_stats
    path = shard_path(Path(output), group, model_id)
    atomic_json(path, shard)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"shard": str(path), "worker_pid": os.getpid(), "trials": len(trials),
            "actions": sum(row["attempted_actions"] for row in trials),
            "memo_stats": memo_stats, "worker_peak_rss_bytes": int(usage.ru_maxrss) * 1024,
            "worker_cpu_seconds": usage.ru_utime + usage.ru_stime}


def validate_existing_shard(path: Path, group: dict[str, Any], model_id: ModelId, contract_sha: str,
                            use_memo: bool = False) -> dict:
    shard = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version": 1, "status": RunState.COMPLETE, "model_id": model_id,
                "family": group["family"], "source_family": group["source_family"], "split": group["split"],
                "layout_id": group["layout_id"], "contract_sha256": contract_sha}
    if any(shard.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Saved shard identity or contract differs: {path}")
    pairs = materialise_pairs(group)
    by_id = {pair.pair_id: pair for pair in pairs}
    env = environment(pairs[0].source)
    trials = shard.get("trials")
    if (shard.get("planned_pair_ids") != [pair.pair_id for pair in pairs] or not isinstance(trials, list)
            or len(trials) != len(pairs) or {row.get("pair_id") for row in trials} != set(by_id)):
        raise ValueError(f"Saved shard is incomplete or duplicated: {path}")
    for row in trials:
        pair = by_id[row["pair_id"]]
        expected_row = decorate_trial(pair, row, group, model_id, env)
        if expected_row != row:
            raise ValueError(f"Saved trial differs from its independent replay: {path}")
    actions = sum(row["attempted_actions"] for row in trials)
    memo_stats = shard.get("memo_stats")
    if use_memo:
        if (not isinstance(memo_stats, dict) or memo_stats.get("requests") != actions
                or memo_stats.get("unique_decisions", -1) + memo_stats.get("previous_cache_hits", -1)
                + memo_stats.get("same_batch_hits", -1) != actions):
            raise ValueError(f"Saved memo statistics disagree with executed actions: {path}")
    elif memo_stats is not None:
        raise ValueError(f"Saved shard unexpectedly includes memo statistics: {path}")
    return {"trials": len(trials), "actions": actions, "memo_stats": memo_stats}


def resource_snapshot() -> dict[str, Any]:
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    process = psutil.Process()
    workers = []
    for child in process.children(recursive=True):
        try:
            if child.is_running():
                workers.append({"pid": child.pid, "rss_bytes": child.memory_info().rss})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"host_total_bytes": memory.total, "host_available_bytes": memory.available,
            "host_used_fraction": (memory.total - memory.available) / memory.total,
            "swap_used_bytes": swap.used, "swap_total_bytes": swap.total,
            "children": workers, "children_rss_bytes": sum(row["rss_bytes"] for row in workers)}


def controller_contract(pairs_path: Path, groups: list[dict[str, Any]], models: list[ModelId],
                        args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    source_paths = [Path(__file__).resolve(), ROOT / "scripts/figure1_cllp_matched.py",
                    ROOT / "scripts/training/calibration.py", ROOT / "carl/icl/boards.py",
                    ROOT / "carl/icl/data_sources/sokoban.py", ROOT / "carl/environment/sokoban/env.py",
                    ROOT / "carl/environment/sokoban/core.py", ROOT / "carl/environment/sokoban/tokenizer.py",
                    ROOT / "carl/environment/training_goal.py"]
    if args.use_memo:
        source_paths.append(ROOT / "scripts/overnight_cpu_memo.py")
    for model in models:
        source_paths.extend((CHECKPOINTS[model] / "config.json", CHECKPOINTS[model] / "pytorch_model.bin"))
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required controller assets are missing: {missing}")
    shards = [{"family": group["family"], "layout_id": group["layout_id"], "model_id": model,
               "pair_ids": [row["pair_id"] for row in group["records"]]} for group in groups for model in models]
    availability = []
    for group in groups:
        contexts = []
        for rank in sorted({row["context_rank"] for row in group["records"]}):
            rows = [row for row in group["records"] if row["context_rank"] == rank]
            available = sorted(row["k"] for row in rows)
            contexts.append({"context_rank": rank, "context_id": rows[0]["context_id"],
                             "available_offsets": available,
                             "unavailable_offsets": [k for k in OFFSETS if k not in available]})
        available_contexts = {item["context_rank"] for item in contexts}
        availability.append({"family": group["family"], "layout_id": group["layout_id"],
                             "board_rank": group["board_rank"], "contexts": contexts,
                             "unavailable_context_ranks": [rank for rank in range(args.max_contexts)
                                                           if rank not in available_contexts]})
    plan = {"schema_version": 1, "split": args.split, "layout_limit_per_family": args.max_boards,
            "contexts_per_layout": args.max_contexts, "seed": args.seed,
            "dispatch_order": "ascending board_rank, alternating 12x12x4 then 12x12x10, all models per board",
            "family_layouts": {family: [group["layout_id"] for group in groups if group["family"] == family]
                               for family in FAMILY_BOXES},
            "model_ids": models, "availability": availability, "planned_shards": shards,
            "planned_trials": sum(len(row["pair_ids"]) for row in shards)}
    contract = {"schema_version": 1, "input_pairs_path": str(pairs_path.resolve()), "input_sha256": file_sha(pairs_path),
                "plan_sha256": digest(plan), "source_sha256": {str(path): file_sha(path) for path in source_paths},
                "models": {str(model): {"checkpoint": str(CHECKPOINTS[model]),
                                          "input": "current_board_only" if model == ModelId.POLICY
                                          else "current_and_complete_target_board"} for model in models},
                "device": "cpu", "dtype": "float32", "decoder": "greedy_argmax", "inference_mode": "eval",
                "tie_break": "first maximum in action order UP, DOWN, LEFT, RIGHT",
                "memo": {"enabled": args.use_memo, "max_entries": args.memo_max_entries if args.use_memo else None,
                         "scope": "one board and one fixed model",
                         "key": "full current and target states for CLLP; current state only for ordinary policy",
                         "simulator_steps_skipped": 0},
                "action_mask": False, "action_cap": CAP, "seed": args.seed, "workers": args.workers,
                "threads_per_worker": args.threads, "blas_threads": args.threads, "batch_size": args.batch_size,
                "cpu_affinity": sorted(os.sched_getaffinity(0)), "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__, "psutil": psutil.__version__,
                "stop": "first full-state hit, actual task terminal, or 32 attempted actions including no-ops"}
    return plan, contract


def status_payload(state: RunState, plan: dict, completed: int, errors: list[dict], started: float,
                   resources: dict, reason: str | None = None) -> dict:
    return {"schema_version": 1, "state": state, "planned_shards": len(plan["planned_shards"]),
            "completed_shards": completed, "missing_shards": len(plan["planned_shards"]) - completed,
            "planned_trials": plan["planned_trials"], "errors": errors, "elapsed_seconds": time.monotonic() - started,
            "stop_reason": reason, "resources": resources}


def execute(args: argparse.Namespace, groups: list[dict[str, Any]], models: list[ModelId]) -> dict:
    """Run a fully frozen plan; dispatch is stopped at deadlines/resource limits."""
    affinity = len(os.sched_getaffinity(0))
    if args.workers < 1 or args.threads < 1 or args.workers * args.threads > affinity:
        raise ValueError("Worker × thread count exceeds CPU affinity")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.use_memo and args.memo_max_entries < 1:
        raise ValueError("memo-max-entries must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    plan, contract = controller_contract(args.pairs, groups, models, args)
    contract_sha = digest(contract)
    with FileLock(str(output / ".run.lock"), timeout=0):
        cp, pp = output / "contract.json", output / "plan.json"
        if cp.exists() or pp.exists():
            if not args.resume or not cp.exists() or not pp.exists():
                raise ValueError("Existing output requires --resume with both frozen files")
            if json.loads(cp.read_text()) != json.loads(json_bytes(contract)) or json.loads(pp.read_text()) != json.loads(json_bytes(plan)):
                raise ValueError("Saved run contract or evaluation prefix differs")
        else:
            if args.resume:
                raise ValueError("No previous frozen contract to resume")
            atomic_json(cp, contract)
            atomic_json(pp, plan)
        pending_jobs = []
        memo_totals = {key: 0 for key in ("requests", "previous_cache_hits", "same_batch_hits", "unique_decisions",
                                          "forward_batches", "evictions")}
        completed, reused_trials, reused_actions = 0, 0, 0
        for group in groups:
            for model in models:
                path = shard_path(output, group, model)
                if path.exists():
                    checked = validate_existing_shard(path, group, model, contract_sha, args.use_memo)
                    completed += 1
                    reused_trials += checked["trials"]
                    reused_actions += checked["actions"]
                    if checked["memo_stats"] is not None:
                        for key in memo_totals:
                            memo_totals[key] += checked["memo_stats"][key]
                else:
                    pending_jobs.append((group, model))
        before = resource_snapshot()
        resources = {"initial": before, "peak_host_used_fraction": before["host_used_fraction"],
                     "peak_children_rss_bytes": before["children_rss_bytes"], "peak_worker_rss_bytes": 0,
                     "peak_swap_used_bytes": before["swap_used_bytes"], "worker_peak_rss_bytes": {},
                     "worker_cpu_seconds": {}, "new_trials": 0, "new_actions": 0,
                     "reused_trials": reused_trials, "reused_actions": reused_actions,
                     "memo_totals": memo_totals if args.use_memo else None}
        errors: list[dict[str, Any]] = []
        reason = None
        atomic_json(output / "status.json", status_payload(RunState.RUNNING, plan, completed, errors, started, resources))
        # The spawn context gives each persistent process a clean Torch runtime.
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(args.threads)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        iterator = iter(pending_jobs)
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                                    initializer=worker_init,
                                                    initargs=(args.threads, contract["source_sha256"])) as pool:
            inflight: dict[concurrent.futures.Future, tuple[dict, ModelId]] = {}
            exhausted = False
            last_progress = 0.0
            while inflight or not exhausted:
                snap = resource_snapshot()
                resources["peak_host_used_fraction"] = max(resources["peak_host_used_fraction"], snap["host_used_fraction"])
                resources["peak_children_rss_bytes"] = max(resources["peak_children_rss_bytes"], snap["children_rss_bytes"])
                resources["peak_swap_used_bytes"] = max(resources["peak_swap_used_bytes"], snap["swap_used_bytes"])
                for child in snap["children"]:
                    resources["peak_worker_rss_bytes"] = max(resources["peak_worker_rss_bytes"], child["rss_bytes"])
                if snap["host_used_fraction"] >= .80:
                    reason = "host_memory_limit"
                elif snap["swap_used_bytes"] - before["swap_used_bytes"] >= 128 * 1024 * 1024:
                    reason = "swap_growth_limit"
                elif args.stop_epoch is not None and time.time() >= args.stop_epoch:
                    reason = "stage_epoch_cutoff"
                elif args.stop_monotonic is not None and time.monotonic() >= args.stop_monotonic:
                    reason = "stage_monotonic_cutoff"
                if reason:
                    exhausted = True
                while not exhausted and len(inflight) < args.workers:
                    try:
                        group, model = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    future = pool.submit(evaluate_shard, group, str(model), str(output), contract_sha, args.batch_size,
                                         args.use_memo, args.memo_max_entries)
                    inflight[future] = (group, model)
                if not inflight:
                    break
                done, _ = concurrent.futures.wait(inflight, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    group, model = inflight.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:  # Preserve every unrelated shard and its missing denominator.
                        errors.append({"family": group["family"], "layout_id": group["layout_id"],
                                       "model_id": model, "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    completed += 1
                    resources["new_trials"] += outcome["trials"]
                    resources["new_actions"] += outcome["actions"]
                    if outcome["memo_stats"] is not None:
                        for key in memo_totals:
                            memo_totals[key] += outcome["memo_stats"][key]
                    pid = str(outcome["worker_pid"])
                    resources["worker_peak_rss_bytes"][pid] = max(
                        resources["worker_peak_rss_bytes"].get(pid, 0), outcome["worker_peak_rss_bytes"])
                    resources["worker_cpu_seconds"][pid] = max(
                        resources["worker_cpu_seconds"].get(pid, 0), outcome["worker_cpu_seconds"])
                now = time.monotonic()
                if done or now - last_progress >= 60:
                    atomic_json(output / "status.json", status_payload(RunState.RUNNING, plan, completed, errors,
                                                                       started, resources, reason))
                    if now - last_progress >= 60:
                        print(json.dumps({"completed_shards": completed, "planned_shards": len(plan["planned_shards"]),
                                          "elapsed_seconds": now - started, "stop_reason": reason}), flush=True)
                        last_progress = now
        state = (RunState.COMPLETE if completed == len(plan["planned_shards"]) and not errors else
                 RunState.FAILED if errors else RunState.PARTIAL)
        result = status_payload(state, plan, completed, errors, started, resources, reason)
        result["elapsed_cpu_seconds"] = sum(resources["worker_cpu_seconds"].values())
        result["new_trials_per_second"] = resources["new_trials"] / max(result["elapsed_seconds"], 1e-9)
        result["new_actions_per_second"] = resources["new_actions"] / max(result["elapsed_seconds"], 1e-9)
        atomic_json(output / "status.json", result)
        print(json.dumps({"state": state, "completed_shards": completed,
                          "planned_shards": len(plan["planned_shards"]), "elapsed_seconds": result["elapsed_seconds"]}),
              flush=True)
        return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "benchmark"):
        item = sub.add_parser(name)
        item.add_argument("--pairs", required=True, type=Path)
        item.add_argument("--output", required=True, type=Path)
        item.add_argument("--threads", type=int, default=1)
        item.add_argument("--batch-size", type=int, default=16)
        item.add_argument("--max-boards", type=int, help="board_rank prefix independently per family")
        item.add_argument("--max-contexts", type=int, choices=(1, 2), default=1)
        item.add_argument("--use-memo", action="store_true", help="Cache deterministic decisions per board/model")
        item.add_argument("--memo-max-entries", type=int, default=8192)
        item.add_argument("--seed", type=int, default=20261115)
        item.add_argument("--stop-epoch", type=float)
        item.add_argument("--stop-monotonic", type=float)
        item.add_argument("--resume", action="store_true")
        if name == "run":
            item.add_argument("--split", choices=("development", "evaluation"), default="evaluation")
            item.add_argument("--workers", type=int, required=True)
            item.add_argument("--models", nargs="+", choices=tuple(ModelId), default=list(DEFAULT_MODELS))
        else:
            item.set_defaults(split="development")
            item.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
            item.add_argument("--configs", nargs="+", help="Explicit worker/thread pairs, e.g. 1x8 2x4")
            item.add_argument("--models", nargs="+", choices=tuple(ModelId), default=[ModelId.CLLP_8])
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    models = [ModelId(value) for value in args.models]
    if not models or len(set(models)) != len(models):
        parser.error("Models must be unique")
    records = read_jsonl(args.pairs)
    groups = select_groups(records, args.split, args.max_boards, max_contexts=args.max_contexts)
    if args.command == "run":
        result = execute(args, groups, models)
        if result["state"] != RunState.COMPLETE:
            raise SystemExit(1)
    else:
        if args.max_boards is None:
            parser.error("Benchmark requires a development-board prefix via --max-boards")
        root = args.output
        root.mkdir(parents=True, exist_ok=True)
        summaries = []
        try:
            configurations = ([tuple(int(value) for value in config.lower().split("x")) for config in args.configs]
                              if args.configs else [(workers, args.threads) for workers in args.workers])
            if any(len(config) != 2 for config in configurations) or len(set(configurations)) != len(configurations):
                raise ValueError("Expected distinct WORKERSxTHREADS pairs")
        except ValueError as error:
            parser.error(f"Invalid benchmark --configs: {error}")
        for workers, threads in configurations:
            settings = argparse.Namespace(**vars(args))
            settings.workers = workers
            settings.threads = threads
            settings.output = root / f"workers_{workers}_threads_{threads}"
            result = execute(settings, groups, models)
            summaries.append({"workers": workers, "threads": settings.threads, "state": result["state"],
                              "elapsed_seconds": result["elapsed_seconds"],
                              "trials_per_second": result["new_trials_per_second"],
                              "actions_per_second": result["new_actions_per_second"],
                              "cpu_seconds": result["elapsed_cpu_seconds"], "resources": result["resources"]})
            atomic_json(root / "benchmark_summary.json", {"runs": summaries, "split": "development",
                                                            "model_ids": models})
            if result["state"] != RunState.COMPLETE:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
