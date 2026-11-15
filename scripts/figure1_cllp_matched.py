"""Evaluate frozen CaRL CLLPs on the exact 20-layout Figure 1 pairs.

The controller receives the current board and the complete target board.  This
is an exact-state controller experiment, not an intent-conditioned rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock
from transformers import BertForSequenceClassification

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from carl.icl.data_sources.sokoban import verify_solution
from scripts.training.calibration import pair_tokens

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/icl/reachability_proposal_hard_12_v1"
CHECKPOINTS = {
    "CLLP-8": ROOT / "rl-data/validation/sokoban/components/full_data/cllp/8/checkpoint-167585",
    "CLLP-32": ROOT / "rl-data/validation/sokoban/components/full_data/cllp/32/checkpoint-20045",
}
OFFSETS = (1, 2, 4, 8, 12, 16)
ACTION_CAP = 32


@dataclass(frozen=True)
class Pair:
    """A verified source and exact target, with original Figure 1 identity."""

    pair_id: str
    layout_id: str
    trajectory_id: str
    rank: int
    position: int
    k: int
    source_hash: str
    target_hash: str
    source: np.ndarray
    target: np.ndarray


def file_sha256(path: Path) -> str:
    """Hash without copying a checkpoint into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Publish a complete JSON document in one filesystem operation."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Preserve every finished trial after an interruption."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    """Read the trusted local trajectory archive without extracting its files."""
    with tarfile.open(path, "r:gz") as archive:
        candidates = [member for member in archive.getmembers() if member.name.endswith("/all.json")]
        if len(candidates) != 1 or not candidates[0].isfile():
            raise ValueError("Expected exactly one regular all.json in the source bundle")
        source = archive.extractfile(candidates[0])
        if source is None:
            raise ValueError("Cannot read trajectory source")
        rows = json.load(source)
    if not isinstance(rows, list):
        raise ValueError("Trajectory source must be a list")
    return rows


def prepare_pairs(
    raw_pairs: Sequence[dict[str, Any]],
    raw_states: dict[str, list[str]],
    source_rows: Sequence[dict[str, Any]],
    *,
    layout_count: int = 20,
    offsets: tuple[int, ...] = OFFSETS,
    board_size: int = 12,
    boxes: int = 10,
) -> list[Pair]:
    """Freeze pairs by original rank and independently replay their trajectories."""
    if layout_count < 1 or not offsets or len(set(offsets)) != len(offsets):
        raise ValueError("Layout count and distinct offsets are required")
    selected = [row for row in raw_pairs if isinstance(row.get("rank"), int) and 0 <= row["rank"] < layout_count]
    if len(selected) != layout_count * len(offsets):
        raise ValueError("Selected pair count does not match the complete layout/offset grid")
    by_trajectory = {row["id"]: row for row in source_rows}
    if len(by_trajectory) != len(source_rows):
        raise ValueError("Duplicate trajectory ID in source bundle")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        groups[row["rank"]].append(row)
    if set(groups) != set(range(layout_count)):
        raise ValueError("Expected one complete group for each selected rank")

    pairs: list[Pair] = []
    seen_ids: set[str] = set()
    seen_layouts: set[str] = set()
    for rank, rows in sorted(groups.items()):
        if {row["k"] for row in rows} != set(offsets) or len({row["layout_id"] for row in rows}) != 1:
            raise ValueError("Every layout must contain every offset exactly once")
        if len({row["trajectory_id"] for row in rows}) != 1 or len({row["i"] for row in rows}) != 1:
            raise ValueError("Offsets for a layout must share the frozen source context")
        source_row = by_trajectory[rows[0]["trajectory_id"]]
        source = from_rows(source_row["initial_state"])
        if source.shape != (board_size, board_size, 7):
            raise ValueError("Unexpected board dimensions")
        tiles = source.argmax(axis=-1)
        if int(np.isin(tiles, (3, 4)).sum()) != boxes:
            raise ValueError("Unexpected box count")
        layout = canonical_layout_id(source)
        if layout != rows[0]["layout_id"] or layout in seen_layouts:
            raise ValueError("Layout identity differs or is duplicated")
        seen_layouts.add(layout)
        trajectory = verify_solution(source, source_row["actions"])
        env = environment(source)
        for row in sorted(rows, key=lambda item: item["k"]):
            pair_id = row["pair_id"]
            if pair_id in seen_ids:
                raise ValueError("Duplicate pair ID")
            seen_ids.add(pair_id)
            i, k = row["i"], row["k"]
            if not isinstance(i, int) or i < 0 or i + k >= len(trajectory):
                raise ValueError("Pair lies outside its replayed trajectory")
            before, after = trajectory[i], trajectory[i + k]
            before_hash, after_hash = state_hash(before), state_hash(after)
            if (before_hash, after_hash) != (row["source_hash"], row["target_hash"]):
                raise ValueError("Replayed endpoint hashes differ from the frozen pair")
            if (state_hash(from_rows(raw_states[before_hash])) != before_hash
                    or state_hash(from_rows(raw_states[after_hash])) != after_hash):
                raise ValueError("Stored Figure 1 state differs from its hash")
            validate_subgoal(before, after)
            if np.array_equal(before, after) or env.is_solved(before):
                raise ValueError("Source is terminal or identical to target")
            pairs.append(Pair(pair_id, layout, row["trajectory_id"], rank, i, k,
                              before_hash, after_hash, before.copy(), after.copy()))
    return pairs


def rollout_pairs(
    pairs: Sequence[Pair],
    predict: Callable[[list[np.ndarray], list[np.ndarray]], Sequence[int]],
    model: str,
    *,
    cap: int = ACTION_CAP,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
    make_environment: Callable[[np.ndarray], Any] = environment,
) -> list[dict[str, Any]]:
    """Run greedy actions; each trial ends on first exact hit, terminal or cap."""
    if cap < 1:
        raise ValueError("Action cap must be positive")
    envs = [make_environment(pair.source) for pair in pairs]
    current = [pair.source.copy() for pair in pairs]
    histories: list[list[int]] = [[] for _ in pairs]
    hashes = [[state_hash(pair.source)] for pair in pairs]
    no_ops = [0] * len(pairs)
    active = list(range(len(pairs)))
    completed: list[dict[str, Any]] = []
    while active:
        actions = list(predict([current[j] for j in active], [pairs[j].target for j in active]))
        if len(actions) != len(active) or any(not isinstance(a, (int, np.integer)) or a not in range(4) for a in actions):
            raise ValueError("Controller must return one action in 0..3 per active pair")
        following = []
        for j, action in zip(active, actions):
            pair, env = pairs[j], envs[j]
            before = current[j]
            after = np.asarray(env.next_state(before, int(action)))
            current[j] = after
            histories[j].append(int(action))
            hashes[j].append(state_hash(after))
            no_ops[j] += int(np.array_equal(before, after))
            hit = bool(np.array_equal(after, pair.target))
            stop = ("exact_target" if hit else "task_terminal" if env.done
                    else "action_limit" if len(histories[j]) == cap else None)
            if stop is None:
                following.append(j)
                continue
            row = {"model": model, "pair_id": pair.pair_id, "layout_id": pair.layout_id,
                   "trajectory_id": pair.trajectory_id, "rank": pair.rank, "i": pair.position, "k": pair.k,
                   "source_hash": pair.source_hash, "target_hash": pair.target_hash, "action_cap": cap,
                   "exact_hit": hit, "first_hit": len(histories[j]) if hit else None,
                   "attempted_actions": len(histories[j]), "actions": histories[j], "state_hashes": hashes[j],
                   "no_op_count": no_ops[j], "stop_reason": stop}
            completed.append(row)
            if on_complete is not None:
                on_complete(row)
        active = following
    return completed


def verify_saved_trial(pair: Pair, row: dict[str, Any], model: str, cap: int = ACTION_CAP) -> None:
    """Replay a saved complete trial before it can be reused during resume."""
    expected = {"model": model, "pair_id": pair.pair_id, "layout_id": pair.layout_id,
                "trajectory_id": pair.trajectory_id, "rank": pair.rank, "i": pair.position, "k": pair.k,
                "source_hash": pair.source_hash, "target_hash": pair.target_hash, "action_cap": cap}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved trial identity differs from the frozen plan")
    actions = row.get("actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= cap:
        raise ValueError("Saved trial has an invalid number of actions")
    env = environment(pair.source)
    state = pair.source.copy()
    hashes = [state_hash(state)]
    no_ops = 0
    stop: str | None = None
    for index, action in enumerate(actions, 1):
        if stop is not None or not isinstance(action, int) or action not in range(4):
            raise ValueError("Saved actions are invalid or continue after a stop")
        before = state
        state = np.asarray(env.next_state(before, action))
        hashes.append(state_hash(state))
        no_ops += int(np.array_equal(before, state))
        stop = ("exact_target" if np.array_equal(state, pair.target) else
                "task_terminal" if env.done else "action_limit" if index == cap else None)
    hit = stop == "exact_target"
    if (stop is None or row.get("stop_reason") != stop or row.get("exact_hit") is not hit
            or row.get("first_hit") != (len(actions) if hit else None) or row.get("state_hashes") != hashes
            or row.get("no_op_count") != no_ops or row.get("attempted_actions") != len(actions)):
        raise ValueError("Saved trial failed independent replay")


def model_predictor(model: Any, tokenizer: Any, device: torch.device, batch_size: int) -> Callable:
    """Create a single-device, FP32 batched greedy controller."""
    def predict(states: list[np.ndarray], targets: list[np.ndarray]) -> list[int]:
        result: list[int] = []
        for first in range(0, len(states), batch_size):
            ids = pair_tokens(tokenizer, states[first:first + batch_size], targets[first:first + batch_size])
            if ids.shape[1] != 291 or ids.shape[1] > model.config.max_position_embeddings:
                raise ValueError("Unexpected or unsupported board-pair token length")
            with torch.inference_mode():
                logits = model(ids.to(device)).logits
            if logits.shape != (len(ids), 4) or not torch.isfinite(logits).all():
                raise ValueError("Controller returned invalid action logits")
            result.extend(int(a) for a in logits.argmax(dim=-1).cpu().tolist())
        return result
    return predict


def run(args: argparse.Namespace, output: Path) -> None:
    """Execute or safely resume one fully specified local model run."""
    started = time.monotonic()
    status_path = output / "status.json"
    rows: list[dict[str, Any]] = []
    contract_validated = not args.resume
    if not args.resume:
        atomic_json(status_path, {"state": "running", "completed": 0, "requested": args.limit})
    try:
        pair_path, state_path = DATA / "core_pairs.json", DATA / "states.json"
        source_path = DATA / "private/source_bundle.tar.gz"
        checkpoint = CHECKPOINTS[args.model]
        pairs = prepare_pairs(json.loads(pair_path.read_text())["pairs"], json.loads(state_path.read_text())["states"],
                              load_source_rows(source_path))[:args.limit]
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(args.device)
        metadata = {"model": args.model, "checkpoint": str(checkpoint), "device": str(device), "dtype": "float32",
                    "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
                    "batch_size": args.batch_size, "threads": args.threads,
                    "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                    "deterministic_algorithms": True, "tf32": False, "action_cap": ACTION_CAP,
                    "token_length": 291, "requested": args.limit,
                    "scope": "complete 20-layout grid" if args.limit == 120 else "partial smoke; not a full grid",
                    "policy": "greedy argmax, no action mask, exact target including player",
                    "source": "original Figure 1; first 20 layout ranks, one source per layout",
                    "sha256": {str(p): file_sha256(p) for p in (pair_path, state_path, source_path,
                                checkpoint / "config.json", checkpoint / "pytorch_model.bin", Path(__file__).resolve())},
                    "argv": sys.argv}
        if args.resume:
            previous = json.loads((output / "provenance.json").read_text())
            if {k: v for k, v in previous.items() if k != "argv"} != {k: v for k, v in metadata.items() if k != "argv"}:
                raise ValueError("Resume contract differs; choose a new output directory")
            trials_path = output / "trials.jsonl"
            if trials_path.exists():
                rows = [json.loads(line) for line in trials_path.read_text().splitlines() if line.strip()]
            by_id = {pair.pair_id: pair for pair in pairs}
            if len({row["pair_id"] for row in rows}) != len(rows):
                raise ValueError("Duplicate saved pair")
            for row in rows:
                if row["pair_id"] not in by_id:
                    raise ValueError("Saved pair is absent from the frozen plan")
                verify_saved_trial(by_id[row["pair_id"]], row, args.model)
            contract_validated = True
            if len(rows) == len(pairs):
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                final_state = "complete" if args.limit == 120 else "partial"
                if status.get("state") != final_state:
                    atomic_json(status_path, {"state": final_state, "completed": len(rows),
                                              "requested": len(pairs), "recovered_from_traces": True,
                                              "seconds": None, "gpu_peak_allocated_bytes": None,
                                              "gpu_peak_reserved_bytes": None})
                print(json.dumps({"output_dir": str(output), "already_completed": len(rows)}), flush=True)
                return
        else:
            atomic_json(output / "provenance.json", metadata)
        atomic_json(status_path, {"state": "running", "completed": len(rows), "requested": len(pairs)})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = BertForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, use_safetensors=False).to(device).float().eval()
        model.requires_grad_(False)
        tokenizer = environment(pairs[0].source).tokenizer
        prior = {row["pair_id"]: row for row in rows}

        def save(row: dict[str, Any]) -> None:
            if row["pair_id"] in prior:
                if prior[row["pair_id"]] != row:
                    raise ValueError("Recomputed trial differs from the saved trace; stopped without overwriting it")
                return
            rows.append(row)
            atomic_jsonl(output / "trials.jsonl", rows)
            atomic_json(status_path, {"state": "running", "completed": len(rows), "requested": len(pairs)})

        rollout_pairs(pairs, model_predictor(model, tokenizer, device, args.batch_size), args.model, on_complete=save)
        atomic_json(status_path, {"state": "complete" if args.limit == 120 else "partial",
                                  "completed": len(rows), "requested": len(pairs),
                                  "seconds": time.monotonic() - started,
                                  "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                                  if device.type == "cuda" else None,
                                  "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                                  if device.type == "cuda" else None})
        print(json.dumps({"output_dir": str(output), "completed": len(rows),
                          "hits": sum(row["exact_hit"] for row in rows)}), flush=True)
    except BaseException as error:
        if contract_validated:
            atomic_json(status_path, {"state": "failed", "completed": len(rows), "requested": args.limit,
                                      "seconds": time.monotonic() - started, "error": repr(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=tuple(CHECKPOINTS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=120, help="Use a separate output directory for a partial smoke")
    parser.add_argument("--resume", action="store_true", help="Replay and retain completed trials before continuing")
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1 or not 1 <= args.limit <= 120:
        parser.error("threads and batch size must be positive; limit must be in 1..120")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".run.lock"), timeout=0):
        if not args.resume and any(path.name != ".run.lock" for path in output.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output}")
        run(args, output)


if __name__ == "__main__":
    main()
