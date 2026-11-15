"""Train matched generator LoRA adapters on equivalent intent representations.

The controller is not loaded or trained. Each example uses the current board
and task only; loss is applied solely to the response tokens. Training and
resource checks select TRAIN/VALIDATION records before examining token lengths.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import board_balanced
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import generate
from carl.icl.intent_generator_experiment import initialise
from carl.icl.intent_generator_experiment import memory
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_generator_experiment import sync_gpu
from carl.icl.intent_generator_experiment import tensor_example
from carl.icl.intent_generator_experiment import validation_contexts
from carl.icl.intent_representation_experiment import Representation
from carl.icl.intent_representation_experiment import messages_for
from carl.icl.intent_representation_experiment import render_representation
from carl.icl.intents.provenance import GENERATION_COMPONENTS
from carl.icl.intents.provenance import REPRESENTATION_COMPONENTS
from carl.icl.intents.provenance import component_hashes


class Mode(StrEnum):
    INSPECT = "inspect"
    BENCHMARK = "benchmark"
    TRAIN = "train"


def encode_example(tokenizer: Any, record: dict[str, Any], representation: Representation | str,
                   max_length: int) -> dict[str, list[int]]:
    """Build an untruncated example with loss masked outside the response."""
    prefix = tokenizer.apply_chat_template(
        messages_for(record, [], representation), tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(prefix, list):
        prefix = prefix.tolist()
    response = render_representation(representation, record["intent"])
    answer = tokenizer.encode(response + (tokenizer.eos_token or ""), add_special_tokens=False)
    if not answer:
        raise ValueError("the complete target must contain response tokens")
    ids = prefix + answer
    if len(ids) > max_length:
        raise ValueError(f"complete example requires {len(ids)} tokens; limit is {max_length}; no truncation")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prefix) + answer}


def training_splits(data: str | Path, max_boards: int, per_board: int) -> tuple[list[dict[str, Any]],
                                                                                list[dict[str, Any]]]:
    """Select checked train and validation records without using test labels."""
    training = read_records(data, DataSplit.TRAIN)
    validation = validation_contexts(read_records(data, DataSplit.VALIDATION), max_boards, per_board)
    if not training or not validation:
        raise ValueError("checked training and validation records are required")
    if any(row.get("validation", {}).get("accepted") is not True for row in training + validation):
        raise ValueError("all training and validation annotations must be checked and accepted")
    if {row["layout_id"] for row in training} & {row["layout_id"] for row in validation}:
        raise ValueError("training and validation layouts overlap")
    if len({row["record_id"] for row in training + validation}) != len(training) + len(validation):
        raise ValueError("training and validation IDs must be unique")
    return training, validation


def token_lengths(tokenizer: Any, records: list[dict[str, Any]], representation: str,
                  max_length: int) -> dict[str, Any]:
    sizes = sorted(len(encode_example(tokenizer, row, representation, max_length)["input_ids"]) for row in records)
    if not sizes:
        raise ValueError("cannot measure lengths on an empty split")
    return {"n": len(sizes), "min": sizes[0], "median": sizes[len(sizes) // 2],
            "p95": sizes[min(len(sizes) - 1, int(0.95 * len(sizes)))], "max": sizes[-1],
            "limit": max_length, "truncated": 0}


def _record_digest(records: list[dict[str, Any]]) -> str:
    allowed = ("record_id", "layout_id", "split", "start_observation", "task_goal", "intent")
    payload = [{key: row[key] for key in allowed} for row in records]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _model_revision(model: Path, revision: str) -> str:
    if not model.is_dir():
        raise ValueError("model must be a complete local snapshot")
    metadata = model / "experiment_revision.json"
    if metadata.exists() and json.loads(metadata.read_text(encoding="utf-8")).get("revision") == revision:
        return "experiment_revision.json"
    if len(model.name) == 40 and all(char in "0123456789abcdef" for char in model.name) and model.name == revision:
        return "huggingface snapshot directory"
    raise ValueError("cannot verify the requested local model revision")


def configuration(args: argparse.Namespace, training: list[dict[str, Any]], validation: list[dict[str, Any]],
                  lengths: dict[str, Any], device: str | None) -> dict[str, Any]:
    model = Path(args.model)
    result = {"schema_version": 1, "mode": args.mode, "representation": args.representation,
              "model": str(model.resolve()), "revision": args.revision,
              "revision_verification": _model_revision(model, args.revision),
              "model_weights_sha256": file_sha256(model / "model.safetensors"),
              "model_config_sha256": file_sha256(model / "config.json"),
              "tokenizer_sha256": file_sha256(model / "tokenizer.json"),
              "data_sha256": file_sha256(args.data), "training_records_sha256": _record_digest(training),
              "validation_records_sha256": _record_digest(validation),
              "train_record_ids": [row["record_id"] for row in training],
              "validation_record_ids": [row["record_id"] for row in validation],
              "train_layouts": sorted({row["layout_id"] for row in training}),
              "validation_layouts": sorted({row["layout_id"] for row in validation}),
              "protocol_sha256": file_sha256(Path(__file__).with_name("intent_representation_experiment.py")),
              "protocol_components_sha256": component_hashes(REPRESENTATION_COMPONENTS),
              "training_code_sha256": file_sha256(__file__),
              "initialiser_code_sha256": file_sha256(Path(__file__).with_name("intent_generator_experiment.py")),
              "generator_components_sha256": component_hashes(GENERATION_COMPONENTS),
              "dtype": args.dtype, "seed": args.seed, "threads": args.threads, "device": device,
              "rank": args.rank, "lora_alpha": 2 * args.rank, "lora_targets": ["q_proj", "v_proj"],
              "microbatch": 1, "accumulation": args.accumulation, "learning_rate": args.lr,
              "gradient_checkpointing": args.gradient_checkpointing, "max_steps": args.max_steps,
              "eval_every": args.eval_every, "max_length": args.max_length, "max_new_tokens": args.max_new_tokens,
              "loss": "response tokens and EOS only", "enable_thinking": False, "token_lengths": lengths,
              "sampling": "board-balanced, seeded", "command": sys.argv}
    return result


def inspect(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    _model_revision(Path(args.model), args.revision)
    training, validation = training_splits(args.data, args.validation_boards, args.validation_contexts_per_board)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.chat_template is None:
        raise ValueError("tokenizer has no chat template")
    lengths = {"train": token_lengths(tokenizer, training, args.representation, args.max_length),
               "validation": token_lengths(tokenizer, validation, args.representation, args.max_length)}
    report = configuration(args, training, validation, lengths, None)
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("use a new inspection output")
    atomic_json(destination, report)
    print(json.dumps({"status": "prepared", "representation": args.representation,
                      "token_lengths": lengths, "output": str(destination)}, indent=2), flush=True)


def _validate_cuda(args: argparse.Namespace) -> None:
    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required but unavailable; no silent CPU fallback")


def _sample_loss(torch: Any, model: Any, tokenizer: Any, record: dict[str, Any], args: argparse.Namespace,
                 device: str):
    example = encode_example(tokenizer, record, args.representation, args.max_length)
    inputs = tensor_example(torch, example, device)
    return model(**inputs).loss, len(example["input_ids"])


def benchmark(args: argparse.Namespace) -> None:
    """Measure backward, optimiser state allocation and representative generation."""
    training, validation = training_splits(args.data, args.validation_boards, args.validation_contexts_per_board)
    if len(training) < args.accumulation:
        raise ValueError("insufficient training examples for the benchmark")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a new benchmark output")
    _model_revision(Path(args.model), args.revision)
    _validate_cuda(args)
    started = time.monotonic()
    torch, tokenizer, model, device = initialise(args, train=True)
    lengths = {"train": token_lengths(tokenizer, training, args.representation, args.max_length),
               "validation": token_lengths(tokenizer, validation, args.representation, args.max_length)}
    metadata = configuration(args, training, validation, lengths, device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    model.train()
    sync_gpu(torch)
    step_start = time.monotonic()
    losses = []
    tokens_processed = 0
    for row in training[:args.accumulation]:
        loss, tokens = _sample_loss(torch, model, tokenizer, row, args, device)
        loss = loss / args.accumulation
        if not math.isfinite(float(loss.detach().cpu())):
            raise RuntimeError("non-finite benchmark loss")
        losses.append(float(loss.detach().cpu()))
        tokens_processed += tokens
        loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sync_gpu(torch)
    seconds = time.monotonic() - step_start
    answer, generated_tokens, generation_seconds = generate(
        torch, model, tokenizer, training[0], args, device, messages=messages_for(training[0], [], args.representation),
    )
    summary = {"status": "complete", **metadata, "complete_step_seconds": seconds, "step_losses": losses,
               "training_tokens": tokens_processed, "training_tokens_per_second": tokens_processed / seconds,
               "generation_sample": answer, "generation_tokens": generated_tokens,
               "generation_seconds": generation_seconds,
               "generation_tokens_per_second": generated_tokens / generation_seconds if generation_seconds else None,
               "elapsed_seconds": time.monotonic() - started,
               "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), **memory(torch)}
    atomic_json(output, summary)
    print(json.dumps({key: summary[key] for key in ("status", "representation", "complete_step_seconds",
                                                     "generation_seconds", "gpu_peak_reserved_bytes")}, indent=2),
          flush=True)


def validation_loss(torch: Any, model: Any, tokenizer: Any, records: list[dict[str, Any]], args: argparse.Namespace,
                    device: str) -> float:
    """Weight independent validation layouts equally when selecting a checkpoint."""
    groups: dict[str, list[float]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for row in records:
            loss, _ = _sample_loss(torch, model, tokenizer, row, args, device)
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError("non-finite validation loss")
            groups[row["layout_id"]].append(value)
    model.train()
    if not groups:
        raise ValueError("no validation layouts")
    return sum(sum(values) / len(values) for values in groups.values()) / len(groups)


def save_adapter(model: Any, output: Path, name: str) -> Path:
    """Make a completed adapter visible only after all of its files exist."""
    destination = output / name
    temporary = output / f".{name}.{os.getpid()}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(f"adapter output already exists: {destination}")
    model.save_pretrained(str(temporary), safe_serialization=True)
    os.replace(temporary, destination)
    return destination


def expired(args: argparse.Namespace) -> bool:
    return ((args.stop_monotonic is not None and time.monotonic() >= args.stop_monotonic)
            or (args.stop_epoch is not None and time.time() >= args.stop_epoch))


def completion_status(completed: int, max_steps: int, last_validation_step: int | None) -> tuple[str, str | None]:
    if completed < max_steps:
        return "stopped", "signal_or_deadline"
    if last_validation_step != completed:
        return "stopped", "final_validation_missing"
    return "finished", None


def train(args: argparse.Namespace) -> None:
    training, validation = training_splits(args.data, args.validation_boards, args.validation_contexts_per_board)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a new training output directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", {"status": "initializing", "completed_steps": 0})
    try:
        _model_revision(Path(args.model), args.revision)
        _validate_cuda(args)
        if expired(args):
            atomic_json(output / "status.json", {"status": "stopped", "reason": "deadline", "completed_steps": 0})
            return
        torch, tokenizer, model, device = initialise(args, train=True)
        lengths = {"train": token_lengths(tokenizer, training, args.representation, args.max_length),
                   "validation": token_lengths(tokenizer, validation, args.representation, args.max_length)}
        atomic_json(output / "configuration.json", configuration(args, training, validation, lengths, device))
        tokenizer.save_pretrained(output / "tokenizer")
    except Exception as exc:
        atomic_json(output / "status.json", {"status": "failed", "completed_steps": 0,
                                             "error_type": type(exc).__name__, "error": str(exc)})
        raise
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.lr)
    examples = board_balanced(training, args.seed)
    stopped = {"requested": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stopped["requested"] = True

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    completed, best, best_path = 0, float("inf"), None
    current_checkpoint: str | None = None
    last_validation_step: int | None = None
    started = time.monotonic()
    model.train()
    try:
        with (output / "training_curve.jsonl").open("x", encoding="utf-8", buffering=1) as curve:
            for step in range(1, args.max_steps + 1):
                if stopped["requested"] or expired(args):
                    break
                step_start = time.monotonic()
                total_loss = 0.0
                for _ in range(args.accumulation):
                    loss, _ = _sample_loss(torch, model, tokenizer, next(examples), args, device)
                    loss = loss / args.accumulation
                    total_loss += float(loss.detach().cpu())
                    if not math.isfinite(total_loss):
                        raise RuntimeError(f"non-finite training loss at step {step}")
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed = step
                row: dict[str, Any] = {"step": step, "train_loss": total_loss,
                                       "seconds": time.monotonic() - step_start}
                if (step % args.eval_every == 0 or step == args.max_steps) and not stopped["requested"] and not expired(args):
                    val = validation_loss(torch, model, tokenizer, validation, args, device)
                    row["validation_loss"] = val
                    current_checkpoint = str(save_adapter(model, output, f"checkpoint-{step}"))
                    last_validation_step = step
                    if val < best:
                        best, best_path = val, current_checkpoint
                        atomic_json(output / "best.json", {"step": step, "validation_loss": val,
                                                           "adapter": best_path})
                curve.write(json.dumps(row) + "\n")
                curve.flush()
                os.fsync(curve.fileno())
                atomic_json(output / "status.json", {"status": "running", "completed_steps": completed,
                                                     "best_adapter": best_path, "latest_checkpoint": current_checkpoint,
                                                     "last_measurement": row, **memory(torch)})
                print(json.dumps(row), flush=True)
        final_path = str(save_adapter(model, output, f"checkpoint-last-{completed}")) if completed else None
        status, reason = completion_status(completed, args.max_steps, last_validation_step)
        atomic_json(output / "status.json", {
            "status": status, "reason": reason, "completed_steps": completed,
            "best_adapter": best_path, "last_adapter": final_path, "latest_checkpoint": current_checkpoint,
            "last_validation_step": last_validation_step,
            "elapsed_seconds": time.monotonic() - started, **memory(torch),
        })
    except Exception as exc:
        atomic_json(output / "status.json", {"status": "failed", "completed_steps": completed,
                                             "best_adapter": best_path, "latest_checkpoint": current_checkpoint,
                                             "error_type": type(exc).__name__, "error": str(exc), **memory(torch)})
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=list(Mode))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--representation", choices=list(Representation), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--validation-boards", type=int, default=12)
    parser.add_argument("--validation-contexts-per-board", type=int, default=4)
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--stop-epoch", type=float)
    parser.add_argument("--require-cuda", action="store_true")
    parser.set_defaults(rank=4)
    args = parser.parse_args()
    if min(args.threads, args.accumulation, args.max_length, args.max_new_tokens, args.max_steps,
           args.eval_every, args.validation_contexts_per_board) < 1 or args.validation_boards < 0:
        parser.error("steps, lengths, thread counts and accumulation must be positive")
    if args.lr <= 0 or not math.isfinite(args.lr):
        parser.error("learning rate must be finite and positive")
    {Mode.INSPECT: inspect, Mode.BENCHMARK: benchmark, Mode.TRAIN: train}[args.mode](args)


if __name__ == "__main__":
    main()
