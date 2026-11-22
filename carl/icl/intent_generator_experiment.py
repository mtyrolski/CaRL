"""Train and evaluate a local intent generator on checked observations."""

import argparse
import hashlib
import json
import math
import signal
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from carl.icl.intents.generation.backend import generate
from carl.icl.intents.generation.backend import initialise
from carl.icl.intents.generation.backend import local_snapshot
from carl.icl.intents.generation.backend import memory
from carl.icl.intents.generation.backend import sync_gpu
from carl.icl.intents.generation.backend import tensor_example
from carl.icl.intents.generation.backend import validation_loss
from carl.icl.intents.generation.cli import run_cli
from carl.icl.intents.generation.checkpoint import (
    ADAPTER_MANIFEST, CHECKPOINT_CRITERION, base_snapshot_identity, predicate_prompt_contract, publish_adapter,
)
from carl.icl.intents.generation.data import DEFAULT_MODEL
from carl.icl.intents.generation.data import DEFAULT_REVISION
from carl.icl.intents.generation.data import SYSTEM
from carl.icl.intents.generation.data import DataSplit
from carl.icl.intents.generation.data import GeneratorMethod
from carl.icl.intents.generation.data import RunMode
from carl.icl.intents.generation.data import TrainingAugmentation
from carl.icl.intents.generation.data import atomic_json
from carl.icl.intents.generation.data import augmented_training_samples
from carl.icl.intents.generation.data import board_balanced
from carl.icl.intents.generation.data import encode_example
from carl.icl.intents.generation.data import evaluation_records
from carl.icl.intents.generation.data import file_sha256
from carl.icl.intents.generation.data import one_per_board
from carl.icl.intents.generation.data import parse_json
from carl.icl.intents.generation.data import predicate_record
from carl.icl.intents.generation.data import prompt_messages
from carl.icl.intents.generation.data import read_records
from carl.icl.intents.generation.data import record_task_group
from carl.icl.intents.generation.data import target_text
from carl.icl.intents.generation.data import validation_contexts
from carl.icl.intents.generation.evaluation import evaluate
from carl.icl.intents.generation.evaluation import evaluation_configuration
from carl.icl.intents.generation.evaluation import prepare_evaluation_output

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "SYSTEM",
    "DataSplit",
    "GeneratorMethod",
    "RunMode",
    "TrainingAugmentation",
    "atomic_json",
    "augmented_training_samples",
    "benchmark",
    "board_balanced",
    "encode_example",
    "evaluate",
    "evaluation_configuration",
    "evaluation_records",
    "file_sha256",
    "generate",
    "initialise",
    "lengths",
    "main",
    "memory",
    "one_per_board",
    "parse_json",
    "prepare_evaluation_output",
    "prompt_messages",
    "read_records",
    "sync_gpu",
    "target_text",
    "tensor_example",
    "train",
    "training_lengths",
    "validation_contexts",
    "validation_loss",
]


def lengths(
    tokenizer: Any, records: list[dict[str, Any]], max_length: int
) -> dict[str, Any]:
    values = [
        len(encode_example(tokenizer, record, max_length)["input_ids"])
        for record in records
    ]
    if not values:
        raise ValueError("dataset has no examples")
    values.sort()
    return {
        "n": len(values),
        "min": values[0],
        "median": values[len(values) // 2],
        "p95": values[min(len(values) - 1, int(0.95 * len(values)))],
        "max": values[-1],
        "limit": max_length,
        "truncated": 0,
    }


def training_lengths(
    tokenizer: Any,
    training: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Verify complete examples in every train orientation and original validation."""
    from carl.icl.intent_augmentation import SYMMETRIES
    from carl.icl.intent_augmentation import transform_training_record

    mode = TrainingAugmentation(
        getattr(args, "augmentation", TrainingAugmentation.NONE)
    )
    if mode == TrainingAugmentation.D4:
        records = [
            transform_training_record(row, symmetry)
            for row in training
            for symmetry in SYMMETRIES
        ]
        records.extend(validation)
    else:
        records = training + validation
    summary = lengths(tokenizer, records, args.max_length)
    summary["training_orientations_per_record"] = (
        len(SYMMETRIES) if mode == TrainingAugmentation.D4 else 1
    )
    padding = getattr(args, "pad_to_length", 0)
    if padding and summary["max"] > padding:
        raise ValueError(
            f"complete example needs {summary['max']} tokens; padding limit is {padding}"
        )
    summary["pad_to_length"] = padding or None
    return summary


def benchmark(args: argparse.Namespace) -> None:
    records = read_records(args.data)
    training = [row for row in records if row["split"] == DataSplit.TRAIN]
    validation = [row for row in records if row["split"] == DataSplit.VALIDATION]
    if len(training) < args.accumulation:
        raise ValueError("not enough train examples for benchmark accumulation")
    started = time.monotonic()
    torch, tokenizer, model, device = initialise(args, train=True)
    # Held-out targets must not affect the training resource check.
    token_lengths = training_lengths(tokenizer, training, validation, args)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    model.train()
    sync_gpu(torch)
    start_step = time.monotonic()
    losses = []
    tokens_processed = 0
    effective_tokens = 0
    samples = augmented_training_samples(
        training, args.seed, getattr(args, "augmentation", TrainingAugmentation.NONE)
    )
    for _ in range(args.accumulation):
        record, _symmetry = next(samples)
        encoded = encode_example(tokenizer, record, args.max_length)
        effective_tokens += len(encoded["input_ids"])
        x = tensor_example(
            torch,
            encoded,
            device,
            pad_to_length=getattr(args, "pad_to_length", 0),
            pad_token_id=tokenizer.pad_token_id,
        )
        tokens_processed += int(x["input_ids"].numel())
        loss = model(**x).loss / args.accumulation
        losses.append(float(loss.detach().cpu()) * args.accumulation)
        if not math.isfinite(losses[-1]):
            raise RuntimeError("benchmark loss is not finite")
        loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sync_gpu(torch)
    step_seconds = time.monotonic() - start_step
    text, tokens, generation_seconds = generate(
        torch, model, tokenizer, training[0], args, device
    )
    summary = {
        "status": "complete",
        "device": device,
        "dtype": args.dtype,
        "model": args.model,
        "revision": args.revision,
        "enable_thinking": False,
        "data_sha256": file_sha256(args.data),
        "token_lengths": token_lengths,
        "accumulation": args.accumulation,
        "rank": args.rank,
        "gradient_checkpointing": args.gradient_checkpointing,
        "complete_step_seconds": step_seconds,
        "step_losses": losses,
        "training_tokens": tokens_processed,
        "effective_training_tokens": effective_tokens,
        "training_tokens_per_second": tokens_processed / step_seconds,
        "generation_seconds": generation_seconds,
        "generation_tokens": tokens,
        "generation_tokens_per_second": tokens / generation_seconds
        if generation_seconds
        else None,
        "generation_sample": text,
        "elapsed_seconds": time.monotonic() - started,
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        **memory(torch),
    }
    atomic_json(Path(args.output), summary)
    print(json.dumps(summary, indent=2), flush=True)


def _demo_splits(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_records(args.data)
    training = [row for row in rows if row["split"] == DataSplit.TRAIN]
    validation_records = [row for row in rows if row["split"] == DataSplit.VALIDATION]
    validation = validation_contexts(
        validation_records,
        args.validation_boards,
        args.validation_contexts_per_board,
    )
    if not training or not validation:
        raise ValueError("train and validation splits must both contain boards")
    if any(row.get("policy_measurements") is not None for row in rows):
        raise ValueError("controller measurements require carl.icl.policy_relative_lora")
    if {row["layout_id"] for row in training} & {
        row["layout_id"] for row in validation_records
    }:
        raise ValueError("training and validation layouts overlap")
    return training, validation, validation_records


def _demo_identity(args: argparse.Namespace, training: list[dict[str, Any]],
                   validation: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data_sha256": file_sha256(args.data),
        "train_layout_ids": sorted({row["layout_id"] for row in training}),
        "validation_layout_ids": sorted({row["layout_id"] for row in validation}),
        "train_task_groups": sorted({group for row in training if (group := record_task_group(row)) is not None}),
        "validation_task_groups": sorted({group for row in validation if (group := record_task_group(row)) is not None}),
        "train_record_ids": [row["record_id"] for row in training],
        "validation_record_ids": [row["record_id"] for row in validation],
    }


def inspect_data(args: argparse.Namespace) -> None:
    """Check demonstration splits without importing a model or the LoRA extra."""
    training, selected, validation = _demo_splits(args)
    summary = {**_demo_identity(args, training, validation), "train_records": len(training),
               "validation_records": len(validation), "selected_validation_records": len(selected),
               "scale": None}
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a fresh inspection output")
    atomic_json(output, summary)
    print(json.dumps(summary, indent=2))


def train(args: argparse.Namespace) -> None:
    training, validation, validation_records = _demo_splits(args)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh training directory")
    output.mkdir(parents=True, exist_ok=True)
    torch, tokenizer, model, device = initialise(args, train=True)
    # Inspect only examples actually used for fitting or validation.
    token_lengths = training_lengths(tokenizer, training, validation, args)
    snapshot = local_snapshot(args.model, args.revision)
    base = base_snapshot_identity(snapshot, args.revision)
    prompt = predicate_prompt_contract("unscaled") if predicate_record(training[0]) else {
        "version": "sokoban-intent-generator-legacy-v1", "scale_arm": "unscaled",
        "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
    }
    provenance = {
        **_demo_identity(args, training, validation_records),
        "validation_record_ids": [row["record_id"] for row in validation],
        "role": "generator", "prompt": prompt,
        "seed": args.seed, "rank": args.rank, "lora_targets": getattr(args, "lora_targets", None) or ["q_proj", "v_proj"],
        "learning_rate": args.lr, "max_steps": args.max_steps, "accumulation": args.accumulation,
        "max_length": args.max_length, "pad_to_length": getattr(args, "pad_to_length", 0) or None,
        "validation_interval": args.eval_every, "checkpoint_selection": CHECKPOINT_CRITERION,
    }
    atomic_json(
        output / "configuration.json",
        {
            "args": vars(args),
            "device": device,
            **provenance,
            "token_lengths": token_lengths,
            "enable_thinking": False,
            "validation_layouts": sorted({r["layout_id"] for r in validation}),
            "command": sys.argv,
        },
    )
    tokenizer.save_pretrained(output / "tokenizer")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    samples = augmented_training_samples(
        training, args.seed, getattr(args, "augmentation", TrainingAugmentation.NONE)
    )
    stop = {"requested": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop["requested"] = True

    best = float("inf")
    best_path = None
    completed = 0
    model.train()
    with ExitStack() as cleanup, (output / "training_curve.jsonl").open(
        "w", encoding="utf-8", buffering=1
    ) as curve:
        for sig in (signal.SIGTERM, signal.SIGINT):
            cleanup.callback(signal.signal, sig, signal.signal(sig, request_stop))
        for step in range(1, args.max_steps + 1):
            if stop["requested"] or (
                args.stop_monotonic is not None
                and time.monotonic() >= args.stop_monotonic
            ):
                break
            started = time.monotonic()
            total_loss = 0.0
            model_tokens = effective_tokens = supervised_tokens = 0
            sampled = []
            for _ in range(args.accumulation):
                record, symmetry = next(samples)
                encoded = encode_example(tokenizer, record, args.max_length)
                effective_tokens += len(encoded["input_ids"])
                supervised_tokens += sum(label != -100 for label in encoded["labels"])
                x = tensor_example(
                    torch,
                    encoded,
                    device,
                    pad_to_length=getattr(args, "pad_to_length", 0),
                    pad_token_id=tokenizer.pad_token_id,
                )
                model_tokens += int(x["input_ids"].numel())
                sampled.append(
                    {
                        "record_id": record["record_id"],
                        "layout_id": record["layout_id"],
                        "symmetry": symmetry,
                    }
                )
                loss = model(**x).loss / args.accumulation
                total_loss += float(loss.detach().cpu())
                if not math.isfinite(total_loss):
                    raise RuntimeError(f"non-finite training loss at step {step}")
                loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed = step
            row = {
                "step": step,
                "train_loss": total_loss,
                "seconds": time.monotonic() - started,
                "model_tokens": model_tokens,
                "effective_tokens": effective_tokens,
                "supervised_tokens": supervised_tokens,
                "samples": sampled,
            }
            if step % args.eval_every == 0 or step == args.max_steps:
                val = validation_loss(torch, model, tokenizer, validation, args, device)
                row["validation_loss"] = val
                if val < best:
                    best = val
                    destination = publish_adapter(
                        model, tokenizer, output, f"checkpoint-{step}", base_identity=base, role="generator",
                        training=provenance, prompt=prompt, step=step, validation_loss=val,
                    )
                    best_path = str(destination.resolve())
                    atomic_json(
                        output / "best.json",
                        {"step": step, "validation_loss": val, "adapter": best_path,
                         "adapter_fingerprint": parse_json((destination / ADAPTER_MANIFEST).read_text())["fingerprint"]},
                    )
            curve.write(json.dumps(row) + "\n")
            atomic_json(
                output / "status.json",
                {
                    "status": "running",
                    "completed_steps": completed,
                    "best_adapter": best_path,
                    "last_measurement": row,
                    **memory(torch),
                },
            )
            print(json.dumps(row), flush=True)
    last_path = publish_adapter(model, tokenizer, output, f"checkpoint-last-{completed}", base_identity=base,
                                role="generator", training=provenance, prompt=prompt, step=completed,
                                validation_loss=None)
    atomic_json(
        output / "status.json",
        {
            "status": "finished" if completed == args.max_steps else "stopped",
            "completed_steps": completed,
            "best_adapter": best_path,
            "last_adapter": str(last_path),
            **memory(torch),
        },
    )


def main() -> None:
    run_cli(benchmark, train, evaluate, inspect=inspect_data, description=__doc__)


if __name__ == "__main__":
    main()
