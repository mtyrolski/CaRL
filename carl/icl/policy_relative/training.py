"""Train and validate LoRA adapters with matched sampling and stopping rules."""

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import generate
from carl.icl.intent_generator_experiment import initialise
from carl.icl.intent_generator_experiment import memory
from carl.icl.intent_generator_experiment import parse_json
from carl.icl.intent_generator_experiment import sync_gpu
from carl.icl.intent_generator_experiment import tensor_example
from carl.icl.intent_representation_training import completion_status
from carl.icl.intent_representation_training import expired
from carl.icl.intent_representation_training import save_adapter
from carl.icl.intents.generation.checkpoint import ADAPTER_MANIFEST, publish_adapter
from carl.icl.policy_relative.configuration import _configuration
from carl.icl.policy_relative.configuration import _model_revision
from carl.icl.policy_relative.configuration import require_scale_information
from carl.icl.policy_relative.data import load_study_data
from carl.icl.policy_relative.prompts import candidate_scale
from carl.icl.policy_relative.prompts import encode_example
from carl.icl.policy_relative.prompts import generator_messages
from carl.icl.policy_relative.prompts import loss_samples
from carl.icl.policy_relative.prompts import training_prompts
from carl.icl.policy_relative.prompts import weighted_study_samples
from carl.icl.policy_relative.types import Arm
from carl.icl.policy_relative.types import Candidate
from carl.icl.policy_relative.types import Context


def _tensors(torch: Any, tokenizer: Any, encoded: dict[str, list[int]], args: argparse.Namespace, device: str) -> Any:
    padding = getattr(args, "pad_to_length", 0)
    return tensor_example(torch, encoded, device, pad_to_length=padding,
                          pad_token_id=tokenizer.pad_token_id if padding else None)


def _loss(
    torch: Any,
    model: Any,
    tokenizer: Any,
    candidate: Candidate,
    args: argparse.Namespace,
    device: str,
) -> Any:
    encoded = encode_example(tokenizer, candidate, Arm(args.arm), args.max_length)
    return model(**_tensors(torch, tokenizer, encoded, args, device)).loss


def _weighted_loss(loss: Any, weight: float, accumulation: int) -> Any:
    """Scale only the target-token loss after the model applies its masks."""
    return (loss if weight == 1 else loss * weight) / accumulation


def validation_loss(
    torch: Any,
    model: Any,
    tokenizer: Any,
    contexts: tuple[Context, ...],
    args: argparse.Namespace,
    device: str,
    demonstration: tuple[Context, ...] | None = None,
    *,
    explicit_weights: bool = False,
) -> float:
    """Use equal fixed validation effort across study arms, preserving legacy scoring."""
    model.eval()

    def mean_loss(pool: tuple[Context, ...]) -> float:
        losses = []
        for context in pool:
            total = 0.0
            for candidate in context.candidates:
                value = float(
                    _loss(torch, model, tokenizer, candidate, args, device)
                    .detach()
                    .cpu()
                )
                if not math.isfinite(value):
                    raise ValueError("validation loss is not finite")
                total += candidate.weight * value
            losses.append(total)
        return sum(losses) / len(losses)

    try:
        with torch.no_grad():
            if demonstration is None:
                value = mean_loss(contexts)
            else:
                base = loss_samples(demonstration, args.seed + 2, explicit_weights=False)
                extra_pool = (
                    demonstration if Arm(args.arm) == Arm.DEMONSTRATION else contexts
                )
                extra = loss_samples(extra_pool, args.seed + 3, explicit_weights=explicit_weights)
                losses = [
                    float(
                        _loss(torch, model, tokenizer, candidate, args, device)
                        .detach()
                        .cpu()
                    ) * weight
                    for _ in demonstration
                    for candidate, weight in (next(base), next(extra))
                ]
                if not all(math.isfinite(loss) for loss in losses):
                    raise ValueError("validation loss is not finite")
                value = sum(losses) / len(losses)
    finally:
        model.train()
    return value


def benchmark(args: argparse.Namespace) -> None:
    """Measure a full optimiser step and generation with the actual prompts."""
    data, demonstration = load_study_data(args)
    require_scale_information(
        data,
        args.arm,
        args.min_informative_train_contexts,
        args.min_informative_validation_contexts,
    )
    _model_revision(Path(args.model), args.revision)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a fresh benchmark output")
    if args.require_cuda:
        import torch as torch_module

        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA required but unavailable; no silent CPU fallback")
    started = time.monotonic()
    torch, tokenizer, model, device = initialise(args, train=True)
    configuration = _configuration(args, data, tokenizer, device, demonstration)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    examples = weighted_study_samples(data, demonstration, Arm(args.arm), args.seed)
    tokens = 0
    model.train()
    sync_gpu(torch)
    step_started = time.monotonic()
    for _ in range(args.accumulation):
        candidate, weight = next(examples)
        example = encode_example(tokenizer, candidate, Arm(args.arm), args.max_length)
        tensors = _tensors(torch, tokenizer, example, args, device)
        tokens += int(tensors["input_ids"].numel())
        loss = _weighted_loss(model(**tensors).loss, weight, args.accumulation)
        if not math.isfinite(float(loss.detach().cpu())):
            raise ValueError("benchmark loss is not finite")
        loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sync_gpu(torch)
    seconds = time.monotonic() - step_started
    candidate = data.train[0].candidates[0]
    requested = candidate_scale(candidate, Arm(args.arm))
    raw, generated, generation_seconds = generate(
        torch,
        model,
        tokenizer,
        candidate.record,
        args,
        device,
        messages=generator_messages(candidate.record, Arm(args.arm), requested),
    )
    summary = {
        **configuration,
        "status": "complete",
        "complete_step_seconds": seconds,
        "training_tokens": tokens,
        "training_tokens_per_second": tokens / seconds,
        "generation_tokens": generated,
        "generation_seconds": generation_seconds,
        "generation_tokens_per_second": generated / generation_seconds
        if generation_seconds
        else None,
        "generation_sample": raw,
        "elapsed_seconds": time.monotonic() - started,
        **memory(torch),
    }
    atomic_json(output, summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "complete_step_seconds",
                    "generation_seconds",
                    "gpu_peak_allocated_bytes",
                    "gpu_peak_reserved_bytes",
                )
            },
            indent=2,
        )
    )


def train(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh training directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "status.json", {"status": "initializing", "completed_steps": 0}
    )
    completed = 0
    best_path = None
    current_checkpoint = None
    torch = None
    try:
        data, demonstration = load_study_data(args)
        require_scale_information(
            data,
            args.arm,
            args.min_informative_train_contexts,
            args.min_informative_validation_contexts,
        )
        _model_revision(Path(args.model), args.revision)
        if expired(args):
            atomic_json(
                output / "status.json",
                {"status": "stopped", "reason": "deadline", "completed_steps": 0},
            )
            return
        if args.require_cuda:
            import torch as torch_module

            if not torch_module.cuda.is_available():
                raise RuntimeError(
                    "CUDA required but unavailable; no silent CPU fallback"
                )
        torch, tokenizer, model, device = initialise(args, train=True)
        configuration = _configuration(args, data, tokenizer, device, demonstration)
        atomic_json(output / "configuration.json", configuration)
        tokenizer.save_pretrained(output / "tokenizer")
        with (output / "prompts.jsonl").open("x", encoding="utf-8") as stream:
            for row in training_prompts(data, args.arm, demonstration):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.lr,
        )
        examples = weighted_study_samples(data, demonstration, Arm(args.arm), args.seed)
        explicit_weights = "measurement_provenance" in data.audit

        def checkpoint(name: str, step: int, loss: float | None) -> Path:
            if "base_snapshot" not in configuration:
                return save_adapter(model, output, name)
            return publish_adapter(model, tokenizer, output, name, base_identity=configuration["base_snapshot"],
                                   role="generator", training=configuration, prompt=configuration["prompt"],
                                   step=step, validation_loss=loss)
        stopped = {"requested": False}

        def request_stop(_signum: int, _frame: Any) -> None:
            stopped["requested"] = True

        previous = {
            sig: signal.signal(sig, request_stop)
            for sig in (signal.SIGTERM, signal.SIGINT)
        }
        best, last_validation = float("inf"), None
        started = time.monotonic()
        try:
            model.train()
            with (output / "training_curve.jsonl").open(
                "x", encoding="utf-8", buffering=1
            ) as curve:
                for step in range(1, args.max_steps + 1):
                    if stopped["requested"] or expired(args):
                        break
                    step_start = time.monotonic()
                    total_loss = 0.0
                    samples = []
                    for _ in range(args.accumulation):
                        candidate, weight = next(examples)
                        loss = _weighted_loss(_loss(torch, model, tokenizer, candidate, args, device),
                                              weight, args.accumulation)
                        samples.append({"record_id": candidate.record["record_id"],
                                        "scale": candidate_scale(candidate, Arm(args.arm)),
                                        "measured_weight": candidate.weight, "loss_multiplier": weight})
                        total_loss += float(loss.detach().cpu())
                        if not math.isfinite(total_loss):
                            raise ValueError("training loss is not finite")
                        loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    completed = step
                    row: dict[str, Any] = {
                        "step": step,
                        "train_loss": total_loss,
                        "seconds": time.monotonic() - step_start,
                        "samples": samples,
                    }
                    if (
                        (step % args.eval_every == 0 or step == args.max_steps)
                        and not stopped["requested"]
                        and not expired(args)
                    ):
                        value = validation_loss(
                            torch,
                            model,
                            tokenizer,
                            data.validation,
                            args,
                            device,
                            demonstration.validation if demonstration else None,
                            explicit_weights=explicit_weights,
                        )
                        row["validation_loss"] = value
                        location = checkpoint(f"checkpoint-{step}", step, value)
                        current_checkpoint = str(location)
                        last_validation = step
                        if value < best:
                            best, best_path = value, current_checkpoint
                            manifest_path = location / ADAPTER_MANIFEST
                            atomic_json(
                                output / "best.json",
                                {
                                    "step": step,
                                    "validation_loss": value,
                                    "adapter": best_path,
                                    **({"adapter_fingerprint": parse_json(manifest_path.read_text())["fingerprint"]}
                                       if manifest_path.is_file() else {}),
                                },
                            )
                    curve.write(json.dumps(row) + "\n")
                    curve.flush()
                    os.fsync(curve.fileno())
                    atomic_json(
                        output / "status.json",
                        {
                            "status": "running",
                            "completed_steps": completed,
                            "best_adapter": best_path,
                            "latest_checkpoint": current_checkpoint,
                            "last_measurement": row,
                            **memory(torch),
                        },
                    )
                    print(json.dumps(row), flush=True)
            last = (
                str(checkpoint(f"checkpoint-last-{completed}", completed, None))
                if completed
                else None
            )
            status, reason = completion_status(
                completed, args.max_steps, last_validation
            )
            atomic_json(
                output / "status.json",
                {
                    "status": status,
                    "reason": reason,
                    "completed_steps": completed,
                    "last_validation_step": last_validation,
                    "best_adapter": best_path,
                    "latest_checkpoint": current_checkpoint,
                    "last_adapter": last,
                    "elapsed_seconds": time.monotonic() - started,
                    **memory(torch),
                },
            )
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    except Exception as error:
        atomic_json(
            output / "status.json",
            {
                "status": "failed",
                "completed_steps": completed,
                "best_adapter": best_path,
                "latest_checkpoint": current_checkpoint,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
