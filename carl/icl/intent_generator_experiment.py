"""Small, leakage-controlled causal-LM experiment for Sokoban intents.

The JSONL input is the checked trajectory data, but the generator is *only*
shown ``start_observation`` and ``task_goal``.  The other fields in each
record are for audit and evaluation, not for conditioning the language model.

Run ``python -m carl.icl.intent_generator_experiment --help`` for commands.
Models are loaded from the local Hugging Face cache with ``local_files_only``.
No implicit truncation, downloading or external logging is performed.
"""

import argparse
import hashlib
import json
import math
import os
import random
import resource
import signal
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

SYSTEM = ("You propose one achievable, non-trivial Sokoban intent using only the current board. "
          "Rows and columns are zero-based. # is wall, - is floor, . is goal, $ is box, "
          "* is box on goal, @ is player and + is player on goal. "
          "Return only a JSON object with keys effects, preserve, instruction. "
          "effects is a nonempty array of objects, each either "
          '{"predicate":"box_at","row":R,"col":C}, '
          '{"predicate":"player_at","row":R,"col":C}, or '
          '{"predicate":"boxes_on_goals_at_least","count":N}. '
          "Every effect must be false on the current board. All effects are required together. "
          "preserve is an array and may be empty; instruction is null or a short string.")
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


class DataSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class GeneratorMethod(StrEnum):
    BASE = "base"
    ADAPTED = "adapted"
    IN_CONTEXT = "in_context"


class RunMode(StrEnum):
    BENCHMARK = "benchmark"
    TRAIN = "train"
    EVALUATE = "evaluate"


def prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Whitelist permitted input fields; do not serialise the record itself."""
    board = record["start_observation"]
    goal = record["task_goal"]
    if not isinstance(board, list) or not board or not all(isinstance(row, str) for row in board):
        raise ValueError("start_observation must be a nonempty list of board rows")
    if not isinstance(goal, str):
        raise ValueError("task_goal must be a string")
    return [
        {
            "role": "system",
            "content": SYSTEM
        },
        {
            "role": "user",
            "content": "Task: " + goal + "\nCurrent board:\n" + "\n".join(board)
        },
    ]


def target_text(record: dict[str, Any]) -> str:
    intent = record["intent"]
    if not isinstance(intent, dict) or not isinstance(intent.get("effects"), list) or not intent["effects"]:
        raise ValueError("checked intent must contain a nonempty effects array")
    return json.dumps(intent, ensure_ascii=False, separators=(",", ":"))


def read_records(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                if split is None or record.get("split") == split:
                    rows.append(record)
    return rows


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    return json.loads(text, parse_constant=reject_constant)


def encode_example(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    prefix = tokenizer.apply_chat_template(
        prompt_messages(record),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prefix, list):
        prefix = prefix.tolist()
    end = tokenizer.eos_token or ""
    answer = tokenizer.encode(target_text(record) + end, add_special_tokens=False)
    ids = prefix + answer
    if len(ids) > max_length:
        raise ValueError(f"complete example needs {len(ids)} tokens; limit is {max_length}; no truncation performed")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prefix) + answer}


def board_balanced(records: list[dict[str, Any]], seed: int) -> Iterator[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["layout_id"])].append(record)
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in groups}
    while groups:
        keys = list(groups)
        rng.shuffle(keys)
        for key in keys:
            if not pools[key]:
                pools[key] = rng.sample(groups[key], len(groups[key]))
            yield pools[key].pop()


def one_per_board(records: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    """Deterministically pick the first context per independent layout."""
    seen: set[str] = set()
    result = []
    for record in records:
        key = str(record["layout_id"])
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
        if limit and len(result) >= limit:
            break
    return result


def validation_contexts(records: list[dict[str, Any]], max_boards: int, per_board: int) -> list[dict[str, Any]]:
    """Select fixed contexts without inspecting model results; diversify effect types."""
    boards: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        boards.setdefault(str(record["layout_id"]), []).append(record)
    selected = []
    for board_records in list(boards.values())[:max_boards or None]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in board_records:
            groups.setdefault(str(record.get("effect_type", "unknown")), []).append(record)
        board_selected: list[dict[str, Any]] = []
        while len(board_selected) < per_board and any(groups.values()):
            for group in groups.values():
                if group and len(board_selected) < per_board:
                    board_selected.append(group.pop(0))
        selected.extend(board_selected)
    return selected


def evaluation_records(records: list[dict[str, Any]], ids_path: str | None, limit: int) -> list[dict[str, Any]]:
    if ids_path is None:
        return one_per_board(records, limit)
    if limit:
        raise ValueError("use either --test-ids or --test-boards, not both")
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))["record_ids"]
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError("frozen test IDs must be a unique list")
    lookup = {record["record_id"]: record for record in records}
    if any(record_id not in lookup for record_id in ids):
        raise ValueError("frozen test ID is absent from test records")
    return [lookup[record_id] for record_id in ids]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sync_gpu(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def initialise(args: argparse.Namespace, adapter: str | None = None, train: bool = False):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU execution is not enabled")
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support bfloat16")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Transformers 4.57.6 may call model_info for a hub identifier even when
    # local_files_only is set.  Resolve the cached snapshot first and hand the
    # resulting directory to both loaders, avoiding an accidental network call.
    source = str(Path(args.model).resolve()) if Path(args.model).is_dir() else snapshot_download(
        args.model,
        revision=args.revision,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if tokenizer.chat_template is None:
        raise ValueError("the selected tokenizer does not supply a chat template")
    model: Any = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model = model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if train:
        from peft import LoraConfig
        from peft import TaskType
        from peft import get_peft_model

        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.rank,
                lora_alpha=2 * args.rank,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ))
    elif adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return torch, tokenizer, model, device


def tensor_example(torch: Any, example: dict[str, list[int]], device: str) -> dict[str, Any]:
    return {key: torch.tensor([value], device=device) for key, value in example.items()}


def lengths(tokenizer: Any, records: list[dict[str, Any]], max_length: int) -> dict[str, Any]:
    values = [len(encode_example(tokenizer, record, max_length)["input_ids"]) for record in records]
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
        "truncated": 0
    }


def memory(torch: Any) -> dict[str, Any]:
    import psutil

    return {
        "host_rss_bytes": psutil.Process().memory_info().rss,
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
    }


def validation_loss(torch: Any, model: Any, tokenizer: Any, records: list[dict[str, Any]], args: argparse.Namespace,
                    device: str) -> float:
    model.eval()
    losses: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for record in records:
            x = tensor_example(torch, encode_example(tokenizer, record, args.max_length), device)
            losses[str(record["layout_id"])].append(float(model(**x).loss.detach().cpu()))
    model.train()
    if not losses:
        raise ValueError("no validation boards")
    return sum(sum(board) / len(board) for board in losses.values()) / len(losses)


def generate(torch: Any,
             model: Any,
             tokenizer: Any,
             record: dict[str, Any],
             args: argparse.Namespace,
             device: str,
             messages: list[dict[str, str]] | None = None):
    ids = tokenizer.apply_chat_template(
        prompt_messages(record) if messages is None else messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)
    if ids.shape[1] + args.max_new_tokens > args.max_length:
        raise ValueError("prompt plus requested generation exceeds max_length; no truncation performed")
    model.eval()
    sync_gpu(torch)
    started = time.monotonic()
    with torch.no_grad():
        result = model.generate(input_ids=ids,
                                attention_mask=torch.ones_like(ids),
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                use_cache=True)
    sync_gpu(torch)
    elapsed = time.monotonic() - started
    generated = result[0, ids.shape[1]:]
    output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return output, int(generated.shape[0]), elapsed


def benchmark(args: argparse.Namespace) -> None:
    records = read_records(args.data)
    training = [row for row in records if row["split"] == DataSplit.TRAIN]
    validation = [row for row in records if row["split"] == DataSplit.VALIDATION]
    if len(training) < args.accumulation:
        raise ValueError("not enough train examples for benchmark accumulation")
    started = time.monotonic()
    torch, tokenizer, model, device = initialise(args, train=True)
    # Held-out targets must not affect the training resource check.
    token_lengths = lengths(tokenizer, training + validation, args.max_length)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    sync_gpu(torch)
    start_step = time.monotonic()
    losses = []
    tokens_processed = 0
    for record in training[:args.accumulation]:
        x = tensor_example(torch, encode_example(tokenizer, record, args.max_length), device)
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
    text, tokens, generation_seconds = generate(torch, model, tokenizer, training[0], args, device)
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
        "training_tokens_per_second": tokens_processed / step_seconds,
        "generation_seconds": generation_seconds,
        "generation_tokens": tokens,
        "generation_tokens_per_second": tokens / generation_seconds if generation_seconds else None,
        "generation_sample": text,
        "elapsed_seconds": time.monotonic() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **memory(torch)
    }
    atomic_json(Path(args.output), summary)
    print(json.dumps(summary, indent=2), flush=True)


def train(args: argparse.Namespace) -> None:
    rows = read_records(args.data)
    training = [row for row in rows if row["split"] == DataSplit.TRAIN]
    validation = validation_contexts(
        [row for row in rows if row["split"] == DataSplit.VALIDATION],
        args.validation_boards,
        args.validation_contexts_per_board,
    )
    if not training or not validation:
        raise ValueError("train and validation splits must both contain boards")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch, tokenizer, model, device = initialise(args, train=True)
    # Inspect only examples actually used for fitting or validation.
    token_lengths = lengths(tokenizer, training + validation, args.max_length)
    atomic_json(
        output / "configuration.json", {
            "args": vars(args),
            "device": device,
            "data_sha256": file_sha256(args.data),
            "token_lengths": token_lengths,
            "enable_thinking": False,
            "validation_layouts": sorted({r["layout_id"] for r in validation}),
            "validation_record_ids": [r["record_id"] for r in validation],
            "command": sys.argv
        })
    tokenizer.save_pretrained(output / "tokenizer")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    samples = board_balanced(training, args.seed)
    stop = {"requested": False}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    best = float("inf")
    best_path = None
    completed = 0
    model.train()
    with (output / "training_curve.jsonl").open("w", encoding="utf-8", buffering=1) as curve:
        for step in range(1, args.max_steps + 1):
            if stop["requested"] or (args.stop_monotonic is not None and time.monotonic() >= args.stop_monotonic):
                break
            started = time.monotonic()
            total_loss = 0.0
            for _ in range(args.accumulation):
                record = next(samples)
                x = tensor_example(torch, encode_example(tokenizer, record, args.max_length), device)
                loss = model(**x).loss / args.accumulation
                total_loss += float(loss.detach().cpu())
                if not math.isfinite(total_loss):
                    raise RuntimeError(f"non-finite training loss at step {step}")
                loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed = step
            row = {"step": step, "train_loss": total_loss, "seconds": time.monotonic() - started}
            if step % args.eval_every == 0 or step == args.max_steps:
                val = validation_loss(torch, model, tokenizer, validation, args, device)
                row["validation_loss"] = val
                if val < best:
                    best = val
                    destination = output / f"checkpoint-{step}"
                    model.save_pretrained(str(destination), safe_serialization=True)
                    best_path = str(destination)
                    atomic_json(output / "best.json", {"step": step, "validation_loss": val, "adapter": best_path})
            curve.write(json.dumps(row) + "\n")
            atomic_json(
                output / "status.json", {
                    "status": "running",
                    "completed_steps": completed,
                    "best_adapter": best_path,
                    "last_measurement": row,
                    **memory(torch)
                })
            print(json.dumps(row), flush=True)
    last_path = output / f"checkpoint-last-{completed}"
    model.save_pretrained(str(last_path), safe_serialization=True)
    atomic_json(
        output / "status.json", {
            "status": "finished" if completed == args.max_steps else "stopped",
            "completed_steps": completed,
            "best_adapter": best_path,
            "last_adapter": str(last_path),
            **memory(torch)
        })


def evaluation_configuration(args: argparse.Namespace) -> dict[str, Any]:
    adapter = Path(args.adapter) if args.adapter else None
    return {
        "model": args.model,
        "revision": args.revision,
        "adapter": args.adapter,
        "adapter_weights_sha256": file_sha256(adapter / "adapter_model.safetensors") if adapter else None,
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json") if adapter else None,
        "data_sha256": file_sha256(args.data),
        "test_ids_sha256": file_sha256(args.test_ids) if args.test_ids else None,
        "prompt_sha256": hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest(),
        "dtype": args.dtype,
        "seed": args.seed,
        "test_boards": args.test_boards,
        "decoding": "greedy",
        "enable_thinking": False,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
    }


def prepare_evaluation_output(output: Path, configuration: dict[str, Any], records: list[dict[str, Any]],
                              resume: bool) -> list[dict[str, Any]]:
    """Verify an interrupted output belongs to this exact frozen experiment."""
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".run.json")
    if not resume and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"Evaluation output already exists: {output}; use --resume")
    if output.exists() and not metadata_path.exists():
        raise ValueError("Cannot resume output without its original configuration")
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Evaluation configuration changed; refusing to resume")
    else:
        atomic_json(metadata_path, configuration)
    if not output.exists():
        return []
    raw = output.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        tail = raw[boundary:]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            saved_tail = output.with_name(f"{output.name}.partial-{hashlib.sha256(tail).hexdigest()[:12]}")
            saved_tail.write_bytes(tail)
            raw = raw[:boundary]
        else:
            raw += b"\n"
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, output)
    rows = read_records(output)
    method = configuration.get("method") or (GeneratorMethod.ADAPTED
                                             if configuration["adapter"] else GeneratorMethod.BASE)
    if len(rows) > len(records):
        raise ValueError("Output contains more contexts than the frozen selection")
    for row, record in zip(rows, records, strict=False):
        if (row.get("record_id") != record["record_id"] or row.get("layout_id") != record["layout_id"]
                or row.get("split") != DataSplit.TEST or row.get("method") != method):
            raise ValueError("Output does not match the ordered frozen test selection")
        if not isinstance(row.get("raw_output"), str) or not isinstance(row.get("json_valid"), bool):
            raise ValueError("Output has malformed generation fields")
        try:
            parsed = parse_json(row["raw_output"])
            valid = True
        except (json.JSONDecodeError, ValueError):
            parsed, valid = None, False
        if row["json_valid"] != valid or row.get("parsed") != parsed:
            raise ValueError("Stored JSON result disagrees with its raw output")
    return rows


def evaluate(args: argparse.Namespace) -> None:
    records = evaluation_records(read_records(args.data, DataSplit.TEST), args.test_ids, args.test_boards)
    if not records:
        raise ValueError("no held-out test boards")
    output = Path(args.output)
    configuration = evaluation_configuration(args)
    previous = prepare_evaluation_output(output, configuration, records, args.resume)
    torch, tokenizer, model, device = initialise(args, adapter=args.adapter, train=False)
    started = time.monotonic()
    with output.open("a", encoding="utf-8", buffering=1) as dest:
        for record in records[len(previous):]:
            output_text, tokens, elapsed = generate(torch, model, tokenizer, record, args, device)
            try:
                parsed = parse_json(output_text)
                valid_json = True
            except (json.JSONDecodeError, ValueError):
                parsed = None
                valid_json = False
            row = {
                "record_id": record["record_id"],
                "layout_id": record["layout_id"],
                "split": DataSplit.TEST,
                "method": GeneratorMethod.ADAPTED if args.adapter else GeneratorMethod.BASE,
                "raw_output": output_text,
                "json_valid": valid_json,
                "parsed": parsed,
                "generated_tokens": tokens,
                "generation_seconds": elapsed
            }
            dest.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({
                "record_id": record["record_id"],
                "json_valid": valid_json,
                "seconds": elapsed
            }),
                  flush=True)
    atomic_json(
        output.with_suffix(".summary.json"), {
            "status": "complete",
            "n": len(records),
            **configuration, "resumed_records": len(previous),
            "elapsed_seconds": time.monotonic() - started,
            **memory(torch)
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(RunMode))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="exact model commit; required when using another model")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--validation-boards", type=int, default=12)
    parser.add_argument("--validation-contexts-per-board", type=int, default=4)
    parser.add_argument("--test-boards", type=int, default=0)
    parser.add_argument("--test-ids", help="JSON file with the pre-frozen record_ids array")
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--adapter")
    parser.add_argument("--resume",
                        action="store_true",
                        help="resume an interrupted evaluation after validating its configuration")
    args = parser.parse_args()
    if args.revision is None:
        if args.model != DEFAULT_MODEL:
            parser.error("--revision is required for a non-default model or local model directory")
        args.revision = DEFAULT_REVISION
    if args.accumulation < 1 or args.max_length < 1 or args.eval_every < 1 or args.validation_contexts_per_board < 1:
        parser.error("accumulation, max-length, eval-every and validation-contexts-per-board must be positive")
    {RunMode.BENCHMARK: benchmark, RunMode.TRAIN: train, RunMode.EVALUATE: evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
