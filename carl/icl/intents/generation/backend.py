"""Local causal-language-model loading, inference and training utilities."""

import argparse
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from carl.icl.intents.generation.data import encode_example
from carl.icl.intents.generation.data import prompt_messages


def sync_gpu(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def initialise(
    args: argparse.Namespace, adapter: str | None = None, train: bool = False
):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    if device == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU execution is not enabled")
    if (
        device == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("this CUDA device does not support bfloat16")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Transformers 4.57.6 may call model_info for a hub identifier even when
    # local_files_only is set.  Resolve the cached snapshot first and hand the
    # resulting directory to both loaders, avoiding an accidental network call.
    source = (
        str(Path(args.model).resolve())
        if Path(args.model).is_dir()
        else snapshot_download(
            args.model,
            revision=args.revision,
            local_files_only=True,
        )
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
            ),
        )
    elif adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return torch, tokenizer, model, device


def tensor_example(
    torch: Any,
    example: dict[str, list[int]],
    device: str,
    *,
    pad_to_length: int = 0,
    pad_token_id: int | None = None,
) -> dict[str, Any]:
    """Optionally equalise model token budgets without supervising padding."""
    length = len(example["input_ids"])
    if pad_to_length and length > pad_to_length:
        raise ValueError(
            f"complete example needs {length} tokens; padding limit is {pad_to_length}"
        )
    padding = pad_to_length - length if pad_to_length else 0
    if padding and pad_token_id is None:
        raise ValueError("a pad token is required for fixed-length training")
    values = {
        "input_ids": example["input_ids"] + [pad_token_id] * padding,
        "attention_mask": example["attention_mask"] + [0] * padding,
        "labels": example["labels"] + [-100] * padding,
    }
    return {key: torch.tensor([value], device=device) for key, value in values.items()}


def memory(torch: Any) -> dict[str, Any]:
    import psutil

    return {
        "host_rss_bytes": psutil.Process().memory_info().rss,
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()
        if torch.cuda.is_available()
        else None,
    }


def validation_loss(
    torch: Any,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> float:
    model.eval()
    losses: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for record in records:
            x = tensor_example(
                torch, encode_example(tokenizer, record, args.max_length), device
            )
            losses[str(record["layout_id"])].append(
                float(model(**x).loss.detach().cpu())
            )
    model.train()
    if not losses:
        raise ValueError("no validation boards")
    return sum(sum(board) / len(board) for board in losses.values()) / len(losses)


def generate(
    torch: Any,
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    args: argparse.Namespace,
    device: str,
    messages: list[dict[str, str]] | None = None,
):
    ids = tokenizer.apply_chat_template(
        prompt_messages(record) if messages is None else messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    ).to(device)
    if ids.shape[1] + args.max_new_tokens > args.max_length:
        raise ValueError(
            "prompt plus requested generation exceeds max_length; no truncation performed"
        )
    model.eval()
    sync_gpu(torch)
    started = time.monotonic()
    with torch.no_grad():
        result = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    sync_gpu(torch)
    elapsed = time.monotonic() - started
    generated = result[0, ids.shape[1] :]
    output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return output, int(generated.shape[0]), elapsed
