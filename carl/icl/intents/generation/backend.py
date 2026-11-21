"""Local causal-language-model loading, inference and training utilities."""

import argparse
import hashlib
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from carl.icl.intents.generation.checkpoint import has_adapter_contract, verify_adapter_artifact, verify_model_revision
from carl.icl.intents.generation.data import encode_example
from carl.icl.intents.generation.data import prompt_messages


def sync_gpu(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def local_snapshot(model: str, revision: str) -> Path:
    """Resolve only a local directory or an already-cached pinned HF snapshot."""
    source = Path(model).expanduser()
    if not source.is_dir():
        # Resolve before calling Transformers to avoid model-info HTTP requests.
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(model, revision=revision, local_files_only=True))
    verify_model_revision(source, revision)
    return source.resolve()


def require_peft() -> Any:
    """Import the optional PEFT extra only when training or loading an adapter."""
    try:
        import peft
    except ModuleNotFoundError as error:
        if error.name != "peft":
            raise
        raise RuntimeError("LoRA requires the optional extra. Run uv sync --locked --extra lora") from error
    return peft


def initialise(
    args: argparse.Namespace, adapter: str | None = None, train: bool = False
):
    import torch
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    selected = "cuda" if getattr(args, "require_cuda", False) else getattr(args, "device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if selected == "auto" else selected
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable; no silent CPU fallback")
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
    peft = require_peft() if train or adapter else None
    source = local_snapshot(args.model, args.revision)
    checkpoint = Path(adapter) if adapter else None
    manifest = (
        verify_adapter_artifact(checkpoint, base=source, revision=args.revision, role="generator")
        if checkpoint and has_adapter_contract(checkpoint) else None
    )
    tokenizer_source = checkpoint / "tokenizer" if manifest is not None and checkpoint else source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    if tokenizer.chat_template is None:
        raise ValueError("the selected tokenizer does not supply a chat template")
    if manifest and hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != manifest["chat_template_sha256"]:
        raise ValueError("adapter chat template differs from training")
    model: Any = AutoModelForCausalLM.from_pretrained(
        source,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model = model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if train:
        assert peft is not None
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=args.rank,
                lora_alpha=2 * args.rank,
                lora_dropout=0.0,
                target_modules=getattr(args, "lora_targets", None) or ["q_proj", "v_proj"],
                bias="none",
                task_type=peft.TaskType.CAUSAL_LM,
            ),
        )
    elif adapter:
        assert peft is not None
        model = peft.PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True)
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
