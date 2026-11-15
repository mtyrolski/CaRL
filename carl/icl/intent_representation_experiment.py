"""Compare Sokoban intent representations using checked support examples."""

import argparse
import json
import os
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import evaluation_records
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import generate
from carl.icl.intent_generator_experiment import initialise
from carl.icl.intent_generator_experiment import memory
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intents.provenance import GENERATION_COMPONENTS
from carl.icl.intents.provenance import REPRESENTATION_COMPONENTS
from carl.icl.intents.provenance import component_hashes
from carl.icl.intents.representation.formats import Representation
from carl.icl.intents.representation.formats import parse_representation
from carl.icl.intents.representation.formats import render_representation
from carl.icl.intents.representation.predictions import _parse_output
from carl.icl.intents.representation.predictions import prepare_output
from carl.icl.intents.representation.prompts import COMMON
from carl.icl.intents.representation.prompts import REPRESENTATION_INSTRUCTIONS
from carl.icl.intents.representation.prompts import SupportSet
from carl.icl.intents.representation.prompts import load_supports
from carl.icl.intents.representation.prompts import messages_for
from carl.icl.intents.representation.prompts import support_bank
from carl.icl.intents.representation.prompts import support_manifest

__all__ = [
    "COMMON",
    "REPRESENTATION_INSTRUCTIONS",
    "Representation",
    "SupportSet",
    "Mode",
    "configuration",
    "evaluate",
    "load_supports",
    "main",
    "messages_for",
    "parse_representation",
    "prepare",
    "prepare_output",
    "prompt_messages",
    "render_representation",
    "support_bank",
    "support_manifest",
]


class Mode(StrEnum):
    FREEZE_SUPPORTS = "freeze-supports"
    INSPECT = "inspect-prompts"
    EVALUATE = "evaluate"


def _write_once(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise ValueError(f"refusing to overwrite a different frozen file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def prepare(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path]:
    from transformers import AutoTokenizer

    all_rows = read_records(args.data, args.split)
    ids = json.loads(Path(args.test_ids).read_text(encoding="utf-8"))
    if ids.get("dataset_sha256") not in (None, file_sha256(args.data)):
        raise ValueError("frozen IDs belong to a different dataset")
    records = evaluation_records(all_rows, args.test_ids, 0)
    if not records or len({r["record_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("empty selection or duplicate records")
    support = load_supports(
        Path(args.support_data),
        Path(args.support_manifest),
        SupportSet(args.support_set),
    )
    if {row["layout_id"] for row in support} & {row["layout_id"] for row in records}:
        raise ValueError("support and evaluated layouts overlap")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.chat_template is None:
        raise ValueError("model tokenizer has no chat template")
    prompts = []
    for record in records:
        messages = messages_for(record, support, args.representation)
        options = {"add_generation_prompt": True, "enable_thinking": False}
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, **options)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        if len(tokens) + args.max_new_tokens > args.max_length:
            raise ValueError(
                f"complete prompt needs {len(tokens)} + {args.max_new_tokens} tokens; "
                f"limit is {args.max_length}; no truncation"
            )
        prompts.append(
            {
                "record_id": record["record_id"],
                "layout_id": record["layout_id"],
                "messages": messages,
                "rendered_prompt": rendered,
                "prompt_tokens": len(tokens),
            }
        )
    prompt_path = Path(args.output).with_suffix(".prompts.jsonl")
    _write_once(
        prompt_path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts),
    )
    return records, prompts, prompt_path


def configuration(args: argparse.Namespace, prompt_path: Path) -> dict[str, Any]:
    adapter = Path(args.adapter) if args.adapter else None
    model = Path(args.model)
    if not model.is_dir():
        raise ValueError("use an existing, explicitly resolved local model snapshot")
    revision_metadata = model / "experiment_revision.json"
    if revision_metadata.exists():
        if (
            json.loads(revision_metadata.read_text(encoding="utf-8")).get("revision")
            != args.revision
        ):
            raise ValueError(
                "local model revision metadata does not match the requested revision"
            )
        revision_verification = "experiment_revision.json"
    elif re.fullmatch(r"[0-9a-f]{40}", model.name) and model.name == args.revision:
        revision_verification = "huggingface snapshot directory"
    else:
        raise ValueError(
            "cannot verify local model revision from metadata or snapshot directory"
        )
    return {
        "schema_version": 1,
        "variant": args.variant or f"{args.representation}_{args.support_set}",
        "representation": args.representation,
        "support_set": args.support_set,
        "model": str(model.resolve()),
        "revision": args.revision,
        "revision_verification": revision_verification,
        "model_weights_sha256": file_sha256(model / "model.safetensors"),
        "model_config_sha256": file_sha256(model / "config.json"),
        "tokenizer_sha256": file_sha256(model / "tokenizer.json"),
        "adapter": str(adapter.resolve()) if adapter else None,
        "adapter_weights_sha256": file_sha256(adapter / "adapter_model.safetensors")
        if adapter
        else None,
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json")
        if adapter
        else None,
        "data_sha256": file_sha256(args.data),
        "test_ids_sha256": file_sha256(args.test_ids),
        "support_data_sha256": file_sha256(args.support_data),
        "support_manifest_sha256": file_sha256(args.support_manifest),
        "prompts_sha256": file_sha256(prompt_path),
        "protocol_sha256": file_sha256(__file__),
        "protocol_components_sha256": component_hashes(REPRESENTATION_COMPONENTS),
        "generator_sha256": file_sha256(
            Path(__file__).with_name("intent_generator_experiment.py")
        ),
        "generator_components_sha256": component_hashes(GENERATION_COMPONENTS),
        "support_selection_sha256": file_sha256(
            Path(__file__).with_name("intent_in_context.py")
        ),
        "split": args.split,
        "dtype": args.dtype,
        "seed": args.seed,
        "threads": args.threads,
        "decoding": "greedy",
        "enable_thinking": False,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "require_cuda": args.require_cuda,
    }


def _write_partial(
    output: Path,
    count: int,
    planned: int,
    reason: str,
    error: BaseException | None = None,
) -> None:
    atomic_json(
        output.with_suffix(".partial-summary.json"),
        {
            "status": "stopped",
            "processed": count,
            "planned": planned,
            "reason": reason,
            "error_type": type(error).__name__ if error else None,
            "error": str(error) if error else None,
        },
    )


def evaluate(args: argparse.Namespace) -> None:
    records, prompts, prompt_path = prepare(args)
    if args.mode == Mode.INSPECT:
        counts = [item["prompt_tokens"] for item in prompts]
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "n": len(records),
                    "min_prompt_tokens": min(counts),
                    "max_prompt_tokens": max(counts),
                    "max_new_tokens": args.max_new_tokens,
                    "max_length": args.max_length,
                    "prompts_path": str(prompt_path),
                },
                indent=2,
            )
        )
        return
    config = configuration(args, prompt_path)
    output = Path(args.output)
    previous = prepare_output(output, config, records, args.resume)
    count = len(previous)
    started = time.monotonic()
    try:
        if (
            args.stop_monotonic is not None and time.monotonic() >= args.stop_monotonic
        ) or (args.stop_epoch is not None and time.time() >= args.stop_epoch):
            _write_partial(output, count, len(records), "deadline")
            raise SystemExit("deadline reached before model initialisation")
        if args.require_cuda:
            import torch as check_torch

            if not check_torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is required but unavailable; no silent CPU fallback"
                )
        torch, tokenizer, model, device = initialise(
            args, adapter=args.adapter, train=False
        )
        with output.open("a", encoding="utf-8", buffering=1) as destination:
            for record, prompt in zip(records[count:], prompts[count:], strict=True):
                if (
                    args.stop_monotonic is not None
                    and time.monotonic() >= args.stop_monotonic
                ) or (args.stop_epoch is not None and time.time() >= args.stop_epoch):
                    _write_partial(output, count, len(records), "deadline")
                    raise SystemExit("deadline reached; partial predictions retained")
                raw, tokens, elapsed = generate(
                    torch,
                    model,
                    tokenizer,
                    record,
                    args,
                    device,
                    messages=prompt["messages"],
                )
                parsed, valid, error = _parse_output(args.representation, raw)
                row = {
                    "record_id": record["record_id"],
                    "layout_id": record["layout_id"],
                    "split": args.split,
                    "variant": config["variant"],
                    "representation": args.representation,
                    "support_set": args.support_set,
                    "raw_output": raw,
                    "syntax_valid": valid,
                    "parsed_intent": parsed,
                    "syntax_error": error,
                    "generated_tokens": tokens,
                    "generation_seconds": elapsed,
                }
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                destination.flush()
                os.fsync(destination.fileno())
                count += 1
                print(
                    json.dumps(
                        {
                            "record_id": record["record_id"],
                            "syntax_valid": valid,
                            "generation_seconds": elapsed,
                        }
                    ),
                    flush=True,
                )
    except Exception as exc:
        _write_partial(output, count, len(records), "infrastructure_error", exc)
        raise
    atomic_json(
        output.with_suffix(".summary.json"),
        {
            "status": "complete",
            "n": count,
            "resumed_records": len(previous),
            "elapsed_seconds": time.monotonic() - started,
            **memory(torch),
        },
    )
    partial = output.with_suffix(".partial-summary.json")
    if partial.exists():
        old = json.loads(partial.read_text(encoding="utf-8"))
        old.update(
            final_status="complete",
            superseded_by=output.with_suffix(".summary.json").name,
        )
        atomic_json(partial, old)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=list(Mode))
    parser.add_argument("--data")
    parser.add_argument("--support-data", required=True)
    parser.add_argument("--support-manifest")
    parser.add_argument("--test-ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--representation", choices=list(Representation))
    parser.add_argument("--support-set", choices=list(SupportSet))
    parser.add_argument("--variant")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--adapter")
    parser.add_argument("--split", choices=list(DataSplit), default=DataSplit.TEST)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--stop-epoch", type=float)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == Mode.FREEZE_SUPPORTS:
        manifest = support_manifest(Path(args.support_data))
        _write_once(
            Path(args.output), json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        print(args.output)
        return
    if not all(
        (
            args.data,
            args.test_ids,
            args.support_manifest,
            args.representation,
            args.support_set,
            args.model,
            args.revision,
        )
    ):
        parser.error(
            "data, test IDs, support manifest/set, representation, model and revision are required"
        )
    if args.max_length < 1 or args.max_new_tokens < 1 or args.threads < 1:
        parser.error("length and thread limits must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
