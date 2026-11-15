#!/usr/bin/env python3
"""Generate paired Sokoban intents from one local, frozen Qwen instance on CPU.

The same original contexts are evaluated under three D4 orientations and three
methods. Only the current board and task enter the model. A fixed historical
support set and a validation-selected adapter are loaded from local files.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import resource
import time
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from filelock import FileLock

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_in_context import in_context_messages
from carl.icl.intent_in_context import load_support
from carl.icl.policy_relative_lora import assess_output
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "analysis/intent_generator_20261114T032232Z"
MODEL = HISTORICAL / "models/Qwen3-0.6B"
ADAPTER = HISTORICAL / "adapter/checkpoint-220"
SUPPORT = HISTORICAL / "icl_followup/support.json"
SUPPORT_DATA = HISTORICAL / "dataset.jsonl"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
GOAL = "Put every box on a goal."
FAMILIES = ("12x12x4", "12x12x10")


class Method(StrEnum):
    BASE = "base"
    ICL = "in_context"
    ADAPTED = "checkpoint220"


class Transform(StrEnum):
    IDENTITY = "identity"
    ROT90 = "rot90_ccw"
    FLIPLR = "horizontal_reflection"


METHODS = tuple(Method)
TRANSFORMS = tuple(Transform)


def transform_rows(rows: list[str], transform: Transform | str) -> list[str]:
    """Apply identity, counterclockwise 90° rotation, or left-right reflection."""
    transform = Transform(transform)
    if not rows or any(not isinstance(row, str) or len(row) != len(rows) for row in rows):
        raise ValueError("Expected a nonempty square character board")
    cells = np.asarray([list(row) for row in rows])
    result = (cells if transform == Transform.IDENTITY else np.rot90(cells) if transform == Transform.ROT90
              else np.fliplr(cells))
    return ["".join(row) for row in result.tolist()]


def map_coordinate(row: int, col: int, transform: Transform | str, size: int = 12) -> tuple[int, int]:
    transform = Transform(transform)
    if (not isinstance(size, int) or size < 1 or isinstance(row, bool) or isinstance(col, bool)
            or not isinstance(row, int) or not isinstance(col, int) or not 0 <= row < size or not 0 <= col < size):
        raise ValueError("Invalid zero-based coordinate")
    if transform == Transform.ROT90:
        return size - 1 - col, row
    if transform == Transform.FLIPLR:
        return row, size - 1 - col
    return row, col


def inverse_coordinate(row: int, col: int, transform: Transform | str, size: int = 12) -> tuple[int, int]:
    transform = Transform(transform)
    # Validate the coordinate even for identity and reflection.
    map_coordinate(row, col, Transform.IDENTITY, size)
    if transform == Transform.ROT90:
        return col, size - 1 - row
    if transform == Transform.FLIPLR:
        return row, size - 1 - col
    return row, col


def transform_actions(actions: list[int], transform: Transform | str, *, inverse: bool = False) -> list[int]:
    """Map UDLR actions to a transformed board, or back when inverse=True."""
    transform = Transform(transform)
    lookup = {Transform.IDENTITY: (0, 1, 2, 3), Transform.ROT90: (2, 3, 1, 0),
              Transform.FLIPLR: (0, 1, 3, 2)}[transform]
    if inverse:
        lookup = tuple(lookup.index(action) for action in range(4))
    if any(not isinstance(action, int) or isinstance(action, bool) or action not in range(4) for action in actions):
        raise ValueError("Expected UDLR actions 0..3")
    return [lookup[action] for action in actions]


def inverse_intent(intent: dict[str, Any], transform: Transform | str, size: int = 12) -> tuple[dict[str, Any], bool]:
    """Map structured predicates back; flag natural-language text as untranslated."""
    result = json.loads(json.dumps(intent, ensure_ascii=False, allow_nan=False))
    for group in ("effects", "preserve"):
        for predicate in result.get(group, []):
            if predicate.get("predicate") in {"box_at", "player_at"}:
                predicate["row"], predicate["col"] = inverse_coordinate(predicate["row"], predicate["col"],
                                                                        transform, size)
    instruction = result.get("instruction")
    return result, bool(isinstance(instruction, str) and instruction.strip())


def replay_transform(pair: dict[str, Any], transform: Transform | str) -> dict[str, Any]:
    """Verify an original and transformed demonstration at every simulator step."""
    original, target = pair["source"], pair["target"]
    actions = pair.get("demonstration_actions")
    if not isinstance(actions, list) or len(actions) != pair["k"]:
        raise ValueError("Demonstration action count differs from its offset")
    if state_hash(from_rows(original)) != pair["source_hash"] or state_hash(from_rows(target)) != pair["target_hash"]:
        raise ValueError("Recorded source or target hash differs")
    transformed = transform_rows(original, transform)
    left, right = environment(from_rows(original)), environment(from_rows(transformed))
    state, altered = from_rows(original), from_rows(transformed)
    for action, mapped in zip(actions, transform_actions(actions, transform), strict=True):
        if left.is_solved(state) or right.is_solved(altered):
            raise ValueError("Demonstration continues after the task terminates")
        state = np.asarray(left.next_state(state, action))
        altered = np.asarray(right.next_state(altered, mapped))
        if to_rows(altered) != transform_rows(to_rows(state), transform):
            raise ValueError("Transformed action does not replay the transformed state")
    if to_rows(state) != target or to_rows(altered) != transform_rows(target, transform):
        raise ValueError("Demonstration endpoint does not replay")
    if canonical_layout_id(from_rows(transformed)) != pair["layout_id"]:
        raise ValueError("Transformed board changes canonical layout")
    return {"reference_pair_id": pair["pair_id"], "demonstration_offset": pair["k"],
            "actions_replayed": len(actions), "transform": str(transform)}


def select_contexts(rows: list[dict[str, Any]], split: str, per_family: int,
                    support_layouts: set[str], training_layouts: set[str] | None = None) -> list[dict[str, Any]]:
    """Use fixed board-rank prefixes, balanced across evaluation source families."""
    if split not in {"development", "evaluation"} or per_family < 1:
        raise ValueError("Invalid split or board count")
    role_by_layout: dict[str, str] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if role_by_layout.setdefault(row["layout_id"], row["split"]) != row["split"]:
            raise ValueError("Canonical layout overlaps development and evaluation")
        if row["split"] == split and row["context_rank"] == 0:
            groups.setdefault(row["context_id"], []).append(row)
    contexts = []
    training_layouts = training_layouts or set()
    for context_id, pairs in groups.items():
        first = pairs[0]
        keys = ("layout_id", "family", "source_family", "split", "board_rank", "source", "source_hash")
        if any(any(pair[key] != first[key] for key in keys) for pair in pairs):
            raise ValueError("A context contains conflicting source data")
        if first["layout_id"] in support_layouts:
            if split == "evaluation":
                raise ValueError("Evaluation layout overlaps an ICL support layout")
            continue
        if split == "evaluation" and first["layout_id"] in training_layouts:
            raise ValueError("Evaluation layout overlaps historical adapter training")
        eligible = [pair for pair in pairs if pair["k"] <= 16]
        if not eligible:
            raise ValueError("No recorded fragment at k<=16 for the D4 replay check")
        reference = max(eligible, key=lambda row: (row["k"], row["pair_id"]))
        contexts.append({**{key: first[key] for key in keys}, "context_id": context_id,
                         "adapter_training_exposure": first["layout_id"] in training_layouts,
                         "reference_pair": reference})
    by_family: dict[str, list[dict[str, Any]]] = {family: [] for family in FAMILIES}
    for context in contexts:
        if context["family"] not in by_family:
            raise ValueError("Unsupported board family")
        by_family[context["family"]].append(context)
    for values in by_family.values():
        values.sort(key=lambda row: (row["board_rank"], row["context_id"]))
    if split == "development":
        if any(len(by_family[family]) < per_family for family in FAMILIES):
            raise ValueError("Not enough development boards after excluding ICL supports")
        # Prefer layouts not used to train the adapter; retain and label known
        # exposure only when the development pool cannot otherwise satisfy N.
        chosen = [row for family in FAMILIES for row in sorted(
            by_family[family], key=lambda row: (row["adapter_training_exposure"], row["board_rank"]))[:per_family]]
    else:
        if per_family % 2:
            raise ValueError("Evaluation per-family count must be even for balanced x10 sources")
        x4 = by_family["12x12x4"][:per_family]
        x10 = by_family["12x12x10"]
        sources = sorted({row["source_family"] for row in x10})
        if len(sources) != 2:
            raise ValueError("Expected exactly two evaluation x10 source distributions")
        source_groups = [[row for row in x10 if row["source_family"] == source][:per_family // 2]
                         for source in sources]
        if len(x4) != per_family or any(len(group) != per_family // 2 for group in source_groups):
            raise ValueError("Insufficient boards for the requested balanced evaluation")
        chosen = x4 + [row for group in source_groups for row in group]
    return sorted(chosen, key=lambda row: (row["board_rank"], FAMILIES.index(row["family"]), row["context_id"]))


def public_record(context: dict[str, Any], transform: Transform | str) -> dict[str, Any]:
    """Whitelist generator-visible fields and the ID needed for support separation."""
    return {"layout_id": context["layout_id"], "start_observation": transform_rows(context["source"], transform),
            "task_goal": GOAL}


def case_id(cohort: str, context_id: str, method: Method | str, transform: Transform | str) -> str:
    return digest({"cohort_id": cohort, "context_id": context_id, "method": str(method),
                   "transform": str(transform)})


def build_plan(contexts: list[dict[str, Any]], support: list[dict[str, Any]], tokenizer: Any,
               cohort: str, max_length: int, max_new_tokens: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze cases, full prompts, and replay checks before loading model weights."""
    prompts, cases, checks = [], [], []
    if not cohort or max_length < 1 or max_new_tokens < 1:
        raise ValueError("Cohort and positive token limits are required")
    for context in contexts:
        original = context["source"]
        for transform in TRANSFORMS:
            checks.append({"context_id": context["context_id"], **replay_transform(context["reference_pair"], transform)})
            record = public_record(context, transform)
            for method in METHODS:
                messages = in_context_messages(record, support) if method == Method.ICL else prompt_messages(record)
                rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                          enable_thinking=False)
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    enable_thinking=False)
                if len(ids) + max_new_tokens > max_length:
                    raise ValueError("Complete prompt and output budget exceed max length; no truncation allowed")
                identifier = case_id(cohort, context["context_id"], method, transform)
                public = {"case_id": identifier, "cohort_id": cohort, "context_id": context["context_id"],
                          "layout_id": context["layout_id"], "family": context["family"],
                          "source_family": context["source_family"], "split": context["split"],
                          "board_rank": context["board_rank"],
                          "adapter_training_exposure": context["adapter_training_exposure"],
                          "method": str(method), "transform": str(transform),
                          "original_rows": original, "source_rows": record["start_observation"],
                          "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                          "prompt_tokens": len(ids)}
                cases.append(public)
                prompts.append({"case_id": identifier, "messages": messages, "rendered_prompt": rendered,
                                "prompt_sha256": public["prompt_sha256"]})
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate planned case ID")
    return {"schema_version": 1, "cohort_id": cohort, "contexts": len(contexts),
            "family_contexts": dict(Counter(context["family"] for context in contexts)),
            "source_contexts": dict(Counter(context["source_family"] for context in contexts)),
            "context_manifest": [{**{key: context[key] for key in ("context_id", "layout_id", "family",
                                  "source_family", "split", "board_rank", "adapter_training_exposure")},
                                  "original_rows": context["source"], "source_hash": context["source_hash"],
                                  "cohort_id": cohort, "expected_methods": [str(method) for method in METHODS],
                                  "expected_transforms": [str(transform) for transform in TRANSFORMS]}
                                 for context in contexts],
            "support_intents": [{"record_id": record["record_id"], "layout_id": record["layout_id"],
                                 "intent": record["intent"]} for record in support],
            "cases": cases, "replay_checks": checks}, prompts


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model(args: argparse.Namespace):
    """One CPU-resident model instance, with its adapter switched on/off in place."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    if tokenizer.chat_template is None:
        raise ValueError("Cached tokenizer has no chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    base = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype).to("cpu")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).to("cpu").eval()
    model.requires_grad_(False)
    return torch, tokenizer, model


def model_batch(torch: Any, tokenizer: Any, model: Any, prompts: list[str], *, adapted: bool,
                max_new_tokens: int, max_length: int) -> tuple[list[tuple[str, int]], float]:
    """Greedy left-padded generation with no implicit input truncation."""
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    if encoded["input_ids"].shape[1] + max_new_tokens > max_length:
        raise ValueError("Padded complete prompts exceed max length; no truncation allowed")
    manager = contextlib.nullcontext() if adapted else model.disable_adapter()
    started = time.monotonic()
    with manager, torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
    elapsed = time.monotonic() - started
    results = []
    prefix = encoded["input_ids"].shape[1]
    for tokens in output[:, prefix:].tolist():
        eos = tokenizer.eos_token_id
        if eos in tokens:
            tokens = tokens[:tokens.index(eos) + 1]
        results.append((tokenizer.decode(tokens, skip_special_tokens=True).strip(), len(tokens)))
    if len(results) != len(prompts):
        raise ValueError("Generation returned a different number of outputs")
    return results, elapsed


def output_record(case: dict[str, Any], raw: str, tokens: int, seconds: float) -> dict[str, Any]:
    """Apply the same strict and one-outer-fence checks for every method."""
    strict = assess_output(raw, case["source_rows"], normalize=False)
    normalized = assess_output(raw, case["source_rows"], normalize=True)
    parsed = normalized["parsed"]
    valid = normalized["json_valid"] and not normalized["validation_errors"]
    original, unsupported_instruction = (inverse_intent(parsed, case["transform"], len(case["source_rows"]))
                                          if valid else (None, False))
    return {**case, "prediction_status": "complete", "raw_output": raw, "generated_tokens": tokens,
            "generation_seconds": seconds, "strict_json_valid": strict["json_valid"],
            "strict_valid_nontrivial": strict["json_valid"] and not strict["validation_errors"],
            "strict_errors": strict["validation_errors"],
            "json_valid": normalized["json_valid"], "valid_nontrivial": valid,
            "validation_errors": normalized["validation_errors"], "fence_removed": normalized["fence_removed"],
            "parsed": parsed, "inverse_intent": original, "instruction_unchanged": unsupported_instruction}


def contract_for(args: argparse.Namespace, plan: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    files = [args.pairs, args.model / "config.json", args.model / "model.safetensors", args.model / "tokenizer.json",
             args.model / "tokenizer_config.json", args.model / "experiment_revision.json", args.adapter / "adapter_config.json",
             args.adapter / "adapter_model.safetensors", args.support, args.support_data, Path(__file__),
             HISTORICAL / "adapter/best.json", HISTORICAL / "adapted_outputs.run.json",
             ROOT / "carl/icl/intent_generator_experiment.py", ROOT / "carl/icl/intent_in_context.py",
             ROOT / "carl/icl/intent_data.py", ROOT / "carl/icl/intent_fence_audit.py",
             ROOT / "carl/icl/policy_relative_lora.py",
             ROOT / "carl/icl/boards.py", ROOT / "carl/icl/data_sources/sokoban.py",
             ROOT / "carl/environment/sokoban/env.py", ROOT / "carl/environment/sokoban/core.py"]
    best = json.loads((HISTORICAL / "adapter/best.json").read_text())
    if args.adapter.resolve() != (ROOT / best["adapter"]).resolve() or best["step"] != 220:
        raise ValueError("Adapter differs from the checkpoint selected by historical validation")
    revision = json.loads((args.model / "experiment_revision.json").read_text())
    if revision.get("revision") != REVISION:
        raise ValueError("Cached model revision differs")
    model_metadata = args.model / ".cache/huggingface/download/model.safetensors.metadata"
    data_manifest = args.pairs.parent / "data_manifest.json"
    for optional in (model_metadata, data_manifest):
        if optional.is_file():
            files.append(optional)
    hashes = {str(path.resolve()): file_sha(path) for path in files}
    historical_run = json.loads((HISTORICAL / "adapted_outputs.run.json").read_text())
    for name, key in (("adapter_model.safetensors", "adapter_weights_sha256"),
                      ("adapter_config.json", "adapter_config_sha256")):
        if hashes[str((args.adapter / name).resolve())] != historical_run.get(key):
            raise ValueError(f"Adapter differs from the approved historical inference: {name}")
    model_provenance = "revision and local file hash only; historical model-content hash unavailable"
    if model_metadata.is_file():
        lines = model_metadata.read_text().splitlines()
        if len(lines) < 2 or lines[0] != REVISION or len(lines[1]) != 64:
            raise ValueError("Historical Qwen download metadata is malformed")
        if hashes[str((args.model / "model.safetensors").resolve())] != lines[1]:
            raise ValueError("Qwen weights differ from historical downloaded-content hash")
        model_provenance = "verified against historical local Hugging Face download metadata"
    pairs_provenance = "file hash recorded; no adjacent data manifest"
    if data_manifest.is_file():
        declared = json.loads(data_manifest.read_text()).get("outputs_sha256", {}).get(args.pairs.name)
        if declared is None or declared != hashes[str(args.pairs.resolve())]:
            raise ValueError("Stage-one pairs disagree with their data manifest")
        pairs_provenance = "verified against adjacent stage-one data manifest"
    return {"schema_version": 1, "plan_sha256": digest(plan), "prompts_sha256": digest(prompts),
            "files_sha256": hashes, "model_provenance": model_provenance, "pairs_provenance": pairs_provenance,
            "model_revision": REVISION, "adapter_selection": "historical validation, checkpoint-220",
            "cohort_id": args.cohort_id, "split": args.split, "per_family": args.per_family,
            "device": "cpu", "dtype": args.dtype, "threads": args.threads, "inter_op_threads": 1,
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")},
            "batch_size": args.batch_size, "seed": args.seed, "decode": "greedy", "enable_thinking": False,
            "max_length": args.max_length, "max_new_tokens": args.max_new_tokens,
            "input": "current board, task goal, and fixed ICL support only",
            "timing": "generation wall time allocated equally over outputs in each batch",
            "parser": "strict JSON plus symmetric removal of at most one complete outer JSON/plain fence"}


def verify_saved(record: dict[str, Any], case: dict[str, Any], contract_sha: str) -> None:
    if (any(record.get(key) != value for key, value in case.items())
            or record.get("contract_sha256") != contract_sha
            or record.get("prediction_status") not in {"complete", "infra_error"}):
        raise ValueError("Saved case identity or contract mismatch")
    if record["prediction_status"] == "complete":
        expected = output_record(case, record["raw_output"], record["generated_tokens"], record["generation_seconds"])
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Saved prediction differs from its raw output")
    elif record.get("raw_output") is not None or not isinstance(record.get("error"), str):
        raise ValueError("Saved infrastructure error has invalid fields")


def reconcile(output: Path, cases: list[dict[str, Any]], contract_sha: str) -> dict[str, dict[str, Any]]:
    """Recover the JSONL index from atomic per-case files and retain torn tails."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "predictions.jsonl"
    old_rows: list[dict[str, Any]] = []
    if path.exists():
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            boundary = raw.rfind(b"\n") + 1
            tail, raw = raw[boundary:], raw[:boundary]
            name = f"torn_tail_{hashlib.sha256(tail).hexdigest()[:16]}.bin"
            (output / name).write_bytes(tail)
        old_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        ids = [row.get("case_id") for row in old_rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Existing prediction index has duplicate case IDs")
    rows = []
    for case in cases:
        saved = output / "cases" / f"{case['case_id']}.json"
        if saved.exists():
            record = json.loads(saved.read_text())
            verify_saved(record, case, contract_sha)
            rows.append(record)
    indexed = {row["case_id"]: row for row in rows}
    if any(row.get("case_id") not in indexed or indexed[row["case_id"]] != row for row in old_rows):
        raise ValueError("Existing prediction index differs from its durable case files")
    atomic_jsonl(path, rows)
    return {row["case_id"]: row for row in rows}


def persist_frozen(output: Path, plan: dict[str, Any], prompts: list[dict[str, Any]],
                   contract: dict[str, Any], *, resume: bool) -> None:
    """Recover interrupted input writes only from the exactly matching contract."""
    cp, pp, prompt_path = (output / name for name in ("contract.json", "plan.json", "prompts.jsonl"))
    if contract["plan_sha256"] != digest(plan) or contract["prompts_sha256"] != digest(prompts):
        raise ValueError("Computed plan or prompts disagree with the contract")
    if cp.exists() or pp.exists() or prompt_path.exists():
        if not resume or not cp.is_file() or json.loads(cp.read_text()) != contract:
            raise ValueError("Existing output requires --resume with its exact frozen contract")
        if pp.exists() and json.loads(pp.read_text()) != plan:
            raise ValueError("Frozen evaluation plan differs")
        if prompt_path.exists() and read_jsonl(prompt_path) != prompts:
            raise ValueError("Frozen evaluation prompts differ")
        if not pp.exists():
            atomic_json(pp, plan)
        if not prompt_path.exists():
            atomic_jsonl(prompt_path, prompts)
    else:
        if resume:
            raise ValueError("No existing stage-three run to resume")
        atomic_json(cp, contract)
        atomic_json(pp, plan)
        atomic_jsonl(prompt_path, prompts)


def method_statistics(records: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Report batch time allocated evenly over its outputs, by method."""
    result = {}
    for method in METHODS:
        rows = [row for row in records.values() if row.get("method") == method
                and row.get("prediction_status") == "complete"]
        seconds = sum(row["generation_seconds"] for row in rows)
        tokens = sum(row["generated_tokens"] for row in rows)
        result[str(method)] = {"outputs": len(rows), "generated_tokens": tokens, "allocated_batch_seconds": seconds,
                               "seconds_per_output": seconds / len(rows) if rows else None,
                               "tokens_per_second": tokens / seconds if seconds else None}
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1 or args.threads > len(os.sched_getaffinity(0)):
        raise ValueError("CPU thread count exceeds affinity")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    from transformers import AutoTokenizer

    support_records = read_jsonl(args.support_data)
    support = load_support(args.support_data, args.support, support_records)
    contexts = select_contexts(read_jsonl(args.pairs), args.split, args.per_family,
                               {row["layout_id"] for row in support},
                               {row["layout_id"] for row in support_records if row.get("split") == "train"})
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    plan, prompts = build_plan(contexts, support, tokenizer, args.cohort_id, args.max_length, args.max_new_tokens)
    contract = contract_for(args, plan, prompts)
    contract_sha = digest(contract)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        persist_frozen(args.output, plan, prompts, contract, resume=args.resume)
        cases = plan["cases"]
        seen = reconcile(args.output, cases, contract_sha)
        initial = psutil.virtual_memory()
        initial_swap = psutil.swap_memory().used
        started = time.monotonic()
        completed = sum(row["prediction_status"] == "complete" for row in seen.values())
        errors = sum(row["prediction_status"] == "infra_error" for row in seen.values())
        base_status = {"schema_version": 1, "state": "running", "planned": len(cases), "completed": completed,
                       "infra_errors": errors, "missing": len(cases) - len(seen),
                       "cohort_id": args.cohort_id, "started_model_loading": True}
        atomic_json(args.output / "status.json", base_status)
        if len(seen) == len(cases):
            base_status.update(state="partial" if errors else "complete",
                               stop_reason="saved_infrastructure_error" if errors else None,
                               elapsed_seconds=0.0, method_statistics=method_statistics(seen))
            atomic_json(args.output / "status.json", base_status)
            return base_status
        torch, tokenizer, model = load_model(args)
        prompts_by_id = {row["case_id"]: row["rendered_prompt"] for row in prompts}
        lookup = {(row["context_id"], row["method"], row["transform"]): row for row in cases}
        stops = None
        peak_memory = (initial.total - initial.available) / initial.total
        peak_swap = initial_swap
        generated, batch_count, tokens_total, seconds_total = 0, 0, 0, 0.0
        with (args.output / "predictions.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
            for context in contexts:
                for method in METHODS:
                    pending = [lookup[(context["context_id"], str(method), str(transform))] for transform in TRANSFORMS
                               if lookup[(context["context_id"], str(method), str(transform))]["case_id"] not in seen]
                    if not pending:
                        continue
                    for offset in range(0, len(pending), args.batch_size):
                        batch = pending[offset:offset + args.batch_size]
                        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
                        fraction = (memory.total - memory.available) / memory.total
                        peak_memory, peak_swap = max(peak_memory, fraction), max(peak_swap, swap.used)
                        if fraction >= .80:
                            stops = "host_memory_limit"
                        elif swap.used - initial_swap >= 128 * 1024 * 1024:
                            stops = "swap_growth_limit"
                        elif args.stop_epoch is not None and time.time() >= args.stop_epoch:
                            stops = "stage_epoch_cutoff"
                        elif args.stop_monotonic is not None and time.monotonic() >= args.stop_monotonic:
                            stops = "stage_monotonic_cutoff"
                        if stops:
                            break
                        batch_prompts = [prompts_by_id[row["case_id"]] for row in batch]
                        try:
                            values, elapsed = model_batch(torch, tokenizer, model, batch_prompts,
                                                          adapted=method == Method.ADAPTED,
                                                          max_new_tokens=args.max_new_tokens, max_length=args.max_length)
                            entries = [output_record(case, raw, token_count, elapsed / len(batch))
                                       for case, (raw, token_count) in zip(batch, values, strict=True)]
                            seconds_total += elapsed
                            tokens_total += sum(token_count for _, token_count in values)
                        except Exception as error:
                            entries = [{**case, "prediction_status": "infra_error", "raw_output": None,
                                        "error": f"{type(error).__name__}: {error}"} for case in batch]
                            stops = "inference_error"
                        batch_count += 1
                        for entry in entries:
                            entry["contract_sha256"] = contract_sha
                            path = args.output / "cases" / f"{entry['case_id']}.json"
                            atomic_json(path, entry)
                            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                            seen[entry["case_id"]] = entry
                            generated += 1
                        completed = sum(row["prediction_status"] == "complete" for row in seen.values())
                        errors = sum(row["prediction_status"] == "infra_error" for row in seen.values())
                        atomic_json(args.output / "status.json", {**base_status, "completed": completed,
                                    "infra_errors": errors, "missing": len(cases) - len(seen),
                                    "new_outputs": generated, "elapsed_seconds": time.monotonic() - started,
                                    "batch_count": batch_count, "generated_tokens": tokens_total,
                                    "generation_seconds": seconds_total, "method_statistics": method_statistics(seen),
                                    "generation_tokens_per_second": tokens_total / seconds_total if seconds_total else None,
                                    "peak_host_memory_fraction": peak_memory,
                                    "peak_swap_growth_bytes": max(0, peak_swap - initial_swap)})
                        if stops:
                            break
                    if stops:
                        break
                if stops:
                    break
        completed = sum(row["prediction_status"] == "complete" for row in seen.values())
        errors = sum(row["prediction_status"] == "infra_error" for row in seen.values())
        result = {**base_status, "state": "complete" if completed == len(cases) and not stops else "partial",
                  "completed": completed, "infra_errors": errors, "missing": len(cases) - len(seen),
                  "stop_reason": stops or ("saved_infrastructure_error" if errors else None),
                  "elapsed_seconds": time.monotonic() - started, "new_outputs": generated,
                  "batch_count": batch_count, "generated_tokens": tokens_total,
                  "generation_seconds": seconds_total,
                  "method_statistics": method_statistics(seen),
                  "generation_tokens_per_second": tokens_total / seconds_total if seconds_total else None,
                  "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "peak_host_memory_fraction": peak_memory,
                  "peak_swap_growth_bytes": max(0, peak_swap - initial_swap)}
        atomic_json(args.output / "status.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--per-family", type=int, required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--support", type=Path, default=SUPPORT)
    parser.add_argument("--support-data", type=Path, default=SUPPORT_DATA)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, choices=(1, 3), default=3)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--stop-epoch", type=float)
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    status = execute(args)
    print(json.dumps(status), flush=True)
    if status["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
