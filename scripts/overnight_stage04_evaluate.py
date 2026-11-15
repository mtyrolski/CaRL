#!/usr/bin/env python3
"""Evaluate matched stage-four LoRA arms on the frozen stage-three boards.

The base and historical adapter responses are copied only from the identical
saved stage-three CPU/FP32 protocol. New adapters share its prompts, decoding,
three orientations and original-board split. Every completed case is durable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import psutil
from filelock import FileLock

from carl.icl.intent_generator_experiment import prompt_messages
from scripts.overnight_stage03_generate import ADAPTER
from scripts.overnight_stage03_generate import MODEL
from scripts.overnight_stage03_generate import REVISION
from scripts.overnight_stage03_generate import TRANSFORMS
from scripts.overnight_stage03_generate import case_id
from scripts.overnight_stage03_generate import load_model
from scripts.overnight_stage03_generate import model_batch
from scripts.overnight_stage03_generate import output_record
from scripts.overnight_stage03_generate import persist_frozen
from scripts.overnight_stage03_generate import reconcile
from scripts.overnight_stage03_generate import transform_rows
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest
from scripts.run_overnight_cpu_policy import file_sha
from scripts.run_overnight_cpu_policy import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {"base": "base", "historical_checkpoint220": "checkpoint220"}
NEW_ARMS = ("wide", "wide_aug", "narrow")
COHORT = "stage04_matched_evaluation"
TASK_GOAL = "Put every box on a goal."
MATCHED_TRAINING_KEYS = ("model", "revision", "dtype", "threads", "seed", "rank", "accumulation", "lr", "max_length",
                         "max_new_tokens", "gradient_checkpointing", "pad_to_length", "max_steps", "eval_every",
                         "validation_boards", "validation_contexts_per_board")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else ROOT / value).resolve()


def select_contexts(contexts: list[dict[str, Any]], per_family: int) -> list[dict[str, Any]]:
    """Select a fixed original-board prefix, balanced across x10 sources."""
    if per_family < 2 or per_family % 2:
        raise ValueError("per-family must be a positive even number of at least two")
    if len({row["layout_id"] for row in contexts}) != len(contexts):
        raise ValueError("Stage-three evaluation contains repeated canonical layouts")
    if any(row["split"] != "evaluation" for row in contexts):
        raise ValueError("Only the frozen evaluation split is permitted")
    x4 = [row for row in contexts if row["family"] == "12x12x4"][:per_family]
    x10 = [row for row in contexts if row["family"] == "12x12x10"]
    sources = sorted({row["source_family"] for row in x10})
    if len(sources) != 2:
        raise ValueError("Expected two frozen x10 source distributions")
    groups = [[row for row in x10 if row["source_family"] == source][:per_family // 2] for source in sources]
    if len(x4) != per_family or any(len(group) != per_family // 2 for group in groups):
        raise ValueError("Not enough frozen boards for the requested paired evaluation")
    chosen = {row["context_id"] for row in x4 + [item for group in groups for item in group]}
    return [row for row in contexts if row["context_id"] in chosen]


def check_splits(contexts: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    splits = manifest["splits"]
    training = set(splits["train_layout_ids"])
    validation = set(splits["validation_layout_ids"])
    evaluation = set(splits["evaluation_layout_ids"])
    if training & validation or training & evaluation or validation & evaluation:
        raise ValueError("Training, validation and evaluation layouts are not disjoint")
    selected = {row["layout_id"] for row in contexts}
    if selected & (training | validation) or not selected <= evaluation:
        raise ValueError("Frozen evaluation layouts are missing or overlap training/validation")


def check_freeze(contexts: list[dict[str, Any]], freeze: dict[str, Any], per_family: int) -> list[str]:
    """Reject any deviation from the evaluation cohort fixed before training."""
    keys = ("context_id", "layout_id", "family", "source_family", "board_rank", "adapter_training_exposure")
    selected = [{key: row[key] for key in keys} for row in contexts]
    frozen = [{key: row[key] for key in keys} for row in freeze["contexts"]]
    if (selected != frozen or freeze["boards_per_family"] != per_family
            or freeze["canonical_original_boards"] != len(selected)
            or freeze["model_revision"] != REVISION or freeze["transforms"] != list(TRANSFORMS)):
        raise ValueError("Selected boards, transforms or model differ from the frozen stage-four evaluation")
    expected = {"device": "cpu", "dtype": "float32", "threads": 4, "batch_size": 3,
                "decode": "greedy", "enable_thinking": False, "max_length": 1024, "max_new_tokens": 128}
    if freeze["inference"] != expected:
        raise ValueError("Frozen inference parameters do not match the reusable reference")
    methods = freeze["planned_methods"]
    if len(methods) != len(set(methods)) or set(methods) != set(BASELINE) | set(NEW_ARMS):
        raise ValueError("Frozen method list does not contain the intended arms")
    return methods


def parse_arm(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in NEW_ARMS or not path:
        raise ValueError("Use --adapter wide=DIR, wide_aug=DIR or narrow=DIR")
    return name, project_path(path)


def selected_adapters(values: list[str], manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load completed, validation-selected adapters and verify matched budgets."""
    arms = dict(parse_arm(value) for value in values)
    if len(arms) != len(values):
        raise ValueError("An adapter arm was repeated")
    selected: dict[str, dict[str, Any]] = {}
    reference = None
    validation_ids = None
    training_counts = set()
    training_curves: dict[str, list[dict[str, Any]]] = {}
    for name, directory in arms.items():
        best_path, config_path, status_path = (directory / key for key in
                                               ("best.json", "configuration.json", "status.json"))
        best, config, status = (read_json(path) for path in (best_path, config_path, status_path))
        args = config["args"]
        checkpoint = project_path(best["adapter"])
        if (status.get("status") != "finished" or status.get("completed_steps") != args["max_steps"]
                or project_path(status["best_adapter"]) != checkpoint or best["step"] < 1
                or not isinstance(best.get("validation_loss"), (int, float))
                or not math.isfinite(best["validation_loss"]) or checkpoint.parent != directory
                or checkpoint.name != f"checkpoint-{best['step']}"):
            raise ValueError(f"{name} lacks a completed validation-selected adapter")
        if args["revision"] != REVISION or project_path(args["model"]) != MODEL.resolve():
            raise ValueError(f"{name} was trained from a different base model")
        adapter_config = read_json(checkpoint / "adapter_config.json")
        expected_lora = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 4,
                         "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none"}
        if (any(adapter_config.get(key) != value for key, value in expected_lora.items())
                or set(adapter_config.get("target_modules") or []) != {"q_proj", "v_proj"}
                or adapter_config.get("rank_pattern") or adapter_config.get("alpha_pattern")
                or adapter_config.get("modules_to_save") or adapter_config.get("use_dora")
                or adapter_config.get("use_rslora")
                or project_path(adapter_config["base_model_name_or_path"]) != MODEL.resolve()):
            raise ValueError(f"{name} LoRA configuration differs from the frozen rank-four setup")
        training_file = project_path(args["data"])
        dataset_name = "dataset_narrow.jsonl" if name == "narrow" else "dataset_wide.jsonl"
        if (file_sha(training_file) != config["data_sha256"]
                or config["data_sha256"] != manifest["output_sha256"][dataset_name]):
            raise ValueError(f"{name} training data hash disagrees with its configuration")
        expected_layouts = set(manifest["arms"]["wide" if name == "wide_aug" else name]["train_layout_ids"])
        if not expected_layouts <= set(manifest["splits"]["train_layout_ids"]):
            raise ValueError(f"{name} training layout is outside the frozen training split")
        records = read_jsonl(training_file)
        actual_layouts = {row["layout_id"] for row in records if row["split"] == "train"}
        actual_validation = {row["layout_id"] for row in records if row["split"] == "validation"}
        valid_record_ids = {row["record_id"] for row in records if row["split"] == "validation"}
        training_counts.add(sum(row["split"] == "train" for row in records))
        if (actual_layouts != expected_layouts
                or not actual_validation <= set(manifest["splits"]["validation_layout_ids"])
                or not actual_validation or not config["validation_record_ids"]
                or not set(config["validation_record_ids"]) <= valid_record_ids):
            raise ValueError(f"{name} training/validation records disagree with the frozen split")
        augmentation = args.get("augmentation")
        if augmentation != ("d4" if name == "wide_aug" else "none"):
            raise ValueError(f"{name} used an unexpected coordinate augmentation")
        curve_path = directory / "training_curve.jsonl"
        curve = read_jsonl(curve_path)
        if len(curve) != args["max_steps"] or any(row["step"] != index for index, row in enumerate(curve, 1)):
            raise ValueError(f"{name} training curve does not contain every completed update")
        validations = [(row["validation_loss"], row["step"]) for row in curve if "validation_loss" in row]
        if (not validations or min(validations) != (best["validation_loss"], best["step"])):
            raise ValueError(f"{name} adapter is not the validation-loss-selected checkpoint")
        training_curves[name] = curve
        matched = {key: args.get(key) for key in MATCHED_TRAINING_KEYS}
        if reference is not None and matched != reference:
            raise ValueError("New arms have unmatched training budgets or initialisation")
        if validation_ids is not None and config["validation_record_ids"] != validation_ids:
            raise ValueError("New arms used different validation records")
        reference, validation_ids = matched, config["validation_record_ids"]
        selected[name] = {"arm_directory": str(directory), "checkpoint": str(checkpoint),
                          "best": best, "completed_steps": status["completed_steps"],
                          "training_data": str(training_file), "training_data_sha256": config["data_sha256"],
                          "checkpoint_weights_sha256": file_sha(checkpoint / "adapter_model.safetensors"),
                          "checkpoint_config_sha256": file_sha(checkpoint / "adapter_config.json"),
                          "best_sha256": file_sha(best_path), "configuration_sha256": file_sha(config_path),
                          "status_sha256": file_sha(status_path), "training_curve_sha256": file_sha(curve_path),
                          "model_tokens": sum(row["model_tokens"] for row in curve),
                          "effective_tokens": sum(row["effective_tokens"] for row in curve),
                          "supervised_tokens": sum(row["supervised_tokens"] for row in curve),
                          "symmetry_counts": dict(Counter(sample["symmetry"] for row in curve
                                                          for sample in row["samples"])),
                          "matched_training": matched,
                          "validation_record_ids": validation_ids}
    if len(training_counts) > 1:
        raise ValueError("New arms use different numbers of original training examples")
    if "wide" in selected and "wide_aug" in selected:
        if selected["wide"]["training_data_sha256"] != selected["wide_aug"]["training_data_sha256"]:
            raise ValueError("Wide and augmented wide arms must share original training examples")
        for plain, augmented in zip(training_curves["wide"], training_curves["wide_aug"], strict=True):
            plain_ids = [(item["record_id"], item["layout_id"]) for item in plain["samples"]]
            augmented_ids = [(item["record_id"], item["layout_id"]) for item in augmented["samples"]]
            if plain_ids != augmented_ids:
                raise ValueError("Wide and augmented wide arms sampled different original records")
    if len({row["model_tokens"] for row in selected.values()}) > 1:
        raise ValueError("New arms have different model token budgets")
    return selected


def verify_reference(contract: dict[str, Any], plan: dict[str, Any], prompts: list[dict[str, Any]]) -> None:
    expected = {"model_revision": REVISION, "device": "cpu", "dtype": "float32", "threads": 4,
                "batch_size": 3, "seed": 1729, "decode": "greedy", "enable_thinking": False,
                "max_length": 1024, "max_new_tokens": 128}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Historical baselines use a different model or inference protocol")
    if digest(plan) != contract["plan_sha256"] or digest(prompts) != contract["prompts_sha256"]:
        raise ValueError("Historical plan or prompts differ from their frozen contract")
    for package, expected_version in contract["packages"].items():
        if importlib.metadata.version(package) != expected_version:
            raise ValueError(f"Historical baseline package version differs: {package}")
    for path in (MODEL / "model.safetensors", MODEL / "tokenizer.json", ADAPTER / "adapter_model.safetensors"):
        if contract["files_sha256"].get(str(path.resolve())) != file_sha(path):
            raise ValueError(f"Historical model/tokenizer/adapter changed: {path}")


def build_plan(reference: dict[str, Any], original_prompts: list[dict[str, Any]],
               contexts: list[dict[str, Any]], methods: list[str], tokenizer: Any
               ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_case = {case["case_id"]: case for case in reference["cases"]}
    by_key = {(case["context_id"], case["method"], case["transform"]): case for case in reference["cases"]}
    prompt_map = {prompt["case_id"]: prompt for prompt in original_prompts}
    if len(by_case) != len(reference["cases"]) or len(prompt_map) != len(original_prompts):
        raise ValueError("Reference cases or prompts have duplicate IDs")
    cases, prompts = [], []
    for context in contexts:
        for transform in TRANSFORMS:
            original = by_key[(context["context_id"], "base", str(transform))]
            prior = prompt_map[original["case_id"]]
            record = {"start_observation": transform_rows(context["original_rows"], transform), "task_goal": TASK_GOAL}
            if (original["source_rows"] != record["start_observation"]
                    or prior["messages"] != prompt_messages(record)
                    or hashlib.sha256(prior["rendered_prompt"].encode()).hexdigest() != prior["prompt_sha256"]):
                raise ValueError("Reference prompt is not the exact current-board/goal serializer")
            options = {"add_generation_prompt": True, "enable_thinking": False}
            rendered = tokenizer.apply_chat_template(prior["messages"], tokenize=False, **options)
            token_ids = tokenizer.apply_chat_template(prior["messages"], tokenize=True, **options)
            rendered_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            if (rendered != prior["rendered_prompt"] or token_ids != rendered_ids
                    or len(token_ids) != original["prompt_tokens"] or len(token_ids) + 128 > 1024):
                raise ValueError("Reconstructed prompt or token IDs differ from the historical baseline")
            old = by_key[(context["context_id"], "checkpoint220", str(transform))]
            if prompt_map[old["case_id"]]["rendered_prompt"] != prior["rendered_prompt"]:
                raise ValueError("Reference base and historical adapter prompts differ")
            for method in methods:
                identifier = case_id(COHORT, context["context_id"], method, transform)
                case = {**original, "case_id": identifier, "cohort_id": COHORT, "method": method}
                cases.append(case)
                prompts.append({**prior, "case_id": identifier})
    manifest = [{**context, "cohort_id": COHORT, "expected_methods": methods,
                 "expected_transforms": list(TRANSFORMS)} for context in contexts]
    return {"schema_version": 1, "cohort_id": COHORT, "contexts": len(contexts),
            "family_contexts": dict(Counter(row["family"] for row in contexts)),
            "source_contexts": dict(Counter(row["source_family"] for row in contexts)),
            "context_manifest": manifest, "cases": cases,
            "selection": "Fixed pre-existing evaluation prefix, balanced across x10 sources; never selected by output."}, prompts


def cached_baselines(reference: list[dict[str, Any]], cases: list[dict[str, Any]],
                     contract_sha: str) -> list[dict[str, Any]]:
    indexed = {(row["context_id"], row["method"], row["transform"]): row for row in reference}
    if len(indexed) != len(reference):
        raise ValueError("Reference predictions contain duplicate cases")
    result = []
    for case in cases:
        if case["method"] not in BASELINE:
            continue
        prior = indexed[(case["context_id"], BASELINE[case["method"]], case["transform"])]
        if (prior.get("prediction_status") != "complete" or prior["source_rows"] != case["source_rows"]
                or prior["prompt_sha256"] != case["prompt_sha256"]):
            raise ValueError("Historical baseline is missing or its current input changed")
        row = output_record(case, prior["raw_output"], prior["generated_tokens"], 0.0)
        row.update({"contract_sha256": contract_sha, "baseline_reused": True,
                    "historical_case_id": prior["case_id"],
                    "historical_generation_seconds": prior["generation_seconds"]})
        result.append(row)
    return result


def persist_record(output: Path, row: dict[str, Any], stream: Any) -> None:
    path = output / "cases" / f"{row['case_id']}.json"
    atomic_json(path, row)
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def reconciled_status(cases: list[dict[str, Any]], seen: dict[str, dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    """Describe recovered durable outputs without treating missing slots as model failures."""
    missing = len(cases) - len(seen)
    if missing < 0:
        raise ValueError("Recovered more cases than were planned")
    return {"state": "complete" if missing == 0 else "partial", "planned": len(cases),
            "completed": len(seen), "missing": missing,
            "baseline_reused": sum(row.get("baseline_reused") is True for row in seen.values()),
            "stop_reason": "reconciled_after_interruption", "methods": methods}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4"
    reference_plan = read_json(args.reference / "plan.json")
    reference_contract = read_json(args.reference / "contract.json")
    reference_prompts = read_jsonl(args.reference / "prompts.jsonl")
    verify_reference(reference_contract, reference_plan, reference_prompts)
    data_manifest = read_json(args.data_manifest)
    contexts = select_contexts(reference_plan["context_manifest"], args.per_family)
    freeze = read_json(args.freeze)
    methods = check_freeze(contexts, freeze, args.per_family)
    check_splits(contexts, data_manifest)
    adapters = selected_adapters(args.adapter, data_manifest)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side="left")
    plan, prompts = build_plan(reference_plan, reference_prompts, contexts, methods, tokenizer)
    reference_predictions_path = args.reference / "predictions.jsonl"
    reference_assessment = read_json(args.reference.parent / "assessment_fresh/assessment_manifest.json")
    expected_reference_sha = reference_assessment["inputs_sha256"].get(str(reference_predictions_path.resolve()))
    if expected_reference_sha != file_sha(reference_predictions_path):
        raise ValueError("Historical predictions differ from the completed stage-three assessment")
    contract = {"schema_version": 1, "plan_sha256": digest(plan), "prompts_sha256": digest(prompts),
                "reference_contract_sha256": file_sha(args.reference / "contract.json"),
                "reference_predictions_sha256": file_sha(reference_predictions_path),
                "data_manifest_sha256": file_sha(args.data_manifest), "freeze_sha256": file_sha(args.freeze),
                "adapters": adapters,
                "source_sha256": file_sha(Path(__file__)), "model_revision": REVISION,
                "device": "cpu", "dtype": "float32", "threads": 4, "batch_size": 3, "seed": 1729,
                "decode": "greedy", "enable_thinking": False, "max_length": 1024, "max_new_tokens": 128,
                "input": "current board and task goal only; historical baselines reuse exact cached responses",
                "limitations": "stage-three held-out layouts were previously inspected; x10 sources differ from x4"}
    contract_sha = digest(contract)
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        persist_frozen(args.output, plan, prompts, contract, resume=args.resume)
        seen = reconcile(args.output, plan["cases"], contract_sha)
        if args.reconcile_only:
            if not args.resume:
                raise ValueError("--reconcile-only requires --resume")
            status = reconciled_status(plan["cases"], seen, methods)
            atomic_json(args.output / "status.json", status)
            return status
        # No test output is read until the complete board-only plan is durable.
        cached = cached_baselines(read_jsonl(reference_predictions_path), plan["cases"], contract_sha)
        started, baseline_swap = time.monotonic(), psutil.swap_memory().used
        (args.output / "cases").mkdir(exist_ok=True)
        atomic_json(args.output / "status.json", {"state": "running", "planned": len(plan["cases"]),
                    "completed": len(seen), "elapsed_seconds": 0})
        try:
            with (args.output / "predictions.jsonl").open("a", encoding="utf-8", buffering=1) as stream:
                for record in cached:
                    if record["case_id"] not in seen:
                        persist_record(args.output, record, stream)
                        seen[record["case_id"]] = record
                atomic_json(args.output / "status.json", {"state": "running", "planned": len(plan["cases"]),
                            "completed": len(seen), "baseline_reused": len(cached),
                            "elapsed_seconds": time.monotonic() - started})
                if adapters:
                    model_args = argparse.Namespace(threads=4, seed=1729, model=MODEL, adapter=ADAPTER, dtype="float32")
                    torch, tokenizer, model = load_model(model_args)
                    for arm, metadata in adapters.items():
                        model.load_adapter(metadata["checkpoint"], adapter_name=arm, is_trainable=False)
                    model.eval().requires_grad_(False)
                    prompt_by_id = {row["case_id"]: row["rendered_prompt"] for row in prompts}
                    for arm in NEW_ARMS:
                        if arm not in adapters:
                            continue
                        model.set_adapter(arm)
                        model.eval().requires_grad_(False)
                        pending = [case for case in plan["cases"] if case["method"] == arm and case["case_id"] not in seen]
                        for offset in range(0, len(pending), 3):
                            batch = pending[offset:offset + 3]
                            if ((args.stop_epoch is not None and time.time() >= args.stop_epoch)
                                    or (args.stop_monotonic is not None and time.monotonic() >= args.stop_monotonic)):
                                break
                            if psutil.virtual_memory().percent >= 80 or psutil.swap_memory().used - baseline_swap >= 128 * 1024 * 1024:
                                raise RuntimeError("Memory/swap guard reached; durable completed cases retained")
                            values, elapsed = model_batch(torch, tokenizer, model,
                                [prompt_by_id[case["case_id"]] for case in batch], adapted=True,
                                max_new_tokens=128, max_length=1024)
                            for case, (raw, tokens) in zip(batch, values, strict=True):
                                row = output_record(case, raw, tokens, elapsed / len(batch))
                                row.update({"contract_sha256": contract_sha, "baseline_reused": False})
                                persist_record(args.output, row, stream)
                                seen[row["case_id"]] = row
                            atomic_json(args.output / "status.json", {"state": "running", "planned": len(plan["cases"]),
                                        "completed": len(seen), "elapsed_seconds": time.monotonic() - started})
        except Exception as error:
            atomic_json(args.output / "status.json", {"state": "failed", "planned": len(plan["cases"]),
                        "completed": len(seen), "missing": len(plan["cases"]) - len(seen),
                        "error": f"{type(error).__name__}: {error}", "elapsed_seconds": time.monotonic() - started})
            raise
        missing = len(plan["cases"]) - len(seen)
        status = {"state": "complete" if missing == 0 else "partial", "planned": len(plan["cases"]),
                  "completed": len(seen), "missing": missing, "baseline_reused": len(cached),
                  "elapsed_seconds": time.monotonic() - started, "methods": methods}
        atomic_json(args.output / "status.json", status)
        return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="completed stage-three fresh_evaluation")
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family", type=int, required=True)
    parser.add_argument("--adapter", action="append", default=[], help="arm=completed-training-directory")
    parser.add_argument("--stop-epoch", type=float)
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reconcile-only", action="store_true", help="recover durable cases without loading weights")
    args = parser.parse_args()
    status = execute(args)
    print(json.dumps(status), flush=True)
    if status["state"] != "complete" and not args.reconcile_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
