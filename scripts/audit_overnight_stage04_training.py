#!/usr/bin/env python3
"""Independently audit the frozen stage-four LoRA updates and sampling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SYMMETRIES = ("identity", "rotate_90", "rotate_180", "rotate_270", "reflect", "reflect_rotate_90",
              "reflect_rotate_180", "reflect_rotate_270")
ARMS = ("wide", "wide_aug", "narrow")
MATCHED_ARGUMENTS = ("model", "revision", "dtype", "threads", "seed", "rank", "accumulation", "lr",
                     "max_length", "max_new_tokens", "gradient_checkpointing", "pad_to_length", "max_steps",
                     "eval_every", "validation_boards", "validation_contexts_per_board")


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(obj, target, indent=2, sort_keys=True, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def independent_transform(record: dict[str, Any], symmetry: str) -> tuple[list[str], dict[str, Any]]:
    """Implement reflection then rotation independently of the training helper."""
    if symmetry not in SYMMETRIES or record["intent"].get("instruction") is not None:
        raise ValueError("unknown symmetry or unsafe free-text instruction")
    board = list(record["start_observation"])
    size = len(board)
    if not size or any(len(row) != size for row in board):
        raise ValueError("expected a square board")
    intent = copy.deepcopy(record["intent"])
    predicates = intent["effects"] + intent["preserve"]
    index = SYMMETRIES.index(symmetry)
    if index >= 4:
        board = [row[::-1] for row in board]
        for predicate in predicates:
            if predicate["predicate"] in ("box_at", "player_at"):
                predicate["col"] = size - 1 - predicate["col"]
    for _ in range(index % 4):
        board = ["".join(board[c][size - 1 - r] for c in range(size)) for r in range(size)]
        for predicate in predicates:
            if predicate["predicate"] in ("box_at", "player_at"):
                predicate["row"], predicate["col"] = size - 1 - predicate["col"], predicate["row"]
    return board, intent


def recompute_tokens(tokenizer: Any, system: str, record: dict[str, Any], symmetry: str,
                     max_length: int) -> tuple[int, int]:
    board, intent = independent_transform(record, symmetry)
    if record["task_goal"] != "Put every box on a goal.":
        raise ValueError("unsupported goal for geometric augmentation")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Task: " + record["task_goal"] + "\nCurrent board:\n" + "\n".join(board)}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    answer = tokenizer.encode(json.dumps(intent, ensure_ascii=False, separators=(",", ":")) + (tokenizer.eos_token or ""),
                              add_special_tokens=False)
    if len(prefix) + len(answer) > max_length:
        raise ValueError("a reconstructed input exceeds the configured maximum")
    return len(prefix) + len(answer), len(answer)


def distribution(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"count": len(ordered), "total": sum(ordered), "minimum": ordered[0] if ordered else None,
            "median": ordered[len(ordered) // 2] if ordered else None,
            "p95": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))] if ordered else None,
            "maximum": ordered[-1] if ordered else None}


def audit_arm(name: str, directory: Path, freeze: dict[str, Any], tokenizer: Any, system: str) -> dict[str, Any]:
    config_path, status_path, best_path, curve_path = (directory / filename for filename in
                                                        ("configuration.json", "status.json", "best.json",
                                                         "training_curve.jsonl"))
    config, status, best, curve = read_json(config_path), read_json(status_path), read_json(best_path), read_jsonl(curve_path)
    args = config["args"]
    expected = {"seed": "seed", "rank": "rank", "accumulation": "accumulation", "lr": "learning_rate",
                "max_length": "max_length", "pad_to_length": "padding_length", "max_steps": "max_steps",
                "eval_every": "eval_every", "dtype": "dtype", "gradient_checkpointing": "gradient_checkpointing",
                "revision": "revision", "validation_boards": "validation_boards",
                "validation_contexts_per_board": "validation_contexts_per_board",
                "stop_monotonic": "training_stop_monotonic"}
    errors = [f"configuration mismatch: {key}" for key, frozen_key in expected.items() if args.get(key) != freeze[frozen_key]]
    if args.get("augmentation") != ("d4" if name == "wide_aug" else "none"):
        errors.append("augmentation mode differs from frozen arm")
    if config.get("device") != freeze["device"] or config.get("enable_thinking") is not False:
        errors.append("device or thinking mode differs from frozen plan")
    if status.get("status") != "finished" or status.get("completed_steps") != freeze["max_steps"]:
        errors.append("training did not complete all frozen updates")
    if len(curve) != freeze["max_steps"] or [row.get("step") for row in curve] != list(range(1, freeze["max_steps"] + 1)):
        errors.append("training curve does not contain exactly the frozen updates")
    dataset = resolve(args["data"])
    if dataset != resolve(freeze["datasets"][name]) or sha(dataset) != config["data_sha256"]:
        errors.append("training dataset path/hash differs")
    if sha(dataset) != freeze["data_sha256"]["wide" if name == "wide_aug" else name]:
        errors.append("training dataset differs from frozen hash")
    rows = read_jsonl(dataset)
    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]
    if len(train) + len(validation) != len(rows):
        errors.append("training dataset contains a non-training/non-validation split")
    lookup = {row["record_id"]: row for row in train}
    if len(lookup) != len(train):
        errors.append("training record IDs are not unique")
    if {row["layout_id"] for row in train} & {row["layout_id"] for row in validation}:
        errors.append("training and validation layouts overlap")
    validation_ids = [row["record_id"] for row in validation]
    if (len(set(validation_ids)) != len(validation_ids)
            or len(config["validation_record_ids"]) != len(validation_ids)
            or set(config["validation_record_ids"]) != set(validation_ids)
            or set(config["validation_layouts"]) != {row["layout_id"] for row in validation}):
        errors.append("selected validation records/layouts differ from the shared held-out data")
    assessments = [(row["validation_loss"], row["step"]) for row in curve if "validation_loss" in row]
    expected_validation_steps = [step for step in range(1, freeze["max_steps"] + 1)
                                 if step % freeze["eval_every"] == 0 or step == freeze["max_steps"]]
    if [step for _, step in assessments] != expected_validation_steps:
        errors.append("validation measurements do not match the frozen cadence")
    if (not assessments or any(isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss)
                               for loss, _ in assessments)):
        errors.append("validation curve has no finite checkpoint measurements")
    elif min(assessments) != (best.get("validation_loss"), best.get("step")):
        errors.append("selected checkpoint is not the first validation argmin")
    checkpoint = resolve(best["adapter"])
    if checkpoint.parent != directory.resolve() or resolve(status["best_adapter"]) != checkpoint:
        errors.append("best checkpoint path is outside its arm or differs from terminal status")
    adapter_config = read_json(checkpoint / "adapter_config.json")
    if (adapter_config.get("r") != freeze["rank"] or adapter_config.get("lora_alpha") != freeze["lora_alpha"]
            or adapter_config.get("lora_dropout") != freeze["lora_dropout"]
            or set(adapter_config.get("target_modules", [])) != set(freeze["target_modules"])
            or resolve(adapter_config["base_model_name_or_path"]) != resolve(args["model"])):
        errors.append("saved adapter configuration differs from frozen LoRA/base")
    metadata = Counter()
    effect_counts, conjunctions, goal_thresholds, coordinates = Counter(), Counter(), Counter(), Counter()
    samples: list[tuple[str, str]] = []
    actual_tokens, response_tokens, model_tokens = [], [], []
    rng = random.Random(freeze["seed"] ^ 0x5D4A7A)
    for row in curve:
        observed = row.get("samples", [])
        if len(observed) != freeze["accumulation"]:
            errors.append(f"update {row.get('step')} has an incorrect microbatch count")
            continue
        effective = supervised = 0
        for sample in observed:
            identifier, layout, symmetry = sample["record_id"], sample["layout_id"], sample["symmetry"]
            original = lookup.get(identifier)
            if original is None or original["layout_id"] != layout:
                errors.append(f"update {row['step']} references an unknown train record/layout")
                continue
            expected_symmetry = rng.choice(SYMMETRIES) if name == "wide_aug" else "identity"
            if symmetry != expected_symmetry:
                errors.append(f"update {row['step']} has an unexpected symmetry RNG draw")
            try:
                length, output_length = recompute_tokens(tokenizer, system, original, symmetry, freeze["max_length"])
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"update {row['step']} cannot reconstruct a training input: {error}")
                continue
            if length > freeze["padding_length"]:
                errors.append(f"update {row['step']} would require truncation")
            effective += length
            supervised += output_length
            metadata[original.get("effect_type", "unknown")] += 1
            effects = original["intent"]["effects"]
            conjunctions[str(len(effects))] += 1
            for item in effects:
                predicate = item["predicate"]
                effect_counts[predicate] += 1
                if predicate == "boxes_on_goals_at_least":
                    goal_thresholds[str(item["count"])] += 1
            _, transformed_intent = independent_transform(original, symmetry)
            for item in transformed_intent["effects"]:
                if item["predicate"] in ("box_at", "player_at"):
                    coordinates[f"{item['predicate']}({item['row']},{item['col']})"] += 1
            samples.append((identifier, layout))
        if row.get("effective_tokens") != effective or row.get("supervised_tokens") != supervised:
            errors.append(f"update {row['step']} token counts differ from independently reconstructed inputs")
        if row.get("model_tokens") != freeze["model_tokens_per_step"]:
            errors.append(f"update {row['step']} has a different padded model token budget")
        actual_tokens.append(effective)
        response_tokens.append(supervised)
        model_tokens.append(row.get("model_tokens", 0))
    source_ids = {row["record_id"] for row in train}
    sampled_ids = {identifier for identifier, _ in samples}
    checkpoint_weights = checkpoint / "adapter_model.safetensors"
    return {"status": "pass" if not errors else "failed", "errors": errors, "arm": name,
            "completed_steps": status.get("completed_steps"), "samples": len(samples),
            "matched_arguments": {key: args.get(key) for key in MATCHED_ARGUMENTS},
            "original_train_records": len(train), "original_train_layouts": len({row['layout_id'] for row in train}),
            "sampled_unique_record_ids": len(sampled_ids), "sampled_unique_layouts": len({layout for _, layout in samples}),
            "source_record_ids": sorted(source_ids), "sampled_record_ids": sorted(sampled_ids),
            "sample_sequence": samples, "training_data_sha256": sha(dataset),
            "validation_record_ids": config["validation_record_ids"], "validation_layouts": config["validation_layouts"],
            "sampled_effect_types": dict(sorted(metadata.items())), "sampled_effect_predicates": dict(sorted(effect_counts.items())),
            "sampled_conjunction_sizes": dict(sorted(conjunctions.items())),
            "sampled_goal_thresholds": dict(sorted(goal_thresholds.items())),
            "sampled_coordinate_counts": dict(sorted(coordinates.items())),
            "sampled_symmetries": dict(sorted(Counter(item["symmetry"] for row in curve for item in row["samples"]).items())),
            "model_tokens_per_update": distribution(model_tokens), "effective_tokens_per_update": distribution(actual_tokens),
            "supervised_tokens_per_update": distribution(response_tokens),
            "validation_candidates": [{"step": step, "loss": loss} for loss, step in assessments],
            "selected_step": best.get("step"), "selected_validation_loss": best.get("validation_loss"),
            "files_sha256": {str(path): sha(path) for path in
                             (dataset, config_path, status_path, best_path, curve_path, checkpoint / "adapter_config.json",
                              checkpoint_weights)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, help="alternative, separately frozen training plan")
    parser.add_argument("--arm", action="append", default=[], help="override one arm directory as name=path")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    stage, output = args.stage.resolve(), args.output.resolve()
    freeze_path = args.freeze.resolve() if args.freeze else stage / "training_freeze.json"
    freeze = read_json(freeze_path)
    directories = {name: stage / f"adapter_{name}" for name in ARMS}
    overridden: set[str] = set()
    for value in args.arm:
        name, separator, directory = value.partition("=")
        if not separator or name not in directories or name in overridden or not directory:
            raise ValueError("arm overrides must be unique name=path entries for known arms")
        directories[name] = resolve(directory)
        overridden.add(name)
    from transformers import AutoTokenizer

    from carl.icl.intent_generator_experiment import SYSTEM

    tokenizer_path = resolve("analysis/intent_generator_20261114T032232Z/models/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    reference = read_json(stage.parent / "stage_03/fresh_evaluation/contract.json")
    model_weights = tokenizer_path / "model.safetensors"
    model_hash = sha(model_weights)
    reference_model_hash = reference["files_sha256"].get(str(model_weights.resolve()))
    arms = {}
    for name in ARMS:
        directory = directories[name]
        status_path = directory / "status.json"
        status = read_json(status_path) if status_path.is_file() else {}
        if status.get("status") != "finished":
            arms[name] = {"status": "not_finished", "observed_status": status.get("status"),
                          "completed_steps": status.get("completed_steps")}
            continue
        arms[name] = audit_arm(name, directory, freeze, tokenizer, SYSTEM)
    completed = [name for name, data in arms.items() if data["status"] in ("pass", "failed")]
    errors = [f"{name}: {error}" for name in completed for error in arms[name]["errors"]]
    if model_hash != reference_model_hash:
        errors.append("cached base model weights differ from the frozen stage-three reference")
    for name in completed:
        if resolve(arms[name]["matched_arguments"]["model"]) != tokenizer_path:
            errors.append(f"{name}: base model path differs from the verified local weights")
    if len(completed) > 1:
        first = arms[completed[0]]
        for name in completed[1:]:
            if arms[name]["matched_arguments"] != first["matched_arguments"]:
                errors.append(f"{name}: base model, seed or training parameters differ across arms")
            if (arms[name]["validation_record_ids"] != first["validation_record_ids"]
                    or arms[name]["validation_layouts"] != first["validation_layouts"]):
                errors.append(f"{name}: validation selection differs across arms")
            if arms[name]["model_tokens_per_update"]["total"] != first["model_tokens_per_update"]["total"]:
                errors.append(f"{name}: padded model token budget differs across arms")
    comparison = {}
    if {"wide", "wide_aug"} <= set(completed):
        a, b = arms["wide"], arms["wide_aug"]
        matching = sum(x == y for x, y in zip(a["sample_sequence"], b["sample_sequence"]))
        comparison["wide_vs_wide_aug"] = {"same_training_dataset_hash": a["training_data_sha256"] == b["training_data_sha256"],
                                          "identical_sample_positions": matching,
                                          "sample_positions": len(a["sample_sequence"]),
                                          "effective_token_difference": b["effective_tokens_per_update"]["total"] - a["effective_tokens_per_update"]["total"],
                                          "supervised_token_difference": b["supervised_tokens_per_update"]["total"] - a["supervised_tokens_per_update"]["total"]}
        if (not comparison["wide_vs_wide_aug"]["same_training_dataset_hash"]
                or matching != len(a["sample_sequence"]) or matching != len(b["sample_sequence"])):
            errors.append("wide and wide_aug do not use the same source dataset and sample sequence")
    if {"narrow", "wide"} <= set(completed):
        a, b = arms["narrow"], arms["wide"]
        comparison["narrow_vs_wide"] = {"common_original_record_ids": len(set(a["source_record_ids"]) & set(b["source_record_ids"])),
                                        "common_sampled_record_ids": len(set(a["sampled_record_ids"]) & set(b["sampled_record_ids"])),
                                        "effective_token_difference": b["effective_tokens_per_update"]["total"] - a["effective_tokens_per_update"]["total"],
                                        "supervised_token_difference": b["supervised_tokens_per_update"]["total"] - a["supervised_tokens_per_update"]["total"]}
        if a["original_train_records"] != b["original_train_records"]:
            errors.append("narrow and wide original example budgets differ")
    state = "failed" if errors else "pass" if len(completed) == len(ARMS) else "partial"
    if state == "partial" and not args.allow_partial:
        raise ValueError("arms are unfinished; use --allow-partial and a separate output path")
    if state != "pass" and output.name.startswith("train_independent_audit") and "partial" not in output.stem:
        raise ValueError("the final audit path is reserved for three completed passing arms")
    result = {"state": state, "errors": errors, "arms": arms, "comparisons": comparison,
              "freeze_sha256": sha(freeze_path), "audit_source_sha256": sha(Path(__file__)),
              "tokenizer_sha256": sha(tokenizer_path / "tokenizer.json"), "base_model_sha256": model_hash,
              "reference_model_sha256": reference_model_hash,
              "tokenisation": "independent reconstruction of visible board, task, response and EOS for every sampled record",
              "arm_directories": {name: str(path) for name, path in directories.items()},
              "freeze_path": str(freeze_path),
              "reproduce_command": shlex.join(["uv", "run", "--offline", "--no-sync", "python", "-m",
                  "scripts.audit_overnight_stage04_training", "--stage", str(stage), "--freeze", str(freeze_path),
                  *[item for name, path in directories.items() for item in ("--arm", f"{name}={path}")],
                  "--output", "<new-output.json>"])}
    atomic_json(output, result)
    print(json.dumps({"state": state, "completed_arms": completed, "errors": len(errors),
                      "output": str(output)}, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
