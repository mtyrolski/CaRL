#!/usr/bin/env python3
"""Audit matched LoRA training without loading weights onto a GPU.

The first 400 training IDs are reconstructed from the saved input order,
seed and hashed sampler implementation. Training did not log per-step IDs;
the reconstruction is evidence of the expected order, not an execution trace.
"""

import argparse
import hashlib
import json
import math
import os
from enum import StrEnum
from itertools import islice
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import board_balanced
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_generator_experiment import validation_contexts
from carl.icl.intent_representation_training import encode_example
from carl.icl.intent_representation_training import messages_for
from carl.icl.intent_representation_training import render_representation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis/intent_generator_20261114T032232Z/dataset.jsonl"
REPRESENTATIONS = ("json", "formal", "natural")
COMMON_FIELDS = (
    "mode", "model", "revision", "revision_verification", "model_weights_sha256", "model_config_sha256",
    "tokenizer_sha256", "data_sha256", "training_records_sha256", "validation_records_sha256",
    "train_record_ids", "validation_record_ids", "train_layouts", "validation_layouts",
    "protocol_sha256", "training_code_sha256", "initialiser_code_sha256", "dtype", "seed", "threads", "device",
    "rank", "lora_alpha", "lora_targets", "microbatch", "accumulation", "learning_rate", "gradient_checkpointing",
    "max_steps", "eval_every", "max_length", "max_new_tokens", "loss", "enable_thinking", "sampling",
)
EXPECTED = {"mode": "train", "dtype": "bfloat16", "seed": 1729, "threads": 8, "device": "cuda",
            "rank": 4, "lora_alpha": 8, "lora_targets": ["q_proj", "v_proj"], "microbatch": 1,
            "accumulation": 4, "learning_rate": 0.0002, "gradient_checkpointing": True,
            "max_steps": 100, "eval_every": 20, "max_length": 512, "max_new_tokens": 128,
            "loss": "response tokens and EOS only", "enable_thinking": False, "sampling": "board-balanced, seeded"}


class AuditStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    MISSING = "missing"
    INCOMPLETE = "incomplete"


def digest_records(records: list[dict[str, Any]]) -> str:
    """Recalculate the trainer's exact whitelist digest from the source data."""
    allowed = ("record_id", "layout_id", "split", "start_observation", "task_goal", "intent")
    rows = [{key: row[key] for key in allowed} for row in records]
    value = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def digest_value(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def command_value(command: list[str], name: str) -> str:
    indices = [index for index, value in enumerate(command) if value == name]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError(f"Expected exactly one command argument: {name}")
    return command[indices[0] + 1]


def verify_command(config: dict[str, Any], directory: Path) -> None:
    """Check the recorded launch matches the saved representation and settings."""
    command = config["command"]
    if not isinstance(command, list) or len(command) < 2 or command[1] != "train":
        raise ValueError("Recorded command is not a training launch")
    flags = {"--representation": "representation", "--revision": "revision", "--dtype": "dtype", "--seed": "seed",
             "--accumulation": "accumulation", "--lr": "learning_rate", "--max-length": "max_length",
             "--max-new-tokens": "max_new_tokens", "--max-steps": "max_steps", "--eval-every": "eval_every"}
    for flag, key in flags.items():
        if command_value(command, flag) != str(config[key]):
            raise ValueError(f"Recorded command differs from configuration: {flag}")
    for flag, expected in (("--data", DATA), ("--model", Path(config["model"])), ("--output", directory)):
        value = Path(command_value(command, flag))
        if (value if value.is_absolute() else ROOT / value).resolve() != expected.resolve():
            raise ValueError(f"Recorded command has a different path: {flag}")
    if command.count("--gradient-checkpointing") != 1 or command.count("--require-cuda") != 1:
        raise ValueError("Recorded command omits required training flags")


def training_inputs(config: dict[str, Any], source: list[dict[str, Any]], test: list[dict[str, Any]]) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve saved IDs and independently re-run deterministic split selection."""
    indexed = {row["record_id"]: row for row in source}
    if len(indexed) != len(source):
        raise ValueError("Duplicate IDs in source dataset")
    train_ids, validation_ids = config["train_record_ids"], config["validation_record_ids"]
    if (not isinstance(train_ids, list) or not isinstance(validation_ids, list)
            or len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(set(validation_ids))
            or any(identifier not in indexed for identifier in train_ids + validation_ids)):
        raise ValueError("Missing or duplicated train/validation IDs")
    training = [indexed[identifier] for identifier in train_ids]
    validation = [indexed[identifier] for identifier in validation_ids]
    expected_train = [row for row in source if row.get("split") == "train"]
    max_boards = int(command_value(config["command"], "--validation-boards"))
    per_board = int(command_value(config["command"], "--validation-contexts-per-board"))
    expected_val = validation_contexts([row for row in source if row.get("split") == "validation"],
                                       max_boards, per_board)
    if training != expected_train or validation != expected_val:
        raise ValueError("Saved training/validation order differs from source selection")
    if (not training or not validation or any(row.get("split") != "train" for row in training)
            or any(row.get("split") != "validation" for row in validation)
            or any(row.get("validation", {}).get("accepted") is not True for row in training + validation)):
        raise ValueError("Training/validation rows must be nonempty, checked and correctly split")
    train_layouts = {row["layout_id"] for row in training}
    val_layouts = {row["layout_id"] for row in validation}
    test_layouts = {row["layout_id"] for row in test}
    test_ids = {row["record_id"] for row in test}
    if (train_layouts & val_layouts or (train_layouts | val_layouts) & test_layouts
            or set(train_ids + validation_ids) & test_ids):
        raise ValueError("Training or validation overlaps the test contexts or layouts")
    if (config["train_layouts"] != sorted(train_layouts) or config["validation_layouts"] != sorted(val_layouts)
            or config["training_records_sha256"] != digest_records(training)
            or config["validation_records_sha256"] != digest_records(validation)):
        raise ValueError("Saved training/validation digest or layout list differs from source")
    return training, validation


def verify_response_mask(tokenizer: Any, record: dict[str, Any], representation: str, limit: int) -> dict[str, int]:
    """Check a real example: board/system tokens masked; full answer and EOS trained."""
    prefix = tokenizer.apply_chat_template(messages_for(record, [], representation), tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False)
    if not isinstance(prefix, list):
        prefix = prefix.tolist()
    answer = tokenizer.encode(render_representation(representation, record["intent"]) + (tokenizer.eos_token or ""),
                              add_special_tokens=False)
    encoded = encode_example(tokenizer, record, representation, limit)
    if (not answer or encoded["input_ids"] != prefix + answer
            or encoded["labels"] != [-100] * len(prefix) + answer
            or encoded["attention_mask"] != [1] * (len(prefix) + len(answer))):
        raise ValueError("Response-only loss mask does not match the complete assistant response")
    altered = {**record, "record_id": "hidden", "source_path": "hidden", "i": 999, "k": 999,
               "later_observation": ["hidden-future"], "fragment_actions": [999],
               "validation": {"hidden": True}, "intent": {"hidden": "label"}}
    if messages_for(record, [], representation) != messages_for(altered, [], representation):
        raise ValueError("The generator input depends on hidden trajectory or label fields")
    return {"prompt_tokens_masked": len(prefix), "response_tokens_supervised": len(answer),
            "total_tokens": len(prefix) + len(answer)}


def verify_curve(status: dict[str, Any], best: dict[str, Any], curve: list[dict[str, Any]],
                 max_steps: int = 100, eval_every: int = 20) -> None:
    """Check completion, all validation checkpoints and deterministic best selection."""
    if (status.get("status") != "finished" or status.get("completed_steps") != max_steps
            or status.get("last_validation_step") != max_steps):
        raise ValueError("Training did not finish with a final validation measurement")
    if [row.get("step") for row in curve] != list(range(1, max_steps + 1)):
        raise ValueError("Training curve does not contain every completed step in order")
    observed = [(row["step"], row["validation_loss"]) for row in curve if "validation_loss" in row]
    expected = sorted(set(range(eval_every, max_steps + 1, eval_every)) | {max_steps})
    if ([step for step, _ in observed] != expected
            or any(isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(loss)
                   for _, loss in observed)):
        raise ValueError("Validation measurement cadence or values are invalid")
    selected = min(observed, key=lambda item: item[1])
    if (best.get("step"), best.get("validation_loss")) != selected:
        raise ValueError("Recorded best checkpoint is not the minimum validation loss")


def resolved_checkpoint(path: str, directory: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(directory.resolve()):
        raise ValueError("Checkpoint lies outside its training run")
    return candidate


def inspect_checkpoint(path: Path, config: dict[str, Any]) -> dict[str, str]:
    settings = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if (settings.get("r") != config["rank"] or settings.get("lora_alpha") != config["lora_alpha"]
            or set(settings.get("target_modules", [])) != set(config["lora_targets"])
            or settings.get("lora_dropout") != 0.0 or settings.get("bias") != "none"
            or settings.get("task_type") != "CAUSAL_LM"):
        raise ValueError("Adapter config differs from recorded LoRA settings")
    return {"weights_sha256": file_sha256(path / "adapter_model.safetensors"),
            "config_sha256": file_sha256(path / "adapter_config.json")}


def compare_common(configs: dict[str, dict[str, Any]], expected_seed: int) -> dict[str, Any]:
    if set(configs) != set(REPRESENTATIONS):
        raise ValueError("Exactly JSON, formal and natural configurations are required")
    mismatches = {field: {representation: value.get(field) for representation, value in configs.items()}
                  for field in COMMON_FIELDS
                  if len({json.dumps(value.get(field), sort_keys=True) for value in configs.values()}) != 1}
    if mismatches:
        raise ValueError(f"Training configurations differ outside representation: {sorted(mismatches)}")
    common = {field: configs["json"].get(field) for field in COMMON_FIELDS}
    expected = {**EXPECTED, "seed": expected_seed}
    for key, value in expected.items():
        if common[key] != value:
            raise ValueError(f"Unexpected matched setting: {key}")
    if any(config.get("representation") != representation for representation, config in configs.items()):
        raise ValueError("A representation configuration has the wrong label")
    return common


def audit_seed(run: Path, seed: int, source: list[dict[str, Any]], tests: list[dict[str, Any]],
               tokenizer: Any) -> dict[str, Any]:
    directories = {name: run / f"adapter_repr_{name}_seed{seed}" for name in REPRESENTATIONS}
    missing = [name for name, path in directories.items() if not (path / "configuration.json").is_file()]
    if missing:
        return {"status": AuditStatus.MISSING, "missing_configurations": missing}
    configs = {name: json.loads((path / "configuration.json").read_text(encoding="utf-8"))
               for name, path in directories.items()}
    statuses = {name: json.loads((path / "status.json").read_text(encoding="utf-8"))
                if (path / "status.json").is_file() else None for name, path in directories.items()}
    incomplete = [name for name, value in statuses.items() if not value or value.get("status") != "finished"]
    if incomplete:
        return {"status": AuditStatus.INCOMPLETE, "incomplete": incomplete}
    common = compare_common(configs, seed)
    source_hash = file_sha256(DATA)
    if common["data_sha256"] != source_hash:
        raise ValueError("Configured training source hash differs from the actual file")
    code_files = {"protocol_sha256": ROOT / "carl/icl/intent_representation_experiment.py",
                  "training_code_sha256": ROOT / "carl/icl/intent_representation_training.py",
                  "initialiser_code_sha256": ROOT / "carl/icl/intent_generator_experiment.py"}
    measured_code = {name: file_sha256(path) for name, path in code_files.items()}
    if any(common[name] != value for name, value in measured_code.items()):
        raise ValueError("Recorded training source hash does not match current source")
    model = Path(common["model"])
    measured_model = {key: file_sha256(model / filename) for key, filename in (
        ("model_weights_sha256", "model.safetensors"), ("model_config_sha256", "config.json"),
        ("tokenizer_sha256", "tokenizer.json"))}
    if any(common[name] != value for name, value in measured_model.items()):
        raise ValueError("Recorded base model or tokenizer hash differs")
    metadata = json.loads((model / "experiment_revision.json").read_text(encoding="utf-8"))
    if metadata.get("revision") != common["revision"]:
        raise ValueError("Model revision differs from metadata")
    training_by_repr: dict[str, list[dict[str, Any]]] = {}
    validation_by_repr: dict[str, list[dict[str, Any]]] = {}
    sample_orders: dict[str, list[str]] = {}
    variants = {}
    for representation, directory in directories.items():
        config, status = configs[representation], statuses[representation]
        assert status is not None
        verify_command(config, directory)
        training, validation = training_inputs(config, source, tests)
        training_by_repr[representation], validation_by_repr[representation] = training, validation
        sequence = [row["record_id"] for row in islice(board_balanced(training, seed), 400)]
        if len(sequence) != 400 or any(row not in config["train_record_ids"] for row in sequence):
            raise ValueError("Reconstructed sampler selected a non-training record")
        sample_orders[representation] = sequence
        best = json.loads((directory / "best.json").read_text(encoding="utf-8"))
        curve = read_records(directory / "training_curve.jsonl")
        verify_curve(status, best, curve)
        best_path = resolved_checkpoint(best["adapter"], directory)
        last_path = resolved_checkpoint(status["last_adapter"], directory)
        if status.get("best_adapter") != best["adapter"]:
            raise ValueError("Status and validation-selected best checkpoint differ")
        best_hashes, last_hashes = inspect_checkpoint(best_path, config), inspect_checkpoint(last_path, config)
        masks = {"train": verify_response_mask(tokenizer, training[0], representation, config["max_length"]),
                 "validation": verify_response_mask(tokenizer, validation[0], representation, config["max_length"])}
        variants[representation] = {
            "configuration_sha256": file_sha256(directory / "configuration.json"),
            "status_sha256": file_sha256(directory / "status.json"),
            "curve_sha256": file_sha256(directory / "training_curve.jsonl"),
            "best_manifest_sha256": file_sha256(directory / "best.json"),
            "best_step": best["step"], "best_validation_loss": best["validation_loss"],
            "best_checkpoint": str(best_path), "best": best_hashes,
            "last_step": status["completed_steps"], "last_checkpoint": str(last_path), "last": last_hashes,
            "best_equals_last_weights": best_hashes["weights_sha256"] == last_hashes["weights_sha256"],
            "best_equals_last_config": best_hashes["config_sha256"] == last_hashes["config_sha256"],
            "validation_steps": [row["step"] for row in curve if "validation_loss" in row],
            "reconstructed_first_400_ids_sha256": digest_value(sequence),
            "response_mask_examples": masks,
        }
    if len({digest_value(values) for values in sample_orders.values()}) != 1:
        raise ValueError("Reconstructed board-balanced sample orders differ")
    if any(training_by_repr[name] != training_by_repr["json"]
           or validation_by_repr[name] != validation_by_repr["json"] for name in REPRESENTATIONS):
        raise ValueError("Matched representations do not use identical ordered training/validation records")
    return {
        "status": AuditStatus.PASSED, "seed": seed,
        "common_fields_sha256": digest_value(common),
        "common_fields": {key: value for key, value in common.items() if key not in
                          {"train_record_ids", "validation_record_ids", "train_layouts", "validation_layouts"}},
        "train_record_ids_sha256": digest_value(common["train_record_ids"]),
        "validation_record_ids_sha256": digest_value(common["validation_record_ids"]),
        "train_layouts_sha256": digest_value(common["train_layouts"]),
        "validation_layouts_sha256": digest_value(common["validation_layouts"]),
        "train_records": len(training_by_repr["json"]), "validation_records": len(validation_by_repr["json"]),
        "train_layouts": len(common["train_layouts"]), "validation_layouts": len(common["validation_layouts"]),
        "test_layouts": len({row["layout_id"] for row in tests}), "split_intersections": 0,
        "measured_code_sha256": measured_code, "measured_model_sha256": measured_model,
        "reconstructed_first_400_train_ids": sample_orders["json"],
        "reconstructed_sequence_note": "Sampler order was reconstructed from saved IDs, code hash and seed; "
                                       "per-step record IDs were not logged during training.",
        "variants": variants,
    }


def audit(run: Path, seeds: list[int]) -> dict[str, Any]:
    source = read_records(DATA)
    test_data = run / "data/dataset.jsonl"
    tests = read_records(test_data, "test")
    frozen = json.loads((run / "data/test_ids.json").read_text(encoding="utf-8"))
    if frozen.get("dataset_sha256") != file_sha256(test_data) or len(tests) != len(frozen["record_ids"]):
        raise ValueError("Test dataset or IDs differ from the frozen manifest")
    if set(frozen["record_ids"]) != {row["record_id"] for row in tests}:
        raise ValueError("Frozen test IDs differ from actual test records")
    from transformers import AutoTokenizer

    model = ROOT / "analysis/intent_generator_20261114T032232Z/models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    primary = audit_seed(run, 1729, source, tests, tokenizer)
    optional = {str(seed): audit_seed(run, seed, source, tests, tokenizer) for seed in seeds if seed != 1729}
    return {
        "schema_version": 1, "status": primary["status"], "primary": primary,
        "optional_seed_replications": optional,
        "source_dataset_sha256": file_sha256(DATA), "test_dataset_sha256": file_sha256(test_data),
        "test_ids_sha256": file_sha256(run / "data/test_ids.json"),
        "limitations": ["Exact per-step sample IDs were not saved during training; first 400 IDs are reconstructed.",
                        "Response-only masking was checked on one real training and validation example per format, "
                        "using the same hashed implementation as training.",
                        "Training losses of differently tokenised formats are not directly comparable as quality metrics."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--optional-seed", action="append", type=int, default=[])
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output or run / "matched_training_audit.json"
    result = audit(run, args.optional_seed)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": result["status"], "output": str(output),
                      "optional": {key: value["status"] for key, value in result["optional_seed_replications"].items()}}))
    if result["status"] != AuditStatus.PASSED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
