"""Portable, dependency-light verification for local PEFT checkpoints."""

import hashlib
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carl.icl.boards import canonical_layout_id, from_rows
from carl.icl.intents.generation.data import DataSplit, file_sha256, record_task_group
from carl.icl.predicate_prompts import GENERATOR_PROMPT_VERSION, generator_instruction

ADAPTER_MANIFEST = "adapter_manifest.json"
ADAPTER_SCHEMA = "carl-lora-adapter-v1"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
CHECKPOINT_CRITERION = "minimum_validation_target_token_loss_first_tie"
_BASE_METADATA = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "chat_template.jinja", "added_tokens.json", "vocab.json", "merges.txt")


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return data


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def verify_model_revision(base: Path, revision: str) -> None:
    """Accept a pinned HF snapshot directory or an explicit local revision record."""
    if not base.is_dir() or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("model must be a local directory with an exact 40-character revision")
    record = base / "experiment_revision.json"
    if record.is_file():
        if _read_json(record).get("revision") != revision:
            raise ValueError("cannot verify the requested local model revision")
    elif base.name != revision:
        raise ValueError("cannot verify the requested local model revision")


def _base_weights(base: Path) -> set[str]:
    index = "model.safetensors.index.json"
    if not (base / index).is_file():
        return {"model.safetensors"}
    weight_map = _read_json(base / index).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("invalid local safetensors index")
    if any(not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors")
           for shard in weight_map.values()):
        raise ValueError("local safetensors index must reference safetensors files in the snapshot")
    return {index, *weight_map.values()}


def base_snapshot_identity(base: Path, revision: str) -> dict[str, Any]:
    """Identify model weights, tokenizer and generation files independently of location."""
    verify_model_revision(base, revision)
    names = {"config.json", "tokenizer.json", *_base_weights(base)}
    names.update(name for name in _BASE_METADATA if (base / name).is_file())
    files = {}
    for name in sorted(names):
        if not (base / name).is_file():
            raise ValueError(f"local model snapshot is missing {name}")
        files[name] = file_sha256(base / name)
    return {"revision": revision, "files": files}


def predicate_prompt_contract(scale_arm: str) -> dict[str, str]:
    """Identify exactly the predicate generator instruction used for tuning."""
    text = generator_instruction(scale_arm)
    return {"version": GENERATOR_PROMPT_VERSION, "scale_arm": scale_arm,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest()}


def _artifact_files(checkpoint: Path) -> dict[str, str]:
    tokenizer = checkpoint / "tokenizer"
    if not (tokenizer / "tokenizer.json").is_file():
        raise ValueError("adapter is missing its saved tokenizer")
    paths = [checkpoint / name for name in ADAPTER_FILES]
    paths.extend(path for path in tokenizer.rglob("*") if path.is_file())
    return {path.relative_to(checkpoint).as_posix(): file_sha256(path) for path in sorted(paths)}


def publish_adapter(
    model: Any, tokenizer: Any, output: Path, name: str, *, base_identity: dict[str, Any], role: str,
    training: dict[str, Any], prompt: dict[str, str], step: int, validation_loss: float | None,
) -> Path:
    """Publish the weights, tokenizer and provenance together from one complete directory."""
    destination = output / name
    if destination.exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    with TemporaryDirectory(dir=output, prefix=f".{name}.") as workspace:
        staged = Path(workspace) / "adapter"
        model.save_pretrained(staged, safe_serialization=True)
        tokenizer.save_pretrained(staged / "tokenizer")
        manifest = {
            "schema_version": ADAPTER_SCHEMA,
            "role": role,
            "base": base_identity,
            "prompt": prompt,
            "training": training,
            "checkpoint": {"step": step, "criterion": CHECKPOINT_CRITERION, "validation_loss": validation_loss},
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "files": _artifact_files(staged),
        }
        manifest["fingerprint"] = _digest(manifest)
        (staged / ADAPTER_MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.rename(destination)
    return destination


def verify_adapter_artifact(checkpoint: Path, *, base: Path, revision: str, role: str) -> dict[str, Any]:
    """Check the adapter's role, base revision and bound files before loading PEFT."""
    path = checkpoint / ADAPTER_MANIFEST
    if not path.is_file():
        raise ValueError(f"adapter is missing {ADAPTER_MANIFEST}")
    manifest = _read_json(path)
    if manifest.get("schema_version") != ADAPTER_SCHEMA or manifest.get("role") != role:
        raise ValueError(f"adapter role or contract does not match {role}")
    fingerprint = manifest.pop("fingerprint", None)
    if fingerprint != _digest(manifest):
        raise ValueError("adapter manifest fingerprint does not match its contents")
    if manifest.get("base") != base_snapshot_identity(base, revision):
        raise ValueError("adapter base files or revision differ from training")
    if manifest.get("files") != _artifact_files(checkpoint):
        raise ValueError("adapter files differ from the pinned checkpoint")
    adapter_config = _read_json(checkpoint / "adapter_config.json")
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("adapter must be PEFT LoRA for a causal language model")
    training = manifest.get("training")
    if not isinstance(training, dict) or not isinstance(manifest.get("prompt"), dict):
        raise ValueError("adapter manifest is missing its training or prompt contract")
    if ("rank" in training and adapter_config.get("r") != training["rank"]
            or "lora_targets" in training and sorted(adapter_config.get("target_modules") or []) != sorted(training["lora_targets"])):
        raise ValueError("adapter rank or targets differ from the training configuration")
    return {**manifest, "fingerprint": fingerprint}


def verify_validation_selection(checkpoint: Path, manifest: dict[str, Any]) -> None:
    """Require the nominated best checkpoint when evaluating on held-out queries."""
    decision = checkpoint.parent / "best.json"
    if not decision.is_file():
        raise ValueError("adapter must have a validation-selected checkpoint")
    best = _read_json(decision)
    checkpoint_record = manifest["checkpoint"]
    if (Path(str(best.get("adapter"))).name != checkpoint.name
            or best.get("step") != checkpoint_record["step"]
            or best.get("validation_loss") != checkpoint_record["validation_loss"]
            or best.get("adapter_fingerprint") != manifest["fingerprint"]
            or checkpoint_record["validation_loss"] is None):
        raise ValueError("adapter does not match the pinned validation-selected checkpoint")


def has_adapter_contract(checkpoint: Path) -> bool:
    """Recognize managed checkpoints even when their manifest was removed."""
    if (checkpoint / ADAPTER_MANIFEST).is_file():
        return True
    configuration = checkpoint.parent / "configuration.json"
    return configuration.is_file() and _read_json(configuration).get("checkpoint_selection") == CHECKPOINT_CRITERION


def verify_held_out_records(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    """Reject evaluation rows that expose a training or model-selection partition."""
    training = manifest["training"]
    layouts = set(training["train_layout_ids"]) | set(training["validation_layout_ids"])
    groups = set(training["train_task_groups"]) | set(training["validation_task_groups"])
    for row in rows:
        if row.get("split") != DataSplit.TEST:
            raise ValueError("adapter evaluation requires the test split")
        layout = canonical_layout_id(from_rows(row["start_observation"]))
        if row["layout_id"] != layout or layout in layouts:
            raise ValueError("test canonical layouts overlap training or validation or have invalid IDs")
        group = record_task_group(row)
        if groups and (group is None or group in groups):
            raise ValueError("test task group is missing or overlaps training or validation")
