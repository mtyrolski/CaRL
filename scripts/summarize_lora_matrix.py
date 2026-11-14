#!/usr/bin/env python3
"""Summarise matched-step and validation-selected LoRA adapters without loading a model."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from safetensors import safe_open


def file_hash(path: Path) -> str | None:
    """Return a checksum for an existing file; absent outputs remain unknown."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_curve(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as source:
        for index, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid training curve at {path}:{index}") from error
            if (not isinstance(row, dict) or not isinstance(row.get("step"), int)
                    or isinstance(row.get("step"), bool)):
                raise ValueError(f"Invalid training step at {path}:{index}")
            if rows and row["step"] <= rows[-1]["step"]:
                raise ValueError(f"Training steps must increase strictly: {path}:{index}")
            rows.append(row)
    return rows


def checkpoint(path: Path | None, step: int | None, curve: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe one existing adapter, including its validation at that step."""
    if path is None or not path.is_dir():
        return {"step": None, "path": None, "weights_sha256": None, "config_sha256": None, "validation_loss": None,
                "saved_adapter_parameters": None, "weights_bytes": None}
    weights = path / "adapter_model.safetensors"
    config = path / "adapter_config.json"
    weight_hash, config_hash = file_hash(weights), file_hash(config)
    if weight_hash is None or config_hash is None:
        return {"step": None, "path": None, "weights_sha256": None, "config_sha256": None, "validation_loss": None,
                "saved_adapter_parameters": None, "weights_bytes": None}
    with safe_open(str(weights), framework="np") as tensors:
        parameter_count = sum(math.prod(tensors.get_slice(key).get_shape()) for key in tensors.keys())
    row = next((row for row in curve if row["step"] == step), {})
    return {"step": step, "path": str(path), "weights_sha256": weight_hash, "config_sha256": config_hash,
            "validation_loss": row.get("validation_loss"), "saved_adapter_parameters": parameter_count,
            "weights_bytes": weights.stat().st_size}


def summarize_variant(name: str, adapter_dir: Path, matched_steps: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep the same-step comparison distinct from validation-selected adapters."""
    directory = adapter_dir.resolve()
    configuration = read_object(directory / "configuration.json")
    status = read_object(directory / "status.json")
    best = read_object(directory / "best.json")
    curve = read_curve(directory / "training_curve.jsonl")
    args = configuration.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"Invalid training arguments: {directory}")
    best_step = best.get("step")
    if best_step is not None and (not isinstance(best_step, int) or isinstance(best_step, bool) or best_step < 1):
        raise ValueError(f"Invalid best checkpoint step: {directory}")
    # A completed fixed-length run always saves its terminal adapter here. A
    # longer historical run may have retained the same step as a best checkpoint.
    terminal = directory / f"checkpoint-last-{matched_steps}"
    archived = directory / f"checkpoint-{matched_steps}"
    matched_path = terminal if status.get("completed_steps") == matched_steps and terminal.is_dir() else archived
    best_path = directory / f"checkpoint-{best_step}" if best_step is not None else None
    matched = checkpoint(matched_path, matched_steps, curve)
    selected = checkpoint(best_path, best_step, curve)
    if best_step is not None and selected["step"] is None:
        raise ValueError(f"Recorded best checkpoint is incomplete or missing: {directory}")
    if selected["step"] is not None and selected["validation_loss"] != best.get("validation_loss"):
        raise ValueError(f"Best checkpoint disagrees with the validation curve: {directory}")
    config_checkpoint = selected["path"] or matched["path"]
    selected_config = read_object(Path(config_checkpoint) / "adapter_config.json") if config_checkpoint else {}
    targets = selected_config.get("target_modules")
    if targets is not None and (not isinstance(targets, list) or not all(isinstance(target, str) for target in targets)):
        raise ValueError(f"Invalid adapter target modules: {directory}")
    seconds = [row.get("seconds") for row in curve]
    measured_seconds = [float(value) for value in seconds if isinstance(value, (int, float)) and not isinstance(value, bool)]
    result = {
        "name": name,
        "adapter_dir": str(directory),
        "status": status.get("status"),
        "completed_steps": status.get("completed_steps"),
        "configured_max_steps": args.get("max_steps"),
        "rank": args.get("rank"),
        "learning_rate": args.get("lr"),
        "accumulation": args.get("accumulation"),
        "seed": args.get("seed"),
        "target_modules": sorted(targets) if targets is not None else None,
        "model": args.get("model"),
        "model_revision": args.get("revision"),
        "data_sha256": configuration.get("data_sha256"),
        "validation_record_ids": configuration.get("validation_record_ids"),
        "configuration_sha256": file_hash(directory / "configuration.json"),
        "curve_sha256": file_hash(directory / "training_curve.jsonl"),
        "train_loss_first": curve[0].get("train_loss") if curve else None,
        "train_loss_last": curve[-1].get("train_loss") if curve else None,
        "measured_step_seconds": sum(measured_seconds) if seconds and len(measured_seconds) == len(seconds) else None,
        "matched": matched,
        "validation_best": selected,
    }
    return result, [{"variant": name, **row} for row in curve]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue())


def summarize(variants: list[tuple[str, Path]], matched_steps: int, output_dir: Path) -> dict[str, Any]:
    if matched_steps < 1 or not variants or len({name for name, _ in variants}) != len(variants):
        raise ValueError("Use a positive matched step and distinct, nonempty variant names")
    if any(not name.strip() for name, _ in variants):
        raise ValueError("Variant names must not be empty")
    summaries, curves = [], []
    for name, path in variants:
        summary, rows = summarize_variant(name, path, matched_steps)
        summaries.append(summary)
        curves.extend(rows)
    result = {"schema_version": 1, "matched_steps": matched_steps, "variants": summaries,
              "note": "Matched-step checkpoints and validation-selected checkpoints are distinct comparisons."}
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "adapter_manifest.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    table = []
    for item in summaries:
        row = {key: value for key, value in item.items() if key not in {"matched", "validation_best", "validation_record_ids"}}
        row["target_modules"] = json.dumps(row["target_modules"])
        for prefix in ("matched", "validation_best"):
            row.update({f"{prefix}_{key}": value for key, value in item[prefix].items()})
        table.append(row)
    fields = list(table[0])
    write_csv(output_dir / "adapter_summary.csv", table, fields)
    write_csv(output_dir / "training_curves.csv", curves,
              ["variant", "step", "train_loss", "validation_loss", "seconds"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=ADAPTER_DIR")
    parser.add_argument("--matched-steps", type=int, default=100)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    variants = []
    for value in args.variant:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("each --variant must be NAME=ADAPTER_DIR")
        variants.append((name, Path(path)))
    summarize(variants, args.matched_steps, args.output_dir)
    print(args.output_dir / "adapter_manifest.json")


if __name__ == "__main__":
    main()
