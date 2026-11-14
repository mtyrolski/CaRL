#!/usr/bin/env python3
"""Freeze the complete, preregistered Sokoban intent representation matrix."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "analysis/intent_generator_20261114T032232Z"
LORA_MATRIX = ROOT / "analysis/lora_matrix_20261114T065804Z"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_MATRIX_SECONDS = 3 * 60 * 60 + 15 * 60


class Group(StrEnum):
    """Disjoint, explicitly ordered groups of model evaluations."""

    CORE = "core"
    SECONDARY = "secondary"
    ADAPTED = "adapted"
    REPLICATE = "replicate"


@dataclass(frozen=True)
class Method:
    name: str
    group: Group
    representation: str
    support_set: str
    adapter: str | None = None
    adapter_pending: bool = False


def methods(run: Path) -> list[Method]:
    """Return the complete matrix without reading any model outputs."""
    r4 = str(ORIGINAL / "adapter/checkpoint-100")
    r8 = str(LORA_MATRIX / "adapter_r8_lr2e4/checkpoint-last-100")
    r2 = str(LORA_MATRIX / "adapter_r2_lr2e4/checkpoint-last-100")
    core = [
        Method("base_json_zero", Group.CORE, "json", "zero"),
        Method("r4_json_zero", Group.CORE, "json", "zero", r4),
        Method("r8_json_zero", Group.CORE, "json", "zero", r8),
        Method("r2_json_zero", Group.CORE, "json", "zero", r2),
        Method("base_json_balanced0", Group.CORE, "json", "balanced0"),
        Method("base_formal_balanced0", Group.CORE, "formal", "balanced0"),
        Method("base_natural_balanced0", Group.CORE, "natural", "balanced0"),
        Method("base_formal_zero", Group.CORE, "formal", "zero"),
        Method("base_natural_zero", Group.CORE, "natural", "zero"),
    ]
    secondary = [Method("r4_json_balanced0", Group.SECONDARY, "json", "balanced0", r4),
                 Method("r8_json_balanced0", Group.SECONDARY, "json", "balanced0", r8)]
    secondary.extend(Method(f"base_{representation}_{support}", Group.SECONDARY, representation, support)
                     for support in ("legacy", "balanced1", "balanced2")
                     for representation in ("json", "formal", "natural"))
    adapted = [Method(f"rep_lora_{representation}_zero", Group.ADAPTED, representation, "zero",
                      str(run / f"adapter_repr_{representation}_seed1729/checkpoint-last-100"), True)
               for representation in ("json", "formal", "natural")]
    replicate = [Method(f"rep_lora_{representation}_seed{seed}", Group.REPLICATE, representation, "zero",
                        str(run / f"adapter_repr_{representation}_seed{seed}/checkpoint-last-100"), True)
                 for seed in (2718, 3141) for representation in ("json", "formal", "natural")]
    return [*core, *secondary, *adapted, *replicate]


def matrix_deadline(clock: dict[str, Any]) -> tuple[float, float]:
    """Enforce both UTC and monotonic deadlines at most 3 h 15 min after start."""
    return float(clock["start_epoch"]) + MAX_MATRIX_SECONDS, float(clock["start_monotonic"]) + MAX_MATRIX_SECONDS


def adapter_hashes(path: str | None, pending: bool) -> dict[str, str] | None:
    if path is None or pending:
        return None
    adapter = Path(path)
    return {name: file_sha256(adapter / name) for name in ("adapter_model.safetensors", "adapter_config.json")}


def build_plan(run: Path, expected_contexts: int = 128) -> dict[str, Any]:
    """Validate input provenance and assign each method a fixed output location."""
    run = run.resolve()
    data, ids = run / "data/dataset.jsonl", run / "data/test_ids.json"
    support_data, supports = ORIGINAL / "dataset.jsonl", run / "support_manifest.json"
    model = ORIGINAL / "models/Qwen3-0.6B"
    files = {
        "data": data, "test_ids": ids, "support_data": support_data, "support_manifest": supports,
        "sampling_plan": run / "data/sampling_plan.json", "independent_audit": run / "data/independent_audit.json",
        "protocol": ROOT / "carl/icl/intent_representation_experiment.py",
        "generator": ROOT / "carl/icl/intent_generator_experiment.py",
        "support_selector": ROOT / "carl/icl/intent_in_context.py",
        "intent_checker": ROOT / "carl/icl/intent_data.py",
        "matrix_runner": ROOT / "scripts/run_representation_matrix.py",
        "matrix_builder": ROOT / "scripts/prepare_representation_matrix.py",
    }
    hashes = {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in files.items()}
    frozen = json.loads(ids.read_text(encoding="utf-8"))
    identifiers = frozen.get("record_ids")
    if (not isinstance(identifiers, list) or len(identifiers) != expected_contexts
            or len(set(identifiers)) != expected_contexts or not all(isinstance(v, str) for v in identifiers)):
        raise ValueError(f"Expected exactly {expected_contexts} unique frozen test IDs")
    if frozen.get("dataset_sha256") != hashes["data"]["sha256"]:
        raise ValueError("Frozen test IDs have an inconsistent dataset hash")
    rows = read_records(data)
    by_id = {row["record_id"]: row for row in rows}
    if len(by_id) != len(rows) or any(identifier not in by_id for identifier in identifiers):
        raise ValueError("Frozen IDs are missing or dataset IDs duplicate")
    test = [by_id[identifier] for identifier in identifiers]
    if any(row.get("split") != "test" for row in test) or len({r["layout_id"] for r in test}) != expected_contexts:
        raise ValueError("Frozen cases must be test records on distinct canonical layouts")
    support = json.loads(supports.read_text(encoding="utf-8"))
    if support.get("support_data_sha256") != hashes["support_data"]["sha256"]:
        raise ValueError("Frozen support does not belong to the checked train data")
    model_revision = json.loads((model / "experiment_revision.json").read_text(encoding="utf-8"))
    if model_revision.get("revision") != REVISION or model_revision.get("model") != "Qwen/Qwen3-0.6B":
        raise ValueError("Local model revision does not match the study")
    model_hashes = {name: file_sha256(model / name)
                    for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")}
    supervisor = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    epoch, monotonic = matrix_deadline(supervisor["clock"])
    epoch = min(epoch, float(supervisor["expensive_start_cutoff_epoch"]))
    monotonic = min(monotonic, float(supervisor["expensive_start_cutoff_monotonic"]))
    jobs = []
    for method in methods(run):
        job = asdict(method)
        job["adapter_sha256"] = adapter_hashes(method.adapter, method.adapter_pending)
        job["output"] = str(run / "predictions" / f"{method.name}.jsonl")
        job["log"] = str(run / "logs" / f"matrix_{method.name}.log")
        jobs.append(job)
    names = [job["name"] for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("Matrix method names must be unique")
    return {
        "schema_version": 2,
        "status": "frozen_before_evaluation",
        "scope": "Predetermined methods and input assets; no model predictions were inspected.",
        "run": str(run),
        "test_contexts": len(identifiers),
        "test_record_ids_sha256": hashlib.sha256(json.dumps(identifiers, separators=(",", ":")).encode()).hexdigest(),
        "inputs": hashes,
        "model": {"path": str(model), "revision": REVISION, "sha256": model_hashes},
        "settings": {"seed": 1729, "dtype": "bfloat16", "threads": 8, "max_length": 1024, "max_new_tokens": 128,
                     "decoding": "greedy", "enable_thinking": False, "split": "test", "require_cuda": True},
        "deadline_epoch": epoch, "deadline_monotonic": monotonic,
        "deadline_utc": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
        "adapter_note": "Representation adapters do not exist at freeze time; their actual hashes are recorded "
                        "immediately before evaluation and in the per-method run configuration.",
        "previous_plan_sha256": file_sha256(run / "matrix_plan_v1.json"),
        "amendment": "Before the first prediction, strengthened resume validation, explicitly fixed thread count, "
                     "and preregistered optional training-seed replicates.",
        "methods": jobs,
    }


def write_frozen(output: Path, plan: dict[str, Any]) -> None:
    content = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    digest = hashlib.sha256(content.encode()).hexdigest() + "\n"
    checksum = output.with_suffix(".sha256")
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Refusing to alter frozen matrix plan: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, output)
    if checksum.exists() and checksum.read_text(encoding="utf-8") != digest:
        raise ValueError("Frozen matrix checksum differs")
    if not checksum.exists():
        temporary = checksum.with_name(f".{checksum.name}.{os.getpid()}.tmp")
        temporary.write_text(digest, encoding="utf-8")
        os.replace(temporary, checksum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output or run / "matrix_plan.json"
    plan = build_plan(run)
    write_frozen(output, plan)
    print(json.dumps({"plan": str(output), "methods": len(plan["methods"]),
                      "contexts_per_method": plan["test_contexts"], "deadline_utc": plan["deadline_utc"]}))


if __name__ == "__main__":
    main()
