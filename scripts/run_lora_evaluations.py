#!/usr/bin/env python3
"""Sequentially evaluate frozen LoRA variants under the campaign watchdog."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path

import psutil

from scripts.intent_experiment_supervisor import StageKind
from scripts.intent_experiment_supervisor import StageStatus

ORIGINAL = Path("analysis/intent_generator_20261114T032232Z")
MODEL = ORIGINAL / "models/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class EvaluationGroup(StrEnum):
    MAIN = "main"
    TRANSFER = "transfer"


class EvaluationStatus(StrEnum):
    STARTED = "started"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MISSING_ADAPTER = "missing_adapter"
    EVALUATION_DEADLINE = "evaluation_deadline"
    TRAINING_STILL_RUNNING = "training_still_running_at_cutoff"


def transfer_icl_ids(run: Path) -> Path:
    """Verify that ICL receives the same ordered contexts and their dataset hash."""
    directory = run / "eval_transfer"
    canonical = json.loads((directory / "test_ids.json").read_text(encoding="utf-8"))
    enriched_path = directory / "icl_test_ids.json"
    enriched = json.loads(enriched_path.read_text(encoding="utf-8"))
    if not isinstance(canonical, dict) or not isinstance(enriched, dict):
        raise ValueError("ICL transfer ID files must contain objects")
    identifiers = canonical.get("record_ids")
    if (not isinstance(identifiers, list) or not identifiers
            or not all(isinstance(identifier, str) for identifier in identifiers)
            or len(set(identifiers)) != len(identifiers) or enriched.get("record_ids") != identifiers):
        raise ValueError("ICL transfer IDs differ from the frozen selection")
    digest = hashlib.sha256((directory / "dataset.jsonl").read_bytes()).hexdigest()
    if enriched.get("dataset_sha256") != digest:
        raise ValueError("ICL transfer IDs refer to a different dataset")
    return enriched_path


def journal(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8", buffering=1) as output:
        output.write(json.dumps(value) + "\n")


def stop_process_tree(process: subprocess.Popen) -> None:
    """Stop only the evaluator and its own descendants, retaining the supervisor group."""
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
        parent = None
    members = [*reversed(children), *([parent] if parent is not None else [])]
    for child in members:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(members, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            continue
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / "run_manifest.json"
    history = run / "evaluation_jobs.jsonl"
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    cutoff = manifest["expensive_start_cutoff_epoch"]
    cutoff_monotonic = manifest["expensive_start_cutoff_monotonic"]
    train_stop = manifest["clock"]["start_epoch"] + 14400
    train_stop_monotonic = manifest["training_stop_monotonic"]
    eval_stop = manifest["clock"]["deadline_epoch"] - 1200
    eval_stop_monotonic = manifest["clock"]["deadline_monotonic"] - 1200
    # Wait beyond the start cutoff so a last-moment training job cannot race
    # the evaluator; no GPU training and inference run concurrently.
    while True:
        state = json.loads(manifest_path.read_text(encoding="utf-8"))
        running = [stage for stage in state["stages"]
                   if stage.get("kind") == StageKind.TRAINING and stage.get("status") == StageStatus.RUNNING]
        cutoff_passed = time.time() >= cutoff + 2 or time.monotonic() >= cutoff_monotonic + 2
        if not running and cutoff_passed:
            break
        if time.time() >= train_stop or time.monotonic() >= train_stop_monotonic:
            journal(history, {"status": EvaluationStatus.TRAINING_STILL_RUNNING, "stages": [stage["name"] for stage in running]})
            return 1
        time.sleep(2)

    adapters = {
        "r4_100": ORIGINAL / "adapter/checkpoint-100",
        "r8_100": run / "adapter_r8_lr2e4/checkpoint-last-100",
        "r2_100": run / "adapter_r2_lr2e4/checkpoint-last-100",
        "r4_local_100": run / "adapter_r4_local_lr2e4/checkpoint-last-100",
        "base": None,
        "r4_best220": ORIGINAL / "adapter/checkpoint-220",
        "icl": None,
    }
    jobs = [(EvaluationGroup.TRANSFER, name, adapter) for name, adapter in adapters.items()]
    jobs += [(EvaluationGroup.MAIN, name, adapters[name]) for name in ("r4_100", "r8_100", "r2_100", "r4_local_100")]
    jobs += [(EvaluationGroup.TRANSFER, "r4_20", ORIGINAL / "adapter/checkpoint-20"),
             (EvaluationGroup.TRANSFER, "r4_last300", ORIGINAL / "adapter/checkpoint-last-300")]
    plan_path = run / "evaluation_plan.json"
    if not plan_path.exists():
        temporary = plan_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"schema_version": 1, "jobs": [{"group": group, "name": name,
                             "adapter": str(adapter) if adapter else None} for group, name, adapter in jobs],
                             "test_order": "fixed IDs, transfer alternates x4/x10", "eval_stop_epoch": eval_stop,
                             "eval_stop_monotonic": eval_stop_monotonic}, indent=2) + "\n")
        os.replace(temporary, plan_path)
    for group, name, adapter in jobs:
        remaining = min(eval_stop - time.time(), eval_stop_monotonic - time.monotonic())
        if remaining <= 5:
            journal(history, {"status": EvaluationStatus.EVALUATION_DEADLINE, "group": group, "name": name})
            break
        if adapter is not None and not (adapter / "adapter_model.safetensors").exists():
            journal(history, {"status": EvaluationStatus.MISSING_ADAPTER, "group": group, "name": name, "adapter": str(adapter)})
            continue
        transfer = group == EvaluationGroup.TRANSFER
        data = run / "eval_transfer/dataset.jsonl" if transfer else ORIGINAL / "dataset.jsonl"
        ids = run / "eval_transfer/test_ids.json" if transfer else ORIGINAL / "frozen_test_ids.json"
        if transfer and name == "icl":
            try:
                ids = transfer_icl_ids(run)
            except (OSError, ValueError) as error:
                journal(history, {"status": EvaluationStatus.FAILED, "group": group, "name": name, "error": str(error)})
                continue
        destination = run / f"eval_{group}/{name}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        module = "carl.icl.intent_in_context" if name == "icl" else "carl.icl.intent_generator_experiment"
        command = [sys.executable, "-m", module, "evaluate", "--data", str(data), "--output", str(destination),
                   "--model", str(MODEL), "--revision", REVISION, "--max-length", "768" if name == "icl" else "512",
                   "--max-new-tokens", "128", "--test-ids", str(ids)]
        if adapter is not None:
            command += ["--adapter", str(adapter)]
        if name == "icl":
            command += ["--support", str(ORIGINAL / "icl_followup/support.json"), "--support-data", str(ORIGINAL / "dataset.jsonl")]
        timeout = min(remaining - 2, 360 if transfer else 600)
        log_path = run / f"logs/eval_{group}_{name}.log"
        journal(history, {"status": EvaluationStatus.STARTED, "group": group, "name": name,
                          "command": command, "timeout_seconds": timeout})
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            try:
                code = process.wait(timeout=timeout)
                status = EvaluationStatus.COMPLETE if code == 0 else EvaluationStatus.FAILED
            except subprocess.TimeoutExpired:
                stop_process_tree(process)
                code, status = process.returncode, EvaluationStatus.TIMEOUT
        journal(history, {"status": status, "group": group, "name": name, "exit_code": code, "log": str(log_path)})
        print(json.dumps({"status": status, "group": group, "name": name}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
