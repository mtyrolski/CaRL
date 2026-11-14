#!/usr/bin/env python3
"""Run frozen representation methods sequentially inside the campaign watchdog."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

import psutil

from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_representation_experiment import SupportSet
from carl.icl.intent_representation_experiment import load_supports
from carl.icl.intent_representation_experiment import messages_for
from carl.icl.intent_representation_experiment import parse_representation
from scripts.prepare_representation_matrix import Group
from scripts.prepare_representation_matrix import matrix_deadline
from scripts.prepare_representation_matrix import methods


class Status(StrEnum):
    STARTED = "started"
    COMPLETE = "complete"
    SKIPPED_COMPLETE = "skipped_complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MISSING_ADAPTER = "missing_adapter"
    DEADLINE = "deadline"
    GPU_BUSY = "gpu_busy"


def journal(path: Path, event: dict[str, Any]) -> None:
    """Persist each start/end status before beginning another method."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_plan(run: Path, plan: dict[str, Any]) -> None:
    """Refuse changed source, inputs, methods, base model, or inference settings."""
    if plan.get("schema_version") != 2 or plan.get("run") != str(run):
        raise ValueError("Matrix plan does not belong to this run")
    if plan.get("test_contexts") != 128:
        raise ValueError("This campaign requires all 128 frozen contexts")
    supervisor = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    latest_epoch, latest_monotonic = matrix_deadline(supervisor["clock"])
    if (plan["deadline_epoch"] > min(latest_epoch, supervisor["expensive_start_cutoff_epoch"])
            or plan["deadline_monotonic"] > min(latest_monotonic, supervisor["expensive_start_cutoff_monotonic"])):
        raise ValueError("Matrix deadline exceeds the frozen campaign limit")
    expected_settings = {"seed": 1729, "dtype": "bfloat16", "threads": 8, "max_length": 1024, "max_new_tokens": 128,
                         "decoding": "greedy", "enable_thinking": False, "split": "test", "require_cuda": True}
    if plan.get("settings") != expected_settings:
        raise ValueError("Inference settings differ from the frozen protocol")
    planned = plan.get("methods")
    if not isinstance(planned, list) or len(planned) != len(methods(run)):
        raise ValueError("Matrix method list has changed")
    for value, expected in zip(planned, methods(run), strict=True):
        if any(value.get(key) != getattr(expected, key)
               for key in ("name", "group", "representation", "support_set", "adapter", "adapter_pending")):
            raise ValueError("Matrix methods differ from the prespecified list")
        if (value.get("output") != str(run / "predictions" / f"{expected.name}.jsonl")
                or value.get("log") != str(run / "logs" / f"matrix_{expected.name}.log")):
            raise ValueError("Matrix output location differs from the prespecified plan")
    for name, metadata in plan["inputs"].items():
        path = Path(metadata["path"])
        if not path.is_file() or file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    model = Path(plan["model"]["path"])
    for name, digest in plan["model"]["sha256"].items():
        if file_sha256(model / name) != digest:
            raise ValueError(f"Frozen base model changed: {name}")


def ensure_supervised(run: Path) -> None:
    """Require this process to share a registered watchdog-owned session."""
    session = os.getsid(0)
    if os.getpgrp() != session:
        raise RuntimeError("The matrix must run in the supervisor's process group")
    state = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if state.get("status") != "active":
        raise RuntimeError("The run supervisor is not active")
    if not any(stage.get("pid") == session and stage.get("status") == "running" for stage in state["stages"]):
        raise RuntimeError("Current process group is not registered with the supervisor")


def training_running(run: Path) -> list[str]:
    state = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    return [stage["name"] for stage in state["stages"]
            if stage.get("kind") == "training" and stage.get("status") == "running"]


def stop_process_tree(process: subprocess.Popen[Any]) -> None:
    """Stop only this child and its descendants, keeping the watchdog's group intact."""
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        parent, children = None, []
    members = [*reversed(children), *([parent] if parent else [])]
    for member in members:
        try:
            member.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(members, timeout=3)
    for member in alive:
        try:
            member.kill()
        except psutil.NoSuchProcess:
            continue
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def current_adapter(job: dict[str, Any], run: Path) -> dict[str, str] | None:
    """Bind pending adapter hashes once and verify them on every resume."""
    if job["adapter"] is None:
        return None
    adapter = Path(job["adapter"])
    paths = ("adapter_model.safetensors", "adapter_config.json")
    if any(not (adapter / name).is_file() for name in paths):
        raise FileNotFoundError(f"Adapter unavailable: {adapter}")
    hashes = {name: file_sha256(adapter / name) for name in paths}
    if not job["adapter_pending"]:
        if hashes != job["adapter_sha256"]:
            raise ValueError("Existing adapter changed after matrix freeze")
        return hashes
    frozen = run / "matrix_adapter_hashes" / f"{job['name']}.json"
    encoded = json.dumps({"adapter": str(adapter), "sha256": hashes}, indent=2) + "\n"
    if frozen.exists():
        if frozen.read_text(encoding="utf-8") != encoded:
            raise ValueError("Representation adapter changed after its first evaluation")
    else:
        frozen.parent.mkdir(parents=True, exist_ok=True)
        temporary = frozen.with_name(f".{frozen.name}.{os.getpid()}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, frozen)
    return hashes


def completed_output(job: dict[str, Any], ids: list[str], plan: dict[str, Any],
                     adapter_hashes: dict[str, str] | None = None) -> bool:
    """A summary is complete only if all raw predictions and settings agree."""
    output = Path(job["output"])
    summary, metadata = output.with_suffix(".summary.json"), output.with_suffix(".run.json")
    prompts = output.with_suffix(".prompts.jsonl")
    if not all(path.is_file() for path in (output, summary, metadata, prompts)):
        return False
    info = json.loads(summary.read_text(encoding="utf-8"))
    config = json.loads(metadata.read_text(encoding="utf-8"))
    if info.get("status") != "complete" or info.get("n") != len(ids):
        return False
    expectations = {"variant": job["name"], "representation": job["representation"],
                    "support_set": job["support_set"], "data_sha256": plan["inputs"]["data"]["sha256"],
                    "test_ids_sha256": plan["inputs"]["test_ids"]["sha256"],
                    "support_data_sha256": plan["inputs"]["support_data"]["sha256"],
                    "support_manifest_sha256": plan["inputs"]["support_manifest"]["sha256"],
                    "protocol_sha256": plan["inputs"]["protocol"]["sha256"],
                    "generator_sha256": plan["inputs"]["generator"]["sha256"],
                    "support_selection_sha256": plan["inputs"]["support_selector"]["sha256"],
                    "prompts_sha256": file_sha256(prompts),
                    "model_weights_sha256": plan["model"]["sha256"]["model.safetensors"],
                    "model_config_sha256": plan["model"]["sha256"]["config.json"],
                    "tokenizer_sha256": plan["model"]["sha256"]["tokenizer.json"],
                    "model": plan["model"]["path"], "revision": plan["model"]["revision"],
                    "adapter": job["adapter"],
                    "adapter_weights_sha256": adapter_hashes["adapter_model.safetensors"] if adapter_hashes else None,
                    "adapter_config_sha256": adapter_hashes["adapter_config.json"] if adapter_hashes else None,
                    **plan["settings"]}
    if any(config.get(key) != value for key, value in expectations.items()):
        raise ValueError(f"Completed output settings differ for {job['name']}")
    rows = read_records(output)
    data = read_records(plan["inputs"]["data"]["path"], "test")
    by_id = {row["record_id"]: row for row in data}
    if len(by_id) != len(data) or any(identifier not in by_id for identifier in ids):
        raise ValueError("Frozen test IDs are missing or duplicated")
    if ([row.get("record_id") for row in rows] != ids
            or any(row.get("layout_id") != by_id[identifier]["layout_id"]
                   or row.get("variant") != job["name"]
                   or row.get("representation") != job["representation"]
                   or row.get("support_set") != job["support_set"]
                   or row.get("split") != "test" for row, identifier in zip(rows, ids, strict=True))):
        raise ValueError(f"Completed output records differ for {job['name']}")
    saved_prompts = read_records(prompts)
    support = load_supports(Path(plan["inputs"]["support_data"]["path"]),
                            Path(plan["inputs"]["support_manifest"]["path"]), SupportSet(job["support_set"]))
    if ([row.get("record_id") for row in saved_prompts] != ids
            or any(row.get("layout_id") != by_id[identifier]["layout_id"]
                   or row.get("messages") != messages_for(by_id[identifier], support, job["representation"])
                   for row, identifier in zip(saved_prompts, ids, strict=True))):
        raise ValueError("Completed output prompt messages differ from the frozen protocol")
    for row in rows:
        raw = row.get("raw_output")
        if not isinstance(raw, str):
            raise ValueError("Completed output lacks raw text")
        try:
            parsed, valid = parse_representation(job["representation"], raw), True
        except ValueError:
            parsed, valid = None, False
        if row.get("syntax_valid") is not valid or row.get("parsed_intent") != parsed:
            raise ValueError("Completed output parse differs from raw text")
    return True


def command_for(job: dict[str, Any], plan: dict[str, Any], resume: bool) -> list[str]:
    """Construct an explicit, identical-input command for one frozen variant."""
    settings = plan["settings"]
    command = [sys.executable, "-m", "carl.icl.intent_representation_experiment", "evaluate",
               "--data", plan["inputs"]["data"]["path"], "--test-ids", plan["inputs"]["test_ids"]["path"],
               "--support-data", plan["inputs"]["support_data"]["path"],
               "--support-manifest", plan["inputs"]["support_manifest"]["path"],
               "--model", plan["model"]["path"], "--revision", plan["model"]["revision"],
               "--output", job["output"], "--variant", job["name"], "--representation", job["representation"],
               "--support-set", job["support_set"], "--split", settings["split"], "--dtype", settings["dtype"],
               "--seed", str(settings["seed"]), "--threads", str(settings["threads"]),
               "--max-length", str(settings["max_length"]),
               "--max-new-tokens", str(settings["max_new_tokens"]), "--require-cuda",
               "--stop-epoch", str(plan["deadline_epoch"]), "--stop-monotonic", str(plan["deadline_monotonic"])]
    if job["adapter"] is not None:
        command.extend(("--adapter", job["adapter"]))
    if resume:
        command.append("--resume")
    return command


def run_group(run: Path, plan_path: Path, group: Group, job_timeout: float) -> int:
    if job_timeout <= 0:
        raise ValueError("Each subprocess timeout must be positive")
    ensure_supervised(run)
    checksum = plan_path.with_suffix(".sha256")
    if not checksum.is_file() or checksum.read_text(encoding="utf-8").strip() != file_sha256(plan_path):
        raise ValueError("Matrix plan checksum is missing or does not match")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verify_plan(run, plan)
    ids = json.loads(Path(plan["inputs"]["test_ids"]["path"]).read_text(encoding="utf-8"))["record_ids"]
    history = run / "matrix_jobs.jsonl"
    selected = [job for job in plan["methods"] if job["group"] == group]
    if not selected:
        raise ValueError("No methods in selected group")
    lock_path = run / ".matrix_gpu.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            journal(history, {"status": Status.GPU_BUSY, "group": group, "reason": "another matrix owns GPU lock"})
            return 1
        failed = False
        for job in selected:
            remaining = min(plan["deadline_epoch"] - time.time(), plan["deadline_monotonic"] - time.monotonic())
            if remaining <= 3:
                journal(history, {"status": Status.DEADLINE, "group": group, "name": job["name"]})
                failed = True
                continue
            active_training = training_running(run)
            if active_training:
                journal(history, {"status": Status.GPU_BUSY, "group": group, "name": job["name"],
                                  "training_stages": active_training})
                failed = True
                break
            try:
                adapter_hashes = current_adapter(job, run)
                if completed_output(job, ids, plan, adapter_hashes):
                    journal(history, {"status": Status.SKIPPED_COMPLETE, "group": group, "name": job["name"]})
                    continue
            except FileNotFoundError as exc:
                journal(history, {"status": Status.MISSING_ADAPTER, "group": group,
                                  "name": job["name"], "error": str(exc)})
                failed = True
                continue
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                journal(history, {"status": Status.FAILED, "group": group, "name": job["name"], "error": str(exc)})
                failed = True
                continue
            output = Path(job["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            log = Path(job["log"])
            log.parent.mkdir(parents=True, exist_ok=True)
            resume = output.exists() or output.with_suffix(".run.json").exists()
            command = command_for(job, plan, resume)
            timeout = min(job_timeout, max(1.0, remaining - 2))
            event = {"status": Status.STARTED, "group": group, "name": job["name"], "command": command,
                     "resume": resume, "adapter_sha256": adapter_hashes, "timeout_seconds": timeout,
                     "output": str(output), "log": str(log), "started_epoch": time.time()}
            journal(history, event)
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false"}
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"\n--- {job['name']} resume={resume} ---\n")
                stream.flush()
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                try:
                    code = process.wait(timeout=timeout)
                    status = Status.COMPLETE if code == 0 else Status.FAILED
                except subprocess.TimeoutExpired:
                    code = process.poll()
                    if code is not None:
                        status = Status.COMPLETE if code == 0 else Status.FAILED
                    else:
                        stop_process_tree(process)
                        code, status = process.returncode, Status.TIMEOUT
            if status == Status.COMPLETE:
                try:
                    if not completed_output(job, ids, plan, adapter_hashes):
                        status = Status.FAILED
                except (ValueError, OSError, json.JSONDecodeError):
                    status = Status.FAILED
            journal(history, {"status": status, "group": group, "name": job["name"],
                              "exit_code": code, "output": str(output), "log": str(log),
                              "finished_epoch": time.time()})
            print(json.dumps({"status": status, "group": group, "name": job["name"]}), flush=True)
            failed |= status != Status.COMPLETE
        fcntl.flock(lock, fcntl.LOCK_UN)
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--group", choices=list(Group), required=True)
    parser.add_argument("--job-timeout", type=float, default=1200)
    args = parser.parse_args()
    run = args.run.resolve()
    raise SystemExit(run_group(run, args.plan or run / "matrix_plan.json", Group(args.group), args.job_timeout))


if __name__ == "__main__":
    main()
