#!/usr/bin/env python3
"""Sequentially execute a preregistered fixed-controller comparison."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import file_sha256
from scripts.run_lora_evaluations import stop_process_tree
from scripts.run_representation_matrix import ensure_supervised
from scripts.run_representation_matrix import training_running


class JobStatus(StrEnum):
    STARTED = "started"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"
    MISSING_GENERATION = "missing_generation"
    NO_VALID_INTENTS = "no_valid_intents"
    CUTOFF = "cutoff"


class InputMode(StrEnum):
    STRICT = "strict"
    FENCE = "fence"


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", buffering=1) as output:
        output.write(json.dumps({"epoch": time.time(), **event}) + "\n")
        output.flush()
        os.fsync(output.fileno())


def run_child(command: list[str], log_path: Path, timeout: float) -> tuple[JobStatus, int | None]:
    """Inherit the watchdog process group and stop only this child's tree."""
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n--- child started at {time.time():.3f}, timeout={timeout:.3f}s ---\n")
        log.flush()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        try:
            code = child.wait(timeout=timeout)
            return (JobStatus.COMPLETE if code == 0 else JobStatus.FAILED), code
        except subprocess.TimeoutExpired:
            stop_process_tree(child)
            return JobStatus.TIMEOUT, child.returncode


def remaining_seconds(manifest: dict[str, Any]) -> float:
    clock = manifest["clock"]
    # Preserve the final 30 minutes for reports and never start expensive
    # subprocesses after the previously fixed 3h45 start cutoff.
    return min(clock["deadline_epoch"] - 1800 - time.time(),
               clock["deadline_monotonic"] - 1800 - time.monotonic())


def start_allowed(manifest: dict[str, Any]) -> bool:
    return (time.time() < manifest["expensive_start_cutoff_epoch"]
            and time.monotonic() < manifest["expensive_start_cutoff_monotonic"]
            and remaining_seconds(manifest) > 10)


def verify_controller_assets(snapshot: Path, selection: Path) -> None:
    """Check all model assets frozen before the controller test."""
    value = json.loads(snapshot.read_text(encoding="utf-8"))
    selected = json.loads(selection.read_text(encoding="utf-8"))["selected_config"]
    model = Path(selected["model"])
    if value["model"] != str(model) or value["selection_sha256"] != file_sha256(selection):
        raise ValueError("Controller asset snapshot differs from the frozen selection")
    names = {"config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"}
    if set(value["files"]) != names:
        raise ValueError("Controller asset snapshot has unexpected files")
    for name in names:
        if file_sha256(model / name) != value["files"][name]["sha256"]:
            raise ValueError(f"Frozen controller asset changed: {name}")


def preserve_partial_tail(output: Path) -> None:
    """Quarantine an interrupted final JSONL line before the verified resume."""
    if not output.exists():
        return
    data = output.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    boundary = data.rfind(b"\n") + 1
    tail = data[boundary:]
    try:
        json.loads(tail)
    except (json.JSONDecodeError, UnicodeDecodeError):
        saved = output.with_name(f"{output.name}.partial-{hashlib.sha256(tail).hexdigest()[:12]}")
        if saved.exists() and saved.read_bytes() != tail:
            raise ValueError("Existing partial-output quarantine differs")
        if not saved.exists():
            saved.write_bytes(tail)
        data = data[:boundary]
    else:
        data += b"\n"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, output)


def verify_fence_plan(plan: dict[str, Any], strict: dict[str, Any], strict_hash: str) -> None:
    """Keep a post-hoc parser sensitivity analysis paired with its primary plan."""
    shared = (
        "frozen_ids", "frozen_ids_sha256", "controller_selection_sha256", "selected_controller",
        "primitive_action_budget", "variants", "decoding", "action_parser", "model_training", "api_calls",
    )
    if (
        plan.get("status") != "posthoc_exploratory"
        or plan.get("generator_output_normalisation") != "remove_one_outer_fence"
        or plan.get("primary_policy_plan_sha256") != strict_hash
        or strict.get("decoding") != "greedy"
        or strict.get("model_training") is not False
        or strict.get("api_calls") != 0
        or any(plan.get(key) != strict.get(key) for key in shared)
    ):
        raise ValueError("Post-hoc policy plan differs from the frozen primary comparison")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--timeout-per-variant", type=float, default=600)
    parser.add_argument("--mode", choices=list(InputMode), default=InputMode.STRICT)
    args = parser.parse_args()
    if args.timeout_per_variant <= 0:
        raise ValueError("Per-variant timeout must be positive")
    run = args.run.resolve()
    ensure_supervised(run)
    manifest = json.loads((run / "run_manifest.json").read_text())
    strict_plan_path = run / "policy_evaluation_plan.json"
    plan_path = run / ("policy_evaluation_fence_plan.json" if args.mode == InputMode.FENCE
                       else "policy_evaluation_plan.json")
    plan = json.loads(plan_path.read_text())
    if args.mode == InputMode.FENCE:
        if plan_path.with_suffix(".sha256").read_text().strip() != file_sha256(plan_path):
            raise ValueError("Post-hoc policy plan checksum differs")
        strict = json.loads(strict_plan_path.read_text())
        verify_fence_plan(plan, strict, file_sha256(strict_plan_path))
    selection = run / "controller_selection.json"
    if plan["controller_selection_sha256"] != file_sha256(selection):
        raise ValueError("The calibrated controller selection has changed")
    if json.loads(selection.read_text())["selected_config"] != plan["selected_controller"]:
        raise ValueError("Controller configuration differs from its frozen selection")
    assets = run / "controller_assets_snapshot.json"
    verify_controller_assets(assets, selection)
    data = run / "data/dataset.jsonl"
    ids = run / "controller_test_ids.json"
    matrix_plan = run / "matrix_plan.json"
    if plan["frozen_ids_sha256"] != file_sha256(ids):
        raise ValueError("The frozen controller contexts have changed")
    input_root = run / ("controller_fence_inputs" if args.mode == InputMode.FENCE else "controller_inputs")
    result_root = run / ("controller_fence_results" if args.mode == InputMode.FENCE else "controller_results")
    history = run / ("controller_fence_jobs.jsonl" if args.mode == InputMode.FENCE else "controller_jobs.jsonl")
    (run / "logs").mkdir(exist_ok=True)
    result_root.mkdir(exist_ok=True)
    lock_path = run / ".matrix_gpu.lock"
    failed = False
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another GPU stage holds the experiment lock") from exc
        for variant in plan["variants"]:
            if not start_allowed(manifest):
                append_event(history, {"variant": variant, "status": JobStatus.CUTOFF})
                failed = True
                break
            if training_running(run):
                raise RuntimeError("A registered training stage is still using the GPU")
            prediction = run / "predictions" / f"{variant}.jsonl"
            if not prediction.exists():
                append_event(history, {"variant": variant, "status": JobStatus.MISSING_GENERATION})
                failed = True
                continue
            input_dir = input_root / variant
            input_manifest = input_dir / "manifest.json"
            if input_manifest.exists():
                prior = json.loads(input_manifest.read_text())
                if (prior["predictions_sha256"] != file_sha256(prediction)
                        or prior["test_ids_sha256"] != file_sha256(ids)
                        or prior["data_sha256"] != file_sha256(data)
                        or prior["matrix_plan_sha256"] != file_sha256(matrix_plan)
                        or prior["variant"] != variant
                        or prior["records_sha256"] != file_sha256(input_dir / "records.jsonl")
                        or prior["eligibility_sha256"] != file_sha256(input_dir / "eligibility.jsonl")
                        or prior["prediction_run_sha256"] != file_sha256(prediction.with_suffix(".run.json"))
                        or prior["prompts_sha256"] != file_sha256(prediction.with_suffix(".prompts.jsonl"))):
                    raise ValueError("Prepared controller inputs differ from current frozen predictions")
                normalisation = prior.get("generator_output_normalisation")
                normaliser = Path(__file__).resolve().parents[1] / "carl/icl/intent_fence_audit.py"
                invalid_fence = args.mode == InputMode.FENCE and (
                    not isinstance(normalisation, dict)
                    or normalisation.get("mode") != "remove_one_outer_fence"
                    or normalisation.get("posthoc") is not True
                    or normalisation.get("raw_predictions_sha256") != file_sha256(prediction)
                    or normalisation.get("normaliser_source_sha256") != file_sha256(normaliser)
                )
                if invalid_fence or (args.mode == InputMode.STRICT and normalisation is not None):
                    raise ValueError("Prepared controller inputs use a different generator-output protocol")
            else:
                prepare = [sys.executable, "scripts/prepare_controller_intents.py", "--data", str(data), "--test-ids",
                           str(ids), "--predictions", str(prediction), "--matrix-plan", str(matrix_plan), "--variant",
                           variant, "--output-dir", str(input_dir)]
                if args.mode == InputMode.FENCE:
                    prepare.append("--remove-one-generator-fence")
                log = run / "logs" / f"controller_{args.mode}_{variant}_prepare.log"
                append_event(history, {"variant": variant, "phase": "prepare", "status": JobStatus.STARTED,
                                       "command": prepare, "log": str(log), "timeout_seconds": 60})
                status, code = run_child(prepare, log, min(60, remaining_seconds(manifest)))
                append_event(history, {"variant": variant, "phase": "prepare", "status": status, "exit_code": code})
                if status != JobStatus.COMPLETE:
                    failed = True
                    continue
            inputs = input_dir / "records.jsonl"
            if not inputs.read_text().strip():
                append_event(history, {"variant": variant, "status": JobStatus.NO_VALID_INTENTS})
                continue
            if not start_allowed(manifest):
                append_event(history, {"variant": variant, "status": JobStatus.CUTOFF})
                failed = True
                break
            output = result_root / f"{variant}.jsonl"
            summary = output.with_suffix(".summary.json")
            if summary.exists() and json.loads(summary.read_text()).get("status") == "complete":
                saved = json.loads(output.with_suffix(".manifest.json").read_text())
                finished = json.loads(summary.read_text())
                if (saved["config"] != plan["selected_controller"] or saved["records_sha256"] != file_sha256(inputs)
                        or saved["action_parser"] != plan["action_parser"]
                        or finished["configuration_sha256"] != saved["configuration_sha256"]):
                    raise ValueError("Completed controller output has different inputs or policy configuration")
                append_event(history, {"variant": variant, "status": JobStatus.COMPLETE, "reused": True})
                continue
            config = plan["selected_controller"]
            command = [sys.executable, "scripts/run_fixed_local_controller.py", "--records", str(inputs), "--output",
                       str(output), "--model", config["model"], "--revision", config["revision"], "--device", config["device"],
                       "--action-cap", str(config["action_cap"]), "--max-new-tokens", str(config["max_new_tokens"]),
                       "--max-context-tokens", str(config["max_context_tokens"])]
            if output.exists() or output.with_suffix(".manifest.json").exists():
                preserve_partial_tail(output)
                command.append("--resume")
            timeout = min(args.timeout_per_variant, remaining_seconds(manifest) - 2)
            if timeout <= 1:
                append_event(history, {"variant": variant, "status": JobStatus.CUTOFF})
                failed = True
                break
            log = run / "logs" / f"controller_{args.mode}_{variant}.log"
            append_event(history, {"variant": variant, "phase": "rollout", "status": JobStatus.STARTED,
                                   "command": command, "log": str(log), "timeout_seconds": timeout})
            status, code = run_child(command, log, timeout)
            verify_controller_assets(assets, selection)
            append_event(history, {"variant": variant, "phase": "rollout", "status": status, "exit_code": code})
            failed |= status != JobStatus.COMPLETE
            print(json.dumps({"variant": variant, "status": status, "exit_code": code}), flush=True)
    if args.mode == InputMode.STRICT and (run / "policy_evaluation_fence_plan.json").is_file():
        if start_allowed(manifest):
            command = [sys.executable, str(Path(__file__).resolve()), "--run", str(run), "--mode", InputMode.FENCE,
                       "--timeout-per-variant", str(args.timeout_per_variant)]
            timeout = max(1.0, remaining_seconds(manifest) - 2)
            log = run / "logs/controller_fence_secondary.log"
            append_event(history, {"phase": "posthoc_fence", "status": JobStatus.STARTED, "command": command,
                                   "log": str(log), "timeout_seconds": timeout})
            status, code = run_child(command, log, timeout)
            append_event(history, {"phase": "posthoc_fence", "status": status, "exit_code": code})
            failed |= status != JobStatus.COMPLETE
            assessment = run / "controller_assessment_fence"
            if assessment.exists() and not (assessment / "summary.json").is_file():
                assessment = run / f"controller_assessment_fence_recovery_{int(time.time())}"
            if (assessment / "summary.json").is_file():
                append_event(history, {"phase": "posthoc_assessment", "status": JobStatus.COMPLETE,
                                       "output": str(assessment), "reused": True})
            else:
                command = [sys.executable, "scripts/assess_fixed_controller.py", "--plan",
                           str(run / "policy_evaluation_fence_plan.json"), "--inputs",
                           str(run / "controller_fence_inputs"), "--results", str(run / "controller_fence_results"),
                           "--output", str(assessment)]
                available = min(manifest["clock"]["deadline_epoch"] - time.time(),
                                manifest["clock"]["deadline_monotonic"] - time.monotonic()) - 45
                if available > 5:
                    timeout = min(600.0, available)
                    log = run / "logs/controller_fence_assessment.log"
                    append_event(history, {"phase": "posthoc_assessment", "status": JobStatus.STARTED,
                                           "command": command, "log": str(log), "timeout_seconds": timeout})
                    assessed, assessed_code = run_child(command, log, timeout)
                    append_event(history, {"phase": "posthoc_assessment", "status": assessed,
                                           "exit_code": assessed_code, "output": str(assessment)})
                    failed |= assessed != JobStatus.COMPLETE
                else:
                    append_event(history, {"phase": "posthoc_assessment", "status": JobStatus.CUTOFF})
                    failed = True
        else:
            append_event(history, {"phase": "posthoc_fence", "status": JobStatus.CUTOFF})
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
