#!/usr/bin/env python3
"""Schedule the fixed-controller LoRA addendum under the existing watchdog.

The coordinator waits for the campaign's first CPU assessment, which is
registered after its GPU matrix stages.  It then registers a separate GPU
stage, followed by CPU assessment and notebook stages.  Each stage and every
child has a bounded timeout; no user process is addressed by name.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from datetime import timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import file_sha256
from scripts.guard_registered_wrapper import pidfd_alive
from scripts.guard_registered_wrapper import watchdog_visible
from scripts.intent_experiment_supervisor import StageStatus
from scripts.intent_experiment_supervisor import append_log
from scripts.intent_experiment_supervisor import boot_id
from scripts.intent_experiment_supervisor import locked
from scripts.intent_experiment_supervisor import owned_process
from scripts.intent_experiment_supervisor import proc_start
from scripts.intent_experiment_supervisor import update_report
from scripts.intent_experiment_supervisor import write_json
from scripts.run_controller_matrix import JobStatus
from scripts.run_controller_matrix import append_event
from scripts.run_controller_matrix import preserve_partial_tail
from scripts.run_controller_matrix import remaining_seconds
from scripts.run_controller_matrix import run_child
from scripts.run_controller_matrix import start_allowed
from scripts.run_controller_matrix import verify_controller_assets
from scripts.run_representation_campaign import ensure_supervised
from scripts.run_representation_matrix import ensure_supervised as ensure_registered
from scripts.run_representation_matrix import training_running

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/intent_experiment_supervisor.py"
PLAN_NAME = "policy_lora_addendum_plan.json"
INPUTS_NAME = "controller_lora_addendum_inputs"
RESULTS_NAME = "controller_lora_addendum_results"
ASSESSMENT_NAME = "controller_lora_addendum_assessment"
WORK_STAGE = "controller_lora_addendum_rollout"
ASSESS_STAGE = "controller_lora_addendum_assessment"
NOTEBOOK_STAGE = "controller_lora_addendum_notebook"
VARIANTS = tuple(name for representation in ("json", "formal", "natural")
                 for name in (f"base_{representation}_zero", f"rep_lora_{representation}_zero"))


class QueueDecision(StrEnum):
    WAIT = "wait"
    START = "start"
    SKIP = "skip"


class AddendumStatus(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def record_status(run: Path, status: AddendumStatus, **details: Any) -> None:
    path = run / "controller_lora_addendum_status.json"
    previous = read_json(path) if path.exists() else {}
    atomic_json(path, {**previous, **details, "status": status, "updated_utc": datetime.now(timezone.utc).isoformat()})


def journal(run: Path, **event: Any) -> None:
    append_event(run / "controller_lora_addendum_jobs.jsonl", event)
    print(json.dumps(event, ensure_ascii=False), flush=True)


def validate_plan(run: Path, root: Path = ROOT) -> dict[str, Any]:
    """Require the six planned variants and the original calibrated controller."""
    path = run / PLAN_NAME
    if path.with_suffix(".sha256").read_text(encoding="utf-8").strip() != file_sha256(path):
        raise ValueError("LoRA addendum plan checksum differs")
    plan = read_json(path)
    strict_path = run / "policy_evaluation_plan.json"
    selection_path = run / "controller_selection.json"
    calibration_path = run / "policy_calibration_05b.manifest.json"
    assets_path = run / "controller_assets_snapshot.json"
    matrix_path = run / "matrix_plan.json"
    paths = ((strict_path, "original_policy_plan_sha256"),
             (selection_path, "controller_selection_sha256"),
             (calibration_path, "controller_calibration_manifest_sha256"),
             (assets_path, "controller_assets_snapshot_sha256"), (matrix_path, "matrix_plan_sha256"))
    if any(file_sha256(source) != plan.get(key) for source, key in paths):
        raise ValueError("A frozen controller or matrix input differs from the addendum plan")
    if matrix_path.with_suffix(".sha256").read_text(encoding="utf-8").strip() != file_sha256(matrix_path):
        raise ValueError("Matrix plan checksum differs")
    strict = read_json(strict_path)
    calibration = read_json(calibration_path)
    selection = read_json(selection_path)
    shared = ("frozen_ids", "frozen_ids_sha256", "controller_selection_sha256", "selected_controller",
              "decoding", "action_parser", "model_training", "primitive_action_budget", "api_calls")
    if (any(plan.get(key) != strict.get(key) for key in shared)
            or plan.get("status") != "posthoc_icl_prior_to_matched_lora_predictions"
            or plan.get("generator_output_normalisation") != "none"
            or plan.get("primitive_action_budget") != 16 or plan.get("model_training") is not False
            or plan.get("api_calls") != 0 or plan.get("decoding") != "greedy"
            or tuple(plan.get("variants", [])) != VARIANTS
            or plan.get("contrasts") != [[f"base_{rep}_zero", f"rep_lora_{rep}_zero"]
                                         for rep in ("json", "formal", "natural")]
            or plan.get("selected_controller") != selection.get("selected_config")
            or plan.get("selected_controller") != calibration.get("config")
            or plan.get("action_parser") != calibration.get("action_parser")
            or plan.get("controller_source_sha256") != calibration.get("source_sha256")):
        raise ValueError("Addendum differs from its fixed controller protocol")
    ids = root / plan["frozen_ids"]
    frozen = read_json(ids)
    if file_sha256(ids) != plan["frozen_ids_sha256"] or len(set(frozen["record_ids"])) != 40:
        raise ValueError("The 40 frozen controller contexts changed")
    if len(frozen["record_ids"]) != 40:
        raise ValueError("The controller IDs contain duplicates")
    for relative, digest in calibration["source_sha256"].items():
        if file_sha256(root / relative) != digest:
            raise ValueError(f"Calibrated controller source changed: {relative}")
    methods = {item["name"]: item for item in read_json(matrix_path)["methods"]}
    for representation in ("json", "formal", "natural"):
        for name, adapted in ((f"base_{representation}_zero", False), (f"rep_lora_{representation}_zero", True)):
            method = methods[name]
            if (method.get("representation") != representation or method.get("support_set") != "zero"
                    or (method.get("adapter") is not None) != adapted):
                raise ValueError(f"Frozen matched method differs: {name}")
    verify_controller_assets(assets_path, selection_path)
    return plan


def queue_decision(manifest: dict[str, Any]) -> QueueDecision:
    """Wait for the campaign's first CPU assessment and an idle GPU."""
    if manifest.get("status") != "active" or not start_allowed(manifest):
        return QueueDecision.SKIP
    if not manifest.get("boot_id") or manifest["boot_id"] != boot_id():
        return QueueDecision.SKIP
    stages = manifest.get("stages", [])
    masters = [stage for stage in stages if stage.get("name") == "representation_campaign"]
    if len(masters) != 1 or not valid_campaign_stage(masters[0], manifest["boot_id"], "run_representation_campaign.py"):
        return QueueDecision.SKIP
    quick = [stage for stage in stages if stage.get("name") == "campaign_assessment_quick"]
    if len(quick) > 1 or (quick and not valid_campaign_stage(quick[0], manifest["boot_id"],
                                                              "assess_intent_representations.py")):
        return QueueDecision.SKIP
    if not quick and masters[0].get("status") != "running":
        return QueueDecision.SKIP
    gpu_busy = any(stage.get("status") == "running" and stage.get("kind") in ("training", "expensive")
                   for stage in stages)
    return QueueDecision.START if quick and not gpu_busy else QueueDecision.WAIT


def watchdog_alive(manifest: dict[str, Any]) -> bool:
    return (manifest.get("status") == "active" and manifest.get("boot_id") == boot_id()
            and watchdog_visible(manifest))


def process_absent(pid: int) -> bool:
    """Check kernel PID visibility without signalling a process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def process_group_absent(pid: int) -> bool:
    """Check the complete host process group without relying on /proc enumeration."""
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def valid_campaign_stage(stage: dict[str, Any], expected_boot: str, script: str) -> bool:
    """Do not trust a marker without the supervisor's process identity."""
    pid, start = stage.get("pid"), stage.get("proc_start")
    command = stage.get("command")
    if (stage.get("kind") != "normal" or stage.get("boot_id") != expected_boot
            or not isinstance(pid, int) or pid <= 0 or not isinstance(start, int) or start <= 0
            or stage.get("status") not in set(StageStatus) or not isinstance(command, list)
            or not any(Path(str(arg)).name == script for arg in command)):
        return False
    return stage.get("status") != "running" or (owned_process(stage) and pidfd_alive(pid))


def final_notebook_decision(manifest: dict[str, Any]) -> QueueDecision:
    """Build last only after the master and its final notebook have exited."""
    if manifest.get("status") != "active" or manifest.get("boot_id") != boot_id():
        return QueueDecision.SKIP
    stages = manifest.get("stages", [])
    masters = [stage for stage in stages if stage.get("name") == "representation_campaign"]
    finishers = [stage for stage in stages if stage.get("name") == "campaign_notebook_finish_request"]
    if len(masters) != 1 or len(finishers) > 1:
        return QueueDecision.SKIP
    master = masters[0]
    if not valid_campaign_stage(master, manifest["boot_id"], "run_representation_campaign.py"):
        return QueueDecision.SKIP
    if not finishers:
        if master.get("status") == "running":
            return QueueDecision.WAIT
        if (master.get("status") in ("failed", "timeout", "deadline_stopped", "lost_or_exited")
                and process_absent(master["pid"]) and process_group_absent(master["pid"])):
            return QueueDecision.START
        return QueueDecision.SKIP
    finisher = finishers[0]
    if not valid_campaign_stage(finisher, manifest["boot_id"], "build_representation_report.py"):
        return QueueDecision.SKIP
    if master.get("status") == "running" or finisher.get("status") == "running":
        return QueueDecision.WAIT
    if any(not process_absent(stage["pid"]) or not process_group_absent(stage["pid"])
           for stage in (master, finisher)):
        return QueueDecision.SKIP
    return QueueDecision.START


def verify_prepared(run: Path, variant: str, prediction: Path, directory: Path) -> list[dict[str, Any]]:
    """Reject stale inputs, unrecorded normalisation and changed provenance."""
    manifest = read_json(directory / "manifest.json")
    checks = {"predictions_sha256": prediction, "test_ids_sha256": run / "controller_test_ids.json",
              "data_sha256": run / "data/dataset.jsonl", "matrix_plan_sha256": run / "matrix_plan.json",
              "records_sha256": directory / "records.jsonl", "eligibility_sha256": directory / "eligibility.jsonl",
              "prediction_run_sha256": prediction.with_suffix(".run.json"),
              "prompts_sha256": prediction.with_suffix(".prompts.jsonl")}
    if (manifest.get("variant") != variant or manifest.get("planned") != 40
            or manifest.get("generator_output_normalisation") is not None
            or any(manifest.get(key) != file_sha256(path) for key, path in checks.items())):
        raise ValueError("Prepared inputs differ from the frozen strict protocol")
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    eligibility = [json.loads(line) for line in (directory / "eligibility.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    frozen = read_json(run / "controller_test_ids.json")["record_ids"]
    if ([row["record_id"] for row in eligibility] != frozen
            or [row["record_id"] for row in eligibility if row["status"] == "ready"]
            != [row["record_id"] for row in records]):
        raise ValueError("Prepared eligibility does not match the frozen 40 contexts")
    return records


def generation_completeness(run: Path, prediction: Path, directory: Path) -> dict[str, Any]:
    """Distinguish partial generation from invalid model output on observed IDs."""
    matrix = read_json(run / "matrix_plan.json")
    expected_ids = read_json(Path(matrix["inputs"]["test_ids"]["path"]))["record_ids"]
    observed = [json.loads(line)["record_id"] for line in prediction.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    summary_path = prediction.with_suffix(".summary.json")
    summary = read_json(summary_path) if summary_path.is_file() else {}
    partial_path = prediction.with_suffix(".partial-summary.json")
    partial = read_json(partial_path) if partial_path.is_file() else {}
    counts = read_json(directory / "manifest.json").get("counts") or {}
    complete = (len(observed) == len(expected_ids) == len(set(observed))
                and set(observed) == set(expected_ids) and summary.get("status") == "complete"
                and summary.get("n") == len(observed))
    return {"complete": complete, "observed": len(observed), "planned": len(expected_ids),
            "summary_status": summary.get("status"), "partial_reason": partial.get("reason"),
            "selected_missing": counts.get("missing_generation", 0)}


def assessment_completeness(run: Path) -> dict[str, Any]:
    """Check the assessment's contents, not merely its process exit code."""
    directory = run / ASSESSMENT_NAME
    try:
        marker = read_json(directory / "complete.json")
        summary_path = directory / "summary.json"
        summary = read_json(summary_path)
        config = summary["config"]
        rows = [row for row in summary["results"] if row.get("family") == "all"]
        known_statuses = {"invalid_representation", "invalid_intent", "achieved", "preservation_violated",
                          "task_solved", "action_limit", "invalid_response"}
        valid_rows = (len(rows) == len(VARIANTS) and {row.get("variant") for row in rows} == set(VARIANTS)
                      and all(row.get("planned") == 40 and row.get("pipeline_evaluated") == 40
                              and isinstance(row.get("statuses"), dict)
                              and set(row["statuses"]).issubset(known_statuses)
                              and all(isinstance(value, int) and value >= 0 for value in row["statuses"].values())
                              and sum(row["statuses"].values()) == 40 for row in rows))
        plan_hash = file_sha256(run / PLAN_NAME)
        complete = (marker.get("status") == "complete" and marker.get("analysis_kind") == "exploratory_addendum"
                    and marker.get("summary_sha256") == file_sha256(summary_path)
                    and marker.get("plan_sha256") == config.get("plan_sha256") == plan_hash
                    and config.get("analysis_kind") == "exploratory_addendum"
                    and config.get("generator_output_normalisation") == "none"
                    and config.get("action_cap") == 16 and summary.get("rejected_variants") == [] and valid_rows)
        return {"complete": complete, "reason": None if complete else "missing_rejected_or_unverified_results"}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"complete": False, "reason": f"{type(error).__name__}: {error}"}


def verify_result(run: Path, plan: dict[str, Any], inputs: Path, output: Path) -> None:
    manifest = read_json(output.with_suffix(".manifest.json"))
    summary = read_json(output.with_suffix(".summary.json"))
    calibration = read_json(run / "policy_calibration_05b.manifest.json")
    expected_ids = [json.loads(line)["record_id"] for line in inputs.read_text(encoding="utf-8").splitlines() if line.strip()]
    observed_ids = [json.loads(line)["record_id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if (manifest.get("config") != plan["selected_controller"]
            or manifest.get("source_sha256") != plan["controller_source_sha256"]
            or manifest.get("weights_sha256") != calibration["weights_sha256"]
            or manifest.get("tokenizer_sha256") != calibration["tokenizer_sha256"]
            or manifest.get("action_parser") != plan["action_parser"]
            or manifest.get("records_sha256") != file_sha256(inputs)
            or manifest.get("record_ids") != expected_ids or observed_ids != expected_ids
            or summary.get("configuration_sha256") != manifest.get("configuration_sha256")
            or summary.get("status") != "complete" or summary.get("records") != len(expected_ids)):
        raise ValueError("Controller results differ from the frozen calibrated policy")


def worker(run: Path, timeout_per_variant: float) -> int:
    ensure_registered(run)
    manifest = read_json(run / "run_manifest.json")
    if not any(stage.get("pid") == os.getpgrp() and stage.get("kind") == "expensive"
               and stage.get("status") == "running" for stage in manifest.get("stages", [])):
        raise RuntimeError("Controller rollout requires its own registered expensive stage")
    plan = validate_plan(run)
    assets, selection = run / "controller_assets_snapshot.json", run / "controller_selection.json"
    data, ids, matrix = run / "data/dataset.jsonl", run / "controller_test_ids.json", run / "matrix_plan.json"
    inputs_root, results_root = run / INPUTS_NAME, run / RESULTS_NAME
    results_root.mkdir(exist_ok=True)
    failed = False
    with (run / ".matrix_gpu.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another stage holds the campaign GPU lock") from exc
        for variant in VARIANTS:
            manifest = read_json(run / "run_manifest.json")
            if not watchdog_alive(manifest):
                journal(run, variant=variant, status=AddendumStatus.FAILED, reason="watchdog_unverified")
                failed = True
                break
            if not start_allowed(manifest):
                journal(run, variant=variant, status=JobStatus.CUTOFF)
                failed = True
                continue
            if training_running(run):
                raise RuntimeError("A registered training stage is using the GPU")
            prediction = run / "predictions" / f"{variant}.jsonl"
            if not all(path.is_file() for path in (prediction, prediction.with_suffix(".run.json"),
                                                   prediction.with_suffix(".prompts.jsonl"))):
                journal(run, variant=variant, status=JobStatus.MISSING_GENERATION)
                failed = True
                continue
            input_dir = inputs_root / variant
            if not (input_dir / "manifest.json").exists():
                prepare = [sys.executable, str(ROOT / "scripts/prepare_controller_intents.py"), "--data", str(data),
                           "--test-ids", str(ids), "--predictions", str(prediction), "--matrix-plan", str(matrix),
                           "--variant", variant, "--output-dir", str(input_dir)]
                timeout = min(60.0, remaining_seconds(manifest) - 2)
                if timeout <= 1:
                    journal(run, variant=variant, status=JobStatus.CUTOFF)
                    failed = True
                    continue
                journal(run, variant=variant, phase="prepare", status=JobStatus.STARTED, command=prepare,
                        timeout_seconds=timeout)
                status, code = run_child(prepare, run / "logs" / f"controller_lora_addendum_{variant}_prepare.log", timeout)
                journal(run, variant=variant, phase="prepare", status=status, exit_code=code)
                if status != JobStatus.COMPLETE:
                    failed = True
                    continue
            records = verify_prepared(run, variant, prediction, input_dir)
            generation = generation_completeness(run, prediction, input_dir)
            if not generation["complete"]:
                journal(run, variant=variant, phase="generation", status=AddendumStatus.PARTIAL, **generation)
                failed = True
            if not records:
                journal(run, variant=variant, status=JobStatus.MISSING_GENERATION
                        if generation["selected_missing"] == 40 else JobStatus.NO_VALID_INTENTS)
                continue
            manifest = read_json(run / "run_manifest.json")
            if not watchdog_alive(manifest):
                journal(run, variant=variant, status=AddendumStatus.FAILED, reason="watchdog_unverified")
                failed = True
                break
            if not start_allowed(manifest):
                journal(run, variant=variant, status=JobStatus.CUTOFF)
                failed = True
                continue
            validate_plan(run)
            output, inputs = results_root / f"{variant}.jsonl", input_dir / "records.jsonl"
            if output.with_suffix(".summary.json").is_file():
                verify_result(run, plan, inputs, output)
                journal(run, variant=variant, status=JobStatus.COMPLETE, reused=True)
                continue
            config = plan["selected_controller"]
            command = [sys.executable, str(ROOT / "scripts/run_fixed_local_controller.py"), "--records", str(inputs),
                       "--output", str(output), "--model", config["model"], "--revision", config["revision"],
                       "--device", config["device"], "--action-cap", str(config["action_cap"]), "--max-new-tokens",
                       str(config["max_new_tokens"]), "--max-context-tokens", str(config["max_context_tokens"])]
            if output.exists() or output.with_suffix(".manifest.json").exists():
                preserve_partial_tail(output)
                command.append("--resume")
            timeout = min(timeout_per_variant, remaining_seconds(manifest) - 2)
            if timeout <= 1:
                journal(run, variant=variant, status=JobStatus.CUTOFF)
                failed = True
                continue
            journal(run, variant=variant, phase="rollout", status=JobStatus.STARTED, command=command,
                    timeout_seconds=timeout)
            status, code = run_child(command, run / "logs" / f"controller_lora_addendum_{variant}.log", timeout,
                                     pass_fds=(lock.fileno(),))
            verify_controller_assets(assets, selection)
            validate_plan(run)
            journal(run, variant=variant, phase="rollout", status=status, exit_code=code)
            if status == JobStatus.COMPLETE:
                verify_result(run, plan, inputs, output)
            else:
                failed = True
    return 1 if failed else 0


def mark_lost_stage(run: Path, name: str, wrapper_pid: int, wrapper_start: int | None) -> bool:
    """Record loss only after repeated checks and a final supervisor-locked check."""
    if wrapper_pid <= 0 or wrapper_start is None or wrapper_start <= 0:
        return False

    def absent(stage: dict[str, Any], state: dict[str, Any]) -> bool:
        pid, start = stage.get("pid"), stage.get("proc_start")
        return (watchdog_alive(state) and state.get("boot_id") == boot_id()
                and stage.get("boot_id") == state.get("boot_id") and stage.get("status") == "running"
                and isinstance(pid, int) and pid > 0 and isinstance(start, int) and start > 0
                and process_absent(wrapper_pid) and proc_start(wrapper_pid) is None
                and process_absent(pid) and proc_start(pid) is None and process_group_absent(pid))

    identity: tuple[int, int] | None = None
    for _ in range(3):
        state = read_json(run / "run_manifest.json")
        matches = [stage for stage in state.get("stages", []) if stage.get("name") == name]
        if len(matches) != 1 or not absent(matches[0], state):
            return False
        current = (matches[0]["pid"], matches[0]["proc_start"])
        if identity is not None and current != identity:
            return False
        identity = current
        time.sleep(.2)
    with locked(run):
        state = read_json(run / "run_manifest.json")
        matches = [stage for stage in state.get("stages", []) if stage.get("name") == name]
        if (len(matches) != 1 or identity != (matches[0].get("pid"), matches[0].get("proc_start"))
                or not absent(matches[0], state)):
            return False
        stage = matches[0]
        stage["status"] = StageStatus.LOST_OR_EXITED
        stage["finished_utc"] = datetime.now(timezone.utc).isoformat()
        stage["exit_code"] = None
        write_json(run / "run_manifest.json", state)
        update_report(run, state)
        append_log(run, f"Guardian recorded {name} as lost_or_exited; exit code is unknown.")
    return True


def run_stage(run: Path, name: str, kind: str, command: list[str], timeout: float) -> str:
    """Register a stage and wait for the supervisor, preserving its log."""
    state = read_json(run / "run_manifest.json")
    existing = [stage for stage in state.get("stages", []) if stage.get("name") == name]
    if existing:
        raise ValueError(f"Stage already registered: {name}")
    if name == WORK_STAGE:
        try:
            ensure_supervised(run, state)
        except RuntimeError as error:
            journal(run, stage=name, status="watchdog_unverified", error=str(error))
            return "watchdog_unverified"
        if not watchdog_alive(state) or queue_decision(state) != QueueDecision.START:
            journal(run, stage=name, status="watchdog_or_trigger_unverified")
            return "watchdog_or_trigger_unverified"
    invocation = [sys.executable, str(SUPERVISOR), "run", "--name", name, "--kind", kind,
                  "--timeout", str(timeout), str(run), "--", *command]
    log_path = run / "logs" / f"{name}_supervisor.log"
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false", "MPLCONFIGDIR": "/tmp/carl-repr-mpl",
                   "IPYTHONDIR": "/tmp/carl-repr-ipython"}
    journal(run, stage=name, status=AddendumStatus.RUNNING, command=command, timeout_seconds=timeout)
    with log_path.open("a", encoding="utf-8") as log:
        wrapper = subprocess.Popen(invocation, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, env=environment)
        wrapper_start = proc_start(wrapper.pid)
        journal(run, stage=name, phase="supervisor", wrapper_pid=wrapper.pid, wrapper_proc_start=wrapper_start)
        wrapper_limit = time.monotonic() + timeout + 30
        heartbeat = time.monotonic() + 300
        while wrapper.poll() is None and time.monotonic() < wrapper_limit:
            if time.monotonic() >= heartbeat:
                journal(run, stage=name, status=AddendumStatus.WAITING, reason="waiting_for_supervisor")
                heartbeat = time.monotonic() + 300
            time.sleep(2)
        if wrapper.poll() is None:
            if wrapper_start is not None and proc_start(wrapper.pid) == wrapper_start:
                wrapper.terminate()
                try:
                    wrapper.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if proc_start(wrapper.pid) == wrapper_start:
                        wrapper.kill()
                    try:
                        wrapper.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            if wrapper.poll() is None:
                journal(run, stage=name, status="running_unresolved", reason="supervisor_wrapper_not_stopped")
                return "running_unresolved"
        code = wrapper.returncode
    # A lost launcher must not be mistaken for a finished supervised child.
    heartbeat = 0.0
    while True:
        state = read_json(run / "run_manifest.json")
        stage = next((item for item in state.get("stages", []) if item.get("name") == name), None)
        if stage is None or stage.get("status") != "running" or state.get("status") != "active":
            break
        if mark_lost_stage(run, name, wrapper.pid, wrapper_start):
            state = read_json(run / "run_manifest.json")
            stage = next(item for item in state["stages"] if item.get("name") == name)
            break
        if (state.get("boot_id") != boot_id() or stage.get("boot_id") != state.get("boot_id")
                or time.monotonic() >= stage["stop_monotonic"] + 20
                or time.time() >= stage["stop_epoch"] + 20):
            journal(run, stage=name, status="running_unresolved", reason="stage_identity_or_stop_unverified")
            return "running_unresolved"
        if time.monotonic() >= heartbeat:
            journal(run, stage=name, status=AddendumStatus.WAITING, reason="supervised_stage_still_running")
            heartbeat = time.monotonic() + 300
        time.sleep(2)
    status = str(stage.get("status")) if stage else "not_registered"
    journal(run, stage=name, status=status, supervisor_exit_code=code)
    return status


def coordinator(run: Path, timeout_per_variant: float) -> int:
    state = read_json(run / "run_manifest.json")
    ensure_supervised(run, state)
    validate_plan(run)
    record_status(run, AddendumStatus.WAITING, plan_sha256=file_sha256(run / PLAN_NAME),
                  interpretation="Exploratory addendum selected after ICL, before new LoRA predictions.")
    heartbeat = 0.0
    while True:
        state = read_json(run / "run_manifest.json")
        decision = queue_decision(state)
        if decision != QueueDecision.WAIT:
            break
        if time.monotonic() >= heartbeat:
            journal(run, status=AddendumStatus.WAITING, reason="waiting_for_campaign_assessment_or_gpu",
                    seconds_before_reporting_reserve=remaining_seconds(state))
            heartbeat = time.monotonic() + 300
        time.sleep(2)
    outcome = "skipped_cutoff_or_unverified_trigger"
    if decision == QueueDecision.START:
        record_status(run, AddendumStatus.RUNNING)
        timeout = min(3600.0, remaining_seconds(state) - 5)
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--run", str(run),
                   "--timeout-per-variant", str(timeout_per_variant)]
        if timeout > 5:
            outcome = run_stage(run, WORK_STAGE, "expensive", command, timeout)
    record_status(run, AddendumStatus.RUNNING, rollout_stage_status=outcome)
    if outcome == "running_unresolved":
        record_status(run, AddendumStatus.PARTIAL, assessment_stage_status="skipped_unresolved_rollout",
                      notebook_stage_status="skipped_unresolved_rollout")
        return 1
    state = read_json(run / "run_manifest.json")
    available = min(state["clock"]["deadline_epoch"] - time.time(),
                    state["clock"]["deadline_monotonic"] - time.monotonic()) - 30
    assessment = "not_started"
    if state.get("status") == "active" and available > 10:
        command = [sys.executable, str(ROOT / "scripts/assess_controller_lora_addendum.py"), "--run", str(run)]
        assessment = run_stage(run, ASSESS_STAGE, "normal", command, min(600.0, available))
    data_assessment = assessment_completeness(run)
    record_status(run, AddendumStatus.RUNNING, assessment_stage_status=assessment,
                  assessment_data_complete=data_assessment["complete"],
                  assessment_data_reason=data_assessment["reason"])
    if assessment == "running_unresolved":
        record_status(run, AddendumStatus.PARTIAL, notebook_stage_status="skipped_unresolved_assessment")
        return 1
    # The master's last notebook stage must not overwrite the addendum report.
    # Keep this independently supervised group alive until the master exits.
    heartbeat = 0.0
    while True:
        state = read_json(run / "run_manifest.json")
        available = min(state["clock"]["deadline_epoch"] - time.time(),
                        state["clock"]["deadline_monotonic"] - time.monotonic()) - 10
        decision = final_notebook_decision(state)
        if decision != QueueDecision.WAIT or available <= 195:
            break
        if time.monotonic() >= heartbeat:
            journal(run, status=AddendumStatus.WAITING, reason="waiting_for_master_final_notebook",
                    seconds_to_deadline=available + 10)
            heartbeat = time.monotonic() + 300
        time.sleep(2)
    if decision == QueueDecision.START and available > 195:
        master = next(stage for stage in state["stages"] if stage.get("name") == "representation_campaign")
        master_status = str(master["status"])
        record_status(run, AddendumStatus.RUNNING, master_stage_status=master_status,
                      master_fallback=master_status != "completed")
        command = [sys.executable, str(ROOT / "scripts/build_representation_report.py"), "--run", str(run)]
        notebook = run_stage(run, NOTEBOOK_STAGE, "normal", command, min(180.0, available))
        if (master_status == "completed" and outcome == assessment == notebook == "completed"
                and data_assessment["complete"]):
            status = AddendumStatus.COMPLETE
        elif (master_status == "completed" and outcome == "skipped_cutoff_or_unverified_trigger"
              and assessment == notebook == "completed"):
            status = AddendumStatus.SKIPPED
        else:
            status = AddendumStatus.PARTIAL
        record_status(run, status, notebook_stage_status=notebook)
    else:
        status = AddendumStatus.PARTIAL
        record_status(run, status, notebook_stage_status="skipped_unverified_master_or_deadline")
    return 0 if status in (AddendumStatus.COMPLETE, AddendumStatus.SKIPPED) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("coordinator", "worker"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--timeout-per-variant", type=float, default=600.0)
    args = parser.parse_args()
    if args.timeout_per_variant <= 0:
        raise ValueError("Per-variant timeout must be positive")
    run = args.run.resolve()
    return coordinator(run, args.timeout_per_variant) if args.action == "coordinator" else worker(run, args.timeout_per_variant)


if __name__ == "__main__":
    sys.exit(main())
