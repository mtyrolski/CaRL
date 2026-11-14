#!/usr/bin/env python3
"""Continue the representation campaign as registered, sequential stages.

Run this coordinator itself as a normal stage of the existing supervisor.  It
never launches an unregistered GPU job.  Previously registered stages are
observed, never retried; a separate watchdog remains responsible for stopping
both this coordinator and each owned stage if the calling session disappears.
"""

import argparse
import fcntl
import json
import math
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
from scripts.intent_experiment_supervisor import boot_id
from scripts.intent_experiment_supervisor import proc_start
from scripts.intent_experiment_supervisor import request_finish

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "analysis/intent_generator_20261114T032232Z"
MODEL = ORIGINAL / "models/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SUPERVISOR = ROOT / "scripts/intent_experiment_supervisor.py"
TRAINING_MODULE = "carl.icl.intent_representation_training"
REPRESENTATIONS = ("json", "formal", "natural")


class Outcome(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def journal(run: Path, event: dict[str, Any]) -> None:
    """Persist decisions even when the parent session or a child disappears."""
    record = {"utc": now_utc(), "epoch": time.time(), "monotonic": time.monotonic(), **event}
    with (run / "campaign_jobs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(record, ensure_ascii=False), flush=True)


def state(run: Path, status: str, **updates: Any) -> None:
    path = run / "campaign_state.json"
    previous = read_json(path) if path.exists() else {}
    atomic_json(path, {**previous, "status": status, "updated_utc": now_utc(), **updates})


def ensure_supervised(run: Path, manifest: dict[str, Any]) -> None:
    """Do not operate outside a live stage of this run's existing watchdog."""
    session = os.getsid(0)
    if os.getpgrp() != session or manifest.get("status") != "active":
        raise RuntimeError("Campaign must run inside an active supervisor-owned process group")
    current = [item for item in manifest.get("stages", [])
               if item.get("pid") == session and item.get("status") == "running"]
    if len(current) != 1 or current[0].get("kind") != "normal":
        raise RuntimeError("Coordinator must itself be a registered normal stage")
    watchdog = manifest.get("watchdog_pid")
    if (not isinstance(watchdog, int) or watchdog <= 0 or manifest.get("boot_id") != boot_id()
            or proc_start(watchdog) != manifest.get("watchdog_proc_start")):
        raise RuntimeError("Existing watchdog is missing or its process identity changed")


def load_plan(run: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = run / "matrix_plan.json"
    checksum = path.with_suffix(".sha256")
    if not checksum.is_file() or checksum.read_text(encoding="utf-8").strip() != file_sha256(path):
        raise ValueError("Frozen matrix plan checksum differs")
    plan = read_json(path)
    if plan.get("schema_version") != 2 or plan.get("run") != str(run):
        raise ValueError("Expected the frozen version-2 matrix for this run")
    clock = manifest["clock"]
    if (plan["deadline_epoch"] > min(clock["start_epoch"] + 11700, manifest["expensive_start_cutoff_epoch"])
            or plan["deadline_monotonic"] > min(clock["start_monotonic"] + 11700,
                                                manifest["expensive_start_cutoff_monotonic"])):
        raise ValueError("Frozen matrix deadline exceeds the campaign limit")
    methods = plan.get("methods")
    if not isinstance(methods, list) or not methods or len({m.get("name") for m in methods}) != len(methods):
        raise ValueError("Matrix method names are missing or duplicated")
    for entry in plan.get("inputs", {}).values():
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {entry['path']}")
    return plan


def remaining(epoch_deadline: float, monotonic_deadline: float) -> float:
    return min(epoch_deadline - time.time(), monotonic_deadline - time.monotonic())


def optional_seed_fits(remaining_seconds: float, training_seconds: list[float],
                       evaluation_seconds: list[float], pending_evaluations: int = 0) -> tuple[bool, float | None]:
    """Gate a whole seed by measured elapsed times, never by result quality."""
    if (not training_seconds or not evaluation_seconds or
            any(not math.isfinite(value) or value <= 0 for value in training_seconds + evaluation_seconds)):
        return False, None
    if pending_evaluations < 0:
        raise ValueError("pending evaluations must be nonnegative")
    required = 1.5 * (3 * max(training_seconds) + (3 + pending_evaluations) * max(evaluation_seconds)) + 300
    return remaining_seconds > required, required


def _stage(manifest: dict[str, Any], name: str) -> dict[str, Any] | None:
    matches = [item for item in manifest.get("stages", []) if item.get("name") == name]
    if len(matches) > 1:
        raise ValueError(f"Duplicate supervisor stage: {name}")
    return matches[0] if matches else None


def _busy_gpu(manifest: dict[str, Any]) -> list[str]:
    return [item["name"] for item in manifest.get("stages", [])
            if item.get("status") == "running" and item.get("kind") in ("training", "expensive")]


def training_command(run: Path, representation: str, seed: int, plan: dict[str, Any]) -> list[str]:
    output = run / f"adapter_repr_{representation}_seed{seed}"
    return [sys.executable, "-m", TRAINING_MODULE, "train", "--data", str(ORIGINAL / "dataset.jsonl"),
            "--output", str(output), "--representation", representation, "--model", str(MODEL),
            "--revision", REVISION, "--dtype", "bfloat16", "--seed", str(seed), "--threads", "8",
            "--max-length", "512", "--max-new-tokens", "128", "--gradient-checkpointing", "--require-cuda",
            "--accumulation", "4", "--lr", "0.0002", "--max-steps", "100", "--eval-every", "20",
            "--validation-boards", "12", "--validation-contexts-per-board", "4",
            "--stop-epoch", str(plan["deadline_epoch"] - 30),
            "--stop-monotonic", str(plan["deadline_monotonic"] - 30)]


def training_complete(run: Path, representation: str, seed: int, expected_data: str) -> bool:
    directory = run / f"adapter_repr_{representation}_seed{seed}"
    if not all((directory / name).is_file() for name in ("status.json", "configuration.json",
                                                            "checkpoint-last-100/adapter_model.safetensors",
                                                            "checkpoint-last-100/adapter_config.json")):
        return False
    status, config = read_json(directory / "status.json"), read_json(directory / "configuration.json")
    fixed = {"representation": representation, "seed": seed, "data_sha256": expected_data,
             "rank": 4, "lora_alpha": 8, "lora_targets": ["q_proj", "v_proj"], "microbatch": 1,
             "accumulation": 4, "learning_rate": .0002, "max_steps": 100, "eval_every": 20,
             "dtype": "bfloat16", "gradient_checkpointing": True, "max_length": 512}
    return (status.get("status") == "finished" and status.get("completed_steps") == 100
            and status.get("last_validation_step") == 100 and all(config.get(k) == v for k, v in fixed.items()))


def elapsed_measurements(run: Path) -> tuple[list[float], list[float]]:
    training = []
    for representation in REPRESENTATIONS:
        path = run / f"adapter_repr_{representation}_seed1729/status.json"
        if path.exists():
            duration = read_json(path).get("elapsed_seconds")
            if isinstance(duration, (int, float)):
                training.append(float(duration) + 30.0)  # include a conservative model-loading allowance
    starts: dict[str, float] = {}
    evaluation = []
    history = run / "matrix_jobs.jsonl"
    if history.exists():
        for line in history.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = event.get("name")
            if event.get("status") == "started" and isinstance(event.get("started_epoch"), (int, float)):
                starts[name] = float(event["started_epoch"])
            elif event.get("status") == "complete" and name in starts and isinstance(event.get("finished_epoch"), (int, float)):
                duration = float(event["finished_epoch"]) - starts.pop(name)
                if duration > 0:
                    evaluation.append(duration)
    return training, evaluation


def gpu_exec(run: Path, command: list[str]) -> None:
    """Hold the same GPU lock as the matrix across the entire training exec."""
    if not command:
        raise ValueError("gpu-exec requires a command")
    manifest = read_json(run / "run_manifest.json")
    group = os.getpgrp()
    if (os.getsid(0) != group or manifest.get("status") != "active" or manifest.get("boot_id") != boot_id()
            or not any(stage.get("pid") == group and stage.get("kind") == "training"
                       and stage.get("status") == "running" for stage in manifest.get("stages", []))):
        raise RuntimeError("GPU training must belong to a registered training stage")
    lock = (run / ".matrix_gpu.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another campaign job holds the GPU lock") from exc
    os.set_inheritable(lock.fileno(), True)
    os.execvpe(command[0], command, os.environ.copy())


class Campaign:
    def __init__(self, run: Path, plan: dict[str, Any]):
        self.run = run
        self.plan = plan
        self.contrasts_path = run / "contrasts_plan.json"
        self.contrasts_sha256 = file_sha256(self.contrasts_path)
        self.failures: list[str] = []
        self.outcomes: dict[str, str] = {}

    def snapshot(self) -> dict[str, Any]:
        return read_json(self.run / "run_manifest.json")

    def remaining_matrix(self) -> float:
        return remaining(self.plan["deadline_epoch"], self.plan["deadline_monotonic"])

    def remaining_report(self) -> float:
        clock = self.snapshot()["clock"]
        return remaining(clock["deadline_epoch"] - 45, clock["deadline_monotonic"] - 45)

    def wait_stage(self, name: str, *, matrix_deadline: bool) -> str | None:
        """Observe a pre-existing registered stage; never relaunch it."""
        reported = False
        heartbeat = time.monotonic() + 300
        while True:
            manifest = self.snapshot()
            if manifest.get("status") != "active":
                return "supervisor_inactive"
            stage = _stage(manifest, name)
            if stage is None:
                return None
            status = stage.get("status")
            if status != "running":
                return str(status)
            left = self.remaining_matrix() if matrix_deadline else self.remaining_report()
            if left <= 0:
                return "waiting_deadline"
            if not reported or time.monotonic() >= heartbeat:
                journal(self.run, {"stage": name, "status": Outcome.WAITING, "remaining_seconds": left})
                reported, heartbeat = True, time.monotonic() + 300
            time.sleep(min(2.0, max(.05, left)))

    def wait_gpu_idle(self) -> bool:
        """Do not overlap any registered training or expensive GPU stage."""
        heartbeat = 0.0
        while self.remaining_matrix() > 5:
            manifest = self.snapshot()
            if manifest.get("status") != "active":
                return False
            busy = _busy_gpu(manifest)
            if not busy:
                return True
            if time.monotonic() >= heartbeat:
                journal(self.run, {"status": Outcome.WAITING, "reason": "gpu_stage_running", "stages": busy})
                heartbeat = time.monotonic() + 300
            time.sleep(2)
        return False

    def run_stage(self, name: str, kind: str, command: list[str], timeout: float, *, gpu: bool) -> str:
        """Register one stage, or observe its prior result without retrying it."""
        manifest = self.snapshot()
        prior = _stage(manifest, name)
        if prior is not None:
            status = self.wait_stage(name, matrix_deadline=gpu) or "missing"
            journal(self.run, {"stage": name, "status": Outcome.SKIPPED, "reason": "already_registered",
                               "registered_status": status})
            self.outcomes[name] = status
            if status != "completed":
                self.failures.append(f"{name}: {status}")
            return status
        if gpu and not self.wait_gpu_idle():
            journal(self.run, {"stage": name, "status": Outcome.SKIPPED, "reason": "gpu_busy_or_matrix_deadline"})
            self.outcomes[name] = "not_started"
            self.failures.append(f"{name}: not started (GPU busy or matrix deadline)")
            return "not_started"
        left = self.remaining_matrix() if gpu else self.remaining_report()
        if left <= 10:
            journal(self.run, {"stage": name, "status": Outcome.SKIPPED, "reason": "deadline"})
            self.outcomes[name] = "not_started"
            self.failures.append(f"{name}: not started (deadline)")
            return "not_started"
        allowed = min(timeout, max(1.0, left - 5))
        invocation = [sys.executable, str(SUPERVISOR), "run", "--name", name, "--kind", kind,
                      "--timeout", str(allowed), str(self.run), "--", *command]
        wrapper_log = self.run / "logs" / f"campaign_supervisor_{name}.log"
        journal(self.run, {"stage": name, "status": Outcome.STARTED, "kind": kind, "timeout_seconds": allowed,
                           "command": command, "wrapper_log": str(wrapper_log)})
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "MPLCONFIGDIR": "/tmp/carl-repr-mpl",
                       "IPYTHONDIR": "/tmp/carl-repr-ipython"}
        with wrapper_log.open("a", encoding="utf-8") as stream:
            child = subprocess.Popen(invocation, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            wrapper_stop = time.monotonic() + allowed + 20
            heartbeat = time.monotonic() + 300
            while child.poll() is None:
                if time.monotonic() >= wrapper_stop:
                    child.terminate()  # Only this known supervisor wrapper, never a process by name.
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
                    break
                if time.monotonic() >= heartbeat:
                    journal(self.run, {"stage": name, "status": Outcome.WAITING, "wrapper_pid": child.pid})
                    heartbeat = time.monotonic() + 300
                time.sleep(2)
            wrapper_code = child.returncode
        observed = self.wait_stage(name, matrix_deadline=gpu)
        status = observed or "not_registered"
        self.outcomes[name] = status
        journal(self.run, {"stage": name, "status": Outcome.COMPLETED if status == "completed" else Outcome.FAILED,
                           "registered_status": status, "wrapper_exit_code": wrapper_code})
        if status != "completed":
            self.failures.append(f"{name}: {status}")
        return status

    def train(self, representation: str, seed: int) -> str:
        stage = (f"train_repr_{representation}_r4" if seed == 1729 else
                 f"train_repr_{representation}_r4_seed{seed}")
        directory = self.run / f"adapter_repr_{representation}_seed{seed}"
        existing = _stage(self.snapshot(), stage)
        if existing is None and directory.exists() and any(directory.iterdir()):
            journal(self.run, {"stage": stage, "status": Outcome.SKIPPED, "reason": "unregistered_output_exists"})
            self.failures.append(f"{stage}: unregistered output")
            self.outcomes[stage] = "unregistered_output"
            return "unregistered_output"
        command = [sys.executable, str(Path(__file__).resolve()), "gpu-exec", "--run", str(self.run), "--",
                   *training_command(self.run, representation, seed, self.plan)]
        status = self.run_stage(stage, "training", command, 900, gpu=True)
        expected = self.plan["inputs"]["support_data"]["sha256"]
        if status == "completed" and not training_complete(self.run, representation, seed, expected):
            journal(self.run, {"stage": stage, "status": Outcome.FAILED, "reason": "adapter_or_training_incomplete"})
            self.failures.append(f"{stage}: incomplete adapter")
            self.outcomes[stage] = "incomplete_adapter"
            return "incomplete_adapter"
        return status

    def matrix(self, group: str) -> str:
        command = [sys.executable, str(ROOT / "scripts/run_representation_matrix.py"), "--run", str(self.run),
                   "--plan", str(self.run / "matrix_plan.json"), "--group", group, "--job-timeout", "1200"]
        return self.run_stage(f"representation_{group}", "expensive", command, 12_000, gpu=True)

    def controller(self) -> str:
        command = [sys.executable, str(ROOT / "scripts/run_controller_matrix.py"), "--run", str(self.run),
                   "--timeout-per-variant", "600"]
        return self.run_stage("controller_five_methods", "expensive", command, 3300, gpu=True)

    def optional_seeds(self) -> None:
        expected = self.plan["inputs"]["support_data"]["sha256"]
        if not all(training_complete(self.run, representation, 1729, expected) for representation in REPRESENTATIONS):
            journal(self.run, {"stage": "optional_seeds", "status": Outcome.SKIPPED,
                               "reason": "matched_seed1729_training_incomplete"})
            return
        pending = 0
        for seed in (2718, 3141):
            training, evaluation = elapsed_measurements(self.run)
            left = self.remaining_matrix()
            fits, required = (optional_seed_fits(left, training, evaluation, pending_evaluations=pending)
                              if len(training) == 3 else (False, None))
            journal(self.run, {"stage": "optional_seed_decision", "seed": seed,
                               "status": "fits" if fits else Outcome.SKIPPED, "remaining_seconds": left,
                               "estimated_required_seconds": required, "training_seconds": training,
                               "evaluation_seconds": evaluation, "pending_evaluations": pending,
                               "basis": "elapsed time and clock only; no model quality measurements"})
            if not fits:
                break
            for representation in REPRESENTATIONS:
                self.train(representation, seed)
            pending += 3
        if pending:
            self.matrix("replicate")

    def generator_assessment(self, *, bfs: bool) -> str:
        name = "campaign_assessment_bfs" if bfs else "campaign_assessment_quick"
        output = self.run / ("assessment_campaign_bfs" if bfs else "assessment_campaign_quick")
        command = [sys.executable, str(ROOT / "scripts/assess_intent_representations.py"),
                   "--data", self.plan["inputs"]["data"]["path"], "--test-ids", self.plan["inputs"]["test_ids"]["path"],
                   "--protocol", str(self.run / "code_snapshot/intent_representation_experiment.py"),
                   "--matrix-plan", str(self.run / "matrix_plan.json"), "--output", str(output),
                   "--reference", "base_json_zero"]
        if file_sha256(self.contrasts_path) != self.contrasts_sha256:
            raise ValueError("Preregistered contrast plan changed during the campaign")
        contrasts = read_json(self.contrasts_path)
        if contrasts.get("status") != "frozen_before_test_predictions":
            raise ValueError("Contrast plan was not frozen before evaluation")
        for before, after in contrasts["primary_contrasts"] + contrasts["secondary_contrasts"]:
            command.extend(("--contrast", f"{before}={after}"))
        if bfs:
            command.extend(("--bfs", "--max-depth", "24", "--max-nodes", "5000", "--seconds", ".2"))
        return self.run_stage(name, "normal", command, 1800 if bfs else 1200, gpu=False)

    def controller_assessment(self) -> str:
        command = [sys.executable, str(ROOT / "scripts/assess_fixed_controller.py"), "--plan",
                   str(self.run / "policy_evaluation_plan.json"), "--output",
                   str(self.run / "controller_assessment_campaign")]
        return self.run_stage("campaign_controller_assessment", "normal", command, 600, gpu=False)

    def notebook(self, name: str) -> str:
        command = [sys.executable, str(ROOT / "scripts/build_representation_report.py"), "--run", str(self.run)]
        return self.run_stage(name, "normal", command, 180, gpu=False)

    def report(self) -> bool:
        """Preserve a quick assessment before attempting bounded search."""
        quick = self.generator_assessment(bfs=False)
        controller = self.controller_assessment()
        # Preserve a useful notebook even if the optional BFS stage is interrupted.
        initial_notebook = self.notebook("campaign_notebook_pre_bfs")
        if self.remaining_report() > 900:
            self.generator_assessment(bfs=True)
        else:
            journal(self.run, {"stage": "campaign_assessment_bfs", "status": Outcome.SKIPPED,
                               "reason": "reporting_time_reserve"})
        notebook = self.notebook("campaign_notebook")
        summary = self.run / "assessment_campaign_quick/summary.json"
        valid_report = (quick == "completed" and controller == "completed" and notebook == "completed"
                        and summary.is_file() and (self.run / "representation_report.ipynb").is_file())
        state(self.run, "report_written" if valid_report else "report_failed", outcomes=self.outcomes,
              failures=self.failures, quick_assessment=quick, controller_assessment=controller,
              initial_notebook_stage=initial_notebook, notebook_stage=notebook)
        return valid_report

    def execute(self) -> None:
        manifest = self.snapshot()
        ensure_supervised(self.run, manifest)
        benchmark = read_json(self.run / "benchmark_representation_json.json")
        if (benchmark.get("status") != "complete" or benchmark.get("rank") != 4
                or benchmark.get("representation") != "json"
                or benchmark.get("data_sha256") != self.plan["inputs"]["support_data"]["sha256"]):
            raise ValueError("Matched generator resource benchmark is missing or inconsistent")
        state(self.run, "running", started_utc=now_utc(), plan_sha256=file_sha256(self.run / "matrix_plan.json"),
              contrasts_sha256=self.contrasts_sha256,
              schedule_sha256=file_sha256(self.run / "campaign_schedule.json"),
              failures=[], outcomes={})
        journal(self.run, {"status": Outcome.STARTED, "stage": "campaign", "matrix_deadline": self.plan["deadline_epoch"]})
        # These names also observe jobs already started by the parent session.
        for representation in REPRESENTATIONS:
            self.train(representation, 1729)
        self.matrix("core")
        self.controller()
        self.matrix("adapted")
        self.matrix("secondary")
        self.optional_seeds()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    coordinator = subparsers.add_parser("run")
    coordinator.add_argument("--run", required=True, type=Path)
    check = subparsers.add_parser("check")
    check.add_argument("--run", required=True, type=Path)
    wrapper = subparsers.add_parser("gpu-exec")
    wrapper.add_argument("--run", required=True, type=Path)
    wrapper.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run = args.run.resolve()
    if args.action == "gpu-exec":
        command = args.command[1:] if args.command and args.command[0] == "--" else args.command
        gpu_exec(run, command)
        return 0
    manifest = read_json(run / "run_manifest.json")
    ensure_supervised(run, manifest)
    plan = load_plan(run, manifest)
    schedule = run / "campaign_schedule.json"
    if not schedule.is_file():
        raise ValueError("The preregistered campaign schedule is missing")
    if args.action == "check":
        contrast_plan = read_json(run / "contrasts_plan.json")
        if contrast_plan.get("status") != "frozen_before_test_predictions":
            raise ValueError("Contrast plan was not frozen before evaluation")
        expected = plan["inputs"]["support_data"]["sha256"]
        print(json.dumps({"status": "ready", "plan_sha256": file_sha256(run / "matrix_plan.json"),
                          "schedule_sha256": file_sha256(schedule),
                          "contrasts_sha256": file_sha256(run / "contrasts_plan.json"),
                          "methods": len(plan["methods"]),
                          "seed1729_adapters": {rep: training_complete(run, rep, 1729, expected)
                                                for rep in REPRESENTATIONS},
                          "registered_gpu_stages": _busy_gpu(manifest),
                          "matrix_seconds_remaining": remaining(plan["deadline_epoch"],
                                                                 plan["deadline_monotonic"])}, indent=2))
        return 0
    lock_path = run / ".campaign_coordinator.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another campaign coordinator is already running") from exc
        campaign = Campaign(run, plan)
        fatal: BaseException | None = None
        try:
            campaign.execute()
        except Exception as exc:
            fatal = exc
            journal(run, {"stage": "campaign", "status": Outcome.FAILED,
                          "error_type": type(exc).__name__, "error": str(exc)})
            state(run, "failed", error_type=type(exc).__name__, error=str(exc), outcomes=campaign.outcomes,
                  failures=campaign.failures)
        try:
            reported = campaign.report()
        except Exception as exc:
            reported = False
            journal(run, {"stage": "reporting", "status": Outcome.FAILED,
                          "error_type": type(exc).__name__, "error": str(exc)})
            state(run, "report_failed", error_type=type(exc).__name__, error=str(exc), outcomes=campaign.outcomes,
                  failures=campaign.failures)
        if fatal is not None or not reported:
            state(run, "failed" if fatal else "report_failed", outcomes=campaign.outcomes, failures=campaign.failures)
            return 1
        state(run, "completed_with_missing" if campaign.failures else "complete",
              outcomes=campaign.outcomes, failures=campaign.failures, finished_utc=now_utc())
        request_finish(run)
        campaign.notebook("campaign_notebook_finish_request")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
