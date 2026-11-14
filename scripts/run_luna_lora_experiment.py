#!/usr/bin/env python3
"""Inspect or explicitly start the prepared, deadline-supervised Luna/LoRA study.

Without ``--execute`` this script performs local preflight checks only. No
watchdog, model inference, training or paid API call is started in that mode.
"""

import argparse
import copy
import datetime as dt
import json
import math
import os
import select
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records
from scripts import intent_experiment_supervisor as supervisor
from scripts.build_luna_lora_preparation_notebook import notebook as notebook_document
from scripts.run_intent_generator_experiment import make_clock

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/icl/luna_lora_prepared.json"
PREPARED = ROOT / "analysis/luna_lora_preparation_20261114"
PYTHON = ["uv", "run", "--offline", "--no-sync", "python"]
PEFT_PYTHON = ["uv", "run", "--offline", "--no-sync", "--with", "peft==0.21.3", "python"]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def check_preparation(config: dict[str, Any], prepared: Path) -> dict[str, Any]:
    """Check the frozen source files and layout separation without model calls."""
    calibration = prepared / "data/calibration_candidates.jsonl"
    test = prepared / "evaluation/test_contexts.jsonl"
    manifest = read_json(prepared / "evaluation/manifest.json")
    if (manifest["calibration_candidates_sha256"] != file_sha256(calibration)
            or manifest["test_contexts_sha256"] != file_sha256(test)):
        raise ValueError("Prepared inputs differ from the frozen evaluation manifest")
    rows, tests = read_records(calibration), read_records(test)
    if not rows or not tests or any(row.get("policy_measurements") is not None for row in rows):
        raise ValueError("Prepared candidates are missing or already contain policy measurements")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") not in {"train", "validation"} or row.get("phase") != "calibration":
            raise ValueError("Calibration input must contain only train and validation")
        if row.get("validation", {}).get("accepted") is not True:
            raise ValueError("Each candidate must be checked before policy measurement")
        groups.setdefault(row["context_id"], []).append(row)
    per_context = config["calibration"]["candidates_per_context"]
    if any(len(group) != per_context for group in groups.values()):
        raise ValueError("Frozen calibration groups have different candidate counts")
    layouts = {name: {row["layout_id"] for row in rows if row["split"] == name}
               for name in ("train", "validation")}
    layouts["test"] = {row["layout_id"] for row in tests}
    if (any(row.get("split") != "test" for row in tests) or len(layouts["test"]) != len(tests)
            or layouts["train"] & layouts["validation"] or layouts["train"] & layouts["test"]
            or layouts["validation"] & layouts["test"]):
        raise ValueError("Training, validation and test layouts must be disjoint")
    families = dict(Counter(row["family"] for row in tests))
    if families != config["evaluation"]["contexts"]:
        raise ValueError("Frozen test family counts differ from the configuration")
    model = ROOT / config["generator"]["model"]
    for name in ("model.safetensors", "config.json", "tokenizer.json", "experiment_revision.json"):
        if not (model / name).is_file():
            raise ValueError(f"Local Qwen snapshot is missing {name}; no API work should start")
    if read_json(model / "experiment_revision.json").get("revision") != config["generator"]["revision"]:
        raise ValueError("Local Qwen revision differs from the configured model")
    if config["controller"]["new_campaign_budget_usd"] != "2.00":
        raise ValueError("The prepared Luna transport requires its configured $2 ceiling")
    if config["deadline_seconds"] != 16200:
        raise ValueError("The prepared supervisor uses a 4 h 30 min deadline")
    expected_controller = {"model": "gpt-6-luna", "service_tier": "default", "reasoning_effort": "none",
                           "temperature": 0, "max_output_tokens": 256, "request_retries": 0}
    if (any(config["controller"].get(key) != value for key, value in expected_controller.items())
            or config["calibration"]["action_cap"] != 16 or config["evaluation"]["action_cap"] != 16
            or config["evaluation"]["attempts_per_proposal"] != 1):
        raise ValueError("Calibration and evaluation must use the same frozen controller settings")
    if (config["expensive_start_cutoff_seconds"] != 13500 or config["training_stop_seconds"] != 14400
            or config["generator"]["rank"] != 4 or config["generator"]["thinking"] is not False
            or config["generator"]["sampling"] != "greedy"):
        raise ValueError("Configured model or deadline differs from the implemented protocol")
    scales = config["evaluation"]["requested_scales"]
    if (not isinstance(scales, list) or len(scales) != 2 or scales[0] == scales[1]
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in scales)):
        raise ValueError("Exactly two distinct requested scales in [0,1] are required")
    return {
        "status": "prepared_not_started", "paid_model_requests": 0, "training_started": False,
        "candidates": len(rows), "contexts": len(groups),
        "layouts": {key: len(value) for key, value in layouts.items()},
        "test_families": families, "candidate_sha256": file_sha256(calibration),
        "test_contexts_sha256": file_sha256(test), "config_sha256": file_sha256(CONFIG),
        "maximum_new_campaign_usd": config["controller"]["new_campaign_budget_usd"],
        "historical_unknown_reservation_usd": config["controller"]["previous_unknown_reservation_usd"],
        "note": "The old reservations remain in their original ledger; no spending is inferred from them.",
    }


def _snapshot(prepared: Path, run_dir: Path) -> None:
    for relative in ("data/calibration_candidates.jsonl", "data/manifest.json", "data/rejected.jsonl",
                     "evaluation/test_contexts.jsonl", "evaluation/manifest.json", "evaluation/test_ids.json"):
        source, target = prepared / relative, run_dir / "inputs" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_sha256(source) != file_sha256(target):
            raise ValueError(f"Input snapshot differs: {relative}")
    shutil.copyfile(CONFIG, run_dir / "inputs/config.json")


def _watchdog_live(run_dir: Path) -> bool:
    """Fail closed if this run's watchdog is absent, exited or from another boot."""
    state = read_json(run_dir / "run_manifest.json")
    pid, start = state.get("watchdog_pid"), state.get("watchdog_proc_start")
    if (state.get("status") != supervisor.RunStatus.ACTIVE or state.get("boot_id") != supervisor.boot_id()
            or isinstance(pid, bool) or not isinstance(pid, int) or isinstance(start, bool) or not isinstance(start, int)
            or not hasattr(os, "pidfd_open")):
        return False
    identity = {"pid": pid, "proc_start": start, "boot_id": state["boot_id"]}
    if not supervisor.owned_process(identity):
        return False
    try:
        descriptor = os.pidfd_open(pid)
    except OSError:
        return False
    try:
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        return not poller.poll(0) and supervisor.owned_process(identity)
    finally:
        os.close(descriptor)


def _stage(run_dir: Path, name: str, kind: supervisor.StageKind, timeout: float, command: list[str]) -> bool:
    if not _watchdog_live(run_dir):
        supervisor.append_log(run_dir, f"Stage {name} was not started: watchdog is absent or unverifiable.")
        return False
    try:
        code = supervisor.run_stage(run_dir, name, kind, timeout, command)
        state = read_json(run_dir / "run_manifest.json")
        matches = [stage for stage in state["stages"] if stage["name"] == name]
        return (code == 0 and len(matches) == 1 and matches[0]["status"] == supervisor.StageStatus.COMPLETED
                and matches[0]["exit_code"] == 0)
    except SystemExit as error:
        supervisor.append_log(run_dir, f"Stage {name} was not started: {error}")
        return False


def _gpu_free_bytes() -> int:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    values = result.stdout.splitlines()
    if len(values) != 1 or not values[0].strip().isdigit():
        raise ValueError("Cannot establish free memory on exactly one GPU")
    return int(values[0].strip()) * 1024 * 1024


def _training_finished(status: dict[str, Any], steps: int) -> bool:
    """A clean exit alone does not prove the two adapters had matched budgets."""
    return (status.get("status") == "finished" and status.get("completed_steps") == steps
            and status.get("last_validation_step") == steps and isinstance(status.get("best_adapter"), str))


def _options(config: dict[str, Any], model: Path) -> list[str]:
    item = config["generator"]
    return ["--model", str(model), "--revision", item["revision"], "--seed", str(config["seed"]),
            "--dtype", item["dtype"], "--rank", str(item["rank"]),
            "--accumulation", str(item["gradient_accumulation"]),
            "--lr", str(item["learning_rate"]), "--max-length", str(item["max_length"]),
            "--max-new-tokens", str(item["max_new_tokens"]), "--max-steps", str(item["max_steps"]),
            "--eval-every", str(item["eval_every"]), "--gradient-checkpointing", "--require-cuda",
            "--min-informative-train-contexts", str(config["calibration"]["min_informative_train_contexts"]),
            "--min-informative-validation-contexts", str(config["calibration"]["min_informative_validation_contexts"])]


def run(config: dict[str, Any], prepared: Path, run_dir: Path, start_epoch: float, start_monotonic: float) -> None:
    """Run sequential stages under a separate watchdog; never retry an unknown API call."""
    clock = make_clock(start_epoch, start_monotonic, config["deadline_seconds"])
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as stream:
        clock_path = Path(stream.name)
        json.dump(clock, stream)
    try:
        supervisor.initialise(run_dir, clock_path, report_mode="status_only")
    finally:
        clock_path.unlink(missing_ok=True)
    state: dict[str, Any] = {"status": "running", "reason": None, "started_utc": clock["start_utc"]}

    def checkpoint(status: str, reason: str | None = None) -> None:
        state.update(status=status, reason=reason, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        supervisor.write_json(run_dir / "study_status.json", state)

    try:
        _snapshot(prepared, run_dir)
        checkpoint("running")
        supervisor.write_json(run_dir / "study.ipynb", notebook_document())
        data = run_dir / "inputs/data/calibration_candidates.jsonl"
        contexts = run_dir / "inputs/evaluation/test_contexts.jsonl"
        budget = run_dir / "budget"
        calibration = run_dir / "calibration"
        ctl = config["controller"]
        model = ROOT / config["generator"]["model"]
        fit_config = copy.deepcopy(config)
        resource_ok = False
        for attempt in range(2):
            if attempt:
                fit_config["generator"]["rank"] = 2
                fit_config["generator"]["gradient_accumulation"] = 2
            free_before = _gpu_free_bytes()
            resource = run_dir / ("resource_check.json" if not attempt else "resource_check_smaller.json")
            command = [*PEFT_PYTHON, "-m", "carl.icl.policy_relative_lora", "resource-check", "--data", str(data),
                       "--output", str(resource), *_options(fit_config, model)]
            name = "resource_check" if not attempt else "resource_check_smaller"
            if _stage(run_dir, name, supervisor.StageKind.EXPENSIVE, 600, command):
                result = read_json(resource)
                peak = result.get("gpu_peak_reserved_bytes")
                if (result.get("measurement_kind") == "resource_only_no_luna_labels" and isinstance(peak, int)
                        and peak <= int(free_before * 0.90)):
                    resource_ok = True
                    break
            supervisor.append_log(run_dir, f"{name} failed or lacked GPU headroom; no Luna request has been made.")
        if not resource_ok:
            checkpoint("partial", "resource_check_failed_before_api")
            return
        ctl_options = ["--budget-dir", str(budget), "--budget-usd", ctl["new_campaign_budget_usd"],
                       "--action-cap", str(config["calibration"]["action_cap"]), "--timeout", str(ctl["timeout_seconds"])]
        calibration_cmd = [*PYTHON, "-m", "scripts.run_luna_lora_controller", "--input", str(data),
                           "--output-dir", str(calibration), "--attempts",
                           str(config["calibration"]["attempts_per_candidate"]), *ctl_options, "--execute"]
        if not _stage(run_dir, "calibration", supervisor.StageKind.EXPENSIVE, 5400, calibration_cmd):
            checkpoint("partial", "calibration_failed_or_timed_out")
            return
        measured = calibration / "calibrated.jsonl"
        inspection = run_dir / "calibration_inspection.json"
        inspect_cmd = [*PYTHON, "-m", "carl.icl.policy_relative_lora", "inspect", "--data", str(measured),
                       "--output", str(inspection), "--arm", "scaled"]
        if not _stage(run_dir, "inspect_calibration", supervisor.StageKind.NORMAL, 120, inspect_cmd):
            checkpoint("partial", "insufficient_complete_calibration")
            return
        audit = read_json(inspection)["audit"]["splits"]
        for split in ("train", "validation"):
            actual = audit[split]["contexts_with_distinct_success_durations"]
            minimum = config["calibration"][f"min_informative_{split}_contexts"]
            if actual < minimum:
                checkpoint("partial", f"insufficient_informative_{split}_contexts")
                return
        benchmark_result = None
        for attempt in range(2):
            if attempt:
                # Both arms use the same smaller adapter and accumulation.
                # Microbatch is already one; never truncate a board.
                fit_config["generator"]["rank"] = 2
                fit_config["generator"]["gradient_accumulation"] = 2
            options = _options(fit_config, model)
            free_before = _gpu_free_bytes()
            benchmark = run_dir / ("benchmark.json" if not attempt else "benchmark_smaller.json")
            benchmark_cmd = [*PEFT_PYTHON, "-m", "carl.icl.policy_relative_lora", "benchmark", "--data", str(measured),
                             "--output", str(benchmark), "--arm", "scaled", *options]
            name = "benchmark" if not attempt else "benchmark_smaller"
            if _stage(run_dir, name, supervisor.StageKind.EXPENSIVE, 600, benchmark_cmd):
                result = read_json(benchmark)
                peak = result.get("gpu_peak_reserved_bytes")
                if isinstance(peak, int) and peak <= int(free_before * 0.90):
                    benchmark_result = result
                    break
            supervisor.append_log(run_dir, f"{name} failed or lacked GPU headroom.")
        if benchmark_result is None:
            checkpoint("partial", "gpu_benchmark_failed_or_insufficient_headroom")
            return
        step_seconds = benchmark_result.get("complete_step_seconds")
        if not isinstance(step_seconds, (int, float)) or step_seconds <= 0:
            checkpoint("partial", "invalid_benchmark_timing")
            return
        remaining = min(clock["start_monotonic"] + 13500 - time.monotonic(),
                        clock["start_epoch"] + 13500 - time.time())
        # Reserve one hour before the expensive-stage cutoff for generation
        # and the fixed controller assessment, plus the final reporting window.
        training_seconds = max(remaining - 3600, 0)
        steps = min(fit_config["generator"]["max_steps"], int(training_seconds / (3 * step_seconds)))
        if steps < fit_config["generator"]["eval_every"]:
            checkpoint("partial", "insufficient_time_for_matched_training")
            return
        fit_config["generator"]["max_steps"] = steps
        options = _options(fit_config, model)
        adapters: dict[str, str] = {}
        for index, arm in enumerate(("unscaled", "scaled")):
            destination = run_dir / "adapters" / arm
            train_cmd = [*PEFT_PYTHON, "-m", "carl.icl.policy_relative_lora", "train", "--data", str(measured),
                         "--output", str(destination), "--arm", arm, *options,
                         "--stop-monotonic", str(clock["start_monotonic"] + 14400),
                         "--stop-epoch", str(clock["start_epoch"] + 14400)]
            available = min(clock["start_monotonic"] + 13500 - time.monotonic(),
                            clock["start_epoch"] + 13500 - time.time()) - 3600
            timeout = min(2400, max(0, available) / (2 - index))
            if timeout <= 0 or not _stage(run_dir, f"train_{arm}", supervisor.StageKind.TRAINING, timeout, train_cmd):
                checkpoint("partial", f"training_{arm}_failed")
                return
            if not _training_finished(read_json(destination / "status.json"), steps):
                checkpoint("partial", f"training_{arm}_incomplete")
                return
            adapters[arm] = read_json(destination / "best.json")["adapter"]
        first = read_json(run_dir / "adapters/unscaled/configuration.json")
        second = read_json(run_dir / "adapters/scaled/configuration.json")
        if first["comparability_sha256"] != second["comparability_sha256"]:
            checkpoint("partial", "training_configurations_not_matched")
            return
        predictions = run_dir / "predictions"
        predictions.mkdir()
        scales = config["evaluation"]["requested_scales"]
        if len(scales) != 2 or scales[0] == scales[1] or any(not 0 <= scale <= 1 for scale in scales):
            raise ValueError("Exactly two distinct requested scales in [0,1] are required")
        for arm, scale, name in (("unscaled", None, "unscaled"), ("scaled", scales[0], "scaled_low"),
                                 ("scaled", scales[1], "scaled_high")):
            destination = predictions / f"{name}.jsonl"
            cmd = [*PEFT_PYTHON, "-m", "carl.icl.policy_relative_lora", "evaluate", "--data", str(contexts),
                   "--output", str(destination), "--arm", arm, "--adapter", adapters[arm], *options]
            if scale is not None:
                cmd += ["--requested-scale", str(scale)]
            if not _stage(run_dir, f"generate_{name}", supervisor.StageKind.EXPENSIVE, 1200, cmd):
                checkpoint("partial", f"generation_{name}_failed")
                return
        for name in ("unscaled", "scaled_low", "scaled_high"):
            metadata = read_json(predictions / f"{name}.run.json")
            if (metadata.get("status") != "complete"
                    or metadata.get("training_comparability_sha256") != first["comparability_sha256"]
                    or metadata.get("controller_hash") != first["controller_hash"]):
                checkpoint("partial", f"generation_{name}_provenance_mismatch")
                return
        prepared_evaluation = run_dir / "evaluation"
        build_cases = [*PYTHON, "-m", "scripts.prepare_luna_lora_controller_evaluation", "--contexts", str(contexts),
                       "--unscaled", str(predictions / "unscaled.jsonl"), "--scaled",
                       str(predictions / "scaled_low.jsonl"), str(predictions / "scaled_high.jsonl"),
                       "--scales", str(scales[0]), str(scales[1]),
                       "--output", str(prepared_evaluation)]
        if not _stage(run_dir, "prepare_evaluation", supervisor.StageKind.NORMAL, 120, build_cases):
            checkpoint("partial", "evaluation_preparation_failed")
            return
        controller_inputs = prepared_evaluation / "controller_inputs.jsonl"
        evaluated = run_dir / "controller_test"
        if controller_inputs.stat().st_size:
            test_cmd = [*PYTHON, "-m", "scripts.run_luna_lora_controller", "--input", str(controller_inputs),
                        "--output-dir", str(evaluated), "--attempts", "1", *ctl_options, "--execute"]
            if not _stage(run_dir, "controller_test", supervisor.StageKind.EXPENSIVE, 3600, test_cmd):
                checkpoint("partial", "controller_test_failed_or_timed_out")
        else:
            evaluated.mkdir()
            (evaluated / "results.jsonl").touch()
        if not (evaluated / "results.jsonl").exists():
            evaluated.mkdir(parents=True, exist_ok=True)
            (evaluated / "results.jsonl").touch()
        controller_hash = read_json(calibration / "plan.json")["controller_hash"]
        assessment = [*PYTHON, "-m", "scripts.assess_luna_lora", "--plan",
                      str(prepared_evaluation / "evaluation_plan.jsonl"), "--rollouts", str(evaluated / "results.jsonl"),
                      "--output", str(run_dir / "assessment"), "--controller-hash", controller_hash,
                      "--action-cap", str(config["evaluation"]["action_cap"]), "--attempts", "1"]
        if not _stage(run_dir, "assessment", supervisor.StageKind.EVALUATION, 600, assessment):
            checkpoint("partial", "assessment_failed")
            return
        if state["status"] != "partial":
            checkpoint("complete")
    except Exception as error:
        checkpoint("partial", f"{type(error).__name__}: {error}")
        raise
    finally:
        notebook = run_dir / "study.ipynb"
        if notebook.exists():
            render = [*PYTHON, "-m", "scripts.execute_luna_lora_report", "--source", str(notebook),
                      "--output", str(notebook), "--run-dir", str(run_dir)]
            if not _stage(run_dir, "render_report", supervisor.StageKind.EVALUATION, 300, render):
                supervisor.append_log(run_dir, "Notebook retained without execution; inspect study_status.json.")
        supervisor.finish(run_dir)


def main() -> None:
    started_epoch, started_monotonic = time.time(), time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--run-dir", type=Path, help="A new output directory, required with --execute")
    parser.add_argument("--execute", action="store_true", help="Start supervised experiments and paid API requests")
    args = parser.parse_args()
    config = read_json(CONFIG)
    plan = check_preparation(config, args.prepared.resolve())
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.run_dir is None:
        parser.error("--execute requires a new --run-dir")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "analysis").resolve()):
        parser.error("--run-dir must be a new directory inside the repository's analysis directory")
    run(config, args.prepared.resolve(), run_dir, started_epoch, started_monotonic)


if __name__ == "__main__":
    main()
