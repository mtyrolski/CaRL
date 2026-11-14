#!/usr/bin/env python3
"""Run or safely resume the deadline-supervised Sokoban intent experiment."""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import TypedDict

from scripts import intent_experiment_supervisor as supervisor
from scripts.report_intent_generator_experiment import read_json
from scripts.report_intent_generator_experiment import write_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/icl/intent_generator_experiment.json"
PYTHON = ["uv", "run", "--no-sync", "python"]
PEFT_PYTHON = ["uv", "run", "--no-sync", "--with", "peft==0.21.3", "python"]


class ExperimentClock(TypedDict):
    start_epoch: float
    start_monotonic: float
    start_utc: str
    deadline_epoch: float
    deadline_monotonic: float
    deadline_utc: str


def make_clock(start_epoch: float, start_monotonic: float, seconds: float) -> ExperimentClock:
    if not 0 < seconds <= 16200:
        raise ValueError("deadline_seconds must be positive and at most 4 h 30 min")

    def utc(epoch: float) -> str:
        return dt.datetime.fromtimestamp(epoch, dt.timezone.utc).isoformat()

    return {
        "start_epoch": start_epoch,
        "start_monotonic": start_monotonic,
        "start_utc": utc(start_epoch),
        "deadline_epoch": start_epoch + seconds,
        "deadline_monotonic": start_monotonic + seconds,
        "deadline_utc": utc(start_epoch + seconds)
    }


def expensive_work_allowed(clock: ExperimentClock, now_monotonic: float, now_epoch: float) -> bool:
    return (now_monotonic < min(clock["start_monotonic"] + 13500, clock["deadline_monotonic"])
            and now_epoch < min(clock["start_epoch"] + 13500, clock["deadline_epoch"]))


def training_steps(clock: ExperimentClock, benchmark: dict, maximum: int, now_monotonic: float,
                   now_epoch: float) -> int:
    """Bound the step count using measured speed and both clocks."""
    seconds = benchmark.get("complete_step_seconds")
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        return 0
    remaining = min(clock["start_monotonic"] + 14400 - now_monotonic, clock["start_epoch"] + 14400 - now_epoch,
                    clock["deadline_monotonic"] - now_monotonic, clock["deadline_epoch"] - now_epoch)
    return max(0, min(maximum, int(max(0, remaining - 120) / (seconds * 2))))


def clock_expired(state: dict) -> bool:
    clock: ExperimentClock = state["clock"]
    return time.monotonic() >= clock["deadline_monotonic"] or time.time() >= clock["deadline_epoch"]


def ensure_watchdog(run_dir: Path) -> None:
    state = read_json(run_dir / "run_manifest.json")
    if state.get("boot_id") != supervisor.boot_id():
        raise RuntimeError("Run boot identity is missing or changed; verify the original watchdog before resuming")
    pid = state.get("watchdog_pid")
    if pid and supervisor.proc_start(pid) == state.get("watchdog_proc_start"):
        return
    if any(stage.get("status") == supervisor.StageStatus.RUNNING for stage in state.get("stages", [])):
        raise RuntimeError(
            "Watchdog is not visible while stages are recorded as running; use the original PID namespace")
    supervisor.refresh_watchdog(run_dir)


def is_current_driver(stage: dict) -> bool:
    """Recognise only the supervisor-registered group that runs this driver."""
    return stage.get("pid") == os.getpgrp() and supervisor.owned_process(stage)


def wait_for_existing_stages(run_dir: Path) -> None:
    """Do not start another GPU stage while a supervised child is alive."""
    while True:
        with supervisor.locked(run_dir):
            state = read_json(run_dir / "run_manifest.json")
            if clock_expired(state):
                raise RuntimeError("The original deadline has elapsed")
            running = [
                stage for stage in state.get("stages", [])
                if stage.get("status") == supervisor.StageStatus.RUNNING and not is_current_driver(stage)
            ]
            # A group may outlive its leader after an interruption.  Waiting
            # for any remaining members is conservative and never signals an
            # unrelated process, even if a process-group number was recycled.
            live = [
                stage for stage in running
                if supervisor.owned_process(stage) or supervisor.group_has_live_members(stage["pid"])
            ]
            for stage in running:
                if stage not in live:
                    stage["status"] = supervisor.StageStatus.LOST_OR_EXITED
                    stage["finished_utc"] = supervisor.utc_now()
                    supervisor.append_log(run_dir,
                                          f"Resumption found unowned stage {stage['name']}; no restart assumed.")
            if len(live) != len(running):
                supervisor.write_json(run_dir / "run_manifest.json", state)
                supervisor.update_report(run_dir, state)
        if not live:
            return
        print("Waiting for supervised stages: " + ", ".join(stage["name"] for stage in live), flush=True)
        write_report(run_dir)
        time.sleep(10)


def run_once(run_dir: Path, name: str, kind: supervisor.StageKind, timeout: float, command: list[str]) -> bool:
    """Use an existing stage result, or start it once under the watchdog."""
    while True:
        state = read_json(run_dir / "run_manifest.json")
        if clock_expired(state):
            return False
        previous = next((stage for stage in state.get("stages", []) if stage["name"] == name), None)
        if previous:
            if previous["status"] == supervisor.StageStatus.RUNNING:
                wait_for_existing_stages(run_dir)
                continue
            return bool(previous["status"] == supervisor.StageStatus.COMPLETED)
        if kind == supervisor.StageKind.EXPENSIVE and not expensive_work_allowed(state["clock"], time.monotonic(),
                                                                                 time.time()):
            supervisor.append_log(run_dir, f"Skipped {name}: expensive-work cutoff reached.")
            return False
        try:
            code: int = supervisor.run_stage(run_dir, name, kind, timeout, command)
        except SystemExit as exc:
            if "already exists" in str(exc):
                continue
            raise
        finally:
            write_report(run_dir)
        return code == 0


def recovery_stage_name(run_dir: Path, name: str) -> str:
    """Use a new ledger entry for an explicitly resumable partial stage."""
    stages = read_json(run_dir / "run_manifest.json").get("stages", [])
    for stage in stages:
        if ((stage["name"] == name or stage["name"].startswith(f"{name}_resume_"))
                and stage["status"] == supervisor.StageStatus.RUNNING):
            return str(stage["name"])
    used = {stage["name"] for stage in stages}
    if name not in used:
        return name
    index = 1
    while f"{name}_resume_{index}" in used:
        index += 1
    return f"{name}_resume_{index}"


def selected_model(run_dir: Path, config: dict) -> tuple[str, str]:
    saved = read_json(run_dir / "model_selection.json")
    if saved.get("model") and saved.get("revision"):
        return saved["model"], saved["revision"]
    benchmark = read_json(run_dir / "qwen3_benchmark.json") or read_json(run_dir / "benchmark.json")
    if benchmark.get("model") and benchmark.get("revision"):
        return benchmark["model"], benchmark["revision"]
    model_config = config["model"]
    destination = run_dir / "models/Qwen3-0.6B"
    script = ("from huggingface_hub import snapshot_download; "
              "import sys; snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
              "allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.tiktoken','*.jinja'])")
    downloaded = run_once(
        run_dir, "download_qwen3", supervisor.StageKind.EXPENSIVE, model_config["download_timeout_seconds"],
        [*PYTHON, "-c", script, model_config["repository"], model_config["revision"],
         str(destination)])
    if downloaded:
        model, revision = str(destination), model_config["revision"]
    else:
        model, revision = model_config["fallback_repository"], model_config["fallback_revision"]
    supervisor.write_json(run_dir / "model_selection.json", {"model": model, "revision": revision})
    return model, revision


def model_options(config: dict, model: str, revision: str, rank: int, accumulation: int) -> list[str]:
    item = config["model"]
    return [
        "--model", model, "--revision", revision, "--max-length",
        str(item["max_length"]), "--max-new-tokens",
        str(item["max_new_tokens"]), "--dtype", item["dtype"], "--rank",
        str(rank), "--accumulation",
        str(accumulation), "--seed",
        str(config["seed"])
    ]


def benchmark_fits(benchmark: dict, reserved_limit: int) -> bool:
    if benchmark.get("status") != "complete":
        return False
    reserved = benchmark.get("gpu_peak_reserved_bytes")
    return reserved is None or reserved <= reserved_limit


def evaluation_needs_resume(output: Path) -> bool:
    """A prewritten run configuration is enough to indicate an interrupted run."""
    return output.exists() or output.with_suffix(".run.json").exists()


def prepare_annotations(run_dir: Path, config: dict, dataset: Path) -> Path:
    settings = config["annotation"]
    selection = run_dir / "terra_selection_final.jsonl"
    if not selection.exists():
        run_once(run_dir, "terra_preselect_final", supervisor.StageKind.NORMAL, 60, [
            *PYTHON, "-m", "carl.icl.intent_terra", "--freeze-from",
            str(dataset), "--selection",
            str(selection), "--selection-count",
            str(settings["selection_count"]), "--selection-seed",
            str(settings["selection_seed"])
        ])
    if not settings["try_isolated_codex"] or not selection.exists():
        return dataset
    selection_manifest = read_json(selection.with_suffix(selection.suffix + ".manifest.json"))
    if (selection_manifest.get("source_sha256") != hashlib.sha256(dataset.read_bytes()).hexdigest()
            or selection_manifest.get("selection_sha256") != hashlib.sha256(selection.read_bytes()).hexdigest()):
        raise ValueError("Frozen annotation selection does not match the checked dataset")
    output = run_dir / "terra"
    clock = run_dir / "clock.json"
    if not clock.exists():
        supervisor.write_json(clock, read_json(run_dir / "run_manifest.json")["clock"])

    def annotate(offset: int, count: int, name: str) -> bool:
        timeout = settings["timeout_seconds"]
        return run_once(run_dir, name, supervisor.StageKind.EXPENSIVE, timeout + 30, [
            *PYTHON, "-m", "carl.icl.intent_terra", "--selection",
            str(selection), "--output",
            str(output), "--clock",
            str(clock), "--registry",
            str(run_dir / "terra_workers.jsonl"), "--offset",
            str(offset), "--batch-size",
            str(count), "--timeout",
            str(timeout), "--retry", "0"
        ])

    # One preliminary request establishes availability; failed calls never
    # silently become local labels or trigger a retry loop.
    pilot_summary = output / "batch_0000_summary.json"
    if not pilot_summary.exists():
        annotate(0, 1, "terra_pilot")
    if read_json(pilot_summary).get("status") != "accepted":
        return dataset
    selection_count = sum(1 for line in selection.read_text().splitlines() if line.strip())
    offset = 1
    while offset < selection_count:
        count = min(settings["batch_size"], selection_count - offset)
        summary = output / f"batch_{offset:04d}_summary.json"
        if not summary.exists():
            annotate(offset, count, f"terra_batch_{offset:04d}")
        if read_json(summary).get("status") != "accepted":
            break
        offset += count
    merged = run_dir / "dataset_terra.jsonl"
    if not merged.exists():
        run_once(run_dir, "merge_terra", supervisor.StageKind.NORMAL, 60, [
            *PYTHON, "-m", "carl.icl.intent_terra", "--merge-from",
            str(dataset), "--summaries",
            str(output), "--merge-output",
            str(merged)
        ])
    return merged if merged.exists() else dataset


def run_pipeline(run_dir: Path, config: dict) -> None:
    dataset = run_dir / "dataset.jsonl"
    ids = run_dir / "frozen_evaluation_contexts.json"
    if not dataset.exists():
        if not run_once(run_dir, "prepare_data_final", supervisor.StageKind.NORMAL, 600, [
                *PYTHON, "scripts/prepare_intent_generator_data.py", "--config", config["data_config"], "--output",
                str(run_dir)
        ]):
            return
    manifest = read_json(run_dir / "dataset_manifest.json")
    if manifest.get("dataset_sha256") != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("Checked dataset does not match its manifest")
    if not ids.exists():
        if not run_once(
                run_dir, "freeze_final_test_contexts", supervisor.StageKind.NORMAL, 60,
            [*PYTHON, "-m", "carl.icl.intent_assessment", "freeze", "--data",
             str(dataset), "--output",
             str(ids)]):
            return
    if read_json(ids).get("dataset_sha256") != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("Frozen evaluation contexts do not match the checked dataset")
    original_dataset = dataset
    dataset = prepare_annotations(run_dir, config, dataset)
    if dataset != original_dataset:
        # The record IDs were frozen before annotation; the train-only overlay
        # changes the data hash but cannot change the held-out selection.
        merged_ids = run_dir / "frozen_evaluation_contexts_annotated.json"
        if not merged_ids.exists():
            frozen = read_json(ids)
            frozen["pre_annotation_dataset_sha256"] = frozen["dataset_sha256"]
            frozen["dataset_sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
            supervisor.write_json(merged_ids, frozen)
        ids = merged_ids
    model, revision = selected_model(run_dir, config)
    rank = config["model"]["lora_rank"]
    accumulation = config["model"]["accumulation"]
    options = model_options(config, model, revision, rank, accumulation)
    bench_path = run_dir / "qwen3_benchmark.json"
    if not bench_path.exists():
        run_once(run_dir, "qwen3_resource_benchmark", supervisor.StageKind.EXPENSIVE, 300, [
            *PEFT_PYTHON, "-m", "carl.icl.intent_generator_experiment", "benchmark", "--data",
            str(dataset), "--output",
            str(bench_path), "--gradient-checkpointing", *options
        ])
    benchmark = read_json(bench_path)
    if (not benchmark_fits(benchmark, config["model"]["gpu_reserved_limit_bytes"]) and expensive_work_allowed(
            read_json(run_dir / "run_manifest.json")["clock"], time.monotonic(), time.time())):
        rank, accumulation = 2, 1
        options = model_options(config, model, revision, rank, accumulation)
        bench_path = run_dir / "smaller_benchmark.json"
        run_once(run_dir, "smaller_resource_benchmark", supervisor.StageKind.EXPENSIVE, 300, [
            *PEFT_PYTHON, "-m", "carl.icl.intent_generator_experiment", "benchmark", "--data",
            str(dataset), "--output",
            str(bench_path), "--gradient-checkpointing", *options
        ])
        benchmark = read_json(bench_path)
    if benchmark:
        rank, accumulation = benchmark.get("rank", rank), benchmark.get("accumulation", accumulation)
        options = model_options(config, model, revision, rank, accumulation)
    base = run_dir / "base_outputs.jsonl"

    def evaluate(output: Path, name: str, adapter: str | None = None) -> None:
        if output.with_suffix(".summary.json").exists():
            return
        partial = evaluation_needs_resume(output)
        command = [
            *PEFT_PYTHON, "-m", "carl.icl.intent_generator_experiment", "evaluate", "--data",
            str(dataset), "--test-ids",
            str(ids), "--output",
            str(output), *options
        ]
        if adapter:
            command.extend(["--adapter", adapter])
        if partial:
            command.append("--resume")
        run_once(run_dir, recovery_stage_name(run_dir, name), supervisor.StageKind.EXPENSIVE,
                 config["evaluation"]["generation_timeout_seconds"], command)

    evaluate(base, "base_evaluation")
    state = read_json(run_dir / "run_manifest.json")
    train_stages = [
        stage for stage in state.get("stages", [])
        if stage.get("kind") == supervisor.StageKind.TRAINING and "smoke" not in stage.get("name", "")
    ]
    adapter_dir = run_dir / "adapter"
    # An interrupted or failed training stage is not restarted from scratch and
    # never overwrites its partial checkpoints.
    can_train = benchmark_fits(
        benchmark, config["model"]["gpu_reserved_limit_bytes"]) and not train_stages and not adapter_dir.exists()
    steps = training_steps(state["clock"], benchmark, config["training"]["max_steps"], time.monotonic(),
                           time.time()) if can_train else 0
    if steps >= config["training"]["eval_every"] and expensive_work_allowed(state["clock"], time.monotonic(),
                                                                            time.time()):
        stop = min(state["training_stop_monotonic"], state["clock"]["deadline_monotonic"] - 1800)
        run_once(run_dir, "qwen3_lora_train", supervisor.StageKind.TRAINING, config["training"]["timeout_seconds"], [
            *PEFT_PYTHON, "-m", "carl.icl.intent_generator_experiment", "train", "--data",
            str(dataset), "--output",
            str(adapter_dir), "--gradient-checkpointing", "--max-steps",
            str(steps), "--eval-every",
            str(config["training"]["eval_every"]), "--lr",
            str(config["training"]["learning_rate"]), "--validation-contexts-per-board",
            str(config["training"]["validation_contexts_per_board"]), "--stop-monotonic",
            str(stop), *options
        ])
    best = read_json(adapter_dir / "best.json").get("adapter")
    adapted = run_dir / "adapted_outputs.jsonl"
    if best:
        evaluate(adapted, "adapted_evaluation", best)
    assessment = run_dir / "assessment"
    if not (assessment / "summary.json").exists():
        partial = (assessment / "assessment.jsonl").exists() and (assessment / "assessment.jsonl").stat().st_size > 0
        command = [
            *PYTHON, "-m", "carl.icl.intent_assessment", "assess", "--data",
            str(dataset), "--test-ids",
            str(ids), "--base",
            str(base), "--adapted",
            str(adapted), "--output",
            str(assessment), "--max-depth",
            str(config["evaluation"]["max_depth"]), "--max-nodes",
            str(config["evaluation"]["max_nodes"]), "--seconds",
            str(config["evaluation"]["seconds_per_proposal"])
        ]
        if partial:
            command.append("--resume")
        run_once(run_dir, recovery_stage_name(run_dir, "assessment"), supervisor.StageKind.EVALUATION,
                 config["evaluation"]["assessment_timeout_seconds"], command)
    if (assessment / "summary.json").exists():
        run_once(run_dir, "plot_assessment", supervisor.StageKind.NORMAL, 120, [
            *PYTHON, "scripts/plot_intent_assessment.py",
            str(assessment / "summary.json"), "--output",
            str(run_dir / "figures/validity_feasibility")
        ])
    if base.exists() and adapted.exists():
        audit = run_dir / "fence_audit"
        if not (audit / "fence_audit_summary.json").exists():
            partial = (audit / "fence_audit.jsonl").exists() and (audit / "fence_audit.jsonl").stat().st_size > 0
            command = [
                *PYTHON, "-m", "carl.icl.intent_fence_audit", "--data",
                str(dataset), "--test-ids",
                str(ids), "--base",
                str(base), "--adapted",
                str(adapted), "--output",
                str(audit)
            ]
            if partial:
                command.append("--resume")
            run_once(run_dir, recovery_stage_name(run_dir, "fence_audit"), supervisor.StageKind.EVALUATION, 300,
                     command)
        feasibility = run_dir / "fence_feasibility"
        if not (feasibility / "summary.json").exists():
            partial = ((feasibility / "feasibility.jsonl").exists()
                       or (feasibility / "feasibility_config.json").exists())
            command = [
                *PYTHON, "-m", "carl.icl.intent_fence_feasibility", "--data",
                str(dataset), "--test-ids",
                str(ids), "--base",
                str(base), "--adapted",
                str(adapted), "--output",
                str(feasibility), "--max-depth",
                str(config["evaluation"]["max_depth"]), "--max-nodes",
                str(config["evaluation"]["max_nodes"]), "--seconds",
                str(config["evaluation"]["seconds_per_proposal"])
            ]
            if partial:
                command.append("--resume")
            run_once(run_dir, recovery_stage_name(run_dir, "fence_feasibility"), supervisor.StageKind.EVALUATION,
                     config["evaluation"]["assessment_timeout_seconds"], command)
        if (feasibility / "summary.json").exists():
            run_once(run_dir, "plot_posthoc_feasibility", supervisor.StageKind.NORMAL, 120, [
                *PYTHON, "scripts/plot_intent_assessment.py",
                str(feasibility / "summary.json"), "--title", "Post-hoc: one outer fence removed", "--output",
                str(run_dir / "figures/posthoc_feasibility")
            ])


def main() -> None:
    start_epoch, start_monotonic = time.time(), time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true", help="Continue an existing run without resetting its clock")
    parser.add_argument("--proposal", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT):
        raise SystemExit("The run directory must be inside the existing checkout")
    if not 1 <= config["annotation"]["batch_size"] <= 4:
        raise SystemExit("The annotation batch size must be between one and four")
    if args.resume:
        if not (run_dir / "run_manifest.json").exists():
            raise SystemExit("--resume requires an existing run manifest")
        saved_config = read_json(run_dir / "experiment_config.json")
        if saved_config:
            config = saved_config
    else:
        clock = make_clock(start_epoch, start_monotonic, config["deadline_seconds"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as temporary:
            json.dump(clock, temporary)
            temporary.flush()
            supervisor.initialise(run_dir, Path(temporary.name), args.proposal)
        supervisor.write_json(run_dir / "clock.json", clock)
        supervisor.write_json(run_dir / "experiment_config.json", config)
    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda _signal, _frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        state = read_json(run_dir / "run_manifest.json")
        if state.get("status") != supervisor.RunStatus.ACTIVE or clock_expired(state):
            return
        ensure_watchdog(run_dir)
        wait_for_existing_stages(run_dir)
        run_pipeline(run_dir, config)
    except KeyboardInterrupt:
        supervisor.append_log(run_dir, "Entry point interrupted; watchdog retains ownership of active children.")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        write_report(run_dir)
    state = read_json(run_dir / "run_manifest.json")
    own_stage = any(
        stage.get("status") == supervisor.StageStatus.RUNNING and is_current_driver(stage)
        for stage in state.get("stages", []))
    if own_stage:
        supervisor.request_finish(run_dir)
        write_report(run_dir)
    elif state.get("status") == supervisor.RunStatus.ACTIVE and not clock_expired(state):
        supervisor.finish(run_dir)
        write_report(run_dir)


if __name__ == "__main__":
    main()
