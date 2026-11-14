"""CPU-only checks for the fixed-controller LoRA addendum scheduler."""

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from carl.icl.intent_generator_experiment import file_sha256
from scripts import run_controller_lora_addendum as addendum


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_queue_waits_for_campaign_and_idle_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(addendum, "start_allowed", lambda _state: True)
    monkeypatch.setattr(addendum, "boot_id", lambda: "boot")
    monkeypatch.setattr(addendum, "owned_process", lambda _stage: True)
    monkeypatch.setattr(addendum, "pidfd_alive", lambda _pid: True)
    master = {"name": "representation_campaign", "status": "running", "kind": "normal", "pid": 100,
              "proc_start": 123, "boot_id": "boot", "command": ["python", "scripts/run_representation_campaign.py"]}
    quick = {"name": "campaign_assessment_quick", "status": "running", "kind": "normal", "pid": 101,
             "proc_start": 123, "boot_id": "boot", "command": ["python", "scripts/assess_intent_representations.py"]}
    state = {"status": "active", "boot_id": "boot", "stages": [master]}
    assert addendum.queue_decision(state) == addendum.QueueDecision.WAIT
    monkeypatch.setattr(addendum, "pidfd_alive", lambda _pid: False)
    assert addendum.queue_decision(state) == addendum.QueueDecision.SKIP
    monkeypatch.setattr(addendum, "pidfd_alive", lambda _pid: True)
    state["stages"] += [quick, {"name": "representation_adapted", "status": "running", "kind": "expensive"}]
    assert addendum.queue_decision(state) == addendum.QueueDecision.WAIT
    state["stages"][2]["status"] = "completed"
    assert addendum.queue_decision(state) == addendum.QueueDecision.START
    state["stages"].append({"name": "training", "status": "running", "kind": "training"})
    assert addendum.queue_decision(state) == addendum.QueueDecision.WAIT
    monkeypatch.setattr(addendum, "start_allowed", lambda _state: False)
    assert addendum.queue_decision(state) == addendum.QueueDecision.SKIP
    state["status"] = "complete"
    assert addendum.queue_decision(state) == addendum.QueueDecision.SKIP


def test_notebook_waits_for_verified_master_and_final_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(addendum, "boot_id", lambda: "boot")
    monkeypatch.setattr(addendum, "owned_process", lambda _stage: True)
    monkeypatch.setattr(addendum, "pidfd_alive", lambda _pid: True)
    monkeypatch.setattr(addendum, "process_absent", lambda _pid: True)
    monkeypatch.setattr(addendum, "process_group_absent", lambda _pid: True)
    master = {"name": "representation_campaign", "status": "running", "kind": "normal", "pid": 100,
              "proc_start": 123, "boot_id": "boot", "command": ["python", "run_representation_campaign.py"]}
    finisher = {"name": "campaign_notebook_finish_request", "status": "running", "kind": "normal", "pid": 101,
                "proc_start": 123, "boot_id": "boot", "command": ["python", "build_representation_report.py"]}
    state = {"status": "active", "boot_id": "boot", "stages": [master]}
    assert addendum.final_notebook_decision(state) == addendum.QueueDecision.WAIT
    state["stages"].append(finisher)
    assert addendum.final_notebook_decision(state) == addendum.QueueDecision.WAIT
    master["status"] = finisher["status"] = "completed"
    assert addendum.final_notebook_decision(state) == addendum.QueueDecision.START
    monkeypatch.setattr(addendum, "process_group_absent", lambda _pid: False)
    assert addendum.final_notebook_decision(state) == addendum.QueueDecision.SKIP
    state["boot_id"] = "different"
    assert addendum.final_notebook_decision(state) == addendum.QueueDecision.SKIP


def test_failed_master_allows_report_only_if_group_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(addendum, "boot_id", lambda: "boot")
    monkeypatch.setattr(addendum, "process_absent", lambda _pid: True)
    monkeypatch.setattr(addendum, "process_group_absent", lambda _pid: True)
    master = {"name": "representation_campaign", "status": "failed", "kind": "normal", "pid": 100,
              "proc_start": 123, "boot_id": "boot", "command": ["python", "run_representation_campaign.py"]}
    state = {"status": "active", "boot_id": "boot", "stages": [master]}
    assert addendum.final_notebook_decision(state) == addendum.QueueDecision.START
    monkeypatch.setattr(addendum, "process_group_absent", lambda _pid: False)
    assert addendum.final_notebook_decision(state) == addendum.QueueDecision.SKIP


def test_guardian_marks_lost_only_after_repeated_absence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = {"name": "stage", "status": "running", "pid": 101, "proc_start": 456, "boot_id": "boot"}
    write_json(tmp_path / "run_manifest.json", {"status": "active", "boot_id": "boot", "stages": [stage]})
    monkeypatch.setattr(addendum, "boot_id", lambda: "boot")
    monkeypatch.setattr(addendum, "watchdog_alive", lambda _state: True)
    monkeypatch.setattr(addendum, "process_absent", lambda _pid: True)
    monkeypatch.setattr(addendum, "proc_start", lambda _pid: None)
    checks: list[int] = []
    def absent_group(pid: int) -> bool:
        checks.append(pid)
        return True
    monkeypatch.setattr(addendum, "process_group_absent", absent_group)
    monkeypatch.setattr(addendum.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(addendum, "update_report", lambda *_args: None)
    assert addendum.mark_lost_stage(tmp_path, "stage", 102, 123)
    recorded = json.loads((tmp_path / "run_manifest.json").read_text())["stages"][0]
    assert recorded["status"] == "lost_or_exited" and recorded["exit_code"] is None
    assert checks == [101, 101, 101, 101]


def test_guardian_does_not_mark_live_group_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = {"name": "stage", "status": "running", "pid": 101, "proc_start": 456, "boot_id": "boot"}
    write_json(tmp_path / "run_manifest.json", {"status": "active", "boot_id": "boot", "stages": [stage]})
    monkeypatch.setattr(addendum, "boot_id", lambda: "boot")
    monkeypatch.setattr(addendum, "watchdog_alive", lambda _state: True)
    monkeypatch.setattr(addendum, "process_absent", lambda _pid: True)
    monkeypatch.setattr(addendum, "proc_start", lambda _pid: None)
    monkeypatch.setattr(addendum, "process_group_absent", lambda _pid: False)
    assert not addendum.mark_lost_stage(tmp_path, "stage", 102, 123)
    assert json.loads((tmp_path / "run_manifest.json").read_text())["stages"][0]["status"] == "running"


def test_guardian_does_not_write_if_watchdog_is_lost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = {"name": "stage", "status": "running", "pid": 101, "proc_start": 456, "boot_id": "boot"}
    write_json(tmp_path / "run_manifest.json", {"status": "active", "boot_id": "boot", "stages": [stage]})
    monkeypatch.setattr(addendum, "watchdog_alive", lambda _state: False)
    assert not addendum.mark_lost_stage(tmp_path, "stage", 102, 123)
    assert json.loads((tmp_path / "run_manifest.json").read_text())["stages"][0]["status"] == "running"


def fake_frozen_plan(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run = root / "analysis/run"
    run.mkdir(parents=True)
    ids = run / "controller_test_ids.json"
    write_json(ids, {"record_ids": [str(i) for i in range(40)]})
    selection = run / "controller_selection.json"
    config = {"model": "frozen", "action_cap": 16}
    write_json(selection, {"selected_config": config})
    source = root / "carl/frozen_policy.py"
    source.parent.mkdir(parents=True)
    source.write_text("frozen prompt and action parser", encoding="utf-8")
    calibration = run / "policy_calibration_05b.manifest.json"
    write_json(calibration, {"config": config, "action_parser": "frozen parser",
                             "source_sha256": {"carl/frozen_policy.py": file_sha256(source)}})
    assets = run / "controller_assets_snapshot.json"
    write_json(assets, {"frozen": True})
    matrix = run / "matrix_plan.json"
    methods = [{"name": name, "representation": rep, "support_set": "zero",
                "adapter": "adapter" if name.startswith("rep_lora_") else None}
               for rep in ("json", "formal", "natural")
               for name in (f"base_{rep}_zero", f"rep_lora_{rep}_zero")]
    write_json(matrix, {"methods": methods})
    matrix.with_suffix(".sha256").write_text(file_sha256(matrix), encoding="utf-8")
    strict = {"frozen_ids": str(ids.relative_to(root)), "frozen_ids_sha256": file_sha256(ids),
              "controller_selection_sha256": file_sha256(selection), "selected_controller": config,
              "decoding": "greedy", "action_parser": "frozen parser", "model_training": False,
              "primitive_action_budget": 16, "api_calls": 0}
    strict_path = run / "policy_evaluation_plan.json"
    write_json(strict_path, strict)
    plan = {**strict, "status": "posthoc_icl_prior_to_matched_lora_predictions",
            "generator_output_normalisation": "none", "variants": list(addendum.VARIANTS),
            "contrasts": [[f"base_{rep}_zero", f"rep_lora_{rep}_zero"] for rep in ("json", "formal", "natural")],
            "original_policy_plan_sha256": file_sha256(strict_path),
            "controller_calibration_manifest_sha256": file_sha256(calibration),
            "controller_assets_snapshot_sha256": file_sha256(assets), "matrix_plan_sha256": file_sha256(matrix),
            "controller_source_sha256": {"carl/frozen_policy.py": file_sha256(source)}}
    plan_path = run / addendum.PLAN_NAME
    write_json(plan_path, plan)
    plan_path.with_suffix(".sha256").write_text(file_sha256(plan_path), encoding="utf-8")
    monkeypatch.setattr(addendum, "verify_controller_assets", lambda *_args: None)
    return run


def test_plan_binds_calibration_sources_and_matched_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = fake_frozen_plan(tmp_path, monkeypatch)
    assert addendum.validate_plan(run, tmp_path)["variants"] == list(addendum.VARIANTS)
    (tmp_path / "carl/frozen_policy.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Calibrated controller source changed"):
        addendum.validate_plan(run, tmp_path)


def test_plan_rejects_generator_normalisation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = fake_frozen_plan(tmp_path, monkeypatch)
    path = run / addendum.PLAN_NAME
    plan = json.loads(path.read_text())
    plan["generator_output_normalisation"] = "remove_one_outer_fence"
    write_json(path, plan)
    path.with_suffix(".sha256").write_text(file_sha256(path), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed controller protocol"):
        addendum.validate_plan(run, tmp_path)


def test_prepared_inputs_bind_predictions_and_disallow_normalisation(tmp_path: Path) -> None:
    run = tmp_path
    ids = [str(i) for i in range(40)]
    write_json(run / "controller_test_ids.json", {"record_ids": ids})
    for path in (run / "data/dataset.jsonl", run / "matrix_plan.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen", encoding="utf-8")
    prediction = run / "predictions/base_json_zero.jsonl"
    prediction.parent.mkdir(parents=True)
    for path in (prediction, prediction.with_suffix(".run.json"), prediction.with_suffix(".prompts.jsonl")):
        path.write_text("frozen", encoding="utf-8")
    directory = run / "controller_lora_addendum_inputs/base_json_zero"
    write_jsonl(directory / "records.jsonl", [{"record_id": ids[0]}])
    write_jsonl(directory / "eligibility.jsonl", [{"record_id": identifier, "status": "ready" if i == 0
                                                    else "invalid_intent"} for i, identifier in enumerate(ids)])
    paths = {"predictions_sha256": prediction, "test_ids_sha256": run / "controller_test_ids.json",
             "data_sha256": run / "data/dataset.jsonl", "matrix_plan_sha256": run / "matrix_plan.json",
             "records_sha256": directory / "records.jsonl", "eligibility_sha256": directory / "eligibility.jsonl",
             "prediction_run_sha256": prediction.with_suffix(".run.json"),
             "prompts_sha256": prediction.with_suffix(".prompts.jsonl")}
    manifest = {"variant": "base_json_zero", "planned": 40,
                **{key: file_sha256(path) for key, path in paths.items()}}
    write_json(directory / "manifest.json", manifest)
    assert len(addendum.verify_prepared(run, "base_json_zero", prediction, directory)) == 1
    manifest["generator_output_normalisation"] = {"mode": "remove_one_outer_fence"}
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="strict protocol"):
        addendum.verify_prepared(run, "base_json_zero", prediction, directory)
    manifest.pop("generator_output_normalisation")
    write_json(directory / "manifest.json", manifest)
    prediction.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="strict protocol"):
        addendum.verify_prepared(run, "base_json_zero", prediction, directory)


def test_missing_generation_is_reported_without_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "logs").mkdir()
    write_json(tmp_path / "run_manifest.json", {"stages": [{"pid": os.getpgrp(), "kind": "expensive",
                                                              "status": "running"}]})
    monkeypatch.setattr(addendum, "ensure_registered", lambda _run: None)
    monkeypatch.setattr(addendum, "validate_plan", lambda _run: {})
    monkeypatch.setattr(addendum, "start_allowed", lambda _state: True)
    monkeypatch.setattr(addendum, "watchdog_alive", lambda _state: True)
    monkeypatch.setattr(addendum, "training_running", lambda _run: [])
    monkeypatch.setattr(addendum, "run_child", lambda *_args: pytest.fail("no GPU child should run"))
    assert addendum.worker(tmp_path, 60) == 1
    events = [json.loads(line) for line in (tmp_path / "controller_lora_addendum_jobs.jsonl").read_text().splitlines()]
    assert [event["variant"] for event in events] == list(addendum.VARIANTS)
    assert all(event["status"] == addendum.JobStatus.MISSING_GENERATION for event in events)


def test_partial_generation_is_not_a_model_failure(tmp_path: Path) -> None:
    ids = tmp_path / "data/test_ids.json"
    write_json(ids, {"record_ids": ["a", "b", "c"]})
    write_json(tmp_path / "matrix_plan.json", {"inputs": {"test_ids": {"path": str(ids)}}})
    prediction = tmp_path / "predictions/base_json_zero.jsonl"
    write_jsonl(prediction, [{"record_id": "a"}, {"record_id": "b"}])
    write_json(prediction.with_suffix(".partial-summary.json"), {"status": "stopped", "reason": "timeout"})
    directory = tmp_path / "inputs"
    write_json(directory / "manifest.json", {"counts": {"missing_generation": 1}})
    assert addendum.generation_completeness(tmp_path, prediction, directory) == {
        "complete": False, "observed": 2, "planned": 3, "summary_status": None,
        "partial_reason": "timeout", "selected_missing": 1,
    }
    write_jsonl(prediction, [{"record_id": "a"}, {"record_id": "b"}, {"record_id": "c"}])
    write_json(prediction.with_suffix(".summary.json"), {"status": "complete", "n": 3})
    write_json(directory / "manifest.json", {"counts": {}})
    assert addendum.generation_completeness(tmp_path, prediction, directory)["complete"] is True


def test_ready_intents_use_same_frozen_policy_and_action_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "logs").mkdir()
    write_json(tmp_path / "run_manifest.json", {"stages": [{"pid": os.getpgrp(), "kind": "expensive",
                                                              "status": "running"}]})
    config = {"model": "/frozen/model", "revision": "frozen-revision", "device": "cuda", "action_cap": 16,
              "max_new_tokens": 32, "max_context_tokens": 2048}
    for variant in addendum.VARIANTS:
        prediction = tmp_path / f"predictions/{variant}.jsonl"
        prediction.parent.mkdir(exist_ok=True)
        for path in (prediction, prediction.with_suffix(".run.json"), prediction.with_suffix(".prompts.jsonl")):
            path.write_text("frozen", encoding="utf-8")
        write_json(tmp_path / f"{addendum.INPUTS_NAME}/{variant}/manifest.json", {})
    monkeypatch.setattr(addendum, "ensure_registered", lambda _run: None)
    monkeypatch.setattr(addendum, "validate_plan", lambda _run: {"selected_controller": config})
    monkeypatch.setattr(addendum, "start_allowed", lambda _state: True)
    monkeypatch.setattr(addendum, "watchdog_alive", lambda _state: True)
    monkeypatch.setattr(addendum, "remaining_seconds", lambda _state: 3600.0)
    monkeypatch.setattr(addendum, "training_running", lambda _run: [])
    monkeypatch.setattr(addendum, "verify_prepared", lambda *_args: [{"record_id": "ready"}])
    monkeypatch.setattr(addendum, "generation_completeness", lambda *_args: {"complete": True, "selected_missing": 0})
    monkeypatch.setattr(addendum, "verify_controller_assets", lambda *_args: None)
    monkeypatch.setattr(addendum, "verify_result", lambda *_args: None)
    commands: list[list[str]] = []
    def fake_child(command: list[str], _log: Path, _timeout: float,
                   pass_fds: tuple[int, ...] = ()) -> tuple[addendum.JobStatus, int]:
        assert len(pass_fds) == 1
        commands.append(command)
        return addendum.JobStatus.COMPLETE, 0
    monkeypatch.setattr(addendum, "run_child", fake_child)
    assert addendum.worker(tmp_path, 60) == 0
    assert len(commands) == 6
    for command in commands:
        assert command[command.index("--model") + 1] == config["model"]
        assert command[command.index("--revision") + 1] == config["revision"]
        assert command[command.index("--action-cap") + 1] == "16"
        assert command[command.index("--max-new-tokens") + 1] == "32"
        assert command[command.index("--max-context-tokens") + 1] == "2048"


def test_completed_result_rejects_different_weights(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    output = tmp_path / "output.jsonl"
    write_jsonl(inputs, [{"record_id": "r"}])
    write_jsonl(output, [{"record_id": "r"}])
    calibration = {"weights_sha256": "original", "tokenizer_sha256": "tokenizer"}
    write_json(tmp_path / "policy_calibration_05b.manifest.json", calibration)
    plan = {"selected_controller": {"model": "frozen"}, "controller_source_sha256": {"policy.py": "source"},
            "action_parser": "parser"}
    saved = {"config": plan["selected_controller"], "source_sha256": plan["controller_source_sha256"],
             "weights_sha256": "original", "tokenizer_sha256": "tokenizer", "action_parser": "parser",
             "records_sha256": file_sha256(inputs), "record_ids": ["r"], "configuration_sha256": "config"}
    write_json(output.with_suffix(".manifest.json"), saved)
    write_json(output.with_suffix(".summary.json"), {"configuration_sha256": "config", "status": "complete",
                                                      "records": 1})
    addendum.verify_result(tmp_path, plan, inputs, output)
    saved["weights_sha256"] = "modified"
    write_json(output.with_suffix(".manifest.json"), saved)
    with pytest.raises(ValueError, match="calibrated policy"):
        addendum.verify_result(tmp_path, plan, inputs, output)


def test_assessment_completion_requires_no_infrastructure_failures(tmp_path: Path) -> None:
    (tmp_path / addendum.PLAN_NAME).write_text("plan", encoding="utf-8")
    plan_hash = file_sha256(tmp_path / addendum.PLAN_NAME)
    directory = tmp_path / addendum.ASSESSMENT_NAME
    rows = [{"family": "all", "variant": variant, "planned": 40, "pipeline_evaluated": 40,
             "statuses": {"invalid_representation": 40}} for variant in addendum.VARIANTS]
    summary = {"config": {"plan_sha256": plan_hash, "analysis_kind": "exploratory_addendum",
                          "generator_output_normalisation": "none", "action_cap": 16},
               "results": rows, "rejected_variants": []}
    write_json(directory / "summary.json", summary)
    marker = {"plan_sha256": plan_hash, "analysis_kind": "exploratory_addendum", "status": "complete",
              "summary_sha256": file_sha256(directory / "summary.json")}
    write_json(directory / "complete.json", marker)
    assert addendum.assessment_completeness(tmp_path)["complete"] is True
    rows[0]["statuses"] = {"invalid_representation": 39, "missing_generation": 1}
    rows[0]["pipeline_evaluated"] = 39
    write_json(directory / "summary.json", summary)
    marker["summary_sha256"] = file_sha256(directory / "summary.json")
    write_json(directory / "complete.json", marker)
    assert addendum.assessment_completeness(tmp_path)["complete"] is False
    rows[0]["statuses"] = {"invalid_representation": 40}
    rows[0]["pipeline_evaluated"] = 40
    summary["rejected_variants"] = [{"variant": addendum.VARIANTS[0]}]
    write_json(directory / "summary.json", summary)
    marker["summary_sha256"] = file_sha256(directory / "summary.json")
    write_json(directory / "complete.json", marker)
    assert addendum.assessment_completeness(tmp_path)["complete"] is False


def test_watchdog_is_checked_before_registering_gpu_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "logs").mkdir()
    write_json(tmp_path / "run_manifest.json", {"stages": []})
    def lost_watchdog(*_args: object) -> None:
        raise RuntimeError("watchdog unavailable")
    monkeypatch.setattr(addendum, "ensure_supervised", lost_watchdog)
    monkeypatch.setattr(addendum.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("must not launch GPU"))
    assert addendum.run_stage(tmp_path, addendum.WORK_STAGE, "expensive", ["no-op"], 60) == "watchdog_unverified"


def test_zombie_watchdog_is_not_alive() -> None:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        until = time.monotonic() + 5
        state = None
        while time.monotonic() < until:
            state = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
            if state == "Z":
                break
            time.sleep(.01)
        assert state == "Z"
        start = addendum.proc_start(process.pid)
        assert start is not None
        manifest = {"status": "active", "boot_id": addendum.boot_id(),
                    "watchdog_pid": process.pid, "watchdog_proc_start": start}
        assert addendum.watchdog_alive(manifest) is False
    finally:
        process.wait(timeout=5)


def test_expired_queue_still_assesses_and_records_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "logs").mkdir()
    write_json(tmp_path / "run_manifest.json", {"status": "active",
               "stages": [{"name": "representation_campaign", "status": "completed"}],
               "clock": {"deadline_epoch": time.time() + 1000, "deadline_monotonic": time.monotonic() + 1000}})
    (tmp_path / addendum.PLAN_NAME).write_text("plan", encoding="utf-8")
    monkeypatch.setattr(addendum, "ensure_supervised", lambda *_args: None)
    monkeypatch.setattr(addendum, "validate_plan", lambda _run: {})
    monkeypatch.setattr(addendum, "start_allowed", lambda _state: False)
    monkeypatch.setattr(addendum, "final_notebook_decision", lambda _state: addendum.QueueDecision.START)
    stages: list[tuple[str, str]] = []
    def fake_stage(_run: Path, name: str, kind: str, _command: list[str], _timeout: float) -> str:
        stages.append((name, kind))
        return "completed"
    monkeypatch.setattr(addendum, "run_stage", fake_stage)
    assert addendum.coordinator(tmp_path, 60) == 0
    assert stages == [(addendum.ASSESS_STAGE, "normal"), (addendum.NOTEBOOK_STAGE, "normal")]
    status = json.loads((tmp_path / "controller_lora_addendum_status.json").read_text())
    assert status["status"] == addendum.AddendumStatus.SKIPPED
    assert status["rollout_stage_status"] == "skipped_cutoff_or_unverified_trigger"


def test_missing_final_notebook_is_partial_and_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "logs").mkdir()
    write_json(tmp_path / "run_manifest.json", {"status": "active", "stages": [],
               "clock": {"deadline_epoch": time.time() + 1000, "deadline_monotonic": time.monotonic() + 1000}})
    (tmp_path / addendum.PLAN_NAME).write_text("plan", encoding="utf-8")
    monkeypatch.setattr(addendum, "ensure_supervised", lambda *_args: None)
    monkeypatch.setattr(addendum, "validate_plan", lambda _run: {})
    monkeypatch.setattr(addendum, "start_allowed", lambda _state: False)
    monkeypatch.setattr(addendum, "final_notebook_decision", lambda _state: addendum.QueueDecision.SKIP)
    monkeypatch.setattr(addendum, "run_stage", lambda *_args: "completed")
    assert addendum.coordinator(tmp_path, 60) == 1
    status = json.loads((tmp_path / "controller_lora_addendum_status.json").read_text())
    assert status["status"] == addendum.AddendumStatus.PARTIAL
    assert status["notebook_stage_status"] == "skipped_unverified_master_or_deadline"


def test_successful_process_exits_with_missing_results_are_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "logs").mkdir()
    write_json(tmp_path / "run_manifest.json", {"status": "active",
               "stages": [{"name": "representation_campaign", "status": "completed"}],
               "clock": {"deadline_epoch": time.time() + 4000, "deadline_monotonic": time.monotonic() + 4000}})
    (tmp_path / addendum.PLAN_NAME).write_text("plan", encoding="utf-8")
    monkeypatch.setattr(addendum, "ensure_supervised", lambda *_args: None)
    monkeypatch.setattr(addendum, "validate_plan", lambda _run: {})
    monkeypatch.setattr(addendum, "queue_decision", lambda _state: addendum.QueueDecision.START)
    monkeypatch.setattr(addendum, "final_notebook_decision", lambda _state: addendum.QueueDecision.START)
    monkeypatch.setattr(addendum, "run_stage", lambda *_args: "completed")
    monkeypatch.setattr(addendum, "assessment_completeness", lambda _run: {"complete": False, "reason": "missing"})
    assert addendum.coordinator(tmp_path, 60) == 1
    status = json.loads((tmp_path / "controller_lora_addendum_status.json").read_text())
    assert status["status"] == addendum.AddendumStatus.PARTIAL
    assert status["assessment_data_complete"] is False and status["notebook_stage_status"] == "completed"


def test_run_child_inherits_lock_fd_and_keeps_default_compatibility(tmp_path: Path) -> None:
    with (tmp_path / "lock").open("w") as lock:
        command = [sys.executable, "-c", f"import os; os.fstat({lock.fileno()})"]
        assert addendum.run_child(command, tmp_path / "child.log", 5, pass_fds=(lock.fileno(),)) == (
            addendum.JobStatus.COMPLETE, 0)
    assert addendum.run_child([sys.executable, "-c", "pass"], tmp_path / "default.log", 5) == (
        addendum.JobStatus.COMPLETE, 0)


def test_lock_survives_parent_close_until_child_exits(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpu.lock"
    ready = tmp_path / "ready"
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    read_fd, write_fd = os.pipe()
    command = [sys.executable, "-c", "import os,sys; from pathlib import Path; "
               "os.fstat(int(sys.argv[1])); Path(sys.argv[2]).touch(); os.read(int(sys.argv[3]),1)",
               str(lock.fileno()), str(ready), str(read_fd)]
    results: list[tuple[addendum.JobStatus, int | None]] = []
    thread = threading.Thread(target=lambda: results.append(
        addendum.run_child(command, tmp_path / "child.log", 10, pass_fds=(lock.fileno(), read_fd))))
    thread.start()
    try:
        until = time.monotonic() + 5
        while not ready.exists() and thread.is_alive() and time.monotonic() < until:
            time.sleep(.01)
        assert ready.exists(), "child did not inherit the lock before the timeout"
        os.close(read_fd)
        read_fd = -1
        lock.close()
        with lock_path.open("a+") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(write_fd, b"x")
            os.close(write_fd)
            write_fd = -1
            thread.join(timeout=5)
            assert not thread.is_alive() and results == [(addendum.JobStatus.COMPLETE, 0)]
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        lock.close()
        thread.join(timeout=5)
