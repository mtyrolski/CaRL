"""Frozen methods, deadlines, adapter hashes and safe orchestration."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.prepare_representation_matrix import Group
from scripts.prepare_representation_matrix import matrix_deadline
from scripts.prepare_representation_matrix import methods
from scripts.prepare_representation_matrix import write_frozen
from scripts.run_representation_matrix import command_for
from scripts.run_representation_matrix import completed_output
from scripts.run_representation_matrix import current_adapter
from scripts.run_representation_matrix import run_group
from scripts.run_representation_matrix import stop_process_tree
from scripts.run_representation_matrix import training_running


def test_frozen_matrix_has_exact_order_and_variants(tmp_path):
    matrix = methods(tmp_path)
    groups = {group: [m.name for m in matrix if m.group == group] for group in Group}
    assert groups[Group.CORE] == [
        "base_json_zero", "r4_json_zero", "r8_json_zero", "r2_json_zero", "base_json_balanced0",
        "base_formal_balanced0", "base_natural_balanced0", "base_formal_zero", "base_natural_zero",
    ]
    assert len(groups[Group.SECONDARY]) == 11
    assert groups[Group.ADAPTED] == ["rep_lora_json_zero", "rep_lora_formal_zero", "rep_lora_natural_zero"]
    assert groups[Group.REPLICATE] == [
        "rep_lora_json_seed2718", "rep_lora_formal_seed2718", "rep_lora_natural_seed2718",
        "rep_lora_json_seed3141", "rep_lora_formal_seed3141", "rep_lora_natural_seed3141",
    ]
    assert len({m.name for m in matrix}) == 29
    assert all(m.adapter_pending for m in matrix if m.group == Group.ADAPTED)
    assert all(m.adapter_pending for m in matrix if m.group == Group.REPLICATE)


def test_deadline_preserves_original_wall_and_monotonic_starts():
    assert matrix_deadline({"start_epoch": 100, "start_monotonic": 200}) == (11800, 11900)


def test_plan_is_write_once_and_checksums_itself(tmp_path):
    path = tmp_path / "matrix_plan.json"
    write_frozen(path, {"methods": ["a"]})
    assert path.with_suffix(".sha256").is_file()
    write_frozen(path, {"methods": ["a"]})
    with pytest.raises(FileExistsError):
        write_frozen(path, {"methods": ["b"]})


def test_pending_adapter_hashes_are_bound_once(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "adapter_config.json").write_text("{}")
    job = {"name": "test", "adapter": str(adapter), "adapter_pending": True}
    first = current_adapter(job, tmp_path)
    assert current_adapter(job, tmp_path) == first
    (adapter / "adapter_model.safetensors").write_bytes(b"different")
    with pytest.raises(ValueError, match="changed"):
        current_adapter(job, tmp_path)
    (adapter / "adapter_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        current_adapter(job, tmp_path)


def test_command_explicitly_binds_method_and_resume(tmp_path):
    job = {"name": "base_formal_balanced0", "representation": "formal", "support_set": "balanced0",
           "adapter": None, "output": str(tmp_path / "output.jsonl")}
    plan = {"inputs": {key: {"path": key} for key in ("data", "test_ids", "support_data", "support_manifest")},
            "model": {"path": "model", "revision": "sha"},
            "settings": {"split": "test", "dtype": "bfloat16", "seed": 1729, "threads": 8,
                         "max_length": 1024, "max_new_tokens": 128},
            "deadline_epoch": 100, "deadline_monotonic": 200}
    command = command_for(job, plan, True)
    assert command[command.index("--variant") + 1] == job["name"]
    assert command[command.index("--representation") + 1] == "formal"
    assert command[command.index("--support-set") + 1] == "balanced0"
    assert "--require-cuda" in command and "--resume" in command and "--adapter" not in command
    job["adapter"] = "path-to-adapter"
    assert command_for(job, plan, False)[-2:] == ["--adapter", "path-to-adapter"]


def test_incomplete_output_never_counts_as_complete(tmp_path):
    job = {"name": "base_json_zero", "representation": "json", "support_set": "zero", "adapter": None,
           "output": str(tmp_path / "x.jsonl")}
    output = Path(job["output"])
    assert not completed_output(job, ["a"], {})
    output.write_text('{"record_id":"a","variant":"base_json_zero"}\n')
    output.with_suffix(".summary.json").write_text(json.dumps({"status": "complete", "n": 0}))
    output.with_suffix(".run.json").write_text("{}")
    assert not completed_output(job, ["a"], {})


def test_complete_resume_verifies_code_prompt_and_saved_parse(tmp_path, monkeypatch):
    import scripts.run_representation_matrix as runner

    job = {"name": "base_json_zero", "representation": "json", "support_set": "zero", "adapter": None,
           "output": str(tmp_path / "x.jsonl")}
    output = Path(job["output"])
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"record_id": "a", "layout_id": "layout", "split": "test"}) + "\n")
    prompts = output.with_suffix(".prompts.jsonl")
    message = [{"role": "user", "content": "current board"}]
    prompts.write_text(json.dumps({"record_id": "a", "layout_id": "layout", "messages": message}) + "\n")
    row = {"record_id": "a", "layout_id": "layout", "split": "test", "variant": "base_json_zero",
           "representation": "json", "support_set": "zero", "raw_output": "broken JSON",
           "syntax_valid": False, "parsed_intent": None}
    output.write_text(json.dumps(row) + "\n")
    output.with_suffix(".summary.json").write_text(json.dumps({"status": "complete", "n": 1}))
    inputs = {key: {"sha256": key, "path": str(data)} for key in
              ("data", "test_ids", "support_data", "support_manifest", "protocol", "generator", "support_selector")}
    settings = {"split": "test", "dtype": "bfloat16", "seed": 1729, "threads": 8,
                "max_length": 1024, "max_new_tokens": 128, "decoding": "greedy",
                "enable_thinking": False, "require_cuda": True}
    model = {"path": "model", "revision": "rev", "sha256": {
        "model.safetensors": "weights", "config.json": "config", "tokenizer.json": "tokenizer"}}
    plan = {"inputs": inputs, "settings": settings, "model": model}
    config = {**settings, "variant": job["name"], "representation": "json", "support_set": "zero",
              "data_sha256": "data", "test_ids_sha256": "test_ids", "support_data_sha256": "support_data",
              "support_manifest_sha256": "support_manifest", "protocol_sha256": "protocol",
              "generator_sha256": "generator", "support_selection_sha256": "support_selector",
              "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
              "model_weights_sha256": "weights", "model_config_sha256": "config", "tokenizer_sha256": "tokenizer",
              "model": "model", "revision": "rev", "adapter": None,
              "adapter_weights_sha256": None, "adapter_config_sha256": None}
    metadata = output.with_suffix(".run.json")
    metadata.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "load_supports", lambda *args: [])
    monkeypatch.setattr(runner, "messages_for", lambda *args: message)
    assert completed_output(job, ["a"], plan)
    metadata.write_text(json.dumps({**config, "protocol_sha256": "changed"}))
    with pytest.raises(ValueError, match="settings differ"):
        completed_output(job, ["a"], plan)
    metadata.write_text(json.dumps(config))
    output.write_text(json.dumps({**row, "syntax_valid": True}) + "\n")
    with pytest.raises(ValueError, match="parse differs"):
        completed_output(job, ["a"], plan)


def test_supervisor_training_detection(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({"stages": [
        {"name": "active", "kind": "training", "status": "running"},
        {"name": "done", "kind": "training", "status": "completed"},
        {"name": "cpu", "kind": "normal", "status": "running"},
    ]}))
    assert training_running(tmp_path) == ["active"]


def test_stopping_owned_subprocess_does_not_stop_unrelated_process():
    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        stop_process_tree(owned)
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        if owned.poll() is None:
            owned.kill()
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_group_records_missing_failure_timeout_and_resume_without_gpu(tmp_path, monkeypatch):
    import scripts.run_representation_matrix as runner

    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"record_ids": ["a", "b"]}))
    jobs = []
    for name in ("missing", "failure", "timeout", "resume"):
        jobs.append({"name": name, "group": "core", "representation": "json", "support_set": "zero",
                     "adapter": None, "output": str(tmp_path / "predictions" / f"{name}.jsonl"),
                     "log": str(tmp_path / "logs" / f"{name}.log")})
    resumed_output = Path(jobs[-1]["output"])
    resumed_output.parent.mkdir()
    resumed_output.write_text('{"partial":true}\n')
    plan = {"inputs": {"test_ids": {"path": str(ids)},
                       **{key: {"path": key} for key in ("data", "support_data", "support_manifest")}},
            "model": {"path": "model", "revision": "sha"},
            "settings": {"split": "test", "dtype": "bfloat16", "seed": 1729, "threads": 8,
                         "max_length": 1024, "max_new_tokens": 128},
            "deadline_epoch": time.time() + 300, "deadline_monotonic": time.monotonic() + 300,
            "methods": jobs}
    plan_path = tmp_path / "matrix_plan.json"
    write_frozen(plan_path, plan)
    monkeypatch.setattr(runner, "ensure_supervised", lambda run: None)
    monkeypatch.setattr(runner, "verify_plan", lambda run, data: None)
    monkeypatch.setattr(runner, "training_running", lambda run: [])
    launched = set()

    def adapter(job, run):
        if job["name"] == "missing":
            raise FileNotFoundError("missing weights")
        return None

    monkeypatch.setattr(runner, "current_adapter", adapter)
    monkeypatch.setattr(runner, "completed_output", lambda job, *_: job["name"] in launched)

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.name = command[command.index("--variant") + 1]
            self.returncode = None
            if self.name == "resume":
                assert "--resume" in command

        def wait(self, timeout):
            if self.name == "timeout":
                raise subprocess.TimeoutExpired("fake", timeout)
            self.returncode = 7 if self.name == "failure" else 0
            if self.returncode == 0:
                launched.add(self.name)
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(runner, "stop_process_tree", lambda process: setattr(process, "returncode", -15))
    assert run_group(tmp_path, plan_path, Group.CORE, 10) == 1
    events = [json.loads(line) for line in (tmp_path / "matrix_jobs.jsonl").read_text().splitlines()]
    terminal = {row["name"]: row["status"] for row in events if row["status"] != "started"}
    assert terminal == {"missing": "missing_adapter", "failure": "failed", "timeout": "timeout", "resume": "complete"}
