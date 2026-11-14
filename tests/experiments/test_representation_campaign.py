"""CPU checks for the autonomous campaign's deadline and ownership guards."""

import json

import pytest

from scripts.run_representation_campaign import Campaign
from scripts.run_representation_campaign import _busy_gpu
from scripts.run_representation_campaign import gpu_exec
from scripts.run_representation_campaign import load_plan
from scripts.run_representation_campaign import optional_seed_fits
from scripts.run_representation_campaign import training_command
from scripts.run_representation_campaign import training_complete


def test_optional_seeds_require_measured_time_and_reserve_pending_evaluations():
    assert optional_seed_fits(10_000, [], [10]) == (False, None)
    assert optional_seed_fits(10_000, [10], []) == (False, None)
    assert optional_seed_fits(10_000, [float("nan")], [10]) == (False, None)
    fits, required = optional_seed_fits(1000, [100, 200, 150], [20, 30])
    assert required == 1.5 * (3 * 200 + 3 * 30) + 300
    assert not fits
    fits, with_pending = optional_seed_fits(2000, [100, 200, 150], [20, 30], pending_evaluations=3)
    assert fits and with_pending is not None and required is not None and with_pending > required
    with pytest.raises(ValueError, match="nonnegative"):
        optional_seed_fits(10_000, [10], [10], pending_evaluations=-1)


def test_training_command_matches_across_representations_except_output_and_format(tmp_path):
    plan = {"deadline_epoch": 1000, "deadline_monotonic": 2000}
    commands = [training_command(tmp_path, representation, 1729, plan) for representation in ("json", "formal", "natural")]
    def normalise(command):
        copy = list(command)
        for option in ("--representation", "--output"):
            copy[copy.index(option) + 1] = "<variable>"
        return copy
    assert normalise(commands[0]) == normalise(commands[1]) == normalise(commands[2])
    assert commands[0][commands[0].index("--accumulation") + 1] == "4"
    assert commands[0][commands[0].index("--max-steps") + 1] == "100"
    assert "--require-cuda" in commands[0]


def test_training_completion_requires_final_validation_and_adapter(tmp_path):
    root = tmp_path / "adapter_repr_formal_seed1729"
    checkpoint = root / "checkpoint-last-100"
    checkpoint.mkdir(parents=True)
    for filename in ("adapter_model.safetensors", "adapter_config.json"):
        (checkpoint / filename).write_text("example")
    status = {"status": "finished", "completed_steps": 100, "last_validation_step": 100}
    config = {"representation": "formal", "seed": 1729, "data_sha256": "dataset", "rank": 4,
              "lora_alpha": 8, "lora_targets": ["q_proj", "v_proj"], "microbatch": 1,
              "accumulation": 4, "learning_rate": .0002, "max_steps": 100, "eval_every": 20,
              "dtype": "bfloat16", "gradient_checkpointing": True, "max_length": 512}
    (root / "status.json").write_text(json.dumps(status))
    (root / "configuration.json").write_text(json.dumps(config))
    assert training_complete(tmp_path, "formal", 1729, "dataset")
    status["last_validation_step"] = 80
    (root / "status.json").write_text(json.dumps(status))
    assert not training_complete(tmp_path, "formal", 1729, "dataset")


def test_registered_gpu_stages_only_are_considered_busy():
    manifest = {"stages": [{"name": "master", "kind": "normal", "status": "running"},
                           {"name": "train", "kind": "training", "status": "running"},
                           {"name": "old", "kind": "expensive", "status": "completed"},
                           {"name": "eval", "kind": "expensive", "status": "running"}]}
    assert _busy_gpu(manifest) == ["train", "eval"]


def test_existing_stage_is_observed_and_never_relaunched(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    (tmp_path / "contrasts_plan.json").write_text("{}")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "active", "stages": [
        {"name": "representation_core", "kind": "expensive", "status": "completed"}]}))
    campaign = Campaign(tmp_path, {"deadline_epoch": 0, "deadline_monotonic": 0})
    def fail_if_launched(*_args, **_kwargs):
        raise AssertionError("must not retry an already registered stage")
    monkeypatch.setattr("subprocess.Popen", fail_if_launched)
    assert campaign.run_stage("representation_core", "expensive", ["never"], 30, gpu=True) == "completed"
    journal = (tmp_path / "campaign_jobs.jsonl").read_text()
    assert '"reason": "already_registered"' in journal


def test_running_stage_is_waited_for_and_never_relaunched(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    (tmp_path / "contrasts_plan.json").write_text("{}")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "active", "stages": [
        {"name": "train_repr_formal_r4", "kind": "training", "status": "running"}]}))
    campaign = Campaign(tmp_path, {"deadline_epoch": 0, "deadline_monotonic": 0})
    waited = []
    monkeypatch.setattr(campaign, "wait_stage", lambda name, *, matrix_deadline: waited.append(name) or "completed")
    def fail_if_launched(*_args, **_kwargs):
        raise AssertionError("must not start another copy of a running stage")
    monkeypatch.setattr("subprocess.Popen", fail_if_launched)
    assert campaign.run_stage("train_repr_formal_r4", "training", ["never"], 30, gpu=True) == "completed"
    assert waited == ["train_repr_formal_r4"]


def test_deadline_prevents_starting_new_expensive_work(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    (tmp_path / "contrasts_plan.json").write_text("{}")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"status": "active", "stages": []}))
    campaign = Campaign(tmp_path, {"deadline_epoch": 0, "deadline_monotonic": 0})
    monkeypatch.setattr(campaign, "wait_gpu_idle", lambda: False)
    def fail_if_launched(*_args, **_kwargs):
        raise AssertionError("must not start GPU work after cutoff")
    monkeypatch.setattr("subprocess.Popen", fail_if_launched)
    assert campaign.run_stage("representation_secondary", "expensive", ["never"], 30, gpu=True) == "not_started"
    assert campaign.failures


def test_gpu_wrapper_holds_shared_lock_through_exec(tmp_path, monkeypatch):
    import fcntl
    import os

    manifest = {"status": "active", "boot_id": "boot", "stages": [
        {"pid": 111, "kind": "training", "status": "running"}]}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("scripts.run_representation_campaign.boot_id", lambda: "boot")
    monkeypatch.setattr(os, "getpgrp", lambda: 111)
    monkeypatch.setattr(os, "getsid", lambda _pid: 111)
    class ExecIntercept(Exception):
        pass
    def fake_exec(_program, command, _env):
        assert command == ["python", "train"]
        with (tmp_path / ".matrix_gpu.lock").open("a+") as another:
            with pytest.raises(BlockingIOError):
                fcntl.flock(another, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raise ExecIntercept
    monkeypatch.setattr(os, "execvpe", fake_exec)
    with pytest.raises(ExecIntercept):
        gpu_exec(tmp_path, ["python", "train"])


def test_plan_refuses_changed_checksum(tmp_path):
    plan = tmp_path / "matrix_plan.json"
    plan.write_text("{}")
    plan.with_suffix(".sha256").write_text("wrong\n")
    with pytest.raises(ValueError, match="checksum"):
        load_plan(tmp_path, {})
