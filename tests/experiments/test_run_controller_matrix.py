"""The sequential controller runner respects time limits and records failures."""

import hashlib
import json
import sys

import pytest

from scripts.run_controller_matrix import JobStatus
from scripts.run_controller_matrix import preserve_partial_tail
from scripts.run_controller_matrix import remaining_seconds
from scripts.run_controller_matrix import run_child
from scripts.run_controller_matrix import start_allowed
from scripts.run_controller_matrix import verify_controller_assets


def test_cutoff_and_reporting_reserve_use_both_clocks(monkeypatch):
    monkeypatch.setattr("scripts.run_controller_matrix.time.time", lambda: 1000)
    monkeypatch.setattr("scripts.run_controller_matrix.time.monotonic", lambda: 500)
    manifest = {"clock": {"deadline_epoch": 4000, "deadline_monotonic": 3000},
                "expensive_start_cutoff_epoch": 2000, "expensive_start_cutoff_monotonic": 1500}
    assert remaining_seconds(manifest) == 700
    assert start_allowed(manifest)
    manifest["expensive_start_cutoff_monotonic"] = 500
    assert not start_allowed(manifest)


def test_child_exit_and_timeout_are_not_reported_as_success(tmp_path):
    log = tmp_path / "child.log"
    log.write_text("previous attempt\n")
    assert run_child([sys.executable, "-c", "raise SystemExit(3)"], log, 5) == (JobStatus.FAILED, 3)
    status, _ = run_child([sys.executable, "-c", "import time; time.sleep(5)"], log, .1)
    assert status == JobStatus.TIMEOUT
    assert "previous attempt" in log.read_text()


def test_frozen_controller_assets_are_checked(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    names = ("config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")
    for name in names:
        (model / name).write_text(name)
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected_config": {"model": str(model)}}))
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot = tmp_path / "assets.json"
    snapshot.write_text(json.dumps({"model": str(model), "selection_sha256": sha(selection),
                                    "files": {name: {"sha256": sha(model / name)} for name in names}}))
    verify_controller_assets(snapshot, selection)
    (model / "generation_config.json").write_text("changed")
    with pytest.raises(ValueError, match="asset changed"):
        verify_controller_assets(snapshot, selection)


def test_interrupted_tail_is_quarantined_before_resume(tmp_path):
    output = tmp_path / "rollouts.jsonl"
    output.write_bytes(b'{"record_id":"a"}\n{"record_id":')
    preserve_partial_tail(output)
    assert output.read_bytes() == b'{"record_id":"a"}\n'
    quarantine = list(tmp_path.glob("rollouts.jsonl.partial-*"))
    assert len(quarantine) == 1 and quarantine[0].read_bytes() == b'{"record_id":'


def test_complete_unterminated_line_is_retained(tmp_path):
    output = tmp_path / "rollouts.jsonl"
    output.write_bytes(b'{"record_id":"a"}')
    preserve_partial_tail(output)
    assert output.read_bytes() == b'{"record_id":"a"}\n'
