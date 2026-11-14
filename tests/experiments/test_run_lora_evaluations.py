"""A vanished child must not prevent stopping its evaluator on timeout."""

import hashlib
import json
from types import SimpleNamespace

import psutil
import pytest

from scripts import run_lora_evaluations as runner


def test_timeout_still_stops_parent_if_a_child_disappears(monkeypatch):
    calls = []

    class Vanished:
        def terminate(self):
            raise psutil.NoSuchProcess(12)

    class Parent:
        def children(self, recursive):
            assert recursive
            return [Vanished()]

        def terminate(self):
            calls.append("parent_terminated")

    monkeypatch.setattr(runner.psutil, "Process", lambda _pid: Parent())
    monkeypatch.setattr(runner.psutil, "wait_procs", lambda members, timeout: (members, []))
    process = SimpleNamespace(pid=10, wait=lambda timeout: calls.append("waited"))
    runner.stop_process_tree(process)
    assert calls == ["parent_terminated", "waited"]


def test_icl_transfer_ids_match_the_frozen_order_and_dataset(tmp_path):
    directory = tmp_path / "eval_transfer"
    directory.mkdir()
    dataset = directory / "dataset.jsonl"
    dataset.write_text('{"record_id":"a"}\n{"record_id":"b"}\n')
    canonical = directory / "test_ids.json"
    canonical.write_text(json.dumps({"record_ids": ["a", "b"]}))
    enriched = directory / "icl_test_ids.json"
    enriched.write_text(json.dumps({"record_ids": ["a", "b"],
                                    "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}))
    assert runner.transfer_icl_ids(tmp_path) == enriched
    enriched.write_text(json.dumps({"record_ids": ["b", "a"],
                                    "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match="frozen selection"):
        runner.transfer_icl_ids(tmp_path)
    enriched.write_text(json.dumps({"record_ids": ["a", "b"], "dataset_sha256": "different"}))
    with pytest.raises(ValueError, match="different dataset"):
        runner.transfer_icl_ids(tmp_path)
