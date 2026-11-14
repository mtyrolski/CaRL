"""Tests of the isolated pair-only annotation protocol; no model calls."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time

from carl.icl.intent_terra import _kill_group
from carl.icl.intent_terra import _proc_start
from carl.icl.intent_terra import attempt
from carl.icl.intent_terra import build_prompt
from carl.icl.intent_terra import freeze_selection
from carl.icl.intent_terra import inspect_events
from carl.icl.intent_terra import merge_accepted
from carl.icl.intent_terra import output_schema
from carl.icl.intent_terra import validate_answer

START = ["#####", "#@$.#", "#####"]
LATER = ["#####", "#-@*#", "#####"]


def test_prompt_selects_only_pair_observations():
    record = {
        "start_observation": START,
        "later_observation": LATER,
        "fragment_actions": ["SECRET_ACTION"],
        "trajectory_id": "SECRET_SOURCE",
        "intent": {
            "text": "SECRET_LABEL"
        },
    }
    prompt = build_prompt([record])
    assert START[1] in prompt and LATER[1] in prompt
    assert "SECRET" not in prompt
    assert output_schema(1)["properties"]["annotations"]["maxItems"] == 1


def test_rejects_false_effect_and_nonempty_preservation():
    record = {"record_id": "example", "start_observation": START, "later_observation": LATER}
    intent = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": [], "instruction": None}
    answer = {"annotations": [{"pair_index": 0, "intent": intent}]}
    checked, errors = validate_answer([record], answer)
    assert not errors and checked[0]["intent"] == intent
    intent["effects"][0]["col"] = 1
    assert validate_answer([record], answer)[1]
    intent["preserve"] = [{"predicate": "player_at", "row": 1, "col": 1}]
    assert any("preserve_must_be_empty" in err for err in validate_answer([record], answer)[1])


def test_rejects_worker_tool_events(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [{"type": "turn.started"}, {"type": "item.completed", "item": {"type": "command_execution"}}]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    errors, count = inspect_events(path)
    assert count == 2
    assert "disallowed_item_type:command_execution" in errors


def test_timeout_registers_and_stops_the_owned_process_group(tmp_path, monkeypatch):
    fake_codex = tmp_path / "codex"
    fake_codex.write_text("#!/bin/sh\nsleep 10\n")
    fake_codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    registry = tmp_path / "workers.jsonl"
    record = {"record_id": "example", "start_observation": START, "later_observation": LATER}
    result = attempt([record], tmp_path / "attempt", 0.1, time.monotonic() + 2, registry)
    assert result["timed_out"]
    assert "timeout" in result["errors"]
    events = [json.loads(line) for line in registry.read_text().splitlines()]
    assert [item["phase"] for item in events] == ["started", "finished"]
    assert events[0]["pid"] == events[0]["pgid"] == events[1]["pid"]
    assert events[0]["proc_start"] == events[1]["proc_start"]


def test_group_cleanup_stops_descendant_that_ignores_termination(tmp_path):
    ready = tmp_path / "ready"
    leaked = tmp_path / "leaked"
    child = ("import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
             "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(1.0); "
             "pathlib.Path(sys.argv[2]).write_text('still running'); time.sleep(10)")
    leader = ("import subprocess,sys,time; "
              "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]]); time.sleep(10)")
    proc = subprocess.Popen([sys.executable, "-c", leader, child, str(ready), str(leaked)], start_new_session=True)
    child_pid = None
    child_start = None
    try:
        end = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < end:
            time.sleep(0.01)
        assert ready.exists()
        child_pid = int(ready.read_text())
        child_start = _proc_start(child_pid)
        _kill_group(proc)
        time.sleep(1.1)
        assert not leaked.exists()
    finally:
        if proc.poll() is None:
            _kill_group(proc)
        if child_pid is not None:
            try:
                if _proc_start(child_pid) == child_start:
                    os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass


def test_api_key_is_mapped_only_in_worker_environment_without_logging(tmp_path, monkeypatch):
    fake_codex = tmp_path / "codex"
    audit = tmp_path / "audit.json"
    fake_key = "test-secret-for-worker-only"
    fake_codex.write_text("#!/usr/bin/python3\nimport json, os, pathlib, sys\n"
                          f"pathlib.Path({str(audit)!r}).write_text(json.dumps({{"
                          f"'mapped': os.environ.get('CODEX_API_KEY') == {fake_key!r}, "
                          "'openai_key_present': 'OPENAI_API_KEY' in os.environ, 'args': sys.argv[1:]}))\n"
                          "print(json.dumps({'type':'turn.started'}))\nsys.exit(1)\n")
    fake_codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("OPENAI_API_KEY", fake_key)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    record = {"record_id": "example", "start_observation": START, "later_observation": LATER}
    result = attempt([record], tmp_path / "attempt", 2, time.monotonic() + 4, api_key_env="OPENAI_API_KEY")
    observed = json.loads(audit.read_text())
    assert observed["mapped"] and not observed["openai_key_present"]
    assert 'shell_environment_policy.inherit="none"' in observed["args"]
    assert result["auth_type"] == "api_key"
    assert fake_key not in json.dumps(result)
    assert fake_key not in (tmp_path / "attempt" / "result.json").read_text()


def test_selection_is_train_only_and_balances_layouts_before_answers(tmp_path):
    rows = [{
        "record_id": f"{split}-{layout}-{i}",
        "layout_id": layout,
        "split": split,
        "start_observation": START,
        "later_observation": LATER,
        "intent": "SECRET",
        "annotation_provenance": {
            "kind": "local_exact_difference"
        }
    } for split in ("train", "test") for layout in ("one", "two") for i in range(3)]
    rows[0]["annotation_provenance"] = {"kind": "historical_shared_context_codex_annotation"}
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    freeze_selection(source, first, count=4, seed=11)
    freeze_selection(source, second, count=4, seed=11)
    assert first.read_text() == second.read_text()
    selected = [json.loads(line) for line in first.read_text().splitlines()]
    assert all(row["split"] == "train" and "intent" not in row for row in selected)
    assert rows[0]["record_id"] not in {row["record_id"] for row in selected}
    assert len({row["layout_id"] for row in selected[:2]}) == 2


def test_merge_keeps_raw_provenance_and_excludes_unchecked_text(tmp_path):
    record = {
        "record_id": "train-1",
        "split": "train",
        "start_observation": START,
        "later_observation": LATER,
        "annotation_provenance": {
            "kind": "local_exact_difference"
        }
    }
    source = tmp_path / "dataset.jsonl"
    source.write_text(json.dumps(record) + "\n")
    summaries = tmp_path / "terra"
    attempt_dir = summaries / "batch_0000_attempt_1"
    attempt_dir.mkdir(parents=True)
    intent = {
        "effects": [{
            "predicate": "box_at",
            "row": 1,
            "col": 3
        }],
        "preserve": [],
        "instruction": "Unverified text stays out of the target."
    }
    accepted = [{"record_id": "train-1", "intent": intent}]
    prompt = "request\n"
    schema = "schema\n"
    result = {
        "errors": [],
        "accepted": accepted,
        "batch_size": 1,
        "batch_coupling_possible": False,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(schema.encode()).hexdigest()
    }
    (attempt_dir / "result.json").write_text(json.dumps(result))
    (attempt_dir / "answer.json").write_text(json.dumps({"annotations": [{"pair_index": 0, "intent": intent}]}))
    (attempt_dir / "events.jsonl").write_text(json.dumps({"type": "turn.started"}) + "\n")
    (attempt_dir / "request.txt").write_text(prompt)
    (attempt_dir / "schema.json").write_text(schema)
    summary = {
        "status": "accepted",
        "accepted": accepted,
        "provenance": {
            "attempts": 1
        },
        "selected_record_ids": ["train-1"]
    }
    (summaries / "batch_0000_summary.json").write_text(json.dumps(summary))
    output = tmp_path / "merged.jsonl"
    manifest = merge_accepted(source, summaries, output)
    merged = json.loads(output.read_text())
    assert manifest["accepted_model_annotations"] == 1
    assert merged["intent"]["instruction"] is None
    assert merged["annotation_provenance"]["kind"] == "fresh_isolated_codex_annotation"
    assert merged["annotation_provenance"]["instruction_excluded_from_training"]
