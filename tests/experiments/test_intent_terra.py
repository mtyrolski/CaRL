"""Tests of the isolated pair-only annotation protocol; no model calls."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from carl.icl.intent_terra import AnnotatorModel
from carl.icl.intent_terra import ReasoningEffort
from carl.icl.intent_terra import _kill_group
from carl.icl.intent_terra import _proc_start
from carl.icl.intent_terra import attempt
from carl.icl.intent_terra import build_prompt
from carl.icl.intent_terra import extract_usage
from carl.icl.intent_terra import freeze_selection
from carl.icl.intent_terra import inspect_events
from carl.icl.intent_terra import main
from carl.icl.intent_terra import merge_accepted
from carl.icl.intent_terra import output_schema
from carl.icl.intent_terra import total_usage
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


def test_usage_counts_are_reported_without_guessing_missing_or_ambiguous_values(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps({"type": "turn.started"}) + "\n")
    unknown = {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None}
    assert extract_usage(path) == (unknown, "not_reported")
    complete = {"input_tokens": 120, "cached_input_tokens": 30, "output_tokens": 15}
    path.write_text(json.dumps({"type": "turn.completed", "usage": complete}) + "\n")
    assert extract_usage(path) == (complete, "complete")
    partial = {"input_tokens": 120, "output_tokens": 15}
    path.write_text(json.dumps({"type": "turn.completed", "usage": partial}) + "\n")
    assert extract_usage(path) == ({**partial, "cached_input_tokens": None}, "partial")
    path.write_text((json.dumps({"type": "turn.completed", "usage": complete}) + "\n") * 2)
    assert extract_usage(path) == (unknown, "ambiguous")
    assert total_usage([{
        "usage": complete
    }, {
        "usage": partial
    }]) == {
        "input_tokens": 240,
        "cached_input_tokens": None,
        "output_tokens": 30
    }
    assert total_usage([]) == unknown


def test_summary_preserves_model_and_unknown_cost_across_retries(tmp_path, monkeypatch):
    selection = tmp_path / "selection.jsonl"
    record = {"record_id": "example", "split": "train", "start_observation": START, "later_observation": LATER}
    selection.write_text(json.dumps(record) + "\n")
    clock = tmp_path / "clock.json"
    now = time.monotonic()
    clock.write_text(json.dumps({"start_monotonic": now, "deadline_monotonic": now + 120}))
    unknown = {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None}
    actual = {"input_tokens": 120, "cached_input_tokens": 30, "output_tokens": 15}
    attempts = []

    def fake_attempt(*args):
        attempts.append(args)
        assert args[6] == AnnotatorModel.LUNA and args[7] == ReasoningEffort.LOW
        return {
            "errors": ["timeout"] if len(attempts) == 1 else [],
            "accepted": [],
            "usage": unknown if len(attempts) == 1 else actual,
            "usage_status": "not_reported" if len(attempts) == 1 else "complete",
            "auth_type": "saved_codex_auth",
            "credential_source": "saved_codex_auth"
        }

    monkeypatch.setattr("carl.icl.intent_terra.attempt", fake_attempt)
    monkeypatch.setattr(sys, "argv", [
        "intent_terra", "--selection",
        str(selection), "--output",
        str(tmp_path / "out"), "--clock",
        str(clock), "--model",
        str(AnnotatorModel.LUNA), "--reasoning-effort",
        str(ReasoningEffort.LOW), "--retry", "1"
    ])
    main()
    summary = json.loads((tmp_path / "out/batch_0000_summary.json").read_text())
    assert summary["provenance"]["model"] == AnnotatorModel.LUNA
    assert summary["provenance"]["reasoning_effort"] == ReasoningEffort.LOW
    assert summary["usage"] == unknown
    assert [row["usage_status"] for row in summary["usage_by_attempt"]] == ["not_reported", "complete"]


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
    assert result["credential_source"] == "environment"
    assert fake_key not in json.dumps(result)
    assert fake_key not in (tmp_path / "attempt" / "result.json").read_text()


def test_dotenv_key_is_selected_for_worker_without_leaking_or_mutating_environment(tmp_path, monkeypatch):
    fake_codex = tmp_path / "codex"
    audit = tmp_path / "audit.json"
    dotenv = tmp_path / "credentials.env"
    fake_key = "test-dotenv-key-for-worker-only"
    dotenv.write_text(f"OPENAI_API_KEY={fake_key}\n")
    fake_codex.write_text("#!/usr/bin/python3\nimport json, os, pathlib, sys\n"
                          f"pathlib.Path({str(audit)!r}).write_text(json.dumps({{"
                          f"'mapped': os.environ.get('CODEX_API_KEY') == {fake_key!r}, "
                          "'openai_key_present': 'OPENAI_API_KEY' in os.environ, 'args': sys.argv[1:]}))\n"
                          "print(json.dumps({'type':'turn.started'}))\nsys.exit(1)\n")
    fake_codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("OPENAI_API_KEY", "other-environment-key")
    monkeypatch.setenv("CODEX_API_KEY", "other-codex-key")
    record = {"record_id": "example", "start_observation": START, "later_observation": LATER}
    result = attempt([record], tmp_path / "attempt", 2, time.monotonic() + 4, api_key_dotenv=dotenv)
    observed = json.loads(audit.read_text())
    assert observed["mapped"] and not observed["openai_key_present"]
    assert result["auth_type"] == "api_key" and result["credential_source"] == "dotenv"
    assert os.environ["OPENAI_API_KEY"] == "other-environment-key"
    assert os.environ["CODEX_API_KEY"] == "other-codex-key"
    assert fake_key not in json.dumps(result)
    assert fake_key not in (tmp_path / "attempt" / "result.json").read_text()


@pytest.mark.parametrize("model,effort", [(AnnotatorModel.TERRA, None), (AnnotatorModel.LUNA, ReasoningEffort.LOW)])
def test_requested_model_and_effort_are_sent_and_recorded_without_model_calls(tmp_path, monkeypatch, model, effort):
    fake_codex = tmp_path / "codex"
    answer = {
        "annotations": [{
            "pair_index": 0,
            "intent": {
                "effects": [{
                    "predicate": "box_at",
                    "row": 1,
                    "col": 3
                }],
                "preserve": [],
                "instruction": None
            }
        }]
    }
    fake_codex.write_text("#!/usr/bin/python3\nimport json,pathlib,sys\n"
                          "args=sys.argv[1:]\n"
                          f"pathlib.Path(args[args.index('-o')+1]).write_text(json.dumps({answer!r}))\n"
                          "sys.stdin.read()\n"
                          "print(json.dumps({'type':'turn.started'}))\n"
                          "print(json.dumps({'type':'item.completed','item':{'type':'agent_message'}}))\n"
                          "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':120,"
                          "'cached_input_tokens':30,'output_tokens':15}}))\n")
    fake_codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    record = {"record_id": "example", "start_observation": START, "later_observation": LATER}
    result = attempt([record], tmp_path / "attempt", 2, time.monotonic() + 4, model=model, reasoning_effort=effort)
    assert not result["errors"]
    command = result["command"]
    assert command[command.index("-m") + 1] == model
    assert result["model"] == model and result["reasoning_effort"] == effort
    assert result["usage"] == {"input_tokens": 120, "cached_input_tokens": 30, "output_tokens": 15}
    assert result["usage_status"] == "complete"
    flags = [item for item in command if item.startswith("model_reasoning_effort=")]
    assert flags == ([] if effort is None else [f'model_reasoning_effort="{effort}"'])


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


@pytest.mark.parametrize("model,effort", [(AnnotatorModel.TERRA, None), (AnnotatorModel.LUNA, ReasoningEffort.LOW)])
def test_merge_keeps_raw_provenance_and_excludes_unchecked_text(tmp_path, model, effort):
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
    command = ["codex", "exec", "-m", model]
    if effort is not None:
        command.extend(["-c", f'model_reasoning_effort="{effort}"'])
    result = {
        "errors": [],
        "accepted": accepted,
        "model": model,
        "reasoning_effort": effort,
        "usage": {
            "input_tokens": 120,
            "cached_input_tokens": 30,
            "output_tokens": 15
        },
        "usage_status": "complete",
        "command": command,
        "batch_size": 1,
        "batch_coupling_possible": False,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(schema.encode()).hexdigest()
    }
    (attempt_dir / "result.json").write_text(json.dumps(result))
    (attempt_dir / "answer.json").write_text(json.dumps({"annotations": [{"pair_index": 0, "intent": intent}]}))
    events = [{"type": "turn.started"}, {"type": "turn.completed", "usage": result["usage"]}]
    (attempt_dir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (attempt_dir / "request.txt").write_text(prompt)
    (attempt_dir / "schema.json").write_text(schema)
    summary = {
        "status": "accepted",
        "accepted": accepted,
        "provenance": {
            "attempts": 1,
            "model": model,
            "reasoning_effort": effort,
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
    assert merged["annotation_provenance"]["model"] == model
    assert merged["annotation_provenance"]["reasoning_effort"] == effort
    assert merged["annotation_provenance"]["usage"]["input_tokens"] == 120
    assert "entire batch" in merged["annotation_provenance"]["usage_scope"]
    assert merged["annotation_provenance"]["instruction_excluded_from_training"]

    summary["provenance"]["model"] = AnnotatorModel.LUNA if model == AnnotatorModel.TERRA else AnnotatorModel.TERRA
    (summaries / "batch_0000_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="model provenance"):
        merge_accepted(source, summaries, tmp_path / "mismatched.jsonl")

    summary["provenance"]["model"] = model
    (summaries / "batch_0000_summary.json").write_text(json.dumps(summary))
    result["command"] = ["codex", "exec", "-m", "gpt-6-luna" if model == AnnotatorModel.TERRA else "gpt-6.1-terra"]
    (attempt_dir / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="command model"):
        merge_accepted(source, summaries, tmp_path / "mismatched_command.jsonl")

    result["command"] = command
    result["usage"]["input_tokens"] = 999
    (attempt_dir / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="usage does not match"):
        merge_accepted(source, summaries, tmp_path / "mismatched_usage.jsonl")

    if model == AnnotatorModel.TERRA:
        result.pop("reasoning_effort")
        result.pop("usage")
        result.pop("usage_status")
        summary["provenance"].pop("reasoning_effort")
        (attempt_dir / "result.json").write_text(json.dumps(result))
        (summaries / "batch_0000_summary.json").write_text(json.dumps(summary))
        (attempt_dir / "events.jsonl").write_text(json.dumps({"type": "turn.started"}) + "\n")
        legacy_output = tmp_path / "legacy.jsonl"
        merge_accepted(source, summaries, legacy_output)
        legacy = json.loads(legacy_output.read_text())["annotation_provenance"]
        assert legacy["model"] == AnnotatorModel.TERRA
        assert legacy["reasoning_effort"] is None
        assert legacy["usage"] == {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None}
        assert legacy["usage_status"] == "not_reported"
