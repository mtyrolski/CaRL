"""Offline tests for the future budgeted Luna controller runner."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_transport import LunaRequestUncertain
from scripts import run_luna_lora_controller as runner

ROWS = ["#######", "#@$-.-#", "#-$-.-#", "#-----#", "#-----#", "#-----#", "#######"]
INTENT = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": []}


def record(identifier: str, context: str = "context", intent: dict | None = None) -> dict:
    variants = {"a": INTENT,
                "b": {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": []},
                "c": {"effects": [{"predicate": "player_at", "row": 3, "col": 1}], "preserve": []}}
    return {"record_id": identifier, "context_id": context, "split": "train", "phase": "calibration",
            "family": "synthetic", "layout_id": "layout", "start_observation": ROWS,
            "task_goal": "Put boxes on goals.", "intent": variants.get(identifier, INTENT) if intent is None else intent}


def make_plan(tmp_path: Path, records: list[dict], *, attempts: int = 1) -> dict:
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return runner.make_plan(records, source, tmp_path / "budget", tmp_path / "output", attempts, 16, 45, "2.00")


def result(identifier: str, attempt: int, status: RolloutStatus, actions: int) -> dict:
    return {"record_id": identifier, "attempt": attempt,
            "result": {"status": status, "actions": [0] * actions, "model_calls": 1}}


def test_plan_only_does_not_make_transport_or_create_run(tmp_path, monkeypatch, capsys):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps(record("a")) + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "make_transport", lambda *args: pytest.fail("No API transport in plan mode"))
    status = runner.main(["--input", str(source), "--budget-dir", str(tmp_path / "budget"),
                          "--output-dir", str(tmp_path / "output")])
    assert status == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "plan_only"
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "budget").exists()


def test_controller_hash_covers_prompt_environment_and_runner_sources():
    first = runner.controller_config(16, 45)
    second = runner.controller_config(16, 45)
    assert runner.sha256_json(first) == runner.sha256_json(second)
    assert {"formatting", "data_source", "types", "environment_core", "environment_tokenizer", "runner"} <= set(
        first["code_sha256"])
    assert all(len(value) == 64 for value in first["code_sha256"].values())


def test_weighted_percentiles_use_success_fraction_and_tied_midpoint(tmp_path):
    records = [record("a"), record("b"), record("c")]
    plan = make_plan(tmp_path, records, attempts=2)
    results = [result("a", 1, RolloutStatus.ACHIEVED, 1), result("a", 2, RolloutStatus.ACHIEVED, 1),
               result("b", 1, RolloutStatus.ACHIEVED, 3), result("b", 2, RolloutStatus.ACTION_LIMIT, 16),
               result("c", 1, RolloutStatus.ACHIEVED, 2), result("c", 2, RolloutStatus.ACHIEVED, 4)]
    values = {row["record_id"]: row["policy_measurements"]
              for row in runner.calibrated_records(records, results, plan)}
    assert all(row["complete"] and row["expected_candidates"] == 3 for row in values.values())
    assert values["a"]["success_fraction"] == 1
    assert values["b"]["success_fraction"] == 0.5
    assert values["c"]["success_fraction"] == 1
    assert values["a"]["weight"] == pytest.approx(0.4)
    assert values["b"]["weight"] == pytest.approx(0.2)
    assert values["c"]["weight"] == pytest.approx(0.4)
    assert values["a"]["u"] == pytest.approx(0.2)
    assert values["b"]["u"] == pytest.approx(0.7)
    assert values["c"]["u"] == pytest.approx(0.7)
    assert values["b"]["attempts"][1] == {"status": "action_limit", "primitive_actions_attempted": 16}


def test_incomplete_context_does_not_assign_partial_labels(tmp_path):
    records = [record("a"), record("b")]
    plan = make_plan(tmp_path, records, attempts=2)
    results = [result("a", 1, RolloutStatus.ACHIEVED, 1), result("a", 2, RolloutStatus.ACHIEVED, 2),
               result("b", 1, RolloutStatus.ACHIEVED, 3)]
    values = [row["policy_measurements"] for row in runner.calibrated_records(records, results, plan)]
    assert all(not row["complete"] for row in values)
    assert all(row["u"] is None and row["weight"] is None and row["success_fraction"] is None
               and row["mean_success_actions"] is None for row in values)
    assert [len(row["attempts"]) for row in values] == [2, 1]


def test_zero_success_has_no_scale_or_duration(tmp_path):
    records = [record("a"), record("b")]
    plan = make_plan(tmp_path, records)
    results = [result("a", 1, RolloutStatus.ACHIEVED, 2), result("b", 1, RolloutStatus.INVALID_RESPONSE, 0)]
    rows = runner.calibrated_records(records, results, plan)
    successful, failed = (row["policy_measurements"] for row in rows)
    assert successful["u"] == 0.5 and successful["weight"] == 1
    assert failed["complete"] and failed["success_fraction"] == 0
    assert failed["mean_success_actions"] is None and failed["u"] is None and failed["weight"] is None


def test_test_phase_never_receives_calibration_scale(tmp_path):
    records = [{**record("a"), "phase": "test", "split": "test"},
               {**record("b"), "phase": "test", "split": "test"}]
    plan = make_plan(tmp_path, records)
    results = [result("a", 1, RolloutStatus.ACHIEVED, 1), result("b", 1, RolloutStatus.ACHIEVED, 3)]
    rows = runner.calibrated_records(records, results, plan)
    assert all(row["policy_measurements"]["complete"] for row in rows)
    assert all(row["policy_measurements"]["u"] is None and row["policy_measurements"]["weight"] is None
               for row in rows)


def test_unexpected_duplicate_or_missing_candidate_cannot_change_reference_set(tmp_path):
    records = [record("a"), record("b")]
    plan = make_plan(tmp_path, records)
    item = result("a", 1, RolloutStatus.ACHIEVED, 1)
    with pytest.raises(ValueError, match="duplicate"):
        runner.calibrated_records(records, [item, item], plan)
    with pytest.raises(ValueError, match="frozen candidate identities"):
        runner.calibrated_records(records[:1], [item], plan)
    with pytest.raises(ValueError, match="expected_candidates"):
        runner.validate_records([{**records[0], "expected_candidates": 3}, records[1]])
    with pytest.raises(ValueError, match="repeats a calibration intent"):
        runner.validate_records([records[0], {**records[1], "intent": records[0]["intent"]}])


class FakeTransport:
    """Answers locally; the recorded call count exposes accidental result reuse."""

    def __init__(self, *, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after

    def create(self, _prompt, *, tag):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("synthetic infrastructure failure")
        return SimpleNamespace(output_text='{"action":"RIGHT"}')


def test_distinct_replicates_make_distinct_transport_calls(tmp_path):
    records = [record("a")]
    plan = make_plan(tmp_path, records, attempts=2)
    output = tmp_path / "output"
    output.mkdir()
    transport = FakeTransport()
    assert runner.execute(records, plan, transport, output) == 0
    assert transport.calls == 2
    results = runner.read_jsonl(output / "results.jsonl")
    assert [row["attempt"] for row in results] == [1, 2]
    row = runner.read_jsonl(output / "calibrated.jsonl")[0]["policy_measurements"]
    assert len(row["attempts"]) == 2 and row["complete"] and row["u"] == 0.5


def test_invalid_and_api_errors_remain_separate_from_unstarted(tmp_path):
    invalid = {"effects": [{"predicate": "box_at", "row": 1, "col": 2}], "preserve": []}
    records = [record("a", "c1"), record("b", "c2", invalid),
               record("c", "c3"), record("d", "c4")]
    plan = make_plan(tmp_path, records)
    output = tmp_path / "output"
    output.mkdir()
    transport = FakeTransport(fail_after=1)
    assert runner.execute(records, plan, transport, output) == 1
    assert transport.calls == 2
    rows = {row["record_id"]: row["policy_measurements"]
            for row in runner.read_jsonl(output / "calibrated.jsonl")}
    assert rows["a"]["status"] == "complete"
    assert rows["b"]["status"] == "invalid_intent" and rows["b"]["attempts"] == []
    assert rows["c"]["status"] == "infrastructure_error" and rows["c"]["attempts"] == []
    assert rows["d"]["status"] == "not_started" and rows["d"]["attempts"] == []
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "stopped_api_error" and status["not_started"] == 1


def test_interrupted_started_attempt_is_not_reported_as_unstarted(tmp_path):
    class InterruptedTransport:
        def __init__(self, output):
            self.output = output
            self.snapshot_checked = False

        def create(self, _prompt, *, tag):
            rows = runner.read_jsonl(self.output / "calibrated.jsonl")
            status = json.loads((self.output / "status.json").read_text(encoding="utf-8"))
            assert rows[0]["policy_measurements"]["status"] == "incomplete"
            assert rows[1]["policy_measurements"]["status"] == "not_started"
            assert status["started_without_saved_result"] == 1 and status["not_started"] == 1
            self.snapshot_checked = True
            raise KeyboardInterrupt

    records = [record("a", "c1"), record("b", "c2")]
    plan = make_plan(tmp_path, records)
    output = tmp_path / "output"
    output.mkdir()
    transport = InterruptedTransport(output)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(records, plan, transport, output)
    assert transport.snapshot_checked
    rows = {row["record_id"]: row["policy_measurements"]
            for row in runner.read_jsonl(output / "calibrated.jsonl")}
    assert rows["a"]["status"] == "incomplete" and rows["a"]["attempts"] == []
    assert rows["b"]["status"] == "not_started"
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "interrupted" and status["started_without_saved_result"] == 1
    assert status["not_started"] == 1


def test_shared_budget_keeps_unknown_and_blocks_second_phase(tmp_path, monkeypatch):
    class NoRequestClient:
        responses = SimpleNamespace(create=lambda **kwargs: pytest.fail("No network call is allowed"))

        def with_options(self, **_kwargs):
            return self

    monkeypatch.setattr(runner, "dotenv_values", lambda _: {"OPENAI_API_KEY": "synthetic-key"})
    monkeypatch.setattr(runner, "OpenAI", lambda **_kwargs: NoRequestClient())
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    one = runner.make_transport(tmp_path / "budget", first, 45, "2")
    two = runner.make_transport(tmp_path / "budget", second, 45, "2")
    assert one.transport.ledger.path == two.transport.ledger.path
    with sqlite3.connect(one.transport.ledger.path) as database:
        database.execute("INSERT INTO attempts(id,stage,tag,endpoint,model,request_hash,reserved_nanos,status) "
                         "VALUES ('synthetic','experiment','','responses','gpt-6-luna','hash',1,'unknown')")
    with pytest.raises(LunaRequestUncertain):
        two.create("synthetic prompt", tag="candidate")
    with sqlite3.connect(one.transport.ledger.path) as database:
        assert database.execute("SELECT status,reserved_nanos FROM attempts").fetchall() == [("unknown", 1)]


def test_history_is_read_only_and_remains_separate(tmp_path, monkeypatch):
    ledger = tmp_path / "historical.sqlite3"
    with sqlite3.connect(ledger) as database:
        database.execute("CREATE TABLE attempts(status TEXT,reserved_nanos INTEGER)")
        database.executemany("INSERT INTO attempts VALUES (?,?)", [("unknown", 3), ("unknown", 5), ("settled", 7)])
    before = ledger.read_bytes()
    monkeypatch.setattr(runner, "HISTORICAL_LEDGER", ledger)
    history = runner.history_summary()
    assert history["unknown_attempts"] == 2 and history["unknown_reserved_usd"] == 8e-9
    assert not history["included_in_new_limit"] and ledger.read_bytes() == before
