"""Offline tests for the future budgeted Luna controller runner."""

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from openai.types.responses import Response

from carl.icl.luna_intent_rollout import OutputLimitError
from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_transport import LunaIntentTransport
from carl.icl.luna_intent_transport import LunaRequestUncertain
from scripts import run_luna_lora_controller as runner

ROWS = ["#######", "#@$-.-#", "#-$-.-#", "#-----#", "#-----#", "#-----#", "#######"]
INTENT = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": []}


@pytest.fixture(autouse=True)
def historical_ledger(tmp_path, monkeypatch):
    """A small local ledger makes the accounting tests independent of user data."""
    root = tmp_path / "historical"
    root.mkdir()
    audits = root / "luna_requests"
    audits.mkdir()
    config = root / "luna_intent_budget.json"
    config.write_text('{"purpose":"test"}\n', encoding="utf-8")
    ledger = root / "luna_intent_ledger.sqlite3"
    with sqlite3.connect(ledger) as database:
        database.execute("CREATE TABLE metadata(key TEXT,value TEXT)")
        database.execute("INSERT INTO metadata VALUES (?,?)", ("config_hash", runner.digest(config)))
        database.execute("CREATE TABLE attempts(id TEXT,status TEXT,reserved_nanos INTEGER, "
                         "charged_nanos INTEGER,request_hash TEXT)")
        for index, reserved in enumerate((3, 5), start=1):
            identifier = f"{index:032x}"
            payload = {"model": "gpt-6-luna", "input": f"synthetic-{index}"}
            request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            database.execute("INSERT INTO attempts VALUES (?,?,?,?,?)", (identifier, "unknown", reserved, None,
                                                                          request_hash))
            audit = {"reservation_id": identifier, "status": "unknown", "request": payload,
                     "request_sha256": request_hash}
            (audits / f"{identifier}.json").write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(runner, "HISTORICAL_LEDGER", ledger)
    return ledger


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


def test_historical_unknowns_are_held_against_ceiling_without_modification(tmp_path, historical_ledger):
    before = historical_ledger.read_bytes()
    plan = make_plan(tmp_path, [record("a")])
    assert plan["campaign_ceiling_usd"] == "2"
    assert plan["budget_usd"] == "1.999999992"
    assert plan["historical_reservations"]["reserved_nanos"] == 8
    assert len(plan["historical_reservations"]["outstanding"]) == 2
    assert historical_ledger.read_bytes() == before


def test_bad_historical_audit_blocks_plan(tmp_path, historical_ledger):
    (historical_ledger.parent / "luna_requests" / f'{1:032x}.json').write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="audit differ"):
        make_plan(tmp_path, [record("a")])


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


def test_output_limit_never_contributes_a_partial_context_label(tmp_path):
    records = [record("a"), record("b")]
    plan = make_plan(tmp_path, records, attempts=2)
    results = [result("a", 1, RolloutStatus.ACHIEVED, 1), result("a", 2, RolloutStatus.OUTPUT_LIMIT, 2),
               result("b", 1, RolloutStatus.ACHIEVED, 3), result("b", 2, RolloutStatus.ACHIEVED, 4)]
    values = [row["policy_measurements"] for row in runner.calibrated_records(records, results, plan)]
    assert [row["status"] for row in values] == ["output_limit", "complete"]
    assert [len(row["attempts"]) for row in values] == [1, 2]
    assert all(not row["complete"] and row["success_fraction"] is None and row["mean_success_actions"] is None
               and row["u"] is None and row["weight"] is None for row in values)


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


def sdk_response(text: str, *, status: str = "completed", reason: str | None = None) -> Response:
    """Build a local Responses object with enough information for the final-message parser."""
    return Response.model_validate({
        "id": "response", "created_at": 1.0, "model": "gpt-6-luna", "object": "response",
        "parallel_tool_calls": False, "tool_choice": "none", "tools": [], "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        "service_tier": "default",
        "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens": 5,
                  "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 105},
        "output": [{"type": "message", "id": "message", "role": "assistant", "status": status,
                    "phase": "final_answer", "content": [{"type": "output_text", "text": text,
                                                             "annotations": []}]}],
    })


@pytest.mark.parametrize(("response", "expected_rollout", "expected_measurement", "exit_code"), [
    (sdk_response("partial", status="incomplete", reason="max_output_tokens"),
     RolloutStatus.OUTPUT_LIMIT, "output_limit", 0),
    (sdk_response("partial", status="incomplete", reason="content_filter"),
     RolloutStatus.API_ERROR, "infrastructure_error", 1),
    (sdk_response("not JSON"), RolloutStatus.INVALID_RESPONSE, "complete", 0),
])
def test_response_cutoffs_are_not_completed_behavioural_attempts(
        tmp_path, response, expected_rollout, expected_measurement, exit_code):
    class Responses:
        def __init__(self):
            self.responses = iter([sdk_response('{"action":"UP"}'), response])

        def create(self, _prompt, *, tag):
            return next(self.responses)

    records = [record("a")]
    plan = make_plan(tmp_path, records)
    output = tmp_path / "output"
    output.mkdir()
    transport = runner.FinalMessageTransport(cast(LunaIntentTransport, Responses()), output, classify_incomplete=True)
    assert runner.execute(records, plan, transport, output) == exit_code
    saved = runner.read_jsonl(output / "results.jsonl")[0]["result"]
    measurement = runner.read_jsonl(output / "calibrated.jsonl")[0]["policy_measurements"]
    assert saved["status"] == expected_rollout
    assert saved["actions"] == [0] and saved["model_calls"] == 2 and saved["steps"][0]["no_op"]
    assert saved["steps"][-1]["action"] is None
    assert measurement["status"] == expected_measurement
    if expected_rollout == RolloutStatus.INVALID_RESPONSE:
        assert measurement["complete"] and measurement["success_fraction"] == 0
    else:
        assert measurement["attempts"] == [] and not measurement["complete"]
        assert all(measurement[key] is None for key in ("success_fraction", "mean_success_actions", "weight", "u"))


def test_output_limit_preserves_response_and_usage_in_ledger_audit(tmp_path, monkeypatch):
    import carl.icl.campaign_budget as campaign_budget
    import carl.icl.luna_intent_transport as luna_transport

    monkeypatch.setattr(campaign_budget, "dotenv_values", lambda _path: {})
    monkeypatch.delenv("CARL_CAMPAIGN_MAX_USD", raising=False)
    monkeypatch.setattr(luna_transport, "_pace_requests", lambda _path: None)
    client = SimpleNamespace()
    client.with_options = lambda **_options: client
    client.responses = SimpleNamespace(create=lambda **_payload: sdk_response(
        "partial", status="incomplete", reason="max_output_tokens"))
    config = runner.create_luna_budget_config(tmp_path / "budget")
    underlying = runner.LunaIntentTransport(config, client=client, structured_actions=True)
    adapter = runner.FinalMessageTransport(underlying, tmp_path / "run", classify_incomplete=True)

    with pytest.raises(OutputLimitError):
        adapter.create("current board", tag="case")
    audits = list((config.parent / "luna_requests").glob("*.json"))
    assert len(audits) == 1
    audit = json.loads(audits[0].read_text())
    assert audit["status"] == "settled" and audit["conservative_charge_usd"] > 0
    assert audit["response"]["status"] == "incomplete"
    assert audit["response"]["incomplete_details"]["reason"] == "max_output_tokens"
    assert audit["response"]["usage"]["input_tokens"] == 100
    assert audit["response"]["usage"]["output_tokens"] == 5


def test_response_model_guard_stops_on_changed_model_and_retains_audit(tmp_path):
    class LocalTransport:
        def __init__(self):
            self.run_dir = tmp_path / "budget"
            self.run_dir.mkdir()
            self.audit_dir = self.run_dir / "luna_requests"
            self.audit_dir.mkdir()
            self.ledger = object()
            self.calls = 0

        def create(self, _prompt, *, tag):
            self.calls += 1
            model = "gpt-6-luna-v1" if self.calls < 3 else "gpt-6-luna-v2"
            audit = {"status": "settled", "response": {"model": model}}
            (self.audit_dir / f"{self.calls}.json").write_text(json.dumps(audit), encoding="utf-8")
            return SimpleNamespace(model=model)

    local = LocalTransport()
    guard = runner.ModelIdentityGuard(cast(LunaIntentTransport, local))
    assert guard.create("first", tag="a").model == "gpt-6-luna-v1"
    assert guard.create("second", tag="b").model == "gpt-6-luna-v1"
    assert json.loads((local.run_dir / "response_model.json").read_text())["model"] == "gpt-6-luna-v1"
    with pytest.raises(LunaRequestUncertain, match="identity changed"):
        guard.create("third", tag="c")
    assert (local.audit_dir / "3.json").exists()
    with pytest.raises(LunaRequestUncertain, match="identity changed"):
        guard.create("no further request", tag="d")
    assert local.calls == 3


def test_response_model_guard_recovers_identity_from_audits(tmp_path):
    root = tmp_path / "budget"
    root.mkdir()
    audits = root / "luna_requests"
    audits.mkdir()
    (audits / "prior.json").write_text(json.dumps({"status": "settled", "response": {"model": "gpt-6-luna-v1"}}))
    calls = []
    transport = SimpleNamespace(run_dir=root, audit_dir=audits, ledger=object(),
                                create=lambda *_args, **_kwargs: calls.append(1) or SimpleNamespace(model="gpt-6-luna-v2"))
    with pytest.raises(LunaRequestUncertain, match="identity changed"):
        runner.ModelIdentityGuard(cast(LunaIntentTransport, transport)).create("prompt", tag="test")
    assert calls == [1]  # The changed identity is only known after the response.


def test_response_model_guard_records_missing_fingerprint_and_stops_on_change(tmp_path):
    class LocalTransport:
        def __init__(self):
            self.run_dir = tmp_path / "budget"
            self.run_dir.mkdir()
            self.audit_dir = self.run_dir / "luna_requests"
            self.audit_dir.mkdir()
            self.ledger = object()
            self.fingerprints = [None, "fp-a", "fp-a", "fp-b"]
            self.calls = 0

        def create(self, _prompt, *, tag):
            fingerprint = self.fingerprints[self.calls]
            self.calls += 1
            response = {"model": "gpt-6-luna"}
            if fingerprint is not None:
                response["system_fingerprint"] = fingerprint
            (self.audit_dir / f"{self.calls}.json").write_text(
                json.dumps({"status": "settled", "response": response}), encoding="utf-8")
            return SimpleNamespace(model="gpt-6-luna", system_fingerprint=fingerprint)

    local = LocalTransport()
    guard = runner.ModelIdentityGuard(cast(LunaIntentTransport, local))
    guard.create("first", tag="a")
    identity = json.loads((local.run_dir / "response_model.json").read_text())
    assert "system_fingerprint" in identity and identity["system_fingerprint"] is None
    assert "does not prove" in identity["note"]
    # Availability of a previously absent fingerprint is not evidence of a change.
    guard.create("second", tag="b")
    guard.create("third", tag="c")
    with pytest.raises(LunaRequestUncertain, match="fingerprint changed"):
        guard.create("changed", tag="d")
    assert (local.audit_dir / "4.json").exists()
    with pytest.raises(LunaRequestUncertain, match="fingerprint changed"):
        guard.create("must not retry", tag="e")
    assert local.calls == 4


def test_response_model_guard_accepts_legacy_identity_without_rewriting_it(tmp_path):
    root = tmp_path / "budget"
    root.mkdir()
    audits = root / "luna_requests"
    audits.mkdir()
    identity_path = root / "response_model.json"
    identity_path.write_text('{"model":"gpt-6-luna","note":"legacy"}\n', encoding="utf-8")
    before = identity_path.read_bytes()
    (audits / "prior.json").write_text(json.dumps({"status": "settled", "response": {"model": "gpt-6-luna"}}))
    calls = []

    def create(_prompt, *, tag):
        calls.append(tag)
        response = {"model": "gpt-6-luna", "system_fingerprint": "fp-a"}
        (audits / "new.json").write_text(json.dumps({"status": "settled", "response": response}))
        return SimpleNamespace(**response)

    transport = SimpleNamespace(run_dir=root, audit_dir=audits, ledger=object(), create=create)
    assert runner.ModelIdentityGuard(cast(LunaIntentTransport, transport)).create(
        "prompt", tag="test").system_fingerprint == "fp-a"
    assert calls == ["test"] and identity_path.read_bytes() == before


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
