"""Recovery freezes the original prefix and never repeats a failed attempt."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_transport import LunaIntentTransport
from carl.icl.luna_intent_transport import LunaRequestUncertain
from carl.icl.luna_intent_transport import create_luna_budget_config
from scripts import resume_luna_expanded as runner


class BadRequest(Exception):
    status_code = 400
    type = "invalid_request_error"


class FakeClient:
    def __init__(self, callback):
        self.responses = SimpleNamespace(create=callback)

    def with_options(self, **_options):
        return self


def records():
    rows = []
    for phase, family, count in (("calibration", "real_12x12x4", 12),
                                 ("test", "real_12x12x4", 20),
                                 ("test", "proposal_gamma2_12x12x10", 20)):
        for index in range(count):
            record_id = f"{phase}-{family}-{index}"
            rows.append({"record_id": record_id, "phase": phase, "family": family, "layout_id": record_id,
                         "context_id": record_id, "start_observation": ["#####", "#####", "#@$.#", "#####", "#####"],
                         "intent": {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": []},
                         "task_goal": "Put boxes on goals."})
    return rows


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import carl.icl.campaign_budget as campaign_budget
    monkeypatch.setattr(campaign_budget, "dotenv_values", lambda _path: {})
    monkeypatch.delenv("CARL_CAMPAIGN_MAX_USD", raising=False)
    root = tmp_path / "project"
    monkeypatch.setattr(runner, "ROOT", root)
    dependencies = {
        "controller": "carl/icl/luna_intent_rollout.py",
        "action_format": "carl/icl/luna_action_format.py",
        "final_response": "carl/icl/luna_response.py",
        "shared_runner": "scripts/run_luna_intent_experiment.py",
        "variant_runner": "scripts/run_luna_expanded.py",
        "boards": "carl/icl/boards.py",
        "predicates": "carl/icl/intent_data.py",
        "formatting": "carl/icl/luna_components/formatting.py",
        "sokoban_data_source": "carl/icl/data_sources/sokoban.py",
        "sokoban_environment": "carl/environment/sokoban/env.py",
        "calibration": "carl/icl/calibration.py",
    }
    hashes = {}
    for name, relative in dependencies.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen-{name}\n")
        hashes[name] = runner.digest(path)
    (root / "carl/icl/luna_intent_transport.py").write_text("current recovery transport\n")
    campaign = root / "analysis/campaign"
    run = campaign / "expanded"
    data_path = run / "data/records.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    selection = runner.order(records())
    data_path.write_text("".join(json.dumps(row) + "\n" for row in selection))
    config = create_luna_budget_config(campaign)

    def fail(**_payload):
        raise BadRequest("private diagnostic")

    client = FakeClient(fail)
    transport = LunaIntentTransport(config, max_output_tokens=256, structured_actions=True, client=client)
    with pytest.raises(LunaRequestUncertain):
        transport.create("board", tag=f"expanded:{selection[2]['record_id']}")
    unknown_id = next((campaign / "luna_requests").glob("*.json")).stem
    original_snapshot = campaign / "code_v4/luna_intent_transport.py"
    original_snapshot.parent.mkdir(parents=True, exist_ok=True)
    original_snapshot.write_text("the frozen original transport\n")
    hashes["transport"] = runner.digest(original_snapshot)
    old_plan = {"code_sha256": hashes, "budget_config_sha256": runner.digest(config),
                "records_sha256": runner.digest(data_path), "record_ids": [row["record_id"] for row in selection]}
    (run / "experiment_plan.json").write_text(json.dumps(old_plan))
    for index, row in enumerate(selection[:3]):
        status = RolloutStatus.API_ERROR if index == 2 else RolloutStatus.ACHIEVED
        runner.append_jsonl(run / "results/rollouts.jsonl", {
            "record_id": row["record_id"], "result": {"status": status, "model_calls": 1, "actions": [3] if index < 2 else [],
                                                       "achieved": index < 2},
        })
    return campaign, unknown_id, selection


def test_recovery_plan_checks_original_transport_snapshot(prepared):
    campaign, identifier, _selection = prepared
    (campaign / "code_v4/luna_intent_transport.py").write_text("changed snapshot\n")
    with pytest.raises(ValueError, match="snapshot"):
        runner.create_recovery_plan(campaign, identifier)


def test_recovery_plan_keeps_initial_result_prefix_immutable(prepared):
    campaign, identifier, selection = prepared
    plan = runner.create_recovery_plan(campaign, identifier)
    assert plan["unknown_reservation_id"] == identifier and plan["unknown_reserved_nanos"] == 577711200
    assert plan["initial_result_count"] == 3 and plan["failed_record_id"] == selection[2]["record_id"]
    path = campaign / "expanded/results/rollouts.jsonl"
    runner.append_jsonl(path, {"record_id": selection[3]["record_id"], "result": {"status": "action_limit"}})
    assert runner.create_recovery_plan(campaign, identifier) == plan
    lines = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    rows[0]["result"]["actions"] = []
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="results or code have changed"):
        runner.create_recovery_plan(campaign, identifier)


def test_resume_skips_failed_attempt_and_stops_on_second_api_error(prepared, monkeypatch):
    campaign, identifier, selection = prepared
    plan = runner.create_recovery_plan(campaign, identifier)
    path = campaign / "expanded/results/rollouts.jsonl"
    prefix = path.read_bytes()
    calls = []

    class Policy:
        def create(self, _prompt, *, tag):
            calls.append(tag)
            if len(calls) == 2:
                raise TimeoutError("fake")
            return SimpleNamespace(output_text='{"action":"RIGHT"}')

    monkeypatch.setattr(runner, "make_transport", lambda *_args: Policy())
    assert runner.resume(campaign, plan) == 1
    outcomes = runner.read_jsonl(path)
    assert path.read_bytes().startswith(prefix)
    assert len(outcomes) == 5
    assert [row["result"]["status"] for row in outcomes[-2:]] == [RolloutStatus.ACHIEVED, RolloutStatus.API_ERROR]
    assert calls == [selection[3]["record_id"], selection[4]["record_id"]]
    assert selection[2]["record_id"] not in calls
    with pytest.raises(ValueError, match="new API failure"):
        runner.resume(campaign, plan)
    assert len(calls) == 2


def test_resume_rejects_started_unfinished_case_without_calling_api(prepared, monkeypatch):
    campaign, identifier, selection = prepared
    plan = runner.create_recovery_plan(campaign, identifier)
    runner.append_jsonl(campaign / "expanded/results/events.jsonl", {"record_id": selection[3]["record_id"], "type": "request"})
    monkeypatch.setattr(runner, "make_transport", lambda *_args: (_ for _ in ()).throw(AssertionError("must not call")))
    with pytest.raises(ValueError, match="interrupted attempts"):
        runner.resume(campaign, plan)


def test_unknown_request_hash_is_bound_to_saved_prefix(prepared):
    campaign, identifier, _selection = prepared
    plan = runner.create_recovery_plan(campaign, identifier)
    path = campaign / "luna_requests" / f"{identifier}.json"
    audit = json.loads(path.read_text())
    assert plan["unknown_request_sha256"] == audit["request_sha256"]
    results_path = campaign / "expanded/results/rollouts.jsonl"
    assert plan["initial_results_sha256"] == hashlib.sha256(results_path.read_bytes()).hexdigest()
