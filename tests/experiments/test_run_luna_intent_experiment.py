"""Exercise the Luna runner without making API calls."""

import json
from types import SimpleNamespace

import pytest

from carl.icl.luna_intent_rollout import RolloutStatus
from scripts import run_luna_intent_experiment as runner


BOARD = ["#####", "#####", "#@$.#", "#####", "#####"]
INTENT = {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": []}


class FakeTransport:
    def __init__(self, *, failure=False):
        self.calls = []
        self.failure = failure

    def create(self, prompt, *, tag):
        self.calls.append({"prompt": prompt, "tag": tag})
        if self.failure:
            raise TimeoutError("fake request")
        return SimpleNamespace(output_text='{"action":"RIGHT"}')


def write_records(run_dir):
    records = []
    for phase, family, count in (("calibration", "real_12x12x4", 12),
                                 ("test", "real_12x12x4", 8),
                                 ("test", "proposal_gamma2_12x12x10", 8)):
        for index in range(count):
            record_id = f"{phase}-{family}-{index}"
            records.append({"record_id": record_id, "phase": phase, "family": family, "layout_id": record_id,
                            "context_id": record_id, "start_observation": BOARD, "task_goal": "Put boxes on goals.",
                            "intent": INTENT, "later_observation": "SECRET-FUTURE", "fragment_actions": [1, 2, 3]})
    path = run_dir / "data/records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    (run_dir / "data/manifest.json").write_text("{}\n")
    return records


def write_pilot(run_dir, status=RolloutStatus.ACHIEVED):
    runner.append_jsonl(run_dir / "results/pilot.jsonl", {"result": {"status": status}})


def test_pilot_is_valid_square_board_and_journals_one_call(tmp_path):
    transport = FakeTransport()
    assert runner.pilot(tmp_path, transport) == 0
    assert len(transport.calls) == 1
    record = runner.read_jsonl(tmp_path / "results/pilot.jsonl")[0]
    assert record["result"]["status"] == RolloutStatus.ACHIEVED
    assert record["result"]["actions"] == [3]
    assert [event["type"] for event in runner.read_jsonl(tmp_path / "results/events.jsonl")] == ["request", "response", "action"]
    with pytest.raises(ValueError, match="already run"):
        runner.pilot(tmp_path, transport)


def test_run_one_does_not_pass_demonstration_future_to_policy(tmp_path):
    record = write_records(tmp_path)[0]
    transport = FakeTransport()
    result = runner.run_one(record, transport, tmp_path, cap=1)
    assert result["result"]["status"] == RolloutStatus.ACHIEVED
    prompt = transport.calls[0]["prompt"]
    assert "SECRET-FUTURE" not in prompt
    assert "later_observation" not in prompt and "fragment_actions" not in prompt
    assert json.loads(prompt.split("\n\n")[-1])["attempt_history"] == []


def test_calibration_uses_successful_durations_and_shared_tie_midpoints(tmp_path):
    records, results = [], []

    def add(record_id, context, status, action_count, *, calls=1, achieved=False):
        records.append({"record_id": record_id, "phase": "calibration", "context_id": context})
        results.append({"record_id": record_id, "result": {
            "status": status, "model_calls": calls, "achieved": achieved, "actions": [0] * action_count,
        }})

    for record_id, length in (("fast", 1), ("tie1", 2), ("tie2", 2), ("slow", 4)):
        add(record_id, "complete", RolloutStatus.ACHIEVED, length, achieved=True)
    add("failed", "complete", RolloutStatus.ACTION_LIMIT, 16)
    add("bad_json", "complete", RolloutStatus.INVALID_RESPONSE, 0)
    add("known_success", "incomplete", RolloutStatus.ACHIEVED, 1, achieved=True)
    add("timeout", "incomplete", RolloutStatus.API_ERROR, 1, calls=2)
    add("invalid", "incomplete", RolloutStatus.INVALID_INTENT, 0, calls=0)
    records.append({"record_id": "not_started", "phase": "calibration", "context_id": "incomplete"})

    path = tmp_path / "calibration.json"
    runner.save_calibration(records, results, path)
    contexts = {context["context_id"]: context for context in json.loads(path.read_text())["contexts"]}
    complete = {row["record_id"]: row for row in contexts["complete"]["candidates"]}
    assert contexts["complete"]["complete"]
    assert {key: complete[key]["u"] for key in ("fast", "tie1", "tie2", "slow")} == {
        "fast": .125, "tie1": .5, "tie2": .5, "slow": .875,
    }
    assert complete["failed"]["u"] is None and complete["bad_json"]["attempts"] == 1
    incomplete = {row["record_id"]: row for row in contexts["incomplete"]["candidates"]}
    assert not contexts["incomplete"]["complete"]
    assert all(row["u"] is None for row in incomplete.values())
    for key in ("timeout", "invalid", "not_started"):
        assert incomplete[key]["attempts"] == 0 and incomplete[key]["success_fraction"] is None


def test_pilot_rejects_an_interrupted_attempt_without_repeating_it(tmp_path):
    transport = FakeTransport()
    runner.append_jsonl(tmp_path / "results/events.jsonl", {"record_id": "synthetic_pilot", "type": "request"})
    with pytest.raises(ValueError, match="interrupted pilot"):
        runner.pilot(tmp_path, transport)
    assert not transport.calls


def test_experiment_requires_a_successful_pilot(tmp_path):
    write_records(tmp_path)
    transport = FakeTransport()
    with pytest.raises(ValueError, match="pilot"):
        runner.experiment(tmp_path, transport)
    write_pilot(tmp_path, RolloutStatus.INVALID_RESPONSE)
    with pytest.raises(ValueError, match="pilot"):
        runner.experiment(tmp_path, transport)
    assert not transport.calls


def test_experiment_never_repeats_an_orphaned_attempt(tmp_path):
    records = write_records(tmp_path)
    write_pilot(tmp_path)
    runner.append_jsonl(tmp_path / "results/events.jsonl", {"record_id": records[0]["record_id"], "type": "request"})
    transport = FakeTransport()
    with pytest.raises(ValueError, match="interrupted attempt"):
        runner.experiment(tmp_path, transport)
    assert not transport.calls


def test_api_failure_preserves_partial_result_and_stops_further_calls(tmp_path):
    write_records(tmp_path)
    write_pilot(tmp_path)
    transport = FakeTransport(failure=True)
    assert runner.experiment(tmp_path, transport) == 1
    assert len(transport.calls) == 1
    results = runner.read_jsonl(tmp_path / "results/rollouts.jsonl")
    assert len(results) == 1 and results[0]["result"]["status"] == RolloutStatus.API_ERROR
    events = runner.read_jsonl(tmp_path / "results/events.jsonl")
    assert [event["type"] for event in events] == ["request", "error"]
    calibration = json.loads((tmp_path / "results/calibration.json").read_text())
    assert all(candidate["u"] is None for context in calibration["contexts"] for candidate in context["candidates"])
    with pytest.raises(ValueError, match="earlier API failure"):
        runner.experiment(tmp_path, transport)
    assert len(transport.calls) == 1


def test_plan_freezes_data_and_interleaves_test_families(tmp_path):
    write_records(tmp_path)
    plan = runner.create_plan(tmp_path)
    assert len(plan["record_ids"]) == 28
    assert [item["family"] for item in plan["record_order"][12:16]] == [
        "real_12x12x4", "proposal_gamma2_12x12x10", "real_12x12x4", "proposal_gamma2_12x12x10",
    ]
    assert runner.create_plan(tmp_path) == plan
    with (tmp_path / "data/records.jsonl").open("a") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="Frozen experiment"):
        runner.create_plan(tmp_path)
