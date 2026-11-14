"""Check the expanded Luna runner without credentials or API requests."""

import json
from types import SimpleNamespace

import pytest
from openai.types.responses import Response

from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_transport import create_luna_budget_config
from scripts import run_luna_expanded as runner


BOARD = ["#####", "#####", "#@$.#", "#####", "#####"]
INTENT = {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": []}


def response(status):
    return Response.model_validate({
        "id": "resp_test", "created_at": 1.0, "model": "gpt-6-luna", "object": "response",
        "output": [{"type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                    "phase": "final_answer", "content": [{"type": "output_text", "text": '{"action":"RIGHT"}',
                                                           "annotations": []}]}],
        "parallel_tool_calls": False, "tool_choice": "none", "tools": [], "status": status,
    })


class FakeTransport:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def create(self, prompt, *, tag):
        self.calls.append({"prompt": prompt, "tag": tag})
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


def records(variant, real_count, synthetic_count):
    result = []
    for phase, family, count in (("calibration", "real_12x12x4", 12),
                                 ("test", "real_12x12x4", real_count),
                                 ("test", "proposal_gamma2_12x12x10", synthetic_count)):
        for index in range(count):
            record_id = f"cal-{index}" if phase == "calibration" else f"{variant}-{family}-{index}"
            result.append({"record_id": record_id, "phase": phase, "family": family, "layout_id": record_id,
                           "context_id": record_id, "start_observation": BOARD, "intent": INTENT,
                           "task_goal": "Put boxes on goals.", "later_observation": "PRIVATE-FUTURE"})
    return result


def write_records(run_dir, rows):
    path = run_dir / "data/records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "data/manifest.json").write_text("{}\n")


def write_pilot(run_dir):
    runner.append_jsonl(run_dir / "results/pilot.jsonl", {"result": {"status": RolloutStatus.ACHIEVED}})


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_service_failure_status_propagates_as_infrastructure_error(tmp_path, status):
    underlying = FakeTransport([response(status)])
    adapter = runner.FinalMessageTransport(underlying, tmp_path)
    with pytest.raises(RuntimeError, match="service did not complete"):
        adapter.create("board", tag="case")
    assert underlying.calls == [{"prompt": "board", "tag": "expanded:case"}]
    assert not (tmp_path / "results/response_errors.jsonl").exists()


def test_incomplete_response_does_not_fabricate_an_action(tmp_path):
    underlying = FakeTransport([response("incomplete")])
    adapter = runner.FinalMessageTransport(underlying, tmp_path)
    assert adapter.create("board", tag="case").output_text == ""
    assert runner.read_jsonl(tmp_path / "results/response_errors.jsonl")[0]["record_id"] == "case"
    assert len(underlying.calls) == 1


def test_factory_uses_original_campaign_budget_and_strict_format(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(runner, "dotenv_values", lambda _path: {"OPENAI_API_KEY": "test-key"})
    monkeypatch.setattr(runner, "OpenAI", lambda **kwargs: SimpleNamespace(settings=kwargs))

    def create_transport(config_path, **kwargs):
        seen.update(config_path=config_path, **kwargs)
        return object()

    monkeypatch.setattr(runner, "LunaIntentTransport", create_transport)
    adapter = runner.make_transport(tmp_path, tmp_path / "expanded")
    assert seen["config_path"] == tmp_path / "luna_intent_budget.json"
    assert seen["max_output_tokens"] == 256 and seen["structured_actions"] is True
    assert seen["client"].settings["max_retries"] == 0
    assert adapter.run_dir == tmp_path / "expanded"


def test_failure_preserves_partial_results_and_stops_all_following_attempts(tmp_path):
    selection = records("expanded", 20, 20)
    write_records(tmp_path, selection)
    write_pilot(tmp_path)
    underlying = FakeTransport([response("completed"), response("failed")])
    adapter = runner.FinalMessageTransport(underlying, tmp_path)
    assert runner.experiment(tmp_path, adapter) == 1
    outcomes = runner.read_jsonl(tmp_path / "results/rollouts.jsonl")
    assert [row["result"]["status"] for row in outcomes] == [RolloutStatus.ACHIEVED, RolloutStatus.API_ERROR]
    assert len(underlying.calls) == 2 and all("PRIVATE-FUTURE" not in call["prompt"] for call in underlying.calls)
    with pytest.raises(ValueError, match="earlier API failure"):
        runner.experiment(tmp_path, adapter)
    assert len(underlying.calls) == 2
    calibration = json.loads((tmp_path / "results/calibration.json").read_text())
    second = next(context for context in calibration["contexts"] if context["context_id"] == "cal-1")
    assert not second["complete"] and second["candidates"][0]["attempts"] == 0


def test_completed_run_rebuilds_a_stale_calibration_without_repeating_calls(tmp_path):
    selection = records("expanded", 20, 20)
    write_records(tmp_path, selection)
    write_pilot(tmp_path)
    for row in selection:
        runner.append_jsonl(tmp_path / "results/rollouts.jsonl", {
            "record_id": row["record_id"], "result": {"status": RolloutStatus.INVALID_RESPONSE,
                                                       "model_calls": 1, "achieved": False, "actions": []},
        })
    (tmp_path / "results/calibration.json").write_text('{}\n')
    underlying = FakeTransport([])
    assert runner.experiment(tmp_path, runner.FinalMessageTransport(underlying, tmp_path)) == 0
    calibration = json.loads((tmp_path / "results/calibration.json").read_text())
    assert len(calibration["contexts"]) == 12
    assert all(context["complete"] for context in calibration["contexts"])
    assert not underlying.calls


def test_orphaned_event_blocks_retries(tmp_path):
    selection = records("expanded", 20, 20)
    write_records(tmp_path, selection)
    write_pilot(tmp_path)
    runner.append_jsonl(tmp_path / "results/events.jsonl", {"record_id": selection[0]["record_id"], "type": "request"})
    underlying = FakeTransport([])
    with pytest.raises(ValueError, match="interrupted attempt"):
        runner.experiment(tmp_path, runner.FinalMessageTransport(underlying, tmp_path))
    assert not underlying.calls


def test_plan_requires_exactly_the_prior_40_test_records(tmp_path):
    campaign = tmp_path / "campaign"
    previous = [records("v1", 8, 8), records("v2", 8, 8), records("v3", 4, 4)]
    expanded = [row for row in previous[0] if row["phase"] == "calibration"]
    expanded += [row for group in previous for row in group if row["phase"] == "test"]
    directories = [campaign, campaign / "structured", campaign / "final_response"]
    for directory, rows in zip(directories, previous, strict=True):
        write_records(directory, rows)
        (directory / "experiment_plan.json").write_text('{}\n')
    run_dir = campaign / "expanded"
    write_records(run_dir, expanded)
    create_luna_budget_config(campaign)
    plan = runner.create_plan(campaign, run_dir)
    assert plan["budget_ceiling_usd"] == "2.00 shared across all variants"
    assert len(plan["record_ids"]) == 52 and plan["max_output_tokens"] == 256
    expanded[-1] = {**expanded[-1], "task_goal": "modified task"}
    write_records(run_dir, expanded)
    with pytest.raises(ValueError, match="exactly match"):
        runner.create_plan(campaign, run_dir)
