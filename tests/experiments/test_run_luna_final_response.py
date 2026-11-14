"""Check the final-message runner with local SDK objects and fake transport."""

import json

import pytest
from openai.types.responses import Response

from carl.icl.luna_intent_rollout import RolloutStatus
from carl.icl.luna_intent_transport import create_luna_budget_config
from scripts import run_luna_final_response as runner


BOARD = ["#####", "#####", "#@$.#", "#####", "#####"]
INTENT = {"effects": [{"predicate": "box_at", "row": 2, "col": 3}], "preserve": []}


def message(phase, text):
    return {"type": "message", "id": "msg_test", "role": "assistant", "status": "completed", "phase": phase,
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def response(*messages, status="completed"):
    return Response.model_validate({
        "id": "resp_test", "created_at": 1.0, "model": "gpt-6-luna", "object": "response",
        "output": list(messages), "parallel_tool_calls": False, "tool_choice": "none", "tools": [], "status": status,
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


def records(variant, real_count=4, synthetic_count=4):
    rows = []
    for phase, family, count in (("calibration", "real_12x12x4", 12),
                                 ("test", "real_12x12x4", real_count),
                                 ("test", "proposal_gamma2_12x12x10", synthetic_count)):
        for index in range(count):
            record_id = f"cal-{index}" if phase == "calibration" else f"{variant}-{family}-{index}"
            rows.append({"record_id": record_id, "phase": phase, "family": family, "layout_id": record_id,
                         "context_id": record_id, "start_observation": BOARD, "intent": INTENT,
                         "task_goal": "Put boxes on goals.", "later_observation": "PRIVATE-FUTURE"})
    return rows


def write_records(run_dir, rows):
    path = run_dir / "data/records.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run_dir / "data/manifest.json").write_text("{}\n")


def write_pilot(run_dir):
    runner.append_jsonl(run_dir / "results/pilot.jsonl", {"result": {"status": "achieved"}})


def test_adapter_extracts_final_and_prefixes_tag(tmp_path):
    result = response(message("commentary", '{"action":"LEFT"}'), message("final_answer", '{"action":"UP"}'))
    underlying = FakeTransport([result])
    adapter = runner.FinalMessageTransport(underlying, tmp_path)
    assert adapter.create("board", tag="case").output_text == '{"action":"UP"}'
    assert underlying.calls == [{"prompt": "board", "tag": "final:case"}]
    assert not (tmp_path / "results/response_errors.jsonl").exists()


def test_incomplete_is_recorded_as_an_output_error_without_retry(tmp_path):
    underlying = FakeTransport([response(message("final_answer", "PRIVATE-PARTIAL"), status="incomplete")])
    adapter = runner.FinalMessageTransport(underlying, tmp_path)
    assert adapter.create("board", tag="case").output_text == ""
    errors = runner.read_jsonl(tmp_path / "results/response_errors.jsonl")
    assert len(errors) == 1 and errors[0]["record_id"] == "case"
    assert "PRIVATE-PARTIAL" not in json.dumps(errors)
    assert len(underlying.calls) == 1


def test_transport_exception_is_not_misclassified_as_a_model_output_error(tmp_path):
    underlying = FakeTransport([TimeoutError("fake")])
    adapter = runner.FinalMessageTransport(underlying, tmp_path)
    with pytest.raises(TimeoutError):
        adapter.create("board", tag="case")
    assert len(underlying.calls) == 1
    assert not (tmp_path / "results/response_errors.jsonl").exists()


def test_experiment_marks_incomplete_as_invalid_and_stops_on_following_transport_error(tmp_path):
    write_records(tmp_path, records("final"))
    write_pilot(tmp_path)
    underlying = FakeTransport([response(message("final_answer", "partial"), status="incomplete"), TimeoutError("fake")])
    adapter = runner.FinalMessageTransport(underlying, tmp_path)
    assert runner.experiment(tmp_path, adapter) == 1
    outcomes = runner.read_jsonl(tmp_path / "results/rollouts.jsonl")
    assert [row["result"]["status"] for row in outcomes] == [RolloutStatus.INVALID_RESPONSE, RolloutStatus.API_ERROR]
    assert all(not row["result"]["actions"] for row in outcomes)
    assert len(underlying.calls) == 2
    assert all(call["tag"].startswith("final:cal-") for call in underlying.calls)
    assert all("PRIVATE-FUTURE" not in call["prompt"] for call in underlying.calls)
    with pytest.raises(ValueError, match="earlier API failure"):
        runner.experiment(tmp_path, adapter)
    assert len(underlying.calls) == 2


def test_orphaned_event_blocks_retry(tmp_path):
    selection = records("final")
    write_records(tmp_path, selection)
    write_pilot(tmp_path)
    runner.append_jsonl(tmp_path / "results/events.jsonl", {"record_id": selection[0]["record_id"], "type": "request"})
    underlying = FakeTransport([])
    with pytest.raises(ValueError, match="interrupted attempt"):
        runner.experiment(tmp_path, runner.FinalMessageTransport(underlying, tmp_path))
    assert not underlying.calls


def test_selection_is_unique_and_interleaves_test_families():
    selection = records("final")
    ordered = runner.order(selection)
    assert [row["family"] for row in ordered[12:16]] == [
        "real_12x12x4", "proposal_gamma2_12x12x10", "real_12x12x4", "proposal_gamma2_12x12x10",
    ]
    selection[-1]["record_id"] = selection[0]["record_id"]
    with pytest.raises(ValueError, match="duplicate"):
        runner.order(selection)


def test_plan_rejects_test_layout_seen_in_preceding_variants(tmp_path):
    campaign = tmp_path / "campaign"
    old1, old2, current = records("old1", 8, 8), records("old2", 8, 8), records("final")
    current[12]["layout_id"] = old1[12]["layout_id"]
    for directory, rows in ((campaign, old1), (campaign / "structured", old2), (campaign / "final_response", current)):
        write_records(directory, rows)
    for path in (campaign / "experiment_plan.json", campaign / "structured/experiment_plan.json"):
        path.write_text("{}\n")
    create_luna_budget_config(campaign)
    with pytest.raises(ValueError, match="overlap"):
        runner.create_plan(campaign, campaign / "final_response")
