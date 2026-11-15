"""The local-rule comparison uses two measured current-board slots."""

import json

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.intent_assessment import local_rule
from carl.icl.intent_assessment import local_rule_with_stats
from carl.icl.intent_data import validate_prediction
from scripts import generate_luna_policy_local_rule as runner
from scripts.prepare_luna_lora_controller_evaluation import prepare_matrix

BOARD = ["######", "#@-$.#", "#----#", "#----#", "#----#", "######"]


def make_context(board=BOARD):
    return {"record_id": "case", "context_id": "context", "layout_id": canonical_layout_id(from_rows(board)),
            "family": "synthetic", "split": "test", "start_observation": board, "task_goal": "Put boxes on goals"}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_rule_statistics_keep_original_proposal_and_count_graph_work():
    proposal, work = local_rule_with_stats(BOARD)
    assert proposal == local_rule(BOARD)
    assert not validate_prediction(proposal, BOARD)
    assert work["walk_graphs"] == 1
    assert work["walk_nodes_expanded"] == work["reachable_tiles"] > 0
    assert work["walk_edges_examined"] == 4 * work["walk_nodes_expanded"]
    assert work["push_checks"] == 4
    assert work["elapsed_seconds"] >= 0


def test_both_slots_are_independently_measured_and_compatible_with_matrix(tmp_path, monkeypatch):
    contexts = tmp_path / "contexts.jsonl"
    contexts.write_text(json.dumps(make_context()) + "\n")
    calls = []
    original = runner.local_rule_with_stats

    def count_calls(board):
        calls.append(board)
        return original(board)

    monkeypatch.setattr(runner, "local_rule_with_stats", count_calls)
    output = tmp_path / "local_rule"
    summary = runner.generate(contexts, output)
    assert len(calls) == 2 and summary["predictions"] == 2 and summary["model_calls"] == 0
    slots = [read_jsonl(output / f"R_slot{i}.jsonl") for i in range(2)]
    assert [slot[0]["slot"] for slot in slots] == [0, 1]
    assert all(slot[0]["status"] == "valid_intent" for slot in slots)
    assert all(slot[0]["arm"] == "R" and slot[0]["requested_scale"] is None for slot in slots)
    assert all(slot[0]["local_rule_work"]["walk_graphs"] == 1 for slot in slots)
    assert slots[0][0]["raw_output"] == slots[1][0]["raw_output"]

    # One synthetic matching arm confirms that the generated files satisfy
    # the same slot and provenance contract as learned generators.
    comparison = []
    for i, slot in enumerate(slots):
        path = tmp_path / f"D_slot{i}.jsonl"
        path.write_text(json.dumps({**slot[0], "arm": "D"}) + "\n")
        comparison.append({"path": str(path), "requested_input": None})
    matrix = {"tiers": [.25, .75], "arms": {
        "R": [{"path": str(output / f"R_slot{i}.jsonl"), "requested_input": None} for i in range(2)],
        "D": comparison,
    }}
    cases, controller = prepare_matrix([make_context()], matrix)
    assert len(cases) == len(controller) == 4


def test_future_fields_cannot_change_rule_proposals(tmp_path):
    record = make_context()
    results = []
    for index, future in enumerate((["first future"], ["entirely different future"])):
        path = tmp_path / f"contexts{index}.jsonl"
        path.write_text(json.dumps({**record, "later_observation": future, "fragment_actions": [index]}) + "\n")
        output = tmp_path / f"out{index}"
        runner.generate(path, output)
        results.append([read_jsonl(output / f"R_slot{slot}.jsonl")[0]["raw_output"] for slot in range(2)])
    assert results[0] == results[1]


def test_immobile_board_is_retained_as_invalid_intent(tmp_path):
    board = ["#####", "#@###", "#####", "#####", "#####"]
    proposal, work = local_rule_with_stats(board)
    assert proposal is None and work["walk_nodes_expanded"] == 1 and work["walk_edges_examined"] == 4
    path = tmp_path / "contexts.jsonl"
    path.write_text(json.dumps(make_context(board)) + "\n")
    output = tmp_path / "out"
    runner.generate(path, output)
    row = read_jsonl(output / "R_slot0.jsonl")[0]
    assert row["raw_output"] == "null" and row["status"] == "invalid_intent" and row["json_valid"] is True
