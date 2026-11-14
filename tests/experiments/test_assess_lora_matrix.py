"""Checks for held-out comparison and interrupted generator outputs."""

import hashlib
import json

import pytest

from scripts.assess_lora_matrix import assess
from scripts.assess_lora_matrix import equivalent_test_ids
from scripts.assess_lora_matrix import intent_signature
from scripts.assess_lora_matrix import paired
from scripts.assess_lora_matrix import read_partial
from scripts.assess_lora_matrix import summarize
from scripts.assess_lora_matrix import variants

BOARD = ["#####", "#####", "#@$.#", "#####", "#####"]
INTENT = {"effects": [{"predicate": "boxes_on_goals_at_least", "count": 1}], "preserve": [], "instruction": None}


def record(identifier="one", layout="board"):
    return {"record_id": identifier, "layout_id": layout, "start_observation": BOARD}


def test_strict_and_fence_are_reported_separately():
    raw = "```json\n" + json.dumps(INTENT) + "\n```"
    row = assess(record(), {"raw_output": raw}, bfs=False, max_depth=8, max_nodes=100, seconds=1)
    assert not row["strict_json"] and not row["strict_valid"]
    assert row["fence_removed"] and row["fence_json"] and row["fence_valid"]
    assert row["fence_whole_goal"] and not row["strict_whole_goal"]


def test_bounded_search_is_an_independently_replayed_solver():
    row = assess(record(), {"raw_output": json.dumps(INTENT)}, bfs=True, max_depth=4, max_nodes=100, seconds=1)
    assert row["strict_valid"] and row["strict_whole_goal"]
    assert row["solver"]["witness_found"] and row["solver"]["witness_actions"] == [3]
    invalid = assess(record("two", "board2"), {"raw_output": "not JSON"}, bfs=True,
                     max_depth=4, max_nodes=100, seconds=1)
    summary = summarize("model", [row, invalid], {"bfs": True})
    assert summary["bfs_attempted"] == summary["bfs_found"] == 1
    assert summary["pipeline_evaluated_for_bfs"] == 2 and summary["bfs_unresolved"] == 0


def test_missing_output_is_not_counted_as_a_failed_proposal():
    good = assess(record(), {"raw_output": json.dumps(INTENT)}, bfs=False, max_depth=4, max_nodes=100, seconds=1)
    missing = assess(record("two", "board2"), None, bfs=False, max_depth=4, max_nodes=100, seconds=1)
    summary = summarize("adapter", [good, missing], {"bfs": False})
    assert (summary["planned"], summary["outputs"], summary["missing"]) == (2, 1, 1)
    assert summary["strict_valid"] == 1 and summary["strict_valid_board_mean"] == 1
    assert summary["strict_valid_95_board_bootstrap"] is None and summary["bfs_found"] is None
    assert summary["generated_tokens"] is None and summary["generation_seconds"] is None


def test_partial_line_is_preserved_and_not_counted(tmp_path):
    path = tmp_path / "predictions.jsonl"
    original = b'{"record_id":"a"}\n{"record_id":'
    path.write_bytes(original)
    rows, partial = read_partial(path)
    assert rows == [{"record_id": "a"}] and partial and path.read_bytes() == original


def test_pairing_uses_only_shared_contexts_and_boards():
    rows = [{"variant": "base", "record_id": "1", "layout_id": "a", "status": "evaluated", "strict_valid": False},
            {"variant": "new", "record_id": "1", "layout_id": "a", "status": "evaluated", "strict_valid": True},
            {"variant": "new", "record_id": "2", "layout_id": "b", "status": "evaluated", "strict_valid": True}]
    delta = paired(rows, "base", "new")
    assert delta["contexts"] == delta["boards"] == 1 and delta["strict_valid_delta"] == 1


def test_distinct_intent_ignores_prose_and_conjunction_order():
    a = {"effects": [{"predicate": "box_at", "row": 1, "col": 2}, {"predicate": "player_at", "row": 1, "col": 3}],
         "preserve": [], "instruction": "one"}
    b = {"effects": list(reversed(a["effects"])), "preserve": [], "instruction": "two"}
    assert intent_signature(a) == intent_signature(b)


def test_duplicate_variant_names_are_rejected():
    with pytest.raises(ValueError):
        variants(["same=a", "same=b"])


def test_equivalent_ids_require_identical_order_and_dataset(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text("{}\n")
    checksum = hashlib.sha256(data.read_bytes()).hexdigest()
    original, alternate = tmp_path / "old.json", tmp_path / "new.json"
    original.write_text(json.dumps({"record_ids": ["a", "b"], "dataset_sha256": checksum}))
    alternate.write_text(json.dumps({"record_ids": ["a", "b"], "note": "same selection"}))
    hashes = equivalent_test_ids(original, [alternate], data)
    assert len(set(hashes.values())) == 2
    alternate.write_text(json.dumps({"record_ids": ["b", "a"]}))
    with pytest.raises(ValueError, match="selection or order"):
        equivalent_test_ids(original, [alternate], data)
    alternate.write_text(json.dumps({"record_ids": ["a", "b"], "dataset_sha256": "wrong"}))
    with pytest.raises(ValueError, match="dataset hash"):
        equivalent_test_ids(original, [alternate], data)
