"""Checks for the exploratory extension of the frozen Sokoban test contexts."""
import copy
import hashlib
import json

import pytest

from scripts.freeze_additional_intent_contexts import select_contexts
from scripts.freeze_additional_intent_contexts import validate_fragment
from scripts.freeze_additional_intent_contexts import validate_metadata

START = ["#####", "#@$.#", "#####"]
LATER = ["#####", "#-@*#", "#####"]
INTENT = {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": [], "instruction": None}


def row(record_id, board, layout="layout", split="test"):
    return {
        "record_id": record_id,
        "split": split,
        "layout_id": layout,
        "start_observation": board,
        "task_goal": "Put every box on a goal.",
    }


def test_selection_is_order_independent_and_deduplicates_generator_inputs():
    original = [row("original", START), row("training", START, "training-layout", "train")]
    candidates = [row("duplicate-existing", START), row("new-b", LATER), row("new-a", LATER),
                  row("ignored-training", START, "training-layout", "train")]
    assert [item["record_id"] for item in select_contexts(original, candidates)] == ["new-a"]
    assert select_contexts(original, candidates) == select_contexts(original, list(reversed(candidates)))
    with pytest.raises(ValueError, match="not in the original evaluation"):
        select_contexts(original, [row("new-layout", START, "other")])


def test_revalidation_uses_recorded_actions_endpoints_and_full_intent():
    record = {
        "i": 0,
        "k": 1,
        "start_observation": START,
        "later_observation": LATER,
        "fragment_actions": [3],
        "intent": INTENT,
    }
    assert validate_fragment(record, [START, LATER], [3]) == []
    bad_actions = {**record, "fragment_actions": [2]}
    assert "recorded actions differ from replay" in validate_fragment(bad_actions, [START, LATER], [3])
    bad_end = {**record, "later_observation": START}
    assert "recorded observation differs from replay" in validate_fragment(bad_end, [START, LATER], [3])
    bad_effect = copy.deepcopy(record)
    bad_effect["intent"]["effects"][0]["col"] = 2
    assert any("already true" in message for message in validate_fragment(bad_effect, [START, LATER], [3]))


def test_metadata_must_match_frozen_test_source_and_unmeasured_provenance():
    source = {"id": "trajectory", "layout_id": "layout", "path": "example.json", "sha256": "abc"}
    record = {
        "schema_version": "sokoban-intent-v1",
        "split": "test",
        "trajectory_id": source["id"],
        "layout_id": source["layout_id"],
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "annotation_provenance": {"kind": "local_exact_difference"},
        "policy_measurements": None,
        "i": 0,
        "k": 1,
        "intent": INTENT,
    }
    identity = f"{record['source_sha256']}:0:1:{json.dumps(INTENT, sort_keys=True)}"
    record["record_id"] = hashlib.sha256(identity.encode()).hexdigest()[:24]
    validate_metadata(record, source)
    with pytest.raises(ValueError, match="source_sha256"):
        validate_metadata({**record, "source_sha256": "different"}, source)
    with pytest.raises(ValueError, match="unmeasured"):
        validate_metadata({**record, "policy_measurements": {"success": True}}, source)
