import json

import pytest

from carl.icl.intent_assessment import AssessmentStatus
from carl.icl.intent_assessment import Method
from carl.icl.intent_assessment import SearchStatus
from carl.icl.intent_assessment import assess_context
from carl.icl.intent_assessment import bounded_witness
from carl.icl.intent_assessment import compare_metadata
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_assessment import local_rule
from carl.icl.intent_assessment import predictions_for
from carl.icl.intent_assessment import step
from carl.icl.intent_assessment import summarise
from carl.icl.intent_data import validate_prediction

BOARD = [
    "######",
    "#@-$.#",
    "#----#",
    "#----#",
    "#----#",
    "######",
]


def intent(row, col):
    return {
        "effects": [{"predicate": "box_at", "row": row, "col": col}],
        "preserve": [],
        "instruction": None,
    }


def test_bounded_search_witness_replays_in_real_environment():
    result = bounded_witness(BOARD, intent(1, 4), max_depth=4, max_nodes=100, seconds=5)
    assert result["witness_found"]
    assert result["witness_actions"] == [3, 3]
    assert result["reached_observation"][1] == "#--@*#"


def test_cutoff_is_unresolved_not_impossible():
    result = bounded_witness(BOARD, intent(1, 4), max_depth=1, max_nodes=100, seconds=5)
    assert not result["witness_found"]
    assert result["status"] == SearchStatus.DEPTH_LIMIT
    assert result["witness_actions"] is None


def test_current_board_rule_emits_valid_intent_without_future_state():
    proposal = local_rule(BOARD)
    assert not validate_prediction(proposal, BOARD)
    assert proposal == intent(1, 4)


def test_action_coordinates_and_blocked_moves():
    walls = (
        {(0, c) for c in range(6)}
        | {(5, c) for c in range(6)}
        | {(r, 0) for r in range(6)}
        | {(r, 5) for r in range(6)}
    )
    assert step((1, 1), frozenset({(1, 3)}), 0, 6, 6, walls) is None
    assert step((1, 1), frozenset({(1, 3)}), 3, 6, 6, walls) == (
        (1, 2),
        frozenset({(1, 3)}),
    )


def test_preservation_is_checked_along_the_search():
    request = intent(1, 4)
    request["preserve"] = [{"predicate": "player_at", "row": 1, "col": 1}]
    result = bounded_witness(BOARD, request, max_depth=8, max_nodes=100, seconds=5)
    assert not result["witness_found"]
    assert result["status"] == SearchStatus.EXHAUSTED


@pytest.mark.parametrize("effects", [None, 0, "box", {"predicate": "box_at"}])
def test_malformed_but_valid_json_is_retained_as_invalid_intent(effects):
    record = {
        "record_id": "r1",
        "layout_id": "board-1",
        "start_observation": BOARD,
        "later_observation": BOARD,
    }
    parsed = {"effects": effects, "preserve": []}
    prediction = {
        "parsed": parsed,
        "raw_output": json.dumps(parsed),
        "json_valid": True,
    }
    row = assess_context(
        record, Method.BASE, {Method.BASE: {"r1": prediction}}, 2, 10, 1
    )
    assert row["json_valid"] is True
    assert row["valid_nontrivial"] is False
    assert row["solver"] is None


def test_incomplete_data_uses_board_weighted_uncertainty():
    def row(layout, valid):
        return {
            "layout_id": layout,
            "record_id": f"{layout}-{valid}",
            "method": Method.BASE,
            "status": AssessmentStatus.EVALUATED,
            "json_valid": valid,
            "valid_nontrivial": valid,
            "effect_types": ["box_at"] if valid else [],
            "conjunction_size": int(valid),
            "solver": {
                "witness_found": True,
                "status": "witness_found",
                "expanded_nodes": 1,
                "elapsed_seconds": 0.1,
            }
            if valid
            else None,
        }

    summary, _ = summarise([row("many", False) for _ in range(10)] + [row("one", True)])
    base = summary[0]
    assert base["valid_fraction"] == pytest.approx(1 / 11)
    assert base["board_mean_valid_fraction"] == pytest.approx(0.5)
    assert base["board_mean_witness_fraction"] == pytest.approx(0.5)
    assert base["valid_fraction_95_board_bootstrap"] is None


def test_rejects_duplicate_and_inconsistent_predictions(tmp_path):
    data, ids, path = (
        tmp_path / name for name in ("dataset.jsonl", "frozen.json", "base.jsonl")
    )
    data.write_text("data")
    ids.write_text('{"record_ids":["r1"]}')
    contexts = [{"record_id": "r1", "layout_id": "board-1", "split": "test"}]
    row = {
        "record_id": "r1",
        "layout_id": "board-1",
        "split": "test",
        "method": "base",
        "raw_output": json.dumps(intent(1, 4)),
        "json_valid": True,
        "parsed": intent(1, 4),
    }
    path.write_text(json.dumps(row) + "\n")
    path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "n": 1,
                "data_sha256": file_hash(data),
                "test_ids_sha256": file_hash(ids),
                "adapter": None,
            }
        )
    )
    assert len(predictions_for(path, Method.BASE, contexts, data, ids)[0]) == 1
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="does not match frozen"):
        predictions_for(path, Method.BASE, contexts, data, ids)
    row["json_valid"] = False
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="disagrees"):
        predictions_for(path, Method.BASE, contexts, data, ids)


def test_rejects_mismatched_model_settings():
    common = {
        "model": "model",
        "revision": "rev",
        "seed": 1,
        "decoding": "greedy",
        "enable_thinking": False,
        "max_new_tokens": 128,
    }
    compare_metadata({Method.BASE: common, Method.ADAPTED: dict(common)})
    with pytest.raises(ValueError, match="revision"):
        compare_metadata(
            {Method.BASE: common, Method.ADAPTED: dict(common, revision="different")}
        )
    with pytest.raises(ValueError, match="decoding"):
        compare_metadata(
            {Method.BASE: common, Method.IN_CONTEXT: dict(common, decoding="sampling")}
        )
    with pytest.raises(ValueError, match="dtype"):
        compare_metadata(
            {
                Method.BASE: common,
                Method.ADAPTED: dict(common, dtype="bfloat16"),
                Method.IN_CONTEXT: dict(common, dtype="float16"),
            }
        )
