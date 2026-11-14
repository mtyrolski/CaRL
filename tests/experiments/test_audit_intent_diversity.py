"""The diversity audit counts matches without treating them as achievements."""

import json

import pytest

from scripts.audit_intent_diversity import Mode
from scripts.audit_intent_diversity import canonical
from scripts.audit_intent_diversity import parse_one
from scripts.audit_intent_diversity import summaries
from scripts.audit_intent_diversity import write_report


def _intent():
    return {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": [], "instruction": None}


def _context(identifier, board):
    return {"record_id": identifier, "layout_id": identifier, "evaluation_family": "12x12x4",
            "start_observation": board}


def _prediction(raw, parsed):
    return {"raw_output": raw, "representation": "json", "syntax_valid": parsed is not None,
            "parsed_intent": parsed}


def test_canonicalisation_ignores_order_but_preserves_predicates():
    box = {"predicate": "box_at", "row": 2, "col": 3}
    goals = {"predicate": "boxes_on_goals_at_least", "count": 2}
    player = {"predicate": "player_at", "row": 1, "col": 1}
    first = {"effects": [box, goals], "preserve": [player]}
    reordered = {"effects": [goals, box], "preserve": [player]}
    assert canonical(first) == canonical(reordered)
    assert canonical(first) != canonical({"effects": [box, goals], "preserve": []})
    assert canonical(first) != canonical({"effects": [box, goals, goals], "preserve": [player]})


def test_one_fence_analysis_is_separate_and_rejects_nested_or_prose():
    intent = _intent()
    raw = json.dumps(intent)
    board = ["#######", "#@$-.-#", "#-----#", "#######"]
    context = _context("a", board)
    signature = [canonical(intent)[0]]
    fenced = _prediction(f"```json\n{raw}\n```", None)
    strict = parse_one(fenced, context, Mode.STRICT, signature)
    posthoc = parse_one(fenced, context, Mode.POSTHOC_FENCE, signature)
    assert strict["parseable"] is False and strict["semantic_valid"] is None
    assert posthoc["fence_removed"] is True and posthoc["semantic_valid"] is True
    assert posthoc["matches_balanced0_support"] == [0]
    for invalid in (f"Here is the intent:\n```json\n{raw}\n```", f"```\n```json\n{raw}\n```\n```"):
        record = parse_one(_prediction(invalid, None), context, Mode.POSTHOC_FENCE, signature)
        assert record["fence_removed"] is False and record["parseable"] is False


def test_same_support_match_is_checked_on_each_board():
    intent = _intent()
    signature = [canonical(intent)[0]]
    prediction = _prediction(json.dumps(intent), intent)
    free = _context("a", ["#######", "#@$-.-#", "#-----#", "#######"])
    already_true = _context("b", ["#######", "#@-$.-#", "#-----#", "#######"])
    rows = [parse_one(prediction, context, Mode.STRICT, signature) for context in (free, already_true)]
    summary, frequencies, matches = summaries("model", "balanced0", rows, 1)
    assert summary["outputs"] == 2 and summary["independent_boards"] == 2
    assert summary["match_support_0"] == 2 and summary["match_support_any_valid"] == 1
    assert frequencies[0]["occurrences"] == 2 and frequencies[0]["valid_on_board"] == 1
    assert frequencies[0]["invalid_on_board"] == 1
    assert matches[0]["matches"] == 2 and matches[0]["valid_on_board"] == 1


def test_recorded_strict_parse_must_match_raw():
    intent = _intent()
    context = _context("a", ["#######", "#@$-.-#", "#-----#", "#######"])
    with pytest.raises(ValueError, match="Saved strict parse"):
        parse_one(_prediction(json.dumps(intent), None), context, Mode.STRICT, [])


def test_report_is_immutable_and_keeps_per_board_records(tmp_path):
    destination = tmp_path / "snapshot"
    result = {"schema_version": 1, "summaries": [{"variant": "m", "outputs": 1}],
              "frequencies": [{"variant": "m", "canonical_intent": _intent()}], "support_matches": [],
              "records": [{"record_id": "a", "semantic_valid": None}]}
    write_report(destination, result)
    assert json.loads((destination / "summary.json").read_text())["schema_version"] == 1
    assert json.loads((destination / "records.jsonl").read_text())["semantic_valid"] is None
    assert "record_id" not in (destination / "summary.json").read_text()
    with pytest.raises(FileExistsError):
        write_report(destination, result)
