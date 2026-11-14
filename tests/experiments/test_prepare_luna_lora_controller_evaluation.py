"""Checks for matched controller inputs and complete generation denominators."""

import json

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from scripts.assess_luna_lora import prepare_cases as assess_cases
from scripts.prepare_luna_lora_controller_evaluation import prepare_cases


def context() -> dict:
    return {"record_id": "source", "context_id": "ctx", "layout_id": "layout", "family": "family",
            "split": "test", "start_observation": ["#######", "#@----#", "#$---.#", "#-----#", "#######"],
            "task_goal": "Put boxes on goals.", "later_observation": ["not permitted"]}


def prediction(arm: str, scale: float | None, status: str) -> dict:
    valid = {"effects": [{"predicate": "box_at", "row": 3, "col": 2}], "preserve": [], "instruction": None}
    return {"record_id": "source", "context_id": "ctx", "layout_id": "layout", "family": "family",
            "arm": arm, "requested_scale": scale, "status": status, "raw_output": "raw",
            "parsed": valid if status == "valid_intent" else None, "json_valid": status == "valid_intent",
            "validation_errors": []}


def test_retains_all_cases_and_sends_only_valid_intents() -> None:
    baseline = [prediction("unscaled", None, "valid_intent")]
    low = [prediction("scaled", 0.25, "invalid_json")]
    high = [prediction("scaled", 0.75, "valid_intent")]
    cases, sent = prepare_cases([context()], baseline, {0.25: low, 0.75: high})
    assert len(cases) == 4
    assert len(sent) == 3
    assert {row["prediction_status"] for row in cases} == {"valid_intent", "invalid_json"}
    assert len({row["record_id"] for row in cases}) == 4
    assert all(row["source_record_id"] == "source" for row in cases)
    assert all("later_observation" not in row for row in cases + sent)
    baseline_cases = [row for row in sent if row["arm"] == "unscaled"]
    assert baseline_cases[0]["intent"] == baseline_cases[1]["intent"]
    assert {row["requested_scale"] for row in baseline_cases} == {0.25, 0.75}


def test_requires_an_explicit_result_for_every_context() -> None:
    with pytest.raises(ValueError, match="Every planned context"):
        prepare_cases([context()], [], {0.25: [prediction("scaled", 0.25, "valid_intent")],
                                        0.75: [prediction("scaled", 0.75, "valid_intent")]})


def test_rechecks_intent_validity_before_sending() -> None:
    invalid = prediction("unscaled", None, "valid_intent")
    invalid["parsed"]["effects"][0]["row"] = 0  # A wall.
    with pytest.raises(ValueError, match="checker"):
        prepare_cases([context()], [invalid], {0.25: [prediction("scaled", 0.25, "valid_intent")],
                                                0.75: [prediction("scaled", 0.75, "valid_intent")]})


def test_prepared_cases_are_accepted_by_the_assessor_without_rollouts() -> None:
    case_context = context()
    case_context["layout_id"] = canonical_layout_id(from_rows(case_context["start_observation"]))
    baseline = prediction("unscaled", None, "valid_intent")
    low = prediction("scaled", 0.25, "invalid_json")
    high = prediction("scaled", 0.75, "valid_intent")
    for item in (baseline, low, high):
        item["layout_id"] = case_context["layout_id"]
        item["raw_output"] = json.dumps(item["parsed"]) if item["parsed"] is not None else "{"
        item["validation_errors"] = [] if item["parsed"] is not None else ["invalid JSON"]
        if item["parsed"] is None:
            item["json_valid"] = False
    plan, _ = prepare_cases([case_context], [baseline], {0.25: [low], 0.75: [high]})
    assessed = assess_cases(plan, [], controller_hash="a" * 64, action_cap=16, expected_attempts=1)
    assert len(assessed) == 4
    assert sum(row["case_status"] == "invalid_prediction" for row in assessed) == 1
    assert sum(row["case_status"] == "not_attempted" for row in assessed) == 3
