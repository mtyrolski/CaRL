"""Checks for matched controller inputs and complete generation denominators."""

import json

import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.policy_relative_lora import assess_output
from scripts.assess_luna_lora import prepare_cases as assess_cases
from scripts.assess_luna_lora import summarise
from scripts.assess_luna_lora import write_results
from scripts.prepare_luna_lora_controller_evaluation import prepare_cases
from scripts.prepare_luna_lora_controller_evaluation import prepare_matrix


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


def test_matrix_preserves_two_slots_per_method_and_distinguishes_duration(tmp_path) -> None:
    case_context = context()
    case_context["layout_id"] = canonical_layout_id(from_rows(case_context["start_observation"]))
    matrix = {"tiers": [0.25, 0.75], "arms": {}}
    for arm in ("B", "D", "N", "T", "U", "R"):
        slots = []
        for index, tier in enumerate(matrix["tiers"]):
            requested = (3.0, 8.0)[index] if arm == "T" else tier if arm == "U" else None
            status = "invalid_json" if arm == "T" and index == 1 else "valid_intent"
            item = prediction(arm, requested, status)
            item["layout_id"] = case_context["layout_id"]
            item["raw_output"] = json.dumps(item["parsed"]) if item["parsed"] is not None else "{"
            item["validation_errors"] = [] if item["parsed"] is not None else ["invalid JSON"]
            if arm == "R":
                item["local_rule_work"] = {"push_checks": 4}
            path = tmp_path / f"{arm}-{index}.jsonl"
            path.write_text(json.dumps(item) + "\n")
            slots.append({"path": str(path), "requested_input": requested})
        matrix["arms"][arm] = slots
    cases, sent = prepare_matrix([case_context], matrix)
    assert len(cases) == 12 and len(sent) == 11
    assert len({row["record_id"] for row in cases}) == 12
    assert all("later_observation" not in row for row in cases + sent)
    assert {row["requested_input"] for row in cases if row["arm"] == "T"} == {3.0, 8.0}
    assert {row["requested_scale"] for row in cases if row["arm"] == "T"} == {0.25, 0.75}
    assert all(row["local_rule_work"] == {"push_checks": 4} for row in cases if row["arm"] == "R")
    assessed = assess_cases(cases, [], controller_hash="a" * 64, action_cap=16, expected_attempts=1)
    assert len(assessed) == 12
    assert all(row["local_rule_work"] == {"push_checks": 4} for row in assessed if row["arm"] == "R")
    assert sum(row["case_status"] == "invalid_prediction" for row in assessed) == 1
    summary, curves, paired = summarise(assessed, action_cap=16, bootstrap_samples=5)
    output = tmp_path / "assessment"
    write_results(output, assessed, summary, curves, paired, {"test": True})
    svg = (output / "intent_achievement.svg").read_text()
    assert "B, tier=0.25 (0/1; unavailable)" in svg
    assert "T, d=3 (tier 0.25) (0/1; unavailable)" in svg
    assert "U, u=0.75 (tier 0.75) (0/1; unavailable)" in svg


def test_matrix_assessor_accepts_and_records_fence_normalisation(tmp_path) -> None:
    case_context = context()
    case_context["layout_id"] = canonical_layout_id(from_rows(case_context["start_observation"]))
    intent = {"effects": [{"predicate": "box_at", "row": 3, "col": 2}], "preserve": [], "instruction": None}
    raw = "```json\n" + json.dumps(intent) + "\n```"
    matrix = {"tiers": [.25, .75], "arms": {}}
    for arm in ("D", "N"):
        slots = []
        for index in range(2):
            row = {key: case_context[key] for key in ("record_id", "context_id", "layout_id", "family")}
            row.update(arm=arm, requested_scale=None, raw_output=raw,
                       **assess_output(raw, case_context["start_observation"], normalize=True))
            path = tmp_path / f"{arm}-{index}.jsonl"
            path.write_text(json.dumps(row) + "\n")
            slots.append({"path": str(path), "requested_input": None})
        matrix["arms"][arm] = slots
    cases, sent = prepare_matrix([case_context], matrix)
    assert len(cases) == len(sent) == 4
    assert all(row["normalization_applied"] is True and row["strict_json_valid"] is False for row in cases)
    assessed = assess_cases(cases, [], controller_hash="a" * 64, action_cap=16, expected_attempts=1)
    assert len(assessed) == 4 and all(row["case_status"] == "not_attempted" for row in assessed)


def matrix_for_validation(tmp_path, inputs):
    """Build separate, checked files without invoking a generator."""
    case_context = context()
    matrix = {"tiers": [.25, .75], "arms": {}}
    for arm, values in inputs.items():
        slots = []
        for index, requested in enumerate(values):
            item = prediction(arm, requested, "invalid_json")
            path = tmp_path / f"{arm}-{index}.jsonl"
            path.write_text(json.dumps(item) + "\n")
            slots.append({"path": str(path), "requested_input": requested})
        matrix["arms"][arm] = slots
    return case_context, matrix


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_matrix_rejects_reusing_a_physical_file_for_two_slots(tmp_path, alias) -> None:
    case_context, matrix = matrix_for_validation(tmp_path, {"B": [None, None], "N": [None, None]})
    source = tmp_path / "N-0.jsonl"
    alternative = tmp_path / "alias.jsonl"
    if alias == "symlink":
        alternative.symlink_to(source)
    elif alias == "hardlink":
        alternative.hardlink_to(source)
    else:
        alternative = source
    matrix["arms"]["N"][1]["path"] = str(alternative)
    with pytest.raises(ValueError, match="distinct prediction file"):
        prepare_matrix([case_context], matrix)


@pytest.mark.parametrize("arm,values", [
    ("D", [None, .5]),
    ("N", [False, None]),
    ("T", [None, 2]),
    ("T", [0, 2]),
    ("T", [1, float("inf")]),
    ("T", [1, 1]),
    ("T", [2, 1]),
    ("U", [None, .75]),
    ("U", [True, .75]),
    ("U", [float("nan"), .75]),
    ("U", [.25, 1.1]),
    ("U", [.5, .5]),
    ("U", [.75, .25]),
])
def test_matrix_rejects_missing_invalid_or_unordered_inputs(tmp_path, arm, values) -> None:
    case_context, matrix = matrix_for_validation(tmp_path, {"B": [None, None], arm: values})
    with pytest.raises(ValueError):
        prepare_matrix([case_context], matrix)


def test_matrix_rejects_undeclared_input_even_for_unscaled_arm(tmp_path) -> None:
    case_context, matrix = matrix_for_validation(tmp_path, {"B": [None, None], "N": [None, None]})
    del matrix["arms"]["N"][0]["requested_input"]
    with pytest.raises(ValueError, match="declared input"):
        prepare_matrix([case_context], matrix)
