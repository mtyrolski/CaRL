"""Check development-only recommendation, board weighting, and freeze guard."""
from __future__ import annotations

import json

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import to_rows
from scripts import overnight_stage03_dev_decision as decision
from scripts import overnight_stage03_generate as generator
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest


def board(index):
    cells = np.ones((12, 12), dtype=int)
    cells[[0, -1], :] = 0
    cells[:, [0, -1]] = 0
    cells[2, 2:5] = 3
    cells[5, 3], cells[5, 4], cells[5, 5] = 5, 4, 2
    cells[8, 2 + index] = 0
    return to_rows(np.eye(7, dtype=np.float32)[cells])


def make_plan(count=4):
    cases = []
    for index in range(count):
        rows = board(index)
        for method in generator.METHODS:
            for transform in generator.TRANSFORMS:
                context = f"ctx-{index}"
                cases.append({"case_id": generator.case_id("dev", context, method, transform),
                              "cohort_id": "dev", "context_id": context,
                              "layout_id": canonical_layout_id(generator.from_rows(rows)), "family": "12x12x4",
                              "source_family": "test", "split": "development", "method": str(method),
                              "transform": str(transform), "original_rows": rows,
                              "source_rows": generator.transform_rows(rows, transform),
                              "adapter_training_exposure": index == 0})
    assert len({case["layout_id"] for case in cases}) == count
    return {"cohort_id": "dev", "cases": cases}


def intent(row=1, col=1):
    return json.dumps({"effects": [{"predicate": "box_at", "row": row, "col": col}],
                       "preserve": [], "instruction": "ignored for structural identity"})


def prediction(case, raw, contract="contract"):
    return {**generator.output_record(case, raw, 5, 0.1), "contract_sha256": contract}


def test_four_distinct_boards_trigger_without_counting_sixteen_transforms_as_boards():
    plan = make_plan()
    predictions = [prediction(case, intent()) for case in plan["cases"]]
    result = decision.analyse(plan, predictions)
    assert result["distinct_layouts"] == 4 and result["known_adapter_training_layouts"] == 1
    for method in generator.METHODS:
        stats = result["methods"][method]
        assert stats["valid_identity_layouts"] == 4 and stats["most_common_intent_share"] == 1.0
        assert stats["eligible_coordinate_layouts"] == 4 and stats["eligible_transform_pairs"] == 8
        assert stats["literal_coordinate_copy_share_equal_layout"] == 1.0
    assert result["recommendation"] == "train_only_d4_augmentation_and_coordinate_balancing"
    assert result["provisional"] is False


def test_transform_reuse_is_averaged_per_original_board_and_small_n_is_provisional():
    plan = make_plan(2)
    predictions = []
    for case in plan["cases"]:
        if case["method"] != "base":
            continue
        if case["context_id"] == "ctx-1" and case["transform"] == "horizontal_reflection":
            continue
        altered = case["context_id"] == "ctx-1" and case["transform"] == "rot90_ccw"
        predictions.append(prediction(case, intent(col=2 if altered else 1)))
    result = decision.analyse(plan, predictions)
    base = result["methods"]["base"]
    assert base["eligible_coordinate_layouts"] == 2 and base["eligible_transform_pairs"] == 3
    assert base["literal_coordinate_copy_pairs"] == 2
    assert base["literal_coordinate_copy_share_equal_layout"] == 0.5
    assert result["provisional"] and result["recommendation"] == "insufficient_development_evidence"


def test_duplicate_keys_and_missing_outputs_are_not_valid_examples():
    plan = make_plan(1)
    identity = next(case for case in plan["cases"] if case["method"] == "base" and case["transform"] == "identity")
    duplicate = '{"effects":[],"effects":[],"preserve":[]}'
    result = decision.analyse(plan, [prediction(identity, duplicate)])
    assert result["methods"]["base"]["invalid_identity_outputs"] == 1
    assert result["methods"]["base"]["valid_identity_layouts"] == 0
    assert result["unobserved_cases"] == 8 and result["provisional"]


def test_freeze_requires_development_only_and_fails_after_evaluation_output(tmp_path):
    plan = make_plan(1)
    run = tmp_path / "benchmark"
    run.mkdir()
    contract = {"split": "development", "plan_sha256": digest(plan)}
    predictions = [prediction(case, intent(), digest(contract)) for case in plan["cases"]]
    atomic_json(run / "plan.json", plan)
    atomic_json(run / "contract.json", contract)
    atomic_json(run / "status.json", {"state": "complete"})
    (run / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    protocol = {"methods": [str(item) for item in generator.METHODS],
                "transformations": [str(item) for item in generator.TRANSFORMS],
                "fresh_diagnostic_intervention_rule": "first averaging over valid transforms within each layout; "
                    "at least 0.5 and at least four observations; Do not use evaluation outputs"}
    protocol_path = tmp_path / "evaluation_plan.json"
    atomic_json(protocol_path, protocol)
    output = tmp_path / "dev_decision.json"
    result = decision.freeze_decision(run, protocol_path, output)
    assert result["test_results_accessed"] is False and len(result["source_sha256"]) == 5
    with pytest.raises(FileExistsError):
        decision.freeze_decision(run, protocol_path, output)
    future = tmp_path / "fresh_evaluation/predictions.jsonl"
    future.parent.mkdir()
    future.write_text("")
    with pytest.raises(ValueError, match="evaluation output already exists"):
        decision.freeze_decision(run, protocol_path, tmp_path / "late.json")
