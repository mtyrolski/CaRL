import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from carl.icl import intent_assessment
from carl.icl import intent_fence_audit
from carl.icl.intent_assessment import AssessmentStatus
from carl.icl.intent_assessment import Method
from carl.icl.intent_assessment import SearchStatus
from carl.icl.intent_assessment import assess_context
from carl.icl.intent_assessment import bounded_witness
from carl.icl.intent_assessment import compare_metadata
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_assessment import local_rule
from carl.icl.intent_assessment import predictions_for
from carl.icl.intent_assessment import recover_progress
from carl.icl.intent_assessment import step
from carl.icl.intent_assessment import summarise
from carl.icl.intent_assessment import write_examples
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
        "effects": [{
            "predicate": "box_at",
            "row": row,
            "col": col
        }],
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
    walls = ({(0, c) for c in range(6)} | {(5, c) for c in range(6)} | {(r, 0) for r in range(6)} |
             {(r, 5) for r in range(6)})
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
    row = assess_context(record, Method.BASE, {Method.BASE: {"r1": prediction}}, 2, 10, 1)
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
            } if valid else None,
        }

    summary, _ = summarise([row("many", False) for _ in range(10)] + [row("one", True)])
    base = summary[0]
    assert base["valid_fraction"] == pytest.approx(1 / 11)
    assert base["board_mean_valid_fraction"] == pytest.approx(0.5)
    assert base["board_mean_witness_fraction"] == pytest.approx(0.5)
    assert base["valid_fraction_95_board_bootstrap"] is None


def test_rejects_duplicate_and_inconsistent_predictions(tmp_path):
    data, ids, path = (tmp_path / name for name in ("dataset.jsonl", "frozen.json", "base.jsonl"))
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
        json.dumps({
            "status": "complete",
            "n": 1,
            "data_sha256": file_hash(data),
            "test_ids_sha256": file_hash(ids),
            "adapter": None,
        }))
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
        compare_metadata({Method.BASE: common, Method.ADAPTED: dict(common, revision="different")})


def test_resume_preserves_partial_measurements_and_interrupted_bytes(tmp_path):
    output = tmp_path / "assessment.jsonl"
    config = {"schema_version": 1, "data_sha256": "abc"}
    contexts = [{"record_id": "r1", "layout_id": "board-1"}]
    assert recover_progress(output, config, contexts, resume=False) == []
    complete = {
        "record_id": "r1",
        "layout_id": "board-1",
        "method": "base",
        "status": "evaluated",
    }
    original = (json.dumps(complete) + '\n{"method":').encode()
    output.write_bytes(original)
    with pytest.raises(ValueError, match="already contains"):
        recover_progress(output, config, contexts, resume=False)
    assert recover_progress(output, config, contexts, resume=True) == [complete]
    assert (list(tmp_path.glob("assessment.jsonl.interrupted-*"))[0].read_bytes() == original)
    with pytest.raises(ValueError, match="configuration"):
        recover_progress(output, dict(config, data_sha256="changed"), contexts, resume=True)


def test_examples_distinguish_demonstration_from_solver_witness(tmp_path):
    reached = ["######", "#--@*#", "#----#", "#----#", "#----#", "######"]
    row = {
        "record_id": "r1",
        "layout_id": "board-1",
        "method": Method.ADAPTED,
        "status": AssessmentStatus.EVALUATED,
        "valid_nontrivial": True,
        "start_observation": BOARD,
        "demonstration_later_observation": BOARD,
        "raw_output": json.dumps(intent(1, 4)),
        "intent": intent(1, 4),
        "validation_errors": [],
        "solver": {
            "witness_found": True,
            "witness_depth": 2,
            "reached_observation": reached,
        },
    }
    write_examples([row], tmp_path)
    example = json.loads((tmp_path / "examples.json").read_text())[0]
    assert example["predicate_truth"][0]["demonstration_end"] is False
    assert example["predicate_truth"][0]["solver_end"] is True
    assert (example["demonstration_end_board"] != example["solver"]["reached_observation"])


def test_complete_and_resumed_assessment_with_posthoc_audit(tmp_path, monkeypatch):
    data = tmp_path / "dataset.jsonl"
    record = {
        "record_id": "r1",
        "layout_id": "board-1",
        "split": "test",
        "start_observation": BOARD,
        "later_observation": BOARD,
    }
    data.write_text(json.dumps(record) + "\n")
    ids = tmp_path / "frozen.json"
    ids.write_text(json.dumps({"record_ids": ["r1"], "dataset_sha256": file_hash(data)}))
    answer = intent(1, 4)
    base, adapted = tmp_path / "base.jsonl", tmp_path / "adapted.jsonl"
    common = {"record_id": "r1", "layout_id": "board-1", "split": "test"}
    base.write_text(
        json.dumps({
            **common,
            "method": "base",
            "raw_output": f"```json\n{json.dumps(answer)}\n```",
            "json_valid": False,
            "parsed": None,
        }) + "\n")
    adapted.write_text(
        json.dumps({
            **common,
            "method": "adapted",
            "raw_output": json.dumps(answer),
            "json_valid": True,
            "parsed": answer,
        }) + "\n")
    for path, adapter in ((base, None), (adapted, "adapter")):
        path.with_suffix(".summary.json").write_text(
            json.dumps({
                "status": "complete",
                "n": 1,
                "model": "model",
                "revision": "rev",
                "seed": 1,
                "decoding": "greedy",
                "enable_thinking": False,
                "max_new_tokens": 128,
                "adapter": adapter,
                "data_sha256": file_hash(data),
                "test_ids_sha256": file_hash(ids),
            }))
    output = tmp_path / "assessment"
    common_args = [
        "--data",
        str(data),
        "--test-ids",
        str(ids),
        "--base",
        str(base),
        "--adapted",
        str(adapted),
    ]
    monkeypatch.setattr(sys, "argv", ["assessment", "assess", *common_args, "--output", str(output)])
    write_examples_real = intent_assessment.write_examples

    def fail_examples(*_args):
        raise OSError("simulated interruption writing examples")

    monkeypatch.setattr(intent_assessment, "write_examples", fail_examples)
    with pytest.raises(OSError, match="simulated interruption"):
        intent_assessment.main()
    assert not (output / "summary.json").exists()
    original = (output / "assessment.jsonl").read_bytes()
    assert len(original.splitlines()) == 3
    monkeypatch.setattr(intent_assessment, "write_examples", write_examples_real)
    monkeypatch.setattr(sys, "argv", ["assessment", "assess", *common_args, "--output", str(output), "--resume"])
    intent_assessment.main()
    summary = json.loads((output / "summary.json").read_text())["results"]
    assert [(row["valid_nontrivial"], row["witness_found"]) for row in summary] == [
        (0, 0),
        (1, 1),
        (1, 1),
    ]
    assert (output / "assessment.jsonl").read_bytes() == original
    monkeypatch.setattr(
        sys,
        "argv",
        ["assessment", "assess", *common_args, "--output",
         str(output), "--resume"],
    )
    intent_assessment.main()
    assert (output / "assessment.jsonl").read_bytes() == original
    fence_output = tmp_path / "fence"
    monkeypatch.setattr(sys, "argv", ["fence_audit", *common_args, "--output", str(fence_output)])
    atomic_write_real = intent_fence_audit.atomic_write

    def fail_fence_csv(path, text):
        if path.name == "fence_audit_summary.csv":
            raise OSError("simulated interruption writing fence CSV")
        atomic_write_real(path, text)

    monkeypatch.setattr(intent_fence_audit, "atomic_write", fail_fence_csv)
    with pytest.raises(OSError, match="simulated interruption"):
        intent_fence_audit.main()
    assert not (fence_output / "fence_audit_summary.json").exists()
    fence_original = (fence_output / "fence_audit.jsonl").read_bytes()
    monkeypatch.setattr(intent_fence_audit, "atomic_write", atomic_write_real)
    monkeypatch.setattr(sys, "argv", ["fence_audit", *common_args, "--output", str(fence_output), "--resume"])
    intent_fence_audit.main()
    assert (fence_output / "fence_audit.jsonl").read_bytes() == fence_original
    assert (fence_output / "fence_audit_summary.csv").exists()
    fence_summary = json.loads((fence_output / "fence_audit_summary.json").read_text())["results"]
    assert fence_summary[0]["strict_valid_nontrivial"] == 0
    assert fence_summary[0]["posthoc_valid_nontrivial"] == 1


def test_publication_figure_exports_vector_and_png_with_missing_method(tmp_path):
    summary = tmp_path / "summary.json"
    result = {
        "outputs": 1,
        "board_mean_valid_fraction": 1.0,
        "board_mean_witness_fraction": 1.0,
        "valid_fraction_95_board_bootstrap": None,
        "witness_fraction_95_board_bootstrap": None,
        "valid_nontrivial": 1,
        "witness_found": 1,
    }
    missing = dict(
        result,
        outputs=0,
        board_mean_valid_fraction=None,
        board_mean_witness_fraction=None,
    )
    summary.write_text(
        json.dumps({
            "results": [
                dict(missing, method="base"),
                dict(result, method="adapted"),
                dict(result, method="local_rule"),
            ]
        }))
    output = tmp_path / "figure"
    script = Path(__file__).resolve().parents[2] / "scripts/plot_intent_assessment.py"
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(summary), "--output",
            str(output), "--title", "Post-hoc: one outer fence removed"
        ],
        check=True,
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert all(output.with_suffix(ext).stat().st_size > 1000 for ext in (".pdf", ".svg", ".png"))
    assert "Post-hoc: one outer fence removed" in output.with_suffix(".svg").read_text()
