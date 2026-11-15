import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from carl.icl import intent_assessment
from carl.icl.intent_assessment import AssessmentStatus
from carl.icl.intent_assessment import Method
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_assessment import recover_progress
from carl.icl.intent_assessment import write_examples

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


def test_in_context_assessment_keeps_method_identity(
    tmp_path, monkeypatch
):
    data, ids = tmp_path / "dataset.jsonl", tmp_path / "frozen.json"
    record = {
        "record_id": "r1",
        "layout_id": "board-1",
        "split": "test",
        "start_observation": BOARD,
        "later_observation": BOARD,
    }
    data.write_text(json.dumps(record) + "\n")
    ids.write_text(
        json.dumps({"record_ids": ["r1"], "dataset_sha256": file_hash(data)})
    )
    answer = intent(1, 4)
    paths = {
        method: tmp_path / f"{method}.jsonl"
        for method in (Method.BASE, Method.ADAPTED, Method.IN_CONTEXT)
    }
    for method, path in paths.items():
        raw = json.dumps(answer)
        if method == Method.IN_CONTEXT:
            raw = f"```json\n{raw}\n```"
        valid = method != Method.IN_CONTEXT
        path.write_text(
            json.dumps(
                {
                    "record_id": "r1",
                    "layout_id": "board-1",
                    "split": "test",
                    "method": method,
                    "raw_output": raw,
                    "json_valid": valid,
                    "parsed": answer if valid else None,
                }
            )
            + "\n"
        )
        path.with_suffix(".summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "n": 1,
                    "model": "model",
                    "revision": "rev",
                    "seed": 1,
                    "decoding": "greedy",
                    "enable_thinking": False,
                    "max_new_tokens": 128,
                    "adapter": "adapter" if method == Method.ADAPTED else None,
                    "data_sha256": file_hash(data),
                    "test_ids_sha256": file_hash(ids),
                }
            )
        )
    arguments = [
        "--data",
        str(data),
        "--test-ids",
        str(ids),
        "--base",
        str(paths[Method.BASE]),
        "--adapted",
        str(paths[Method.ADAPTED]),
        "--in-context",
        str(paths[Method.IN_CONTEXT]),
    ]
    strict = tmp_path / "strict"
    monkeypatch.setattr(
        sys, "argv", ["assessment", "assess", *arguments, "--output", str(strict)]
    )
    intent_assessment.main()
    result = json.loads((strict / "summary.json").read_text())
    by_method = {row["method"]: row for row in result["results"]}
    assert by_method["in_context"]["valid_json"] == 0
    assert result["paired_in_context_minus_base"]["contexts"] == 1
    assert result["paired_in_context_minus_base"]["valid_delta"] == -1
    assert result["comparison_metadata_missing"]["base"] == ["dtype", "prompt_sha256"]

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
    assert (
        list(tmp_path.glob("assessment.jsonl.interrupted-*"))[0].read_bytes()
        == original
    )
    with pytest.raises(ValueError, match="configuration"):
        recover_progress(
            output, dict(config, data_sha256="changed"), contexts, resume=True
        )


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
    assert (
        example["demonstration_end_board"] != example["solver"]["reached_observation"]
    )


def test_complete_and_resumed_assessment(tmp_path, monkeypatch):
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
    ids.write_text(
        json.dumps({"record_ids": ["r1"], "dataset_sha256": file_hash(data)})
    )
    answer = intent(1, 4)
    base, adapted = tmp_path / "base.jsonl", tmp_path / "adapted.jsonl"
    common = {"record_id": "r1", "layout_id": "board-1", "split": "test"}
    base.write_text(
        json.dumps(
            {
                **common,
                "method": "base",
                "raw_output": f"```json\n{json.dumps(answer)}\n```",
                "json_valid": False,
                "parsed": None,
            }
        )
        + "\n"
    )
    adapted.write_text(
        json.dumps(
            {
                **common,
                "method": "adapted",
                "raw_output": json.dumps(answer),
                "json_valid": True,
                "parsed": answer,
            }
        )
        + "\n"
    )
    for path, adapter in ((base, None), (adapted, "adapter")):
        path.with_suffix(".summary.json").write_text(
            json.dumps(
                {
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
                }
            )
        )
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
    monkeypatch.setattr(
        sys, "argv", ["assessment", "assess", *common_args, "--output", str(output)]
    )
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
    monkeypatch.setattr(
        sys,
        "argv",
        ["assessment", "assess", *common_args, "--output", str(output), "--resume"],
    )
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
        ["assessment", "assess", *common_args, "--output", str(output), "--resume"],
    )
    intent_assessment.main()
    assert (output / "assessment.jsonl").read_bytes() == original
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
        json.dumps(
            {
                "results": [
                    dict(missing, method="base"),
                    dict(result, method="adapted"),
                    dict(result, method="local_rule"),
                ]
            }
        )
    )
    output = tmp_path / "figure"
    script = Path(__file__).resolve().parents[2] / "scripts/plot_intent_assessment.py"
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(summary),
            "--output",
            str(output),
            "--title",
            "Post-hoc: one outer fence removed",
        ],
        check=True,
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert all(
        output.with_suffix(ext).stat().st_size > 1000
        for ext in (".pdf", ".svg", ".png")
    )
    assert "Post-hoc: one outer fence removed" in output.with_suffix(".svg").read_text()
