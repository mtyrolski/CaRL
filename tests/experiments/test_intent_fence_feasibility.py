import json
import sys

import pytest

from carl.icl import intent_fence_feasibility
from carl.icl.intent_assessment import Method
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_fence_feasibility import normalise_prediction

BOARD = ["######", "#@-$.#", "#----#", "#----#", "#----#", "######"]
INTENT = {"effects": [{"predicate": "box_at", "row": 1, "col": 4}], "preserve": [], "instruction": None}


def test_normalisation_removes_only_one_complete_outer_fence():
    fenced = f"```json\n{json.dumps(INTENT)}\n```"
    prediction = {"raw_output": fenced, "json_valid": False, "parsed": None}
    normalised, provenance = normalise_prediction(prediction, BOARD)
    assert normalised["raw_output"] == fenced
    assert normalised["json_valid"] and normalised["parsed"] == INTENT
    assert provenance == {"strict_json_valid": False, "fence_removed": True}
    bad, provenance = normalise_prediction(dict(prediction, raw_output="extra\n" + fenced), BOARD)
    assert not bad["json_valid"] and not provenance["fence_removed"]


def test_posthoc_witnesses_are_paired_and_resumable(tmp_path, monkeypatch):
    data = tmp_path / "dataset.jsonl"
    record = {
        "record_id": "r1",
        "layout_id": "board-1",
        "split": "test",
        "start_observation": BOARD,
        "later_observation": BOARD
    }
    data.write_text(json.dumps(record) + "\n")
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"record_ids": ["r1"], "dataset_sha256": file_hash(data)}))
    base, adapted = tmp_path / "base.jsonl", tmp_path / "adapted.jsonl"
    common = {"record_id": "r1", "layout_id": "board-1", "split": "test"}
    base.write_text(
        json.dumps({
            **common, "method": "base",
            "raw_output": f"```json\n{json.dumps(INTENT)}\n```",
            "json_valid": False,
            "parsed": None
        }) + "\n")
    adapted.write_text(
        json.dumps({
            **common, "method": "adapted",
            "raw_output": json.dumps(INTENT),
            "json_valid": True,
            "parsed": INTENT
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
    input_hashes = file_hash(base), file_hash(adapted)
    output = tmp_path / "posthoc"
    args = [
        "--data",
        str(data), "--test-ids",
        str(ids), "--base",
        str(base), "--adapted",
        str(adapted), "--output",
        str(output)
    ]
    assess_real = intent_fence_feasibility.assess_context

    def interrupt_after_base(record, method, *rest):
        if method == Method.ADAPTED:
            raise OSError("simulated interruption")
        return assess_real(record, method, *rest)

    monkeypatch.setattr(intent_fence_feasibility, "assess_context", interrupt_after_base)
    monkeypatch.setattr(sys, "argv", ["feasibility", *args])
    with pytest.raises(OSError, match="simulated interruption"):
        intent_fence_feasibility.main()
    assert not (output / "summary.json").exists()
    first = (output / "feasibility.jsonl").read_bytes()
    assert len(first.splitlines()) == 1
    monkeypatch.setattr(intent_fence_feasibility, "assess_context", assess_real)
    monkeypatch.setattr(sys, "argv", ["feasibility", *args, "--resume"])
    intent_fence_feasibility.main()
    assert (output / "feasibility.jsonl").read_bytes().startswith(first)
    summary = json.loads((output / "summary.json").read_text())
    assert [(row["outputs"], row["valid_nontrivial"], row["witness_found"]) for row in summary["results"]
           ] == [(1, 1, 1), (1, 1, 1)]
    assert summary["results"][0]["strict_valid_json"] == 0
    assert summary["results"][0]["single_outer_fence_removed"] == 1
    assert summary["paired_adapted_minus_base"]["boards"] == 1
    assert summary["paired_adapted_minus_base"]["witness_delta"] == 0
    assert "not controller" in summary["solver"]["meaning"]
    assert (output / "summary.csv").exists() and (output / "examples.md").exists()
    assert (file_hash(base), file_hash(adapted)) == input_hashes
