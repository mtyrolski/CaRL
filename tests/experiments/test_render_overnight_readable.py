"""Checks for the saved-results presentation, without model inference."""

import hashlib
import json

import pytest

from scripts.render_overnight_readable import prompt_for
from scripts.render_overnight_readable import record_table
from scripts.render_overnight_readable import render_board
from scripts.render_overnight_readable import verify_manifest


def test_board_displays_coordinates_and_escapes_label():
    rows = ["#" * 12 for _ in range(12)]
    rendered = render_board(rows, '<test "board">', (2, 4))

    assert len(rendered.split('class="tile')) == 145
    assert 'class="tile wall target"' in rendered
    assert "row 2, column 4: wall" in rendered
    assert "&lt;test &quot;board&quot;&gt;" in rendered
    with pytest.raises(ValueError, match="12 by 12"):
        render_board(rows[:-1], "broken")


def test_manifest_rejects_a_changed_saved_result(tmp_path):
    source = tmp_path / "source.json"
    output = tmp_path / "report.html"
    source.write_text('{"count": 2}', encoding="utf-8")
    output.write_text("report", encoding="utf-8")
    manifest = {
        "input_sha256": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
        "output_sha256": {output.name: hashlib.sha256(output.read_bytes()).hexdigest()},
    }

    assert verify_manifest(tmp_path, manifest) == 2
    source.write_text('{"count": 3}', encoding="utf-8")
    with pytest.raises(ValueError, match="input_sha256 mismatch"):
        verify_manifest(tmp_path, manifest)


def test_prompt_is_the_saved_input_for_the_selected_prediction(tmp_path):
    folder = tmp_path / "evaluation"
    folder.mkdir()
    text = "<system> Board and task only."
    prompt_hash = hashlib.sha256(text.encode()).hexdigest()
    prediction = {
        "method": "wide",
        "family": "12x12x4",
        "board_rank": 0,
        "transform": "identity",
        "case_id": "first",
        "prompt_sha256": prompt_hash,
    }
    prompt = {
        "case_id": "first",
        "messages": [{"role": "system", "content": "Board and task only."}],
        "rendered_prompt": text,
        "prompt_sha256": prompt_hash,
    }
    (folder / "predictions.jsonl").write_text(
        json.dumps(prediction) + "\n", encoding="utf-8"
    )
    (folder / "prompts.jsonl").write_text(json.dumps(prompt) + "\n", encoding="utf-8")

    found_prompt, found_prediction = prompt_for(
        tmp_path, "evaluation", "wide", "12x12x4"
    )
    assert found_prompt == prompt
    assert found_prediction == prediction
    prompt["rendered_prompt"] += " changed"
    (folder / "prompts.jsonl").write_text(json.dumps(prompt) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt hash"):
        prompt_for(tmp_path, "evaluation", "wide", "12x12x4")


def test_example_table_keeps_unresolved_and_invalid_cases_separate(tmp_path):
    folder = tmp_path / "stage_04/evaluation"
    folder.mkdir(parents=True)
    outputs = [
        {
            "method": "base",
            "raw_output": "<bad>",
            "valid_nontrivial": False,
            "bfs_status": "not_applicable_invalid_intent",
            "bounded_path_found": None,
            "bfs_search": None,
        },
        {
            "method": "wide",
            "raw_output": "{}",
            "valid_nontrivial": True,
            "bfs_status": "node_limit",
            "bounded_path_found": False,
            "bfs_search": {},
        },
    ]
    predictions = [
        {
            "context_id": "board",
            "transform": "identity",
            "method": out["method"],
            "raw_output": out["raw_output"],
            "validation_errors": ["invalid"] if out["method"] == "base" else [],
        }
        for out in outputs
    ]
    (folder / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )

    rendered = record_table(tmp_path, {"context_id": "board", "outputs": outputs})
    assert "&lt;bad&gt;" in rendered
    assert "Not run for invalid intent" in rendered
    assert "Unresolved: node limit" in rendered
