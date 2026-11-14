"""The local-only ablation must never filter validation or test examples."""

import json

import pytest

from scripts.prepare_lora_local_only import filter_records


def test_filters_only_historical_terra_train_and_preserves_other_lines():
    rows = [
        {"record_id": "local", "split": "train", "annotation_provenance": {"kind": "local_exact_difference", "model": None}},
        {"record_id": "terra", "split": "train", "annotation_provenance": {"kind": "historical_shared_context_codex_annotation", "model": "gpt-6.1-terra"}},
        {"record_id": "val", "split": "validation", "annotation_provenance": {"kind": "historical_shared_context_codex_annotation", "model": "gpt-6.1-terra"}},
        {"record_id": "test", "split": "test", "annotation_provenance": {"kind": "historical_shared_context_codex_annotation", "model": "gpt-6.1-terra"}},
    ]
    lines = [json.dumps(row, indent=None, separators=(", ", ": ")) + "\n" for row in rows]
    output, removed, counts = filter_records("".join(lines))
    assert output == lines[0] + lines[2] + lines[3]
    assert removed == ["terra"]
    assert counts == {"train": 1, "validation": 1, "test": 1}


def test_ambiguous_terra_train_provenance_is_not_silently_removed():
    row = {"record_id": "ambiguous", "split": "train", "annotation_provenance": {"kind": "unknown", "model": "gpt-6.1-terra"}}
    with pytest.raises(ValueError, match="provenance"):
        filter_records(json.dumps(row) + "\n")
