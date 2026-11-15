"""Merging preserves missing jobs and rejects altered or mismatched records."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.merge_overnight_stage05_shards import merge


def setup_stage(stage: Path) -> dict:
    board = {"layout_id": "layout-a", "family": "12x12x4", "source_family": "fixture", "split": "evaluation"}
    boards = stage / "boards.jsonl"
    data = (json.dumps(board) + "\n").encode()
    boards.write_bytes(data)
    freeze = {"input_boards": str(boards), "input_boards_sha256": hashlib.sha256(data).hexdigest(),
              "methods": ["adasubs", "direct_policy"]}
    (stage / "evaluation_freeze.json").write_text(json.dumps(freeze))
    return board


def test_partial_and_complete_merge_keeps_original_bytes(tmp_path: Path) -> None:
    board = setup_stage(tmp_path)
    source = tmp_path / "adasubs_evaluation/shards/result.json"
    source.parent.mkdir(parents=True)
    original = json.dumps({**board, "method": "adasubs", "status": "completed"}).encode()
    source.write_bytes(original)
    result = merge(tmp_path)
    assert (result["state"], result["merged_jobs"], result["planned_jobs"]) == ("partial", 1, 2)
    assert (tmp_path / "evaluation/shards/adasubs/layout-a.json").read_bytes() == original
    policy = tmp_path / "policy_evaluation/shards/direct_policy/result.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(json.dumps({**board, "method": "direct_policy", "status": "infra_error"}))
    result = merge(tmp_path)
    assert (result["state"], result["merged_jobs"]) == ("complete", 2)
    source.write_text(json.dumps({**board, "method": "adasubs", "status": "changed"}))
    with pytest.raises(ValueError, match="Previously merged shard changed"):
        merge(tmp_path)


def test_rejects_mismatched_board_metadata(tmp_path: Path) -> None:
    board = setup_stage(tmp_path)
    source = tmp_path / "adasubs_evaluation/shards/result.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({**board, "method": "adasubs", "family": "12x12x10"}))
    with pytest.raises(ValueError, match="metadata"):
        merge(tmp_path)
