"""Campaign layout identity, test-data isolation, and reproducibility contracts."""
import json
from pathlib import Path

import numpy as np
import pytest

from carl.icl.boards import canonical_layout_id, from_rows, state_hash
from carl.icl.config import ExperimentConfig, load_config
from carl.icl.datasets import _public_instances, load_data

ROOT = Path(__file__).resolve().parents[2]
BOARD = ["######", "#-@--#", "#-$.-#", "#----#", "######"]


def test_canonical_layout_respects_all_rotations_and_reflections() -> None:
    board = from_rows(BOARD)
    expected = canonical_layout_id(board)
    for reflected in (board, np.fliplr(board)):
        for rotation in range(4):
            assert canonical_layout_id(np.rot90(reflected, rotation)) == expected
    moved = from_rows(["######", "#--@-#", "#-$.-#", "#----#", "######"])
    assert canonical_layout_id(moved) == expected
    assert state_hash(moved) != state_hash(board)
    changed_goals = from_rows(["######", "#-@--#", "#-$.--", "#----#", "######"])
    assert canonical_layout_id(changed_goals) != expected


@pytest.mark.parametrize("field,value", [("actions", [0, 1]), ("witness", [1]),
                                         ("future_state", BOARD), ("subgoal", BOARD)])
def test_public_manifest_rejects_private_and_future_fields(tmp_path: Path, field: str, value: object) -> None:
    from carl.icl.boards import level_key

    row = {"id": "board", "state": BOARD, "level_key": level_key(from_rows(BOARD)), field: value}
    path = tmp_path / "test.json"
    path.write_text(json.dumps({"provenance": {}, "instances": [row]}))
    with pytest.raises(ValueError, match="private or future|Unexpected fields"):
        _public_instances(path)
    path.write_text(json.dumps({"provenance": {"hidden": {field: value}},
                                "instances": [{k: v for k, v in row.items() if k != field}]}))
    with pytest.raises(ValueError, match="private or future"):
        _public_instances(path)


def test_pilot_loader_does_not_open_holdout_private_or_source(monkeypatch: pytest.MonkeyPatch) -> None:
    data = ROOT / "data/icl/campaign_dataset_v2"
    if not (data / "campaign_dataset_manifest.json").exists():
        pytest.skip("Prepare campaign dataset first")
    base = load_config("configs/icl/experiments/tuned_fixed.json")
    original_text, original_bytes = Path.read_text, Path.read_bytes

    def guarded_text(path: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if "holdout1000" in str(path) or "campaign_dataset_v2_private" in str(path) or path.suffix == ".pkl":
            raise AssertionError("Pilot loader accessed a held-out or private source")
        return original_text(path, encoding=encoding, errors=errors)

    def guarded_bytes(path: Path) -> bytes:
        if "holdout1000" in str(path) or "campaign_dataset_v2_private" in str(path) or path.suffix == ".pkl":
            raise AssertionError("Pilot loader accessed a held-out or private source")
        return original_bytes(path)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    for family in ("real_12x12x4", "generated_32x32x10"):
        config = ExperimentConfig.model_validate({**base.model_dump(), "data_dir": str(data),
                    "family": family, "split": "pilot100", "instances": 50, "easy": False})
        _training, instances = load_data(config)
        assert len(instances) == 50
        assert all(set(record) == {"id", "state", "level_key"} for record in instances)
    config = ExperimentConfig.model_validate({**base.model_dump(), "data_dir": str(data),
                                               "split": "pilot100", "easy": True})
    with pytest.raises(ValueError, match="explicit full-board"):
        load_data(config)


def test_campaign_split_summary_and_explicit_history_exclusions() -> None:
    audit = ROOT / "analysis/luna_icl/campaign_100/data_integrity.json"
    if not audit.exists():
        pytest.skip("Prepare campaign dataset first")
    data = json.loads(audit.read_text())
    summary = data["counts_and_witness_statistics"]
    assert data["shards_indexed"] == 112
    assert data["cross_shard_duplicate_layouts"] > 0
    for split, family, count in (("pilot100", "real_12x12x4", 50),
                                 ("pilot100", "generated_32x32x10", 50),
                                 ("holdout1000", "real_12x12x4", 500),
                                 ("holdout1000", "generated_32x32x10", 500)):
        group = summary["groups"][f"{split}/{family}"]
        assert group["count"] == group["unique_layouts"] == count
    assert not any(summary["intersections"].values())
    history = json.loads((audit.parent / "history_snapshot.json").read_text())
    assert {199, 434, 732}.issubset(history["observed_legacy_trajectory_indices"])


def test_validation_diagnostics_require_explicit_private_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from carl.icl.api import CachedOpenAI
    from scripts.evaluate_trajectory_components import parser, run

    public = ROOT / "data/icl/campaign_dataset_v2"
    private = ROOT / "data/icl/campaign_dataset_v2_private"
    if not public.exists() or not private.exists():
        pytest.skip("Prepare campaign dataset first")
    monkeypatch.setattr(CachedOpenAI, "_client", lambda _self: pytest.fail("Mock must not call an API"))
    args = ["--config", "configs/icl/campaign_data/component_validation_real.json", "--component", "policy",
            "--split", "component_validation", "--indices", "0", "--horizons", "4", "--positions", "start",
            "--max-requests", "8", "--output", str(tmp_path / "report"), "--cache-dir", str(tmp_path / "cache"), "--mock"]
    with pytest.raises(ValueError, match="explicit --private-data-dir"):
        run(parser().parse_args(args))
    report = run(parser().parse_args([*args, "--private-data-dir", str(private)]))
    data = json.loads(report.read_text())
    assert data["provider"] == "mock" and data["summary"]["cases"] == 1
