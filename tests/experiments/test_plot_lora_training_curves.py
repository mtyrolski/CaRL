"""Checks for plotting recorded LoRA diagnostics without a model or GPU."""

import csv
import json
from pathlib import Path

import pytest

from scripts import plot_lora_training_curves as curves


def training_fixture(run: Path, representation: str) -> Path:
    directory = run / f"adapter_repr_{representation}_seed1729"
    directory.mkdir(parents=True)
    rows = [{"step": 1, "train_loss": 1.0, "seconds": .5},
            {"step": 2, "train_loss": .7, "validation_loss": .6, "seconds": .5},
            {"step": 3, "train_loss": .5, "validation_loss": .8, "seconds": .5}]
    (directory / "training_curve.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (directory / "configuration.json").write_text(json.dumps({"representation": representation, "seed": 1729, "rank": 4}))
    (directory / "best.json").write_text(json.dumps({"step": 2, "validation_loss": .6, "adapter": "checkpoint-2"}))
    (directory / "status.json").write_text(json.dumps({"status": "finished", "completed_steps": 3,
                                                        "best_adapter": "checkpoint-2",
                                                        "last_adapter": "checkpoint-last-3"}))
    for name in ("checkpoint-2", "checkpoint-last-3"):
        (directory / name).mkdir()
        (directory / name / "adapter_model.safetensors").write_bytes(name.encode())
    return directory


def test_figure_contains_observed_rows_and_provenance(tmp_path: Path) -> None:
    for representation in curves.REPRESENTATIONS:
        training_fixture(tmp_path, representation)
    destination = tmp_path / "figure"
    assert curves.create_outputs(tmp_path, destination) == destination
    with (destination / "lora_training_loss.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert sum(row["validation_loss"] != "" for row in rows) == 6
    assert sum(row["selected_checkpoint"] == "True" for row in rows) == 3
    provenance = json.loads((destination / "provenance.json").read_text())
    assert {item["best_step"] for item in provenance["methods"]} == {2}
    assert all(not item["selected_weights_match_last"] for item in provenance["methods"])
    for name, digest in provenance["outputs"].items():
        assert curves.sha256(destination / name) == digest
    with pytest.raises(FileExistsError):
        curves.create_outputs(tmp_path, destination)


def test_rejects_checkpoint_that_is_not_recorded_validation_minimum(tmp_path: Path) -> None:
    directory = training_fixture(tmp_path, "json")
    (directory / "best.json").write_text(json.dumps({"step": 3, "validation_loss": .8,
                                                     "adapter": "checkpoint-3"}))
    with pytest.raises(ValueError, match="validation minimum"):
        curves.read_training(tmp_path, "json")


def test_rejects_missing_training_step(tmp_path: Path) -> None:
    directory = training_fixture(tmp_path, "formal")
    lines = (directory / "training_curve.jsonl").read_text().splitlines()
    (directory / "training_curve.jsonl").write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    with pytest.raises(ValueError, match="invalid or missing observations"):
        curves.read_training(tmp_path, "formal")
