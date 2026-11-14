"""The LoRA matrix preserves matched-step and validation-best provenance."""

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from scripts.summarize_lora_matrix import summarize


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def save_adapter(directory, name, rank):
    destination = directory / name
    destination.mkdir(parents=True)
    weights = np.full((rank, 2), sum(name.encode()), dtype=np.float32)
    save_file({"lora_A": weights}, str(destination / "adapter_model.safetensors"))
    save_json(destination / "adapter_config.json", {"r": rank, "target_modules": ["v_proj", "q_proj"]})


def test_matched_and_best_are_separate_and_hashed(tmp_path):
    adapter = tmp_path / "adapter"
    save_json(adapter / "configuration.json", {
        "args": {"rank": 4, "lr": 0.0002, "max_steps": 300, "seed": 1729, "accumulation": 4},
        "data_sha256": "source-hash", "validation_record_ids": ["validation-a"]})
    save_json(adapter / "status.json", {"status": "finished", "completed_steps": 300})
    save_json(adapter / "best.json", {"step": 220, "validation_loss": 0.2})
    save_adapter(adapter, "checkpoint-100", 4)
    save_adapter(adapter, "checkpoint-220", 4)
    rows = [{"step": 100, "train_loss": 0.4, "validation_loss": 0.3, "seconds": 1.0},
            {"step": 220, "train_loss": 0.3, "validation_loss": 0.2, "seconds": 1.0}]
    (adapter / "training_curve.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = summarize([("baseline", adapter)], 100, tmp_path / "report")
    item = result["variants"][0]
    assert item["matched"]["step"] == 100
    assert item["validation_best"]["step"] == 220
    assert item["matched"]["weights_sha256"] != item["validation_best"]["weights_sha256"]
    assert item["matched"]["validation_loss"] == 0.3
    assert item["matched"]["saved_adapter_parameters"] == 8
    assert item["matched"]["weights_bytes"] > 0
    assert item["target_modules"] == ["q_proj", "v_proj"]
    assert (tmp_path / "report/adapter_summary.csv").is_file()
    assert (tmp_path / "report/training_curves.csv").is_file()


def test_stopped_run_does_not_invent_a_matched_checkpoint(tmp_path):
    adapter = tmp_path / "adapter"
    save_json(adapter / "configuration.json", {"args": {"rank": 8}})
    save_json(adapter / "status.json", {"status": "stopped", "completed_steps": 55})
    save_json(adapter / "best.json", {"step": 20, "validation_loss": 0.5})
    save_adapter(adapter, "checkpoint-20", 8)
    (adapter / "training_curve.jsonl").write_text('{"step":20,"validation_loss":0.5}\n')
    item = summarize([("partial", adapter)], 100, tmp_path / "report")["variants"][0]
    assert item["matched"]["path"] is None
    assert item["validation_best"]["step"] == 20
    assert item["completed_steps"] == 55


def test_invalid_curve_and_mismatched_best_are_rejected(tmp_path):
    adapter = tmp_path / "adapter"
    save_json(adapter / "configuration.json", {"args": {"rank": 4}})
    save_json(adapter / "best.json", {"step": 20, "validation_loss": 0.1})
    save_adapter(adapter, "checkpoint-20", 4)
    curve = adapter / "training_curve.jsonl"
    curve.write_text('{"step":20,"validation_loss":0.5}\n')
    with pytest.raises(ValueError, match="validation curve"):
        summarize([("broken", adapter)], 100, tmp_path / "report")
    curve.write_text('{"step":20}\n{"step":20}\n')
    with pytest.raises(ValueError, match="increase strictly"):
        summarize([("broken", adapter)], 100, tmp_path / "report")
