#!/usr/bin/env python3
"""Plot recorded LoRA training and validation losses without running a model."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "analysis/intent_representations_20261114T075324Z"
REPRESENTATIONS = ("json", "formal", "natural")
TITLES = {"json": "JSON", "formal": "Formal grammar", "natural": "Controlled English"}
SEED = 1729


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def finite_loss(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def read_training(run: Path, representation: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read observed steps and verify the validation-selected checkpoint."""
    if representation not in REPRESENTATIONS:
        raise ValueError("Unknown intent representation")
    directory = run / f"adapter_repr_{representation}_seed{SEED}"
    curve_path, best_path = directory / "training_curve.jsonl", directory / "best.json"
    config_path, status_path = directory / "configuration.json", directory / "status.json"
    config, status, best = read_json(config_path), read_json(status_path), read_json(best_path)
    if (config.get("representation") != representation or config.get("seed") != SEED
            or config.get("rank") != 4 or status.get("status") != "finished"):
        raise ValueError(f"Incomplete or incompatible training: {representation}")
    rows = [json.loads(line) for line in curve_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_steps = list(range(1, int(status["completed_steps"]) + 1))
    if (not rows or [row.get("step") for row in rows] != expected_steps
            or any(not finite_loss(row.get("train_loss")) for row in rows)
            or any("validation_loss" in row and not finite_loss(row["validation_loss"]) for row in rows)):
        raise ValueError(f"Training curve has invalid or missing observations: {representation}")
    validation = [row for row in rows if "validation_loss" in row]
    if not validation:
        raise ValueError(f"No recorded validation losses: {representation}")
    selected = min(validation, key=lambda row: row["validation_loss"])
    if (best.get("step") != selected["step"] or best.get("validation_loss") != selected["validation_loss"]
            or best.get("adapter") != status.get("best_adapter")):
        raise ValueError(f"Best checkpoint differs from the recorded validation minimum: {representation}")
    selected_dir = directory / f"checkpoint-{selected['step']}"
    selected_weights = selected_dir / "adapter_model.safetensors"
    last_dir = directory / f"checkpoint-last-{status['completed_steps']}"
    last_weights = last_dir / "adapter_model.safetensors"
    if Path(best["adapter"]).name != selected_dir.name or Path(status["last_adapter"]).name != last_dir.name:
        raise ValueError(f"Checkpoint names do not match the recorded steps: {representation}")
    source_files = (curve_path, best_path, config_path, status_path, selected_weights, last_weights)
    summary = {"representation": representation, "seed": SEED, "rank": config["rank"],
               "train_steps": len(rows), "validation_points": len(validation),
               "best_step": selected["step"], "best_validation_loss": selected["validation_loss"],
               "selected_weights_match_last": sha256(selected_weights) == sha256(last_weights),
               "inputs": {str(path.relative_to(run)): sha256(path) for path in source_files}}
    return rows, summary


def plot(rows_by_representation: dict[str, list[dict[str, Any]]], output: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharex=True)
    for axis, representation in zip(axes, REPRESENTATIONS, strict=True):
        rows = rows_by_representation[representation]
        validation = [row for row in rows if "validation_loss" in row]
        selected = min(validation, key=lambda row: row["validation_loss"])
        axis.plot([row["step"] for row in rows], [row["train_loss"] for row in rows],
                  color="#0072B2", linewidth=.9, alpha=.75, label="Training")
        axis.scatter([row["step"] for row in validation], [row["validation_loss"] for row in validation],
                     color="#D55E00", marker="o", s=27, label="Validation", zorder=3)
        axis.scatter([selected["step"]], [selected["validation_loss"]], color="#111111", marker="*", s=85,
                     label="Selected checkpoint", zorder=4)
        axis.set_title(TITLES[representation])
        axis.set_xlabel("Training step")
        axis.set_ylabel("Response-token loss")
        axis.set_xlim(left=0, right=max(row["step"] for row in rows) * 1.05)
        axis.grid(axis="y", alpha=.2, linewidth=.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .09))
    figure.text(.5, .01, "Loss is a diagnostic; held-out intent and controller results are primary.",
                ha="center", va="bottom", fontsize=8)
    figure.tight_layout(rect=(0, .18, 1, 1))
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"lora_training_loss.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_outputs(run: Path, destination: Path) -> Path:
    """Create a fresh, self-contained figure directory with input/output hashes."""
    run = run.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Use a fresh output directory: {destination}")
    rows_by_representation: dict[str, list[dict[str, Any]]] = {}
    summaries = []
    for representation in REPRESENTATIONS:
        rows, summary = read_training(run, representation)
        rows_by_representation[representation] = rows
        summaries.append(summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        csv_path = temporary / "lora_training_loss.csv"
        fields = ("representation", "seed", "step", "train_loss", "validation_loss", "step_seconds",
                  "selected_checkpoint")
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for summary in summaries:
                representation = summary["representation"]
                for row in rows_by_representation[representation]:
                    writer.writerow({"representation": representation, "seed": SEED, "step": row["step"],
                                     "train_loss": row["train_loss"], "validation_loss": row.get("validation_loss"),
                                     "step_seconds": row.get("seconds"),
                                     "selected_checkpoint": row["step"] == summary["best_step"]})
        plot(rows_by_representation, temporary)
        output_names = ("lora_training_loss.csv", "lora_training_loss.png", "lora_training_loss.pdf",
                        "lora_training_loss.svg")
        audit = run / "validation_checkpoint_audit.json"
        provenance = {"schema_version": 1, "seed": SEED, "analysis": "observed_training_diagnostics",
                      "methods": summaries, "script_sha256": sha256(Path(__file__).resolve()),
                      "validation_checkpoint_audit_sha256": sha256(audit) if audit.is_file() else None,
                      "outputs": {name: sha256(temporary / name) for name in output_names},
                      "interpretation": ["Each recorded training step is shown without smoothing.",
                                         "Validation markers show only measured steps; the star marks the recorded minimum.",
                                         "Validation weights layouts equally; training loss is measured on sampled batches.",
                                         "Separate vertical scales and output formats preclude a direct loss comparison.",
                                         "Loss is diagnostic, not the held-out intent or fixed-controller result."]}
        (temporary / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    print(create_outputs(run, args.output or run / "lora_training_curves_seed1729"))


if __name__ == "__main__":
    main()
