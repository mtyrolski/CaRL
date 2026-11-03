"""Estimate batch capacity from measured allocated GPU bytes at two batch sizes.

Input JSON keys: batch_sizes and allocated_bytes (parallel measured arrays).
This is linear extrapolation, not a guarantee or a training launcher.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from carl.icl.api import write_json


def _parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item <= 0 for item in values):
        raise ValueError("Values must be positive integers")
    return values


def _load_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    missing = {"batch_sizes", "allocated_bytes"} - artifact.keys()
    if missing:
        raise ValueError(f"Artifact missing keys: {sorted(missing)}")
    sizes, memory = np.asarray(artifact["batch_sizes"], float), np.asarray(
        artifact["allocated_bytes"], float
    )
    if sizes.ndim != 1 or memory.shape != sizes.shape or len(sizes) < 2 or len(set(sizes)) < 2:
        raise ValueError("Need at least two distinct measured batch sizes")
    if (
        not np.isfinite(sizes).all()
        or not np.isfinite(memory).all()
        or np.any(sizes <= 0)
        or np.any(memory <= 0)
    ):
        raise ValueError("Measurements must be finite and positive")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--available-gib", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = _load_artifact(args.artifact)
    slope, intercept = np.polyfit(artifact["batch_sizes"], artifact["allocated_bytes"], 1)
    if slope <= 0 or not np.isfinite(args.available_gib) or args.available_gib <= 0:
        raise ValueError("Positive memory slope and available memory required")
    estimate = max(0, int((args.available_gib * 2**30 * 0.8 - max(0, intercept)) / slope))
    write_json(
        args.output,
        {
            "estimated_batch_capacity": estimate,
            "safety_fraction": 0.8,
            "bytes_per_example": float(slope),
            "fixed_bytes": float(intercept),
            "measured": artifact,
        },
    )


if __name__ == "__main__":
    main()
