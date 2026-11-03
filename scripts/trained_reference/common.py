"""Small deterministic helpers for the historical reference export."""
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def selected_indices(population: int, count: int, seed: int) -> list[int]:
    """Sample without replacement before inspecting any solver outcomes."""
    if not 0 < count <= population:
        raise ValueError('Require 0 < sample_size <= population_size')
    return sorted(np.random.Generator(np.random.PCG64(seed)).choice(population, count, replace=False).tolist())


def save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(solved: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = solved / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return center - half, center + half
