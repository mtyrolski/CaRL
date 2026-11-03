"""Paths and deterministic tabular output for the campaign export."""
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'analysis/luna_icl/campaign_100'
DATA = ROOT / 'data/icl/campaign_dataset_v2'
FREEZE = CAMPAIGN / 'scale_tuning/real12_v1/freeze.json'
PILOT = CAMPAIGN / 'pilot_small_v1'
CALIBRATION = DATA / 'calibration/08e8317905e28ebc90073b94aad3ffea004ef265f5ad2d09065e44c2137d80ab'
OUTPUT = CAMPAIGN / 'plot_data'


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='raise')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: scalar(row.get(key)) for key in columns})
