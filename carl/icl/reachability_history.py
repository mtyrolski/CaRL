"""Read-only inventory of known public Sokoban layout sources and history.

This inventories declared cohorts; a cohort's presence does not prove a model
saw it. Only the history snapshot's separate observed list claims prior exposure.
No private witness, response cache, or model is opened.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id, from_rows

CANONICALIZATION = "carl-layout-d4-v1"
PUBLIC_NAMES = frozenset({
    "training.json", "demonstrations.json", "validation.json", "test.json", "pilot100.json",
    "component_validation.json", "scale_validation.json", "holdout1000.json",
})
MANDATORY = (
    "data/icl/training.json",
    "data/icl/real_12x12x4/validation.json",
    "data/icl/real_12x12x4/test.json",
    "data/icl/component_study_100_clean/training.json",
    "data/icl/component_study_100_clean/real_12x12x4/validation.json",
    "data/icl/component_study_100_clean/real_12x12x4/test.json",
    "data/icl/campaign_dataset_v2/real_12x12x4/pilot100.json",
    "data/icl/campaign_dataset_12x12_v1/real_12x12x4/pilot100.json",
    "configs/icl/checkpoints/launch100/assets/cohort_checks.json",
    "configs/icl/checkpoints/launch100/assets/demonstrations.json",
    "configs/icl/checkpoints/launch100/assets/checkpoint/generator.json",
    "configs/icl/checkpoints/launch100/assets/verbal.json",
    "configs/icl/search_demonstrations.json",
    "analysis/luna_icl/campaign_100/history_snapshot.json",
    "analysis/luna_icl/campaign_100/goal_progress_probe/data_validation_queries.json",
)


def _public_files(root: Path) -> list[Path]:
    """Discover public prepared data without traversing private or hidden trees."""
    directory = root / "data/icl"
    result = []
    if not directory.is_dir():
        return result
    for location, folders, files in os.walk(directory, followlinks=False):
        folders[:] = sorted(name for name in folders if not name.startswith(".")
                            and "private" not in name.lower() and "staging" not in name.lower()
                            and name not in {"cache", "calibration", "private_witnesses"})
        base = Path(location)
        # Standard trees have either root/family/file or root/dataset/family/file.
        if len(base.relative_to(directory).parts) > 2:
            folders[:] = []
            continue
        for name in sorted(files):
            item = base / name
            if name in PUBLIC_NAMES and item.is_file() and not item.is_symlink():
                result.append(item)
    return result


def _id(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA-256 canonical layout ID")
    return value


def _record_rows(rows: Any, *, state_fields: tuple[str, ...]) -> list[str]:
    if not isinstance(rows, list):
        raise ValueError("Expected a list of records")
    ids = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Expected a record object")
        board = next((row[field] for field in state_fields if field in row), None)
        if not isinstance(board, list):
            raise ValueError("A declared public record has no board rows")
        actual = canonical_layout_id(from_rows(board))
        if "layout_id" in row and _id(row["layout_id"]) != actual:
            raise ValueError("Declared canonical layout ID does not match board")
        ids.add(actual)
    return sorted(ids)


def _record_sources(relative: str, payload: Any) -> list[tuple[str, list[str]]]:
    """Extract known schema fields only; unknown schemas fail closed."""
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    if relative.endswith("/history_snapshot.json"):
        if payload.get("canonicalization") != CANONICALIZATION:
            raise ValueError("Unknown history snapshot canonicalization")
        return [(field, sorted({_id(value) for value in payload[field]}))
                for field in ("observed_layout_ids", "reserved_layout_ids")]
    if relative.endswith("/cohort_checks.json"):
        return [(field, sorted({_id(value) for value in payload[field]}))
                for field in ("training_layout_ids", "validation_layout_ids", "pilot_layout_ids")]
    if relative.endswith("/verbal.json"):
        return [("demonstrations", _record_rows(payload["demonstrations"], state_fields=("state", "source"))),
                ("calibration_examples", _record_rows(payload["examples"], state_fields=("source", "state")))]
    if "contexts" in payload:
        return [("contexts", _record_rows(payload["contexts"], state_fields=("state",)))]
    for field in ("instances", "demonstrations", "examples"):
        if field in payload:
            return [(field, _record_rows(payload[field], state_fields=("state", "source")))]
    raise ValueError("No supported public board records in file")


def scan_history(root: Path) -> dict[str, Any]:
    """Return per-source canonical layout sets, input hashes and explicit coverage.

    `kind` describes file provenance, not actual exposure. The two snapshot
    fields remain separate, so `reserved_layout_ids` must not be read as model
    observations. Missing sources and parse errors are reported explicitly.
    """
    root = Path(root).resolve()
    discovered = set(_public_files(root))
    discovered.update(root / name for name in MANDATORY if (root / name).is_file())
    for relative in (
        "configs/icl/checkpoints/launch100/assets/data/training.json",
        "configs/icl/checkpoints/launch100/assets/data/real_12x12x4/pilot100.json",
    ):
        item = root / relative
        if item.is_file():
            discovered.add(item)
    # A Figure 1 manifest contains evaluation inputs and its provider, not a
    # proof of executed/model-seen cases. Never traverse trial/cache records.
    for item in sorted((root / "analysis/luna_icl").glob("figure1*/manifest.json")):
        if item.is_file() and not item.is_symlink():
            discovered.add(item)
    sources: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for item in sorted(discovered, key=lambda p: p.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix()
        try:
            data = item.read_bytes()
            payload = json.loads(data)
            digest = hashlib.sha256(data).hexdigest()
            for field, ids in _record_sources(relative, payload):
                row: dict[str, Any] = {"path": relative, "kind": field, "status": "available", "sha256": digest,
                                       "count": len(ids), "layout_ids": ids}
                if isinstance(payload, dict) and relative.startswith("analysis/luna_icl/figure1"):
                    row["provider"] = payload.get("provider")
                sources[f"{relative}#{field}"] = row
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"path": relative, "error": error})
            sources[f"{relative}#error"] = {"path": relative, "kind": "error", "status": "error",
                                            "sha256": None, "count": 0, "layout_ids": [], "error": error}
    missing = [name for name in MANDATORY if not (root / name).is_file()]
    for name in missing:
        sources[f"{name}#missing"] = {"path": name, "kind": "missing", "status": "missing",
                                      "sha256": None, "count": 0, "layout_ids": []}
    return {
        "version": 1,
        "canonicalization": CANONICALIZATION,
        "sources": sources,
        "missing": missing,
        "errors": errors,
        "limitations": [
            "Private witnesses, response caches and arbitrary historical result payloads are not scanned.",
            "Missing or unpublished history and exposure after the saved snapshot are not established.",
            "Cohort and reservation membership does not prove that a model observed a layout.",
            "Absence from these sources does not prove a layout is an untouched test case.",
        ],
        "scope": {
            "public_data": "Recognized training, demonstrations, validation, test, pilot and holdout files under data/icl; hidden and private trees excluded.",
            "explicit": "launch100 cohort/train/pilot/progress/verbal, search demonstrations, history snapshot, goal-progress validation inputs, Figure 1 manifests.",
            "not_scanned": "Private witnesses, response caches, arbitrary historical result payloads, unpublished or missing history, and prior exposure after the snapshot.",
            "interpretation": "Files and reserved cohorts show selection, not model use. Absence from scanned history does not prove no earlier exposure.",
        },
    }
