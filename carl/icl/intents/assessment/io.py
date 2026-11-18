"""Input validation, persistence and board-level statistics for intent evaluation."""

import hashlib
import json
import os
import random
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, cast


class Method(StrEnum):
    BASE = "base"
    ADAPTED = "adapted"
    IN_CONTEXT = "in_context"
    LOCAL_RULE = "local_rule"


class AssessmentStatus(StrEnum):
    EVALUATED = "evaluated"
    MISSING = "missing_infrastructure_output"


Record = dict[str, Any]


def test_contexts(records: list[Record], ids_path: Path | None = None) -> list[Record]:
    test = [row for row in records if row["split"] == "test"]
    if ids_path is None:
        return test
    ids = json.loads(ids_path.read_text(encoding="utf-8"))["record_ids"]
    by_id = {row["record_id"]: row for row in test}
    if (
        len(by_id) != len(test)
        or not isinstance(ids, list)
        or len(ids) != len(set(ids))
        or any(identifier not in by_id for identifier in ids)
    ):
        raise ValueError("Frozen context IDs must be distinct test records")
    return [by_id[identifier] for identifier in ids]


def read_jsonl(path: str | Path) -> list[Record]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reject_non_json(value: str) -> None:
    raise ValueError(f"non-JSON constant: {value}")


def check_raw_prediction(row: Record) -> None:
    """Make JSON validity a property of the saved raw text, never repair it."""
    if not isinstance(row.get("raw_output"), str) or not isinstance(
        row.get("json_valid"), bool
    ):
        raise ValueError("Prediction is missing its raw output or JSON validity flag")
    try:
        parsed = json.loads(row["raw_output"], parse_constant=_reject_non_json)
    except ValueError:
        if row["json_valid"] or row.get("parsed") is not None:
            raise ValueError(
                "Invalid raw JSON disagrees with saved prediction"
            ) from None
    else:
        if not row["json_valid"] or row.get("parsed") != parsed:
            raise ValueError("Parsed JSON disagrees with saved raw output")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions_for(
    path: Path | None,
    method: Method,
    contexts: list[Record],
    data_path: Path,
    ids_path: Path | None,
) -> tuple[dict[str, Record], Record | None]:
    """Reject mismatched, duplicate or extra predictions before pairing."""
    if not path or not path.exists():
        return {}, None
    expected = {row["record_id"]: row["layout_id"] for row in contexts}
    predictions: dict[str, Record] = {}
    for row in read_jsonl(path):
        record_id = cast(str, row.get("record_id"))
        if (
            record_id in predictions
            or record_id not in expected
            or row.get("layout_id") != expected[record_id]
            or row.get("split") != "test"
            or row.get("method") != method
        ):
            raise ValueError(f"Prediction does not match frozen contexts: {record_id}")
        check_raw_prediction(row)
        predictions[record_id] = row
    summary_path = path.with_suffix(".summary.json")
    if not summary_path.exists():
        return predictions, None
    summary = json.loads(summary_path.read_text())
    data_hash = file_hash(data_path)
    ids_hash = file_hash(ids_path) if ids_path else None
    if (
        summary.get("status") != "complete"
        or summary.get("n") != len(predictions)
        or summary.get("data_sha256") != data_hash
        or summary.get("test_ids_sha256") != ids_hash
    ):
        raise ValueError(
            f"Prediction summary does not match data or frozen IDs: {summary_path}"
        )
    if (summary.get("adapter") is not None) != (method == Method.ADAPTED):
        raise ValueError(f"Adapter metadata does not match {method}: {summary_path}")
    return predictions, summary


def compare_metadata(metadata: dict[Method, Record | None]) -> None:
    available = [
        (method, metadata.get(method))
        for method in (Method.BASE, Method.ADAPTED, Method.IN_CONTEXT)
    ]
    available = [(method, value) for method, value in available if value is not None]
    if len(available) < 2:
        return
    reference_method, reference = available[0]
    assert reference is not None
    for method, candidate in available[1:]:
        assert candidate is not None
        for key in (
            "model",
            "revision",
            "seed",
            "decoding",
            "enable_thinking",
            "max_new_tokens",
        ):
            if key not in reference or reference[key] != candidate.get(key):
                raise ValueError(
                    f"Inference settings disagree for {reference_method} and {method}: {key}"
                )
    for key in ("dtype", "prompt_sha256"):
        present = [
            (method, value[key])
            for method, value in available
            if value is not None and key in value
        ]
        if present and any(value != present[0][1] for _, value in present[1:]):
            raise ValueError(
                f"Inference settings disagree for {key}: {[method for method, _ in present]}"
            )


def missing_comparison_metadata(
    metadata: dict[Method, Record | None]
) -> dict[Method, list[str]]:
    """Record historical settings that were not captured rather than guessing."""
    fields = ("dtype", "prompt_sha256")
    return {
        method: [field for field in fields if value is None or field not in value]
        for method, value in metadata.items()
    }


def bootstrap(
    values: list[float], rng: random.Random, count: int = 2000
) -> list[float] | None:
    if len(values) < 5:
        return None
    n = len(values)
    draws = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(count)
    )
    return [draws[int(count * 0.025)], draws[min(count - 1, int(count * 0.975))]]


def board_means(rows: list[Record], value: Any) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["layout_id"], []).append(value(row))
    return [sum(values) / len(values) for _, values in sorted(grouped.items())]


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def recover_progress(
    path: Path,
    config: Record,
    contexts: list[Record],
    resume: bool,
    config_name: str = "assessment_config.json",
) -> list[Record]:
    """Keep each completed measurement and validate its provenance on resume."""
    config_path = path.with_name(config_name)
    if path.exists() and path.stat().st_size and not resume:
        raise ValueError(
            f"{path} already contains results; use --resume or a new output directory"
        )
    if (
        config_path.exists()
        and json.loads(config_path.read_text(encoding="utf-8")) != config
    ):
        raise ValueError(
            "Assessment configuration or input hashes changed; use a new output directory"
        )
    if path.exists() and path.stat().st_size and not config_path.exists():
        raise ValueError("Cannot resume without the original assessment configuration")
    atomic_write(config_path, json.dumps(config, indent=2) + "\n")
    if not path.exists():
        path.touch()
        return []
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        # Preserve the exact interrupted file before repairing a partial line.
        backup = path.with_name(f"{path.name}.interrupted-{time.time_ns()}")
        backup.write_bytes(content)
        previous, separator, tail = content.rpartition(b"\n")
        try:
            json.loads(tail)
        except (ValueError, UnicodeDecodeError):
            content = previous + separator
        else:
            content += b"\n"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    expected = {record["record_id"]: record["layout_id"] for record in contexts}
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        key = row.get("method"), row.get("record_id")
        if (
            key in seen
            or row.get("method") not in set(Method)
            or row.get("record_id") not in expected
            or row.get("layout_id") != expected[row["record_id"]]
            or row.get("status") not in set(AssessmentStatus)
        ):
            raise ValueError(f"Invalid saved assessment record: {key}")
        seen.add(key)
    return rows
