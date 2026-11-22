"""Resume and validate generated outputs for intent representation experiments."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from carl.icl.intents.generation.data import atomic_json
from carl.icl.intents.generation.data import read_records
from carl.icl.intents.representation.formats import parse_representation


def _parse_output(
    representation: str, output: str
) -> tuple[dict[str, Any] | None, bool, str | None]:
    try:
        return parse_representation(representation, output), True, None
    except ValueError as exc:
        return None, False, str(exc)


def prepare_output(
    output: Path, config: dict[str, Any], records: list[dict[str, Any]], resume: bool
) -> list[dict[str, Any]]:
    """Validate a partial run against exact settings, order, and raw outputs."""
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = output.with_suffix(".run.json")
    if not resume and (output.exists() or metadata.exists()):
        raise FileExistsError("output already exists; use --resume for this exact run")
    if output.exists() and not metadata.exists():
        raise ValueError("cannot resume without a frozen run configuration")
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8")) != config:
            raise ValueError("run configuration changed; refusing to resume")
    else:
        atomic_json(metadata, config)
    if not output.exists():
        return []
    raw = output.read_bytes()
    if raw and not raw.endswith(b"\n"):
        boundary = raw.rfind(b"\n") + 1
        tail = raw[boundary:]
        try:
            json.loads(tail)
        except (json.JSONDecodeError, UnicodeDecodeError):
            saved = output.with_name(
                f"{output.name}.partial-{hashlib.sha256(tail).hexdigest()[:12]}"
            )
            saved.write_bytes(tail)
            raw = raw[:boundary]
        else:
            raw += b"\n"
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_bytes(raw)
        os.replace(temporary, output)
    previous = read_records(output)
    if len(previous) > len(records):
        raise ValueError("output exceeds frozen selection")
    for row, record in zip(previous, records, strict=False):
        if (
            row.get("record_id") != record["record_id"]
            or row.get("layout_id") != record["layout_id"]
            or any(row.get(key) != config[key] for key in ("split", "variant", "representation", "support_set"))
            or not isinstance(row.get("raw_output"), str)
        ):
            raise ValueError(
                "saved output does not match frozen selection or configuration"
            )
        parsed, syntax_valid, _ = _parse_output(
            config["representation"], row["raw_output"]
        )
        if (
            row.get("syntax_valid") is not syntax_valid
            or row.get("parsed_intent") != parsed
        ):
            raise ValueError("saved parse does not match raw output")
    return previous
