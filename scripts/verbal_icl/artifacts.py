"""Atomic, fsynced local experiment artifacts for interruption-safe inspection."""
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


def replace_text(path: Path, text: str) -> None:
    """Persist bytes, rename atomically, then persist the directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, data: Any) -> None:
    """Commit a complete JSON checkpoint."""
    replace_text(path, json.dumps(data, indent=2, allow_nan=False) + '\n')


def append_jsonl(path: Path, data: Any) -> None:
    """Commit a complete journal line; caller holds the experiment's writer lock."""
    previous = path.read_text(encoding='utf-8') if path.exists() else ''
    replace_text(path, previous + json.dumps(data, allow_nan=False) + '\n')
