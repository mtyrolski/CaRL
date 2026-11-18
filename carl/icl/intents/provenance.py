"""Source files needed to identify intent-generation implementations."""

import hashlib
from collections.abc import Iterable
from pathlib import Path

GENERATION_COMPONENTS = (
    "intents/__init__.py",
    "intents/provenance.py",
    "intents/generation/__init__.py",
    "intents/generation/data.py",
    "intents/generation/backend.py",
    "intents/generation/evaluation.py",
    "intents/generation/cli.py",
    "boards.py",
    "intent_augmentation.py",
    "intent_data.py",
)

REPRESENTATION_COMPONENTS = (
    "intents/__init__.py",
    "intents/provenance.py",
    "intents/representation/__init__.py",
    "intents/representation/formats.py",
    "intents/representation/prompts.py",
    "intents/representation/predictions.py",
    "boards.py",
    "intent_data.py",
    "intent_in_context.py",
)

ASSESSMENT_COMPONENTS = (
    "intents/__init__.py",
    "intents/provenance.py",
    "intents/assessment/__init__.py",
    "intents/assessment/search.py",
    "boards.py",
    "intent_data.py",
)


def component_hashes(names: Iterable[str], root: Path | None = None) -> dict[str, str]:
    """Hash named source files relative to the ICL package without changing them."""
    root = root or Path(__file__).resolve().parent.parent
    hashes = {}
    for name in names:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Expected a relative source path: {name}")
        with (root / path).open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return hashes
