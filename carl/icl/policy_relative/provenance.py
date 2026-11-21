"""Identify the complete implementation used by a generator run."""

import hashlib
import json
from pathlib import Path

from carl.icl.intents.provenance import ASSESSMENT_COMPONENTS
from carl.icl.intents.provenance import GENERATION_COMPONENTS
from carl.icl.intents.provenance import REPRESENTATION_COMPONENTS

SHARED_CODE_FILES = tuple(dict.fromkeys((
    "boards.py",
    "intent_assessment.py",
    "intent_augmentation.py",
    "intent_data.py",
    "intent_generator_experiment.py",
    "intent_in_context.py",
    "intent_representation_experiment.py",
    "intent_representation_training.py",
    "predicate_prompts.py",
    "intents/provenance.py",
    "intents/assessment/io.py",
    "intents/assessment/reporting.py",
    "intents/generation/checkpoint.py",
    "intents/representation/normalization.py",
    *ASSESSMENT_COMPONENTS,
    *GENERATION_COMPONENTS,
    *REPRESENTATION_COMPONENTS,
)))


def code_provenance() -> tuple[str, dict[str, str]]:
    """Hash the entry point, package modules, and shared experiment code."""
    package = Path(__file__).resolve().parent
    paths = [
        package.parent / "policy_relative_lora.py",
        *sorted(package.glob("*.py")),
        *(package.parent / name for name in SHARED_CODE_FILES),
    ]
    hashes = {
        str(path.relative_to(package.parent)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), hashes
