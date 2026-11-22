"""Export and reload immutable calibrated generator examples without API calls."""
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from carl.icl.boards import from_rows, state_hash, validate_subgoal
from carl.icl.checkpoints.types import CalibratedExample, GeneratorCheckpoint, GeneratorDefaults, SourceArtifact
from carl.icl.config import ROOT, ExperimentConfig, ModelSettings
from carl.icl.data_sources.sokoban import environment
from carl.icl.types import JSONObject, JSONValue


def _json_bytes(value: JSONValue) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def checkpoint_digest(checkpoint: GeneratorCheckpoint) -> str:
    """Bind all settings, ordered examples and provenance except the digest itself."""
    return _digest(_json_bytes(checkpoint.model_dump(mode="json", exclude={"sha256"})))


def _source_path(path: str) -> Path:
    result = Path(path).expanduser()
    return result if result.is_absolute() else ROOT / result


def _verify_sources(sources: tuple[SourceArtifact, ...]) -> None:
    for source in sources:
        with _source_path(source.path).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != source.sha256:
            raise ValueError(f"Checkpoint source hash mismatch: {source.path}")


def _write_immutable(path: Path, content: bytes) -> None:
    """Publish complete bytes atomically and never replace a different artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite immutable checkpoint artifact: {path}") from None
    finally:
        temporary.unlink()


def _examples(calibration: JSONObject, action_cap: int, denominator: float) -> tuple[CalibratedExample, ...]:
    from carl.icl.calibration import weighted_median, weighted_percentiles

    result = tuple(
        CalibratedExample.model_validate({key: row[key] for key in CalibratedExample.model_fields.keys()})
        for row in calibration["examples"]
    )
    contexts: dict[str, list[CalibratedExample]] = defaultdict(list)
    identities: set[tuple[str, int]] = set()
    targets: set[tuple[str, str]] = set()
    for example in result:
        identity = (example.context, example.target_index)
        source, target = from_rows(list(example.state)), from_rows(list(example.subgoal))
        validate_subgoal(source, target)
        target_key = (example.context, state_hash(target))
        if identity in identities or target_key in targets:
            raise ValueError("Calibration repeats a target identity or target board within a context")
        if np.array_equal(source, target):
            raise ValueError("Checkpoint target copies its source board")
        if example.d > action_cap:
            raise ValueError("Checkpoint first-hit mean exceeds the calibrated action cap")
        if contexts[example.context] and contexts[example.context][0].state != example.state:
            raise ValueError("Calibration context has inconsistent source boards")
        env, current = environment(source), source
        for action in example.actions:
            if env.is_solved(current):
                raise ValueError("Checkpoint action witness continues beyond a solved task")
            current = np.asarray(env.next_state(current, action))
        if not np.array_equal(current, target):
            raise ValueError("Checkpoint action witness does not reach its target")
        identities.add(identity)
        targets.add(target_key)
        contexts[example.context].append(example)
    for examples in contexts.values():
        times, hits = [row.d for row in examples], [row.hits for row in examples]
        weights, percentiles = weighted_percentiles(times, hits)
        median = weighted_median(times, [float(hit) for hit in hits])
        for example, weight, percentile in zip(examples, weights, percentiles, strict=True):
            checks = (
                (example.weight, float(weight)), (example.u, float(percentile)),
                (example.a_c, median), (example.ratio_d_over_ac, example.d / median),
                (example.normalized_offset, example.k / denominator),
            )
            if any(not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12) for actual, expected in checks):
                raise ValueError("Checkpoint labels disagree with calibrated hit counts and times")
    return result


def _validate_calibration(
    calibration: JSONObject, *, policy_fingerprint: str, action_cap: int, provider: str,
) -> None:
    from carl.icl.calibration import calibration_result_digest

    if calibration.get("status") != "completed":
        raise ValueError("Checkpoint requires completed calibration")
    if calibration.get("policy_fingerprint") != policy_fingerprint:
        raise ValueError("Checkpoint calibration policy fingerprint mismatch")
    if calibration.get("provider") != provider:
        raise ValueError("Checkpoint calibration provider mismatch")
    if calibration.get("config", {}).get("action_cap") != action_cap:
        raise ValueError("Checkpoint calibration action cap mismatch")
    if calibration.get("result_digest") != calibration_result_digest(calibration):
        raise ValueError("Checkpoint calibration result digest mismatch")
    trials = calibration.get("config", {}).get("trials")
    if not isinstance(trials, int) or trials < 1 or any(row.get("trials") != trials for row in calibration["examples"]):
        raise ValueError("Checkpoint examples require every configured calibration trial")


def export_generator_checkpoint(
    path: str | Path, *, config: ExperimentConfig, policy_fingerprint: str,
    calibration: JSONObject, source_hashes: dict[str, str],
    example_ids: Sequence[tuple[str, int]] | None = None,
) -> GeneratorCheckpoint:
    """Snapshot real calibration records and their original ordered demonstration IDs."""
    path = Path(path)
    provider = calibration.get("provider")
    if provider not in {"mock", "openai"}:
        raise ValueError("Checkpoint calibration provider must be mock or openai")
    _validate_calibration(calibration, policy_fingerprint=policy_fingerprint,
                          action_cap=config.calibration.action_cap, provider=provider)
    records = _examples(calibration, config.calibration.action_cap, config.offset_denominator)
    indexed = {(row.context, row.target_index): row for row in records}
    if example_ids is None:
        selected = calibration.get("selected_example_ids")
        example_ids = ([(row["context"], row["target_index"]) for row in selected]
                       if selected is not None else list(indexed))
    if any(identity not in indexed for identity in example_ids):
        raise ValueError("Checkpoint selected an example absent from calibration evidence")
    sources = tuple(SourceArtifact(path=key, sha256=value) for key, value in sorted(source_hashes.items()))
    _verify_sources(sources)
    evidence = _json_bytes(calibration)
    evidence_hash = _digest(evidence)
    evidence_name = f"calibration-{evidence_hash}.json"
    generator = config.role("generator")
    transport = config.explicit_role("generator")
    checkpoint = GeneratorCheckpoint(
        version="carl-generator-checkpoint-v2" if transport is not None else "carl-generator-checkpoint-v1",
        model=generator.model, generator_transport=transport, provider=provider, policy_fingerprint=policy_fingerprint,
        policy_few_shot=config.few_shot, action_cap=config.calibration.action_cap,
        calibration_signature=calibration["signature"], calibration_result_digest=calibration["result_digest"],
        calibration_evidence=SourceArtifact(path=evidence_name, sha256=evidence_hash), source_hashes=sources,
        defaults=GeneratorDefaults(
            prompts=config.prompts, reasoning_effort=generator.reasoning_effort,
            response_format=generator.response_format,
            layout_schema=config.generator_layout_schema, max_output_tokens=config.generator_tokens,
            max_attempts=config.generator_max_attempts, num_subgoals=config.num_subgoals,
            offset_denominator=config.offset_denominator,
        ),
        actual_example_count=len(example_ids), examples=tuple(indexed[identity] for identity in example_ids), sha256="0" * 64,
    )
    checkpoint = checkpoint.model_copy(update={"sha256": checkpoint_digest(checkpoint)})
    _write_immutable(path.parent / evidence_name, evidence)
    _write_immutable(path, _json_bytes(checkpoint.model_dump(mode="json")))
    return checkpoint


def load_generator_checkpoint(
    path: str | Path, *, model: str | None = None, policy_fingerprint: str | None = None,
    action_cap: int | None = None, provider: str | None = None, verify_sources: bool = False,
    generator_transport: ModelSettings | None = None,
) -> GeneratorCheckpoint:
    """Validate the complete snapshot; optionally verify external source files too."""
    path = Path(path)
    checkpoint = GeneratorCheckpoint.model_validate_json(path.read_bytes())
    if checkpoint.sha256 != checkpoint_digest(checkpoint):
        raise ValueError("Generator checkpoint content digest mismatch")
    expected = {"model": model, "policy_fingerprint": policy_fingerprint, "action_cap": action_cap,
                "provider": "openai" if provider == "replay" else provider}
    for field, value in expected.items():
        if value is not None and getattr(checkpoint, field) != value:
            raise ValueError(f"Generator checkpoint {field} mismatch")
    if (model is not None or generator_transport is not None) and checkpoint.generator_transport != generator_transport:
        raise ValueError("Generator checkpoint transport mismatch")
    evidence_path = path.parent / checkpoint.calibration_evidence.path
    if evidence_path.resolve().parent != path.parent.resolve():
        raise ValueError("Checkpoint calibration evidence must be adjacent to the checkpoint")
    evidence = evidence_path.read_bytes()
    if _digest(evidence) != checkpoint.calibration_evidence.sha256:
        raise ValueError("Checkpoint calibration evidence hash mismatch")
    calibration: JSONObject = json.loads(evidence)
    _validate_calibration(calibration, policy_fingerprint=checkpoint.policy_fingerprint,
                          action_cap=checkpoint.action_cap, provider=checkpoint.provider)
    if (calibration.get("signature") != checkpoint.calibration_signature
            or calibration.get("result_digest") != checkpoint.calibration_result_digest):
        raise ValueError("Checkpoint calibration provenance mismatch")
    records = _examples(calibration, checkpoint.action_cap, checkpoint.defaults.offset_denominator)
    indexed = {(row.context, row.target_index): row for row in records}
    if any(indexed.get((row.context, row.target_index)) != row for row in checkpoint.examples):
        raise ValueError("Checkpoint example differs from calibration evidence")
    if verify_sources:
        _verify_sources(checkpoint.source_hashes)
    return checkpoint
