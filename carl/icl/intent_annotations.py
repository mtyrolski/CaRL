"""Build checked Sokoban intent labels from offline demonstration fragments."""

import json
import random
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from carl.icl.boards import canonical_layout_id, from_rows
from carl.icl.intent_data import DA_VERSION, EffectType, SCHEMA_VERSION, SokobanPredicate
from carl.icl.intent_domain import (
    DOMAIN_SPEC,
    DOMAIN_VERSION,
    TASK_GOAL,
    CheckStatus,
    DomainSpec,
    Intent,
    Predicate,
    SokobanObservation,
    SokobanSnapshot,
    annotation_digest,
    annotation_identity,
    annotation_json,
    check_intent,
    context_identity,
    decode_intent,
    predicate_truth,
    render_instruction,
    replay_fragment,
)
from carl.icl.intents.generation.data import DataSplit, file_sha256, parse_json
from carl.icl.intents.provenance import component_hashes
from carl.icl.types import JSONValue

DEFAULT_ANNOTATOR = "deterministic-mock-v1"
ActionIndex = Annotated[int, Field(strict=True, ge=0, le=3)]


class _Demonstration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    task_group: str | None = Field(default=None, pattern=r"\S")
    split: DataSplit
    start_observation: tuple[str, ...]
    actions: tuple[ActionIndex, ...] = Field(min_length=1)
    observations: tuple[tuple[str, ...], ...] | None = None


class _Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectories: list[_Demonstration] = Field(min_length=1)


@dataclass(frozen=True)
class AnnotationRequest:
    """The only information an annotator can inspect about a fragment."""

    start_observation: SokobanObservation
    later_observation: SokobanObservation
    domain: DomainSpec = DOMAIN_SPEC


@dataclass(frozen=True)
class SegmentSelection:
    """An evaluator-only selection of a fragment's original endpoints."""

    source_id: str
    start: int
    end: int


class UnlabelledStatus(StrEnum):
    ABSTAINED = "abstained"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class UnlabelledAnnotation:
    """Preserve an annotator abstention or a malformed raw response in the audit."""

    status: UnlabelledStatus
    reason: str
    json_valid: bool = True


AnnotationSketch: TypeAlias = JSONValue | Intent | UnlabelledAnnotation
Annotator: TypeAlias = Callable[[AnnotationRequest], AnnotationSketch]


@dataclass(frozen=True)
class _Trace:
    demonstration: _Demonstration
    layout_id: str
    snapshots: tuple[SokobanSnapshot, ...]


@dataclass(frozen=True)
class _Fragment:
    trace: _Trace
    start: int
    offset: int

    @property
    def observations(self) -> tuple[SokobanObservation, ...]:
        return tuple(
            snapshot.observation
            for snapshot in self.trace.snapshots[self.start : self.start + self.offset + 1]
        )


@dataclass(frozen=True)
class _Decision:
    intent: Intent | None
    status: CheckStatus | UnlabelledStatus
    reasons: tuple[str, ...]
    json_valid: bool
    first_hit: int | None = None


def _positions(rows: tuple[str, ...], symbols: str) -> set[tuple[int, int]]:
    return {
        (row_index, col)
        for row_index, row in enumerate(rows)
        for col, cell in enumerate(row)
        if cell in symbols
    }


def mock_annotator(request: AnnotationRequest) -> Intent:
    """Propose position changes and candidate preserves from the two endpoints."""
    before, after = request.start_observation.rows, request.later_observation.rows
    old_boxes, new_boxes = _positions(before, "$*"), _positions(after, "$*")
    old_player, new_player = _positions(before, "@+"), _positions(after, "@+")
    effects: list[Predicate] = [
        {"predicate": "box_at", "row": row, "col": col}
        for row, col in sorted(new_boxes - old_boxes)
    ]
    if not effects:
        effects.extend(
            {"predicate": "player_at", "row": row, "col": col}
            for row, col in sorted(new_player - old_player)
        )
    preserve: list[Predicate] = [
        {"predicate": "box_at", "row": row, "col": col}
        for row, col in sorted(old_boxes & new_boxes)
    ]
    preserve.extend(
        {"predicate": "player_at", "row": row, "col": col}
        for row, col in sorted(old_player & new_player)
    )
    return {"effects": effects, "preserve": preserve, "instruction": None}


def _load_traces(payload: bytes) -> tuple[list[_Trace], dict[DataSplit, list[str]]]:
    source = _Source.model_validate_json(payload)
    layout_split: dict[str, DataSplit] = {}
    group_split: dict[str, DataSplit] = {}
    source_ids: set[str] = set()
    traces = []
    for demonstration in sorted(source.trajectories, key=lambda item: item.source_id):
        if not demonstration.source_id.strip() or demonstration.source_id in source_ids:
            raise ValueError("source_id must be nonempty and unique")
        source_ids.add(demonstration.source_id)
        group = demonstration.task_group or demonstration.source_id
        if group_split.setdefault(group, demonstration.split) != demonstration.split:
            raise ValueError(f"Task group overlaps splits: {group}")
        try:
            snapshot = SokobanSnapshot(demonstration.start_observation)
            layout_id = canonical_layout_id(from_rows(list(snapshot.rows)))
            snapshots = replay_fragment(snapshot, demonstration.actions)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{demonstration.source_id}: {error}") from error
        if (demonstration.observations is not None
                and demonstration.observations != tuple(snapshot.rows for snapshot in snapshots)):
            raise ValueError(f"{demonstration.source_id}: recorded observations disagree with replay")
        previous = layout_split.setdefault(layout_id, demonstration.split)
        if previous != demonstration.split:
            raise ValueError(
                f"Canonical layout overlap between {previous} and {demonstration.split}: {demonstration.source_id}"
            )
        traces.append(_Trace(demonstration, layout_id, snapshots))
    by_split = {
        split: sorted(layout for layout, assigned in layout_split.items() if assigned == split)
        for split in DataSplit
    }
    return traces, by_split


def _fragments(traces: list[_Trace], max_offset: int, seed: int) -> Iterator[_Fragment]:
    rng = random.Random(seed)
    for offset in range(1, min(max_offset, max(len(trace.snapshots) - 1 for trace in traces)) + 1):
        candidates = [
            _Fragment(trace, start, offset)
            for trace in traces
            for start in range(len(trace.snapshots) - offset)
        ]
        rng.shuffle(candidates)
        yield from candidates


def _selected_fragments(traces: list[_Trace], selections: Sequence[SegmentSelection]) -> Iterator[_Fragment]:
    by_source = {trace.demonstration.source_id: trace for trace in traces}
    for selection in selections:
        trace = by_source.get(selection.source_id)
        if trace is None or not 0 <= selection.start < selection.end < len(trace.snapshots):
            raise ValueError("Selected fragment references an invalid source or bounds")
        yield _Fragment(trace, selection.start, selection.end - selection.start)


def _check(sketch: AnnotationSketch, observations: tuple[SokobanObservation, ...], *, first_hit: bool = False) -> _Decision:
    if isinstance(sketch, UnlabelledAnnotation):
        return _Decision(None, UnlabelledStatus(sketch.status), (sketch.reason,), sketch.json_valid)
    invalid_status = UnlabelledStatus.MALFORMED if first_hit else CheckStatus.REJECTED
    try:
        if isinstance(sketch, str):
            parse_json(sketch)
        else:
            annotation_json(sketch)
    except (TypeError, ValueError) as error:
        return _Decision(None, invalid_status, (f"invalid JSON: {error}",), False)
    try:
        intent = decode_intent(sketch)
    except (TypeError, ValueError) as error:
        return _Decision(None, invalid_status, (f"invalid intent: {error}",), True)
    if first_hit:
        endpoints = check_intent({**intent, "preserve": []}, (observations[0], observations[-1]))
        if endpoints.status != CheckStatus.ACCEPTED:
            return _Decision(intent, endpoints.status, endpoints.errors, True)
        hit = next(i for i, observation in enumerate(observations)
                   if all(predicate_truth(effect, observation) is True for effect in intent["effects"]))
        prefix = check_intent(intent, observations[:hit + 1])
        return _Decision(intent, prefix.status, prefix.errors, True, hit if prefix.status == CheckStatus.ACCEPTED else None)
    check = check_intent(intent, observations)
    return _Decision(intent, check.status, check.errors, True)


def _effect_type(intent: Intent, endpoint: SokobanObservation) -> EffectType:
    effects = intent["effects"]
    if len(effects) > 1:
        if all(
            effect["predicate"] == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST
            or (effect["predicate"] == SokobanPredicate.BOX_AT.value and endpoint.rows[effect["row"]][effect["col"]] == "*")
            for effect in effects
        ):
            return EffectType.GOAL_CONJUNCTION
        return EffectType.BOX_CONJUNCTION
    if effects[0]["predicate"] == SokobanPredicate.BOX_AT:
        return EffectType.BOX
    if effects[0]["predicate"] == SokobanPredicate.BOXES_ON_GOALS_AT_LEAST:
        return EffectType.GOAL_PROGRESS
    return EffectType.WALK_ONLY


def _annotation_rows(
    fragments: Iterator[_Fragment], annotator: Annotator, *, first_hit: bool = False,
) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for fragment in fragments:
        observations = fragment.observations
        start, end = observations[0], observations[-1]
        result = _check(annotator(AnnotationRequest(start, end)), observations, first_hit=first_hit)
        offset = fragment.offset if result.first_hit is None else result.first_hit
        demonstration = fragment.trace.demonstration
        context_id = context_identity(start.rows)
        row: dict[str, Any] = {
            "schema_version": DA_VERSION,
            "intent_version": SCHEMA_VERSION,
            "context_id": context_id,
            "layout_id": fragment.trace.layout_id,
            "split": demonstration.split,
            "start_observation": start.rows,
            "task_goal": TASK_GOAL,
            "source_id": demonstration.source_id,
            "demonstration_offset": offset,
            "validation": {
                "accepted": result.status == CheckStatus.ACCEPTED,
                "status": result.status.value,
                "reasons": list(result.reasons),
                "json_valid": result.json_valid,
            },
            "evidence": {
                "fragment_id": annotation_digest(
                    annotation_json((demonstration.source_id, fragment.start, fragment.offset)).encode(),
                ),
                "start_index": fragment.start,
                "actions": demonstration.actions[fragment.start : fragment.start + offset],
                "observations": [observation.rows for observation in observations[:offset + 1]],
            },
        }
        if first_hit:
            row["evidence"].update(
                original_end_index=fragment.start + fragment.offset, original_end_observation=end.rows,
                first_verified_hit_offset=result.first_hit,
            )
        if demonstration.task_group is not None:
            row["task_group"] = demonstration.task_group
        if result.intent is not None:
            row["intent"] = result.intent
        if result.status == CheckStatus.ACCEPTED and result.intent is not None:
            intent = result.intent
            record_id = annotation_identity(context_id, intent)
            row["record_id"] = record_id
            if record_id in seen:
                row["validation"].update(
                    accepted=False, status="duplicate", reasons=["duplicate semantic intent for the same context"],
                    duplicate_of=record_id,
                )
            else:
                seen.add(record_id)
                row["effect_type"] = _effect_type(intent, observations[offset])
                row["instruction"] = render_instruction(intent)
        yield row


def prepare_annotations(
    source: Path,
    output: Path,
    *,
    max_offset: int,
    seed: int,
    annotator: Annotator = mock_annotator,
    annotator_id: str = DEFAULT_ANNOTATOR,
    selections: Sequence[SegmentSelection] | None = None,
) -> dict[str, Any]:
    """Replay demonstrations, check pair-only drafts, and publish a complete dataset."""
    if not isinstance(max_offset, int) or isinstance(max_offset, bool) or max_offset < 1:
        raise ValueError("max_offset must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or not isinstance(annotator_id, str) or not annotator_id.strip():
        raise ValueError("an integer seed and nonempty annotator ID are required")
    if output.exists():
        raise FileExistsError(f"Annotation output already exists: {output}")
    payload = source.read_bytes()
    traces, layouts = _load_traces(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with TemporaryDirectory(dir=output.parent, prefix=f".{output.name}.") as directory:
        stage = Path(directory)
        data_file, audit_file = stage / "D_A.jsonl", stage / "annotation_audit.jsonl"
        with data_file.open("w", encoding="utf-8") as accepted, audit_file.open("w", encoding="utf-8") as audit:
            fragments = _fragments(traces, max_offset, seed) if selections is None else _selected_fragments(traces, selections)
            for row in _annotation_rows(fragments, annotator, first_hit=selections is not None):
                status = row["validation"]["status"]
                counts[status] += 1
                audit.write(annotation_json(row) + "\n")
                if status == "accepted":
                    accepted.write(annotation_json({key: value for key, value in row.items() if key != "evidence"}) + "\n")
        manifest = {
            "schema_version": DA_VERSION,
            "intent_version": SCHEMA_VERSION,
            "domain_version": DOMAIN_VERSION,
            "annotator": annotator_id,
            "seed": seed,
            "max_offset": max_offset,
            "source": str(source.resolve()),
            "source_sha256": annotation_digest(payload),
            "source_ids": [trace.demonstration.source_id for trace in traces],
            "layout_ids_by_split": layouts,
            "attempted": counts.total(),
            "accepted": counts["accepted"],
            "rejected": counts["rejected"],
            "unknown": counts["unknown"],
            "duplicates": counts["duplicate"],
            "code_sha256": component_hashes((
                "boards.py", "data_sources/sokoban.py", "intent_data.py",
                "intent_domain.py", "intent_annotations.py", "intents/generation/data.py",
                "intents/provenance.py", "intents/representation/formats.py",
            )),
            "files_sha256": {path.name: file_sha256(path) for path in (data_file, audit_file)},
        }
        if selections is not None:
            manifest["abstained"] = counts["abstained"]
            manifest["malformed"] = counts["malformed"]
        manifest_file = stage / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output.mkdir()
        for path in (data_file, audit_file, manifest_file):
            path.replace(output / path.name)
    return manifest
