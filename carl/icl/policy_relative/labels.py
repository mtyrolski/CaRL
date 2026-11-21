"""Shared identities and scale labels for frozen intent measurement pools."""

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, TypedDict

from carl.icl.calibration import weighted_median, weighted_percentiles

MEASUREMENT_VERSION: Final = "carl-dc-v2"


class ScaleLabels(TypedDict):
    unscaled: None
    absolute_d: float | None
    ratio_d_over_ac: float | None
    offset_rank: float
    normalized_offset: float
    relative_u: float | None
    weight: float | None


def contract_hash(value: object) -> str:
    """Hash canonical JSON, with non-finite values rejected."""
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def pool_identity(rows: Sequence[Mapping[str, Any]], planned_attempts: Mapping[str, int]) -> str:
    """Bind the selected records, split, context, offsets, types and planned counts."""
    fields = ("record_id", "context_id", "layout_id", "split", "demonstration_offset", "effect_type")
    members = []
    for row in sorted(rows, key=lambda item: item["record_id"]):
        members.append({
            **{key: row[key] for key in fields},
            "planned_attempts": planned_attempts[row["record_id"]],
        })
    return contract_hash(members)


def label_candidates(
    durations: Sequence[float | None],
    success_fractions: Sequence[float],
    offsets: Sequence[int],
    *,
    offset_denominator: float,
) -> list[ScaleLabels]:
    """Label a complete pool using success masses and two distinct offset controls."""
    if not durations or len(durations) != len(success_fractions) or len(durations) != len(offsets):
        raise ValueError("a complete pool needs a duration, success fraction and offset for every candidate")
    if isinstance(offset_denominator, bool) or not math.isfinite(offset_denominator) or offset_denominator <= 0:
        raise ValueError("offset denominator must be a positive finite number")
    if any(isinstance(offset, bool) or not isinstance(offset, int) or offset < 1 for offset in offsets):
        raise ValueError("demonstration offsets must be positive integers")

    reached: list[tuple[int, float, float]] = []
    for index, (duration, fraction) in enumerate(zip(durations, success_fractions, strict=True)):
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("success fractions must be between zero and one")
        if duration is None:
            if fraction != 0:
                raise ValueError("duration is required when an intent had a success")
        elif isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0 or fraction == 0:
            raise ValueError("duration must be positive and finite and have a successful attempt")
        else:
            reached.append((index, duration, fraction))

    # The demonstration rank is independent of success and uses the entire frozen pool.
    _, ranks = weighted_percentiles(offsets, [1] * len(offsets))
    labels: list[ScaleLabels] = [
        {
            "unscaled": None,
            "absolute_d": duration,
            "ratio_d_over_ac": None,
            "offset_rank": float(rank),
            "normalized_offset": offset / offset_denominator,
            "relative_u": None,
            "weight": None,
        }
        for duration, offset, rank in zip(durations, offsets, ranks, strict=True)
    ]
    if reached:
        times = [duration for _, duration, _ in reached]
        masses = [fraction for _, _, fraction in reached]
        weights, relative = weighted_percentiles(times, masses)
        median = weighted_median(times, masses)
        for (index, duration, _), weight, percentile in zip(reached, weights, relative, strict=True):
            labels[index].update(
                ratio_d_over_ac=duration / median, relative_u=float(percentile), weight=float(weight),
            )
    return labels
