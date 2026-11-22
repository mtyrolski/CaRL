"""Build leakage-controlled generator prompts and weighted training samples."""

import random
from collections.abc import Iterator
from typing import Any, cast

from carl.icl.intents.generation.data import (
    SYSTEM, DataSplit, encode_messages, predicate_record, prompt_messages, target_text,
)
from carl.icl.policy_relative.checks import _number
from carl.icl.policy_relative.types import MISSING_SCALE
from carl.icl.policy_relative.types import SCALE_INSTRUCTION
from carl.icl.policy_relative.types import Arm
from carl.icl.policy_relative.types import Candidate
from carl.icl.policy_relative.types import Context
from carl.icl.policy_relative.types import FitData
from carl.icl.policy_relative.types import VERSIONED_SCALE_ARMS
from carl.icl.predicate_prompts import generator_messages as predicate_messages


def prompt_scale_arm(arm: Arm) -> str:
    """Translate historical names to the shared predicate scale vocabulary."""
    if arm in VERSIONED_SCALE_ARMS:
        return arm.value
    if arm in (Arm.SCALED, Arm.RELATIVE):
        return "relative_u"
    return "absolute_d" if arm == Arm.ABSOLUTE else "unscaled"


def candidate_scale(candidate: Candidate, arm: Arm) -> float | None:
    if not candidate.measured:
        return None
    if arm in VERSIONED_SCALE_ARMS:
        return cast(float | None, candidate.record["policy_measurements"]["scale_labels"][arm.value])
    if arm in (Arm.SCALED, Arm.RELATIVE):
        return candidate.u
    if arm == Arm.ABSOLUTE:
        return candidate.mean_success_actions
    return None


def generator_messages(
    record: dict[str, Any], arm: Arm | str, requested_scale: float | None = None, *, predicate: bool = False,
) -> list[dict[str, str]]:
    """Whitelist board, task and explicit scale; never serialise the source record."""
    arm = Arm(arm)
    if predicate or predicate_record(record):
        return predicate_messages(record["start_observation"], record["task_goal"], requested_scale,
                                  prompt_scale_arm(arm), allow_missing=arm in (Arm.ABSOLUTE, Arm.RELATIVE))
    if arm in (Arm.UNSCALED, Arm.BASE, Arm.DEMONSTRATION, Arm.MEASURED):
        if requested_scale is not None:
            raise ValueError("unscaled prompts do not accept a requested scale")
        value = MISSING_SCALE
    elif requested_scale is None and arm in (Arm.ABSOLUTE, Arm.RELATIVE):
        value = MISSING_SCALE
    else:
        value_number = _number(requested_scale, "requested_scale")
        if arm in (Arm.SCALED, Arm.RELATIVE) and not 0 <= value_number <= 1:
            raise ValueError("requested_scale must be between zero and one")
        if arm == Arm.ABSOLUTE and value_number <= 0:
            raise ValueError("requested duration must be positive")
        value = repr(value_number)
    messages = prompt_messages(record)
    if arm == Arm.ABSOLUTE:
        messages[0]["content"] = (
            SYSTEM
            + " The requested duration is the expected number of primitive controller "
            "actions conditional on success. A missing-scale marker means no duration is requested."
        )
        messages[1]["content"] += f"\nRequested duration in actions: {value}"
    else:
        messages[0]["content"] = SYSTEM + SCALE_INSTRUCTION
        messages[1]["content"] += f"\nRequested relative scale: {value}"
    return messages


def encode_target(
    tokenizer: Any,
    record: dict[str, Any],
    arm: Arm | str,
    requested_scale: float | None,
    max_length: int,
) -> dict[str, list[int]]:
    """Encode an entire prompt and checked target, masking every prompt token."""
    return encode_messages(tokenizer, generator_messages(record, arm, requested_scale), target_text(record), max_length)


def encode_example(
    tokenizer: Any, candidate: Candidate, arm: Arm | str, max_length: int
) -> dict[str, list[int]]:
    arm = Arm(arm)
    return encode_target(
        tokenizer, candidate.record, arm, candidate_scale(candidate, arm), max_length
    )


def balanced_samples(contexts: tuple[Context, ...], seed: int) -> Iterator[Candidate]:
    """Sample contexts equally and candidates within each context by measured w."""
    if not contexts:
        raise ValueError("at least one context is required")
    rng = random.Random(seed)
    while True:
        order = list(contexts)
        rng.shuffle(order)
        for context in order:
            yield rng.choices(
                context.candidates,
                weights=[item.weight for item in context.candidates],
                k=1,
            )[0]


def study_samples(
    data: FitData, demonstration: FitData | None, arm: Arm | str, seed: int
) -> Iterator[Candidate]:
    """Pair half demonstration with half measured examples, or two demo streams for D."""
    arm = Arm(arm)
    if demonstration is None:
        yield from balanced_samples(data.train, seed)
        return
    base = balanced_samples(demonstration.train, seed)
    extra = balanced_samples(
        demonstration.train if arm == Arm.DEMONSTRATION else data.train, seed + 1
    )
    for baseline, alternate in zip(base, extra, strict=True):
        yield baseline
        yield alternate


def loss_samples(contexts: tuple[Context, ...], seed: int, *, explicit_weights: bool) -> Iterator[tuple[Candidate, float]]:
    """Keep one response per draw, with an unbiased multiplier for a uniform candidate draw."""
    if not explicit_weights:
        for candidate in balanced_samples(contexts, seed):
            yield candidate, 1.0
        return
    if not contexts:
        raise ValueError("at least one context is required")
    rng = random.Random(seed)
    while True:
        order = list(contexts)
        rng.shuffle(order)
        for context in order:
            candidate = rng.choice(context.candidates)
            yield candidate, len(context.candidates) * candidate.weight


def weighted_study_samples(
    data: FitData, demonstration: FitData | None, arm: Arm, seed: int,
) -> Iterator[tuple[Candidate, float]]:
    """Use explicit target objective weights for checked calibration and preserve legacy schedules."""
    if "measurement_provenance" not in data.audit:
        for candidate in study_samples(data, demonstration, arm, seed):
            yield candidate, 1.0
        return
    if demonstration is None:
        yield from loss_samples(data.train, seed, explicit_weights=True)
        return
    demo = loss_samples(demonstration.train, seed, explicit_weights=False)
    measured = loss_samples(data.train, seed + 1, explicit_weights=True)
    for baseline, alternate in zip(demo, measured, strict=True):
        yield baseline
        yield alternate


def training_prompts(
    data: FitData, arm: Arm | str, demonstration: FitData | None = None
) -> Iterator[dict[str, Any]]:
    """Save exact visible messages and response targets for reproducible review."""
    arm = Arm(arm)
    pools = [("measured" if demonstration else "legacy", data)]
    if demonstration:
        pools = [("demonstration", demonstration)]
        if arm != Arm.DEMONSTRATION:
            pools.append(("measured", data))
    for source, pool in pools:
        for split, contexts in (
            (DataSplit.TRAIN, pool.train),
            (DataSplit.VALIDATION, pool.validation),
        ):
            for context in contexts:
                for candidate in context.candidates:
                    scale = candidate_scale(candidate, arm)
                    yield {
                        "record_id": candidate.record["record_id"],
                        "context_id": context.context_id,
                        "split": split,
                        "source": source,
                        "weight": candidate.weight,
                        "messages": generator_messages(candidate.record, arm, scale),
                        "response_target": target_text(candidate.record),
                    }
