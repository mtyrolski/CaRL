"""Build leakage-controlled generator prompts and weighted training samples."""

import random
from collections.abc import Iterator
from typing import Any

from carl.icl.intent_generator_experiment import SYSTEM
from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import target_text
from carl.icl.policy_relative.checks import _number
from carl.icl.policy_relative.types import MISSING_SCALE
from carl.icl.policy_relative.types import SCALE_INSTRUCTION
from carl.icl.policy_relative.types import Arm
from carl.icl.policy_relative.types import Candidate
from carl.icl.policy_relative.types import Context
from carl.icl.policy_relative.types import FitData


def candidate_scale(candidate: Candidate, arm: Arm) -> float | None:
    if not candidate.measured:
        return None
    if arm in (Arm.SCALED, Arm.RELATIVE):
        return candidate.u
    if arm == Arm.ABSOLUTE:
        return candidate.mean_success_actions
    return None


def generator_messages(
    record: dict[str, Any], arm: Arm | str, requested_scale: float | None = None
) -> list[dict[str, str]]:
    """Whitelist board, task and explicit scale; never serialise the source record."""
    arm = Arm(arm)
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
    prefix = tokenizer.apply_chat_template(
        generator_messages(record, arm, requested_scale),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prefix, list):
        prefix = prefix.tolist()
    answer = tokenizer.encode(
        target_text(record) + (tokenizer.eos_token or ""), add_special_tokens=False
    )
    if not answer:
        raise ValueError("the complete intent response must contain tokens")
    ids = prefix + answer
    if len(ids) > max_length:
        raise ValueError(
            f"complete example requires {len(ids)} tokens; limit is {max_length}; no truncation"
        )
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prefix) + answer,
    }


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
    while True:
        yield next(base)
        yield next(extra)


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
                        "messages": generator_messages(candidate.record, arm, scale),
                        "response_target": target_text(candidate.record),
                    }
