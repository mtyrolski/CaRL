"""Prepare matched LoRA generators from measured policy-relative intents.

This module never trains the controller or calls its API.  A calibration file
contains all candidates in each context, including failures, and one complete
set of measured attempts per candidate.  The supplied ``u`` and ``weight``
are checked against those attempts before successful candidates are used.

Every row requires ``context_id`` and ``policy_measurements`` with
``controller_hash``, ``action_cap``, ``expected_candidates``, ``complete``,
``attempts``, ``success_fraction``, ``mean_success_actions``, ``u`` and
``weight``.  Each attempt has ``status`` and ``primitive_actions_attempted``.
The producer must separately freeze the planned candidate IDs: a declared
candidate count alone cannot prove that the selected pool was not changed.

Generator inputs are restricted to the current observation, task and either
an explicitly requested scale or the missing-scale marker.  Evaluation uses
separate test contexts without reference intents or future observations.
"""

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from collections import Counter
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.intent_data import validate_prediction
from carl.icl.intent_fence_audit import remove_one_outer_fence
from carl.icl.intent_generator_experiment import SYSTEM
from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import atomic_json
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import generate
from carl.icl.intent_generator_experiment import initialise
from carl.icl.intent_generator_experiment import memory
from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_generator_experiment import sync_gpu
from carl.icl.intent_generator_experiment import target_text
from carl.icl.intent_generator_experiment import tensor_example
from carl.icl.intent_representation_training import completion_status
from carl.icl.intent_representation_training import expired
from carl.icl.intent_representation_training import save_adapter

MISSING_SCALE = "<missing-scale>"
SCALE_INSTRUCTION = (" The requested relative scale is a within-context percentile of successful "
                     "controller action durations. A missing-scale marker means no scale is requested.")


class Arm(StrEnum):
    UNSCALED = "unscaled"
    SCALED = "scaled"
    BASE = "B"
    DEMONSTRATION = "D"
    MEASURED = "N"
    ABSOLUTE = "T"
    RELATIVE = "U"


class Mode(StrEnum):
    INSPECT = "inspect"
    RESOURCE_CHECK = "resource-check"
    BENCHMARK = "benchmark"
    TRAIN = "train"
    EVALUATE = "evaluate"


class AttemptStatus(StrEnum):
    ACHIEVED = "achieved"
    ACTION_LIMIT = "action_limit"
    TASK_SOLVED = "task_solved"
    PRESERVATION_VIOLATED = "preservation_violated"
    INVALID_RESPONSE = "invalid_response"


class PredictionStatus(StrEnum):
    VALID = "valid_intent"
    INVALID_JSON = "invalid_json"
    INVALID_INTENT = "invalid_intent"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class Candidate:
    record: dict[str, Any]
    success_fraction: float | None
    mean_success_actions: float | None
    u: float | None
    weight: float
    measured: bool = True


@dataclass(frozen=True)
class Context:
    context_id: str
    layout_id: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class FitData:
    train: tuple[Context, ...]
    validation: tuple[Context, ...]
    controller_hash: str | None
    action_cap: int | None
    attempts_per_candidate: int | None
    audit: dict[str, Any]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _digest_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _equal(actual: Any, expected: float, name: str) -> float:
    value = _number(actual, name)
    if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{name} disagrees with the measured attempts")
    return value


def _identity(record: dict[str, Any]) -> tuple[str, str, str]:
    values = tuple(record.get(name) for name in ("record_id", "context_id", "layout_id"))
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("record_id, context_id and layout_id must be nonempty strings")
    board, goal = record.get("start_observation"), record.get("task_goal")
    if not isinstance(board, list) or not board or not all(isinstance(row, str) for row in board):
        raise ValueError("start_observation must contain board rows")
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("task_goal must be a nonempty string")
    try:
        actual_layout = canonical_layout_id(from_rows(board))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid start board: {error}") from error
    if actual_layout != values[2]:
        raise ValueError("layout_id differs from the canonical wall/goal layout")
    return values    # type: ignore[return-value]


def _intent_key(intent: dict[str, Any]) -> str:
    return json.dumps(
        {key: sorted(json.dumps(item, sort_keys=True) for item in intent[key]) for key in ("effects", "preserve")},
        sort_keys=True)


def _attempts(measurement: dict[str, Any], cap: int) -> tuple[int, list[int]]:
    attempts = measurement.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("attempts must be a nonempty list of completed attempts")
    successful = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise ValueError("each attempt must be an object")
        try:
            status = AttemptStatus(attempt.get("status"))
        except (TypeError, ValueError) as error:
            raise ValueError("incomplete, infrastructure-error or unknown attempt status") from error
        count = attempt.get("primitive_actions_attempted")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= cap:
            raise ValueError("primitive_actions_attempted must be an integer within the cap")
        if status == AttemptStatus.ACHIEVED:
            if count == 0:
                raise ValueError("a non-trivial intent cannot be achieved in zero actions")
            successful.append(count)
        if status == AttemptStatus.ACTION_LIMIT and count != cap:
            raise ValueError("an action-limit attempt must have used the complete cap")
    return len(attempts), successful


def _checked_context(rows: list[dict[str, Any]]) -> tuple[Context | None, tuple[str, int, int]]:
    first = rows[0]
    first_id, context_id, layout_id = _identity(first)
    del first_id
    seen_intents: set[str] = set()
    observed: list[tuple[dict[str, Any], float, float | None]] = []
    signature: tuple[str, int, int] | None = None
    for row in rows:
        _, row_context, row_layout = _identity(row)
        if (row_context != context_id or row_layout != layout_id or row["split"] != first["split"]
                or row["start_observation"] != first["start_observation"] or row["task_goal"] != first["task_goal"]):
            raise ValueError("all candidates in a context must share split, board, layout and task")
        validation = row.get("validation")
        if not isinstance(validation, dict) or validation.get("accepted") is not True:
            raise ValueError("all calibration intents must be checked and accepted")
        intent = row.get("intent")
        errors = validate_prediction(intent, row["start_observation"])
        if errors or not isinstance(intent, dict):
            raise ValueError(f"invalid or already satisfied intent: {errors}")
        key = _intent_key(intent)
        if key in seen_intents:
            raise ValueError("duplicate candidate intent within a context")
        seen_intents.add(key)
        measurement = row.get("policy_measurements")
        if not isinstance(measurement, dict) or measurement.get("complete") is not True:
            raise ValueError("all context measurements must be complete")
        if _positive_int(measurement.get("expected_candidates"), "expected_candidates") != len(rows):
            raise ValueError("context does not contain every expected candidate")
        controller = _digest_string(measurement.get("controller_hash"), "controller_hash")
        cap = _positive_int(measurement.get("action_cap"), "action_cap")
        attempts, successful = _attempts(measurement, cap)
        current = controller, cap, attempts
        if signature is not None and signature != current:
            raise ValueError("controller, action cap and attempt count must match")
        signature = current
        success_fraction = _equal(measurement.get("success_fraction"), len(successful) / attempts, "success_fraction")
        if successful:
            duration = _equal(measurement.get("mean_success_actions"),
                              sum(successful) / len(successful), "mean_success_actions")
        else:
            duration = None
            if any(measurement.get(field) is not None for field in ("mean_success_actions", "u", "weight")):
                raise ValueError("a candidate without successes must not have duration, u or weight")
        observed.append((row, success_fraction, duration))

    assert signature is not None
    reached = [(row, fraction, duration) for row, fraction, duration in observed if duration is not None]
    if not reached:
        return None, signature
    total = sum(fraction for _, fraction, _ in reached)
    candidates = []
    for row, fraction, duration in reached:
        assert duration is not None
        weight = fraction / total
        before = sum(
            other_fraction / total for _, other_fraction, other_duration in reached if other_duration < duration)
        tied = sum(
            other_fraction / total for _, other_fraction, other_duration in reached if other_duration == duration)
        relative_scale = before + tied / 2
        measurement = row["policy_measurements"]
        _equal(measurement.get("weight"), weight, "weight")
        _equal(measurement.get("u"), relative_scale, "u")
        candidates.append(Candidate(row, fraction, duration, relative_scale, weight))
    return Context(context_id, layout_id, tuple(sorted(candidates,
                                                       key=lambda item: item.record["record_id"]))), signature


def load_fit_data(path: str | Path) -> FitData:
    """Validate complete measured pools, splits, and weighted tied midpoints."""
    rows = read_records(path)
    if not rows:
        raise ValueError("calibration data is empty")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids: set[str] = set()
    layouts: dict[str, set[str]] = defaultdict(set)
    visible_contexts: dict[tuple[tuple[str, ...], str], str] = {}
    for row in rows:
        record_id, context_id, layout_id = _identity(row)
        split = row.get("split")
        if not isinstance(split, str) or split not in (DataSplit.TRAIN, DataSplit.VALIDATION):
            raise ValueError("calibration file may contain only train and validation records")
        if record_id in ids:
            raise ValueError("record IDs must be unique")
        ids.add(record_id)
        layouts[split].add(layout_id)
        visible = tuple(row["start_observation"]), row["task_goal"]
        if visible in visible_contexts and visible_contexts[visible] != context_id:
            raise ValueError("one visible context cannot have multiple context IDs")
        visible_contexts[visible] = context_id
        groups[context_id].append(row)
    if layouts[DataSplit.TRAIN] & layouts[DataSplit.VALIDATION]:
        raise ValueError("train and validation canonical layouts overlap")
    split_contexts: dict[str, list[Context]] = defaultdict(list)
    signatures: set[tuple[str, int, int]] = set()
    empty = Counter()
    rejected = []
    for context_id in sorted(groups):
        group = groups[context_id]
        first = group[0]
        if any(row["split"] != first["split"] or row["layout_id"] != first["layout_id"]
               or row["start_observation"] != first["start_observation"] or row["task_goal"] != first["task_goal"]
               for row in group):
            raise ValueError("all candidates in a context must share split, board, layout and task")
        if any(not isinstance(row.get("policy_measurements"), dict)
               or row["policy_measurements"].get("complete") is not True for row in group):
            rejected.append({
                "context_id": context_id,
                "split": first["split"],
                "record_ids": [row["record_id"] for row in group],
                "reason": "incomplete_measurements"
            })
            continue
        expected = {
            _positive_int(row["policy_measurements"].get("expected_candidates"), "expected_candidates") for row in group
        }
        if len(expected) != 1 or len(group) > next(iter(expected)):
            raise ValueError("inconsistent expected candidate count")
        if len(group) < next(iter(expected)):
            rejected.append({
                "context_id": context_id,
                "split": first["split"],
                "record_ids": [row["record_id"] for row in group],
                "reason": "missing_candidates"
            })
            continue
        context, signature = _checked_context(group)
        signatures.add(signature)
        if context is None:
            empty[group[0]["split"]] += 1
        else:
            split_contexts[group[0]["split"]].append(context)
    training, validation = split_contexts[DataSplit.TRAIN], split_contexts[DataSplit.VALIDATION]
    if not training or not validation:
        raise ValueError("train and validation each need at least one complete context with a measured success; "
                         f"rejected context IDs: {[row['context_id'] for row in rejected]}")
    if len(signatures) != 1:
        raise ValueError("controller, action cap and attempt count must match across splits")
    controller, cap, attempts = next(iter(signatures))
    audit = {
        "rows": len(rows),
        "contexts": len(groups),
        "zero_success_contexts": dict(empty),
        "rejected_contexts": rejected,
        "rejected_context_count": len(rejected),
        "layout_ids_by_split": {
            name: sorted(layouts[name]) for name in (DataSplit.TRAIN, DataSplit.VALIDATION)
        },
        "splits": {
            name: {
                "layouts":
                    len(layouts[name]),
                "contexts_with_success":
                    len(split_contexts[name]),
                "labelled_candidates":
                    sum(len(context.candidates)
                        for context in split_contexts[name]),
                "contexts_with_multiple_successes":
                    sum(len(context.candidates) > 1
                        for context in split_contexts[name]),
                "contexts_with_distinct_success_durations":
                    sum(
                        len({candidate.mean_success_actions
                             for candidate in context.candidates}) > 1
                        for context in split_contexts[name]),
                "distinct_scales":
                    sorted({candidate.u
                            for context in split_contexts[name]
                            for candidate in context.candidates if candidate.u is not None})
            } for name in (DataSplit.TRAIN, DataSplit.VALIDATION)
        }
    }
    return FitData(tuple(training), tuple(validation), controller, cap, attempts, audit)


def load_demo_data(path: str | Path) -> FitData:
    """Read every checked demonstration, including unsuccessful measured candidates."""
    rows = read_records(path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[str] = set()
    visible_contexts: dict[tuple[tuple[str, ...], str], str] = {}
    layouts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        identifier, context, layout = _identity(row)
        split = row.get("split")
        if not isinstance(split, str) or split not in (DataSplit.TRAIN, DataSplit.VALIDATION):
            raise ValueError("demonstrations may contain only train and validation records")
        if identifier in identities:
            raise ValueError("demonstration record IDs must be unique")
        identities.add(identifier)
        if row.get("validation", {}).get("accepted") is not True:
            raise ValueError("demonstration intents must be checked")
        errors = validate_prediction(row.get("intent"), row["start_observation"])
        if errors:
            raise ValueError(f"invalid demonstration intent: {errors}")
        visible = tuple(row["start_observation"]), row["task_goal"]
        if visible in visible_contexts and visible_contexts[visible] != context:
            raise ValueError("one visible demonstration context cannot have multiple context IDs")
        visible_contexts[visible] = context
        groups[context].append(row)
        layouts[split].add(layout)
    if layouts[DataSplit.TRAIN] & layouts[DataSplit.VALIDATION]:
        raise ValueError("demonstration train and validation layouts overlap")
    splits: dict[str, list[Context]] = defaultdict(list)
    for context_id, group in sorted(groups.items()):
        first = group[0]
        signature = (first["split"], first["layout_id"], first["start_observation"], first["task_goal"])
        if any((row["split"], row["layout_id"], row["start_observation"], row["task_goal"]) != signature
               for row in group):
            raise ValueError("demonstration context must share split, board, layout and task")
        if len({_intent_key(row["intent"]) for row in group}) != len(group):
            raise ValueError("duplicate demonstration intent within a context")
        candidates = tuple(Candidate(row, None, None, None, 1 / len(group), False)
                           for row in sorted(group, key=lambda item: item["record_id"]))
        splits[first["split"]].append(Context(context_id, first["layout_id"], candidates))
    if not splits[DataSplit.TRAIN] or not splits[DataSplit.VALIDATION]:
        raise ValueError("demonstrations need train and validation contexts")
    audit = {"rows": len(rows), "contexts": len(groups),
             "layout_ids_by_split": {name: sorted(layouts[name]) for name in (DataSplit.TRAIN, DataSplit.VALIDATION)},
             "splits": {name: {"contexts": len(splits[name]),
                                "candidates": sum(len(context.candidates) for context in splits[name])}
                        for name in (DataSplit.TRAIN, DataSplit.VALIDATION)}}
    return FitData(tuple(splits[DataSplit.TRAIN]), tuple(splits[DataSplit.VALIDATION]), None, None, None, audit)


def load_study_data(args: argparse.Namespace) -> tuple[FitData, FitData | None]:
    """Require the same checked demonstration base for all D/N/T/U arms."""
    arm = Arm(args.arm)
    if arm in (Arm.UNSCALED, Arm.SCALED):
        if getattr(args, "demo_data", None) is not None:
            raise ValueError("legacy arms do not accept a demonstration mix")
        return load_fit_data(args.data), None
    if not getattr(args, "demo_data", None):
        raise ValueError("D/N/T/U require --demo-data")
    demonstration = load_demo_data(args.demo_data)
    if arm == Arm.DEMONSTRATION:
        return demonstration, demonstration
    if not args.data:
        raise ValueError("N/T/U require --data with calibrated measurements")
    demo_rows = {candidate.record["record_id"]: candidate.record
                 for context in (*demonstration.train, *demonstration.validation) for candidate in context.candidates}
    for row in read_records(args.data):
        original = demo_rows.get(row.get("record_id"))
        if original is None or any(row.get(key) != original.get(key) for key in
                                   ("context_id", "layout_id", "split", "start_observation", "task_goal", "intent")):
            raise ValueError("every calibrated candidate must match the common demonstration pool")
    return load_fit_data(args.data), demonstration


def candidate_scale(candidate: Candidate, arm: Arm) -> float | None:
    if not candidate.measured:
        return None
    if arm in (Arm.SCALED, Arm.RELATIVE):
        return candidate.u
    if arm == Arm.ABSOLUTE:
        return candidate.mean_success_actions
    return None


def generator_messages(record: dict[str, Any],
                       arm: Arm | str,
                       requested_scale: float | None = None) -> list[dict[str, str]]:
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
        messages[0]["content"] = (SYSTEM + " The requested duration is the expected number of primitive controller "
                                  "actions conditional on success. A missing-scale marker means no duration is requested.")
        messages[1]["content"] += f"\nRequested duration in actions: {value}"
    else:
        messages[0]["content"] = SYSTEM + SCALE_INSTRUCTION
        messages[1]["content"] += f"\nRequested relative scale: {value}"
    return messages


def encode_target(tokenizer: Any, record: dict[str, Any], arm: Arm | str, requested_scale: float | None,
                  max_length: int) -> dict[str, list[int]]:
    """Encode an entire prompt and checked target, masking every prompt token."""
    prefix = tokenizer.apply_chat_template(generator_messages(record, arm, requested_scale),
                                           tokenize=True,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
    if not isinstance(prefix, list):
        prefix = prefix.tolist()
    answer = tokenizer.encode(target_text(record) + (tokenizer.eos_token or ""), add_special_tokens=False)
    if not answer:
        raise ValueError("the complete intent response must contain tokens")
    ids = prefix + answer
    if len(ids) > max_length:
        raise ValueError(f"complete example requires {len(ids)} tokens; limit is {max_length}; no truncation")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prefix) + answer}


def encode_example(tokenizer: Any, candidate: Candidate, arm: Arm | str, max_length: int) -> dict[str, list[int]]:
    arm = Arm(arm)
    return encode_target(tokenizer, candidate.record, arm, candidate_scale(candidate, arm), max_length)


def balanced_samples(contexts: tuple[Context, ...], seed: int) -> Iterator[Candidate]:
    """Sample contexts equally and candidates within each context by measured w."""
    if not contexts:
        raise ValueError("at least one context is required")
    rng = random.Random(seed)
    while True:
        order = list(contexts)
        rng.shuffle(order)
        for context in order:
            yield rng.choices(context.candidates, weights=[item.weight for item in context.candidates], k=1)[0]


def study_samples(data: FitData, demonstration: FitData | None, arm: Arm | str, seed: int) -> Iterator[Candidate]:
    """Pair half demonstration with half measured examples, or two demo streams for D."""
    arm = Arm(arm)
    if demonstration is None:
        yield from balanced_samples(data.train, seed)
        return
    base = balanced_samples(demonstration.train, seed)
    extra = balanced_samples(demonstration.train if arm == Arm.DEMONSTRATION else data.train, seed + 1)
    while True:
        yield next(base)
        yield next(extra)


def training_prompts(data: FitData, arm: Arm | str, demonstration: FitData | None = None) -> Iterator[dict[str, Any]]:
    """Save exact visible messages and response targets for reproducible review."""
    arm = Arm(arm)
    pools = [("measured" if demonstration else "legacy", data)]
    if demonstration:
        pools = [("demonstration", demonstration)]
        if arm != Arm.DEMONSTRATION:
            pools.append(("measured", data))
    for source, pool in pools:
        for split, contexts in ((DataSplit.TRAIN, pool.train), (DataSplit.VALIDATION, pool.validation)):
            for context in contexts:
                for candidate in context.candidates:
                    scale = candidate_scale(candidate, arm)
                    yield {
                        "record_id": candidate.record["record_id"],
                        "context_id": context.context_id,
                        "split": split,
                        "source": source,
                        "messages": generator_messages(candidate.record, arm, scale),
                        "response_target": target_text(candidate.record)
                    }


def _model_revision(path: Path, revision: str) -> None:
    # A Hugging Face revision is a 40-character commit, not a SHA-256.
    if not path.is_dir() or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("model must be a local directory with an exact 40-character revision")
    metadata = path / "experiment_revision.json"
    if path.name != revision and (not metadata.is_file()
                                  or json.loads(metadata.read_text()).get("revision") != revision):
        raise ValueError("cannot verify the requested local model revision")
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        if not (path / name).is_file():
            raise ValueError(f"local model snapshot is missing {name}")


def _candidate_lengths(tokenizer: Any, contexts: tuple[Context, ...], arm: Arm, limit: int) -> dict[str, int]:
    lengths = sorted(
        len(encode_example(tokenizer, candidate, arm, limit)["input_ids"])
        for context in contexts
        for candidate in context.candidates)
    return {
        "count": len(lengths),
        "minimum": lengths[0],
        "median": lengths[len(lengths) // 2],
        "maximum": lengths[-1],
        "limit": limit,
        "truncated": 0
    }


def _configuration(args: argparse.Namespace, data: FitData, tokenizer: Any, device: str,
                   demonstration: FitData | None = None) -> dict[str, Any]:
    model = Path(args.model).resolve()
    data_path = args.data if Arm(args.arm) != Arm.DEMONSTRATION else args.demo_data
    comparable = {
        "data_sha256": file_sha256(data_path),
        "model_revision": args.revision,
        "model_weights_sha256": file_sha256(model / "model.safetensors"),
        "model_config_sha256": file_sha256(model / "config.json"),
        "tokenizer_sha256": file_sha256(model / "tokenizer.json"),
        "controller_hash": data.controller_hash,
        "action_cap": data.action_cap,
        "attempts_per_candidate": data.attempts_per_candidate,
        "seed": args.seed,
        "dtype": args.dtype,
        "threads": getattr(args, "threads", 8),
        "rank": args.rank,
        "lora_targets": ["q_proj", "v_proj"],
        "accumulation": args.accumulation,
        "learning_rate": args.lr,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "max_length": args.max_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        "min_informative_train_contexts": getattr(args, "min_informative_train_contexts", 4),
        "min_informative_validation_contexts": getattr(args, "min_informative_validation_contexts", 1),
        "train_context_ids": [item.context_id for item in data.train],
        "validation_context_ids": [item.context_id for item in data.validation],
        "train_layout_ids": data.audit["layout_ids_by_split"][DataSplit.TRAIN],
        "validation_layout_ids": data.audit["layout_ids_by_split"][DataSplit.VALIDATION],
        "sampling": "equal contexts, candidates sampled by measured weight",
        "loss": "response and EOS tokens only",
        "code_sha256": file_sha256(__file__)
    }
    if demonstration is not None:
        comparable.update({
            "demonstration_sha256": file_sha256(args.demo_data),
            "demonstration_train_context_ids": [item.context_id for item in demonstration.train],
            "demonstration_validation_context_ids": [item.context_id for item in demonstration.validation],
            "train_layout_ids": sorted(set(comparable["train_layout_ids"])
                                       | set(demonstration.audit["layout_ids_by_split"][DataSplit.TRAIN])),
            "validation_layout_ids": sorted(set(comparable["validation_layout_ids"])
                                            | set(demonstration.audit["layout_ids_by_split"][DataSplit.VALIDATION])),
            "sampling": "alternating demonstration and controller-weighted draws; D replaces measured with demo",
            "demonstration_fraction": .5,
            "validation_sampling": "fixed weighted draws, one per demo context per pool; seeds +2 and +3",
            "validation_draws_per_pool": len(demonstration.validation),
        })
    parity = hashlib.sha256(json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    token_lengths = {
        "train": _candidate_lengths(tokenizer, data.train, Arm(args.arm), args.max_length),
        "validation": _candidate_lengths(tokenizer, data.validation, Arm(args.arm), args.max_length)
    }
    if demonstration is not None:
        token_lengths["demonstration_train"] = _candidate_lengths(tokenizer, demonstration.train, Arm(args.arm),
                                                                  args.max_length)
        token_lengths["demonstration_validation"] = _candidate_lengths(tokenizer, demonstration.validation,
                                                                       Arm(args.arm), args.max_length)
    return {
        **comparable, "comparability_sha256": parity,
        "arm": args.arm,
        "model": str(model),
        "device": device,
        "token_lengths": token_lengths,
        "enable_thinking": False,
        "command": sys.argv,
        "data_audit": data.audit
    }


def _loss(torch: Any, model: Any, tokenizer: Any, candidate: Candidate, args: argparse.Namespace, device: str) -> Any:
    encoded = encode_example(tokenizer, candidate, Arm(args.arm), args.max_length)
    return model(**tensor_example(torch, encoded, device)).loss


def require_scale_information(data: FitData,
                              arm: Arm | str,
                              minimum_train: int = 4,
                              minimum_validation: int = 1) -> None:
    """Do not present a singleton or all-tied calibration as a scale study."""
    if Arm(arm) in (Arm.UNSCALED, Arm.DEMONSTRATION, Arm.MEASURED):
        return
    if minimum_train < 1 or minimum_validation < 1:
        raise ValueError("minimum informative context counts must be positive")
    for split, minimum in ((DataSplit.TRAIN, minimum_train), (DataSplit.VALIDATION, minimum_validation)):
        count = data.audit["splits"][split]["contexts_with_distinct_success_durations"]
        if count < minimum:
            raise ValueError(f"scaled training needs at least {minimum} {split} contexts with two distinct "
                             f"measured successful durations; found {count}")


def validation_loss(torch: Any, model: Any, tokenizer: Any, contexts: tuple[Context, ...], args: argparse.Namespace,
                    device: str, demonstration: tuple[Context, ...] | None = None) -> float:
    """Use equal fixed validation effort across study arms, preserving legacy scoring."""
    model.eval()

    def mean_loss(pool: tuple[Context, ...]) -> float:
        losses = []
        for context in pool:
            total = 0.0
            for candidate in context.candidates:
                value = float(_loss(torch, model, tokenizer, candidate, args, device).detach().cpu())
                if not math.isfinite(value):
                    raise ValueError("validation loss is not finite")
                total += candidate.weight * value
            losses.append(total)
        return sum(losses) / len(losses)

    try:
        with torch.no_grad():
            if demonstration is None:
                value = mean_loss(contexts)
            else:
                base = balanced_samples(demonstration, args.seed + 2)
                extra_pool = demonstration if Arm(args.arm) == Arm.DEMONSTRATION else contexts
                extra = balanced_samples(extra_pool, args.seed + 3)
                losses = [float(_loss(torch, model, tokenizer, candidate, args, device).detach().cpu())
                          for _ in demonstration for candidate in (next(base), next(extra))]
                if not all(math.isfinite(loss) for loss in losses):
                    raise ValueError("validation loss is not finite")
                value = sum(losses) / len(losses)
    finally:
        model.train()
    return value


def resource_records(path: str | Path) -> list[dict[str, Any]]:
    """Select checked, unmeasured fitting candidates without using test records."""
    rows = [row for row in read_records(path) if row.get("split") in (DataSplit.TRAIN, DataSplit.VALIDATION)]
    layouts: dict[str, set[str]] = defaultdict(set)
    identifiers: set[str] = set()
    for row in rows:
        record_id, _, layout = _identity(row)
        if record_id in identifiers:
            raise ValueError("resource-check record IDs must be unique")
        identifiers.add(record_id)
        layouts[row["split"]].add(layout)
        validation = row.get("validation")
        if not isinstance(validation, dict) or validation.get("accepted") is not True:
            raise ValueError("resource-check requires checked candidate intents")
        if row.get("policy_measurements") is not None:
            raise ValueError("resource-check accepts unmeasured candidates only")
        errors = validate_prediction(row.get("intent"), row["start_observation"])
        if errors:
            raise ValueError(f"resource-check candidate intent is invalid: {errors}")
    if not layouts[DataSplit.TRAIN] or not layouts[DataSplit.VALIDATION]:
        raise ValueError("resource-check requires both train and validation candidates")
    if layouts[DataSplit.TRAIN] & layouts[DataSplit.VALIDATION]:
        raise ValueError("resource-check train and validation layouts overlap")
    return rows


def resource_encodings(tokenizer: Any, rows: list[dict[str, Any]], max_length: int,
                       max_new_tokens: int) -> list[tuple[dict[str, Any], dict[str, list[int]], dict[str, Any]]]:
    """Measure complete prompt and target lengths at a diagnostic scale of 0.5."""
    result = []
    for row in rows:
        encoded = encode_target(tokenizer, row, Arm.SCALED, .5, max_length)
        prompt_length = sum(value == -100 for value in encoded["labels"])
        if prompt_length + max_new_tokens > max_length:
            raise ValueError("resource-check prompt plus generation exceeds max_length; no truncation")
        lengths = {
            "record_id": row["record_id"],
            "context_id": row["context_id"],
            "layout_id": row["layout_id"],
            "split": row["split"],
            "prompt_tokens": prompt_length,
            "response_tokens": len(encoded["input_ids"]) - prompt_length,
            "complete_example_tokens": len(encoded["input_ids"]),
            "truncated": False
        }
        result.append((row, encoded, lengths))
    return result


def resource_check(args: argparse.Namespace) -> None:
    """Benchmark a real checked intent with a synthetic scale input, not a label."""
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a fresh resource-check output")
    started = time.monotonic()
    metadata: dict[str, Any] = {
        "measurement_kind": "resource_only_no_luna_labels",
        "synthetic_requested_scale": .5,
        "scale_interpretation": "Diagnostic input only. It is not measured u and must never be used as a label.",
        "data_sha256": file_sha256(args.data),
        "model": str(Path(args.model).resolve()),
        "revision": args.revision,
        "dtype": args.dtype,
        "seed": args.seed,
        "threads": args.threads,
        "rank": args.rank,
        "lora_targets": ["q_proj", "v_proj"],
        "accumulation": args.accumulation,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "gradient_checkpointing": args.gradient_checkpointing,
        "enable_thinking": False,
        "code_sha256": file_sha256(__file__),
        "command": sys.argv,
    }
    atomic_json(output, {**metadata, "status": "initializing"})
    torch = None
    try:
        rows = resource_records(args.data)
        _model_revision(Path(args.model), args.revision)
        if expired(args):
            raise TimeoutError("resource-check deadline reached before loading the model")
        if args.require_cuda:
            import torch as torch_module
            if not torch_module.cuda.is_available():
                raise RuntimeError("CUDA required but unavailable; no silent CPU fallback")
        torch, tokenizer, model, device = initialise(args, train=True)
        prepared = resource_encodings(tokenizer, rows, args.max_length, args.max_new_tokens)
        training = [item for item in prepared if item[0]["split"] == DataSplit.TRAIN]
        # Repeated use of one long example stresses the real microbatch without
        # presenting repeated examples as policy measurements or new data.
        longest = max(training, key=lambda item: len(item[1]["input_ids"]))
        generation_case = max(prepared, key=lambda item: item[2]["prompt_tokens"])
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                      lr=args.lr)
        model.train()
        sync_gpu(torch)
        step_started = time.monotonic()
        tokens = 0
        for _ in range(args.accumulation):
            tensors = tensor_example(torch, longest[1], device)
            tokens += len(longest[1]["input_ids"])
            loss = model(**tensors).loss / args.accumulation
            if not math.isfinite(float(loss.detach().cpu())):
                raise ValueError("resource-check loss is not finite")
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync_gpu(torch)
        seconds = time.monotonic() - step_started
        if expired(args):
            raise TimeoutError("resource-check deadline reached before generation")
        raw, generated, generation_seconds = generate(
            torch,
            model,
            tokenizer,
            generation_case[0],
            args,
            device,
            messages=generator_messages(generation_case[0], Arm.SCALED, .5),
        )
        summary = {
            **metadata, "status": "complete",
            "device": device,
            "model_weights_sha256": file_sha256(Path(args.model) / "model.safetensors"),
            "model_config_sha256": file_sha256(Path(args.model) / "config.json"),
            "tokenizer_sha256": file_sha256(Path(args.model) / "tokenizer.json"),
            "token_lengths": [item[2] for item in prepared],
            "training_record_id": longest[0]["record_id"],
            "generation_record_id": generation_case[0]["record_id"],
            "complete_step_seconds": seconds,
            "training_tokens": tokens,
            "training_tokens_per_second": tokens / seconds,
            "generation_tokens": generated,
            "generation_seconds": generation_seconds,
            "generation_tokens_per_second": generated / generation_seconds if generation_seconds else None,
            "generation_sample": raw,
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "Scale 0.5 is a diagnostic input, not a Luna measurement.",
                "Later measured scales can tokenise differently and require their own length check.",
                "No adapter checkpoint or policy-relative labels are produced."
            ],
            **memory(torch)
        }
        atomic_json(output, summary)
        print(
            json.dumps(
                {
                    key: summary[key] for key in ("status", "measurement_kind", "complete_step_seconds",
                                                 "generation_seconds", "gpu_peak_reserved_bytes")
                },
                indent=2))
    except Exception as error:
        try:
            measured_memory = memory(torch) if torch is not None else {}
        except Exception:
            measured_memory = {}
        atomic_json(
            output, {
                **metadata, "status": "failed",
                "elapsed_seconds": time.monotonic() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                **measured_memory
            })
        raise


def benchmark(args: argparse.Namespace) -> None:
    """Measure a full optimiser step and generation with the actual prompts."""
    data, demonstration = load_study_data(args)
    require_scale_information(data, args.arm, args.min_informative_train_contexts,
                              args.min_informative_validation_contexts)
    _model_revision(Path(args.model), args.revision)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a fresh benchmark output")
    if args.require_cuda:
        import torch as torch_module
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA required but unavailable; no silent CPU fallback")
    started = time.monotonic()
    torch, tokenizer, model, device = initialise(args, train=True)
    configuration = _configuration(args, data, tokenizer, device, demonstration)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.lr)
    examples = study_samples(data, demonstration, args.arm, args.seed)
    tokens = 0
    model.train()
    sync_gpu(torch)
    step_started = time.monotonic()
    for _ in range(args.accumulation):
        example = encode_example(tokenizer, next(examples), Arm(args.arm), args.max_length)
        tensors = tensor_example(torch, example, device)
        tokens += len(example["input_ids"])
        loss = model(**tensors).loss / args.accumulation
        if not math.isfinite(float(loss.detach().cpu())):
            raise ValueError("benchmark loss is not finite")
        loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sync_gpu(torch)
    seconds = time.monotonic() - step_started
    candidate = data.train[0].candidates[0]
    requested = candidate_scale(candidate, Arm(args.arm))
    raw, generated, generation_seconds = generate(
        torch,
        model,
        tokenizer,
        candidate.record,
        args,
        device,
        messages=generator_messages(candidate.record, Arm(args.arm), requested),
    )
    summary = {
        **configuration, "status": "complete",
        "complete_step_seconds": seconds,
        "training_tokens": tokens,
        "training_tokens_per_second": tokens / seconds,
        "generation_tokens": generated,
        "generation_seconds": generation_seconds,
        "generation_tokens_per_second": generated / generation_seconds if generation_seconds else None,
        "generation_sample": raw,
        "elapsed_seconds": time.monotonic() - started,
        **memory(torch)
    }
    atomic_json(output, summary)
    print(
        json.dumps(
            {
                key: summary[key] for key in ("status", "complete_step_seconds", "generation_seconds",
                                             "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")
            },
            indent=2))


def train(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh training directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", {"status": "initializing", "completed_steps": 0})
    completed = 0
    best_path = None
    current_checkpoint = None
    torch = None
    try:
        data, demonstration = load_study_data(args)
        require_scale_information(data, args.arm, args.min_informative_train_contexts,
                                  args.min_informative_validation_contexts)
        _model_revision(Path(args.model), args.revision)
        if expired(args):
            atomic_json(output / "status.json", {"status": "stopped", "reason": "deadline", "completed_steps": 0})
            return
        if args.require_cuda:
            import torch as torch_module
            if not torch_module.cuda.is_available():
                raise RuntimeError("CUDA required but unavailable; no silent CPU fallback")
        torch, tokenizer, model, device = initialise(args, train=True)
        atomic_json(output / "configuration.json", _configuration(args, data, tokenizer, device, demonstration))
        tokenizer.save_pretrained(output / "tokenizer")
        with (output / "prompts.jsonl").open("x", encoding="utf-8") as stream:
            for row in training_prompts(data, args.arm, demonstration):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                      lr=args.lr)
        examples = study_samples(data, demonstration, args.arm, args.seed)
        stopped = {"requested": False}

        def request_stop(_signum: int, _frame: Any) -> None:
            stopped["requested"] = True

        previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        best, last_validation = float("inf"), None
        started = time.monotonic()
        try:
            model.train()
            with (output / "training_curve.jsonl").open("x", encoding="utf-8", buffering=1) as curve:
                for step in range(1, args.max_steps + 1):
                    if stopped["requested"] or expired(args):
                        break
                    step_start = time.monotonic()
                    total_loss = 0.0
                    for _ in range(args.accumulation):
                        loss = _loss(torch, model, tokenizer, next(examples), args, device) / args.accumulation
                        total_loss += float(loss.detach().cpu())
                        if not math.isfinite(total_loss):
                            raise ValueError("training loss is not finite")
                        loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    completed = step
                    row: dict[str, Any] = {
                        "step": step,
                        "train_loss": total_loss,
                        "seconds": time.monotonic() - step_start
                    }
                    if ((step % args.eval_every == 0 or step == args.max_steps) and not stopped["requested"]
                            and not expired(args)):
                        value = validation_loss(torch, model, tokenizer, data.validation, args, device,
                                                demonstration.validation if demonstration else None)
                        row["validation_loss"] = value
                        current_checkpoint = str(save_adapter(model, output, f"checkpoint-{step}"))
                        last_validation = step
                        if value < best:
                            best, best_path = value, current_checkpoint
                            atomic_json(output / "best.json", {
                                "step": step,
                                "validation_loss": value,
                                "adapter": best_path
                            })
                    curve.write(json.dumps(row) + "\n")
                    curve.flush()
                    os.fsync(curve.fileno())
                    atomic_json(
                        output / "status.json", {
                            "status": "running",
                            "completed_steps": completed,
                            "best_adapter": best_path,
                            "latest_checkpoint": current_checkpoint,
                            "last_measurement": row,
                            **memory(torch)
                        })
                    print(json.dumps(row), flush=True)
            last = str(save_adapter(model, output, f"checkpoint-last-{completed}")) if completed else None
            status, reason = completion_status(completed, args.max_steps, last_validation)
            atomic_json(
                output / "status.json", {
                    "status": status,
                    "reason": reason,
                    "completed_steps": completed,
                    "last_validation_step": last_validation,
                    "best_adapter": best_path,
                    "latest_checkpoint": current_checkpoint,
                    "last_adapter": last,
                    "elapsed_seconds": time.monotonic() - started,
                    **memory(torch)
                })
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    except Exception as error:
        atomic_json(
            output / "status.json", {
                "status": "failed",
                "completed_steps": completed,
                "best_adapter": best_path,
                "latest_checkpoint": current_checkpoint,
                "error_type": type(error).__name__,
                "error": str(error)
            })
        raise


def _strict_json(raw: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)


def assess_output(raw: str, board: list[str], *, normalize: bool = False) -> dict[str, Any]:
    """Parse complete JSON, optionally removing one enclosing Markdown fence."""
    try:
        _strict_json(raw)
        strict_valid = True
    except (TypeError, ValueError, json.JSONDecodeError):
        strict_valid = False
    normalized, removed = remove_one_outer_fence(raw) if normalize else (raw, False)
    try:
        intent = _strict_json(normalized)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": PredictionStatus.INVALID_JSON,
            "json_valid": False,
            "strict_json_valid": strict_valid,
            "normalized_json_valid": False,
            "fence_removed": removed,
            "parsed": None,
            "validation_errors": [str(error)]
        }
    errors = validate_prediction(intent, board)
    return {
        "status": PredictionStatus.INVALID_INTENT if errors else PredictionStatus.VALID,
        "json_valid": True,
        "strict_json_valid": strict_valid,
        "normalized_json_valid": True,
        "fence_removed": removed,
        "parsed": intent,
        "validation_errors": errors
    }


def test_contexts(path: str | Path) -> list[dict[str, Any]]:
    """Read one frozen query per context without consulting any target label."""
    rows = read_records(path)
    if not rows:
        raise ValueError("no test contexts")
    identifiers: set[str] = set()
    contexts: set[str] = set()
    for row in rows:
        record_id, context_id, _ = _identity(row)
        if row.get("split") != DataSplit.TEST or record_id in identifiers or context_id in contexts:
            raise ValueError("test contexts require unique IDs and split=test")
        identifiers.add(record_id)
        contexts.add(context_id)
    return rows


def verify_adapter(adapter: Path, model: Path, revision: str, arm: Arm, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind evaluation to this arm's validation-selected adapter and layouts."""
    training_path = adapter.parent / "configuration.json"
    best_path = adapter.parent / "best.json"
    if not training_path.is_file() or not best_path.is_file():
        raise ValueError("adapter must come from a completed validation-selected training run")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    saved = best.get("adapter")
    if not isinstance(saved, str) or Path(saved).resolve() != adapter.resolve():
        raise ValueError("adapter does not match the validation-selected checkpoint")
    expected = {
        "arm": arm,
        "model_revision": revision,
        "model_weights_sha256": file_sha256(model / "model.safetensors"),
        "model_config_sha256": file_sha256(model / "config.json"),
        "tokenizer_sha256": file_sha256(model / "tokenizer.json")
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise ValueError("adapter arm, model revision or source files differ from training")
    known_layouts = set(training.get("train_layout_ids", [])) | set(training.get("validation_layout_ids", []))
    if not known_layouts:
        raise ValueError("training configuration does not record its layouts")
    if known_layouts & {row["layout_id"] for row in rows}:
        raise ValueError("test layouts overlap generator training or validation")
    return training


def evaluate(args: argparse.Namespace) -> None:
    rows = test_contexts(args.data)
    arm = Arm(args.arm)
    is_scaled = arm in (Arm.SCALED, Arm.RELATIVE, Arm.ABSOLUTE)
    requested = args.requested_scale if is_scaled else None
    if not is_scaled and args.requested_scale is not None:
        raise ValueError("unscaled evaluation must not receive --requested-scale")
    if is_scaled and requested is None:
        raise ValueError("scaled evaluation requires --requested-scale")
    messages = [generator_messages(row, arm, requested) for row in rows]
    _model_revision(Path(args.model), args.revision)
    if arm == Arm.BASE:
        if args.adapter is not None:
            raise ValueError("the base arm must not load an adapter")
        adapter = None
        training = None
    else:
        if not args.adapter:
            raise ValueError("adapted evaluation requires a validation-selected adapter")
        adapter = Path(args.adapter)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            if not (adapter / name).is_file():
                raise ValueError(f"adapter is missing {name}")
        training = verify_adapter(adapter, Path(args.model), args.revision, arm, rows)
    output = Path(args.output)
    meta_path = output.with_suffix(".run.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError("use a fresh evaluation output")
    output.parent.mkdir(parents=True, exist_ok=True)
    configuration = {
        "data_sha256": file_sha256(args.data),
        "model": str(Path(args.model).resolve()),
        "revision": args.revision,
        "model_weights_sha256": file_sha256(Path(args.model) / "model.safetensors"),
        "model_config_sha256": file_sha256(Path(args.model) / "config.json"),
        "tokenizer_sha256": file_sha256(Path(args.model) / "tokenizer.json"),
        "adapter": str(adapter.resolve()) if adapter is not None else None,
        "adapter_weights_sha256": file_sha256(adapter / "adapter_model.safetensors") if adapter else None,
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json") if adapter else None,
        "code_sha256": file_sha256(__file__),
        "arm": arm,
        "requested_scale": requested,
        "training_configuration_sha256": file_sha256(adapter.parent / "configuration.json") if adapter else None,
        "training_comparability_sha256": training["comparability_sha256"] if training is not None else None,
        "controller_hash": training["controller_hash"] if training is not None else None,
        "record_ids": [row["record_id"] for row in rows],
        "dtype": args.dtype,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "output_normalisation": ("remove_one_outer_fence" if arm not in (Arm.UNSCALED, Arm.SCALED) else "strict"),
        "enable_thinking": False,
        "command": sys.argv
    }
    atomic_json(meta_path, {**configuration, "status": "initializing"})
    started, failure = time.monotonic(), None
    model_tuple = None
    try:
        if args.require_cuda:
            import torch as torch_module
            if not torch_module.cuda.is_available():
                raise RuntimeError("CUDA required but unavailable; no silent CPU fallback")
        model_tuple = initialise(args, adapter=str(adapter) if adapter is not None else None, train=False)
        torch, tokenizer, model, device = model_tuple
        for prompt in messages:
            tokens = tokenizer.apply_chat_template(prompt,
                                                   tokenize=True,
                                                   add_generation_prompt=True,
                                                   enable_thinking=False)
            if len(tokens) + args.max_new_tokens > args.max_length:
                raise ValueError("prompt plus generation exceeds max_length; no truncation")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    with output.open("x", encoding="utf-8", buffering=1) as stream:
        for record, prompt in zip(rows, messages, strict=True):
            row: dict[str, Any] = {key: record.get(key) for key in ("record_id", "context_id", "layout_id", "family")}
            row.update(arm=arm,
                       requested_scale=requested,
                       prompt=prompt,
                       raw_output=None,
                       parsed=None,
                       json_valid=None,
                       strict_json_valid=None,
                       normalized_json_valid=None,
                       fence_removed=None,
                       validation_errors=[],
                       generated_tokens=None,
                       generation_seconds=None)
            if failure is not None:
                row.update(status=PredictionStatus.NOT_ATTEMPTED, error=failure)
            elif expired(args):
                failure = "evaluation deadline reached"
                row.update(status=PredictionStatus.NOT_ATTEMPTED, error=failure)
            else:
                try:
                    assert model_tuple is not None
                    torch, tokenizer, model, device = model_tuple
                    raw, tokens, seconds = generate(torch, model, tokenizer, record, args, device, messages=prompt)
                    row.update(raw_output=raw,
                               generated_tokens=tokens,
                               generation_seconds=seconds,
                               **assess_output(raw, record["start_observation"],
                                               normalize=arm not in (Arm.UNSCALED, Arm.SCALED)))
                except Exception as error:
                    failure = f"{type(error).__name__}: {error}"
                    row.update(status=PredictionStatus.INFRASTRUCTURE_ERROR, error=failure)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    atomic_json(
        meta_path, {
            **configuration, "status": "failed" if failure else "complete",
            "error": failure,
            "elapsed_seconds": time.monotonic() - started
        })
    if failure:
        raise RuntimeError(f"evaluation stopped: {failure}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=list(Mode))
    parser.add_argument("--data", help="calibration JSONL or separate test-context JSONL")
    parser.add_argument("--demo-data", help="common checked demonstration pool for D/N/T/U training")
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=list(Arm), help="required except for the unmeasured resource check")
    parser.add_argument("--model", help="verified local model snapshot; required except for inspect")
    parser.add_argument("--revision", help="exact local Hugging Face commit; required except for inspect")
    parser.add_argument("--adapter", help="saved generator adapter; required for evaluate")
    parser.add_argument("--requested-scale", type=float, help="relative u or absolute action duration at evaluation")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--rank", type=int, choices=(2, 4), default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--min-informative-train-contexts", type=int, default=4)
    parser.add_argument("--min-informative-validation-contexts", type=int, default=1)
    parser.add_argument("--stop-monotonic", type=float)
    parser.add_argument("--stop-epoch", type=float)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.accumulation, args.max_length, args.max_new_tokens, args.max_steps, args.eval_every,
           args.min_informative_train_contexts, args.min_informative_validation_contexts) < 1:
        parser.error("steps, lengths, threads and accumulation must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be finite and positive")
    if args.mode == Mode.RESOURCE_CHECK and args.arm is not None:
        parser.error("resource-check uses diagnostic scale 0.5 and does not accept --arm")
    if args.mode != Mode.RESOURCE_CHECK and args.arm is None:
        parser.error("this mode requires --arm")
    if not args.data and not (args.arm == Arm.DEMONSTRATION and args.mode in (Mode.INSPECT, Mode.BENCHMARK, Mode.TRAIN)):
        parser.error("--data is required except for D inspection, benchmarking and training")
    if args.mode in (Mode.INSPECT, Mode.BENCHMARK, Mode.TRAIN) and args.arm in (
            Arm.DEMONSTRATION, Arm.MEASURED, Arm.ABSOLUTE, Arm.RELATIVE) and not args.demo_data:
        parser.error("D/N/T/U require --demo-data")
    if args.mode in (Mode.BENCHMARK, Mode.TRAIN) and args.arm in (
            Arm.DEMONSTRATION, Arm.MEASURED, Arm.ABSOLUTE, Arm.RELATIVE) and args.accumulation % 2:
        parser.error("D/N/T/U require even gradient accumulation for an exact 50/50 mix")
    if args.mode != Mode.INSPECT and not all((args.model, args.revision)):
        parser.error("resource-check, benchmark, train and evaluate require --model and --revision")
    if args.arm == Arm.BASE and args.mode != Mode.EVALUATE:
        parser.error("the base arm is available for evaluation only")
    if args.mode == Mode.EVALUATE and args.arm == Arm.BASE and args.adapter is not None:
        parser.error("the base arm must not load an adapter")
    if args.mode == Mode.EVALUATE and args.arm != Arm.BASE and not args.adapter:
        parser.error("evaluate requires --adapter")
    if args.mode == Mode.INSPECT:
        data, demonstration = load_study_data(args)
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("use a fresh inspection output")
        atomic_json(
            output, {
                "data_sha256": file_sha256(args.demo_data if args.arm == Arm.DEMONSTRATION else args.data),
                "demonstration_sha256": file_sha256(args.demo_data) if demonstration else None,
                "arm": args.arm,
                "controller_hash": data.controller_hash,
                "action_cap": data.action_cap,
                "attempts_per_candidate": data.attempts_per_candidate,
                "audit": data.audit
            })
        print(json.dumps(data.audit, indent=2))
    elif args.mode == Mode.RESOURCE_CHECK:
        resource_check(args)
    elif args.mode == Mode.BENCHMARK:
        benchmark(args)
    elif args.mode == Mode.TRAIN:
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
