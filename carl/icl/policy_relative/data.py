"""Load and validate demonstration and controller-measurement datasets."""

import argparse
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from carl.icl.intent_data import validate_prediction
from carl.icl.intent_generator_experiment import DataSplit
from carl.icl.intent_generator_experiment import read_records
from carl.icl.policy_relative.checks import _checked_context
from carl.icl.policy_relative.checks import _identity
from carl.icl.policy_relative.checks import _intent_key
from carl.icl.policy_relative.checks import _positive_int
from carl.icl.policy_relative.types import Arm
from carl.icl.policy_relative.types import Candidate
from carl.icl.policy_relative.types import Context
from carl.icl.policy_relative.types import FitData


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
        if not isinstance(split, str) or split not in (
            DataSplit.TRAIN,
            DataSplit.VALIDATION,
        ):
            raise ValueError(
                "calibration file may contain only train and validation records"
            )
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
        if any(
            row["split"] != first["split"]
            or row["layout_id"] != first["layout_id"]
            or row["start_observation"] != first["start_observation"]
            or row["task_goal"] != first["task_goal"]
            for row in group
        ):
            raise ValueError(
                "all candidates in a context must share split, board, layout and task"
            )
        if any(
            not isinstance(row.get("policy_measurements"), dict)
            or row["policy_measurements"].get("complete") is not True
            for row in group
        ):
            rejected.append(
                {
                    "context_id": context_id,
                    "split": first["split"],
                    "record_ids": [row["record_id"] for row in group],
                    "reason": "incomplete_measurements",
                }
            )
            continue
        expected = {
            _positive_int(
                row["policy_measurements"].get("expected_candidates"),
                "expected_candidates",
            )
            for row in group
        }
        if len(expected) != 1 or len(group) > next(iter(expected)):
            raise ValueError("inconsistent expected candidate count")
        if len(group) < next(iter(expected)):
            rejected.append(
                {
                    "context_id": context_id,
                    "split": first["split"],
                    "record_ids": [row["record_id"] for row in group],
                    "reason": "missing_candidates",
                }
            )
            continue
        context, signature = _checked_context(group)
        signatures.add(signature)
        if context is None:
            empty[group[0]["split"]] += 1
        else:
            split_contexts[group[0]["split"]].append(context)
    training, validation = (
        split_contexts[DataSplit.TRAIN],
        split_contexts[DataSplit.VALIDATION],
    )
    if not training or not validation:
        raise ValueError(
            "train and validation each need at least one complete context with a measured success; "
            f"rejected context IDs: {[row['context_id'] for row in rejected]}"
        )
    if len(signatures) != 1:
        raise ValueError(
            "controller, action cap and attempt count must match across splits"
        )
    controller, cap, attempts = next(iter(signatures))
    audit = {
        "rows": len(rows),
        "contexts": len(groups),
        "zero_success_contexts": dict(empty),
        "rejected_contexts": rejected,
        "rejected_context_count": len(rejected),
        "layout_ids_by_split": {
            name: sorted(layouts[name])
            for name in (DataSplit.TRAIN, DataSplit.VALIDATION)
        },
        "splits": {
            name: {
                "layouts": len(layouts[name]),
                "contexts_with_success": len(split_contexts[name]),
                "labelled_candidates": sum(
                    len(context.candidates) for context in split_contexts[name]
                ),
                "contexts_with_multiple_successes": sum(
                    len(context.candidates) > 1 for context in split_contexts[name]
                ),
                "contexts_with_distinct_success_durations": sum(
                    len(
                        {
                            candidate.mean_success_actions
                            for candidate in context.candidates
                        }
                    )
                    > 1
                    for context in split_contexts[name]
                ),
                "distinct_scales": sorted(
                    {
                        candidate.u
                        for context in split_contexts[name]
                        for candidate in context.candidates
                        if candidate.u is not None
                    }
                ),
            }
            for name in (DataSplit.TRAIN, DataSplit.VALIDATION)
        },
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
        if not isinstance(split, str) or split not in (
            DataSplit.TRAIN,
            DataSplit.VALIDATION,
        ):
            raise ValueError(
                "demonstrations may contain only train and validation records"
            )
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
            raise ValueError(
                "one visible demonstration context cannot have multiple context IDs"
            )
        visible_contexts[visible] = context
        groups[context].append(row)
        layouts[split].add(layout)
    if layouts[DataSplit.TRAIN] & layouts[DataSplit.VALIDATION]:
        raise ValueError("demonstration train and validation layouts overlap")
    splits: dict[str, list[Context]] = defaultdict(list)
    for context_id, group in sorted(groups.items()):
        first = group[0]
        signature = (
            first["split"],
            first["layout_id"],
            first["start_observation"],
            first["task_goal"],
        )
        if any(
            (row["split"], row["layout_id"], row["start_observation"], row["task_goal"])
            != signature
            for row in group
        ):
            raise ValueError(
                "demonstration context must share split, board, layout and task"
            )
        if len({_intent_key(row["intent"]) for row in group}) != len(group):
            raise ValueError("duplicate demonstration intent within a context")
        candidates = tuple(
            Candidate(row, None, None, None, 1 / len(group), False)
            for row in sorted(group, key=lambda item: item["record_id"])
        )
        splits[first["split"]].append(
            Context(context_id, first["layout_id"], candidates)
        )
    if not splits[DataSplit.TRAIN] or not splits[DataSplit.VALIDATION]:
        raise ValueError("demonstrations need train and validation contexts")
    audit = {
        "rows": len(rows),
        "contexts": len(groups),
        "layout_ids_by_split": {
            name: sorted(layouts[name])
            for name in (DataSplit.TRAIN, DataSplit.VALIDATION)
        },
        "splits": {
            name: {
                "contexts": len(splits[name]),
                "candidates": sum(len(context.candidates) for context in splits[name]),
            }
            for name in (DataSplit.TRAIN, DataSplit.VALIDATION)
        },
    }
    return FitData(
        tuple(splits[DataSplit.TRAIN]),
        tuple(splits[DataSplit.VALIDATION]),
        None,
        None,
        None,
        audit,
    )


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
    demo_rows = {
        candidate.record["record_id"]: candidate.record
        for context in (*demonstration.train, *demonstration.validation)
        for candidate in context.candidates
    }
    for row in read_records(args.data):
        original = demo_rows.get(row.get("record_id"))
        if original is None or any(
            row.get(key) != original.get(key)
            for key in (
                "context_id",
                "layout_id",
                "split",
                "start_observation",
                "task_goal",
                "intent",
            )
        ):
            raise ValueError(
                "every calibrated candidate must match the common demonstration pool"
            )
    return load_fit_data(args.data), demonstration
