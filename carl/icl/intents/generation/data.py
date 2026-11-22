"""Inputs, prompts, encoding and sampling for Sokoban intent generation."""

import hashlib
import json
import os
import random
from collections import defaultdict
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_data import DA_VERSION
from carl.icl.predicate_prompts import generator_messages as predicate_messages

SYSTEM = (
    "You propose one achievable, non-trivial Sokoban intent using only the current board. "
    "Rows and columns are zero-based. # is wall, - is floor, . is goal, $ is box, "
    "* is box on goal, @ is player and + is player on goal. "
    "Return only a JSON object with keys effects, preserve, instruction. "
    "effects is a nonempty array of objects, each either "
    '{"predicate":"box_at","row":R,"col":C}, '
    '{"predicate":"player_at","row":R,"col":C}, or '
    '{"predicate":"boxes_on_goals_at_least","count":N}. '
    "Every effect must be false on the current board. All effects are required together. "
    "preserve is an array and may be empty; instruction is null or a short string."
)
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


class DataSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class GeneratorMethod(StrEnum):
    BASE = "base"
    ADAPTED = "adapted"
    IN_CONTEXT = "in_context"


class RunMode(StrEnum):
    INSPECT = "inspect"
    BENCHMARK = "benchmark"
    TRAIN = "train"
    EVALUATE = "evaluate"


class TrainingAugmentation(StrEnum):
    NONE = "none"
    D4 = "d4"


def prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Whitelist permitted input fields; do not serialise the record itself."""
    board = record["start_observation"]
    goal = record["task_goal"]
    if (
        not isinstance(board, list)
        or not board
        or not all(isinstance(row, str) for row in board)
    ):
        raise ValueError("start_observation must be a nonempty list of board rows")
    if not isinstance(goal, str):
        raise ValueError("task_goal must be a string")
    if predicate_record(record):
        return predicate_messages(board, goal, None, "unscaled")
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": "Task: " + goal + "\nCurrent board:\n" + "\n".join(board),
        },
    ]


def target_text(record: dict[str, Any]) -> str:
    intent = record["intent"]
    if (
        not isinstance(intent, dict)
        or not isinstance(intent.get("effects"), list)
        or not intent["effects"]
    ):
        raise ValueError("checked intent must contain a nonempty effects array")
    return json.dumps(intent, ensure_ascii=False, separators=(",", ":"))


def read_records(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    """Read legacy rows and validate versioned annotations before selecting a split."""
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("intent dataset records must be objects")
                rows.append(record)
    if any(row.get("schema_version") == DA_VERSION for row in rows):
        _check_versioned_annotations(rows, DA_VERSION)
    return [row for row in rows if split is None or row.get("split") == split]


def _check_versioned_annotations(rows: list[dict[str, Any]], version: str) -> None:
    from carl.icl.boards import canonical_layout_id, from_rows
    from carl.icl.intent_data import SCHEMA_VERSION, validate_prediction
    from carl.icl.intent_domain import TASK_GOAL, annotation_identity, context_identity, decode_intent, render_instruction

    layouts: dict[str, DataSplit] = {}
    groups: dict[str, DataSplit] = {}
    records: set[str] = set()
    for row in rows:
        if row.get("schema_version") != version or row.get("intent_version") != SCHEMA_VERSION:
            raise ValueError("mixed schema versions in checked annotations")
        identifier = row.get("record_id")
        if not isinstance(identifier, str) or not identifier or identifier in records:
            raise ValueError("checked annotation record IDs must be unique and nonempty")
        records.add(identifier)
        raw_split = row.get("split")
        if not isinstance(raw_split, str):
            raise ValueError(f"{raw_split!r} is not a valid DataSplit")
        split = DataSplit(raw_split)
        group = record_task_group(row)
        if group is not None and groups.setdefault(group, split) != split:
            raise ValueError("checked annotation task groups overlap across splits")
        board = row.get("start_observation")
        if not isinstance(board, list):
            raise ValueError("checked annotation start_observation must contain board rows")
        layout = canonical_layout_id(from_rows(board))
        if row.get("layout_id") != layout:
            raise ValueError("checked annotation layout_id differs from the canonical layout")
        if layouts.setdefault(layout, split) != split:
            raise ValueError("checked annotation layouts overlap across splits")
        verification = row.get("validation")
        if (row.get("task_goal") != TASK_GOAL or not isinstance(verification, dict)
                or verification.get("accepted") is not True or verification.get("status") != "accepted"):
            raise ValueError("expected a checked and accepted Sokoban annotation")
        intent = decode_intent(row.get("intent"))
        errors = validate_prediction(intent, board)
        if errors or row.get("instruction") != render_instruction(intent):
            raise ValueError(f"invalid checked annotation {identifier}: {errors or ['instruction mismatch']}")
        context_id = context_identity(board)
        if row.get("context_id") != context_id or annotation_identity(context_id, intent) != identifier:
            raise ValueError("checked annotation identity differs from the board or intent")


def predicate_record(record: dict[str, Any]) -> bool:
    """Distinguish checked annotations from the historical JSONL format."""
    return record.get("schema_version") == DA_VERSION


def record_task_group(record: dict[str, Any]) -> str | None:
    """Find the declared task group, falling back to the demonstration source."""
    declared = [record[key] for key in ("task_group", "task_group_id") if key in record]
    if any(not isinstance(group, str) or not group.strip() for group in declared) or (
            len(declared) == 2 and declared[0] != declared[1]):
        raise ValueError("task group fields must be nonempty and agree")
    group = declared[0] if declared else record.get("source_id")
    return group if isinstance(group, str) and group.strip() else None


def same_prompt_layout(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Also check canonical wall and goal identity with versioned support data."""
    if DA_VERSION in (first.get("schema_version"), second.get("schema_version")):
        from carl.icl.boards import canonical_layout_id, from_rows

        return canonical_layout_id(from_rows(first["start_observation"])) == canonical_layout_id(
            from_rows(second["start_observation"]),
        )
    return bool(first["layout_id"] == second["layout_id"])


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-JSON constant {value}")

    return json.loads(text, parse_constant=reject_constant)


def encode_example(
    tokenizer: Any, record: dict[str, Any], max_length: int
) -> dict[str, list[int]]:
    return encode_messages(tokenizer, prompt_messages(record), target_text(record), max_length)


def encode_messages(
    tokenizer: Any, messages: list[dict[str, str]], target: str, max_length: int,
) -> dict[str, list[int]]:
    """Encode an entire response with the visible chat prefix masked from loss."""
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prefix, list):
        prefix = prefix.tolist()
    end = tokenizer.eos_token or ""
    answer = tokenizer.encode(target + end, add_special_tokens=False)
    if not answer:
        raise ValueError("the complete intent response must contain tokens")
    ids = prefix + answer
    if len(ids) > max_length:
        raise ValueError(
            f"complete example needs {len(ids)} tokens; limit is {max_length}; no truncation performed"
        )
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prefix) + answer,
    }


def board_balanced(
    records: list[dict[str, Any]], seed: int
) -> Iterator[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["layout_id"])].append(record)
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in groups}
    while groups:
        keys = list(groups)
        rng.shuffle(keys)
        for key in keys:
            if not pools[key]:
                pools[key] = rng.sample(groups[key], len(groups[key]))
            yield pools[key].pop()


def augmented_training_samples(
    records: list[dict[str, Any]], seed: int, augmentation: TrainingAugmentation | str
) -> Iterator[tuple[dict[str, Any], str]]:
    """Keep board sampling unchanged while drawing symmetries from a separate RNG."""
    from carl.icl.intent_augmentation import SYMMETRIES
    from carl.icl.intent_augmentation import BoardSymmetry
    from carl.icl.intent_augmentation import transform_training_record

    mode = TrainingAugmentation(augmentation)
    rng = random.Random(seed ^ 0x5D4A7A)
    for record in board_balanced(records, seed):
        symmetry = (
            rng.choice(SYMMETRIES)
            if mode == TrainingAugmentation.D4
            else BoardSymmetry.IDENTITY
        )
        yield (
            transform_training_record(record, symmetry)
            if mode == TrainingAugmentation.D4
            else record,
            symmetry.value,
        )


def one_per_board(
    records: list[dict[str, Any]], limit: int = 0
) -> list[dict[str, Any]]:
    """Deterministically pick the first context per independent layout."""
    seen: set[str] = set()
    result = []
    for record in records:
        key = str(record["layout_id"])
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
        if limit and len(result) >= limit:
            break
    return result


def validation_contexts(
    records: list[dict[str, Any]], max_boards: int, per_board: int
) -> list[dict[str, Any]]:
    """Select fixed contexts without inspecting model results; diversify effect types."""
    boards: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        boards.setdefault(str(record["layout_id"]), []).append(record)
    selected = []
    for board_records in list(boards.values())[: max_boards or None]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in board_records:
            groups.setdefault(str(record.get("effect_type", "unknown")), []).append(
                record
            )
        board_selected: list[dict[str, Any]] = []
        while len(board_selected) < per_board and any(groups.values()):
            for group in groups.values():
                if group and len(board_selected) < per_board:
                    board_selected.append(group.pop(0))
        selected.extend(board_selected)
    return selected


def evaluation_records(
    records: list[dict[str, Any]], ids_path: str | None, limit: int
) -> list[dict[str, Any]]:
    if ids_path is None:
        return one_per_board(records, limit)
    if limit:
        raise ValueError("use either --test-ids or --test-boards, not both")
    ids = json.loads(Path(ids_path).read_text(encoding="utf-8"))["record_ids"]
    if not isinstance(ids, list) or len(ids) != len(set(ids)):
        raise ValueError("frozen test IDs must be a unique list")
    lookup = {record["record_id"]: record for record in records}
    if any(record_id not in lookup for record_id in ids):
        raise ValueError("frozen test ID is absent from test records")
    return [lookup[record_id] for record_id in ids]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)
