"""Replay-verified, layout-disjoint demonstrations and evaluation boards."""
import hashlib
import json
from pathlib import Path
from typing import Final, TypeAlias, cast

import joblib
import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows, level_key, state_hash, to_rows, validate_subgoal
from carl.icl.config import ExperimentConfig
from carl.icl.data_sources.sokoban import (
    DIRECTIONS,
    INVERSE,
    _walks,
    environment,
    reverse_board,
    verify_solution,
)
from carl.icl.types import (
    Board,
    Candidate,
    Instance,
    JSONObject,
    JSONValue,
    TrainingContext,
    TrainingData,
    ValueExample,
)

FAMILIES: Final = ((12, 4), (12, 10), (16, 4), (24, 7), (32, 10))

TrajectoryRecord: TypeAlias = tuple[str, list[Board], list[int]]

__all__ = [
    "DIRECTIONS", "FAMILIES", "INVERSE", "PRIVATE_FIELDS", "TrajectoryRecord", "_walks",
    "dataset_signature", "environment", "load_data", "prepare", "prepare_value_examples",
    "reverse_board", "verify_solution",
]


def dataset_signature(config: ExperimentConfig) -> JSONObject:
    """Identify data-selection settings independently of inference conditions."""
    return {
        "seed": config.seed,
        "train_boards": config.train_boards,
        "validation_boards": config.validation_boards,
        "test_boards": config.test_boards,
        "reference_window": config.calibration.reference_window,
        "candidates": config.calibration.candidates,
        "source": str(config.path("trajectory_source")),
    }


def prepare(config: ExperimentConfig) -> JSONObject:
    """Export reproducible, layout-disjoint demonstrations and verified evaluation boards."""
    root = config.path("data_dir")
    source = config.path("trajectory_source")
    raw = joblib.load(source)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    rng = np.random.default_rng(config.seed)
    required = config.train_boards + config.validation_boards + config.test_boards
    selected, seen = [], set()
    transitions = 0
    for index in rng.permutation(len(trajectories)):
        trajectory = np.asarray(trajectories[int(index)])
        if len(trajectory) <= config.calibration.reference_window:
            continue
        env = environment(trajectory[0])
        key = level_key(trajectory[0])
        if key in seen or env.is_solved(trajectory[0]) or not env.is_solved(trajectory[-1]):
            continue
        actions = []
        for before, after in zip(trajectory[:-1], trajectory[1:]):
            if env.is_solved(before):
                break
            validate_subgoal(before, after)
            matches = [a for a in range(4) if np.array_equal(env.next_state(before, a), after)]
            if not matches or np.array_equal(before, after):
                raise ValueError(f"Trajectory {index} has a non-replayable or no-op transition")
            actions.append(matches[0])
        if len(actions) < config.calibration.reference_window:
            continue
        states = verify_solution(trajectory[0], actions)
        selected.append((f"trajectory_{int(index)}", states, actions))
        seen.add(key)
        transitions += len(actions)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise ValueError(f"Need {required} distinct successful layouts; found {len(selected)}")
    provenance = {
        "signature": dataset_signature(config),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "replayed_transitions": transitions,
        "source_trajectories": len(trajectories),
        "split_unit": "walls + goals + board shape",
        "generator_version": 1,
    }
    contexts: list[TrainingContext] = []
    for identifier, states, actions in selected[: config.train_boards]:
        candidates: list[Candidate] = []
        unique: set[str] = set()
        for k in range(1, min(len(states), config.calibration.reference_window + 1)):
            candidates.append(Candidate(subgoal=to_rows(states[k]), k=k, actions=actions[:k]))
        indices = []
        for j, candidate in enumerate(candidates):
            key = json.dumps(candidate["subgoal"])
            if key != json.dumps(to_rows(states[0])) and key not in unique:
                unique.add(key)
                indices.append(j)
        chosen = rng.choice(
            indices, min(len(indices), config.calibration.candidates), replace=False
        ).tolist()
        contexts.append(
            {
                "id": identifier,
                "state": to_rows(states[0]),
                "remaining_actions": len(actions),
                "candidates": candidates,
                "calibration_indices": chosen,
            }
        )
    write_json(root / "training.json", {"provenance": provenance, "contexts": contexts})
    families: dict[str, JSONObject] = {}

    def export(family: str, split: str, records: list[TrajectoryRecord]) -> None:
        public, easy, private = [], [], []
        for identifier, states, actions in records:
            public.append(
                {"id": identifier, "state": to_rows(states[0]), "level_key": level_key(states[0])}
            )
            easy.append(
                {
                    "id": identifier + "_one_move",
                    "state": to_rows(states[-2]),
                    "level_key": level_key(states[0]),
                }
            )
            private.append({"id": identifier, "actions": actions})
        write_json(root / family / f"{split}.json", {"provenance": provenance, "instances": public})
        write_json(
            root / family / f"{split}_easy.json", {"provenance": provenance, "instances": easy}
        )
        write_json(root / "private_witnesses" / family / f"{split}.json", private)
        families.setdefault(family, {})[split] = {
            "count": len(public),
            "shapes": sorted({str(np.asarray(r[1][0]).shape) for r in records}),
        }

    start = config.train_boards
    export("real_12x12x4", "validation", selected[start : start + config.validation_boards])
    export("real_12x12x4", "test", selected[start + config.validation_boards :])
    generation = []
    for family_index, (size, boxes) in enumerate(FAMILIES):
        family = f"generated_{size}x{size}x{boxes}"
        for split_index, (split, count) in enumerate(
            (("validation", config.validation_boards), ("test", config.test_boards))
        ):
            records = []
            for index in range(count):
                board_seed = config.seed + 100000 * (family_index + 1) + 1000 * split_index + index
                state, actions, metadata = reverse_board(size, boxes, board_seed)
                if level_key(state) in seen:
                    raise ValueError("Generated board overlaps a prepared layout")
                seen.add(level_key(state))
                identifier = f"{family}_{split}_{index}"
                records.append((identifier, verify_solution(state, actions), actions))
                generation.append({"id": identifier, **metadata})
            export(family, split, records)
    manifest = {
        "provenance": provenance,
        "families": families,
        "generation": generation,
        "train_contexts": len(contexts),
        "total_unique_layouts": len(seen),
        "synthetic_distribution": "connected random-walk floor, reverse pulls, verified witness; not legacy generator",
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def prepare_value_examples(config: ExperimentConfig) -> int:
    """Export replay-verified value demonstrations from training trajectories only."""
    root = config.path("data_dir")
    training_path = root / "training.json"
    training = cast(TrainingData, json.loads(training_path.read_text(encoding="utf-8")))
    provenance = training.get("provenance", {})
    if provenance.get("signature") != dataset_signature(config):
        raise ValueError("Prepared training data differs from config")
    source = config.path("trajectory_source")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if provenance.get("source_sha256") != source_hash:
        raise ValueError("Trajectory shard differs from the prepared training data")
    raw = joblib.load(source)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    examples: list[ValueExample] = []
    for context in training["contexts"]:
        identifier = context["id"]
        if not identifier.startswith("trajectory_") or not identifier[11:].isdigit():
            raise ValueError(f"Unsupported training trajectory identifier: {identifier}")
        states = np.asarray(trajectories[int(identifier[11:])])
        remaining = context["remaining_actions"]
        if len(states) <= remaining or not np.array_equal(states[0], from_rows(context["state"])):
            raise ValueError(f"Training trajectory does not match {identifier}")
        env = environment(states[0])
        actions = []
        for before, after in zip(states[:remaining], states[1 : remaining + 1]):
            matches = [a for a in range(4) if np.array_equal(env.next_state(before, a), after)]
            if not matches or np.array_equal(before, after):
                raise ValueError(f"Non-replayable training transition in {identifier}")
            actions.append(matches[0])
        verified = verify_solution(states[0], actions)
        for candidate in context["candidates"]:
            if candidate["k"] >= len(verified) or to_rows(verified[candidate["k"]]) != candidate["subgoal"]:
                raise ValueError(f"Training example mismatch for {identifier}")
        for position in (0, remaining // 2, remaining - 1, remaining):
            examples.append(ValueExample(context=identifier, state=to_rows(verified[position]),
                                         remaining_actions=remaining - position))
    write_json(root / "value_examples.json", {
        "version": 1, "training_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "source_sha256": source_hash, "examples": examples,
    })
    return len(examples)


PUBLIC_INSTANCE_FIELDS: Final = frozenset({
    "id", "state", "level_key", "board_id", "family", "trajectory_id", "source_path",
    "source_sha256", "source_index", "generator_version", "seed", "state_hash", "layout_id", "split",
})
CAMPAIGN_SPLITS: Final = ("component_validation", "scale_validation", "pilot100", "holdout1000")
PRIVATE_FIELDS: Final = frozenset({"actions", "witness", "witnesses", "solution", "solutions", "future_state",
                                  "future_states", "subgoal", "subgoals", "target", "targets", "action_path"})


def _reject_private_fields(value: JSONValue) -> None:
    if isinstance(value, dict):
        if PRIVATE_FIELDS.intersection(value):
            raise ValueError("Public evaluation manifest contains private or future data")
        for item in value.values():
            _reject_private_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_private_fields(item)


def _public_instances(path: Path) -> list[Instance]:
    """Validate public records and return only solver inputs, never arbitrary metadata."""
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(dataset, dict) or set(dataset) - {"provenance", "instances"}
            or not isinstance(dataset.get("instances"), list)):
        raise ValueError(f"Invalid public dataset: {path}")
    _reject_private_fields(dataset)
    result: list[Instance] = []
    for record in dataset["instances"]:
        if not isinstance(record, dict) or set(record) - PUBLIC_INSTANCE_FIELDS:
            raise ValueError(f"Unexpected fields in public dataset: {path}")
        if not isinstance(record.get("id"), str) or not isinstance(record.get("level_key"), str):
            raise ValueError(f"Invalid public board identifier: {path}")
        for key in set(record) - {"state", "id", "level_key"}:
            if not isinstance(record[key], (str, int, type(None))) or isinstance(record[key], bool):
                raise ValueError(f"Invalid public metadata field {key}: {path}")
        rows = record.get("state")
        if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
            raise ValueError(f"Invalid public board state: {path}")
        state = from_rows(rows)
        validate_subgoal(state, state)
        if level_key(state) != record["level_key"]:
            raise ValueError("Instance layout checksum mismatch")
        if "layout_id" in record and record["layout_id"] != canonical_layout_id(state):
            raise ValueError("Instance canonical layout checksum mismatch")
        if "state_hash" in record and record["state_hash"] != state_hash(state):
            raise ValueError("Instance state checksum mismatch")
        result.append({"id": record["id"], "state": to_rows(state), "level_key": record["level_key"]})
    return result


def load_data(config: ExperimentConfig) -> tuple[TrainingData, list[Instance]]:
    """Load prepared examples and reject split overlap or mismatched provenance."""
    root = config.path("data_dir")
    training = cast(TrainingData, json.loads((root / "training.json").read_text(encoding="utf-8")))
    if training.get("provenance", {}).get("signature") != dataset_signature(config):
        raise ValueError(
            "Prepared data differs from config; use a separate data_dir and prepare it"
        )
    if config.value_examples == "trajectory":
        value_path = root / "value_examples.json"
        if not value_path.exists():
            raise FileNotFoundError("Run 'python -m scripts.run_icl_search prepare-value' first")
        value_data = json.loads(value_path.read_text(encoding="utf-8"))
        training_hash = hashlib.sha256((root / "training.json").read_bytes()).hexdigest()
        if (value_data.get("version") != 1 or value_data.get("training_sha256") != training_hash
                or value_data.get("source_sha256") != training.get("provenance", {}).get("source_sha256")):
            raise ValueError("Value examples do not match the prepared training data")
        context_layouts = {context["id"]: level_key(from_rows(context["state"]))
                           for context in training["contexts"]}
        examples = cast(list[ValueExample], value_data["examples"])
        if any(example["context"] not in context_layouts
               or level_key(from_rows(example["state"])) != context_layouts[example["context"]]
               or not isinstance(example["remaining_actions"], int)
               or example["remaining_actions"] < 0 for example in examples):
            raise ValueError("Invalid or non-training value example")
        training["value_examples"] = examples
    campaign_path = root / "campaign_dataset_manifest.json"
    is_campaign = campaign_path.exists()
    if is_campaign and (config.easy or config.split not in CAMPAIGN_SPLITS):
        raise ValueError("Campaign data require an explicit full-board campaign split")
    suffix = "_easy" if config.easy else ""
    selected_path = root / config.family / f"{config.split}{suffix}.json"
    if is_campaign:
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        expected = campaign["manifest_sha256"][f"{config.family}/{config.split}.json"]
        if hashlib.sha256(selected_path.read_bytes()).hexdigest() != expected:
            raise ValueError("Campaign public manifest checksum mismatch")
    instances = _public_instances(selected_path)[: config.instances]
    if len(instances) < config.instances or len(training["contexts"]) < config.few_shot:
        raise ValueError("Insufficient prepared instances or distinct few-shot contexts")
    layouts = {canonical_layout_id(from_rows(context["state"])) for context in training["contexts"]}
    # A pilot run must not open the separate, unexamined 1k holdout.
    splits = (config.split,) if is_campaign else ("validation", "test")
    groups = [layouts]
    for split in splits:
        records = _public_instances(root / config.family / f"{split}.json")
        ids = [canonical_layout_id(from_rows(row["state"])) for row in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate or symmetric layouts in {split}")
        groups.append(set(ids))
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups))):
        raise ValueError("Train/validation/test layouts overlap")
    return training, instances
