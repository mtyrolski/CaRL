"""Freeze and verify train-only support for the predicate ICL pilot."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from filelock import FileLock

from carl.icl.boards import canonical_layout_id, from_rows
from carl.icl.intent_annotations import DA_VERSION
from carl.icl.intent_domain import Action, SokobanObservation, SokobanSnapshot, check_intent, decode_intent, replay_fragment
from carl.icl.intents.generation.data import DataSplit, atomic_json, file_sha256, parse_json, read_records
from carl.icl.policy_relative.data import load_fit_data
from carl.icl.policy_relative.labels import MEASUREMENT_VERSION, contract_hash
from carl.icl.types import JSONObject

SUPPORT_VERSION = "sokoban-icl-support-v1"
SCALE_ARMS = frozenset({
    "unscaled", "absolute_d", "ratio_d_over_ac", "offset_rank", "normalized_offset", "relative_u",
})
PromptExample = tuple[JSONObject, JSONObject]


@dataclass(frozen=True)
class FrozenSupport:
    """The frozen manifest and role examples, exposed as independent copies."""

    _serialized: str

    @property
    def manifest(self) -> JSONObject:
        return cast(JSONObject, json.loads(self._serialized))

    @property
    def fingerprint(self) -> str:
        return contract_hash(self.manifest)

    @property
    def policy_fingerprint(self) -> str:
        return cast(str, self.manifest["policy"]["fingerprint"])

    @property
    def generator_fingerprint(self) -> str:
        return cast(str, self.manifest["generator"]["fingerprint"])

    @property
    def policy(self) -> tuple[PromptExample, ...]:
        return self._examples("policy")

    @property
    def generator(self) -> tuple[PromptExample, ...]:
        return self._examples("generator")

    def _examples(self, role: Literal["policy", "generator"]) -> tuple[PromptExample, ...]:
        return tuple((entry["input"], entry["output"]) for entry in self.manifest[role]["examples"])

    def assert_query(
        self,
        observation: SokobanObservation,
        *,
        split: Literal["validation", "test"],
        task_group: str | None = None,
    ) -> None:
        """Check held-out layout and task-group identity before rendering a prompt."""
        if split not in (DataSplit.VALIDATION, DataSplit.TEST):
            raise ValueError("predicate ICL query split must be validation or test")
        layout = _query_layout(observation, task_group)
        exposed = self.manifest["partitions"]
        splits = (DataSplit.TRAIN, DataSplit.VALIDATION) if split == DataSplit.TEST else (DataSplit.TRAIN,)
        for partition in splits:
            if layout in exposed[partition]["layout_ids"]:
                raise ValueError(f"query canonical layout overlaps the {partition} partition")
            if task_group in exposed[partition]["task_groups"]:
                raise ValueError(f"query task group overlaps the {partition} partition")
        if split == DataSplit.TEST and task_group is None:
            raise ValueError("test query requires a task group for leakage checks")

    def assert_policy_holdout(self, observation: SokobanObservation, *, task_group: str | None = None) -> None:
        """Keep measurement contexts separate from the policy examples."""
        examples = self.manifest["policy"]["examples"]
        if not examples:
            return
        layout = _query_layout(observation, task_group)
        if any(layout == entry["layout_id"] for entry in examples):
            raise ValueError("measurement canonical layout overlaps policy support")
        if any(task_group == entry["task_group"] for entry in examples):
            raise ValueError("measurement task group overlaps policy support")


def _query_layout(observation: SokobanObservation, task_group: str | None) -> str:
    if observation.partial:
        raise ValueError("cannot verify the canonical layout of a partial held-out observation")
    if task_group is not None and (not isinstance(task_group, str) or not task_group.strip()):
        raise ValueError("query task group must be nonempty")
    return canonical_layout_id(from_rows(list(observation.rows)))


def task_group_id(row: JSONObject) -> str:
    """Use an explicit task group when supplied, or the source trajectory ID."""
    declared = {row[key] for key in ("task_group", "task_group_id") if key in row and isinstance(row[key], str)}
    if len(declared) > 1 or any(
        key in row and (not isinstance(row[key], str) or not row[key].strip())
        for key in ("task_group", "task_group_id")
    ):
        raise ValueError("task group fields must agree and be nonempty strings")
    group = next(iter(declared), row.get("source_id"))
    if not isinstance(group, str) or not group.strip():
        raise ValueError("checked support data needs a task group or source_id")
    return group


def _checked_rows(path: Path) -> tuple[list[JSONObject], str]:
    digest = file_sha256(path)
    rows = read_records(path)
    if file_sha256(path) != digest:
        raise ValueError("support dataset changed while reading")
    if not rows or any(row.get("schema_version") != DA_VERSION for row in rows):
        raise ValueError("predicate ICL needs checked carl-da-v1 annotations or carl-dc-v2 measurements")
    if any(not isinstance(row.get(key), str) or not row[key].strip()
           for row in rows for key in ("source_id", "effect_type")):
        raise ValueError("checked support records need source_id and effect_type")
    return rows, digest


def _partitions(rows: Iterable[JSONObject]) -> JSONObject:
    layouts: dict[str, DataSplit] = {}
    groups: dict[str, DataSplit] = {}
    for row in rows:
        split, layout, group = DataSplit(row["split"]), row["layout_id"], task_group_id(row)
        if layouts.setdefault(layout, split) != split:
            raise ValueError("canonical layouts overlap across support dataset splits")
        if groups.setdefault(group, split) != split:
            raise ValueError("task groups overlap across support dataset splits")
    return {
        split: {
            "layout_ids": sorted(key for key, value in layouts.items() if value == split),
            "task_groups": sorted(key for key, value in groups.items() if value == split),
        }
        for split in (DataSplit.TRAIN, DataSplit.VALIDATION)
    }


def _select(rows: Iterable[JSONObject], validation: JSONObject, *, count: int, seed: int, role: str) -> list[JSONObject]:
    if not count:
        return []
    candidates = sorted(
        (row for row in rows if row["split"] == DataSplit.TRAIN),
        key=lambda row: (
            row["effect_type"] != validation["effect_type"],
            contract_hash([SUPPORT_VERSION, role, seed, validation["record_id"], row["record_id"]]),
            row["record_id"],
        ),
    )
    selected: list[JSONObject] = []
    used_layouts: set[str] = set()
    for row in candidates:
        layout = row["layout_id"]
        if layout not in used_layouts:
            used_layouts.add(layout)
            selected.append(row)
            if len(selected) == count:
                return selected
    raise ValueError(f"{role} needs {count} checked training layouts, found {len(selected)}")


def _policy_actions(path: Path, examples: list[JSONObject]) -> tuple[dict[str, str], str | None]:
    if not examples:
        return {}, None
    if not path.is_file():
        raise ValueError("policy support requires the annotation audit with verified action fragments")
    digest = file_sha256(path)
    by_id = {row["record_id"]: row for row in examples}
    actions: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            entry = parse_json(line)
            if not isinstance(entry, dict):
                raise ValueError("annotation audit rows must be objects")
            identifier = entry.get("record_id")
            validation = entry.get("validation")
            if (
                not isinstance(identifier, str) or identifier not in by_id
                or not isinstance(validation, dict) or validation.get("accepted") is not True
            ):
                continue
            if identifier in actions:
                raise ValueError("annotation audit repeats a checked support record")
            actions[identifier] = _verified_first_action(by_id[identifier], entry)
    if actions.keys() != by_id.keys() or file_sha256(path) != digest:
        raise ValueError("annotation audit lacks checked support evidence or changed while reading")
    return actions, digest


def _verified_first_action(row: JSONObject, entry: JSONObject) -> str:
    keys = (
        "schema_version", "intent_version", "split", "source_id", "context_id", "layout_id",
        "start_observation", "task_goal", "intent", "instruction", "demonstration_offset",
    )
    if any(entry.get(key) != row[key] for key in keys):
        raise ValueError("annotation audit disagrees with the checked support record")
    if task_group_id(entry) != task_group_id(row):
        raise ValueError("annotation audit disagrees with the checked support task group")
    evidence = entry.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("checked policy support has no action witness")
    actions = evidence.get("actions")
    if not isinstance(actions, list) or not actions or len(actions) != row["demonstration_offset"]:
        raise ValueError("checked policy support has an invalid action witness")
    fragment = replay_fragment(SokobanSnapshot(tuple(row["start_observation"])), actions)
    if (
        evidence.get("observations") != [list(step.rows) for step in fragment]
        or check_intent(decode_intent(row["intent"]), [step.observation for step in fragment]).status != "accepted"
    ):
        raise ValueError("policy support action witness fails simulator replay or intent checking")
    return Action(actions[0]).name


def _entry(row: JSONObject, example: PromptExample) -> JSONObject:
    query, answer = example
    return {
        "record_id": row["record_id"],
        "context_id": row["context_id"],
        "layout_id": row["layout_id"],
        "task_group": task_group_id(row),
        "split": row["split"],
        "source_id": row["source_id"],
        "effect_type": row["effect_type"],
        "input": query,
        "output": answer,
        "content_sha256": contract_hash([query, answer]),
    }


def _policy_entry(row: JSONObject, action: str) -> JSONObject:
    return _entry(row, (
        {"observation": row["start_observation"], "instruction": row["instruction"], "attempt_history": []},
        {"action": action},
    ))


def _generator_entry(row: JSONObject, arm: str) -> JSONObject:
    scale = None if arm == "unscaled" else row["policy_measurements"]["scale_labels"][arm]
    return _entry(row, (
        {"observation": row["start_observation"], "task_goal": row["task_goal"], "scale": scale},
        cast(JSONObject, decode_intent(row["intent"])),
    ))


def _generator_pool(data: Path, rows: list[JSONObject], digest: str, arm: str) -> tuple[list[JSONObject], JSONObject | None]:
    measured = any("policy_measurements" in row for row in rows)
    if not measured:
        if arm != "unscaled":
            raise ValueError("controller scale labels require a checked D_C dataset")
        return rows, None
    if any(not isinstance(item, dict) or item.get("schema_version") != MEASUREMENT_VERSION
           for item in (row.get("policy_measurements") for row in rows)):
        raise ValueError("generator support requires a versioned D_C measurement dataset")
    fit = load_fit_data(data)
    if file_sha256(data) != digest:
        raise ValueError("measurement dataset changed while validating scale labels")
    usable = [candidate.record for context in (*fit.train, *fit.validation) for candidate in context.candidates]
    return usable, fit.audit["measurement_provenance"]


def _build(
    data: Path,
    *,
    seed: int,
    policy_shots: int,
    generator_shots: int,
    scale_arm: str,
    validation_id: str,
    selection: JSONObject,
    policy_data: Path | None,
    policy_audit: Path | None,
) -> FrozenSupport:
    if not isinstance(seed, int) or isinstance(seed, bool) or any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in (policy_shots, generator_shots)
    ):
        raise ValueError("support requires an integer seed and nonnegative shot counts")
    if not isinstance(scale_arm, str) or scale_arm not in SCALE_ARMS:
        raise ValueError("unsupported predicate generator scale arm")
    if not isinstance(validation_id, str) or not validation_id or not isinstance(selection, dict):
        raise ValueError("support requires a validation record ID and selection settings")
    rows, digest = _checked_rows(data)
    policy_rows, policy_digest = (rows, digest) if policy_data is None else _checked_rows(policy_data)
    partitions = _partitions((*rows, *policy_rows))
    candidates, measurement = _generator_pool(data, rows, digest, scale_arm)
    anchor = next((row for row in candidates if row["record_id"] == validation_id
                   and row["split"] == DataSplit.VALIDATION), None)
    if anchor is None:
        raise ValueError("support selection needs a checked validation record in the generator dataset")
    policy = _select(policy_rows, anchor, count=policy_shots, seed=seed, role="policy")
    generator = _select(candidates, anchor, count=generator_shots, seed=seed, role="generator")
    audit_path = policy_audit or (policy_data or data).with_name("annotation_audit.jsonl")
    actions, audit_digest = _policy_actions(audit_path, policy)
    policy_entries = [_policy_entry(row, actions[row["record_id"]]) for row in policy]
    generator_entries = [_generator_entry(row, scale_arm) for row in generator]
    manifest = {
        "schema_version": SUPPORT_VERSION,
        "seed": seed,
        "selection": selection,
        "selection_rule": "match validation effect type, then seed-ranked records on distinct training layouts",
        "validation": {
            **{key: anchor[key] for key in ("record_id", "layout_id", "split", "effect_type", "task_goal")},
            "observation_sha256": contract_hash(anchor["start_observation"]),
            "task_group": task_group_id(anchor),
            "scale_label": None if scale_arm == "unscaled" else anchor["policy_measurements"]["scale_labels"][scale_arm],
            "measured_success": measurement is not None,
        },
        "partitions": partitions,
        "scale_arm": scale_arm,
        "policy": {
            "shots": policy_shots, "dataset_sha256": policy_digest, "action_audit_sha256": audit_digest,
            "examples": policy_entries, "fingerprint": contract_hash(policy_entries),
        },
        "generator": {
            "shots": generator_shots, "dataset_sha256": digest, "measurement_provenance": measurement,
            "examples": generator_entries, "fingerprint": contract_hash(generator_entries),
        },
    }
    return FrozenSupport(json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False))


def freeze_support(
    data: Path,
    destination: Path,
    *,
    seed: int,
    policy_shots: int,
    generator_shots: int,
    scale_arm: str,
    validation_id: str,
    selection: JSONObject,
    policy_data: Path | None = None,
    policy_audit: Path | None = None,
) -> FrozenSupport:
    """Freeze ordered role examples and their hashes without overwriting a manifest."""
    support = _build(
        data, seed=seed, policy_shots=policy_shots, generator_shots=generator_shots, scale_arm=scale_arm,
        validation_id=validation_id, selection=selection, policy_data=policy_data, policy_audit=policy_audit,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if destination.exists():
            raise FileExistsError(f"support manifest already exists: {destination}")
        atomic_json(destination, support.manifest)
    return support


def load_support(
    path: Path,
    data: Path,
    *,
    policy_data: Path | None = None,
    policy_audit: Path | None = None,
) -> FrozenSupport:
    """Recreate the selection and reject any change to data, order or support contents."""
    manifest = parse_json(path.read_text(encoding="utf-8"))
    try:
        if manifest["schema_version"] != SUPPORT_VERSION:
            raise ValueError("unsupported predicate support manifest version")
        support = _build(
            data, seed=manifest["seed"], policy_shots=manifest["policy"]["shots"],
            generator_shots=manifest["generator"]["shots"], scale_arm=manifest["scale_arm"],
            validation_id=manifest["validation"]["record_id"], selection=manifest["selection"],
            policy_data=policy_data, policy_audit=policy_audit,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid predicate support manifest") from exc
    if support.fingerprint != contract_hash(manifest):
        raise ValueError("frozen support manifest differs from the checked data or selection")
    return support
