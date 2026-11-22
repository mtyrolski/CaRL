"""Measure checked intent pools from independent, reproducible policy attempts."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from filelock import FileLock, Timeout

from carl.icl.api import write_json
from carl.icl.intent_annotations import DA_VERSION
from carl.icl.intent_domain import DOMAIN_VERSION, SokobanSnapshot, decode_intent
from carl.icl.intents.generation.data import file_sha256, read_records
from carl.icl.intents.provenance import component_hashes
from carl.icl.luna_intent_rollout import MEASUREMENT_PROMPT, Action, ActionAttempt
from carl.icl.policy_relative.labels import MEASUREMENT_VERSION, contract_hash, label_candidates, pool_identity, reached_midpoints
from carl.icl.policy_relative.rollout import (
    ACTION_PARSER_VERSION, MEASUREMENT_PROMPT_VERSION, PolicyFactory, run_measurement_attempt,
)

COMPLETE_STATUSES = frozenset({
    "achieved", "action_limit", "task_solved", "preservation_violated", "invalid_response", "unknown",
})
EXPORTED_ATTEMPT_FIELDS = (
    "index", "seed", "status", "primitive_actions_attempted", "first_hit_actions", "task_solved", "model_calls", "metrics",
)

DIRECTED_FAILURES = frozenset({
    "intent_timeout", "invalid_response", "episode_ended", "task_solved", "preservation_failed", "preservation_unknown",
})


def directed_labels(
    attempts: Sequence[Sequence[dict[str, Any]]], planned: Sequence[int],
) -> list[dict[str, Any]]:
    """Summarize independent eligible futures and mask incomplete or noncomparative pools."""
    counts, successes, durations, fractions = [], [], [], []
    for candidates, expected in zip(attempts, planned, strict=True):
        futures: dict[str, dict[str, Any]] = {}
        for trial in candidates:
            if trial["status"] not in DIRECTED_FAILURES | {"intent_reached"}:
                continue
            future = trial["future_id"]
            if future in futures:
                previous = futures[future]
                if (trial["status"], trial["first_hit"]) != (previous["status"], previous["first_hit"]):
                    raise ValueError("An identical future has conflicting outcomes")
                continue
            futures[future] = trial
        n = len(futures)
        if not 0 < expected or n > expected:
            raise ValueError("Eligible futures do not match the frozen trial counts")
        hits = [row["first_hit"] for row in futures.values() if row["status"] == "intent_reached"]
        if any(isinstance(hit, bool) or not isinstance(hit, int) or hit < 1 for hit in hits):
            raise ValueError("A successful unmet intent requires a positive executed first hit")
        counts.append(n)
        successes.append(len(hits))
        durations.append(sum(hits) / len(hits) if hits else None)
        fractions.append(len(hits) / n if n else None)

    complete = counts == list(planned)
    comparative = complete and len({duration for duration in durations if duration is not None}) >= 2
    scales = reached_midpoints(durations, [p or 0.0 for p in fractions]) if complete else {}
    return [{
        "eligible_n": count, "successes": success, "p": fraction, "d": duration,
        "w": scales[index][0] if index in scales else None,
        "u": scales[index][1] if comparative and index in scales else None,
        "planned_n": planned[index], "context_complete": complete,
        "scale_supervision": comparative and index in scales,
        "uncertainty": [flag for condition, flag in (
            (count < planned[index], "incomplete_trials"), (0 < count <= 2, "small_n"),
            (count > 0 and success == 0, "no_success"), (not complete, "incomplete_context"),
            (complete and not comparative, "noncomparative_durations"),
        ) if condition],
    } for index, (count, success, fraction, duration) in enumerate(zip(counts, successes, fractions, durations, strict=True))]


@dataclass(frozen=True)
class MeasurementSettings:
    """A cap and trial plan shared across the frozen pool."""

    seed: int = 0
    action_cap: int = 8
    trials: int = 1
    planned_attempts: dict[str, int] = field(default_factory=dict)
    quota: dict[str, int] = field(default_factory=dict)
    offset_denominator: float = 8.0

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for value in (self.action_cap, self.trials, *self.planned_attempts.values(), *self.quota.values()):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("action cap, trial counts and quotas must be positive integers")
        if (isinstance(self.offset_denominator, bool) or not isinstance(self.offset_denominator, (int, float))
                or not math.isfinite(self.offset_denominator) or self.offset_denominator <= 0):
            raise ValueError("offset denominator must be positive and finite")


def trial_seed(seed: int, record_id: str, index: int) -> int:
    """Assign an independent stream keyed by candidate and zero-based attempt."""
    data = f"carl-policy-trial-v1:{seed}:{record_id}:{index}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _policy_contract(controller: dict[str, Any], memory: Sequence[ActionAttempt]) -> dict[str, Any]:
    if not isinstance(controller.get("id"), str) or not controller["id"]:
        raise ValueError("a frozen controller requires a nonempty id")
    if not isinstance(controller.get("deterministic", False), bool):
        raise ValueError("deterministic must be a boolean")
    if any(not isinstance(item, ActionAttempt) or not isinstance(item.action, Action) or not isinstance(item.no_op, bool)
           for item in memory):
        raise ValueError("initial memory must contain typed action attempts")
    component = controller.get("predicate_policy", {})
    return {
        "controller": json.loads(json.dumps(controller, allow_nan=False)),
        "initial_memory": [{"action": item.action.value, "no_op": item.no_op} for item in memory],
        "prompt_version": component.get("prompt_version", MEASUREMENT_PROMPT_VERSION),
        "prompt_sha256": component.get("prompt_sha256", hashlib.sha256(MEASUREMENT_PROMPT.encode()).hexdigest()),
        "parser_version": component.get("parser_version", ACTION_PARSER_VERSION),
        "sampling": "per-candidate-and-attempt-sha256-v1",
        "code": component_hashes(("luna_intent_rollout.py", "policy_relative/rollout.py")),
    }


def _select_pool(rows: list[dict[str, Any]], settings: MeasurementSettings) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] in ("train", "validation"):
            groups[row["context_id"], row["effect_type"]].append(row)
    result = []
    for (_, kind), records in sorted(groups.items()):
        limit = settings.quota.get(kind, 0) if settings.quota else len(records)
        records.sort(key=lambda row: (trial_seed(settings.seed, row["record_id"], 0), row["record_id"]))
        result.extend(records[:limit])
    if not result:
        raise ValueError("there are no selected train or validation candidates")
    if not set(settings.planned_attempts) <= {row["record_id"] for row in result}:
        raise ValueError("planned_attempts must refer to selected candidate IDs")
    return sorted(result, key=lambda row: (row["context_id"], row["record_id"]))


def _freeze(
    data: Path, settings: MeasurementSettings, controller: dict[str, Any], memory: Sequence[ActionAttempt],
    candidate_check: Callable[[dict[str, Any]], None] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    data_hash = file_sha256(data)
    rows = read_records(data)
    if not rows or any(row.get("schema_version") != DA_VERSION for row in rows):
        raise ValueError(f"expected checked {DA_VERSION} annotations")
    if file_sha256(data) != data_hash:
        raise ValueError("D_A changed while loading the candidate pool")
    pool = _select_pool(rows, settings)
    if candidate_check is not None:
        for row in pool:
            candidate_check(row)
    planned = {row["record_id"]: settings.planned_attempts.get(row["record_id"], settings.trials) for row in pool}
    policy = _policy_contract(controller, memory)
    environment_root = Path(__file__).resolve().parents[2] / "environment/sokoban"
    environment_hashes = {name: file_sha256(environment_root / name) for name in ("core.py", "env.py", "tokenizer.py")}
    environment_hashes.update(component_hashes(("boards.py", "intent_data.py", "intent_domain.py", "data_sources/sokoban.py")))
    provenance = {
        "schema_version": MEASUREMENT_VERSION,
        "domain_version": DOMAIN_VERSION,
        "data_sha256": data_hash,
        "environment_sha256": contract_hash(environment_hashes),
        "controller_hash": contract_hash(policy),
        "candidate_pool_hash": pool_identity(pool, planned),
        "split_sha256": contract_hash(sorted({(row["split"], row["layout_id"]) for row in pool})),
        "measurement_code": component_hashes(("policy_relative/measurements.py", "policy_relative/labels.py", "calibration.py")),
    }
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pool:
        counts[row["context_id"]][row["effect_type"]] += 1
    frozen = {
        "provenance": provenance,
        "controller_contract": policy,
        "settings": asdict(settings),
        "environment_sources": environment_hashes,
        "type_counts": {context: dict(kinds) for context, kinds in sorted(counts.items())},
        "candidates": [
            {**{key: row[key] for key in (
                "record_id", "context_id", "split", "layout_id", "effect_type", "demonstration_offset",
            )},
             "snapshot_sha256": contract_hash(row["start_observation"]),
             "planned_attempts": planned[row["record_id"]]}
            for row in pool
        ],
    }
    return frozen, pool, planned


def _read_attempts(
    output: Path, planned: dict[str, int], settings: MeasurementSettings,
) -> dict[str, dict[int, dict[str, Any]]]:
    stored: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in sorted((output / "attempts").glob("*/*.json")):
        identifier, index = path.parent.name, int(path.stem)
        if identifier not in planned or not 0 <= index < planned[identifier] or path.stem != str(index):
            raise ValueError("stored attempt falls outside the frozen pool")
        result = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(result, dict) or result.get("index") != index
                or result.get("seed") != trial_seed(settings.seed, identifier, index)
                or result.get("status") not in COMPLETE_STATUSES):
            raise ValueError("stored attempt does not match its frozen identity")
        stored[identifier][index] = result
    return stored


def _measurement_rows(
    pool: list[dict[str, Any]], completed: dict[str, dict[int, dict[str, Any]]],
    planned: dict[str, int], frozen: dict[str, Any], settings: MeasurementSettings,
) -> list[dict[str, Any]]:
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        contexts[row["context_id"]].append(row)
    provenance = frozen["provenance"]
    result = []
    for rows in contexts.values():
        trials = [list(completed.get(row["record_id"], {}).values()) for row in rows]
        successful = [
            [item["first_hit_actions"] for item in attempts if item["status"] == "achieved"]
            for attempts in trials
        ]
        denominators = [planned[row["record_id"]] for row in rows]
        complete = all(len(attempts) == count for attempts, count in zip(trials, denominators, strict=True))
        p = [len(hits) / count for hits, count in zip(successful, denominators, strict=True)]
        d = [sum(hits) / len(hits) if hits else None for hits in successful]
        offsets = [row["demonstration_offset"] for row in rows]
        labels = label_candidates(d, p, offsets, offset_denominator=settings.offset_denominator) if complete else None
        for index, row in enumerate(rows):
            scale = labels[index] if labels is not None else None
            attempts = [
                {key: attempt[key] for key in EXPORTED_ATTEMPT_FIELDS}
                for attempt in sorted(trials[index], key=lambda item: item["index"])
            ]
            result.append({
                **row,
                "measurement_provenance": provenance,
                "policy_measurements": {
                    "schema_version": MEASUREMENT_VERSION,
                    "complete": complete,
                    "expected_candidates": len(rows),
                    "planned_attempts": denominators[index],
                    "completed_attempts": len(attempts),
                    "controller_hash": provenance["controller_hash"],
                    "controller_contract": frozen["controller_contract"],
                    "candidate_pool_hash": provenance["candidate_pool_hash"],
                    "action_cap": settings.action_cap,
                    "offset_denominator": settings.offset_denominator,
                    "attempts": attempts,
                    "success_fraction": p[index] if complete else None,
                    "mean_success_actions": d[index] if complete else None,
                    "weight": scale["weight"] if scale else None,
                    "u": scale["relative_u"] if scale else None,
                    "scale_labels": scale,
                },
            })
    return result


def _publish(
    output: Path, pool: list[dict[str, Any]], completed: dict[str, dict[int, dict[str, Any]]],
    *, planned: dict[str, int], frozen: dict[str, Any], settings: MeasurementSettings,
) -> dict[str, Any]:
    rows = _measurement_rows(pool, completed, planned, frozen, settings)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        except BaseException:
            temporary.unlink()
            raise
    path = output / "D_C.jsonl"
    temporary.replace(path)
    outcomes = Counter(attempt["status"] for group in completed.values() for attempt in group.values())
    done, target = outcomes.total(), sum(planned.values())
    manifest = {
        **frozen["provenance"],
        "status": "complete" if done == target else "incomplete",
        "planned_attempts": target,
        "completed_attempts": done,
        "outcomes": dict(outcomes),
        "labelled_candidates": sum(row["policy_measurements"]["u"] is not None for row in rows),
        "output_sha256": file_sha256(path),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _run_pool(
    output: Path, pool: list[dict[str, Any]], completed: dict[str, dict[int, dict[str, Any]]],
    *, planned: dict[str, int], settings: MeasurementSettings, memory: Sequence[ActionAttempt],
    policy_factory: PolicyFactory,
) -> None:
    for row in pool:
        identifier = row["record_id"]
        for index in range(planned[identifier]):
            if index in completed[identifier]:
                continue
            seed = trial_seed(settings.seed, identifier, index)
            outcome = run_measurement_attempt(
                SokobanSnapshot(tuple(row["start_observation"])), decode_intent(row["intent"]), row["task_goal"],
                cap=settings.action_cap, seed=seed, initial_memory=memory, policy_factory=policy_factory,
            )
            attempt = {**outcome, "index": index, "seed": seed}
            if outcome["status"] not in COMPLETE_STATUSES:
                write_json(output / "interruptions" / identifier / f"{index}-{uuid4().hex}.json", attempt)
                return
            write_json(output / "attempts" / identifier / f"{index}.json", attempt)
            completed[identifier][index] = attempt


def measure(
    data: Path, output: Path, *, settings: MeasurementSettings, controller: dict[str, Any],
    policy_factory: PolicyFactory,
    initial_memory: Sequence[ActionAttempt] = (), resume: bool = False,
    candidate_check: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Freeze a pool, run independent attempts, and publish checked V2 measurements."""
    memory = tuple(initial_memory)
    frozen, pool, planned = _freeze(data, settings, controller, memory, candidate_check)
    output.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(output / ".measurement.lock", timeout=0):
            pool_path = output / "pool.json"
            if resume:
                if not pool_path.is_file() or json.loads(pool_path.read_text(encoding="utf-8")) != frozen:
                    raise ValueError("cannot resume a changed frozen pool or controller configuration")
            else:
                if any(path.name != ".measurement.lock" for path in output.iterdir()):
                    raise FileExistsError("measurement output already exists")
                write_json(pool_path, frozen)
            completed = _read_attempts(output, planned, settings)
            try:
                _run_pool(output, pool, completed, planned=planned, settings=settings, memory=memory, policy_factory=policy_factory)
            except BaseException:
                _publish(output, pool, completed, planned=planned, frozen=frozen, settings=settings)
                raise
            return _publish(output, pool, completed, planned=planned, frozen=frozen, settings=settings)
    except Timeout as error:
        raise ValueError("the measurement output is locked by another process") from error
