"""Blind failed prefixes, verify Codex labels, and freeze training candidates."""

import csv
import io
from collections import Counter
from pathlib import Path

from carl.icl.experiments.intent_pilot_annotations import (
    PROMPT, PROMPT_SHA256, _craftax_audit, _emit, _jsonl, _packet, _response, _rows, _sokoban_audit,
)
from carl.icl.experiments.intent_pilot_attempts import DOMAINS, MODELS, freeze, plan, read
from carl.icl.intent_annotations import UnlabelledAnnotation
from carl.icl.intent_domain import IntentCheck, SokobanObservation, annotation_json, check_intent, decode_intent
from carl.icl.intents.generation.data import file_sha256, read_records
from carl.icl.planning.active_intent import INTENT_CAP, digest
from carl.icl.planning.craftax_tasks import CraftaxIntent, predicate_holds
from carl.icl.types import JSONObject


POOL = "candidate-pool-v1.json"


def _progress(domain: str, first: JSONObject | list[str], current: JSONObject | list[str]) -> bool:
    if domain == "craftax":
        assert isinstance(first, dict) and isinstance(current, dict)
        return any(amount > first["inventory"].get(item, 0) for item, amount in current["inventory"].items())
    assert isinstance(first, list) and isinstance(current, list)
    return any(tile in "$*" and old not in "$*" for before, after in zip(first, current, strict=True)
               for old, tile in zip(before, after, strict=True))


def _prefix(record: JSONObject, observations: list) -> int | None:
    if record["status"] not in ("intent_timeout", "budget_cutoff"):
        return None
    return next((index for index, step in enumerate(record["steps"], 1) if step["executed_primitives"] and
                 _progress(record["domain"], observations[0], observations[index])), None)


def _attempts(root: Path, split: str) -> list[tuple[JSONObject, JSONObject]]:
    frozen = plan(root, split)
    directory = root / "hindsight/private"
    fixtures = directory / "fixtures"
    expected = [(domain, model, f"{model}-{domain}") for domain in DOMAINS for model in MODELS]
    expected.extend((domain, "scripted_train", f"scripted-{domain}") for domain in DOMAINS)
    attempts = []
    for domain, model, name in expected:
        path = directory / "attempts" / f"{name}.json"
        record = read(path)
        replay = read(directory / "replays" / path.name)
        kind = "scripted_train" if model == "scripted_train" else "local"
        context = frozen["contexts"][domain][kind]
        if (record["attempt_id"] != name or record["domain"] != domain or record["pilot_split"] != "train-support"
                or record["source_episode_id"] != context["episode_id"] or record["group_id"] != context["group_id"]
                or record["world_id"] != context.get("world_id") or record["public_task"] != context["task_goal"]
                or _semantic(record["requested_h"]) != _semantic(context["requested_h"])
                or record["mode"] != "live" or not record["training_eligible"] or record["provenance"]["fixture"]):
            raise ValueError(f"Attempt {name} is outside its frozen training context")
        expected_kind = "scripted_train_support" if kind == "scripted_train" else "real_local_policy"
        if record["provenance"]["kind"] != expected_kind or replay["attempt_sha256"] != file_sha256(path):
            raise ValueError(f"Attempt {name} does not match its original provenance and replay")
        if kind == "local":
            config = read(root / "direct" / f"{model}.json")
            provenance = record["provenance"]
            if provenance["model_id"] != config["model_id"] or provenance["model"]["revision"] != config["local"]["revision"]:
                raise ValueError(f"Attempt {name} used different weights from the frozen model")
        if replay["observations"] != [state["observation"] for state in record["states"]]:
            raise ValueError(f"Attempt {name} observations disagree with native replay")
        if (any(call["query"]["assigned_intent"] != record["requested_h"] for call in record["decisions"])
                or any(digest(state["memory"]) != state["memory_id"] for state in [*record["states"], record["continuation"]])):
            raise ValueError(f"Attempt {name} has inconsistent intent or memory evidence")
        attempts.append((record, replay))
    for domain in DOMAINS:
        fixture = read(fixtures / f"{domain}.json")
        origin = next(record for record, _ in attempts if record["attempt_id"] == fixture["source_attempt"])
        branch, replan = fixture["speculative_attempt"], fixture["live_replan"]
        if (fixture["classification"] != "integration_only" or fixture["live_timeout_status"] != "intent_timeout"
                or origin["domain"] != domain or origin["status"] != "intent_timeout" or origin["executed_primitives"] != INTENT_CAP
                or branch["status"] != "intent_timeout" or branch["executed_primitives"] != INTENT_CAP
                or branch["continuation"] != branch["states"][0] or branch["states"][0] != origin["states"][0]
                or branch["budget_before"]["actions"] - branch["budget_after"]["actions"] != INTENT_CAP
                or replan["states"][0] != origin["continuation"] or replan["budget_before"] != origin["budget_after"]
                or replan["status"] not in ("intent_reached", "task_solved")):
            raise ValueError(f"Missing verified timeout and replan fixture for {domain}")
    return attempts


def _index(attempts: list[tuple[JSONObject, JSONObject]]) -> tuple[list[JSONObject], list[JSONObject]]:
    unique: dict[str, JSONObject] = {}
    index = []
    for record, replay in attempts:
        domain = record["domain"]
        observations = replay["observations"]
        end = _prefix(record, observations)
        if end is None:
            continue
        packet = _packet(domain, record["public_task"], observations[0], observations[end])
        unique.setdefault(packet["packet_id"], packet)
        index.append({"packet_id": packet["packet_id"], "episode_id": record["attempt_id"], "start": 0, "end": end})
    return list(unique.values()), index


def packets(root: Path, split: str) -> JSONObject:
    """Expose only endpoint observations, the public task, and domain vocabulary."""
    directory = root / "hindsight"
    attempts = _attempts(root, split)
    blinded, index = _index(attempts)
    _emit(directory / "annotation-prompt.txt", PROMPT)
    _jsonl(directory / "packets.jsonl", blinded)
    _jsonl(directory / "private/packet-index.jsonl", index)
    result = {
        "packets": len(blinded), "prefixes": len(index), "by_domain": dict(Counter(row["domain"] for row in blinded)),
        "packet_sha256": file_sha256(directory / "packets.jsonl"), "annotation_prompt_sha256": PROMPT_SHA256,
    }
    freeze(directory / "packets-manifest.json", result)
    return result


def _semantic(intent: JSONObject) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(map(annotation_json, intent.get(field, [])))) for field in ("effects", "preserve"))


def _full_prefix(domain: str, answer: JSONObject, observations: list, steps: list) -> IntentCheck:
    """Reject a prefix whose later ticks lose preservation after an earlier hit."""
    if domain == "sokoban":
        intent = decode_intent(answer)
        checked = check_intent(intent, [SokobanObservation(tuple(frame)) for frame in observations])
        if checked.status != "accepted":
            return checked
    else:
        native_intent = CraftaxIntent.model_validate(answer)
        unknown = False
        for index, observation in enumerate(observations):
            holds = [predicate_holds(condition, observation["inventory"]) for condition in native_intent.preserve]
            if False in holds:
                return IntentCheck("rejected", (f"preservation false at prefix state {index}",))
            unknown = unknown or None in holds
        if unknown:
            return IntentCheck("unknown", ("preservation unknown within the selected prefix",))
    if any(step["native_ticks"] != 1 for step in steps):
        return IntentCheck("unknown", ("missing per-tick evidence anywhere in the selected prefix",))
    return IntentCheck("accepted")


def _verified_row(
    directory: Path, record: JSONObject, replay: JSONObject, ref: JSONObject, answer: JSONObject | UnlabelledAnnotation,
) -> JSONObject:
    name = record["attempt_id"]
    domain = record["domain"]
    observations = replay["observations"]
    trace = {
        "episode_id": name, "pilot_split": "train-support", "group_id": record["group_id"],
        "task_goal": record["public_task"], "observations": observations, "actions": replay["actions"],
        "trajectory_provenance": record["provenance"],
        "snapshots": [{key: state[key] for key in ("state_id", "memory_id")} for state in record["states"]],
    }
    end = ref["end"]
    answers = {ref["packet_id"]: answer}
    if domain == "sokoban":
        trace["source"] = {"source_id": name, "split": "train", "task_group": record["group_id"],
                           "start_observation": observations[0], "actions": replay["actions"]}
        row, = _sokoban_audit(directory, {name: trace}, [ref], answers)
    else:
        trace.update(
            world_id=record["world_id"], facts=[state["inventory"] for state in observations],
            steps=replay["steps"],
            snapshots=[{**native, "memory_id": state["memory_id"]}
                       for native, state in zip(replay["native_states"], record["states"], strict=True)],
        )
        row, = _craftax_audit({name: trace}, [ref], answers)
    if row["validation"]["accepted"]:
        full = _full_prefix(domain, row["intent"], observations[:end + 1], record["steps"][:end])
        if _semantic(row["intent"]) == _semantic(record["requested_h"]):
            full = IntentCheck("rejected", ("the proposed label is identical to the failed request",))
        if full.status != "accepted":
            row["validation"].update(accepted=False, status=full.status, reasons=list(full.errors))
    row["lineage"] = {
        "source_attempt_id": name, "source_attempt_sha256": replay["attempt_sha256"],
        "source_episode_id": record["source_episode_id"], "requested_h": record["requested_h"],
        "source_status": record["status"], "source_budget_reason": record["budget_reason"],
        "source_primitive_count": record["executed_primitives"], "prefix_native_transitions": end,
        "duration_kind": "observed_under_original_request", "directed_duration_available": False,
    }
    if row["validation"]["accepted"]:
        row["lineage"]["observed_first_hit_primitives"] = row["demonstration_offset"]
    return row


def _checked_outputs(root: Path, split: str) -> tuple[JSONObject, list[JSONObject], dict[str, list[JSONObject]]]:
    directory = root / "hindsight"
    attempts = _attempts(root, split)
    blinded, index = _index(attempts)
    if blinded != _rows(directory / "packets.jsonl") or index != _rows(directory / "private/packet-index.jsonl"):
        raise ValueError("Blinded packets differ from the native prefixes")
    if (directory / "annotation-prompt.txt").read_text(encoding="utf-8") != PROMPT:
        raise ValueError("The annotation instruction differs from the frozen protocol")
    raw = directory / "annotations.raw.jsonl"
    answers = {packet["packet_id"]: _response(line, packet["packet_id"]) for packet, line in
               zip(blinded, raw.read_text(encoding="utf-8").splitlines(), strict=True)}
    base = {domain: _rows(root / "annotations" / domain / "D_A.jsonl") for domain in DOMAINS}
    seen = {domain: {row["record_id"] for row in rows} for domain, rows in base.items()}
    verified: dict[str, list[JSONObject]] = {domain: [] for domain in DOMAINS}
    audit = []
    by_id = {record["attempt_id"]: (record, replay) for record, replay in attempts}
    for ref in index:
        record, replay = by_id[ref["episode_id"]]
        domain = record["domain"]
        row = _verified_row(directory, record, replay, ref, answers[ref["packet_id"]])
        audit.append({"domain": domain, **row})
        if row["validation"]["accepted"]:
            key = row["record_id"]
            if key in seen[domain]:
                row["validation"].update(accepted=False, status="duplicate", reasons=["intent already in checked D_A"],
                                         duplicate_of=key)
            else:
                seen[domain].add(key)
                verified[domain].append({key: value for key, value in row.items() if key != "evidence"})
    pool = {
        "schema": "carl-hindsight-candidate-pool-v1", "version": 1, "base_split_sha256": split,
        "plan_sha256": file_sha256(directory / "private/plan.json"),
        "packets_sha256": file_sha256(directory / "packets.jsonl"), "annotations_sha256": file_sha256(raw),
        "annotation_prompt_sha256": PROMPT_SHA256,
        "attempt_sha256": {record["attempt_id"]: replay["attempt_sha256"] for record, replay in attempts},
        "replay_sha256": {record["attempt_id"]: file_sha256(directory / "private/replays" / f"{record['attempt_id']}.json")
                          for record, _ in attempts},
        "base_da_sha256": {domain: file_sha256(root / "annotations" / domain / "D_A.jsonl") for domain in DOMAINS},
        "candidates": {domain: [
            {
                "candidate_id": f"{domain}:{row['record_id']}", "support_context_id": row["context_id"],
                "group_id": row["independent_group_id"], "intent": row["intent"], "source_kind": (
                    "verified_hindsight" if "lineage" in row else "checked_demonstration"
                ), **({"lineage": row["lineage"]} if "lineage" in row else {}),
            } for row in [*base[domain], *verified[domain]] if row["pilot_split"] == "train-support"
        ] for domain in DOMAINS},
    }
    return pool, audit, {domain: [*base[domain], *verified[domain]] for domain in DOMAINS}


def _report(directory: Path, attempts: list[tuple[JSONObject, JSONObject]], audit: list[JSONObject], summary: JSONObject) -> None:
    data = io.StringIO()
    writer = csv.writer(data, lineterminator="\n")
    writer.writerow(("attempt_id", "domain", "source", "status", "budget_reason", "executed_primitives", "first_hit"))
    for record, _ in attempts:
        writer.writerow([record[key] for key in ("attempt_id", "domain")] + [record["provenance"]["kind"]]
                        + [record[key] for key in ("status", "budget_reason", "executed_primitives", "first_hit")])
    _emit(directory / "attempts.csv", data.getvalue())
    lines = [
        "# Prompt 04 assigned intents and hindsight", "",
        "| Attempt | Source | Status | Executed primitives | First hit |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for record, _ in attempts:
        reason = f" ({record['budget_reason']})" if record["budget_reason"] else ""
        hit = record["first_hit"] if record["first_hit"] is not None else "—"
        lines.append(f"| {record['attempt_id']} | {record['provenance']['kind']} | {record['status']}{reason} | "
                     f"{record['executed_primitives']} | {hit} |")
    lines += ["", f"Pool SHA-256 `{summary['pool_sha256']}`", "", "## Accepted hindsight labels", ""]
    for row in audit:
        if row["validation"]["accepted"]:
            origin = row["lineage"]
            lines.append(f"- `{origin['source_attempt_id']}` yielded `{row['record_id']}` at "
                         f"{origin['observed_first_hit_primitives']} observed primitives under the original request.")
    lines.extend(["", "The fixtures directory records the successful live replans and speculative checkpoint restores.",
                  "The candidate pool carries training support and lineage. Directed duration remains unmeasured."])
    report = directory / "RESULTS.md"
    temporary = report.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(report)


def build(root: Path, split: str, verify_only: bool) -> JSONObject:
    """Rebuild the pool byte-for-byte and validate each new dataset."""
    directory = root / "hindsight"
    if verify_only and not (directory / POOL).is_file():
        raise ValueError("Freeze a candidate pool before verifying it")
    pool, audit, datasets = _checked_outputs(root, split)
    files = [directory / POOL, directory / "private/annotation_audit.jsonl", directory / "pool-manifest.json"]
    files += [directory / domain / "D_A.jsonl" for domain in DOMAINS]
    if verify_only and any(not path.is_file() for path in files):
        raise ValueError("A versioned hindsight output is missing")
    _jsonl(directory / "private/annotation_audit.jsonl", audit)
    for domain, rows in datasets.items():
        output = directory / domain
        output.mkdir(exist_ok=True)
        path = output / "D_A.jsonl"
        _jsonl(path, rows)
        if domain == "sokoban":
            read_records(path)
        else:
            for row in rows:
                CraftaxIntent.model_validate(row["intent"])
    freeze(directory / POOL, pool)
    summary = {
        "pool_version": pool["version"], "pool_sha256": file_sha256(directory / POOL), "base_split_sha256": split,
        "candidates": {domain: len(rows) for domain, rows in pool["candidates"].items()},
        "hindsight_outcomes": {
            domain: dict(Counter(row["validation"]["status"] for row in audit if row["domain"] == domain))
            for domain in DOMAINS
        },
        "da_sha256": {domain: file_sha256(directory / domain / "D_A.jsonl") for domain in DOMAINS},
        "audit_sha256": file_sha256(directory / "private/annotation_audit.jsonl"),
    }
    freeze(directory / "pool-manifest.json", summary)
    _report(directory, _attempts(root, split), audit, summary)
    return summary
