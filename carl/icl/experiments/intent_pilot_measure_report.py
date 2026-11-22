"""Publish sectioned D_C records without dropping missing or censored trials."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from carl.icl.api import write_json
from carl.icl.experiments.intent_pilot_attempts import DOMAINS, MODELS, read
from carl.icl.experiments.intent_pilot_measure import ELIGIBLE, INITIAL_MEMORY, _records, _slot, load_plan
from carl.icl.intents.generation.data import file_sha256
from carl.icl.planning.active_intent import digest
from carl.icl.policy_relative.measurements import directed_labels
from carl.icl.types import JSONObject


def _jsonl(path: Path, records: list[JSONObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        temp = Path(stream.name)
        try:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        except BaseException:
            temp.unlink()
            raise
    temp.replace(path)


def _headline(directory: Path, rows: list[JSONObject], pool_hash: str, *, final: bool) -> None:
    lines = ["# Prompt 05 directed native measurements", "", f"Frozen measurement pool `{pool_hash}`.", "",
             "| Controller | Domain | Eligible / planned | Successes | Censored | Informative contexts |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['domain']} | {row['eligible_n']}/{row['planned_n']} | "
                     f"{row['successes']} | {row['censored_n']} | {row.get('informative_contexts', '—')} |")
    lines += ["", "Incomplete contexts and pools with fewer than two distinct successful durations carry no comparative u.",
              "The per-cell attempts retain original failures, in-flight work and undispatched slots."]
    if not final:
        lines.append("This is a progress report from the bounded run.")
    if final:
        with (directory / "cells.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    path = directory / "RESULTS.md"
    with NamedTemporaryFile(dir=directory, mode="w", encoding="utf-8", delete=False) as stream:
        stream.write("\n".join(lines) + "\n")
        temp = Path(stream.name)
    temp.replace(path)


def _observed_root(domain: str, case: JSONObject, native: JSONObject) -> tuple[object, str | None, list[str | None]]:
    if domain == "sokoban":
        board = case["start_observation"]
        identifier = digest(board)
        return board, identifier, [identifier]
    root = native.get(case["context_id"])
    if root is None:
        return None, None, [None] * len(case["future_streams"])
    if len(set(root["future_ids"])) != len(case["future_streams"]):
        raise ValueError("A native measurement context repeats a future identity")
    return root["observation"], root["physical_start_id"], root["future_ids"]


def publish(root: Path, *, progress_only: bool = False) -> JSONObject:
    plan, pool_hash = load_plan(root)
    directory = root / "measurements"
    path = directory / "private/native-starts.json"
    native = read(path) if path.is_file() else {}
    records: dict[str, list[JSONObject]] = defaultdict(list)
    for record in _records(directory):
        records[record["slot_id"]].append(record)
    recognized = set()
    cells = []
    for model in MODELS:
        policy = plan["policies"][model]
        for domain in DOMAINS:
            rows: list[JSONObject] = []
            attempts: list[JSONObject] = []
            membership: list[JSONObject] = []
            output = directory / model / domain
            provenance = {**plan["inputs"], "policy_sha256": policy["sha256"], "pool_sha256": pool_hash}
            for case in plan["contexts"][domain]:
                observation, physical, future_ids = _observed_root(domain, case, native)
                candidates, groups = case["candidates"], []
                for candidate in candidates:
                    trial_group: list[JSONObject] = []
                    for index, future in enumerate(future_ids):
                        slot = _slot(model, domain, case, candidate, index)
                        recognized.add(slot["slot_id"])
                        trials = records.get(slot["slot_id"], [])
                        for trial in trials:
                            if (any(trial[key] != value for key, value in slot.items())
                                    or trial["pool_sha256"] != pool_hash or trial["policy_sha256"] != policy["sha256"]
                                    or trial["requested_h"] != candidate["intent"]
                                    or trial["physical_start_id"] != physical or trial["future_id"] != future
                                    or trial["memory_id"] != digest(INITIAL_MEMORY)
                                    or any(call["query"]["assigned_intent"] != trial["requested_h"] for call in trial["decisions"])):
                                raise ValueError("Attempt does not match its frozen candidate, policy or future")
                        trial_group.extend(trials)
                        counted = False
                        for trial in trials:
                            is_eligible = trial["status"] in ELIGIBLE
                            attempts.append({**trial, "eligible": is_eligible and not counted,
                                             "censored": not is_eligible, "duplicate_future": is_eligible and counted})
                            counted = counted or is_eligible
                        if not trials:
                            attempts.append({**slot, "status": "not_started", "eligible": False, "censored": False,
                                             "future_id": future, "physical_start_id": physical, "first_hit": None})
                    groups.append(trial_group)
                n = len(future_ids)
                labels = directed_labels(groups, [n] * len(groups))
                membership.append({"context_id": case["context_id"], "source_id": case["source_id"], "group_id": case["group_id"],
                                   "physical_start_id": physical, "future_ids": future_ids, "future_streams": case["future_streams"],
                                   "memory_id": digest(INITIAL_MEMORY), "candidates": candidates})
                for candidate, stats in zip(candidates, labels, strict=True):
                    rows.append({
                        "schema_version": "carl-pilot-directed-v1", "split": "train", "pilot_split": "train-measurement",
                        "context_id": case["context_id"], "task_group": case["group_id"], "start_observation": observation,
                        "task_goal": case["task_goal"], **candidate, "record_id": candidate["candidate_id"],
                        "measurement_provenance": provenance, "policy_measurements": stats,
                    })
            outcomes = Counter(trial["status"] for trial in attempts)
            eligible = sum(row["policy_measurements"]["eligible_n"] for row in rows)
            planned = sum(row["policy_measurements"]["planned_n"] for row in rows)
            successes = sum(row["policy_measurements"]["successes"] for row in rows)
            informative = {row["context_id"] for row in rows if row["policy_measurements"]["scale_supervision"]}
            summary = {
                "model": policy["contract"]["model_id"], "domain": domain,
                "planned_n": planned, "eligible_n": eligible, "successes": successes,
                "censored_n": sum(trial["censored"] for trial in attempts), "unstarted_n": outcomes["not_started"],
                "complete_contexts": len({row["context_id"] for row in rows if row["policy_measurements"]["context_complete"]}),
                "informative_contexts": len(informative), "ranked_candidates": sum(row["policy_measurements"]["scale_supervision"] for row in rows),
                "status": "complete" if eligible == planned else "incomplete",
            }
            cells.append(summary)
            if progress_only:
                continue
            _jsonl(output / "attempts.jsonl", attempts)
            _jsonl(output / "D_C.jsonl", rows)
            write_json(output / "pool.json", {"provenance": provenance, "type_quotas": plan["type_quotas"][domain],
                                                "controller": policy["contract"], "contexts": membership})
            write_json(output / "manifest.json", {
                **summary, "outcomes": dict(outcomes), "provenance": provenance,
                "sha256": {name: file_sha256(output / name) for name in ("attempts.jsonl", "D_C.jsonl", "pool.json")},
            })
    if set(records) - recognized:
        raise ValueError("An attempt refers to an unplanned trial slot")
    result = {"pool_sha256": pool_hash, "planned_n": plan["planned_trials"], "cells": cells}
    _headline(directory, cells, pool_hash, final=not progress_only)
    if not progress_only:
        write_json(directory / "manifest.json", result)
    return result
