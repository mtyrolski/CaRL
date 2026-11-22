"""Summarize useful proposals, native reachability, overlap and measured cost."""

import csv
from pathlib import Path
from tempfile import NamedTemporaryFile

from carl.icl.api import write_json
from carl.icl.experiments.intent_pilot_attempts import read
from carl.icl.experiments.intent_pilot_measure import ELIGIBLE
from carl.icl.intents.generation.data import file_sha256
from carl.icl.planning.paired_generator import SCALES
from carl.icl.types import JSONObject


def _cost(calls: list[JSONObject], rolls: list[JSONObject]) -> JSONObject:
    decisions = [decision for rollout in rolls for decision in rollout["decisions"]]
    completions = [decision for decision in decisions if "input_tokens" in decision and not decision.get("cache_hit")]
    return {
        "generator_dispatched": len(calls), "generator_calls": sum(call["status"] == "finished" for call in calls),
        "generator_input_tokens": sum(call.get("input_tokens", 0) for call in calls),
        "generator_output_tokens": sum(call.get("output_tokens", 0) for call in calls),
        "policy_logical_calls": len(decisions), "policy_calls": len(completions),
        "policy_cache_hits": sum(bool(decision.get("cache_hit")) for decision in decisions),
        "policy_input_tokens": sum(decision["input_tokens"] for decision in completions),
        "policy_output_tokens": sum(decision["output_tokens"] for decision in completions),
        "incomplete_cost_records": sum(call["status"] == "running" for call in calls)
        + sum("input_tokens" not in decision for decision in decisions),
    }


def _measurement(slots: list[JSONObject], rolls: list[JSONObject], trials_per_slot: int) -> JSONObject:
    eligible = [r for r in rolls if r["status"] in ELIGIBLE]
    hits = [r["first_hit"] for r in eligible if r["status"] == "intent_reached"]
    valid = sum(slot["status"] == "valid" for slot in slots)
    return {
        "submitted_slots": len(slots), "valid_n": valid, "validity": valid / len(slots) if slots else None,
        "duplicate_n": sum(slot["status"] == "duplicate" for slot in slots),
        "invalid_or_missing_n": sum(slot["status"] in ("invalid", "missing") for slot in slots),
        "eligible_n": len(eligible), "successes": len(hits), "success_probability": len(hits) / len(eligible) if eligible else None,
        "first_hits": hits, "mean_first_hit": sum(hits) / len(hits) if hits else None,
        "censored_n": len(rolls) - len(eligible),
        "unstarted_n": valid * trials_per_slot - len(rolls),
    }


def _overlap(calls: list[JSONObject], rolls: list[JSONObject]) -> JSONObject:
    complete = {(row["case_id"], row["arm"]): row for row in calls if row["status"] == "finished"}
    overlap, crossing = [], []
    for (case_id, arm), relative in complete.items():
        if arm != "relative":
            continue
        unlabelled = complete.get((case_id, "unlabelled"))
        if unlabelled is not None:
            relative_effects = {slot["effect_id"] for slot in relative["slots"] if "effect_id" in slot}
            unlabelled_effects = {slot["effect_id"] for slot in unlabelled["slots"] if "effect_id" in slot}
            overlap.append({"case_id": case_id, "common_effect_ids": sorted(relative_effects & unlabelled_effects)})
        if not all(slot["status"] == "valid" for slot in relative["slots"]):
            continue
        hits = [[row["first_hit"] for row in rolls if row["case_id"] == case_id and row["arm"] == "relative"
                 and row["index"] == index and row["status"] == "intent_reached"] for index in range(len(SCALES))]
        if all(hits):
            lower, upper = (sum(values) / len(values) for values in hits)
            crossing.append({"case_id": case_id, "mean_at_025": lower, "mean_at_075": upper,
                             "ordered": upper > lower})
    return {
        "within_call_overlaps": [dict(case_id=case_id, arm=arm) for (case_id, arm), row in complete.items() if row["within_overlap"]],
        "between_arms": overlap, "successful_distinct_scale_pairs": crossing,
    }


def publish(root: Path, *, edition: int = 1) -> JSONObject:
    directory = root / ("generator" if edition == 1 else f"generator-v{edition}")
    plan_path = directory / "private/plan.json"
    plan = read(plan_path)
    config = file_sha256(plan_path)
    cells = {}
    table = []
    for name, condition in plan["cells"].items():
        calls = [read(path) for path in sorted((directory / "private/calls" / name).glob("*.json"))]
        rolls = [read(path) for path in sorted((directory / "private/rollouts" / name).glob("*.json"))]
        if any(row["config_sha256"] != config or row["cell"] != name for row in [*calls, *rolls]):
            raise ValueError("A validation record has the wrong frozen profile")
        arms = []
        for arm, requested in (("unlabelled", None), *(("relative", u) for u in SCALES)):
            arm_calls = [row for row in calls if row["arm"] == arm]
            slots = [slot for row in arm_calls for slot in row.get("slots", []) if slot["requested_u"] == requested]
            arm_rolls = [row for row in rolls if row["arm"] == arm and
                         (requested is None or SCALES[row["index"]] == requested)]
            stats = _measurement(slots, arm_rolls, 2 if condition["domain"] == "craftax" else 1)
            record = {"arm": arm, "requested_u": requested, "status": condition["arms"][arm], **stats}
            table.append({"controller": condition["model_id"], "domain": condition["domain"],
                          **{k: v for k, v in record.items() if k != "first_hits"}})
            arms.append(record)
        cells[name] = {"arms": arms, "reached_support": condition["reached_candidates"],
                       "ranked_support": condition["ranked_candidates"], "cost": _cost(calls, rolls),
                       "overlap": _overlap(calls, rolls)}
    result = {"config_sha256": config, "values": plan["values"], "cells": cells}
    write_json(directory / "results.json", result)
    with NamedTemporaryFile(dir=directory, mode="w", encoding="utf-8", newline="", delete=False) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
        temporary = Path(stream.name)
    temporary.replace(directory / "cells.csv")
    lines = ["# Prompt 06 matched generator validation", "", f"Frozen configuration `{config}`.", "",
             "| Controller | Domain | Arm | Requested u | Status | Valid / slots | Successes / eligible | Mean first hit |",
             "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for row in table:
        mean = f"{row['mean_first_hit']:.3f}" if row["mean_first_hit"] is not None else "—"
        valid = f"{row['valid_n']}/{row['submitted_slots']}" if row["submitted_slots"] else "—"
        hits = f"{row['successes']}/{row['eligible_n']}" if row["eligible_n"] else "—"
        lines.append(f"| {row['controller']} | {row['domain']} | {row['arm']} | "
                     f"{row['requested_u'] if row['requested_u'] is not None else '—'} | {row['status']} | "
                     f"{valid} | {hits} | {mean} |")
    lines.extend(["", "Reported costs and overlap are in `results.json`. Raw prompts and simulator traces are under `private/`.",
                  "The fitted Sokoban value uses training demonstration returns. Craftax's configured fallback is the shared constant V=0.5."])
    with NamedTemporaryFile(dir=directory, mode="w", encoding="utf-8", delete=False) as stream:
        stream.write("\n".join(lines) + "\n")
        temporary = Path(stream.name)
    temporary.replace(directory / "RESULTS.md")
    return result
