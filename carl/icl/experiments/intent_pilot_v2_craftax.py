"""Certify and measure a frozen native Craftax context for the follow-up pilot."""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from carl.environment.craftax import CraftaxAdapter, CraftaxSnapshot
from carl.icl.api import write_json
from carl.icl.config import LocalModelSettings
from carl.icl.experiments.intent_pilot_annotation_sources import _advance
from carl.icl.experiments.intent_pilot_attempts import freeze, read
from carl.icl.experiments.intent_pilot_measure import ELIGIBLE, INITIAL_MEMORY, LocalDecider, _records, _run_slot, load_plan
from carl.icl.experiments.intent_pilot_measure_report import _jsonl
from carl.icl.experiments.intent_pilot_v2 import PROFILE, V1, V2, jsonl, seed
from carl.icl.experiments.intent_pilot_v2_run import V2Ledger
from carl.icl.intents.generation.data import file_sha256
from carl.icl.local_models import bind_local_role
from carl.icl.planning.active_intent import INTENT_CAP, digest
from carl.icl.planning.craftax_tasks import CraftaxIntent, CraftaxTask, conjunction, render_instruction
from carl.icl.planning.local_direct import CraftaxDirect
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.policy_relative.measurements import directed_labels
from carl.icl.types import JSONObject


DIRECTORY = V2 / "craftax"
PLAN = DIRECTORY / "private/native-plan.json"
WITNESSES = DIRECTORY / "private/native-witnesses.json"
ARM = "directed"
SCHEMA = "intent-pilot-v2-craftax-directed-v1"
NATIVE_ACTIVITY = "craftax-native-certify"
MEASURE_ACTIVITY = "craftax-directed-block"


def _evidence_row(row: JSONObject, label: str, *, direct: bool) -> JSONObject:
    result = row["result"] if direct else row
    decisions = result["decisions"]
    steps = [step for decision in decisions for step in decision["steps"]] if direct else row["steps"]
    query = decisions[0]["query"]
    return {
        "label": label, "id": row["case_id"] if direct else row["slot_id"],
        "status": result["stop"] if direct else row["status"],
        "observation": query["observation"], "public_task": query["public_task"],
        "requested_h": query["public_task"] if direct else query["assigned_intent"],
        "first_response": decisions[0]["response"], "first_transition": steps[0],
        "executed_actions": [step["applied"] for step in steps if step["executed"]],
        "primitives": result["executed_primitives"],
        "ticks": result["native_ticks"] if direct else sum(step["native_ticks"] for step in steps),
        "first_hit": result["first_verified_hit_primitives"] if direct else row["first_hit"],
    }


def _historical_evidence(v1_plan: JSONObject) -> JSONObject:
    direct_path = V1 / f"direct/{PROFILE}-results.json"
    measurement_path = V1 / f"measurements/{PROFILE}/craftax/attempts.jsonl"
    direct = [_evidence_row(row, row["skill"], direct=True)
              for row in read(direct_path)["domains"]["craftax"]["records"]]
    sword_ids = {candidate["candidate_id"] for case in v1_plan["contexts"]["craftax"] for candidate in case["candidates"]
                 if candidate["type"] == "tool"}
    calibration = [_evidence_row(row, "directed_sword", direct=False) for row in jsonl(measurement_path)
                   if row["candidate_id"] in sword_ids and row["status"] in ELIGIBLE]
    if len(direct) != 4 or len(calibration) != 4:
        raise ValueError("The v1 evidence does not match the closed run's coverage")
    return {"inputs": {str(path): file_sha256(path) for path in (direct_path, measurement_path)},
            "direct": direct, "calibration": calibration}


def prepare() -> JSONObject:
    splits = V2 / "splits.json"
    group, = (row for row in read(splits)["groups"] if row["domain"] == "craftax" and row["partition"] == "calibration")
    task = CraftaxTask.model_validate(group["task"])
    if group["group_id"] != contract_hash(task.chain_signature) or group["seed"] != task.seed:
        raise ValueError("The Craftax task does not match its frozen split identity")
    v1_plan, _ = load_plan(V1)
    context_id = contract_hash(["craftax", group["world_id"], task.goal.model_dump(mode="json")])
    preserve = [{"predicate": "tool_tier_at_least", "tool": "pickaxe", "tier": 2}]
    witnesses = {"stone": ["DO"], "wood": ["UP", "DO"]}
    candidates = []
    for stage in task.chain:
        effect, = stage.effects
        predicate = effect.model_dump(mode="json")
        resource = predicate["item"]
        intent = CraftaxIntent.model_validate({"effects": [predicate], "preserve": preserve}).model_dump(mode="json")
        candidates.append({"candidate_id": contract_hash([context_id, intent]), "intent": intent,
                           "request_id": resource, "witness_actions": witnesses[resource]})
    if len(candidates) != 2 or {candidate["request_id"] for candidate in candidates} != set(witnesses):
        raise ValueError("Expected the prespecified stone and wood requests")
    times = read(V2 / "forecast.json")
    native = times["craftax"]["previous_setup_seconds"]
    model_setup = times["observed_cold_seconds"]
    warm = times["previous_regime"]["craftax"]["warm_median_seconds"]
    median_forecast = native + model_setup + 4 * INTENT_CAP * warm
    forecast = {"native_setup_reference_seconds": native, "model_setup_reference_seconds": model_setup,
                "historical_decision_median_seconds": warm, "median_full_block_seconds": median_forecast,
                "native_deadline_seconds": math.ceil(native + 15), "block_deadline_seconds": math.ceil(median_forecast + 15)}
    case = {**group, "context_id": context_id, "native_task": group["task"], "task_goal": render_instruction(task.goal),
            "candidates": candidates, "future_streams": [seed("craftax-v2-future", context_id, i) for i in range(2)]}
    preparation_code = _plan()["code_sha256"] if PLAN.is_file() else file_sha256(Path(__file__))
    plan = {"schema": SCHEMA, "case": case, "policy": v1_plan["policies"][PROFILE], "arm": ARM,
            "forecast": forecast, "primitive_cap": INTENT_CAP, "split_sha256": file_sha256(splits),
            "reference_da_sha256": v1_plan["inputs"]["da_sha256"]["craftax"], "code_sha256": preparation_code}
    evidence = _historical_evidence(v1_plan)
    freeze(PLAN, plan)
    freeze(DIRECTORY / "v1-evidence.json", evidence)
    return report()


def _plan() -> JSONObject:
    plan: JSONObject = read(PLAN)
    if (plan["split_sha256"] != file_sha256(V2 / "splits.json") or plan["schema"] != SCHEMA
            or plan["primitive_cap"] != INTENT_CAP or plan["arm"] != ARM):
        raise ValueError("Frozen Craftax split or trial protocol changed")
    return plan


def _job_deadline(phase: str, activity: str) -> float:
    active = [job for job in V2Ledger(V1).status()["jobs"] if job["prompt"] == 18 and job["domain"] == "craftax"
              and job["phase"] == phase and job["activity"] == activity and job["status"] == "reserved"]
    if len(active) != 1 or active[0]["workers"] != 1:
        raise ValueError("This operation requires its single-worker reservation in the shared v2 ledger")
    return time.monotonic() + int(active[0]["reserved_ms"]) / 1000 - 5


def _native(plan: JSONObject) -> tuple[CraftaxAdapter, CraftaxSnapshot]:
    case = plan["case"]
    task = CraftaxTask.model_validate(case["native_task"])
    adapter = CraftaxAdapter(map_size=case["map_size"], max_steps=case["max_steps"])
    root = adapter.stage(adapter.reset(task.seed), **task.scenario.model_dump())
    streams = case["future_streams"]
    forks = [adapter.fork_future(root, stream) for stream in streams]
    if (len({fork.fingerprint for fork in forks}) != 2
            or any(fork.physical_fingerprint != root.physical_fingerprint for fork in forks)):
        raise ValueError("The planned future keys must differ and share one physical start")
    records = []
    for candidate in case["candidates"]:
        intent = CraftaxIntent.model_validate(candidate["intent"])
        for stream, fork in zip(streams, forks, strict=True):
            observations, states, steps = _advance(adapter, fork, candidate["witness_actions"])
            effects = [conjunction(intent.effects, observation["inventory"]) for observation in observations]
            preservation = [conjunction(intent.preserve, observation["inventory"]) for observation in observations]
            if (effects != [False] * len(steps) + [True]
                    or any(value is not True for value in preservation)
                    or any(not s["executed_primitive"] or s["native_ticks"] != 1 or s["automatic_ticks"]
                           or s["terminated"] or s["truncated"] for s in steps)):
                raise ValueError(f"Uncertified native witness for {candidate['request_id']} on stream {stream}")
            records.append({"candidate_id": candidate["candidate_id"], "future_id": fork.fingerprint, "stream": stream,
                            "observations": observations, "states": states, "steps": steps, "effects": effects,
                            "preservation": preservation, "first_hit_primitives": len(steps)})
    visible = CraftaxDirect(adapter, root, task.goal).observe()
    freeze(WITNESSES, {"plan_sha256": file_sha256(PLAN), "physical_start_id": root.physical_fingerprint,
                       "initial_memory_id": digest(INITIAL_MEMORY), "future_ids": [fork.fingerprint for fork in forks],
                       "observation": visible, "provenance": "private scripted native verification", "witnesses": records})
    return adapter, root


def certify() -> JSONObject:
    _job_deadline("preparation", NATIVE_ACTIVITY)
    plan = _plan()
    with FileLock(str(DIRECTORY / "private/native.lock"), timeout=0):
        _native(plan)
    return report()


def _slots(plan: JSONObject, candidate: JSONObject) -> list[JSONObject]:
    case = plan["case"]
    profile = plan["policy"]["sha256"]
    return [{"slot_id": contract_hash([profile, case["context_id"], candidate["candidate_id"], ARM, stream]),
             "model": PROFILE, "domain": "craftax", "context_id": case["context_id"], "candidate_id": candidate["candidate_id"],
             "index": index, "stream": stream, "request_id": candidate["request_id"], "arm": ARM,
             "pool_sha256": file_sha256(PLAN), "policy_sha256": profile}
            for index, stream in enumerate(case["future_streams"])]


def _worker(policy: JSONObject) -> LocalDecider:
    import torch

    cfg = read(V1 / f"direct/{PROFILE}.json")
    torch.set_num_threads(policy["contract"]["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(policy["contract"]["policy_seed"])
    role = bind_local_role(LocalModelSettings.model_validate(cfg["local"]), "policy")
    if role.identity() != policy["contract"]["model"]:
        raise ValueError("Cached Craftax policy weights differ from the v1 profile")
    _ = role._stack
    _ = role._tokenizer
    cache = DIRECTORY / "private/cache" / policy["sha256"]
    check = V1 / "measurements/private/cache" / policy["sha256"] / "greedy-check.json"
    freeze(cache / "greedy-check.json", read(check))
    return LocalDecider(role, cache)


def measure() -> JSONObject:
    deadline = _job_deadline("inference", MEASURE_ACTIVITY)
    started = time.monotonic()
    plan = _plan()
    v1_plan, _ = load_plan(V1)
    if plan["policy"] != v1_plan["policies"][PROFILE]:
        raise ValueError("The frozen v1 policy contract changed")
    with FileLock(str(DIRECTORY / "private/native.lock"), timeout=0):
        case = plan["case"]
        complete = {row["slot_id"] for row in _records(DIRECTORY) if row["status"] in ELIGIBLE}
        pending = [(candidate, slot) for candidate in case["candidates"] for slot in _slots(plan, candidate)
                   if slot["slot_id"] not in complete]
        if not pending:
            return report()
        forecast = plan["forecast"]
        minimum = (forecast["native_setup_reference_seconds"] + forecast["model_setup_reference_seconds"]
                   + len(pending) * INTENT_CAP * forecast["historical_decision_median_seconds"])
        if deadline - time.monotonic() < minimum:
            raise ValueError(f"Reserve at least {math.ceil(minimum + 5)} local seconds for the pending Craftax block")
        adapter, root = _native(plan)
        native_ready = time.monotonic()
        decide = _worker(plan["policy"])
        run_id = uuid4().hex
        freeze(DIRECTORY / "private/sessions" / f"{run_id}.json", {
            "profile_sha256": plan["policy"]["sha256"], "native_setup_seconds": native_ready - started,
            "model_setup_seconds": time.monotonic() - native_ready, "active_slots": len(pending),
            "code_sha256": file_sha256(Path(__file__)),
        })
        for candidate, slot in pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            row = _run_slot(DIRECTORY, run_id, slot, case, candidate, adapter, {case["context_id"]: root}, decide, remaining)
            print(slot["request_id"], slot["index"], row["status"], row["first_hit"], flush=True)
            report()
    return report()


def _attempt_summary(row: JSONObject) -> JSONObject:
    steps, calls = row["steps"], row["decisions"]
    return {**{key: row[key] for key in (
                "slot_id", "run_id", "request_id", "stream", "index", "status", "first_hit", "executed_primitives",
                "physical_start_id", "future_id", "memory_id", "elapsed_seconds",
            )}, "eligible": row["status"] in ELIGIBLE, "censored": row["status"] not in ELIGIBLE,
            "budget_reason": row.get("budget_reason"),
            "native_ticks": sum(step["native_ticks"] for step in steps),
            "automatic_ticks": sum(step["automatic_ticks"] for step in steps),
            "no_ops": sum(step["no_op"] for step in steps), "policy_calls": len(calls),
            "physical_policy_calls_with_reply": sum(call.get("cache_hit") is False for call in calls),
            "input_tokens_with_reply": sum(call.get("input_tokens", 0) for call in calls),
            "output_tokens_with_reply": sum(call.get("output_tokens", 0) for call in calls),
            "trace_path": str(DIRECTORY / "private/trials" / row["run_id"] / f"{row['slot_id']}.json")}


def _results(plan: JSONObject, native: JSONObject) -> tuple[list[JSONObject], list[JSONObject]]:
    records: dict[str, list[JSONObject]] = defaultdict(list)
    for row in _records(DIRECTORY):
        records[row["slot_id"]].append(row)
    attempts: list[JSONObject] = []
    memberships = []
    groups = []
    case = plan["case"]
    for candidate in case["candidates"]:
        trials = []
        slots = _slots(plan, candidate)
        memberships.append({"candidate_id": candidate["candidate_id"], "intent": candidate["intent"], "slots": slots})
        for slot in slots:
            seen = records.pop(slot["slot_id"], [])
            for row in seen:
                if (any(row[key] != value for key, value in slot.items())
                        or row["physical_start_id"] != native["physical_start_id"]
                        or row["future_id"] != native["future_ids"][slot["index"]]
                        or row["memory_id"] != digest(INITIAL_MEMORY)
                        or row["requested_h"] != candidate["intent"]
                        or any(call["query"]["assigned_intent"] != row["requested_h"] for call in row["decisions"])):
                    raise ValueError("A Craftax trial changed frozen inputs")
            if sum(row["status"] in ELIGIBLE for row in seen) > 1:
                raise ValueError("A completed Craftax future was rerun")
            trials.extend(seen)
            attempts.extend(_attempt_summary(row) for row in seen)
            if not seen:
                attempts.append({**slot, "status": "not_started", "eligible": False, "censored": False, "first_hit": None})
        groups.append(trials)
    if records:
        raise ValueError("Unexpected slot in the Craftax trial journal")
    freeze(DIRECTORY / "pool-membership.json", memberships)
    labels = [{**{key: candidate[key] for key in ("candidate_id", "request_id", "intent")},
               "model": PROFILE, "domain": "craftax", "context_id": case["context_id"], "group_id": case["group_id"],
               "world_id": case["world_id"], "task_goal": case["task_goal"], "start_observation": native.get("observation"),
               "policy_sha256": plan["policy"]["sha256"], "pool_sha256": file_sha256(PLAN),
               "split_sha256": plan["split_sha256"], "reference_da_sha256": plan["reference_da_sha256"],
               "policy_measurements": stats}
              for candidate, stats in zip(case["candidates"], directed_labels(groups, [2, 2]), strict=True)]
    return attempts, labels


def _command(phase: str, activity: str, seconds: int, operation: str) -> str:
    return (f"uv run --locked --offline --no-sync python -m scripts.intent_pilot_v2 run "
            f"--prompt 18 --domain craftax --phase {phase} --activity {activity} --seconds {seconds} -- "
            "env JAX_COMPILATION_CACHE_DIR=$PWD/analysis/intent-pilot-v2/craftax/cache/jax "
            f"python -m carl.icl.experiments.intent_pilot_v2_craftax {operation}")


def report() -> JSONObject:
    plan = _plan()
    native = read(WITNESSES) if WITNESSES.is_file() else {}
    if native and native["plan_sha256"] != file_sha256(PLAN):
        raise ValueError("Native witnesses refer to a different frozen plan")
    attempts, labels = _results(plan, native)
    _jsonl(DIRECTORY / "attempts.jsonl", attempts)
    _jsonl(DIRECTORY / "D_C.jsonl", labels)
    forecast = plan["forecast"]
    answer = {
        "case_id": plan["case"]["case_id"], "world_seed": plan["case"]["seed"], "input_sha256": file_sha256(PLAN),
        "native_status": "verified" if native else "configured_unverified", "planned_model_attempts": 4,
        "scripted_native_witnesses": len(native.get("witnesses", [])),
        "eligible_model_attempts": sum(row["eligible"] for row in attempts),
        "censored_attempts": sum(row["censored"] for row in attempts),
        "unstarted_slots": sum(row["status"] == "not_started" for row in attempts),
        "informative_scale_rows": sum(row["policy_measurements"]["scale_supervision"] for row in labels),
        "forecast": forecast, "commands": {
            "certify": _command("preparation", NATIVE_ACTIVITY, forecast["native_deadline_seconds"], "certify"),
            "measure": _command("inference", MEASURE_ACTIVITY, forecast["block_deadline_seconds"], "measure"),
        },
    }
    write_json(DIRECTORY / "summary.json", answer)
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = {"prepare": prepare, "certify": certify, "measure": measure, "report": report}
    parser.add_argument("operation", choices=operations)
    args = parser.parse_args()
    print(json.dumps(operations[args.operation](), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
