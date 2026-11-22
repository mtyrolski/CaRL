"""Measure constructed Sokoban intent pools with the pinned v2 controller."""

import json
import math
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.experiments.intent_pilot_attempts import freeze, read
from carl.icl.experiments.intent_pilot_measure import ELIGIBLE, INITIAL_MEMORY, load_plan
from carl.icl.experiments.intent_pilot_v2 import PROFILE, V1, V2
from carl.icl.experiments.intent_pilot_v2_run import (
    V2Ledger, _attempts, _finished, _run_attempt, _trial_plan, load_worker, verify_trials,
)
from carl.icl.intents.generation.data import file_sha256
from carl.icl.planning.active_intent import INTENT_CAP, digest
from carl.icl.policy_relative.measurements import directed_labels
from carl.icl.types import JSONObject


CAL = V2 / "calibration"
PLAN = CAL / "plan.json"
WITNESSES = CAL / "private/native-witnesses.json"
CHECK = CAL / "verification.json"


def prepare_calibration() -> JSONObject:
    from carl.icl.experiments.intent_pilot_v2_pools import calibration_cases

    diagnostic = read(V2 / "results.json")
    original, _ = load_plan(V1)
    profile = original["policies"][PROFILE]
    if diagnostic["profile_for_calibration"] != profile["sha256"]:
        raise ValueError("The diagnostic did not select the frozen Qwen3 profile")
    cases, witnesses = calibration_cases()
    source = read(V2 / "private/diagnostic-plan.json")
    observed_test_seconds = sum(row["attempt_seconds"] or 0 for row in diagnostic["rows"])
    worker_seconds = sum(job["charged_ms"] / 1000 for job in diagnostic["jobs"]
                         if job["prompt"] == 15 and job["activity"] == "pinned-diagnostic" and job["charged_ms"] is not None)
    # Keep the same trial wall ceiling and budget startup plus publishing separately.
    overhead = math.ceil(max(0, worker_seconds - observed_test_seconds)) + 12
    cap = source["attempt_wall_seconds"]
    plan = {
        "schema": "carl-pilot-v2-sokoban-calibration-v1", "split_sha256": file_sha256(V2 / "splits.json"),
        "profile": profile, "arm": "directed", "cases": cases, "attempt_wall_seconds": cap,
        "intent_cap": INTENT_CAP, "initial_memory_sha256": digest(INITIAL_MEMORY),
        "futures_per_candidate": 1, "determinism": "native Sokoban and pinned CPU greedy Torch",
        "sampling_scope": "deliberately_constructed_feasibility_test",
        "diagnostic_plan_sha256": file_sha256(V2 / "private/diagnostic-plan.json"),
        "context_deadline_seconds": 2 * cap + overhead, "forecast_setup_and_reporting_seconds": overhead,
    }
    freeze(PLAN, plan)
    freeze(WITNESSES, witnesses)
    return publish_calibration()


def _require_lease(plan: JSONObject) -> None:
    amount = (plan["context_deadline_seconds"] + 1) * 1000
    if not any(job["prompt"] == 16 and job["phase"] == "inference" and job["status"] == "reserved"
               and job["reserved_ms"] >= amount for job in V2Ledger(V1).status()["jobs"]):
        raise ValueError(f"Reserve a full context before dispatch ({amount / 1000:.0f} worker-seconds including cleanup)")


def measure_context(index: int) -> JSONObject:
    plan, slots = _trial_plan(PLAN)
    _require_lease(plan)
    if not 0 <= index < len(plan["cases"]):
        raise ValueError(f"Unknown frozen context index {index}")
    with FileLock(str(CAL / "private/calibration.lock"), timeout=0):
        done = {key: _finished(_attempts(key), retry_censored=True) is not None for _, _, key in slots}
        if any(case["index"] < index and not done[key] for case, _, key in slots):
            raise ValueError("Finish the preceding context's eligible trials before opening the next context")
        pending = [(case, request, key) for case, request, key in slots if case["index"] == index and not done[key]]
        if pending:
            original, _ = load_plan(V1)
            profile = plan["profile"]
            if profile != original["policies"][PROFILE]:
                raise ValueError("The selected frozen controller differs from v1")
            run_id = uuid4().hex
            decider = load_worker(profile, run_id)
            for case, request, key in pending:
                _run_attempt(plan, case, request, key, run_id, decider)
                print(case["case_id"], request["request_id"], _attempts(key)[-1]["status"], flush=True)
        verify_trials(PLAN, WITNESSES, CHECK)
        return publish_calibration()


def _jsonl(path: Path, rows: list[JSONObject]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text("".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def context_command(index: int, deadline: int) -> str:
    return ("uv run --locked --offline --no-sync python -m scripts.intent_pilot_v2 run "
            f"--prompt 16 --domain sokoban --phase inference --activity calibration-{index} --seconds {deadline} -- "
            f"uv run --locked --offline --no-sync python -m scripts.intent_pilot_v2 calibrate --context {index}")


def publish_calibration() -> JSONObject:
    plan, slots = _trial_plan(PLAN)
    verified = read(CHECK)["trials"] if CHECK.exists() else []
    verified_hashes = {row["trace_sha256"] for row in verified}
    rows, labels = [], []
    for state in plan["cases"]:
        group = [(case, request, key) for case, request, key in slots if case["case_id"] == state["case_id"]]
        attempts_for_requests = []
        for _, request, key in group:
            records = _attempts(key)
            marked = []
            for attempt in records:
                trace = V2 / "private/attempts" / key / f"{attempt['run_id']}.json"
                valid = attempt["status"] in ELIGIBLE and file_sha256(trace) in verified_hashes
                marked.append({**attempt, "eligible": valid, "assigned": True, "censored": not valid,
                               "trace": str(trace), "trace_sha256": file_sha256(trace)})
            if not marked:
                marked.append({"case_id": state["case_id"], "context_id": state["context_id"], "request_id": request["request_id"],
                               "trial_key": key, "status": "not_started", "assigned": False, "eligible": False,
                               "censored": False, "first_hit": None})
            rows.extend(marked)
            attempts_for_requests.append([attempt for attempt in marked if attempt["eligible"]])
        stats = directed_labels(attempts_for_requests, [1] * len(group))
        for (case, request, key), entry in zip(group, stats, strict=True):
            labels.append({
                "schema_version": "carl-pilot-directed-v2", "split": "train", "pilot_split": "train-measurement",
                "case_id": case["case_id"], "context_id": case["context_id"], "task_group": case["group_id"],
                "request_id": request["request_id"], "record_id": key, "candidate_id": key, "type": "box_at",
                "task_goal": case["task_goal"], "start_observation": case["observation"], "intent": request["intent"],
                "measurement_provenance": {"pool_sha256": file_sha256(PLAN), "split_sha256": plan["split_sha256"],
                                           "policy_sha256": plan["profile"]["sha256"]},
                "policy_measurements": entry,
            })
    _jsonl(CAL / "attempts.jsonl", rows)
    _jsonl(CAL / "D_C.jsonl", labels)
    ledger = V2Ledger(V1)
    balance = ledger.balance()
    jobs = [job for job in ledger.status()["jobs"] if job["prompt"] == 16]
    summary = {"pool_sha256": file_sha256(PLAN), "profile_sha256": plan["profile"]["sha256"], "rows": [
        {"case_id": label["case_id"], "request_id": label["request_id"], **label["policy_measurements"]}
        for label in labels
    ], "eligible": sum(row["eligible_n"] for row in (label["policy_measurements"] for label in labels)),
        "planned": len(labels), "censored": sum(row["censored"] for row in rows),
        "unassigned": sum(not row["assigned"] for row in rows), "jobs": jobs, **balance}
    informative = [label for label in labels if label["policy_measurements"]["scale_supervision"]]
    summary["informative_contexts"] = len({label["context_id"] for label in informative})
    summary["unprotected_seconds"] = balance["available_seconds"] - balance["paired_reserve_seconds"] - balance["closeout_reserve_seconds"]
    write_json(CAL / "results.json", summary)

    def value(item: object) -> str:
        return "—" if item is None else str(item)

    lines = [
        "## Prompt 16 Sokoban calibration", "",
        "Deliberately constructed feasibility sample. One deterministic trial per explicitly requested intent. "
        "Durations are executed primitives to first verified hit.", "",
        f"Pool SHA-256 `{summary['pool_sha256']}`. Frozen profile `{summary['profile_sha256']}`.", "",
        "| Context | Request | Eligible / planned | p | d | w | u |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        *[f"| {row['case_id']} | {row['request_id']} | {row['eligible_n']}/{row['planned_n']} | " +
          " | ".join(value(row[key]) for key in ("p", "d", "w", "u")) + " |" for row in summary["rows"]],
        "", f"Informative contexts {summary['informative_contexts']}. Censored attempts {summary['censored']}. "
        f"Unassigned slots {summary['unassigned']}. Full traces are retained in the private attempt archive.", "",
    ]
    if informative:
        context_id = informative[0]["context_id"]
        low, high = sorted((item for item in informative if item["context_id"] == context_id),
                           key=lambda item: item["policy_measurements"]["d"])
        early, late = low["policy_measurements"], high["policy_measurements"]
        lines += [
            f"First-hit primitive counts were {early['d']:g} for {low['request_id']} "
            f"and {late['d']:g} for {high['request_id']}. Each was requested once and succeeded.",
            "Their success rates are 1 each, so both weights are 1/2. The shorter duration's CDF interval "
            "runs from 0 to 1/2, with midpoint 1/4. The longer duration's interval runs from 1/2 to 1, "
            "with midpoint 3/4.", "",
        ]
    if any(row["assigned"] for row in rows) and not any(
        row["assigned"] and row["status"] != "intent_reached" for row in rows
    ):
        lines += ["Both recorded requests succeeded. Hindsight added zero rows to D_A.", ""]
    spent = sum(job["charged_ms"] or 0 for job in jobs)
    lines += [f"Settled calibration charge {spent / 1000:.3f} s. Available outside the protected blocks "
              f"{max(0, summary['unprotected_seconds']):.3f} s.", ""]
    for case in plan["cases"]:
        if any(row["case_id"] == case["case_id"] and not row["eligible_n"] for row in summary["rows"]):
            seconds = plan["context_deadline_seconds"]
            fit = "fits" if seconds + 1 <= summary["unprotected_seconds"] else "exceeds the currently available allowance"
            lines += [f"Context {case['index']} needs a {seconds + 1} s reservation including cleanup. It {fit}.", "",
                      "```bash", "source analysis/intent-pilot/env.sh", context_command(case["index"], seconds), "```", ""]
            break
    (CAL / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
