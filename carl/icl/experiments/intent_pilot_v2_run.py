"""Meter v2 experiments and run the pinned Sokoban diagnostic with resumable trials."""

import csv
import json
import math
import time
from sqlite3 import Connection
from statistics import median
from uuid import uuid4

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.config import LocalModelSettings
from carl.icl.experiments.intent_pilot_attempts import freeze, read
from carl.icl.experiments.intent_pilot_budget import DOMAINS, LIMIT_MS, PHASES, PilotLedger, _now
from carl.icl.experiments.intent_pilot_compare import _within
from carl.icl.experiments.intent_pilot_measure import ELIGIBLE, INITIAL_MEMORY, LocalDecider, load_plan
from carl.icl.experiments.intent_pilot_v2 import PROFILE, V1, V2, jsonl
from carl.icl.intent_domain import Action, SokobanSnapshot, check_intent, decode_intent, replay_fragment
from carl.icl.intents.generation.data import file_sha256
from carl.icl.local_models import bind_local_role
from carl.icl.planning.active_intent import INTENT_CAP, IntentBudget, IntentSession, digest
from carl.icl.planning.local_direct import SokobanDirect, parse_batch
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject


V2_LIMIT_MS = 728_278
PAIRED_MS = 230_000
CLOSEOUT_MS = 25_000
PROMPTS = range(14, 21)


class V2Ledger(PilotLedger):
    """Reuse the v1 settlement path and add a single, transactional v2 allowance."""

    def _balance(self, conn: Connection) -> tuple[int, int]:
        total, spent = conn.execute("""
            SELECT COALESCE(SUM(COALESCE(charged_ms, reserved_ms)), 0),
                   COALESCE(SUM(CASE WHEN prompt BETWEEN 14 AND 20 THEN COALESCE(charged_ms, reserved_ms) END), 0)
            FROM jobs
        """).fetchone()
        limit = conn.execute("SELECT value FROM contract WHERE key='v2_limit_ms'").fetchone()
        cap = int(limit[0]) if limit else min(V2_LIMIT_MS, max(0, LIMIT_MS - total + spent))
        return cap, max(0, min(cap - spent, LIMIT_MS - total))

    def balance(self) -> JSONObject:
        with self._connection() as conn:
            cap, remaining = self._balance(conn)
        return {"limit_seconds": cap / 1000, "available_seconds": remaining / 1000,
                "paired_reserve_seconds": PAIRED_MS / 1000, "closeout_reserve_seconds": CLOSEOUT_MS / 1000}

    def _reserve(
        self, prompt: int, domain: str, phase: str, activity: str,
        command: list[str], seconds: int, workers: int,
    ) -> str:
        if prompt not in PROMPTS or domain not in DOMAINS or phase not in PHASES:
            raise ValueError("Invalid v2 prompt, domain or phase")
        if min(seconds, workers) < 1 or not command:
            raise ValueError("A positive deadline, worker count and local command are required")
        requested = (seconds + 1) * workers * 1000
        protected = PAIRED_MS + CLOSEOUT_MS if prompt <= 15 else CLOSEOUT_MS if prompt < 20 else 0
        with self._connection(write=True) as conn:
            limit, remaining = self._balance(conn)
            if remaining - requested < protected:
                raise ValueError(f"Reservation needs {requested / 1000:.3f}s plus {protected / 1000:.3f}s protected")
            conn.execute("INSERT OR IGNORE INTO contract VALUES ('v2_limit_ms', ?)", (str(limit),))
            job = uuid4().hex
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 'reserved', NULL)",
                (job, prompt, domain, phase, activity, json.dumps(command), _now(), workers, requested),
            )
        return job


def _player(rows: list[str]) -> tuple[int, int]:
    return next((y, x) for y, row in enumerate(rows) for x, tile in enumerate(row) if tile in "@+")


def _v1_diagnosis() -> JSONObject:
    path = V1 / "final/validation/private/episodes/generated_12x12x10_validation_1-direct.json"
    right = read(path)["live"][0]
    blocked = next(i for i, step in enumerate(right["steps"]) if step["no_op"])
    y, x = _player(right["states"][blocked]["observation"])
    hit_path = V1 / "final/calibration/qwen3-0.6b/sokoban/attempts.jsonl"
    hit = next(row for row in jsonl(hit_path) if row["status"] == "intent_reached")
    empty_path = V1 / "measurements/qwen2.5-1.5b-instruct/sokoban/attempts.jsonl"
    empty = next(row for row in jsonl(empty_path) if row["status"] == "invalid_response")
    wood_path = V1 / "direct/qwen3-0.6b-results.json"
    wood = next(row for row in read(wood_path)["domains"]["craftax"]["records"] if row["skill"] == "gather_wood")
    decision = wood["result"]["decisions"][0]
    visible = decision["query"]["observation"]
    depth_path = V1 / "final/validation/private/episodes/generated_12x12x10_validation_1-unlabelled.json"
    depth = read(depth_path)
    return {
        "right_only": {"file": str(path), "requested": [step["requested"] for step in right["steps"]],
                       "first_no_op": blocked + 1, "player": [y, x], "forward_tile": right["states"][blocked]["observation"][y][x + 1],
                       "first_reply": right["decisions"][0]["response"], "h": right["requested_h"]},
        "one_action_hit": {"file": str(hit_path), "id": hit["attempt_id"], "h": hit["requested_h"],
                           "reply": hit["decisions"][0]["response"], "step": hit["steps"][0], "first_hit": hit["first_hit"]},
        "empty_list": {"file": str(empty_path), "id": empty["attempt_id"], "reply": empty["decisions"][-1]["response"],
                       "output_tokens": empty["decisions"][-1]["output_tokens"], "error": empty["decisions"][-1]["error"]},
        "craftax_wood": {"file": str(wood_path), "id": wood["case_id"], "reply": decision["response"],
                         "facing": visible["facing"], "front_tile": visible["neighbors"][visible["facing"]],
                         "first_transition": decision["steps"][0]},
        "depth_one": {"file": str(depth_path), "slots": depth["batches"][0]["generator"]["slots"],
                      "probe": depth["probes"][0]["status"], "first_no_op":
                      next((i + 1 for i, step in enumerate(depth["probes"][0]["steps"]) if step["no_op"]), None),
                      "committed": len(depth["live"])},
    }


def diagnostics_and_forecast() -> None:
    from transformers import AutoTokenizer

    diagnosis = _v1_diagnosis()
    recipe = read(V2 / "private/diagnostic-plan.json")
    pin = recipe["profile"]["contract"]
    paths = {
        PROFILE: ("UP", "DO"), "qwen2.5-1.5b-instruct": ("RIGHT",),
    }
    fit = {}
    for model, actions in paths.items():
        cfg = read(V1 / f"direct/{model}.json")["local"]
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"], local_files_only=True, trust_remote_code=False)
        for action in actions:
            reply = json.dumps({"actions": [action]}, separators=(",", ":"))
            vocabulary = (action,) if action == "DO" else tuple(a.name for a in Action)
            parsed, _ = parse_batch(reply, vocabulary)
            tokens = len(tokenizer.encode(reply, add_special_tokens=False))
            if parsed != [action] or tokens > pin["decoding"]["output_tokens"]:
                raise ValueError("A single legal action cannot fit the frozen output contract")
            fit[f"{model}/{action}"] = {"tokens": tokens, "reply": reply}
    diagnosis["wire_checks"] = {"zero_based_coordinates": True, "action_ids": {a.name: int(a) for a in Action},
                                 "legal_single_action_tokens": fit, "output_cap": pin["decoding"]["output_tokens"]}
    freeze(V2 / "v1-diagnostics.json", diagnosis)

    previous = {
        domain: jsonl(V1 / f"final/calibration/{PROFILE}/{domain}/attempts.jsonl")
        for domain in ("sokoban", "craftax")
    }
    stats = {}
    for domain, rows in previous.items():
        times = [d["inference_seconds"] for row in rows for d in row.get("decisions", [])
                 if d.get("inference_seconds") is not None and not d.get("cache_hit")]
        stats[domain] = {"warm_median_seconds": median(times), "longest_attempt_seconds":
                         max(row.get("elapsed_seconds", 0) for row in rows)}
    cold = max(read(V1 / p)["setup_seconds"] for p in (
        "comparison/private/warmup.json", "final/validation/private/warmup.json", "effort/results.json",
    ))
    native = max(read(p)["native_setup_seconds"] for p in (V1 / "measurements/private/sessions").glob("*.json"))
    block = math.ceil(cold) + 4 * recipe["attempt_wall_seconds"] + 21
    forecast = {
        "previous_regime": stats, "observed_cold_seconds": cold,
        "diagnostic": {"deadline_seconds": block, "reserved_seconds": block + 1, "calls_share_one_loaded_model": True,
                       "assigned_attempts": 4, "per_attempt_wall_seconds": recipe["attempt_wall_seconds"]},
        "paired_reserve_seconds": PAIRED_MS / 1000, "closeout_reserve_seconds": CLOSEOUT_MS / 1000,
        "craftax": {"previous_setup_seconds": native, "planned_attempts": 4,
                    "median_full_cap_forecast_seconds":
                    native + 4 * INTENT_CAP * stats["craftax"]["warm_median_seconds"],
                    "status": "reassess_after_sokoban_settles"},
    }
    freeze(V2 / "forecast.json", forecast)


def _trial_plan() -> tuple[JSONObject, list[tuple[JSONObject, JSONObject, str]]]:
    plan: JSONObject = read(V2 / "private/diagnostic-plan.json")
    if file_sha256(V2 / "splits.json") != plan["split_sha256"]:
        raise ValueError("The v2 split changed")
    keys = []
    profile = plan["profile"]
    for case in plan["cases"]:
        for request in case["requests"]:
            identifier = contract_hash([profile["sha256"], case["context_id"], request, plan["arm"],
                                        {"environment_seed": case["seed"], "policy_seed": profile["contract"]["policy_seed"]}])
            keys.append((case, request, identifier))
    return plan, keys


def _attempts(trial: str) -> list[JSONObject]:
    rows = []
    for path in sorted((V2 / "private/attempts" / trial).glob("*.json"), key=lambda item: item.stat().st_mtime_ns):
        row = read(path)
        rows.append({**row, "status": "campaign_interrupted" if row["status"] == "running" else row["status"]})
    return rows


def _finished(records: list[JSONObject], *, retry_censored: bool = False) -> JSONObject | None:
    completed = next((row for row in records if row["status"] in ELIGIBLE), None)
    if completed is not None or retry_censored:
        return completed
    return next((row for row in reversed(records) if row["status"] in ("budget_cutoff", "controller_error")), None)


def _stop_reason(row: JSONObject) -> str | None:
    # The shared intent engine categorizes BudgetExceeded from a controller call as model_context.
    calls = row.get("decisions", [])
    reason = row.get("budget_reason")
    if reason == "model_context" and calls and calls[-1].get("error") == "wall":
        return "wall"
    return reason if isinstance(reason, str) else None


def diagnose(*, retry_censored: bool = False) -> None:
    ledger = V2Ledger(V1).status()
    if not any(job["prompt"] == 15 and job["phase"] == "inference" and job["status"] == "reserved" for job in ledger["jobs"]):
        raise ValueError("Run the diagnostic through the Prompt 15 ledger reservation")
    plan, keys = _trial_plan()
    source, _ = load_plan(V1)
    profile = plan["profile"]
    if profile != source["policies"][PROFILE]:
        raise ValueError("The original profile fingerprint changed")
    with FileLock(str(V2 / "private/diagnostic.lock"), timeout=0):
        pending = [(case, request, k) for case, request, k in keys
                   if _finished(_attempts(k), retry_censored=retry_censored) is None]
        if not pending:
            return
        import torch

        started = time.monotonic()
        cfg = read(V1 / "direct/qwen3-0.6b.json")
        torch.set_num_threads(cfg["threads"])
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(profile["contract"]["policy_seed"])
        role = bind_local_role(LocalModelSettings.model_validate(cfg["local"]), "policy")
        if role.identity() != profile["contract"]["model"]:
            raise ValueError("The cached local weights differ from v1")
        _ = role._stack
        _ = role._tokenizer
        cache = V2 / "private/cache" / profile["sha256"]
        check = V1 / "measurements/private/cache" / profile["sha256"] / "greedy-check.json"
        freeze(cache / "greedy-check.json", read(check))
        decider = LocalDecider(role, cache)
        run_id = uuid4().hex
        freeze(V2 / "private/workers" / f"{run_id}.json", {"setup_seconds": time.monotonic() - started, "profile": profile["sha256"]})
        for case, request, key in pending:
            _run_attempt(plan, case, request, key, run_id, decider)
            print(case["case_id"], request["request_id"], _attempts(key)[-1]["status"], flush=True)
            report()


def _run_attempt(plan: JSONObject, case: JSONObject, request: JSONObject, key: str, run_id: str,
                 decider: LocalDecider) -> None:
    start = SokobanSnapshot(tuple(case["observation"]))
    cap = plan["attempt_wall_seconds"]
    budget = IntentBudget(INTENT_CAP, 32, 32, cap)
    session = IntentSession(SokobanDirect(start, decode_intent(request["intent"])), case["task_goal"], budget)
    deadline = budget.started + cap
    identity = {"trial_key": key, "case_id": case["case_id"], "context_id": case["context_id"],
                "profile_sha256": plan["profile"]["sha256"], "request_id": request["request_id"], "arm": plan["arm"],
                "environment_seed": case["seed"], "policy_seed": plan["profile"]["contract"]["policy_seed"],
                "future_id": digest(case["observation"]), "initial_memory_id": digest(INITIAL_MEMORY), "run_id": run_id}
    path = V2 / "private/attempts" / key / f"{run_id}.json"

    def save(row: JSONObject) -> None:
        write_json(path, {**identity, **row})

    def decide(wire: list[dict[str, str]]) -> JSONObject:
        with _within(deadline):
            return decider(wire)

    result = session.attempt(key, request["intent"], decide, cap=INTENT_CAP, stop_on_invalid=True, checkpoint=save)
    if result["states"][0]["state_id"] != identity["future_id"] or result["states"][0]["memory_id"] != identity["initial_memory_id"]:
        raise ValueError("Trial did not restore the frozen state and memory")
    result["budget_reason"] = _stop_reason(result)
    save(result)


def verify_trials() -> JSONObject:
    """Independently replay every stored native step and separate failures from witnesses."""
    plan, keys = _trial_plan()
    paths = {(row["case_id"], path["request_id"]): path["actions"]
             for row in read(V2 / "private/native-witnesses.json") for path in row["paths"]}
    checked = []
    for case, request, key in keys:
        for attempt in _attempts(key):
            actions = [step["requested"] for step in attempt["steps"] if step["executed"]]
            snapshot = SokobanSnapshot(tuple(case["observation"]))
            frames = replay_fragment(snapshot, [int(Action[name]) for name in actions])
            if [list(frame.rows) for frame in frames] != [row["observation"] for row in attempt["states"]]:
                raise ValueError("Independent replay disagrees with a recorded trial")
            if any(call["query"]["assigned_intent"] != request["intent"] or call["query"]["public_task"] != case["task_goal"]
                   for call in attempt["decisions"]):
                raise ValueError("The model-facing query differs from the exact frozen task or intent")
            if attempt["initial_memory_id"] != digest(INITIAL_MEMORY) or attempt["profile_sha256"] != plan["profile"]["sha256"]:
                raise ValueError("A trial uses a different memory or profile")
            hit = attempt.get("first_hit")
            if hit is not None and (hit != len(actions) or check_intent(
                    decode_intent(request["intent"]), [frame.observation for frame in frames]).status != "accepted"):
                raise ValueError("The recorded first hit lacks verified native effects and preservation")
            expected = paths[(case["case_id"], request["request_id"])]
            deviation = next(({"action_index": i + 1, "observed": action, "minimum_witness": reference,
                               "player": list(_player(list(frames[i].rows)))}
                              for i, (action, reference) in enumerate(zip(actions, expected)) if action != reference), None)
            no_op = next((i + 1 for i, step in enumerate(attempt["steps"]) if step["no_op"]), None)
            path = V2 / "private/attempts" / key / f"{attempt['run_id']}.json"
            checked.append({
                "case_id": case["case_id"], "request_id": request["request_id"], "status": attempt["status"],
                "actions": actions, "first_hit": hit, "first_no_op": no_op, "stop_reason": _stop_reason(attempt),
                "first_witness_deviation": deviation,
                "first_request_error": next((call["error"] for call in attempt["decisions"] if "error" in call), None),
                "trace": str(path), "trace_sha256": file_sha256(path),
            })
    result = {"independent_replays": len(checked), "trials": checked}
    write_json(V2 / "trial-verification.json", result)
    return result


def report() -> JSONObject:
    plan, keys = _trial_plan()
    ledger = V2Ledger(V1)
    balance = ledger.balance()
    jobs = [job for job in ledger.status()["jobs"] if job["prompt"] in PROMPTS]
    forecast = read(V2 / "forecast.json")
    evidence = read(V2 / "private/native-witnesses.json")
    native = {item["case_id"]: {p["request_id"]: p for p in item["paths"]} for item in evidence}
    rows = []
    for case, request, key in keys:
        attempts = _attempts(key)
        item = _finished(attempts)
        rows.append({
            "case_id": case["case_id"], "request_id": request["request_id"], "trial_key": key,
            "witness_length": native[case["case_id"]][request["request_id"]]["minimum_primitives"],
            "status": item["status"] if item else "campaign_interrupted" if attempts else "not_started",
            "budget_reason": _stop_reason(item) if item else None,
            "first_hit": item.get("first_hit") if item else None,
            "no_ops": sum(step["no_op"] for step in item.get("steps", [])) if item else None,
            "requests": len(item.get("decisions", [])) if item else None,
            "attempt_seconds": item.get("elapsed_seconds") if item else None,
            "attempt_history": [row["status"] for row in attempts],
        })
    hits = [row["first_hit"] for row in rows if row["first_hit"] is not None]
    chosen = plan["profile"]["sha256"] if any(hit > 1 for hit in hits) else None
    result = {"profile_sha256": plan["profile"]["sha256"], "split_sha256": plan["split_sha256"],
              "pool_sha256": file_sha256(V2 / "private/diagnostic-plan.json"), "rows": rows,
              "succeeded": len(hits), "assigned": len(rows), "profile_for_calibration": chosen,
              "jobs": jobs, **balance}
    native_block = forecast["craftax"]["median_full_cap_forecast_seconds"]
    free_after_reserves = balance["available_seconds"] - (PAIRED_MS + CLOSEOUT_MS) / 1000
    result["craftax"] = {**forecast["craftax"], "unprotected_seconds": free_after_reserves,
                         "full_block_fits_forecast": free_after_reserves >= native_block + 1}
    write_json(V2 / "results.json", result)
    with (V2 / "diagnostic.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ready = forecast["diagnostic"]["deadline_seconds"]
    pending = sum(row["status"] in ("not_started", "campaign_interrupted") for row in rows)
    resume = min(ready, 60 + 75 * pending)
    command = ("uv run --locked --offline --no-sync python -m scripts.intent_pilot_v2 run --prompt 15 --domain sokoban "
               f"--phase inference --activity pinned-diagnostic --seconds {resume} -- "
               "uv run --locked --offline --no-sync python -m scripts.intent_pilot_v2 diagnostic")
    lines = [
        "# CaRL intent pilot v2", "", f"Frozen v1 Qwen3-0.6B profile `{result['profile_sha256']}`.",
        f"Split SHA-256 `{result['split_sha256']}` and diagnostic pool SHA-256 `{result['pool_sha256']}`.", "",
        f"Available {balance['available_seconds']:.3f} s of {balance['limit_seconds']:.3f} s. "
        f"Protected {PAIRED_MS/1000:.0f} s for pairing and {CLOSEOUT_MS/1000:.0f} s for closeout.", "",
        "| Diagnostic case | Request | Native minimum | Status | Model first hit | No-ops | Requests | Seconds |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        *[f"| {r['case_id']} | {r['request_id']} | {r['witness_length']} | {r['status']} | "
          f"{r['first_hit'] if r['first_hit'] is not None else '—'} | {r['no_ops'] if r['no_ops'] is not None else '—'} | "
          f"{r['requests'] if r['requests'] is not None else '—'} | "
          f"{round(r['attempt_seconds'], 3) if r['attempt_seconds'] is not None else '—'} |" for r in rows],
        "", f"Selected profile for the separate calibration is `{chosen or 'pending'}`.", "",
    ]
    charged = sum((job["charged_ms"] or 0) for job in jobs)
    lines += [f"Settled v2 jobs charged {charged / 1000:.3f} worker-seconds and USD 0.",
              "Prompt 14 checked the third RIGHT against a wall, the preserved one-action hit, the empty Qwen2.5 "
              "reply, the Craftax LEFT no-op facing a tree, and the depth-one probe cutoff with zero commitments.",
              "Native IDs are UP=0, DOWN=1, LEFT=2, RIGHT=3. Each tested one-action reply encodes to 5 of 64 tokens.",
              "The v1 data and sealed-report checksums passed. Inherited support references only the original training rows.",
              "The model only reads frozen observations and requests. Native oracle lengths and snapshots are separate.",
              "", f"The four-attempt Craftax forecast is {native_block:.3f} s at the historical decision median "
              f"and setup cost. Available outside protected work {max(0, free_after_reserves):.3f} s.", ""]
    proof = V2 / "trial-verification.json"
    if proof.is_file():
        replay = read(proof)
        lines += [f"Independent native replays passed for {replay['independent_replays']} recorded attempts. "
                  "Trace paths and the first witness deviation are in `trial-verification.json`.", ""]
        failure = next((row for row in replay["trials"] if row["first_hit"] is None and row["first_witness_deviation"]), None)
        if failure:
            deviation = failure["first_witness_deviation"]
            reason = failure.get("stop_reason") or failure.get("first_request_error") or failure["status"]
            lines += [f"On {failure['case_id']}/{failure['request_id']}, the first witness deviation was action "
                      f"{deviation['action_index']} at {deviation['player']}. The model chose {deviation['observed']}. "
                      f"The minimum witness uses {deviation['minimum_witness']}. The recorded stop was `{reason}`.", ""]
    if pending:
        lines += ["Resume from the first unset outcome.", "", "```bash", "source analysis/intent-pilot/env.sh", command, "```", ""]
    elif chosen:
        lines += ["Prompt 15 selected the original profile. The calibration partition remains separate from these diagnostics.",
                  "The test groups are unopened. Refresh the evidence before the next stage.", "",
                  "```bash", "source analysis/intent-pilot/env.sh",
                  "uv run --locked --offline --no-sync python -m scripts.intent_pilot_v2 report", "```", ""]
    (V2 / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    return result
