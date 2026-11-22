"""Run the matched local Sokoban comparison on a fresh validation composition."""

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median
from typing import Literal, cast
from uuid import uuid4

from filelock import FileLock, Timeout

from carl.icl.api import BudgetExceeded, write_json
from carl.icl.boards import canonical_layout_id, from_rows
from carl.icl.experiments.intent_pilot_attempts import freeze, read
from carl.icl.experiments.intent_pilot_compare import COUNT_LIMITS, _Episode, _TaskDomain, _episode_summary, _menu, _within
from carl.icl.experiments.intent_pilot_generate import DISCOUNT, _support
from carl.icl.experiments.intent_pilot_v2 import PROFILE, V1, V2, _terrain, _witness, jsonl, seed
from carl.icl.experiments.intent_pilot_v2_run import V2Ledger, load_worker
from carl.icl.intent_domain import TASK_GOAL, Action, SokobanSnapshot, check_intent, decode_intent, replay_fragment
from carl.icl.intents.generation.data import file_sha256
from carl.icl.planning.active_intent import INTENT_CAP, IntentBudget, digest
from carl.icl.planning.intent_value import load_value
from carl.icl.planning.paired_generator import Arm, effect_id, messages, proposals, score, system_prompt
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject


PlanningArm = Literal["direct", "unlabelled", "relative"]
ARMS: tuple[PlanningArm, ...] = ("direct", "unlabelled", "relative")
GENERATOR_ARMS: tuple[Arm, ...] = ("unlabelled", "relative")
DOMAIN = "sokoban"
DIRECTORY = V2 / "validation"
CASE = DIRECTORY / "private/case.json"
PLAN = DIRECTORY / "private/plan.json"
ARCHIVED_RUNNER = DIRECTORY / "private/runner.py"
VALUE = V1 / "generator-v2/value.json"
CALIBRATION = V2 / "calibration/D_C.jsonl"
ACTIVITY = "paired-sokoban"
ARM_SECONDS = 60.0
GENERATOR_SECONDS = 21.0
MIN_PROBE_SECONDS = 10.0
LIVE_RESERVE_SECONDS = 15.0
WORKER_SECONDS = 220


def _task(row: int, col: int, kept_col: int) -> JSONObject:
    return {"effects": [{"predicate": "box_at", "row": row, "col": col}],
            "preserve": [{"predicate": "box_at", "row": 1, "col": kept_col}], "instruction": None}


def _case(split: JSONObject, profile: JSONObject) -> JSONObject:
    group = next(row for row in split["groups"] if row["domain"] == DOMAIN and row["partition"] == "validation" and row["index"] == 0)
    terrain, _, col, kept_col, goal_col = _terrain("validation", group["index"], group["seed"])
    goal_row = next(y for y, row in enumerate(terrain[2:], 2) if row[goal_col] == ".")
    terrain[goal_row + 3][col], terrain[goal_row + 2][col], terrain[1][kept_col] = "@", "$", "*"
    rows = ["".join(row) for row in terrain]
    if canonical_layout_id(from_rows(rows)) != group["group_id"]:
        raise ValueError("The new validation start differs from its frozen D4 layout")
    goal = _task(goal_row, goal_col, kept_col)
    witness = []
    for length, request in enumerate((_task(goal_row + 1, col, kept_col), goal), 1):
        path = _witness(rows, request, length)
        snapshots = replay_fragment(SokobanSnapshot(tuple(rows)), list(path))
        if len(path) != length or check_intent(decode_intent(request), [s.observation for s in snapshots]).status != "accepted":
            raise ValueError("The new composition requires a verified one-push component and two-push finish")
        witness.append({"intent": request, "actions": [action.name for action in path], "primitives": length})
    checked = read(V2 / "trial-verification.json")
    skills = [row for row in checked["trials"] if row["case_id"] == "v2-sokoban-support-diagnostic-0" and row["status"] == "intent_reached"]
    if sorted(row["actions"] for row in skills) != [["UP"], ["UP", "UP"]]:
        raise ValueError("The frozen profile has no prior verified evidence for both components")
    for row in skills:
        trace = Path(row["trace"])
        if file_sha256(trace) != row["trace_sha256"] or not read(trace)["requested_h"]["preserve"]:
            raise ValueError("Validation skill evidence is not a preserved push from the original diagnostics")
    result = {**group, "observation": rows, "intention": goal, "task": TASK_GOAL,
              "context_id": contract_hash([group["group_id"], rows, goal]),
              "initial_memory": profile["contract"]["initial_memory"],
              "rng": {"environment": "deterministic", "group_seed": group["seed"], "policy": profile["contract"]["policy_seed"]},
              "evaluator_witness": witness, "prior_real_skills": skills}
    freeze(CASE, result)
    return result


def _examples(split: JSONObject, profile_sha: str, split_sha: str) -> tuple[list[JSONObject], int, str | None]:
    support = split["inherited_training"][DOMAIN]
    source = Path(support["source"])
    if file_sha256(source) != support["sha256"]:
        raise ValueError("Inherited support changed after the v2 split freeze")
    ids = set(support["training_ids"])
    da = [row for row in jsonl(source) if row["record_id"] in ids]
    if {row["record_id"] for row in da} != ids or any(row["pilot_split"] != "train-support" or not row["validation"]["accepted"] for row in da):
        raise ValueError("Inherited generator examples must resolve to accepted v1 training support")
    measured = jsonl(CALIBRATION) if CALIBRATION.is_file() else []
    calibration_groups = {row["group_id"] for row in split["groups"] if row["domain"] == DOMAIN and row["partition"] == "calibration"}
    for row in measured:
        provenance = row["measurement_provenance"]
        if (row["pilot_split"] != "train-measurement" or row["task_group"] not in calibration_groups or
                provenance["policy_sha256"] != profile_sha or provenance["split_sha256"] != split_sha):
            raise ValueError("Relative examples must come from this frozen policy and its training measurement groups")
    examples, _, _ = _support(da, measured)
    # This permutation hides the low-to-high ordering used by the support selector.
    examples.sort(key=lambda row: seed("validation-support-order", row["record_id"]))
    informative = {row["context_id"] for row in measured if row["policy_measurements"].get("scale_supervision")}
    return examples, len(informative), file_sha256(CALIBRATION) if measured else None


def _timings(case: JSONObject) -> JSONObject:
    diagnostic = read(V2 / "results.json")
    proof = case["prior_real_skills"]
    successful = [row for row in diagnostic["rows"] if any(row["case_id"] == hit["case_id"] and row["request_id"] == hit["request_id"] for hit in proof)]
    generator = V1 / "final/validation/private/episodes"
    times = [call["elapsed_seconds"] for path in sorted(generator.glob("*-unlabelled.json")) for call in read(path)["calls"]
             if call["role"] == "generator" and call["dispatched"] and call["status"] == "ok"]
    warm = read(V1 / "final/validation/private/warmup.json")
    return {"v2_successful_request_seconds": {str(row["first_hit"]): row["attempt_seconds"] for row in successful},
            "v1_generator_median_seconds": median(times), "v1_setup_seconds": warm["setup_seconds"],
            "warmed_case_cap": ARM_SECONDS, "primary_pair_case_cap_seconds": 2 * ARM_SECONDS,
            "additional_relative_case_cap_seconds": ARM_SECONDS, "shared_worker_seconds": WORKER_SECONDS}


def _validate_plan(plan: JSONObject, *, executing: bool = True) -> None:
    runner = Path(__file__)
    if not executing and file_sha256(runner) != plan["files"]["runner"] and ARCHIVED_RUNNER.is_file():
        runner = ARCHIVED_RUNNER
    paths = {"runner": runner, "split": V2 / "splits.json", "diagnostic": V2 / "private/diagnostic-plan.json", "value": VALUE,
             "case": CASE, "local_config": V1 / "direct" / f"{PROFILE}.json"}
    for name, path in paths.items():
        if plan["files"][name] != file_sha256(path):
            raise ValueError(f"The frozen comparison input changed: {name}")
    if contract_hash(plan["support"]) != plan["support_sha256"]:
        raise ValueError("The frozen generator support changed")
    if plan["case"]["initial_memory"] != {"history": [], "note": None, "feedback": None}:
        raise ValueError("The paired runner requires the original empty controller memory")


def freeze_validation() -> JSONObject:
    """Freeze one case, matched support, scoring and all admission rules before model work."""
    if PLAN.is_file():
        plan: JSONObject = read(PLAN)
        _validate_plan(plan)
        return plan
    split_file = V2 / "splits.json"
    diagnostic = V2 / "private/diagnostic-plan.json"
    source = read(diagnostic)
    profile = source["profile"]
    split_sha = file_sha256(split_file)
    if split_sha != source["split_sha256"]:
        raise ValueError("Validation does not use the previously frozen split")
    split = read(split_file)
    case = _case(split, profile)
    support, informative_contexts, calibration_sha = _examples(split, profile["sha256"], split_sha)
    value = load_value(VALUE, discount=DISCOUNT)
    if value.identity()["fingerprint"] != read(V1 / "generator-v2/selection.json")["value_fingerprint"]:
        raise ValueError("The fitted value differs from the original selected value")
    value.assert_held_out(SokobanSnapshot(tuple(case["observation"])).observation, case["group_id"], split="validation")
    relative = len({row["measured_u"] for row in support if row["measured_u"] is not None}) >= 2
    scored_arms = ARMS if relative else ARMS[:2]
    plan = {
        "schema": "carl-pilot-v2-paired-validation", "case": case, "profile": profile, "arms": list(scored_arms),
        "support": support, "support_sha256": contract_hash(support), "D_C_sha256": calibration_sha,
        "relative_status": "exploratory" if relative else "unmeasured_no_informative_scale",
        "informative_training_contexts": informative_contexts, "generator_edition": 2,
        "generator_system_prompts": {arm: system_prompt(DOMAIN, arm, edition=2) for arm in GENERATOR_ARMS},
        "initial_options": _options(case["observation"], case["intention"]),
        "value": value.identity(), "limits": {**COUNT_LIMITS, "arm_seconds": ARM_SECONDS, "intent_primitives": INTENT_CAP,
                                              "generator_seconds": GENERATOR_SECONDS, "min_probe_seconds": MIN_PROBE_SECONDS,
                                              "live_reserve_seconds": LIVE_RESERVE_SECONDS, "probe_count_fraction": 0.5},
        "forecast": _timings(case), "partition": "validation", "policy_cache": "exact_intra_arm_wire_only",
        "tools": "Every runner has the same policy interface and native simulator. B and C perform hypothetical probes.",
        "files": {"runner": file_sha256(Path(__file__)), "split": split_sha, "diagnostic": file_sha256(diagnostic),
                  "value": file_sha256(VALUE), "case": file_sha256(CASE),
                  "local_config": file_sha256(V1 / "direct" / f"{PROFILE}.json")},
    }
    freeze(PLAN, plan)
    return plan


def _options(rows: list[str], goal: JSONObject) -> list[JSONObject]:
    options = _menu(rows, goal)
    boxes = [row for row in options if all(p["predicate"] == "box_at" for p in row["intent"]["effects"])]
    return boxes if len(boxes) >= 2 else options


class _PairedEpisode(_Episode):
    def _restore_live_limits(self, deadline: float) -> None:
        self.budget.max_actions = COUNT_LIMITS["primitives"]
        self.budget.max_ticks = COUNT_LIMITS["native_ticks"]
        self.budget.max_calls = COUNT_LIMITS["requests"]
        self.budget.wall_seconds = deadline - self.budget.started
        self.calls.phase, self.calls.deadline, self.calls.token_limit = "live", deadline, COUNT_LIMITS["tokens"]

    def _probe_slots(self, batch: JSONObject, deadline: float) -> None:
        slots = batch["generator"]["slots"]
        for index, slot in enumerate(slots):
            slot_deadline = deadline - MIN_PROBE_SECONDS * (len(slots) - index - 1)
            self.calls.deadline = slot_deadline
            self.budget.wall_seconds = slot_deadline - self.budget.started
            if self.budget.cutoff() is not None:
                slot["probe_status"] = "not_dispatched_probe_cap"
                continue
            slot["probe_index"] = len(self.trace["probes"])
            record = self.attempt(slot["intent"], probe=True)
            slot.update(probe_status=record["status"], probe_first_hit=record["first_hit"], task_solved=record["task_solved"],
                        probe_primitives=record["executed_primitives"], probe_budget_reason=record["budget_reason"])
            initial = record["states"][0]
            if (initial["state_id"], initial["memory_id"]) != (batch["parent_state_id"], batch["parent_memory_id"]):
                raise ValueError("Probe did not restore the current live parent and memory")
            if record["status"] in ("intent_reached", "task_solved"):
                self.calls.deadline = deadline
                endpoint = record["states"][-1]["observation"]
                value = self.calls.value(self.budget, self.value, endpoint)
                length = record["executed_primitives"] if record["task_solved"] else record["first_hit"]
                slot["score"] = score(length, value, DISCOUNT)
                batch["scored"].append(slot["index"])
            self.save()

    def choose(self, deadline: float) -> JSONObject | None:
        domain = self.session.domain
        assert isinstance(domain, _TaskDomain)
        rows = list(domain.snapshot.rows)
        goal = self.case["intention"]
        options = _options(rows, goal)
        before = self.budget.actions, self.budget.ticks, self.budget.calls
        self._restrict_probes(deadline)
        probe_end = deadline - LIVE_RESERVE_SECONDS
        batch: JSONObject = {"status": "started", "options": options, "scored": [], "live_index": None,
                             "parent_state_id": digest(rows),
                             "parent_memory_id": digest({"history": self.session.history, "note": self.session.note, "feedback": self.session.feedback})}
        self.trace["batches"].append(batch)
        try:
            if len(options) < 2:
                batch["status"] = "insufficient_distinct_candidates"
                return None
            if min(probe_end - 2 * MIN_PROBE_SECONDS, time.monotonic() + GENERATOR_SECONDS) <= time.monotonic() or self.budget.cutoff() is not None:
                raise BudgetExceeded("probe_admission")
            self.calls.deadline = min(probe_end - 2 * MIN_PROBE_SECONDS, time.monotonic() + GENERATOR_SECONDS)
            arm = cast(Arm, self.arm)
            wire = messages(DOMAIN, arm, rows, self.case["task"], self.support, options, edition=2)
            self.budget.calls += 1
            generated = self.calls.invoke("generator", wire)
            parsed = proposals(DOMAIN, arm, generated["response"], rows, options)
            batch["generator"] = parsed
            accepted = {effect_id(option["intent"]): option for option in options}
            for slot in parsed["slots"]:
                if slot["status"] != "valid":
                    continue
                option = accepted.get(slot["effect_id"])
                if option is None or slot["intent"] != option["intent"]:
                    slot.update(status="invalid", error="outside_frozen_menu")
                else:
                    slot["proposal_id"] = option["id"]
            if any(slot["status"] != "valid" for slot in parsed["slots"]):
                batch["status"] = "insufficient_distinct_proposals"
                return None
            self._probe_slots(batch, probe_end)
            if any("probe_index" not in slot for slot in parsed["slots"]):
                batch["status"] = "incomplete_probe_batch"
                return None
            winner = max((slot for slot in parsed["slots"] if "score" in slot), key=lambda slot: (slot["score"], -slot["index"]), default=None)
            if winner is None:
                batch["status"] = "no_verified_probe"
                return None
            batch.update(status="selected", committed_slot=winner["index"], live_index=len(self.trace["live"]))
            return dict(winner["intent"])
        except (BudgetExceeded, OSError, RuntimeError, ValueError) as error:
            batch.update(status="probe_cutoff" if isinstance(error, BudgetExceeded) else "proposal_error", error=str(error))
            return None
        finally:
            batch.update(executed_primitives=self.budget.actions - before[0], ticks=self.budget.ticks - before[1], requests=self.budget.calls - before[2])
            self._restore_live_limits(deadline)
            self.save()


def _trial_key(plan: JSONObject, arm: PlanningArm) -> str:
    case = plan["case"]
    return contract_hash([plan["profile"]["sha256"], case["context_id"], case["intention"], arm, case["rng"], file_sha256(PLAN)])


def _attempt_paths(plan: JSONObject, arm: PlanningArm) -> list[Path]:
    return sorted((DIRECTORY / "private/episodes" / _trial_key(plan, arm)).glob("*.json"), key=lambda path: path.stat().st_mtime_ns)


def _done(paths: list[Path]) -> bool:
    return any(read(path)["status"] != "running" for path in paths)


def _require_lease(worker_seconds: int) -> None:
    jobs = V2Ledger(V1).status()["jobs"]
    if not any(job["prompt"] == 17 and job["phase"] == "inference" and job["activity"] == ACTIVITY and job["status"] == "reserved"
               and job["workers"] == 1 and job["reserved_ms"] >= (worker_seconds + 5) * 1000 for job in jobs):
        raise ValueError("Reserve the full local validation worker through the v2 ledger before dispatch")


def run_validation(worker_seconds: int = WORKER_SECONDS, *, primary_only: bool = False) -> JSONObject:
    """Use a single warm local controller for the whole predeclared comparison."""
    started = time.monotonic()
    _require_lease(worker_seconds)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with FileLock(str(DIRECTORY / "worker.lock"), timeout=0):
        plan = freeze_validation()
        publish_validation()
        pending = [arm for arm in plan["arms"] if not (primary_only and arm == "relative") and not _done(_attempt_paths(plan, arm))]
        if not pending:
            return publish_validation()
        mandatory = sum(arm != "relative" for arm in pending) or 1
        hard_deadline = started + worker_seconds
        if hard_deadline - time.monotonic() < plan["forecast"]["v1_setup_seconds"] + ARM_SECONDS * mandatory:
            raise ValueError("The remaining block cannot admit the complete primary comparison after setup")
        run_id = uuid4().hex
        with _within(hard_deadline - ARM_SECONDS * mandatory):
            worker = load_worker(plan["profile"], run_id)
        role = worker.role
        case = plan["case"]
        role.assert_query(SokobanSnapshot(tuple(case["observation"])).observation, split="validation", task_group=case["group_id"])
        freeze(DIRECTORY / "private/workers" / f"{run_id}.json", {
            "load": str(V2 / "private/workers" / f"{run_id}.json"), "worker_setup_seconds": time.monotonic() - started,
            "profile_sha256": plan["profile"]["sha256"], "worker_seconds": worker_seconds,
        })
        value = load_value(VALUE, discount=DISCOUNT)
        for arm in pending:
            if time.monotonic() + ARM_SECONDS > hard_deadline:
                break
            key = _trial_key(plan, arm)
            path = DIRECTORY / "private/episodes" / key / f"{run_id}.json"
            budget = IntentBudget(COUNT_LIMITS["primitives"], COUNT_LIMITS["requests"], COUNT_LIMITS["native_ticks"], ARM_SECONDS)
            episode = _PairedEpisode(case, arm, role, plan["support"], value, path, budget)
            episode.trace.update(trial_key=key, run_id=run_id, plan_sha256=file_sha256(PLAN))
            episode.execute()
            print(case["case_id"], arm, episode.trace["status"], episode.trace["first_hit"], flush=True)
            publish_validation()
        return publish_validation()


def _verify_trace(case: JSONObject, trace: JSONObject) -> None:
    live = case["observation"]
    memory = case["initial_memory"]
    for attempt in trace["live"]:
        initial = attempt["states"][0]
        if initial["observation"] != live or initial["memory"] != memory:
            raise ValueError("A committed attempt did not continue from the actual live state and memory")
        replay = replay_fragment(SokobanSnapshot(tuple(live)), [int(Action[step["applied"]]) for step in attempt["steps"]])
        if [list(snapshot.rows) for snapshot in replay] != [row["observation"] for row in attempt["states"]]:
            raise ValueError("The live replay differs from the saved transitions")
        live, memory = attempt["states"][-1]["observation"], attempt["states"][-1]["memory"]
    if trace["status"] == "task_solved":
        start = SokobanSnapshot(tuple(case["observation"]))
        replay = replay_fragment(start, [int(Action[step["applied"]]) for attempt in trace["live"] for step in attempt["steps"]])
        if check_intent(decode_intent(case["intention"]), [state.observation for state in replay]).status != "accepted":
            raise ValueError("The claimed live task success did not pass independent preservation and effect checks")


def _summary(plan: JSONObject, arm: PlanningArm) -> JSONObject:
    paths = _attempt_paths(plan, arm)
    history = [{"path": str(path), "status": read(path)["status"]} for path in paths]
    best = next((path for path in paths if read(path)["status"] != "running"), paths[-1] if paths else None)
    common = {"case_id": plan["case"]["case_id"], "arm": arm, "trial_key": _trial_key(plan, arm), "history": history}
    if best is None:
        return {**common, "status": "not_started" if arm in plan["arms"] else plan["relative_status"], "assigned": False, "solved": None}
    trace = read(best)
    interrupted = trace["status"] == "running"
    if not interrupted:
        _verify_trace(plan["case"], trace)
    row = _episode_summary(trace, interrupted=interrupted)
    slots: list[JSONObject] = []
    for batch in trace["batches"]:
        live_index = batch.get("live_index")
        result = trace["live"][live_index] if live_index is not None and live_index < len(trace["live"]) else {}
        for slot in batch.get("generator", {}).get("slots", []):
            selected = batch.get("committed_slot") == slot["index"]
            slots.append({**slot, "selected": selected, "live_status": result.get("status") if selected else None,
                          "live_first_hit": result.get("first_hit") if selected else None, "live_task_solved": result.get("task_solved") if selected else None})
    return {**row, **common, "assigned": True, "solved": None if interrupted else row["status"] == "task_solved",
            "trace": str(best), "trace_sha256": file_sha256(best), "slots": slots,
            "no_ops": sum(step["no_op"] for attempt in [*trace["probes"], *trace["live"]] for step in attempt["steps"])}


def publish_validation() -> JSONObject:
    plan = read(PLAN)
    _validate_plan(plan, executing=False)
    rows = [_summary(plan, arm) for arm in ARMS]
    pairings = []
    for left, right in ((rows[0], rows[1]), (rows[0], rows[2]), (rows[1], rows[2])):
        a, b = left["solved"], right["solved"]
        pairings.append({"left": left["arm"], "right": right["arm"],
                         "outcome": "incomplete" if a is None or b is None else "tie" if a == b else left["arm"] if a else right["arm"]})
    jobs = [job for job in V2Ledger(V1).status()["jobs"] if job["prompt"] == 17 and job["activity"] == ACTIVITY]
    report = {"plan_sha256": file_sha256(PLAN), "partition": "validation", "relative_status": plan["relative_status"],
              "rows": rows, "pairs": pairings, "forecast": plan["forecast"], "jobs": jobs}
    write_json(DIRECTORY / "results.json", report)
    columns = ("case_id", "arm", "assigned", "solved", "status", "first_hit", "live_primitives", "executed_primitives", "native_ticks",
               "requests", "physical_model_requests", "charged_tokens", "no_ops", "elapsed_seconds")
    with (DIRECTORY / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Prompt 17 paired Sokoban validation", "", "One fresh held-out validation composition. The sealed split remains unopened.", "",
             f"Frozen policy `{plan['profile']['sha256']}` and plan `{report['plan_sha256']}`.",
             f"Relative arm `{plan['relative_status']}`. Informative training contexts {plan['informative_training_contexts']}.", "",
             "All variants use the same local action policy, memory bounds, typed task, 48 primitives/ticks, 16 logical calls, 16,000 tokens and 60 warmed seconds.",
             "Each planned arm uses the same two-slot generator grammar, support order, value and scheduler. Public box options take priority when two are available.",
             "Probes and live steps share one budget. Live execution invokes the same policy in the real state. "
             "An exact same-wire response may be reused within an arm and still incurs the logical token and call charges.",
             plan["tools"], "",
             "| Arm | Solved / assigned | Stop | First hit | All primitives | Logical / physical requests | Charged tokens | Seconds |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        solved = int(row["solved"]) if row["solved"] is not None else "—"
        elapsed = row.get("elapsed_seconds")
        lines.append(f"| {row['arm']} | {solved}/{int(row['assigned'])} | {row['status']} | {row.get('first_hit') or '—'} | "
                     f"{row.get('executed_primitives', '—')} | {row.get('requests', '—')}/{row.get('physical_model_requests', '—')} | "
                     f"{row.get('charged_tokens', '—')} | {f'{elapsed:.3f}' if elapsed is not None else '—'} |")
    if any(row.get("slots") for row in rows):
        lines += ["", "| Arm | Requested u | Parsed option | Probe first hit | Selected | Live first hit |",
                  "| --- | ---: | --- | ---: | --- | ---: |"]
        for row in rows:
            for slot in row.get("slots", []):
                scale = slot.get("requested_u")
                selected = "yes" if slot["selected"] else "no"
                lines.append(f"| {row['arm']} | {scale if scale is not None else '—'} | {slot.get('proposal_id', '—')} | "
                             f"{slot.get('probe_first_hit') or '—'} | {selected} | {slot.get('live_first_hit') or '—'} |")
    lines += ["", "Case-level comparisons", "", *[f"- {p['left']} versus {p['right']} — {p['outcome']}" for p in pairings], "",
              f"Settled ledger charge {sum((job['charged_ms'] or 0) for job in jobs) / 1000:.3f} worker-seconds and USD 0.",
              "The incremental relative-arm envelope is 60 warmed seconds on the shared worker.",
              "Parsed choices, separate speculative and committed hits, and interruption evidence are in `results.json` and `private/episodes/`.",
              "", "```bash", "source analysis/intent-pilot/env.sh",
              "uv run --locked --offline --no-sync python -m carl.icl.experiments.intent_pilot_v2_validation report", "```", ""]
    (DIRECTORY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "run", "report"))
    parser.add_argument("--worker-seconds", type=int, default=WORKER_SECONDS)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.operation == "freeze":
            freeze_validation()
        result = run_validation(args.worker_seconds, primary_only=args.primary_only) if args.operation == "run" else publish_validation()
    except (BudgetExceeded, OSError, Timeout, ValueError) as error:
        parser.exit(1, f"Sokoban paired validation: {error}\n")
    print(json.dumps({"directory": str(DIRECTORY), "plan_sha256": result["plan_sha256"], "pairs": result["pairs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
