"""Compare committed direct and depth-one episodes on held-out Sokoban compositions."""

import csv
import os
import signal
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from carl.environment.sokoban.env import SokobanEnv
from carl.environment.sokoban.tokenizer import SokobanTokenizer
from carl.icl.api import BudgetExceeded, write_json
from carl.icl.boards import to_rows
from carl.icl.config import LocalModelSettings
from carl.icl.experiments.intent_pilot_attempts import freeze, read
from carl.icl.experiments.intent_pilot_budget import PilotLedger
from carl.icl.experiments.intent_pilot_generate import DISCOUNT, GENERATOR_CONTEXT, GENERATOR_TOKENS, _options, _sokoban_options
from carl.icl.experiments.intent_pilot_measure import CONTEXT_LIMIT, OUTPUT_TOKENS, load_plan
from carl.icl.intent_domain import TASK_GOAL, SokobanObservation, SokobanSnapshot, decode_intent, predicate_truth
from carl.icl.intent_domain import render_instruction, replay_fragment
from carl.icl.intents.generation.data import file_sha256
from carl.icl.local_models import LocalRoleModel, bind_local_role
from carl.icl.planning.active_intent import INTENT_CAP, IntentBudget, IntentSession
from carl.icl.planning.intent_value import FittedIntentValue, load_value
from carl.icl.planning.local_direct import SokobanDirect
from carl.icl.planning.paired_generator import Arm, messages, proposals, score
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject
from carl.utils.aliases import State


MODEL = "qwen3-0.6b"
DOMAIN = "sokoban"
CELL = f"{MODEL}/{DOMAIN}"
ARMS = ("direct", "unlabelled", "relative")
COUNT_LIMITS = {"primitives": 48, "native_ticks": 48, "requests": 16, "tokens": 16_000}
CASE_SECONDS = 15.0
COMMIT_SECONDS = 0.75
CASE_COUNT = 2


def _condition(cell: list[int]) -> JSONObject:
    row, col = cell
    return {"predicate": "box_at", "row": row, "col": col}


def _compose(episode: JSONObject) -> JSONObject | None:
    for index, (start, walk, push) in enumerate(zip(episode["facts"], episode["facts"][1:], episode["facts"][2:])):
        initial_goals = set(map(tuple, start["occupied_goals"]))
        walked_goals = set(map(tuple, walk["occupied_goals"]))
        final_goals = set(map(tuple, push["occupied_goals"]))
        kept = sorted(initial_goals & walked_goals & final_goals)
        added = sorted(final_goals - walked_goals)
        if not kept or not added or start["boxes"] != walk["boxes"]:
            continue
        rows = episode["observations"][index]
        h = {"effects": [_condition(list(added[0]))], "preserve": [_condition(list(kept[0]))], "instruction": None}
        if len(_menu(rows, h)) < 2:
            continue
        actions = episode["actions"][index:index + 2]
        verified = replay_fragment(SokobanSnapshot(tuple(rows)), actions)
        expected = episode["observations"][index:index + 3]
        if [list(snapshot.rows) for snapshot in verified] != expected or expected[0] == expected[1]:
            raise ValueError("The held-out composition does not replay as a walk followed by filling a goal")
        return {
            "case_id": episode["episode_id"], "group_id": episode["group_id"], "partition": "validation",
            "observation": rows, "intention": h, "task": render_instruction(decode_intent(h)),
            "initial_memory": {"history": [], "note": None, "feedback": None},
            "live_environment": "deterministic_native_sokoban",
            "evaluator_witness": {"start": index, "actions": actions, "trace_sha256": contract_hash(expected)},
        }
    return None


def _menu(rows: list[str], task: JSONObject) -> list[JSONObject]:
    observation = SokobanObservation(tuple(rows))
    keep = task["preserve"]
    if any(predicate_truth(predicate, observation) is not True for predicate in decode_intent(task)["preserve"]):
        return []
    intents = [task]
    for option in _sokoban_options(rows):
        h = option["intent"]
        if any(predicate in keep for predicate in h["effects"]):
            continue
        intents.append({**h, "preserve": keep})
    return _options(DOMAIN, rows, intents)


def freeze_validation(root: Path) -> tuple[JSONObject, FittedIntentValue, list[JSONObject]]:
    directory = root / "generator-v2"
    selection = read(directory / "selection.json")
    source = directory / "private/plan.json"
    generator = read(source)
    measured, pool_hash = load_plan(root)
    if (file_sha256(source) != selection["config_sha256"] or
            generator["directed_pool_sha256"] != pool_hash or
            selection["selected"].get(CELL) != "unlabelled_diagnostic"):
        raise ValueError("The comparison requires the matched frozen validation selection and policy")
    value = load_value(directory / "value.json", discount=DISCOUNT)
    if contract_hash(value.model_dump(mode="json")) != selection["value_fingerprint"]:
        raise ValueError("The validation value differs from the selected value")
    cell = generator["cells"][CELL]
    if cell["policy"] != measured["policies"][MODEL]["contract"]:
        raise ValueError("The policy differs from the directed measurement")
    annotations = root / "annotations/sokoban"
    episodes = annotations / "private/episodes.json"
    split = read(annotations / "split_manifest.json")
    if (file_sha256(episodes) != split["locked_files"]["private/episodes.json"] or
            split["base_split_sha256"] != PilotLedger(root).split_digest()):
        raise ValueError("The composition source differs from the frozen campaign")
    used = {case["group_id"] for case in generator["validation"][DOMAIN]}
    cases = []
    for episode in sorted(read(episodes), key=lambda row: row["episode_id"]):
        if episode["pilot_split"] != "validation" or episode["group_id"] in used or episode["fixture"]:
            continue
        case = _compose(episode)
        if case:
            value.assert_held_out(SokobanObservation(tuple(case["observation"])), case["group_id"], split="validation")
            cases.append(case)
    if len(cases) != CASE_COUNT or len({case["group_id"] for case in cases}) != CASE_COUNT:
        raise ValueError("Expected two new compositions on distinct held-out validation layouts")
    evidence = root / "generator-v2/private/rollouts" / CELL
    skill_hits: dict[str, list[str]] = {"walk": [], "push_while_preserving_goal": []}
    for path in sorted(evidence.glob("*.json")):
        row = read(path)
        if row.get("status") != "intent_reached" or row.get("first_hit") != 1:
            continue
        h = row["requested_h"]
        if h["effects"][0]["predicate"] == "player_at":
            skill_hits["walk"].append(file_sha256(path))
        if h["effects"][0]["predicate"] == "box_at" and h["preserve"]:
            skill_hits["push_while_preserving_goal"].append(file_sha256(path))
    if not all(skill_hits.values()):
        raise ValueError("The frozen controller has no independent evidence for a required skill")
    freeze(root / "comparison/private/plan.json", {
        "schema": "carl-pilot-paired-validation-v1", "split_sha256": split["base_split_sha256"],
        "selection_sha256": file_sha256(directory / "selection.json"), "episodes_sha256": file_sha256(episodes),
        "value": selection["value_fingerprint"], "value_target": TASK_GOAL, "policy_sha256": cell["policy_sha256"],
        "runner_sha256": file_sha256(Path(__file__)), "skill_hit_sha256": skill_hits,
        "other_cells": {name: settings["arms"] for name, settings in generator["cells"].items() if name != CELL},
        "relative_status": selection["comparative_scale_status"], "limits": {**COUNT_LIMITS, "wall_seconds": CASE_SECONDS},
        "commit_wall_reserve": COMMIT_SECONDS, "commit_count_reserve_fraction": 0.5, "cases": cases,
    })
    return cell, value, cases


class _GoalEnvironment(SokobanEnv):
    def __init__(self, snapshot: SokobanSnapshot, goal: JSONObject) -> None:
        super().__init__(
            SokobanTokenizer(size_of_board=(len(snapshot.rows), len(snapshot.rows[0]))),
            num_boxes=sum(row.count("$") + row.count("*") for row in snapshot.rows),
        )
        self.goal = decode_intent(goal)

    def is_solved(self, board: State) -> bool:
        visible = SokobanObservation(tuple(to_rows(board)))
        return all(predicate_truth(p, visible) is True for p in (*self.goal["effects"], *self.goal["preserve"]))


class _TaskDomain(SokobanDirect):
    def __init__(self, snapshot: SokobanSnapshot, goal: JSONObject) -> None:
        self.snapshot = snapshot
        self.intent = decode_intent(goal)
        self.goal = goal
        self.env = _GoalEnvironment(snapshot, goal)

    def __copy__(self) -> "_TaskDomain":
        duplicate = _TaskDomain(self.snapshot, self.goal)
        duplicate.intent = self.intent
        return duplicate


def _deadline(_: int, _frame: object) -> None:
    raise BudgetExceeded("wall")


@contextmanager
def _within(deadline: float) -> Iterator[None]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BudgetExceeded("wall")
    previous = signal.signal(signal.SIGALRM, _deadline)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class _Calls:
    role: LocalRoleModel
    save: Callable[[], None]
    rows: list[JSONObject] = field(default_factory=list)
    cache: dict[str, JSONObject] = field(default_factory=dict)
    phase: str = "live"
    deadline: float = 0.0
    token_limit: int = COUNT_LIMITS["tokens"]

    def invoke(self, role: str, wire: list[dict[str, str]]) -> JSONObject:
        limit, context = (OUTPUT_TOKENS, CONTEXT_LIMIT) if role == "policy" else (GENERATOR_TOKENS, GENERATOR_CONTEXT)
        started = time.monotonic()
        row: JSONObject = {"role": role, "phase": self.phase, "status": "admission", "dispatched": False}
        self.rows.append(row)
        try:
            n_in = self.role.input_tokens(
                [{"role": message["role"], "content": [{"text": message["content"]}]} for message in wire]
            )
            if n_in + limit > self.role.context_limit(context):
                raise BudgetExceeded("model_context")
            used = sum(event.get("charged_tokens", 0) for event in self.rows)
            if used + n_in + limit > self.token_limit:
                raise BudgetExceeded("tokens")
            key = contract_hash([role, wire])
            if key in self.cache:
                row.update(**self.cache[key], cache_hit=True, status="ok", request_sha256=key)
            else:
                payload: JSONObject = {}

                def checkpoint(actual: JSONObject) -> None:
                    payload.update(actual)
                    self.save()

                with _within(self.deadline):
                    row.update(status="in_flight", dispatched=True, request_sha256=key, input_tokens=n_in, output_tokens=None,
                               charged_tokens=n_in + limit, payload=payload, started_at_ns=time.time_ns())
                    self.save()
                    raw, actual_in, n_out = self.role.complete_chat(
                        wire, output_tokens=limit, context_limit=context, on_payload=checkpoint,
                    )
                if n_in != actual_in:
                    raise ValueError("The admitted token count differs from the actual input")
                response = {"response": raw, "input_tokens": actual_in, "output_tokens": n_out,
                            "charged_tokens": actual_in + n_out, "payload": payload}
                self.cache[key] = response
                row.update(**response, cache_hit=False, status="ok")
        except (BudgetExceeded, OSError, RuntimeError, ValueError) as error:
            row.update(status="cutoff" if isinstance(error, BudgetExceeded) else "error", error=str(error))
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            self.save()
        return row

    def value(self, budget: IntentBudget, value: FittedIntentValue, rows: list[str]) -> float:
        if time.monotonic() >= self.deadline or budget.remaining()["calls"] <= 0:
            raise BudgetExceeded("value_admission")
        started = time.monotonic()
        budget.calls += 1
        result = value(SokobanObservation(tuple(rows)), TASK_GOAL)
        self.rows.append({"role": "value", "phase": self.phase, "status": "ok", "value": result,
                          "dispatched": False, "elapsed_seconds": time.monotonic() - started})
        self.save()
        return result


@dataclass
class _Episode:
    case: JSONObject
    arm: Literal["direct", "unlabelled", "relative"]
    role: LocalRoleModel
    support: list[JSONObject]
    value: FittedIntentValue
    path: Path
    budget: IntentBudget
    trace: JSONObject = field(init=False)
    calls: _Calls = field(init=False)
    session: IntentSession = field(init=False)

    def __post_init__(self) -> None:
        observation = SokobanSnapshot(tuple(self.case["observation"]))
        task = self.case["intention"]
        self.session = IntentSession(_TaskDomain(observation, task), self.case["task"], self.budget)
        self.calls = _Calls(self.role, self.save)
        self.trace = {"case_id": self.case["case_id"], "arm": self.arm, "status": "running",
                      "live": [], "probes": [], "batches": [], "calls": self.calls.rows, "first_hit": None}
        self.save()

    def save(self) -> None:
        write_json(self.path, self.trace)

    def attempt(self, h: JSONObject, *, probe: bool = False) -> JSONObject:
        records: list[JSONObject] = self.trace["probes" if probe else "live"]
        identifier = f"{self.case['case_id']}/{self.arm}/{self.calls.phase}/{len(records)}"
        records.append({})

        def checkpoint(row: JSONObject) -> None:
            records[-1] = row
            self.save()

        row = self.session.attempt(
            identifier, h, lambda wire: self.calls.invoke("policy", wire),
            speculative=probe, cap=INTENT_CAP, stop_on_invalid=True, checkpoint=checkpoint,
        )
        if row["budget_reason"] == "model_context" and row["decisions"]:
            row["budget_reason"] = row["decisions"][-1].get("error", "model_context")
        checkpoint(row)
        return row

    def _restrict_probes(self, absolute_deadline: float) -> None:
        """Leave half of the count limits live, even after multiple rounds of probes."""
        for attr, limit, column in (("actions", "primitives", "executed_primitives"), ("calls", "requests", "requests"),
                                    ("ticks", "native_ticks", "ticks")):
            cap = COUNT_LIMITS[limit]
            spent = sum(row[column] for row in self.trace["batches"])
            setattr(self.budget, f"max_{attr}", min(cap - 1, getattr(self.budget, attr) + max(0, cap // 2 - spent)))
        live_tokens = sum(row.get("charged_tokens", 0) for row in self.calls.rows if row["phase"] == "live")
        self.calls.token_limit = min(COUNT_LIMITS["tokens"] - CONTEXT_LIMIT, live_tokens + COUNT_LIMITS["tokens"] // 2)
        self.calls.deadline = absolute_deadline - COMMIT_SECONDS
        self.budget.wall_seconds = self.calls.deadline - self.budget.started
        self.calls.phase = "probe"

    def choose(self, deadline: float) -> JSONObject | None:
        domain = self.session.domain
        assert isinstance(domain, _TaskDomain)
        rows = list(domain.snapshot.rows)
        options = _menu(rows, self.case["intention"])
        arm = cast(Arm, self.arm)
        before = self.budget.actions, self.budget.ticks, self.budget.calls
        self._restrict_probes(deadline)
        batch: JSONObject = {"status": "started", "options": options, "scored": []}
        self.trace["batches"].append(batch)
        try:
            if len(options) < 2:
                batch["status"] = "insufficient_distinct_candidates"
                return None
            if self.budget.cutoff() is not None:
                raise BudgetExceeded(str(self.budget.cutoff()))
            self.budget.calls += 1
            wire = messages(DOMAIN, arm, rows, self.case["task"], self.support, options, edition=2)
            generated = self.calls.invoke("generator", wire)
            batch["generator"] = proposals(DOMAIN, arm, generated["response"], rows, options)
            for slot in batch["generator"]["slots"]:
                if slot["status"] != "valid":
                    continue
                last = next((r["elapsed_seconds"] for r in reversed(self.calls.rows)
                             if r["role"] == "policy" and r["status"] == "ok" and r["dispatched"]), 0)
                if self.budget.cutoff() is not None or (batch["scored"] and time.monotonic() + last >= self.calls.deadline):
                    slot["probe_status"] = "not_dispatched_probe_cap"
                    continue
                h = decode_intent(slot["intent"])
                if h["preserve"] != decode_intent(self.case["intention"])["preserve"]:
                    slot["probe_status"] = "incompatible_preservation"
                    continue
                attempt = self.attempt(dict(h), probe=True)
                if attempt["status"] not in ("intent_reached", "task_solved"):
                    continue
                if attempt["task_solved"]:
                    candidate_score = score(attempt["executed_primitives"], 1.0, DISCOUNT)
                else:
                    endpoint = attempt["states"][-1]["observation"]
                    candidate_score = score(attempt["first_hit"], self.calls.value(self.budget, self.value, endpoint), DISCOUNT)
                batch["scored"].append({"slot": slot["index"], "h": h, "score": candidate_score})
            batch["status"] = "selected" if batch["scored"] else "no_verified_probe"
        except (BudgetExceeded, OSError, RuntimeError, ValueError) as error:
            batch.update(status="probe_cutoff" if isinstance(error, BudgetExceeded) else "proposal_error", error=str(error))
        finally:
            batch.update(executed_primitives=self.budget.actions - before[0],
                         ticks=self.budget.ticks - before[1], requests=self.budget.calls - before[2])
            self.budget.max_actions = COUNT_LIMITS["primitives"]
            self.budget.max_calls = COUNT_LIMITS["requests"]
            self.budget.max_ticks = COUNT_LIMITS["native_ticks"]
            self.budget.wall_seconds = deadline - self.budget.started
            self.calls.phase, self.calls.deadline, self.calls.token_limit = "live", deadline, COUNT_LIMITS["tokens"]
            self.save()
        winner = max(batch["scored"], key=lambda row: (row["score"], -row["slot"]), default=None)
        if winner is None:
            return None
        batch["committed_slot"] = winner["slot"]
        self.save()
        return dict(winner["h"])

    def execute(self) -> None:
        deadline = self.budget.started + self.budget.wall_seconds
        self.calls.deadline = deadline
        h = self.case["intention"]
        final = "budget_cutoff"
        while self.budget.cutoff() is None:
            assignment = h if self.arm == "direct" else self.choose(deadline)
            if assignment is None:
                final = self.trace["batches"][-1]["status"]
                break
            result = self.attempt(assignment)
            final = result["status"]
            safe = final not in ("preservation_failed", "preservation_unknown")
            if safe and result["task_solved"]:
                self.trace["first_hit"] = sum(row["executed_primitives"] for row in self.trace["live"])
                final = "task_solved"
                break
            if ((self.arm == "direct" and final != "intent_timeout") or
                    final not in ("intent_reached", "intent_timeout") or not result["executed_primitives"]):
                break
        self.trace.update(status=final, elapsed_seconds=time.monotonic() - self.budget.started,
                          budget=asdict(self.budget))
        self.save()


def run(root: Path, wall_seconds: int) -> JSONObject:
    """Reserve once externally. A completed or interrupted assigned episode is immutable."""
    started = time.monotonic()
    if wall_seconds <= CASE_SECONDS:
        raise ValueError("The comparison deadline must admit at least one full warmed episode")
    directory = root / "comparison"
    cell, value, cases = freeze_validation(root)
    publish(root)
    print("Frozen paired validation cases:", ", ".join(case["case_id"] for case in cases), flush=True)
    import torch

    policy = cell["policy"]
    config = read(root / "direct" / f"{MODEL}.json")
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(policy["policy_seed"])
    role = bind_local_role(LocalModelSettings.model_validate(config["local"]), "policy")
    if role.identity() != policy["model"]:
        raise ValueError("The actual local weights differ from the frozen policy")
    setup = time.monotonic()
    with _within(started + wall_seconds - CASE_SECONDS):
        _ = role._stack
        _ = role._tokenizer
    write_json(directory / "private/warmup.json", {
        "elapsed_seconds": time.monotonic() - setup, "setup_seconds": time.monotonic() - started,
        "policy": role.identity(), "threads": torch.get_num_threads(),
    })
    for index, case in enumerate(cases):
        order: tuple[Literal["direct", "unlabelled"], ...] = ("direct", "unlabelled") if index % 2 == 0 else ("unlabelled", "direct")
        for arm in order:
            path = directory / "private/episodes" / f"{case['case_id']}-{arm}.json"
            if path.exists():
                continue
            seconds = min(CASE_SECONDS, started + wall_seconds - time.monotonic())
            if seconds <= COMMIT_SECONDS:
                return publish(root)
            budget = IntentBudget(COUNT_LIMITS["primitives"], COUNT_LIMITS["requests"], COUNT_LIMITS["native_ticks"], seconds)
            episode = _Episode(case, arm, role, cell["examples"], value, path, budget)
            episode.execute()
            print(case["case_id"], arm, episode.trace["status"], episode.trace["first_hit"], flush=True)
            publish(root)
    return publish(root)


def _episode_summary(row: JSONObject, *, interrupted: bool) -> JSONObject:
    attempts = [*row["probes"], *row["live"]]
    calls = row["calls"]
    live = [step for attempt in row["live"] for step in attempt["steps"]]
    steps = [step for attempt in attempts for step in attempt["steps"]]
    return {
        "case_id": row["case_id"], "arm": row["arm"],
        "status": "campaign_interrupted" if interrupted and row["status"] == "running" else row["status"],
        "first_hit": row["first_hit"],
        "executed_primitives": sum(step["executed"] for step in steps),
        "live_primitives": sum(step["executed"] for step in live), "native_ticks": sum(step["native_ticks"] for step in steps),
        "automatic_ticks": sum(step["automatic_ticks"] for step in steps),
        "logical_calls": dict(Counter(call["role"] for call in calls)), "requests": len(calls),
        "physical_model_requests": sum(call["dispatched"] for call in calls),
        "input_tokens": sum(call.get("input_tokens", 0) for call in calls),
        "known_output_tokens": sum(call.get("output_tokens") or 0 for call in calls),
        "unknown_output_calls": sum(call["dispatched"] and call.get("output_tokens") is None for call in calls),
        "charged_tokens": sum(call.get("charged_tokens", 0) for call in calls),
        "elapsed_seconds": row.get("elapsed_seconds"), "attempt_statuses": [attempt["status"] for attempt in attempts],
        "budget_reasons": [attempt["budget_reason"] for attempt in attempts if attempt.get("budget_reason")],
        "probe_batches": [batch["status"] for batch in row["batches"]],
    }


def publish(root: Path) -> JSONObject:
    """Preserve paired denominators, including undispatched or unavailable episodes."""
    directory = root / "comparison"
    plan = read(directory / "private/plan.json")
    ledger = PilotLedger(root).status()
    jobs = [job for job in ledger["jobs"] if job["prompt"] == 7 and job["activity"] == "paired-validation"]
    interrupted = bool(jobs and jobs[-1]["status"] in ("timed_out", "failed", "launch_failed"))
    runs: list[JSONObject] = []
    for case in plan["cases"]:
        for arm in ARMS:
            path = directory / "private/episodes" / f"{case['case_id']}-{arm}.json"
            if path.is_file():
                runs.append(_episode_summary(read(path), interrupted=interrupted))
            else:
                status = plan["relative_status"] if arm == "relative" else "not_dispatched"
                runs.append({"case_id": case["case_id"], "arm": arm, "status": status, "first_hit": None})
    summaries = {}
    for arm in ARMS:
        assigned = [row for row in runs if row["arm"] == arm]
        measured = [row for row in assigned if "requests" in row]
        completed = [row for row in measured if row["status"] not in ("running", "campaign_interrupted")]
        statuses = dict(Counter(row["status"] for row in assigned))
        selected = [row["first_hit"] for row in measured if row["first_hit"] is not None]
        total: dict[str, Any] = {}
        for key in ("executed_primitives", "live_primitives", "native_ticks", "requests", "physical_model_requests",
                    "input_tokens", "known_output_tokens", "unknown_output_calls", "charged_tokens", "elapsed_seconds"):
            total[key] = sum(row[key] for row in measured) if measured and all(row[key] is not None for row in measured) else None
        summaries[arm] = {"assigned": len(assigned), "started": len(measured), "completed": len(completed),
                          "solved": len(selected) if completed else None, "first_hits": selected, **total, "statuses": statuses}
    report = {"cell": CELL, "other_cells": plan["other_cells"], "limits": plan["limits"],
              "setup": read(directory / "private/warmup.json") if (directory / "private/warmup.json").is_file() else None,
              "job": jobs[-1] if jobs else None,
              "plan_sha256": file_sha256(directory / "private/plan.json"), "arms": summaries, "episodes": runs}
    write_json(directory / "comparison.json", report)
    with (directory / "comparison.csv.tmp").open("w", encoding="utf-8", newline="") as stream:
        columns = list(dict.fromkeys(key for row in runs for key in row))
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(runs)
    os.replace(directory / "comparison.csv.tmp", directory / "comparison.csv")
    lines = ["# Paired validation", "", f"Controller / domain: {CELL}", "",
             "| Arm | Solved / assigned | First hits | All / live actions | Requests | Charged tokens | Seconds | Stops |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, row in summaries.items():
        shown = {key: "—" if result is None else str(result) for key, result in row.items()}
        seconds = row["elapsed_seconds"]
        lines.append(f"| {arm} | {shown['solved']} / {row['assigned']} | {row['first_hits'] or '—'} | "
                     f"{shown['executed_primitives']} / {shown['live_primitives']} | {shown['requests']} | "
                     f"{shown['charged_tokens']} | {'—' if seconds is None else f'{seconds:.3f}'} | {row['statuses']} |")
    (directory / "RESULTS.md.tmp").write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(directory / "RESULTS.md.tmp", directory / "RESULTS.md")
    return report
