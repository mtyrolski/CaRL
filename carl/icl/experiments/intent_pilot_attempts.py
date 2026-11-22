"""Collect failed assigned intents on the pilot's frozen training support."""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carl.environment.craftax import CraftaxAdapter, CraftaxSnapshot
from carl.icl.api import write_json
from carl.icl.config import LocalModelSettings
from carl.icl.experiments.intent_pilot_annotation_sources import _advance
from carl.icl.intent_domain import TASK_GOAL, Action, SokobanSnapshot, annotation_json, decode_intent, replay_fragment
from carl.icl.intents.generation.data import file_sha256
from carl.icl.local_models import bind_local_role
from carl.icl.planning.active_intent import INTENT_CAP, ActiveDomain, Decide, IntentBudget, IntentSession
from carl.icl.planning.craftax_tasks import CraftaxIntent, CraftaxTask, render_instruction
from carl.icl.planning.local_direct import CraftaxDirect, SokobanDirect
from carl.icl.types import JSONObject


MODELS = ("qwen3-0.6b", "qwen2.5-1.5b-instruct")
DOMAINS = ("sokoban", "craftax")


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(path: Path, value: object) -> None:
    """Write an immutable pilot artefact or check its existing contents."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=path.parent, prefix=".freeze-") as directory:
            candidate = Path(directory) / path.name
            write_json(candidate, value)
            try:
                path.hardlink_to(candidate)
            except FileExistsError:
                pass
    if annotation_json(read(path)) != annotation_json(value):
        raise ValueError(f"Frozen hindsight artefact differs at {path}")


def _positions(rows: Sequence[str], symbols: str) -> set[tuple[int, int]]:
    return {(y, x) for y, row in enumerate(rows) for x, tile in enumerate(row) if tile in symbols}


def _box(cell: tuple[int, int]) -> JSONObject:
    return {"predicate": "box_at", "row": cell[0], "col": cell[1]}


def _sokoban_context(traces: list[JSONObject], *, controlled: bool) -> JSONObject:
    for trace in traces:
        rows = trace["observations"]
        goals = [_positions(observation, "*") for observation in rows]
        for start, action in enumerate(trace["actions"]):
            if start == trace["start"] or (not controlled and action != int(Action.RIGHT)):
                continue
            first = goals[start + 1] - goals[start]
            kept = goals[start + 1] & goals[start]
            if len(first) != 1 or not kept:
                continue
            (y, x), = _positions(rows[start + 1], "@+")
            block = next((name for name, dy, dx in (("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1))
                          if rows[start + 1][y + dy][x + dx] == "#"), None)
            if controlled and block is None:
                continue
            for end in range(start + 2, min(len(rows), start + INTENT_CAP + 1)):
                kept = kept & goals[end]
                if not kept or not first <= goals[end]:
                    break
                others = goals[end] - goals[start] - first
                if others:
                    second = _box(min(others))
                    preserve = [_box(min(kept))]
                    return {
                        "episode_id": trace["episode_id"], "group_id": trace["group_id"], "pilot_split": "train-support",
                        "rows": rows[start], "start_index": start, "task_goal": TASK_GOAL,
                        "requested_h": {"effects": [_box(next(iter(first))), second], "preserve": preserve},
                        "script": [Action(action).name, *([block] * (INTENT_CAP - 1))] if controlled else None,
                        "replan": {"effects": [second], "preserve": preserve},
                        "replan_script": [Action(a).name for a in trace["actions"][start + 1:end]] if controlled else None,
                    }
    raise ValueError("No eligible two-goal context in the frozen Sokoban training traces")


def _craftax_context(trace: JSONObject, *, controlled: bool) -> JSONObject:
    task = CraftaxTask.model_validate(trace["native_task"])
    requested = task.goal.model_dump(mode="json")
    extra: JSONObject = {}
    if controlled:
        preserve = [{"predicate": "tool_tier_at_least", "tool": "pickaxe", "tier": 1}]
        requested = {"effects": [{"predicate": "inventory_at_least", "item": "stone", "count": 3}], "preserve": preserve}
        extra = {
            "script": ["LEFT", "DO", *(["NOOP"] * (INTENT_CAP - 2))],
            "replan": requested, "replan_script": ["UP", "DO"],
        }
    return {
        "episode_id": trace["episode_id"], "group_id": trace["group_id"], "world_id": trace["world_id"],
        "pilot_split": "train-support", "native_task": trace["native_task"], "task_goal": render_instruction(task.goal),
        "setup_actions": ["DO"] if controlled else [], "requested_h": requested, **extra,
    }


def plan(root: Path, split: str) -> JSONObject:
    """Lock train starts, model configs and protocol before collection."""
    paths = {}
    training = {}
    for domain in DOMAINS:
        source = root / "annotations" / domain
        trace_file = source / "private/episodes.json"
        split_manifest = read(source / "split_manifest.json")
        if split_manifest["base_split_sha256"] != split or file_sha256(trace_file) != split_manifest["locked_files"]["private/episodes.json"]:
            raise ValueError(f"{domain} training source differs from the frozen split")
        training[domain] = [row for row in read(trace_file) if row["pilot_split"] == "train-support" and not row["fixture"]]
        if {row["group_id"] for row in training[domain]} != set(split_manifest["group_membership"]["train-support"]):
            raise ValueError(f"{domain} training groups differ from the frozen split")
        paths[domain] = {
            "trace_sha256": file_sha256(trace_file),
            "da_sha256": file_sha256(source / "D_A.jsonl"),
            "manifest_sha256": file_sha256(source / "manifest.json"),
        }
        if paths[domain]["da_sha256"] != read(source / "manifest.json")["file_sha256"]["D_A.jsonl"]:
            raise ValueError(f"{domain} D_A differs from the verified dataset")
    native = {row["episode_id"]: row for row in training["craftax"]}
    sokoban = _sokoban_context(training["sokoban"], controlled=False)
    sokoban_script = _sokoban_context([row for row in training["sokoban"] if row["group_id"] != sokoban["group_id"]], controlled=True)
    result = {
        "schema": "intent-pilot-hindsight-plan-v1", "base_split_sha256": split, "inputs": paths,
        "intent_cap": INTENT_CAP, "session_limits": {"max_actions": 24, "max_calls": 3, "max_ticks": 24, "wall_seconds": 150},
        "controller_configs": {name: file_sha256(root / "direct" / f"{name}.json") for name in MODELS},
        "contexts": {
            "sokoban": {"local": sokoban, "scripted_train": sokoban_script},
            "craftax": {"local": _craftax_context(native["cx-stage03-craft-2"], controlled=False),
                        "scripted_train": _craftax_context(native["cx-stage03-gather-stone"], controlled=True)},
        },
    }
    private = root / "hindsight/private"
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    freeze(private / "plan.json", result)
    return result


@dataclass(frozen=True)
class TrainingContext:
    domain: str
    spec: JSONObject
    start: SokobanSnapshot | CraftaxSnapshot
    create: Callable[[], ActiveDomain]
    craftax_task: CraftaxIntent | None = None

    def session(self, budget: IntentBudget) -> IntentSession:
        return IntentSession(self.create(), self.spec["task_goal"], budget, self.craftax_task)

    def replay(self, record: JSONObject) -> JSONObject:
        states = record["states"]
        steps = record["steps"]
        if isinstance(self.start, SokobanSnapshot):
            compact_actions = [int(Action[step["requested"]]) for step in steps]
            check = replay_fragment(self.start, compact_actions)
            if [list(frame.rows) for frame in check] != [state["observation"] for state in states]:
                raise ValueError("Sokoban attempt did not match native replay")
            return {"observations": [list(frame.rows) for frame in check], "actions": compact_actions}
        domain = self.create()
        assert isinstance(domain, CraftaxDirect)
        actions: list[str] = [entry["requested"] for entry in steps]
        observations, native_states, native_steps = _advance(domain.adapter, self.start, actions)
        if observations != [state["observation"] for state in states] or (
            [state["fingerprint"] for state in native_states] != [state["state_id"] for state in states]
        ):
            raise ValueError("Craftax attempt did not match same-key native replay")
        for recorded, native in zip(steps, native_steps, strict=True):
            comparisons = {"applied": native["native_action"], "executed": native["executed_primitive"],
                           "native_ticks": native["native_ticks"], "automatic_ticks": native["automatic_ticks"]}
            if any(recorded[key] != value for key, value in comparisons.items()):
                raise ValueError("Craftax attempt disagrees with native action accounting")
        return {"observations": observations, "actions": actions, "steps": native_steps, "native_states": native_states}


def _context(domain: str, spec: JSONObject, adapter: CraftaxAdapter | None = None) -> TrainingContext:
    if domain == "sokoban":
        board = SokobanSnapshot(tuple(spec["rows"]))
        intent = decode_intent(spec["requested_h"])
        return TrainingContext(domain, spec, board, lambda: SokobanDirect(board, intent))
    assert adapter is not None
    task = CraftaxTask.model_validate(spec["native_task"])
    snapshot = adapter.stage(adapter.reset(task.seed), **task.scenario.model_dump())
    for action in spec["setup_actions"]:
        snapshot = adapter.step(snapshot, action).snapshot
    native_h = CraftaxIntent.model_validate(spec["requested_h"])
    return TrainingContext(domain, spec, snapshot, lambda: CraftaxDirect(adapter, snapshot, native_h), task.goal)


def _script(actions: Sequence[str]) -> Decide:
    pending = iter(actions)

    def decide(_messages: list[dict[str, str]]) -> JSONObject:
        action = next(pending, None)
        if action is None:
            raise ValueError("Controlled policy exhausted its script")
        return {"response": json.dumps({"actions": [action], "intent": "continue this assignment"})}

    return decide


def _record(private: Path, context: TrainingContext, result: JSONObject, source: JSONObject, *, eligible: bool) -> None:
    record = {
        **result, "domain": context.domain, "pilot_split": context.spec["pilot_split"],
        "source_episode_id": context.spec["episode_id"], "group_id": context.spec["group_id"],
        "world_id": context.spec.get("world_id"), "public_task": context.spec["task_goal"],
        "provenance": source, "training_eligible": eligible,
    }
    path = private / "attempts" / f"{record['attempt_id']}.json"
    freeze(path, record)
    evidence = context.replay(record)
    freeze(private / "replays" / path.name, {"attempt_sha256": file_sha256(path), **evidence})
    print(f"{record['attempt_id']} {record['status']} {record['executed_primitives']}", flush=True)


def _recover(private: Path, context: TrainingContext, attempt_id: str) -> JSONObject | None:
    path = private / "attempts" / f"{attempt_id}.json"
    if not path.exists():
        return None
    record: JSONObject = read(path)
    if record["source_episode_id"] != context.spec["episode_id"] or record["domain"] != context.domain:
        raise ValueError("Existing attempt has a different frozen context")
    replay = private / "replays" / path.name
    if replay.exists():
        if read(replay)["attempt_sha256"] != file_sha256(path):
            raise ValueError("Existing attempt differs from its verified replay")
    else:
        freeze(replay, {"attempt_sha256": file_sha256(path), **context.replay(record)})
    return record


def _controlled(private: Path, context: TrainingContext) -> None:
    domain, spec = context.domain, context.spec
    source = {"kind": "scripted_train_support", "fixture": False}
    prefix = f"scripted-{domain}"
    fixture_path = private / "fixtures" / f"{domain}.json"
    saved = _recover(private, context, prefix)
    if saved is not None and fixture_path.exists():
        return
    session = context.session(IntentBudget(32, 32, 32, 180))
    attempt = session.attempt(prefix, spec["requested_h"], _script(spec["script"]))
    if saved is None:
        _record(private, context, attempt, source, eligible=True)
    elif attempt["steps"] != saved["steps"] or attempt["continuation"] != saved["continuation"]:
        raise ValueError("The controlled live state differs from the original attempt")
    if attempt["status"] != "intent_timeout":
        raise ValueError(f"Controlled {domain} rollout did not reach its intent timeout")

    replanned = session.attempt(f"fixture-replan-{domain}", spec["replan"], _script(spec["replan_script"]))
    if replanned["states"][0] != attempt["continuation"] or replanned["budget_before"] != attempt["budget_after"]:
        raise ValueError("Live replan did not retain actual state, memory and remaining budget")
    branch = context.session(IntentBudget(32, 32, 32, 180)).attempt(
        f"fixture-branch-{domain}", spec["requested_h"], _script(spec["script"]), speculative=True,
    )
    if branch["status"] != "intent_timeout" or branch["continuation"] != branch["states"][0]:
        raise ValueError("Failed speculative attempt did not retain its frontier checkpoint")
    freeze(fixture_path, {
        "classification": "integration_only", "source_attempt": prefix, "source_trajectory_kind": source["kind"],
        "live_timeout_status": attempt["status"], "speculative_attempt": branch, "live_replan": replanned,
    })


def collect(root: Path, split: str) -> JSONObject:
    """Share one native runtime across both real-local policies and the controls."""
    import torch

    frozen = plan(root, split)
    private = root / "hindsight/private"
    configs = {model: read(root / "direct" / f"{model}.json") for model in MODELS}
    threads, = {config["threads"] for config in configs.values()}
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)

    started = time.monotonic()
    adapter = CraftaxAdapter(map_size=16, max_steps=32)
    contexts = {
        domain: {kind: _context(domain, spec, adapter) for kind, spec in frozen["contexts"][domain].items()}
        for domain in DOMAINS
    }
    preparation = time.monotonic() - started

    for model, config in configs.items():
        pending = [domain for domain in DOMAINS
                   if _recover(private, contexts[domain]["local"], f"{model}-{domain}") is None]
        if not pending:
            continue
        torch.manual_seed(73)
        settings = LocalModelSettings.model_validate(config["local"])
        if settings.adapter is not None or settings.device != "cpu":
            raise ValueError("This collection requires the pinned base CPU controllers")
        role = bind_local_role(settings, "policy")

        def decide(messages: list[dict[str, str]], role: Any = role) -> JSONObject:
            call_started = time.monotonic()
            payload: JSONObject = {}
            raw, inputs, outputs = role.complete_chat(messages, output_tokens=96, context_limit=1024, on_payload=payload.update)
            return {"response": raw, "input_tokens": inputs, "output_tokens": outputs, "payload": payload,
                    "latency_seconds": time.monotonic() - call_started}

        source = {"kind": "real_local_policy", "fixture": False, "model_id": config["model_id"], "model": role.identity()}
        for domain in pending:
            context = contexts[domain]["local"]
            attempt_id = f"{model}-{domain}"
            session = context.session(IntentBudget(**frozen["session_limits"]))
            if isinstance(session.domain, SokobanDirect):
                # Base-only rollout. Preserve adapter evaluation guards in the public validation path.
                if role.adapter is not None:
                    raise ValueError("The frozen train support policy cannot use an evaluation adapter")
            record = session.attempt(attempt_id, context.spec["requested_h"], decide)
            _record(private, context, record, source, eligible=True)
    for domain in DOMAINS:
        _controlled(private, contexts[domain]["scripted_train"])
    summary = {"attempts": 6, "control_and_native_setup_seconds": preparation,
               "plan_sha256": file_sha256(private / "plan.json"), "models": list(MODELS)}
    if not (private / "collection.json").exists():
        freeze(private / "collection.json", summary)
    return summary
