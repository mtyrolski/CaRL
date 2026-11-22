"""Freeze and execute a small matched, directed native controller block."""

import time
from pathlib import Path
from uuid import uuid4

from filelock import FileLock

from carl.environment.craftax import CraftaxAdapter, CraftaxSnapshot
from carl.icl.api import write_json
from carl.icl.boards import to_rows
from carl.icl.config import ROOT, LocalModelSettings
from carl.icl.data_sources.sokoban import environment
from carl.icl.experiments.intent_pilot import verify as verify_split
from carl.icl.experiments.intent_pilot_attempts import DOMAINS, MODELS, _positions, freeze, read
from carl.icl.experiments.intent_pilot_budget import PilotLedger
from carl.icl.intent_domain import TASK_GOAL, Action, SokobanSnapshot, decode_intent, validate_intent_proposal
from carl.icl.intents.generation.data import file_sha256
from carl.icl.local_models import LocalRoleModel, bind_local_role
from carl.icl.planning.active_intent import HISTORY_CAP, INTENT_CAP, SYSTEM_PROMPT, IntentBudget, IntentSession, digest
from carl.icl.planning.craftax_tasks import CraftaxIntent, CraftaxTask, predicate_holds, render_instruction
from carl.icl.planning.local_direct import CRAFTAX_RULES, SOKOBAN_RULES, CraftaxDirect, SokobanDirect
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.policy_relative.measurements import DIRECTED_FAILURES, trial_seed
from carl.icl.types import JSONObject


SEED = 2026112205
OUTPUT_TOKENS = 64
CONTEXT_LIMIT = 1024
INITIAL_MEMORY: JSONObject = {"history": [], "note": None, "feedback": None}
ELIGIBLE = DIRECTED_FAILURES | {"intent_reached"}


def _inputs(root: Path) -> JSONObject:
    ledger = PilotLedger(root)
    split = verify_split(root)["split_sha256"]
    pool_path = root / "hindsight/candidate-pool-v1.json"
    pool = read(pool_path)
    pool_hash = file_sha256(pool_path)
    if split != ledger.split_digest() or split != pool["base_split_sha256"] or pool_hash != ledger.candidate_pool_digest(1):
        raise ValueError("The frozen campaign split or support pool differs from its ledger identity")
    annotations = {domain: file_sha256(root / "hindsight" / domain / "D_A.jsonl") for domain in DOMAINS}
    if annotations != read(root / "hindsight/pool-manifest.json")["da_sha256"]:
        raise ValueError("The checked D_A files differ from the frozen support pool")
    return {"split_sha256": split, "support_pool_sha256": pool_hash, "da_sha256": annotations,
            "measurement_split_sha256": {d: file_sha256(root / f"splits/{d}-train-measurement.json") for d in DOMAINS}}


def _candidate(context_id: str, kind: str, intent: JSONObject, parent: JSONObject) -> JSONObject:
    return {"candidate_id": contract_hash([context_id, kind, intent]), "type": kind, "intent": intent,
            "support_candidate_id": parent["candidate_id"], "support_source_kind": parent["source_kind"]}


def _sokoban_context(source: JSONObject, templates: dict[str, JSONObject]) -> JSONObject | None:
    demo = source["demonstration"]
    root = SokobanSnapshot(tuple(demo["start_observation"]))
    board = root.restore()
    simulator = environment(board)
    frames = [root]
    for action in demo["actions"]:
        if simulator.is_solved(board):
            break
        successor = simulator.next_state(board, action)
        if isinstance(successor, str):
            raise ValueError("The numeric Sokoban witness produced a string state")
        board = successor
        frames.append(SokobanSnapshot(tuple(to_rows(board))))
    goals = [_positions(frame.rows, "*") for frame in frames]
    boxes = [_positions(frame.rows, "$*") for frame in frames]
    # Probe short reachable witnesses. The actions and future frames never enter model messages.
    starts = sorted(range(len(frames) - 1), key=lambda i: demo["actions"][i] != int(Action.RIGHT))
    for start in starts:
        added = goals[start + 1] - goals[start]
        kept = goals[start] & goals[start + 1]
        if len(added) != 1 or not kept or len(goals[start + 1]) <= len(goals[start]):
            continue
        for end in range(start + 2, min(len(frames), start + INTENT_CAP + 1)):
            kept &= goals[end]
            if not kept or not added <= goals[end]:
                break
            other = boxes[end] - boxes[start] - added
            if not other:
                continue
            identifier = contract_hash(["sokoban", frames[start].rows, TASK_GOAL])
            first, second, keep = [_positions_to_predicate(min(cells)) for cells in (added, other, kept)]
            effects = {"box_at": [first], "goal_count": [{"predicate": "boxes_on_goals_at_least", "count": len(goals[start]) + 1}],
                       "box_conjunction": [first, second]}
            candidates = [
                _candidate(identifier, kind, {"effects": terms, "preserve": [keep], "instruction": None}, templates[kind])
                for kind, terms in effects.items()
            ]
            if any(validate_intent_proposal(decode_intent(row["intent"]), frames[start].observation) for row in candidates):
                raise ValueError("A selected Sokoban candidate is illegal or already attained")
            return {"context_id": identifier, "source_id": source["episode_id"], "group_id": source["canonical_layout_id"],
                    "start_index": start, "start_observation": list(frames[start].rows), "task_goal": TASK_GOAL,
                    "candidates": candidates, "future_streams": [None]}
    return None


def _positions_to_predicate(cell: tuple[int, int]) -> JSONObject:
    return {"predicate": "box_at", "row": cell[0], "col": cell[1]}


def _sokoban_contexts(sources: list[JSONObject], support: list[JSONObject]) -> list[JSONObject]:
    templates = {
        "box_at": next(row for row in support if row["source_kind"] == "verified_hindsight"),
        "goal_count": next(row for row in support if row["intent"]["effects"][0]["predicate"] == "boxes_on_goals_at_least"),
        "box_conjunction": next(row for row in support if len(row["intent"]["effects"]) == 2),
    }
    contexts = []
    for source in sources:
        context = _sokoban_context(source, templates)
        if context is not None:
            contexts.append(context)
        if len(contexts) == 3:
            return contexts
    raise ValueError("Fewer than three short Sokoban contexts fit the frozen candidate types")


def _tool(tool: str, tier: int) -> JSONObject:
    return {"predicate": "tool_tier_at_least", "tool": tool, "tier": tier}


def _inventory(item: str, count: int) -> JSONObject:
    return {"predicate": "inventory_at_least", "item": item, "count": count}


def _craftax_contexts(sources: list[JSONObject], support: list[JSONObject]) -> list[JSONObject]:
    contexts = []
    seen = set()
    for source in sources:
        chain = source["chain_id"]
        if chain in seen:
            continue
        seen.add(chain)
        task = CraftaxTask.model_validate(source["task"])
        tier = task.scenario.inventory.get("pickaxe", 0) + 1
        identifier = contract_hash(["craftax", source["world_id"], task.goal.model_dump(mode="json")])
        effect_sets = {"tool": [_tool("sword", 1)], "inventory": [_inventory("stone", 1)],
                       "tool_conjunction": [_tool("sword", 1), _tool("pickaxe", tier)]}
        candidates = []
        for kind, effects in effect_sets.items():
            preserve = [_inventory("wood", 1)]
            parent = next(row for row in support if sorted(map(contract_hash, row["intent"]["effects"])) ==
                          sorted(map(contract_hash, effects)) and row["intent"]["preserve"] == preserve)
            candidates.append(_candidate(identifier, kind, parent["intent"], parent))
        streams = [trial_seed(SEED, identifier, index) % 2**32 for index in range(2)]
        if len(set(streams)) != 2:
            raise ValueError("Craftax future key collision")
        contexts.append({"context_id": identifier, "source_id": task.task_id, "group_id": chain,
                         "world_id": source["world_id"], "native_task": source["task"],
                         "task_goal": render_instruction(task.goal), "candidates": candidates, "future_streams": streams})
    if len(contexts) != 3:
        raise ValueError("Expected three distinct Craftax measurement chains")
    return contexts


def _policy_contract(root: Path, model: str) -> JSONObject:
    config = read(root / "direct" / f"{model}.json")
    previous = read(root / "direct" / f"{model}-results.json")["model"]
    code = ("icl/planning/active_intent.py", "icl/planning/local_direct.py", "icl/local_models.py",
            "icl/intent_domain.py", "environment/craftax/adapter.py", "environment/sokoban/env.py")
    return {
        "model_id": config["model_id"], "model": previous["identity"], "config_sha256": file_sha256(root / "direct" / f"{model}.json"),
        "threads": config["threads"], "torch_deterministic": True, "policy_seed": SEED,
        "prompts": {"sokoban": SYSTEM_PROMPT + "\n" + SOKOBAN_RULES, "craftax": SYSTEM_PROMPT + "\n" + CRAFTAX_RULES},
        "examples": [], "initial_memory": INITIAL_MEMORY, "memory_id": digest(INITIAL_MEMORY), "history_cap": HISTORY_CAP,
        "decoding": {"do_sample": False, "enable_thinking": False, "output_tokens": OUTPUT_TOKENS, "context_limit": CONTEXT_LIMIT},
        "effort": {"primitive_cap": INTENT_CAP, "native_tick_cap": 32, "call_cap": 32, "batch_cap": 4,
                   "stop_on_invalid": True, "interrupt": ["progress_changed", "no_op", "automatic_tick"]},
        "code_sha256": {name: file_sha256(ROOT / "carl" / name) for name in code},
    }


def freeze_plan(root: Path) -> JSONObject:
    inputs = _inputs(root)
    pool = read(root / "hindsight/candidate-pool-v1.json")["candidates"]
    sources = {domain: read(root / f"splits/{domain}-train-measurement.json")["episodes"] for domain in DOMAINS}
    contexts = {"sokoban": _sokoban_contexts(sources["sokoban"], pool["sokoban"]),
                "craftax": _craftax_contexts(sources["craftax"], pool["craftax"])}
    for domain, rows in contexts.items():
        groups = {row["group_id"] for row in rows}
        support = {row["group_id"] for row in pool[domain]}
        if len(groups) != 3 or groups & support:
            raise ValueError("Measurement contexts overlap frozen support groups")
    contracts = {model: _policy_contract(root, model) for model in MODELS}
    planned = {
        "schema": "intent-pilot-directed-plan-v1", "inputs": inputs, "seed": SEED, "contexts": contexts,
        "type_quotas": {d: {row["type"]: 1 for row in contexts[d][0]["candidates"]} for d in DOMAINS},
        "policies": {model: {"contract": contract, "sha256": contract_hash(contract)} for model, contract in contracts.items()},
        "planned_trials": sum(len(context["candidates"]) * len(context["future_streams"]) for rows in contexts.values()
                              for context in rows) * len(MODELS),
    }
    directory = root / "measurements"
    (directory / "private").mkdir(parents=True, exist_ok=True, mode=0o700)
    freeze(directory / "private/pool-v1.json", planned)
    return {"pool_sha256": file_sha256(directory / "private/pool-v1.json"), "planned_trials": planned["planned_trials"],
            "type_quotas": planned["type_quotas"], **inputs}


def load_plan(root: Path) -> tuple[JSONObject, str]:
    path = root / "measurements/private/pool-v1.json"
    plan: JSONObject = read(path)
    if plan["inputs"] != _inputs(root):
        raise ValueError("The frozen measurement inputs changed")
    for model, policy in plan["policies"].items():
        if policy["contract"] != _policy_contract(root, model) or contract_hash(policy["contract"]) != policy["sha256"]:
            raise ValueError("The frozen controller protocol changed")
    return plan, file_sha256(path)


class LocalDecider:
    """Reuse only exact greedy model calls; environment futures remain independent."""

    def __init__(self, role: LocalRoleModel, path: Path) -> None:
        self.role, self.path = role, path

    def __call__(self, messages: list[dict[str, str]]) -> JSONObject:
        key = contract_hash(messages)
        path = self.path / "responses" / f"{key}.json"
        if path.is_file():
            return {**read(path), "cache_hit": True, "request_sha256": key}
        result = self._generate(messages)
        proof = self.path / "greedy-check.json"
        if not proof.is_file():
            repeat = self._generate(messages)
            for field in ("response", "input_tokens", "output_tokens"):
                if result[field] != repeat[field] or result["payload"]["input_ids"] != repeat["payload"]["input_ids"]:
                    raise ValueError("The local greedy policy failed its identical-payload repeat")
            freeze(proof, {"request_sha256": key, "response_sha256": contract_hash(result["response"]),
                           "verified": True, "repeat_inference_seconds": repeat["inference_seconds"],
                           "repeat_input_tokens": repeat["input_tokens"], "repeat_output_tokens": repeat["output_tokens"]})
        freeze(path, result)
        return {**result, "cache_hit": False, "request_sha256": key}

    def _generate(self, messages: list[dict[str, str]]) -> JSONObject:
        started = time.monotonic()
        payload: JSONObject = {}
        text, n_in, n_out = self.role.complete_chat(
            messages, output_tokens=OUTPUT_TOKENS, context_limit=CONTEXT_LIMIT, on_payload=payload.update,
        )
        return {"response": text, "input_tokens": n_in, "output_tokens": n_out, "payload": payload,
                "inference_seconds": time.monotonic() - started}


def _native_roots(plan: JSONObject) -> tuple[CraftaxAdapter, dict[str, CraftaxSnapshot]]:
    adapter = CraftaxAdapter(max_steps=32, map_size=16)
    roots = {}
    for case in plan["contexts"]["craftax"]:
        task = CraftaxTask.model_validate(case["native_task"])
        snapshot = adapter.stage(adapter.reset(task.seed), **task.scenario.model_dump())
        facts = adapter.inventory(snapshot)
        for candidate in case["candidates"]:
            h = CraftaxIntent.model_validate(candidate["intent"])
            if (any(predicate_holds(value, facts) is not False for value in h.effects)
                    or any(predicate_holds(value, facts) is not True for value in h.preserve)):
                raise ValueError("A frozen Craftax intent is already attained or has invalid preservation")
        roots[case["context_id"]] = snapshot
    adapter.step(next(iter(roots.values())), "NOOP")
    return adapter, roots


def _slot(model: str, domain: str, case: JSONObject, candidate: JSONObject, index: int) -> JSONObject:
    return {"slot_id": contract_hash([model, domain, candidate["candidate_id"], index]),
            "model": model, "domain": domain, "context_id": case["context_id"], "candidate_id": candidate["candidate_id"],
            "index": index, "stream": case["future_streams"][index]}


def _records(directory: Path) -> list[JSONObject]:
    records = []
    for path in sorted((directory / "private/trials").glob("*/*.json")):
        row = read(path)
        if row["status"] == "running":
            row["status"] = "campaign_interrupted"
            row["first_hit"] = None
        records.append(row)
    return records


def _run_slot(
    directory: Path, run_id: str, metadata: JSONObject, case: JSONObject, candidate: JSONObject,
    adapter: CraftaxAdapter, roots: dict[str, CraftaxSnapshot], decide: LocalDecider, remaining: float,
) -> JSONObject:
    path = directory / "private/trials" / run_id / f"{metadata['slot_id']}.json"
    h = candidate["intent"]
    budget = IntentBudget(INTENT_CAP, 32, 32, remaining)
    if metadata["domain"] == "sokoban":
        start = SokobanSnapshot(tuple(case["start_observation"]))
        physical = future = digest(list(start.rows))
        session = IntentSession(SokobanDirect(start, decode_intent(h)), case["task_goal"], budget)
    else:
        root = roots[case["context_id"]]
        physical = root.physical_fingerprint
        fork = adapter.fork_future(root, metadata["stream"])
        other = adapter.fork_future(root, case["future_streams"][1 - metadata["index"]])
        if fork.physical_fingerprint != physical or other.physical_fingerprint != physical or fork.fingerprint == other.fingerprint:
            raise ValueError("Craftax futures do not share a physical root with distinct native keys")
        future = fork.fingerprint
        goal = CraftaxTask.model_validate(case["native_task"]).goal
        session = IntentSession(CraftaxDirect(adapter, fork, CraftaxIntent.model_validate(h)), case["task_goal"], budget, goal)
    identity = {**metadata, "physical_start_id": physical, "future_id": future, "run_id": run_id,
                "memory_id": digest(INITIAL_MEMORY)}

    def checkpoint(record: JSONObject) -> None:
        write_json(path, {**identity, **record})

    result = session.attempt(metadata["slot_id"], h, decide, cap=INTENT_CAP, stop_on_invalid=True, checkpoint=checkpoint)
    if result["initial_check"] != {"effects": False, "preservation": True, "task_solved": False, "episode_ended": False}:
        raise ValueError("A frozen attempt did not begin in its verified unmet context")
    if result["states"][0]["state_id"] != future or result["states"][0]["memory_id"] != identity["memory_id"]:
        raise ValueError("Attempt started from a different state or controller memory")
    checkpoint(result)
    return {**identity, **result}


def run(root: Path, wall_seconds: int) -> JSONObject:
    """Reserve externally, make progress atomically, and retry only incomplete slots."""
    import torch
    from carl.icl.experiments.intent_pilot_measure_report import publish

    directory = root / "measurements"
    started = time.monotonic()
    with FileLock(str(directory / "private/run.lock"), timeout=0):
        plan, pool_hash = load_plan(root)
        saved = _records(directory)
        complete = {row["slot_id"] for row in saved if row["status"] in ELIGIBLE}
        run_id = uuid4().hex
        torch.set_num_threads(6)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(SEED)
        adapter, roots = _native_roots(plan)
        setup_seconds = time.monotonic() - started
        freeze(directory / "private/native-starts.json", {
            case["context_id"]: {"observation": adapter.observe(roots[case["context_id"]]),
                                 "physical_start_id": roots[case["context_id"]].physical_fingerprint,
                                 "future_ids": [adapter.fork_future(roots[case["context_id"]], stream).fingerprint
                                                for stream in case["future_streams"]]}
            for case in plan["contexts"]["craftax"]
        })
        freeze(directory / "private/sessions" / f"{run_id}.json",
               {"pool_sha256": pool_hash, "wall_seconds": wall_seconds, "native_setup_seconds": setup_seconds})
        policies = {}
        for model, policy in plan["policies"].items():
            config = read(root / "direct" / f"{model}.json")
            role = bind_local_role(LocalModelSettings.model_validate(config["local"]), "policy")
            if role.identity() != policy["contract"]["model"] or config["threads"] != torch.get_num_threads():
                raise ValueError("The loaded policy does not match its planned weights and threads")
            policies[model] = LocalDecider(role, directory / "private/cache" / policy["sha256"])
        for context_index in range(3):
            for candidate_index in range(3):
                for model in MODELS:
                    for domain in DOMAINS:
                        case = plan["contexts"][domain][context_index]
                        candidate = case["candidates"][candidate_index]
                        for index in range(len(case["future_streams"])):
                            metadata = _slot(model, domain, case, candidate, index)
                            if metadata["slot_id"] in complete:
                                continue
                            remaining = wall_seconds - (time.monotonic() - started)
                            if remaining <= 15:
                                return publish(root)
                            metadata.update(pool_sha256=pool_hash, policy_sha256=plan["policies"][model]["sha256"])
                            row = _run_slot(directory, run_id, metadata, case, candidate, adapter, roots, policies[model], remaining)
                            print(model, domain, context_index, candidate["type"], index,
                                  row["status"], row["first_hit"], flush=True)
                        publish(root, progress_only=True)
        return publish(root)
