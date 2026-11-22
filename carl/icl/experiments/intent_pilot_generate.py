"""Freeze matched generator arms and validate proposals with the measured policy."""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from carl.environment.craftax import CraftaxAdapter, CraftaxSnapshot
from carl.icl.api import BudgetExceeded, write_json
from carl.icl.config import ROOT, LocalModelSettings
from carl.icl.experiments.intent_pilot_attempts import DOMAINS, MODELS, freeze, read
from carl.icl.experiments.intent_pilot_measure import ELIGIBLE, LocalDecider, load_plan
from carl.icl.intent_domain import TASK_GOAL, SokobanSnapshot, decode_intent
from carl.icl.intents.generation.data import file_sha256
from carl.icl.local_models import bind_local_role
from carl.icl.planning.active_intent import INTENT_CAP, IntentBudget, IntentSession
from carl.icl.planning.craftax_tasks import CraftaxIntent, CraftaxTask, recipe_objectives, render_instruction
from carl.icl.planning.intent_value import FittedIntentValue, fit_value, load_value
from carl.icl.planning.local_direct import CraftaxDirect, SokobanDirect
from carl.icl.planning.paired_generator import SCALES, Arm, admissible, effect_id, messages, proposals, score, system_prompt
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.policy_relative.measurements import trial_seed
from carl.icl.types import JSONObject


SEED = 2026112206
GENERATOR_TOKENS = 64
GENERATOR_CONTEXT = 2048
DISCOUNT = 0.99
CONSTANT_VALUE: JSONObject = {"kind": "constant-value-diagnostic-v1", "value": 0.5, "discount": DISCOUNT}
ARMS: tuple[Arm, ...] = ("unlabelled", "relative")


def _rows(path: Path) -> list[JSONObject]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _example(row: JSONObject, source: str) -> JSONObject:
    stats = row.get("policy_measurements", {})
    measured = stats.get("u") if stats.get("scale_supervision") and stats.get("context_complete") else None
    # Unranked but reached rows carry their observed success probability for diagnostic selection.
    weight = stats.get("w") if measured is not None else stats.get("p", 1.0)
    return {"record_id": row["record_id"], "source": source, "observation": row["start_observation"],
            "task": row["task_goal"], "intent": row["intent"], "measured_u": measured,
            "selection_weight": weight,
            "weight_kind": "measured_w" if measured is not None else "observed_p" if stats else "uniform_demo"}


def _support(demo: list[JSONObject], measured: list[JSONObject]) -> tuple[list[JSONObject], int, int]:
    da = [_example(row, "D_A") for row in demo if row["pilot_split"] == "train-support" and row["validation"]["accepted"]]
    reached = [_example(row, "D_C") for row in measured if row["pilot_split"] == "train-measurement" and
               row["policy_measurements"]["successes"] > 0]

    def ordering(row: JSONObject) -> tuple[float, str]:
        return -row["selection_weight"], contract_hash([SEED, row["record_id"]])

    ranked = sorted((row for row in reached if row["measured_u"] is not None),
                    key=lambda row: (row["measured_u"], *ordering(row)))
    selected = [ranked[0], ranked[-1]] if ranked and ranked[0]["measured_u"] != ranked[-1]["measured_u"] else []
    selected_ids = {row["record_id"] for row in selected}
    selected.extend(row for row in sorted(reached, key=ordering) if row["record_id"] not in selected_ids)
    return [*sorted(da, key=ordering)[:2], *selected[:2]], len(reached), len(ranked)


def _options(domain: str, observation: list[str] | JSONObject, intents: Sequence[JSONObject]) -> list[JSONObject]:
    distinct: dict[str, JSONObject] = {}
    for h in intents:
        intent = admissible(domain, h, observation)
        distinct.setdefault(effect_id(intent), intent)
    order = sorted(distinct, key=lambda key: contract_hash([SEED, observation, key]))
    return [{"id": chr(ord("A") + index), "intent": distinct[key]} for index, key in enumerate(order[:6])]


def _sokoban_options(rows: list[str]) -> list[JSONObject]:
    y, x = next((row, col) for row, tiles in enumerate(rows) for col, tile in enumerate(tiles) if tile in "@+")
    occupied = [(row, col) for row, tiles in enumerate(rows) for col, tile in enumerate(tiles) if tile == "*"]
    intents = []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        near = y + dy, x + dx
        if not (0 <= near[0] < len(rows) and 0 <= near[1] < len(rows[0])):
            continue
        tile = rows[near[0]][near[1]]
        if tile in "-.":
            effect = {"predicate": "player_at", "row": near[0], "col": near[1]}
        elif tile in "$*":
            far = y + 2 * dy, x + 2 * dx
            if not (0 <= far[0] < len(rows) and 0 <= far[1] < len(rows[0])) or rows[far[0]][far[1]] not in "-.":
                continue
            effect = {"predicate": "box_at", "row": far[0], "col": far[1]}
        else:
            continue
        keep = [{"predicate": "box_at", "row": row, "col": col} for row, col in occupied if (row, col) != near][:1]
        intents.append({"effects": [effect], "preserve": keep, "instruction": None})
    return _options("sokoban", rows, intents)


def _validation_cases(root: Path) -> dict[str, list[JSONObject]]:
    directory = root / "annotations/sokoban"
    path = directory / "private/episodes.json"
    if file_sha256(path) != read(directory / "split_manifest.json")["locked_files"]["private/episodes.json"]:
        raise ValueError("Sokoban validation witnesses no longer match their checked snapshot package")
    used = {tuple(row["initial_rows"]) for row in read(root / "direct/validation.json")["cases"]["sokoban"]}
    used |= {tuple(row["start_observation"]) for row in _rows(directory / "D_A.jsonl") if row["split"] == "validation"}
    validation = read(root / "splits/sokoban-validation.json")["episodes"]
    train_groups = {r["canonical_layout_id"] for r in read(root / "splits/sokoban-train-support.json")["episodes"]}
    allowed = {row["episode_id"]: row["canonical_layout_id"] for row in validation}
    sokoban = []
    for trace in read(path):
        source = trace["episode_id"]
        if trace["pilot_split"] != "validation" or source not in allowed:
            continue
        group = allowed[source]
        if group in train_groups or group != trace["group_id"]:
            raise ValueError("A generator validation group overlaps training membership")
        for rows in trace["observations"]:
            if tuple(rows) in used or not any("$" in row for row in rows):
                continue
            options = _sokoban_options(rows)
            if len(options) >= 2 and any(o["intent"]["effects"][0]["predicate"] == "box_at" for o in options):
                sokoban.append({"case_id": contract_hash([source, rows]), "source_id": source, "group_id": group,
                                "observation": rows, "task": TASK_GOAL, "options": options})
                break
        if len(sokoban) == 2:
            break
    if len(sokoban) != 2:
        raise ValueError("Need two fresh validation boards with diverse visible options")
    craftax = []
    seen = set()
    for row in read(root / "splits/craftax-validation.json")["episodes"]:
        if row["chain_id"] not in seen:
            craftax.append({"case_id": contract_hash([row["world_id"], "RIGHT"]), "source_id": row["episode_id"],
                            "group_id": row["chain_id"], "native_task": row["task"], "setup_action": "RIGHT"})
            seen.add(row["chain_id"])
    return {"sokoban": sokoban, "craftax": craftax}


def _value(root: Path, output: Path) -> JSONObject:
    source = root / "splits/sokoban-train-support.json"
    trajectories = [{**row["demonstration"], "split": "train", "task_group": row["canonical_layout_id"]}
                    for row in read(source)["episodes"]]
    data = output / "private/value-training.json"
    freeze(data, {"trajectories": trajectories})
    path = output / "value.json"
    if path.exists():
        value = load_value(path, discount=DISCOUNT)
        if value.data_sha256 != file_sha256(data):
            raise ValueError("The value fit was trained on a different source")
    else:
        earlier = root / "generator/value.json"
        value = load_value(earlier, discount=DISCOUNT) if earlier.exists() else fit_value(data, discount=DISCOUNT)
        if value.data_sha256 != file_sha256(data):
            raise ValueError("The value fit was trained on a different source")
        freeze(path, value.identity())
    return value.identity()


def freeze_plan(root: Path, *, edition: int = 1) -> JSONObject:
    measured, pool_hash = load_plan(root)
    output = root / ("generator" if edition == 1 else f"generator-v{edition}")
    (output / "private").mkdir(parents=True, exist_ok=True, mode=0o700)
    cases = _validation_cases(root)
    cells = {}
    for model in MODELS:
        policy = measured["policies"][model]
        for domain in DOMAINS:
            directory = root / "measurements" / model / domain
            data = directory / "D_C.jsonl"
            summary = read(directory / "manifest.json")
            if (file_sha256(data) != summary["sha256"]["D_C.jsonl"] or summary["provenance"]["pool_sha256"] != pool_hash
                    or summary["provenance"]["policy_sha256"] != policy["sha256"]):
                raise ValueError("A directed measurement cell differs from the audited policy and pool")
            support, reached, ranked = _support(_rows(root / "hindsight" / domain / "D_A.jsonl"), _rows(data))
            labelled_values = {row["measured_u"] for row in support if row["measured_u"] is not None}
            validation_groups = {case["group_id"] for case in cases[domain]}
            if (validation_groups & {case["group_id"] for case in measured["contexts"][domain]} or
                    validation_groups & {row["group_id"] for row in read(root / "hindsight/candidate-pool-v1.json")["candidates"][domain]}):
                raise ValueError("A validation context overlaps generator support or policy measurements")
            cells[f"{model}/{domain}"] = {
                "model": model, "domain": domain, "model_id": policy["contract"]["model_id"],
                "policy_sha256": policy["sha256"], "policy": policy["contract"],
                "dc_sha256": file_sha256(data), "examples": support, "reached_candidates": reached, "ranked_candidates": ranked,
                "arms": {"unlabelled": "matched" if len(labelled_values) >= 2 else "diagnostic" if reached else "no_reached_examples",
                         "relative": "matched" if len(labelled_values) >= 2 else "no_informative_scale"},
                "generator": {**policy["contract"]["model"], "role": "generator",
                              "prompts": {arm: system_prompt(domain, arm, edition=edition) for arm in ARMS},
                              "history": [], "context_limit": GENERATOR_CONTEXT, "output_tokens": GENERATOR_TOKENS},
            }
    plan = {
        "schema": "carl-pilot-paired-generator-v1", "inputs": measured["inputs"], "directed_pool_sha256": pool_hash,
        "scales": list(SCALES), "proposal_slots": len(SCALES), "calls_per_arm_context": 1, "seed": SEED,
        "scheduler": "fixed-slot-interleaved-native-rollouts-v1", "score": "verified_hit_discounted_endpoint_value",
        "intent_cap": INTENT_CAP, "cells": cells, "validation": cases,
        "values": {"sokoban": _value(root, output), "craftax": CONSTANT_VALUE},
        "prompt_code_sha256": file_sha256(ROOT / "carl/icl/planning/paired_generator.py"),
    }
    if edition != 1:
        plan["prompt_edition"] = edition
    freeze(output / "private/plan.json", plan)
    return {"config_sha256": file_sha256(output / "private/plan.json"),
            "cells": {name: cell["arms"] for name, cell in cells.items()}}


@dataclass
class Validation:
    case: JSONObject
    observation: list[str] | JSONObject
    task: str
    options: list[JSONObject]
    native: CraftaxSnapshot | None = None
    goal: CraftaxIntent | None = None


def _runtime_cases(domain: str, cases: list[JSONObject]) -> tuple[list[Validation], CraftaxAdapter | None]:
    if domain == "sokoban":
        return [Validation(row, row["observation"], row["task"], row["options"]) for row in cases], None
    adapter = CraftaxAdapter(map_size=16, max_steps=32)
    result = []
    for row in cases:
        task = CraftaxTask.model_validate(row["native_task"])
        start = adapter.stage(adapter.reset(task.seed), **task.scenario.model_dump())
        changed = adapter.step(start, row["setup_action"])
        if changed.observation == adapter.observe(start) or changed.terminated or changed.truncated:
            raise ValueError("The fresh native validation observation is unavailable")
        menu = _options(domain, changed.observation, [intent.model_dump(mode="json")
                                                       for intent in recipe_objectives(changed.observation, task.goal)])
        result.append(Validation(row, changed.observation, render_instruction(task.goal), menu, changed.snapshot, task.goal))
    return result, adapter


def _rollout(
    validation: Validation, domain: str, adapter: CraftaxAdapter | None, h: JSONObject, future: int, decide: LocalDecider,
    path: Path, remaining: float, value: FittedIntentValue, metadata: JSONObject,
) -> None:
    budget = IntentBudget(INTENT_CAP, 32, 32, remaining)
    if domain == "sokoban":
        assert isinstance(validation.observation, list)
        start = SokobanSnapshot(tuple(validation.observation))
        session = IntentSession(SokobanDirect(start, decode_intent(h)), validation.task, budget)
    else:
        assert adapter is not None and validation.native is not None
        stream = trial_seed(SEED, validation.case["case_id"], future) % 2**32
        start_native = adapter.fork_future(validation.native, stream)
        if start_native.physical_fingerprint != validation.native.physical_fingerprint:
            raise ValueError("Validation future changed the physical start")
        session = IntentSession(CraftaxDirect(adapter, start_native, CraftaxIntent.model_validate(h)), validation.task,
                                budget, validation.goal)

    def checkpoint(record: JSONObject) -> None:
        write_json(path, {**metadata, **record})

    record = session.attempt(str(metadata["proposal_id"]), h, decide, stop_on_invalid=True, checkpoint=checkpoint)
    endpoint_value = None
    if record["status"] in ELIGIBLE:
        if domain == "sokoban":
            assert isinstance(session.domain, SokobanDirect)
            endpoint_value = value(session.domain.snapshot.observation, validation.task)
        else:
            endpoint_value = CONSTANT_VALUE["value"]
    checkpoint({**record, "endpoint_value": endpoint_value, "score": score(record["first_hit"], endpoint_value, DISCOUNT)})


def run(root: Path, wall_seconds: int, *, edition: int = 1) -> JSONObject:
    import torch
    from carl.icl.experiments.intent_pilot_generate_report import publish

    output = root / ("generator" if edition == 1 else f"generator-v{edition}")
    frozen_path = output / "private/plan.json"
    plan = read(frozen_path)
    frozen = freeze_plan(root, edition=edition)
    config_sha = frozen["config_sha256"]
    value = load_value(output / "value.json", discount=DISCOUNT)
    started = time.monotonic()
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for name, cell in plan["cells"].items():
        arms: list[Arm] = [arm for arm in ARMS if cell["arms"][arm] in ("diagnostic", "matched")]
        if not arms:
            continue
        model, domain = cell["model"], cell["domain"]
        validations, adapter = _runtime_cases(domain, plan["validation"][domain])
        settings = LocalModelSettings.model_validate(read(root / "direct" / f"{model}.json")["local"])
        role = bind_local_role(settings, "policy")
        if role.identity() != cell["policy"]["model"]:
            raise ValueError("The generator backend differs from the frozen controller weights")
        torch.manual_seed(cell["policy"]["policy_seed"])
        cache = output / "private/policy-cache" / cell["policy_sha256"]
        freeze(cache / "greedy-check.json", read(root / "measurements/private/cache" / cell["policy_sha256"] / "greedy-check.json"))
        decide = LocalDecider(role, cache)
        batches = []
        for validation in validations:
            if domain == "sokoban":
                assert isinstance(validation.observation, list)
                value.assert_held_out(SokobanSnapshot(tuple(validation.observation)).observation,
                                      validation.case["group_id"], split="validation")
            for arm in arms:
                if time.monotonic() - started >= wall_seconds - 15:
                    return publish(root, edition=edition)
                case_id = validation.case["case_id"]
                call = output / "private/calls" / name / f"{case_id}-{arm}.json"
                if call.exists():
                    batches.append((validation, read(call)))
                    continue
                wire = messages(domain, arm, validation.observation, validation.task, cell["examples"], validation.options,
                                edition=edition)
                metadata = {"config_sha256": config_sha, "cell": name, "case_id": case_id, "arm": arm, "role": "generator"}
                write_json(call, {**metadata, "status": "running", "messages": wire})
                payload: JSONObject = {}
                try:
                    text, n_in, n_out = role.complete_chat(wire, output_tokens=GENERATOR_TOKENS,
                                                          context_limit=GENERATOR_CONTEXT, on_payload=payload.update)
                except (OSError, RuntimeError, ValueError) as error:
                    batch = {**metadata, **proposals(domain, arm, "", validation.observation, validation.options),
                             "status": "context_limit" if isinstance(error, BudgetExceeded) else "call_error", "error": str(error)}
                else:
                    batch = {**metadata, **proposals(domain, arm, text, validation.observation, validation.options),
                             "status": "finished", "response": text, "input_tokens": n_in, "output_tokens": n_out, "payload": payload}
                write_json(call, {**batch, "messages": wire})
                batches.append((validation, batch))
                print(name, case_id[:8], arm, batch["status"], batch["distinct_effects"], flush=True)
                publish(root, edition=edition)
        for index in range(len(SCALES)):
            for validation, batch in batches:
                if "slots" not in batch or batch["slots"][index]["status"] != "valid":
                    continue
                proposal = batch["slots"][index]
                for future in range(2 if domain == "craftax" else 1):
                    remaining = wall_seconds - (time.monotonic() - started)
                    if remaining <= 15:
                        return publish(root, edition=edition)
                    identifier = contract_hash([name, batch["case_id"], batch["arm"], index, future])
                    path = output / "private/rollouts" / name / f"{identifier}.json"
                    if path.exists():
                        continue
                    _rollout(validation, domain, adapter, proposal["intent"], future, decide, path, remaining, value,
                             {"config_sha256": config_sha, "cell": name, "case_id": batch["case_id"],
                              "arm": cast(Arm, batch["arm"]), "index": index, "future": future, "proposal_id": identifier})
                    print(name, batch["case_id"][:8], batch["arm"], index, read(path)["status"], flush=True)
                    publish(root, edition=edition)
    return publish(root, edition=edition)
