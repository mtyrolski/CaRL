"""Predeclare and evaluate the pilot's matched real-local direct validation."""

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from carl.environment.craftax import CraftaxAdapter
from carl.icl.api import write_json
from carl.icl.experiments.intent_pilot import verify
from carl.icl.intent_domain import SokobanSnapshot, decode_intent as decode_sokoban_intent, render_instruction as sokoban_text
from carl.icl.intents.generation.data import file_sha256
from carl.icl.planning.craftax_tasks import CraftaxIntent, CraftaxScenario, render_instruction as craftax_text
from carl.icl.planning.local_direct import CraftaxDirect, DirectLimits, SokobanDirect, load_direct_controller, run_direct
from carl.icl.types import JSONObject


LIMITS = DirectLimits()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cells(rows: list[str], symbols: str) -> set[tuple[int, int]]:
    return {(y, x) for y, line in enumerate(rows) for x, tile in enumerate(line) if tile in symbols}


def _sokoban_case(trace: JSONObject, skill: str, slot: int) -> JSONObject:
    frames = trace["observations"]
    for index, (before, after) in enumerate(zip(frames, frames[1:])):
        keep = _cells(before, "*") & _cells(after, "*")
        if skill == "walk_to_push_position":
            old_player, = _cells(before, "@+")
            player, = _cells(after, "@+")
            boxes = _cells(after, "$*")
            if old_player == player or boxes != _cells(before, "$*"):
                continue
            y, x = player
            if not any(abs(y - by) + abs(x - bx) == 1 for by, bx in boxes):
                continue
            effect = {"predicate": "player_at", "row": y, "col": x}
        else:
            new_goals = _cells(after, "*") - _cells(before, "*")
            if len(new_goals) != 1 or not keep:
                continue
            (y, x), = new_goals
            effect = {"predicate": "box_at", "row": y, "col": x}
        preserve = []
        if keep:
            y, x = min(keep)
            preserve.append({"predicate": "box_at", "row": y, "col": x})
        intent = decode_sokoban_intent({"effects": [effect], "preserve": preserve})
        return {
            "case_id": f"sokoban-{slot}", "skill": skill, "initial_rows": before, "intent": intent,
            "source_id": trace["episode_id"], "task_group": trace["group_id"], "start_index": index,
        }
    raise ValueError(f"No eligible {skill} context in frozen Sokoban source {trace['episode_id']}")


def freeze_validation(root: Path) -> JSONObject:
    """Choose all validation starts before either controller sees any task."""
    directory = root / "direct"
    directory.mkdir(parents=True, exist_ok=True)
    sokoban_file = root / "annotations/sokoban/private/episodes.json"
    collection = _read(root / "annotations/sokoban/split_manifest.json")
    if file_sha256(sokoban_file) != collection["locked_files"]["private/episodes.json"]:
        raise ValueError("Sokoban validation evidence differs from the frozen native traces")
    traces = [trace for trace in _read(sokoban_file) if trace["pilot_split"] == "validation"]
    if len(traces) != 4 or len({row["group_id"] for row in traces}) != 4:
        raise ValueError("Require four disjoint Sokoban validation layouts")
    skills = ("walk_to_push_position", "walk_to_push_position", "fill_and_keep_goal", "fill_and_keep_goal")
    sokoban = [_sokoban_case(trace, skill, slot) for slot, (trace, skill) in enumerate(zip(traces, skills, strict=True), 1)]

    craftax = []
    native_file = root / "splits/craftax-validation.json"
    for slot, row in enumerate(_read(native_file)["episodes"], 1):
        task = row["task"]
        if task["task_id"].startswith("cx-val-01"):
            effect = {"predicate": "tool_tier_at_least", "tool": "sword", "tier": 1}
            skill = "craft_wood_sword"
        elif task["task_id"].startswith("cx-val-02"):
            effect = {"predicate": "inventory_at_least", "item": "wood", "count": task["scenario"]["inventory"]["wood"] + 1}
            skill = "gather_wood"
        else:
            raise ValueError("Unknown frozen Craftax validation chain")
        intent = CraftaxIntent.model_validate({
            "effects": [effect], "preserve": [{"predicate": "inventory_at_least", "item": "wood", "count": 1}],
        })
        craftax.append({
            "case_id": f"craftax-{slot}", "skill": skill, "source_id": row["episode_id"], "world_id": row["world_id"],
            "seed": task["seed"], "scenario": task["scenario"], "intent": intent.model_dump(mode="json"),
        })
    if len(craftax) != 4 or len({row["world_id"] for row in craftax}) != 4:
        raise ValueError("Require four distinct native validation worlds")
    result = {
        "schema": "intent-pilot-direct-validation-v1", "split_sha256": verify(root)["split_sha256"],
        "inputs": {str(path.relative_to(root)): file_sha256(path) for path in (native_file, sokoban_file)},
        "limits": asdict(LIMITS), "cases": {"sokoban": sokoban, "craftax": craftax},
    }
    path = directory / "validation.json"
    if path.exists() and _read(path) != result:
        raise ValueError("The predeclared direct validation roster has changed")
    if not path.exists():
        write_json(path, result)
    return {"path": str(path), "sha256": file_sha256(path), "cases": {domain: len(rows) for domain, rows in result["cases"].items()}}


def _aggregate(rows: list[JSONObject]) -> JSONObject:
    calls = [call for row in rows for call in row["result"]["decisions"] if "input_tokens" in call]
    warm = [call["latency_seconds"] for call in calls if not call["payload"]["cold_load"]]
    cold = [call["latency_seconds"] for call in calls if call["payload"]["cold_load"]]
    summary: JSONObject = {"attempts": len(rows), "successes": sum(row["result"]["success"] for row in rows)}
    for name in ("errors", "loops", "input_tokens", "output_tokens", "executed_primitives", "executed_no_ops",
                 "requested_primitives", "native_ticks"):
        summary[name] = sum(row["result"][name] for row in rows)
    summary.update(
        model_seconds=sum(call["latency_seconds"] for call in calls),
        cold_seconds=cold[0] if cold else None, warm_median_seconds=statistics.median(warm) if warm else None,
    )
    summary["skills"] = {
        skill: {"successes": sum(row["result"]["success"] for row in rows if row["skill"] == skill),
                "attempts": sum(row["skill"] == skill for row in rows)}
        for skill in sorted({row["skill"] for row in rows})
    }
    return summary


def _completed_prefix(result: JSONObject, domain: str, cases: list[JSONObject]) -> list[JSONObject]:
    records: list[JSONObject] = result["domains"].get(domain, {}).get("records", [])
    completed = [(row["case_id"], row["skill"]) for row in records]
    planned = [(case["case_id"], case["skill"]) for case in cases]
    if completed != planned[:len(completed)]:
        raise ValueError(f"Completed {domain} results are not a prefix of frozen validation")
    return records


def run_validation(root: Path, config: Path, output: Path, *, resume: bool = False) -> JSONObject:
    """Run the exact same four starts in each domain with one shared weight instance."""
    roster_path = root / "direct/validation.json"
    roster = _read(roster_path)
    reference = freeze_validation(root)
    if roster["split_sha256"] != verify(root)["split_sha256"] or roster["limits"] != asdict(LIMITS):
        raise ValueError("The frozen direct validation contract differs")
    if output.exists() and not resume:
        raise FileExistsError(f"Direct run output already exists: {output}")
    role, identity = load_direct_controller(config)
    header: JSONObject = {
        "schema": "intent-pilot-direct-report-v1", "execution_mode": "real-local",
        "validation_sha256": reference["sha256"], "limits": roster["limits"],
    }
    if resume:
        result = _read(output)
        pinned = ("model_id", "requested_model_id", "identity", "threads", "seed")
        if any(result.get(key) != value for key, value in header.items()) or any(
            result["model"][key] != identity[key] for key in pinned
        ):
            raise ValueError("Direct resume requires the exact frozen validation and controller")
        completed = {domain: len(_completed_prefix(result, domain, cases))
                     for domain, cases in roster["cases"].items()}
        if all(completed[domain] == len(cases) for domain, cases in roster["cases"].items()):
            raise ValueError("All direct validation starts are already complete")
        result.setdefault("resumes", []).append({
            "prior_report_sha256": file_sha256(output), "completed_starts": completed,
            "runner_sha256": file_sha256(Path(__file__)),
        })
    else:
        result = {**header, "model": identity, "domains": {}}

    def save(domain: str, rows: list[JSONObject], preparation: JSONObject) -> None:
        result["domains"][domain] = {"records": rows, "totals": _aggregate(rows), "preparation": preparation}
        write_json(output, result)

    sokoban_rows = _completed_prefix(result, "sokoban", roster["cases"]["sokoban"])
    pending_sokoban = roster["cases"]["sokoban"][len(sokoban_rows):]
    for case in pending_sokoban:
        board = SokobanSnapshot(tuple(case["initial_rows"]))
        role.assert_query(board.observation, split="validation", task_group=case["task_group"])
        sokoban_intent = decode_sokoban_intent(case["intent"])
        task = {"instruction": sokoban_text(sokoban_intent), "effects": sokoban_intent["effects"],
                "preserve": sokoban_intent["preserve"]}
        outcome = run_direct(role, SokobanDirect(board, sokoban_intent), task, LIMITS)
        sokoban_rows.append({"case_id": case["case_id"], "skill": case["skill"], "result": outcome})
        save("sokoban", sokoban_rows, {})

    craftax_rows = _completed_prefix(result, "craftax", roster["cases"]["craftax"])
    pending = roster["cases"]["craftax"][len(craftax_rows):]
    if not pending:
        return {name: record["totals"] for name, record in result["domains"].items()}
    if resume and not pending_sokoban:
        # The initial sequence reaches Craftax with the same weights already warm.
        started = time.perf_counter()
        payload: JSONObject = {}
        response, inputs, outputs = role.complete_chat(
            [{"role": "user", "content": "Reply with OK."}], output_tokens=8, context_limit=128,
            on_payload=payload.update,
        )
        result["resumes"][-1]["controller_warmup"] = {
            "seconds": time.perf_counter() - started, "input_tokens": inputs, "output_tokens": outputs,
            "response": response, "payload": payload,
        }
        write_json(output, result)
    started = time.perf_counter()
    adapter = CraftaxAdapter(max_steps=32, map_size=16)
    roots = [adapter.stage(adapter.reset(case["seed"]), **CraftaxScenario.model_validate(case["scenario"]).model_dump())
             for case in pending]
    warmup = adapter.step(roots[0], "NOOP")
    preparation = {"seconds": time.perf_counter() - started, "warmup_native_ticks": warmup.native_ticks}
    if craftax_rows:
        previous = result["domains"]["craftax"]["preparation"]
        preparation = {
            "seconds": previous["seconds"] + preparation["seconds"],
            "warmup_native_ticks": previous["warmup_native_ticks"] + preparation["warmup_native_ticks"],
        }
    save("craftax", craftax_rows, preparation)
    for case, snapshot in zip(pending, roots, strict=True):
        intent = CraftaxIntent.model_validate(case["intent"])
        serialized = intent.model_dump(mode="json")
        task = {"instruction": craftax_text(intent), "effects": serialized["effects"], "preserve": serialized["preserve"]}
        outcome = run_direct(role, CraftaxDirect(adapter, snapshot, intent), task, LIMITS)
        craftax_rows.append({"case_id": case["case_id"], "skill": case["skill"], "result": outcome})
        save("craftax", craftax_rows, preparation)
    return {name: record["totals"] for name, record in result["domains"].items()}
