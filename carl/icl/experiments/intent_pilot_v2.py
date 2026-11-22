"""Freeze a fresh two-start intent diagnostic against the closed v1 release."""

import json
import re
import subprocess
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from random import Random

from carl.icl.boards import canonical_layout_id, from_rows, to_rows
from carl.icl.data_sources.sokoban import environment
from carl.icl.experiments.intent_pilot import verify as verify_v1
from carl.icl.experiments.intent_pilot_annotation_sources import _task
from carl.icl.experiments.intent_pilot_attempts import freeze, read
from carl.icl.experiments.intent_pilot_budget import PilotLedger
from carl.icl.experiments.intent_pilot_measure import load_plan
from carl.icl.intent_domain import (
    TASK_GOAL, Action, SokobanSnapshot, check_intent, decode_intent, predicate_truth,
    render_instruction, replay_fragment, validate_intent_proposal,
)
from carl.icl.intents.generation.data import file_sha256
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject


V1 = Path("analysis/intent-pilot")
V2 = Path("analysis/intent-pilot-v2")
SEED = 2026112214
PROFILE = "qwen3-0.6b"
PARTITIONS = ("support-diagnostic", "calibration", "validation", "sealed-test")
ID_LINE = re.compile(rb'^ {6}"(episode_id|canonical_layout_id|chain_id|world_id)": "([^"]+)",?$')


def jsonl(path: Path) -> list[JSONObject]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def seed(*parts: object) -> int:
    return int(contract_hash([SEED, *parts])[:8], 16)


def _v1_ids() -> dict[str, set[str]]:
    """Read only outer group IDs from the sealed rosters."""
    checked = verify_v1(V1)
    if checked["split_sha256"] != PilotLedger(V1).split_digest():
        raise ValueError("The v1 split differs from the campaign ledger")
    result: dict[str, set[str]] = {"sokoban": set(), "craftax": set()}
    for entry in read(V1 / "splits.json")["files"]:
        domain = entry["domain"]
        path = V1 / entry["path"]
        fields = {"episode_id", "canonical_layout_id"} if domain == "sokoban" else {"episode_id", "chain_id", "world_id"}
        if entry["split"] == "sealed-test":
            with path.open("rb") as stream:
                values = [match.groups() for line in stream
                          if (match := ID_LINE.fullmatch(line.rstrip(b"\r\n"))) is not None
                          and match[1].decode() in fields]
            if len(values) != entry["episodes"] * len(fields):
                raise ValueError(f"Could not identify all sealed {domain} groups")
            result[domain].update(value.decode() for _, value in values)
        else:
            result[domain].update(row[key] for row in read(path)["episodes"] for key in fields)
    for domain, identities in result.items():
        for row in read(V1 / f"annotations/{domain}/private/episodes.json"):
            identities.update(value for key in ("episode_id", "group_id", "world_id") if (value := row.get(key)))
    return result


def _terrain(partition: str, index: int, random_seed: int) -> tuple[list[list[str]], int, int, int, int]:
    """Return the walls and goals. Piece positions are applied only to unsealed diagnostics."""
    family = PARTITIONS.index(partition)
    rng = Random(random_seed)
    col = 3 + rng.randrange(3)
    turn = index == 1
    player_row = 8 + (family == 3)
    goal_row, goal_col = player_row - (2 + family) - (not turn), col + turn
    rows = [list("#" * 12) for _ in range(12)]
    for row in range(1, goal_row + 1):
        for x in range(1 + family % 2, 10 + family % 2):
            rows[row][x] = "-"
    for row in range(goal_row + 1, player_row + 1):
        rows[row][col] = "-"
        if turn and row <= player_row - 1:
            rows[row][col - 1] = "-"
    if turn:
        rows[goal_row][col - 1] = "-"
    kept_col = 8 + rng.randrange(2)
    rows[1][kept_col] = "."
    rows[goal_row][goal_col] = "."
    return rows, player_row, col, kept_col, goal_col


def _sokoban_roster(used: set[str]) -> tuple[list[JSONObject], list[JSONObject]]:
    roster, cases = [], []
    for partition in PARTITIONS:
        for index in range(2):
            for retry in range(64):
                random_seed = seed("sokoban", partition, index, retry)
                terrain, y, col, keep, goal_col = _terrain(partition, index, random_seed)
                layout = canonical_layout_id(from_rows(["".join(row) for row in terrain]))
                if layout not in used:
                    break
            else:
                raise ValueError(f"No distinct Sokoban layout for {partition}/{index}")
            used.add(layout)
            item = {
                "domain": "sokoban", "partition": partition, "index": index, "seed": random_seed,
                "family": f"local-tunnel-v2/{partition}", "group_id": layout,
                "case_id": f"v2-sokoban-{partition}-{index}",
            }
            roster.append(item)
            if partition == "support-diagnostic":
                terrain[y][col], terrain[y - 1][col], terrain[1][keep] = "@", "$", "*"
                board = ["".join(row) for row in terrain]
                # Straight tunnel needs two upward pushes. The other endpoint is reached after a side-step and a turn.
                near, far = (y - 2, col), (y - (2 if index == 1 else 3), goal_col)
                targets = (("A", near), ("B", far)) if index == 0 else (("A", far), ("B", near))
                requests = [{
                    "request_id": name,
                    "intent": {"effects": [{"predicate": "box_at", "row": row, "col": x}],
                               "preserve": [{"predicate": "box_at", "row": 1, "col": keep}], "instruction": None},
                } for name, (row, x) in targets]
                cases.append({**item, "observation": board, "task_goal": TASK_GOAL, "requests": requests,
                              "context_id": contract_hash([layout, board, TASK_GOAL])})
    return roster, cases


def _craftax_roster(used: set[str]) -> list[JSONObject]:
    # The frozen chains vary by resource order, not count increments of the same chain.
    resources = [("wood", "TREE"), ("stone", "STONE"), ("coal", "COAL"), ("iron", "IRON")]
    options = [(x, y) for x in resources for y in resources if x != y]
    options.sort(key=lambda pair: seed("craftax-chain", pair))
    roster: list[JSONObject] = []
    for partition in PARTITIONS:
        for items in options:
            start = seed("craftax-world", partition, items)
            identifier = f"v2-craftax-{partition}"
            task = _task(identifier, start, {"pickaxe": 2},
                         dict(zip(("RIGHT", "UP"), (tile for _, tile in items), strict=True)),
                         "RIGHT", [(item, 1) for item, _ in items])
            chain = contract_hash(task.chain_signature)
            world = contract_hash(["craftax-1.1.0-symbolic", 16, start])
            if chain in used or world in used:
                continue
            used.update((chain, world))
            entry = {"domain": "craftax", "partition": partition, "seed": start, "case_id": identifier,
                     "group_id": chain, "world_id": world, "map_size": 16, "max_steps": 32}
            if partition != "sealed-test":
                entry["task"] = task.model_dump(mode="json")
            roster.append(entry)
            break
        else:
            raise ValueError(f"No distinct Craftax world and chain for {partition}")
    return roster


def _witness(rows: list[str], predicate: JSONObject, limit: int) -> tuple[Action, ...]:
    """Bounded native BFS with preservation checked on every transition."""
    start = SokobanSnapshot(tuple(rows))
    intent = decode_intent(predicate)
    if errors := validate_intent_proposal(intent, start.observation):
        raise ValueError(f"The requested diagnostic intent is inadmissible: {errors}")
    env = environment(start.restore())
    queue: deque[tuple[SokobanSnapshot, tuple[Action, ...]]] = deque([(start, ())])
    visited = {start.rows}
    while queue:
        snapshot, path = queue.popleft()
        if all(predicate_truth(effect, snapshot.observation) is True for effect in intent["effects"]):
            return path
        if len(path) == limit or env.is_solved(snapshot.restore()):
            continue
        for action in Action:
            state = env.next_state(snapshot.restore(), int(action))
            successor = SokobanSnapshot(tuple(to_rows(state)))
            if successor.rows not in visited and all(predicate_truth(p, successor.observation) is True for p in intent["preserve"]):
                visited.add(successor.rows)
                queue.append((successor, (*path, action)))
    raise ValueError(f"No preserved native witness in {limit} actions")


def certify(cases: Sequence[JSONObject]) -> list[JSONObject]:
    evidence = []
    for case in cases:
        paths = []
        lengths = []
        for request in case["requests"]:
            intent = decode_intent(request["intent"])
            start = SokobanSnapshot(tuple(case["observation"]))
            actions = _witness(case["observation"], request["intent"], 4)
            snapshots = replay_fragment(start, [int(action) for action in actions])
            if check_intent(intent, [snap.observation for snap in snapshots]).status != "accepted":
                raise ValueError("A minimum-action native witness failed verification")
            paths.append({"request_id": request["request_id"], "actions": [a.name for a in actions],
                          "observations": [list(snap.rows) for snap in snapshots],
                          "instruction": render_instruction(intent), "minimum_primitives": len(actions),
                          "turns": sum(left != right for left, right in zip(actions, actions[1:]))})
            lengths.append(len(actions))
        if sorted(lengths)[0] != 1 or not 2 <= max(lengths) <= 4:
            raise ValueError("Expected one 1-step and one 2–4-step request")
        evidence.append({"case_id": case["case_id"], "context_id": case["context_id"], "paths": paths})
    if not any(path["turns"] for item in evidence for path in item["paths"]):
        raise ValueError("The diagnostic needs a longer witness with a change of direction")
    return evidence


def prepare() -> JSONObject:
    subprocess.run(["sha256sum", "-c", "delivery/RELEASE.sha256"], cwd=V1, check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=V1 / "final/sealed-report", check=True)
    groups = _v1_ids()
    sokoban, cases = _sokoban_roster(groups["sokoban"])
    craftax = _craftax_roster(groups["craftax"])
    support = {}
    for domain in ("sokoban", "craftax"):
        source = V1 / f"hindsight/{domain}/D_A.jsonl"
        rows = jsonl(source)
        training = [row["record_id"] for row in rows if row["pilot_split"] == "train-support" and row["validation"]["accepted"]]
        support[domain] = {"source": str(source), "sha256": file_sha256(source), "checked_rows": len(rows),
                           "training_ids": training}
    freeze(V2 / "splits.json", {
        "schema": "intent-pilot-v2-splits", "seed": SEED, "v1_release_sha256": file_sha256(V1 / "DATASET_MANIFEST.json"),
        "v1_split_sha256": file_sha256(V1 / "splits.json"), "inherited_training": support,
        "groups": [*sokoban, *craftax], "sealed_evaluation": "unopened_group_descriptors_only",
    })
    policy, _ = load_plan(V1)
    profile = policy["policies"][PROFILE]
    plan = {"schema": "intent-pilot-v2-diagnostic", "profile": profile, "split_sha256": file_sha256(V2 / "splits.json"),
            "cases": cases, "attempt_wall_seconds": 75, "arm": "assigned-intent"}
    freeze(V2 / "private/diagnostic-plan.json", plan)
    evidence = certify(cases)
    freeze(V2 / "private/native-witnesses.json", evidence)
    return {"split_sha256": plan["split_sha256"], "pool_sha256": file_sha256(V2 / "private/diagnostic-plan.json"),
            "profile_sha256": profile["sha256"], "cases": len(cases),
            "native_lengths": {row["case_id"]: {path["request_id"]: path["minimum_primitives"] for path in row["paths"]}
                               for row in evidence}}
