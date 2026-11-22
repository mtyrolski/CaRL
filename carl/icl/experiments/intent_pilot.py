"""Freeze whole-episode membership before the offline pilot extracts fragments."""

import fcntl
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows, validate_subgoal
from carl.icl.config import ROOT
from carl.icl.planning.craftax_tasks import CraftaxTask, load_tasks


SPLITS = ("train-support", "train-measurement", "validation", "sealed-test")
LEGACY_SPLITS = {split: role for split, role in zip(SPLITS, ("train", "train", "validation", "test"), strict=True)}
SEEDS = {"sokoban_selection": 2026112201, "craftax_world": 2026112202, "fragment_extraction": 2026112203}
SCHEMA = "intent-pilot-splits-v1"
MOVES = {"U": 0, "D": 1, "L": 2, "R": 3}
GAMMA2 = ROOT / "analysis/icl_reachability/proposal_hard_sources_v1/12"
FIXTURE_ROOT = ROOT / "configs/icl"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _rows(board: str | list[str]) -> list[str]:
    return [line.replace(" ", "-") for line in (board.splitlines() if isinstance(board, str) else board)]


def _actions(actions: str | list[int]) -> list[int]:
    result = [MOVES[action] for action in actions] if isinstance(actions, str) else actions
    if not result or any(
        isinstance(action, bool) or not isinstance(action, int) or action not in range(4) for action in result
    ):
        raise ValueError("Sokoban witness must contain native action indices")
    return result


def _sokoban_sources() -> Iterator[tuple[str, str, tuple[Path, ...], str | list[str], str | list[int], int | None]]:
    support = ROOT / "data/icl/campaign_dataset_v2/demonstrations.json"
    support_witness = ROOT / "data/icl/campaign_dataset_v2_private/demonstration/real_12x12x4.json"
    witnesses = {record["board_id"]: record["actions"] for record in _read(support_witness)["witnesses"]}
    for row in _read(support)["instances"]:
        yield SPLITS[0], row["trajectory_id"], (support, support_witness), row["state"], witnesses[row["board_id"]], row["seed"]

    for witness in sorted((ROOT.parent / "data/witnesses").glob("gamma1_s*.json")):
        board = witness.parent.parent / "boards" / f"{witness.stem}.txt"
        trace = _read(witness)
        yield SPLITS[1], trace["id"], (witness, board), board.read_text(), trace["actions"], int(witness.stem.split("_s")[1])

    validation = ROOT / "data/icl/generated_12x12x10/validation.json"
    validation_witness = ROOT / "data/icl/private_witnesses/generated_12x12x10/validation.json"
    witnesses = {record["id"]: record["actions"] for record in _read(validation_witness)}
    for row in _read(validation)["instances"]:
        yield SPLITS[2], row["id"], (validation, validation_witness), row["state"], witnesses[row["id"]], None

    for path in sorted(GAMMA2.glob("*.json")):
        source = _read(path)
        if source["settings"]["family"] != "gamma2":
            raise ValueError("The sealed Sokoban source pool must be the gamma2 family")
        yield SPLITS[3], source["id"], (path,), source["ascii"], source["actions"], source["seed"]


def _sokoban_fixtures() -> tuple[set[str], dict[str, str]]:
    demonstrations = FIXTURE_ROOT / "quickstart/sokoban_demonstrations.json"
    queries = FIXTURE_ROOT / "quickstart/sokoban_test_queries.json"
    published_example = GAMMA2 / "110278.json"
    boards = [row["start_observation"] for row in _read(demonstrations)["trajectories"]]
    boards.extend(_read(queries).values())
    boards.append(_read(published_example)["ascii"])
    return (
        {canonical_layout_id(from_rows(_rows(board))) for board in boards},
        {str(path.relative_to(ROOT)): _file_digest(path) for path in (demonstrations, queries, published_example)},
    )


def _sokoban() -> tuple[dict[str, list[dict]], dict]:
    families = dict(zip(SPLITS, ("real_12x12x4", "gamma1_12x12x10", "generated_12x12x10", "gamma2_12x12x10"), strict=True))
    cohorts: dict[str, list[dict]] = {split: [] for split in SPLITS}
    fixtures, fixture_files = _sokoban_fixtures()
    sources: dict[str, str] = {}
    omitted: Counter[str] = Counter()
    owners: dict[str, str] = {}
    layouts: set[str] = set()
    episodes: set[str] = set()
    for split, episode, inputs, board, actions, seed in _sokoban_sources():
        source_paths = []
        for path in inputs:
            name = str(path.relative_to(ROOT.parent))
            source_paths.append(name)
            if name not in sources:
                sources[name] = _file_digest(path)
        rows = _rows(board)
        state = from_rows(rows)
        validate_subgoal(state, state)
        layout = canonical_layout_id(state)
        if layout in fixtures:
            omitted["fixture_layout"] += 1
            continue
        # Source aliases and all rotated/reflected versions share their partition.
        keys = (f"episode:{episode}", f"layout:{layout}", f"family:{families[split]}")
        if any(owners.setdefault(key, split) != split for key in keys):
            raise ValueError("Sokoban source episode, symmetric layout or task family crosses splits")
        if layout in layouts or episode in episodes:
            omitted["duplicate_episode_or_layout"] += 1
            continue
        layouts.add(layout)
        episodes.add(episode)
        cohorts[split].append({
            "episode_id": episode, "task_family": families[split], "canonical_layout_id": layout,
            "source_files_from_workspace": source_paths, "source_seed": seed,
            "demonstration": {
                "source_id": episode, "task_group": families[split], "split": LEGACY_SPLITS[split],
                "start_observation": rows, "actions": _actions(actions),
            },
        })
    for split, rows in cohorts.items():
        rows.sort(key=lambda row: _digest([SEEDS["sokoban_selection"], row["episode_id"], row["canonical_layout_id"]]))
        if split == "sealed-test":
            omitted["outside_seeded_sealed_sample"] += max(0, len(rows) - 8)
            del rows[8:]
    return cohorts, {"sources_from_workspace": sources, "fixture_files": fixture_files, "omitted": dict(omitted)}


def _predicates(requirements: dict[str, int]) -> list[dict]:
    return [
        {"predicate": "tool_tier_at_least", "tool": item, "tier": amount}
        if item in {"pickaxe", "sword"} else {"predicate": "inventory_at_least", "item": item, "count": amount}
        for item, amount in sorted(requirements.items())
    ]


def _craftax(path: Path) -> tuple[dict[str, list[dict]], dict]:
    spec = _read(path)
    fixture_file = FIXTURE_ROOT / "craftax/tasks.json"
    fixtures = load_tasks(fixture_file)
    example_chain = fixtures.by_id("wood-to-stone-pickaxe").chain_signature
    fixture_seeds = {task.seed for task in fixtures.tasks}
    if set(spec["groups"]) != set(SPLITS):
        raise ValueError("Craftax source roster must declare all four partitions")
    cohorts: dict[str, list[dict]] = {split: [] for split in SPLITS}
    owners: dict[tuple[tuple[str, ...], ...], str] = {}
    worlds: set[str] = set()
    ids: set[str] = set()
    for split in SPLITS:
        for item in spec["groups"][split]:
            for repeat in range(spec["worlds_per_chain"]):
                seed = int(_digest([SEEDS["craftax_world"], item["id"], repeat])[:8], 16)
                world_id = _digest(["craftax-1.1.0-symbolic", spec["map_size"], seed])
                episode_id = f"{item['id']}-{repeat}"
                task = CraftaxTask.model_validate({
                    "task_id": episode_id, "split": LEGACY_SPLITS[split], "seed": seed,
                    "scenario": {
                        "inventory": {**spec["base_inventory"], **item["inventory"]},
                        "tiles": spec["tiles"], "facing": spec["facing"],
                    },
                    "chain": [{"effects": _predicates(step)} for step in item["milestones"]],
                    "goal": {"effects": _predicates(item["goal"]), "preserve": _predicates(item.get("preserve", {}))},
                })
                chain = task.chain_signature
                if seed in fixture_seeds or chain == example_chain:
                    raise ValueError("Craftax cohort overlaps a documented fixture")
                if owners.setdefault(chain, split) != split or world_id in worlds or episode_id in ids:
                    raise ValueError("Craftax chain, world or episode is repeated across the frozen roster")
                worlds.add(world_id)
                ids.add(episode_id)
                cohorts[split].append({
                    "episode_id": episode_id, "world_id": world_id, "chain_id": _digest(chain),
                    "map_size": spec["map_size"], "max_steps": spec["max_steps"],
                    "task": task.model_dump(mode="json"), "status": "scheduled_native_episode",
                })
    known_skills = {skill for chain, split in owners.items() if split.startswith("train") for stage in chain for skill in stage}
    if any(skill not in known_skills for chain, split in owners.items()
           if split == "sealed-test" for stage in chain for skill in stage):
        raise ValueError("A sealed Craftax chain contains a skill absent from training")
    return cohorts, {"roster_sha256": _file_digest(path), "fixture_sha256": _file_digest(fixture_file)}


def _split_path(domain: str, split: str) -> str:
    directory = "sealed" if split == "sealed-test" else "splits"
    return f"{directory}/{domain}-{split}.json"


def _freeze_file(path: Path, payload: dict) -> None:
    if path.exists():
        if _read(path) != payload:
            raise ValueError(f"Frozen artefact differs at {path}; preserve the original campaign")
    else:
        write_json(path, payload)


def freeze(root: Path) -> dict:
    """Write complete episodes and native world assignments with no fragment extraction."""
    with (root / ".freeze.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sokoban, sokoban_provenance = _sokoban()
        craftax, craftax_provenance = _craftax(root / "private-inputs/craftax-roster.json")
        (root / "sealed").mkdir(mode=0o700, exist_ok=True)
        (root / "splits").mkdir(exist_ok=True)
        provenance = {"sokoban": sokoban_provenance, "craftax": craftax_provenance}
        files = []
        for domain, cohorts in (("sokoban", sokoban), ("craftax", craftax)):
            for split, rows in cohorts.items():
                if not rows:
                    raise ValueError(f"No {domain} episodes for {split}")
                path = _split_path(domain, split)
                payload = {"schema": SCHEMA, "domain": domain, "split": split, "episodes": rows}
                _freeze_file(root / path, payload)
                files.append({
                    "domain": domain, "split": split, "path": path, "episodes": len(rows),
                    "sha256": _file_digest(root / path),
                })
        manifest = {"schema": SCHEMA, "seeds": SEEDS, "files": files, "provenance": provenance}
        _freeze_file(root / "splits.json", manifest)
        return manifest


def verify(root: Path) -> dict:
    """Verify split integrity and return only unsealed paths plus aggregate counts and hashes."""
    path = root / "splits.json"
    manifest = _read(path)
    paths = {_split_path(domain, split) for domain in ("sokoban", "craftax") for split in SPLITS}
    if manifest["schema"] != SCHEMA or manifest["seeds"] != SEEDS:
        raise ValueError("Unexpected split schema or seeds")
    if sorted(row["path"] for row in manifest["files"]) != sorted(paths):
        raise ValueError("Missing or unexpected frozen split files")
    result: dict[str, Any] = {"split_sha256": _file_digest(path), "seeds": SEEDS, "splits": []}
    for row in manifest["files"]:
        if row["path"] != _split_path(row["domain"], row["split"]) or row["episodes"] < 1:
            raise ValueError("Invalid split manifest entry")
        if _file_digest(root / row["path"]) != row["sha256"]:
            raise ValueError(f"{row['domain']} {row['split']} differs from the frozen split")
        result["splits"].append({key: val for key, val in row.items()
                                 if key != "path" or row["split"] != "sealed-test"})
    return result
