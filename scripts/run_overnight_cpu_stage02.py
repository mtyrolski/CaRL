#!/usr/bin/env python3
"""Reuse stage-one CLLP traces and evaluate new full-state targets on CPUs.

Demonstrations are imported, never re-run. New BFS targets receive the same
frozen controller input, model, decoding, action budget and stopping rules.
Every completed candidate/model attempt is committed separately, so an
interruption leaves both the original results and new completed work intact.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from filelock import FileLock
from transformers import BertForSequenceClassification

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from scripts import run_overnight_cpu_policy as stage1
from scripts.figure1_cllp_matched import Pair
from scripts.figure1_cllp_matched import model_predictor
from scripts.figure1_cllp_matched import rollout_pairs
from scripts.overnight_cpu_memo import MemoPredictor

MODELS = (stage1.ModelId.CLLP_8, stage1.ModelId.CLLP_32)
REQUIRED = {
    "candidate_id", "context_id", "layout_id", "family", "source_family", "split",
    "board_rank", "context_rank", "source", "source_hash", "target", "target_hash",
    "candidate_origin", "demo_pair_ids", "demo_offsets", "in_fixed_pool",
}


def selected_candidates(rows: list[dict[str, Any]], split: str, max_boards: int | None) -> list[dict[str, Any]]:
    """Freeze demonstrations, selected BFS targets, and every valid generated target.

    A generated target may already be represented by a BFS candidate outside
    the fixed pool. Preserve its origin, but schedule the shared target once.
    """
    if max_boards is not None and max_boards < 1:
        raise ValueError("max-boards must be positive")
    identifiers: set[str] = set()
    targets: set[tuple[str, str]] = set()
    split_by_layout: dict[str, str] = {}
    contexts: dict[str, tuple[str, str, str]] = {}
    selected = []
    for row in rows:
        if not REQUIRED <= row.keys():
            raise ValueError(f"Candidate missing fields: {sorted(REQUIRED - row.keys())}")
        cid = row["candidate_id"]
        if not isinstance(cid, str) or len(cid) != 64 or any(c not in "0123456789abcdef" for c in cid):
            raise ValueError("candidate_id must be a lowercase SHA-256 digest")
        if cid in identifiers:
            raise ValueError("Duplicate candidate_id")
        identifiers.add(cid)
        if row["split"] not in ("train", "development", "evaluation") or row["family"] not in stage1.FAMILY_BOXES:
            raise ValueError("Unknown split or board family")
        if split_by_layout.setdefault(row["layout_id"], row["split"]) != row["split"]:
            raise ValueError("Canonical layout occurs in multiple splits")
        if (not isinstance(row["board_rank"], int) or isinstance(row["board_rank"], bool)
                or row["board_rank"] < 0 or row["context_rank"] not in (0, 1)):
            raise ValueError("Invalid frozen board or context rank")
        identity = (row["layout_id"], row["source_hash"], row["split"])
        if contexts.setdefault(row["context_id"], identity) != identity:
            raise ValueError("Context identifies multiple source states")
        target_key = (row["context_id"], row["target_hash"])
        if target_key in targets:
            raise ValueError("Duplicate full target in one context")
        targets.add(target_key)
        if (not isinstance(row["in_fixed_pool"], bool)
                or row["candidate_origin"] not in ("demonstration", "bfs", "generator")
                or not isinstance(row["demo_pair_ids"], list) or not isinstance(row["demo_offsets"], list)):
            raise ValueError("Invalid candidate provenance")
        also_generated = row.get("also_generated", False)
        generator_ranks = row.get("generator_ranks", [])
        if (not isinstance(also_generated, bool) or not isinstance(generator_ranks, list)
                or any(not isinstance(rank, int) or isinstance(rank, bool) or rank < 0 for rank in generator_ranks)
                or (also_generated and not generator_ranks)):
            raise ValueError("Invalid generated-target alias")
        if row["candidate_origin"] == "demonstration" and not row["demo_pair_ids"]:
            raise ValueError("A demonstration must retain its old pair IDs")
        if row["candidate_origin"] != "demonstration" and (row["demo_pair_ids"] or row["demo_offsets"]):
            raise ValueError("A new target cannot have demonstration offsets")
        if (row["split"] == split and (max_boards is None or row["board_rank"] < max_boards)
                and (row["candidate_origin"] != "bfs" or row["in_fixed_pool"] or also_generated)):
            selected.append(row)
    if not selected:
        raise ValueError(f"No eligible candidates for split {split!r}")
    family_order = {family: i for i, family in enumerate(stage1.FAMILY_BOXES)}
    return sorted(selected, key=lambda row: (row["board_rank"], family_order[row["family"]],
                                             row["context_rank"], row["candidate_id"]))


def validate_candidate(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Validate the exact full board supplied to a controller."""
    source, target = from_rows(row["source"]), from_rows(row["target"])
    if source.shape != (12, 12, 7) or target.shape != source.shape:
        raise ValueError("Expected 12x12 full-state boards")
    expected = stage1.FAMILY_BOXES[row["family"]]
    for state in (source, target):
        cells = state.argmax(-1)
        if (int(np.isin(cells, (3, 4)).sum()) != expected
                or int(np.isin(cells, (2, 3, 6)).sum()) != expected):
            raise ValueError("Wrong box or goal count")
    if (canonical_layout_id(source) != row["layout_id"]
            or state_hash(source) != row["source_hash"] or state_hash(target) != row["target_hash"]):
        raise ValueError("Candidate board or layout hash mismatch")
    validate_subgoal(source, target)
    if np.array_equal(source, target) or environment(source).is_solved(source):
        raise ValueError("Candidate is trivial or source is terminal")
    return source, target


def load_reference(directory: Path) -> dict[str, Any]:
    """Verify the old frozen plan, data, model files and execution code."""
    contract = json.loads((directory / "contract.json").read_text())
    plan = json.loads((directory / "plan.json").read_text())
    if contract["plan_sha256"] != stage1.digest(plan):
        raise ValueError("Stage-one plan hash mismatch")
    pairs_path = Path(contract["input_pairs_path"])
    if stage1.file_sha(pairs_path) != contract["input_sha256"]:
        raise ValueError("Stage-one input pairs hash mismatch")
    expected = {
        "action_cap": 32, "action_mask": False, "batch_size": 16, "blas_threads": 2,
        "decoder": "greedy_argmax", "device": "cpu", "dtype": "float32", "inference_mode": "eval",
        "threads_per_worker": 2,
        "stop": "first full-state hit, actual task terminal, or 32 attempted actions including no-ops",
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage-one controller settings differ from the frozen protocol")
    if (contract.get("torch") != torch.__version__ or contract.get("numpy") != np.__version__
            or contract.get("memo", {}).get("enabled") is not True):
        raise ValueError("Stage-one library or memo configuration differs")
    for model in MODELS:
        expected_model = {"checkpoint": str(stage1.CHECKPOINTS[model]), "input": "current_and_complete_target_board"}
        if contract["models"].get(str(model)) != expected_model:
            raise ValueError("Stage-one checkpoint or input contract differs")
    # All historical source entries are verified, including checkpoint assets;
    # do this once per reference, never once per candidate.
    for path, value in contract["source_sha256"].items():
        if not Path(path).is_file() or stage1.file_sha(Path(path)) != value:
            raise ValueError(f"Stage-one source or checkpoint hash mismatch: {path}")
    rows = stage1.read_jsonl(pairs_path)
    groups = stage1.select_groups(rows, plan["split"], plan["layout_limit_per_family"],
                                  max_contexts=plan["contexts_per_layout"])
    groups_by_layout = {(g["family"], g["layout_id"]): g for g in groups}
    pairs_by_id = {r["pair_id"]: r for r in rows}
    if len(pairs_by_id) != len(rows):
        raise ValueError("Duplicate old pair ID")
    return {"directory": directory, "contract": contract, "contract_sha256": stage1.digest(contract),
            "plan": plan, "groups": groups_by_layout, "pairs": pairs_by_id, "shards": {}}


def existing_attempt(row: dict[str, Any], model: stage1.ModelId, reference: dict[str, Any]) -> dict[str, Any] | None:
    """Replay each referenced old shard and ensure all aliases agree exactly."""
    if row["split"] == "train":
        raise ValueError("New campaign training boards cannot reuse stage-one attempts")
    if row["candidate_origin"] != "demonstration":
        return None
    old_rows = []
    for pair_id in row["demo_pair_ids"]:
        pair = reference["pairs"].get(pair_id)
        if pair is None:
            raise ValueError(f"Unknown demonstration pair ID: {pair_id}")
        for key in ("family", "source_family", "split", "layout_id", "context_id", "source_hash", "target_hash"):
            if pair[key] != row[key]:
                raise ValueError(f"Demonstration alias differs in {key}")
        old_rows.append(pair)
    if sorted({r["k"] for r in old_rows}) != sorted(set(row["demo_offsets"])):
        raise ValueError("Demonstration offsets disagree with saved pair IDs")
    group = reference["groups"].get((row["family"], row["layout_id"]))
    if group is None:
        return None
    path = stage1.shard_path(reference["directory"], group, model)
    if not path.is_file():
        return None
    cache_key = (str(path), str(model))
    if cache_key not in reference["shards"]:
        stage1.validate_existing_shard(path, group, model, reference["contract_sha256"], use_memo=True)
        shard = json.loads(path.read_text())
        reference["shards"][cache_key] = ({t["pair_id"]: t for t in shard["trials"]}, stage1.file_sha(path))
    trials, shard_sha = reference["shards"][cache_key]
    if any(pair["pair_id"] not in trials for pair in old_rows):
        return None  # The old evaluation did not include this context.
    aliases = [trials[pair["pair_id"]] for pair in old_rows]
    fields = ("exact_hit", "first_hit", "attempted_actions", "actions", "state_hashes", "no_op_count",
              "task_solved", "stop_reason", "final_state", "final_player", "action_cap")
    if any(any(alias[field] != aliases[0][field] for field in fields) for alias in aliases[1:]):
        raise ValueError("Equivalent old source/target aliases have inconsistent traces")
    return {**{key: aliases[0][key] for key in fields},
            "reuse_ref": {"path": str(path.resolve()), "sha256": shard_sha,
                          "contract_path": str((reference["directory"] / "contract.json").resolve()),
                          "contract_sha256": reference["contract_sha256"],
                          "pair_ids": [pair["pair_id"] for pair in old_rows]}}


def attempt_path(output: Path, model: stage1.ModelId, candidate_id: str) -> Path:
    return output / "attempts" / str(model) / f"{candidate_id}.json"


def attempt_record(row: dict[str, Any], model: stage1.ModelId, trace: dict[str, Any],
                   contract_sha: str) -> dict[str, Any]:
    keys = ("candidate_id", "context_id", "layout_id", "family", "source_family", "split", "board_rank",
            "context_rank", "source_hash", "target_hash")
    fields = ("exact_hit", "first_hit", "attempted_actions", "actions", "state_hashes", "no_op_count",
              "task_solved", "stop_reason", "final_state", "final_player", "action_cap")
    reference = trace.get("reuse_ref")
    measurement = ("stage1_reuse" if reference is not None else
                   "new_train_demonstration" if row["split"] == "train" and row["candidate_origin"] == "demonstration"
                   else "new_target")
    return {"schema_version": 1, "status": "complete", **{key: row[key] for key in keys},
            "controller": model, **{key: trace[key] for key in fields}, "reuse_ref": reference,
            "measurement_origin": measurement,
            "contract_sha256": contract_sha}


def validate_attempt(row: dict[str, Any], model: stage1.ModelId, record: dict[str, Any],
                     contract_sha: str) -> None:
    """Replay a stage-two attempt and validate identity before resume or merge."""
    expected = {"schema_version": 1, "status": "complete", "controller": model,
                "contract_sha256": contract_sha}
    expected.update({key: row[key] for key in ("candidate_id", "context_id", "layout_id", "family", "source_family",
                                                 "split", "board_rank", "context_rank", "source_hash", "target_hash")})
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved attempt identity or contract mismatch")
    reference = record.get("reuse_ref")
    requires_reuse = row["candidate_origin"] == "demonstration" and row["split"] != "train"
    if requires_reuse != isinstance(reference, dict):
        raise ValueError("Demonstration reuse provenance is absent or unexpected")
    measurement = ("stage1_reuse" if reference is not None else
                   "new_train_demonstration" if row["split"] == "train" and row["candidate_origin"] == "demonstration"
                   else "new_target")
    if record.get("measurement_origin") != measurement:
        raise ValueError("Saved measurement origin differs from the split and reuse provenance")
    from scripts.assess_overnight_cpu_policy import replay_trial    # Local import avoids plotting in workers.

    replay_trial(row, record)
    if reference is not None:
        path = Path(reference["path"])
        if not path.is_file() or stage1.file_sha(path) != reference["sha256"]:
            raise ValueError("Original reference shard changed")


def run_context(rows: list[dict[str, Any]], model_value: str, output: str,
                contract_sha: str) -> dict[str, Any]:
    """Batch only new targets; save each finished trace atomically."""
    model_id = stage1.ModelId(model_value)
    checkpoint = stage1.CHECKPOINTS[model_id]
    model = stage1.WORKER["models"].get(model_id)
    if model is None:
        for asset in (checkpoint / "config.json", checkpoint / "pytorch_model.bin"):
            if stage1.file_sha(asset) != stage1.WORKER["expected_hashes"].get(str(asset)):
                raise ValueError("Checkpoint changed after the controller contract was frozen")
        model = BertForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, use_safetensors=False).to("cpu").float().eval()
        model.requires_grad_(False)
        stage1.WORKER["models"][model_id] = model
    pairs, by_id = [], {}
    for row in rows:
        source, target = validate_candidate(row)
        pair = Pair(row["candidate_id"], row["layout_id"], "", row["context_rank"], 0, None,
                    row["source_hash"], row["target_hash"], source, target)
        pairs.append(pair)
        by_id[pair.pair_id] = (pair, row)
    env = environment(pairs[0].source)
    predictor = MemoPredictor(model_predictor(model, env.tokenizer, torch.device("cpu"), 16),
                              conditioned=True, batch_size=16, max_entries=8192)

    def save(raw: dict[str, Any]) -> None:
        pair, row = by_id[raw["pair_id"]]
        group = {"family": row["family"], "layout_id": row["layout_id"], "split": row["split"],
                 "source_family": row["source_family"], "board_rank": row["board_rank"],
                 "records": [{"pair_id": row["candidate_id"], "context_id": row["context_id"],
                              "context_rank": row["context_rank"], "board_rank": row["board_rank"]}]}
        trial = stage1.decorate_trial(pair, raw, group, model_id, env)
        value = attempt_record(row, model_id, trial, contract_sha)
        stage1.atomic_json(attempt_path(Path(output), model_id, pair.pair_id), value)

    rollout_pairs(pairs, predictor, str(model_id), cap=32, on_complete=save, make_environment=lambda _state: env)
    return {"trials": len(pairs), "memo_stats": predictor.stats.to_dict()}


def merge(output: Path) -> dict[str, int]:
    """Atomically reconstruct the completed-only JSONL and report missing attempts."""
    contract = json.loads((output / "contract.json").read_text())
    plan = json.loads((output / "plan.json").read_text())
    if contract["plan_sha256"] != stage1.digest(plan):
        raise ValueError("Plan hash mismatch")
    contract_sha = stage1.digest(contract)
    candidates = {row["candidate_id"]: row for row in stage1.read_jsonl(Path(contract["candidates_path"]))}
    if stage1.file_sha(Path(contract["candidates_path"])) != contract["candidates_sha256"]:
        raise ValueError("Candidate file changed after plan was frozen")
    target = output / "rollouts.jsonl"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    complete = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for item in plan["planned_attempts"]:
                model = stage1.ModelId(item["controller"])
                row = candidates[item["candidate_id"]]
                path = attempt_path(output, model, row["candidate_id"])
                if not path.exists():
                    continue
                record = json.loads(path.read_text())
                validate_attempt(row, model, record, contract_sha)
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                complete += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"completed": complete, "planned": len(plan["planned_attempts"]),
            "missing": len(plan["planned_attempts"]) - complete}


def freeze_plan(output: Path, contract: dict[str, Any], plan: dict[str, Any], *, resume: bool) -> None:
    """Recover a missing plan only from an identical contract and plan digest."""
    if contract.get("plan_sha256") != stage1.digest(plan):
        raise ValueError("Computed plan does not match its contract")
    cp, pp = output / "contract.json", output / "plan.json"
    if cp.exists() or pp.exists():
        if not resume or not cp.is_file():
            raise ValueError("Existing output requires --resume and its frozen contract")
        if json.loads(cp.read_text()) != json.loads(stage1.json_bytes(contract)):
            raise ValueError("Stage-two saved contract changed")
        if pp.exists():
            if json.loads(pp.read_text()) != plan:
                raise ValueError("Stage-two saved plan changed")
        else:
            # The contract was durably replaced before a crash wrote the plan.
            # Exact contract equality and its plan digest bind every candidate.
            stage1.atomic_json(pp, plan)
    else:
        if resume:
            raise ValueError("No stage-two plan to resume")
        stage1.atomic_json(cp, contract)
        stage1.atomic_json(pp, plan)


def verify_train_exposure(rows: list[dict[str, Any]], exposure_path: Path,
                          stage1_reference: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless new training layouts are disjoint from known exposure."""
    manifest = json.loads(exposure_path.read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("role") != "campaign_train"
            or manifest.get("canonicalization") != "carl-layout-d4-v1"
            or manifest.get("old_checkpoint_exposure") != "unknown"):
        raise ValueError("Invalid training exposure manifest or undocumented old checkpoint exposure")

    def checked_path(field: str, sha_field: str) -> Path:
        value, expected = manifest.get(field), manifest.get(sha_field)
        if not isinstance(value, str) or not isinstance(expected, str):
            raise ValueError(f"Missing training provenance: {field}")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else stage1.ROOT / path
        if not path.is_file() or stage1.file_sha(path) != expected:
            raise ValueError(f"Training provenance hash mismatch: {field}")
        return path.resolve()

    old_data = Path(stage1_reference["contract"]["input_pairs_path"]).resolve().parent
    old_exposure = checked_path("stage01_exposure_path", "stage01_exposure_sha256")
    if old_exposure != (old_data / "exposure_register.json").resolve():
        raise ValueError("Training manifest points to a different stage-one exposure register")
    campaign_path = old_data.parent.parent / "campaign_manifest.json"
    if not campaign_path.is_file():
        raise ValueError("Original campaign manifest is unavailable")
    campaign = json.loads(campaign_path.read_text())
    frozen_data = campaign.get("stages", {}).get("01", {}).get("data_freeze")
    if not isinstance(frozen_data, dict) or frozen_data.get("exposure_sha256") != stage1.file_sha(old_exposure):
        raise ValueError("Training exposure disagrees with the frozen stage-one campaign manifest")
    old_trajectories = old_data / "trajectories.jsonl"
    if not old_trajectories.is_file() or stage1.file_sha(old_trajectories) != manifest.get("stage01_trajectories_sha256"):
        raise ValueError("Stage-one trajectory inventory hash mismatch")
    history_path = checked_path("fresh_history_path", "fresh_history_sha256")
    for prefix in ("source", "indexed_source", "alias_source"):
        checked_path(f"{prefix}_path", f"{prefix}_sha256")
    previous = json.loads(old_exposure.read_text())
    history = json.loads(history_path.read_text())
    if (previous.get("canonicalization") != "carl-layout-d4-v1"
            or history.get("canonicalization") != "carl-layout-d4-v1"):
        raise ValueError("Training exposure uses an incompatible layout canonicalization")
    mappings = (("historical_observed_layout_ids", history, "observed_layout_ids"),
                ("historical_protected_layout_ids", history, "previously_protected_layout_ids"),
                ("stage01_development_layout_ids", previous, "development_layout_ids"),
                ("stage01_evaluation_layout_ids", previous, "campaign_evaluation_layout_ids"),
                ("stage01_training_prohibited_layout_ids", previous, "training_prohibited_layout_ids"))
    protected: set[str] = set()
    for manifest_key, source, source_key in mappings:
        actual, expected = manifest.get(manifest_key), source.get(source_key)
        if not isinstance(actual, list) or not isinstance(expected, list) or set(actual) != set(expected):
            raise ValueError(f"Training exposure inventory mismatch: {manifest_key}")
        protected.update(actual)
    old_layouts = {row["layout_id"] for row in stage1.read_jsonl(old_trajectories)}
    protected.update(old_layouts)
    train_ids = manifest.get("train_layout_ids")
    if (not isinstance(train_ids, list) or not train_ids or len(set(train_ids)) != len(train_ids)
            or any(not isinstance(value, str) or len(value) != 64 for value in train_ids)):
        raise ValueError("Training exposure must list distinct canonical layout IDs")
    if set(train_ids) & protected:
        raise ValueError("Training layouts overlap protected or previously exposed layouts")
    if not rows or any(row["split"] != "train" or row["layout_id"] not in train_ids for row in rows):
        raise ValueError("Training candidate file contains an unregistered layout or another split")
    hashes = {manifest[key] for key in ("source_sha256", "indexed_source_sha256", "alias_source_sha256")}
    if any(row.get("source_sha256") not in hashes for row in rows):
        raise ValueError("Training candidate source hash is absent from its exposure manifest")
    return {"path": str(exposure_path.resolve()), "sha256": stage1.file_sha(exposure_path),
            "campaign_manifest_path": str(campaign_path.resolve()),
            "stage01_campaign_data_freeze_sha256": stage1.digest(frozen_data),
            "stage01_exposure_sha256": manifest["stage01_exposure_sha256"],
            "stage01_trajectories_sha256": manifest["stage01_trajectories_sha256"],
            "fresh_history_sha256": manifest["fresh_history_sha256"],
            "source_sha256": manifest["source_sha256"], "indexed_source_sha256": manifest["indexed_source_sha256"],
            "alias_source_sha256": manifest["alias_source_sha256"], "old_checkpoint_exposure": "unknown",
            "historical_split": manifest.get("historical_split")}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.workers * 2 > len(os.sched_getaffinity(0)):
        raise ValueError("workers × 2 threads exceed available CPU affinity")
    source_rows = stage1.read_jsonl(args.candidates)
    rows = selected_candidates(source_rows, args.split, args.max_boards)
    refs = {"development": load_reference(args.development_reference.resolve()),
            "evaluation": load_reference(args.evaluation_reference.resolve())}
    dev, evaluation = refs["development"]["contract"], refs["evaluation"]["contract"]
    for key in ("models", "source_sha256", "action_cap", "action_mask", "batch_size", "blas_threads", "decoder",
                "device", "dtype", "inference_mode", "memo", "threads_per_worker", "stop", "tie_break", "torch"):
        if dev[key] != evaluation[key]:
            raise ValueError(f"Development and evaluation controller contracts disagree: {key}")
    if args.split == "train":
        if args.train_exposure is None:
            raise ValueError("Training requires a separate verified exposure manifest")
        exposure = verify_train_exposure(source_rows, args.train_exposure.resolve(), refs["evaluation"])
    else:
        if args.train_exposure is not None:
            raise ValueError("A training exposure manifest is only valid for the train split")
        exposure = None
    for row in rows:
        validate_candidate(row)
    plan = {"schema_version": 1, "split": args.split, "max_boards_per_family": args.max_boards,
            "selection": "all demonstrations, selected BFS targets, and all unique valid generator targets "
                         "including aliases of BFS targets outside the fixed pool",
            "controllers": [str(model) for model in MODELS], "training_exposure": exposure,
            "planned_attempts": [{**{key: row[key] for key in ("candidate_id", "context_id", "layout_id", "family",
                                                               "split", "source_hash", "target_hash")},
                                  "candidate_origin": row["candidate_origin"],
                                  "measurement_origin": ("stage1_reuse" if row["split"] != "train"
                                                         and row["candidate_origin"] == "demonstration"
                                                         else "new_train_demonstration" if row["split"] == "train"
                                                         and row["candidate_origin"] == "demonstration" else "new_target"),
                                  "controller": str(model)} for row in rows for model in MODELS]}
    contract = {"schema_version": 1, "candidates_path": str(args.candidates.resolve()),
                "candidates_sha256": stage1.file_sha(args.candidates), "plan_sha256": stage1.digest(plan),
                "stage1_development_contract_sha256": refs["development"]["contract_sha256"],
                "stage1_evaluation_contract_sha256": refs["evaluation"]["contract_sha256"],
                "training_exposure": exposure,
                "controller_configuration": {key: evaluation[key] for key in (
                    "models", "source_sha256", "action_cap", "action_mask", "batch_size", "blas_threads", "decoder",
                    "device", "dtype", "inference_mode", "memo", "threads_per_worker", "stop", "tie_break", "torch")},
                "stage2_runner_sha256": stage1.file_sha(Path(__file__)), "workers": args.workers,
                "reuse": "verified complete stage-one full-state traces, no controller rerun"}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".run.lock"), timeout=0):
        freeze_plan(output, contract, plan, resume=args.resume)
        contract_sha = stage1.digest(contract)
        grouped: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        imported = reused = 0
        missing_demo = []
        for row in rows:
            for model in MODELS:
                path = attempt_path(output, model, row["candidate_id"])
                if path.exists():
                    validate_attempt(row, model, json.loads(path.read_text()), contract_sha)
                    reused += 1
                elif row["candidate_origin"] == "demonstration" and row["split"] != "train":
                    trace = existing_attempt(row, model, refs[row["split"]])
                    if trace is None:
                        missing_demo.append({"candidate_id": row["candidate_id"], "controller": str(model)})
                    else:
                        stage1.atomic_json(path, attempt_record(row, model, trace, contract_sha))
                        imported += 1
                else:
                    grouped[(row["board_rank"], row["family"], row["context_id"], str(model))].append(row)
        started = time.monotonic()
        baseline = stage1.resource_snapshot()
        errors = []
        state: dict[str, Any] = {"state": "running", "planned": len(plan["planned_attempts"]),
                                 "imported": imported, "previously_saved": reused, "new": 0,
                                 "missing_demonstrations": missing_demo, "errors": errors,
                                 "resources_initial": baseline}
        stage1.atomic_json(output / "status.json", state)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "2"
        os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        family_order = {family: i for i, family in enumerate(stage1.FAMILY_BOXES)}
        tasks = sorted(grouped, key=lambda key: (key[0], family_order[key[1]], key[2], key[3]))
        pending = iter(tasks)
        stopped, exhausted = None, False
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                                    initializer=stage1.worker_init,
                                                    initargs=(2, evaluation["source_sha256"])) as pool:
            active: dict[concurrent.futures.Future, tuple[int, str, str, str]] = {}
            last_log = 0.0
            while active or not exhausted:
                snap = stage1.resource_snapshot()
                if snap["host_used_fraction"] >= .80:
                    stopped = "host_memory_limit"
                elif snap["swap_used_bytes"] - baseline["swap_used_bytes"] >= 128 * 1024 * 1024:
                    stopped = "swap_growth_limit"
                elif args.stop_epoch is not None and time.time() >= args.stop_epoch:
                    stopped = "stage_epoch_cutoff"
                elif args.stop_monotonic is not None and time.monotonic() >= args.stop_monotonic:
                    stopped = "stage_monotonic_cutoff"
                if stopped:
                    exhausted = True
                while not exhausted and len(active) < args.workers:
                    try:
                        key = next(pending)
                    except StopIteration:
                        exhausted = True
                        break
                    active[pool.submit(run_context, grouped[key], key[3], str(output), contract_sha)] = key
                if not active:
                    break
                done, _ = concurrent.futures.wait(active, timeout=1,
                                                   return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    key = active.pop(future)
                    try:
                        outcome = future.result()
                        state["new"] += outcome["trials"]
                    except Exception as error:
                        errors.append({"context_id": key[2], "controller": key[3],
                                       "error": f"{type(error).__name__}: {error}"})
                if done or time.monotonic() - last_log >= 60:
                    state.update(elapsed_seconds=time.monotonic() - started, stop_reason=stopped, resources=snap)
                    stage1.atomic_json(output / "status.json", state)
                    if time.monotonic() - last_log >= 60:
                        print(json.dumps({key: state[key] for key in ("planned", "imported", "new")}), flush=True)
                        last_log = time.monotonic()
        counts = merge(output)
        state.update(counts, state="complete" if not counts["missing"] and not errors else "partial",
                     elapsed_seconds=time.monotonic() - started, stop_reason=stopped)
        stage1.atomic_json(output / "status.json", state)
        return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--candidates", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--evaluation-reference", type=Path, required=True)
    run.add_argument("--development-reference", type=Path, required=True)
    run.add_argument("--split", choices=("train", "development", "evaluation"), required=True)
    run.add_argument("--train-exposure", type=Path, help="verified separate campaign training exposure manifest")
    run.add_argument("--max-boards", type=int)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--stop-epoch", type=float)
    run.add_argument("--stop-monotonic", type=float)
    run.add_argument("--resume", action="store_true")
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "merge":
        with FileLock(str(args.output / ".run.lock"), timeout=0):
            print(json.dumps(merge(args.output)))
        return
    result = execute(args)
    print(json.dumps({key: result[key] for key in ("state", "planned", "completed", "missing")}), flush=True)
    if result["state"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
