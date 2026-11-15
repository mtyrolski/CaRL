"""Analyse fixed-controller temporal profiles without imputing missing attempts.

Models receive complete target boards. All labels are retrospective diagnostics,
not evidence of intent-conditioned planning or of a deployable generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
import numpy as np

from carl.icl.calibration import weighted_percentiles

CONTROLLERS = ("cllp8", "cllp32")
LIMITS = (8, 16, 32)
DEMO_SLOTS = ("demo_01", "demo_02", "demo_03", "demo_04")
DEMO_OFFSETS = (1, 4, 16, 32, 8, 2)
BFS_SLOTS = ("bfs_d01_04_p0", "bfs_d01_04_p1plus", "bfs_d05_08_p0", "bfs_d05_08_p1plus",
             "bfs_d09_max_p0", "bfs_d09_max_p1plus")


class Pool(StrEnum):
    FIXED = "fixed"
    REDUCED = "reduced"
    DEMO_ONLY = "demo_only"
    BFS_ONLY = "bfs_only"
    GENERATOR_BASE = "generator_base"
    GENERATOR_AUGMENTED = "generator_augmented"


class ProfileStatus(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNDERFILLED = "underfilled"
    UNAVAILABLE = "unavailable"
    NO_CANDIDATES = "no_candidates"


POOLS = tuple(Pool)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonblank JSON object records."""
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def sha_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def midpoint(values: Sequence[float], masses: Sequence[float]) -> list[float]:
    """Success-weighted empirical CDF midpoints; ties share one midpoint."""
    if len(values) != len(masses) or not values:
        raise ValueError("A nonempty equally sized sample is required")
    if any(not math.isfinite(float(x)) for x in values) or any(
        not math.isfinite(float(w)) or w <= 0 for w in masses
    ):
        raise ValueError("Values must be finite and masses positive")
    _weights, ranks = weighted_percentiles(list(values), list(masses))
    return [float(value) for value in ranks]


def prefix(attempt: dict[str, Any] | None, limit: int) -> dict[str, Any]:
    """Derive a shorter budget only from a completed fixed-policy rollout."""
    if limit not in LIMITS:
        raise ValueError("Unsupported action limit")
    if attempt is None:
        return {"status": "missing", "p": None, "d": None, "stop_reason": None}
    if attempt.get("status") != "complete":
        return {"status": "infrastructure_failure", "p": None, "d": None, "stop_reason": None}
    hit, first, count = attempt.get("exact_hit"), attempt.get("first_hit"), attempt.get("attempted_actions")
    actions, hashes = attempt.get("actions"), attempt.get("state_hashes")
    reason = attempt.get("stop_reason")
    if (not isinstance(hit, bool) or not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 32
            or not isinstance(actions, list) or len(actions) != count or not isinstance(hashes, list)
            or len(hashes) != count + 1 or reason not in {"exact_target", "task_terminal", "action_limit"}
            or (hit and (not isinstance(first, int) or isinstance(first, bool) or first != count
                         or reason != "exact_target"))
            or (not hit and (first is not None or reason == "exact_target"))
            or (reason == "action_limit" and count != 32)):
        raise ValueError("Invalid completed rollout or contradictory first hit")
    reached = bool(hit and first <= limit)
    prefix_stop = ("exact_target" if reached else "task_terminal" if reason == "task_terminal" and count <= limit
                   else "action_limit")
    return {"status": "complete", "p": int(reached), "d": first if reached else None,
            "stop_reason": prefix_stop}


def validate_candidates(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Enforce one context and one full target identity for each candidate."""
    required = {"candidate_id", "context_id", "layout_id", "family", "source_family", "split", "board_rank",
                "context_rank", "source_hash", "target_hash", "candidate_origin", "demo_offsets", "demo_pair_ids",
                "in_fixed_pool", "pool_variant", "slot"}
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids: set[str] = set()
    for row in rows:
        if not required <= row.keys():
            raise ValueError("Candidate identity or pool fields are missing")
        if row["candidate_id"] in ids or not isinstance(row["candidate_id"], str) or not row["candidate_id"]:
            raise ValueError("Duplicate or invalid candidate ID")
        ids.add(row["candidate_id"])
        if row["source_hash"] == row["target_hash"] or row["candidate_origin"] not in {"demonstration", "bfs", "generator"}:
            raise ValueError("Invalid source/target or candidate origin")
        offsets, pair_ids = row["demo_offsets"], row["demo_pair_ids"]
        if (not isinstance(offsets, list) or not isinstance(pair_ids, list) or len(offsets) != len(pair_ids)
                or len(set(offsets)) != len(offsets) or any(not isinstance(k, int) or isinstance(k, bool) or k <= 0
                                                          for k in offsets)):
            raise ValueError("Invalid demonstration aliases")
        if (row["candidate_origin"] == "demonstration") != bool(offsets):
            raise ValueError("Only demonstration candidates have demonstration offsets")
        if not isinstance(row["in_fixed_pool"], bool):
            raise ValueError("Pool membership must be boolean")
        if not isinstance(row.get("in_generator_pool", False), bool):
            raise ValueError("Generator pool membership must be boolean")
        if not isinstance(row.get("also_generated", False), bool):
            raise ValueError("Generated-alias flag must be boolean")
        contexts[row["context_id"]].append(row)
    for context, items in contexts.items():
        for key in ("layout_id", "family", "source_family", "split", "source_hash", "board_rank", "context_rank"):
            if len({item[key] for item in items}) != 1:
                raise ValueError(f"Context {context} has inconsistent {key}")
        if len({item["target_hash"] for item in items}) != len(items):
            raise ValueError("Targets must be unique within each context")
        offsets = [k for item in items for k in item["demo_offsets"]]
        if len(offsets) != len(set(offsets)):
            raise ValueError("A demonstration offset cannot name multiple targets")
        slots = [item["slot"] for item in items if item["in_fixed_pool"]]
        if len(set(slots)) != len(slots) or any(slot not in DEMO_SLOTS + BFS_SLOTS for slot in slots):
            raise ValueError("Selected candidates need distinct declared pool slots")
        variants = {item["pool_variant"] for item in items if item["pool_variant"] is not None}
        if len(variants) > 1 or variants - {"demo4_bfs6", "demo4_bfs4", "underfilled"}:
            raise ValueError("Inconsistent fixed-pool variant")
    return dict(contexts)


def validate_rollouts(
    candidates: list[dict[str, Any]], plan: dict[str, Any], attempts: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Ensure every saved attempt is an individually planned model/target pair."""
    by_id = {row["candidate_id"]: row for row in candidates}
    scheduled = set()
    for row in plan.get("planned_attempts", []):
        key = (row["controller"], row["candidate_id"])
        candidate = by_id.get(row["candidate_id"])
        if key in scheduled or row["controller"] not in CONTROLLERS or candidate is None:
            raise ValueError("Duplicate, unknown or wrong-controller planned attempt")
        for field in ("context_id", "layout_id", "family", "split", "source_hash", "target_hash"):
            if row.get(field) != candidate[field]:
                raise ValueError(f"Planned attempt has mismatched {field}")
        scheduled.add(key)
    expected = {(model, row["candidate_id"]) for row in candidates for model in CONTROLLERS
                if row["candidate_origin"] in {"demonstration", "generator"} or row["in_fixed_pool"]
                or row.get("also_generated", False)}
    if scheduled != expected:
        raise ValueError("Plan must contain all demonstrations, generated aliases and selected BFS pairs")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in attempts:
        key = (row.get("controller"), row.get("candidate_id"))
        if key not in scheduled or key in result:
            raise ValueError("Unplanned or duplicate completed attempt")
        candidate = by_id[row["candidate_id"]]
        for field in ("context_id", "layout_id", "family", "split", "source_hash", "target_hash"):
            if row.get(field) != candidate[field]:
                raise ValueError(f"Completed attempt has mismatched {field}")
        if row.get("status") != "complete":
            raise ValueError("Consolidated file must contain only completed attempts")
        prefix(row, 32)
        if row["state_hashes"][0] != candidate["source_hash"]:
            raise ValueError("Rollout does not start from planned state")
        if row["exact_hit"] and row["state_hashes"][-1] != candidate["target_hash"]:
            raise ValueError("Hit does not end at the planned exact target")
        result[key] = row
    return result


def select_pools(rows: list[dict[str, Any]]) -> tuple[str, dict[str, list[dict[str, Any]]]]:
    """Derive declared nested subsets from slot order, never from outcomes."""
    variants = {row["pool_variant"] for row in rows if row["pool_variant"] is not None}
    variant = next(iter(variants), "underfilled")
    fixed = [row for row in rows if row["in_fixed_pool"]]
    demo = sorted((row for row in fixed if row["candidate_origin"] == "demonstration"),
                  key=lambda row: DEMO_SLOTS.index(row["slot"]))
    bfs = sorted((row for row in fixed if row["candidate_origin"] == "bfs"),
                 key=lambda row: BFS_SLOTS.index(row["slot"]))
    expected_bfs = {"demo4_bfs6": 6, "demo4_bfs4": 4}.get(variant)
    if expected_bfs is not None and (len(demo) != 4 or len(bfs) != expected_bfs):
        raise ValueError("Fixed pool counts disagree with its declared variant")
    reduced_bfs = 3 if variant == "demo4_bfs6" else 2
    pools = {"fixed": demo + bfs, "reduced": demo[:2] + bfs[:reduced_bfs],
             "demo_only": demo, "bfs_only": bfs}
    selected = [row for row in rows if row.get("in_generator_pool", False)]
    if selected:
        base = [row for row in selected if row["candidate_origin"] != "generator"]
        pools["generator_base"] = base
        pools["generator_augmented"] = selected
    return variant, pools


def profile(
    rows: list[dict[str, Any]], attempts: dict[tuple[str, str], dict[str, Any]], controller: str, limit: int,
    *, pool: str, variant: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the CDF from a complete reference pool, retaining all failures."""
    outcomes = {row["candidate_id"]: prefix(attempts.get((controller, row["candidate_id"])), limit) for row in rows}
    missing = sum(item["status"] != "complete" for item in outcomes.values())
    reached = [row for row in rows if outcomes[row["candidate_id"]]["p"] == 1]
    if pool.startswith("generator_"):
        origins = [row["candidate_origin"] for row in rows]
        expected = {"demonstration": 4, "bfs": 4, "generator": int(pool == "generator_augmented")}
        pool_filled = all(origins.count(origin) == number for origin, number in expected.items())
    else:
        pool_filled = variant != "underfilled"
    status = (ProfileStatus.UNDERFILLED if not pool_filled else
              ProfileStatus.INCOMPLETE if missing else ProfileStatus.COMPLETE)
    durations = [outcomes[row["candidate_id"]]["d"] for row in reached]
    centre = median(durations) if durations and status == "complete" else None
    labels = {}
    if durations and status == "complete":
        # M=1 deterministic: every successful fraction is exactly 1.
        for row, u in zip(reached, midpoint(durations, [1.0] * len(durations))):
            ident = row["candidate_id"]
            labels[ident] = {"p": 1.0, "d": outcomes[ident]["d"], "weight": 1 / len(reached),
                             "u": u, "d_over_median": outcomes[ident]["d"] / centre if centre else None}
    distinct = len(set(durations)) if status == "complete" else None
    return ({"controller": controller, "L": limit, "pool": pool, "pool_variant": variant, "status": status,
             "planned_targets": len(rows), "completed_targets": len(rows) - missing, "missing_targets": missing,
             "successful_targets": len(reached) if status == "complete" else None, "median_d": centre,
             "mean_success_d": sum(durations) / len(durations) if durations and status == "complete" else None,
             "distinct_successful_d": distinct, "fewer_than_two_distinct_d": distinct < 2 if distinct is not None else None,
             "all_fail": len(reached) == 0 if status == "complete" else None,
             "singleton_success": len(reached) == 1 if status == "complete" else None,
             "all_tied_multiple": len(reached) > 1 and distinct == 1 if status == "complete" else None}, labels)


def rank_counts(ks: Sequence[int], durations: Sequence[int]) -> dict[str, Any]:
    """Count strict inversions and each disjoint tie category explicitly."""
    if len(ks) != len(durations):
        raise ValueError("Rank vectors differ in length")
    result = {"pairwise": 0, "strict_comparable": 0, "concordant": 0, "inversions": 0,
              "k_tie_only": 0, "d_tie_only": 0, "both_tied": 0}
    for a, b in itertools.combinations(range(len(ks)), 2):
        dk, dd = ks[a] - ks[b], durations[a] - durations[b]
        result["pairwise"] += 1
        if dk == 0 and dd == 0:
            result["both_tied"] += 1
        elif dk == 0:
            result["k_tie_only"] += 1
        elif dd == 0:
            result["d_tie_only"] += 1
        else:
            result["strict_comparable"] += 1
            result["inversions" if dk * dd < 0 else "concordant"] += 1
    result["inversion_rate"] = (result["inversions"] / result["strict_comparable"]
                                if result["strict_comparable"] else None)
    return result


def cluster_mean(
    values: Sequence[tuple[str, float]], *, seed: int = 20261115, replicates: int = 2000,
) -> dict[str, Any]:
    """Equal-context mean with whole-layout resampling; each value is one context."""
    boards: dict[str, list[float]] = defaultdict(list)
    for layout, value in values:
        if not math.isfinite(float(value)):
            raise ValueError("A cluster metric must be finite")
        boards[layout].append(float(value))
    if not boards:
        return {"n_contexts": 0, "n_layouts": 0, "mean": None, "ci_low": None, "ci_high": None}
    sums = np.asarray([sum(boards[key]) for key in sorted(boards)], dtype=float)
    counts = np.asarray([len(boards[key]) for key in sorted(boards)], dtype=float)
    mean = float(sums.sum() / counts.sum())
    if len(boards) < 2:
        low = high = None
    else:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(boards), size=(replicates, len(boards)))
        estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        low, high = (float(value) for value in np.quantile(estimates, [.025, .975]))
    return {"n_contexts": int(counts.sum()), "n_layouts": len(boards), "mean": mean,
            "ci_low": low, "ci_high": high}



def context_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Fields that identify one original board and restored context."""
    return {key: row.get(key) for key in (
        "context_id", "layout_id", "family", "source_family", "split", "board_rank", "context_rank",
    )}


def common_demo_ranks(
    rows: list[dict[str, Any]], attempts: dict[tuple[str, str], dict[str, Any]], limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank k and duration on exactly the targets reached by both controllers.

    When one target has several demonstration offsets, the earliest offset is
    its canonical k; all aliases remain in the individual target output.
    """
    demo = [row for row in rows if row["candidate_origin"] == "demonstration"]
    missing = sum(attempts.get((model, row["candidate_id"])) is None
                  for model in CONTROLLERS for row in demo)
    common = [row for row in demo if all(
        prefix(attempts.get((model, row["candidate_id"])), limit)["p"] == 1 for model in CONTROLLERS
    )] if not missing else []
    ks = [min(row["demo_offsets"]) for row in common]
    k_u = midpoint(ks, [1] * len(ks)) if ks else []
    contexts, targets = [], []
    for controller in CONTROLLERS:
        durations = [prefix(attempts[(controller, row["candidate_id"])], limit)["d"] for row in common]
        d_u = midpoint(durations, [1] * len(durations)) if durations else []
        errors = [abs(a - b) for a, b in zip(k_u, d_u)]
        counts = rank_counts(ks, durations)
        contexts.append({"controller": controller, "L": limit,
                         "status": "incomplete" if missing else "complete",
                         "demo_targets": len(demo), "missing_attempts": missing,
                         "common_success_targets": len(common) if not missing else None,
                         "distinct_d": len(set(durations)) if not missing else None,
                         "mean_abs_u_difference": sum(errors) / len(errors) if errors else None,
                         "identical_u_fraction": sum(error < 1e-12 for error in errors) / len(errors)
                         if errors else None, **counts})
        for row, k, ku, duration, du in zip(common, ks, k_u, durations, d_u):
            targets.append({"controller": controller, "L": limit, "candidate_id": row["candidate_id"],
                            "k_earliest_alias": k, "all_demo_offsets": row["demo_offsets"],
                            "demo_pair_ids": row["demo_pair_ids"], "d": duration,
                            "u_duration": du, "u_demo_offset": ku, "abs_u_difference": abs(du - ku)})
    return contexts, targets


def analyse(
    context_rows: list[dict[str, Any]], candidates: list[dict[str, Any]],
    attempts: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped = validate_candidates(candidates)
    context_ids = [row["context_id"] for row in context_rows]
    if len(set(context_ids)) != len(context_ids) or set(grouped) - set(context_ids):
        raise ValueError("Missing or duplicate contexts")
    results: dict[str, list[dict[str, Any]]] = {key: [] for key in (
        "profiles", "profile_labels", "demo_offsets", "common_ranks", "common_targets", "pool_sensitivity",
        "slot_selection", "offset_availability", "candidate_outcomes", "bfs_rank_contexts", "bfs_rank_targets",
    )}
    for context in context_rows:
        meta = context_metadata(context)
        rows = grouped.get(context["context_id"], [])
        for row in rows:
            for key in ("layout_id", "family", "source_family", "split", "board_rank", "context_rank"):
                if row[key] != context[key]:
                    raise ValueError(f"Candidate/context mismatch: {key}")
        variant, pools = select_pools(rows)
        availability = {item["offset"]: item for item in context.get("demo_offset_availability", [])}
        if len(availability) != len(context.get("demo_offset_availability", [])):
            raise ValueError("Repeated demonstration availability offset")
        if context.get("status") == "complete" and set(availability) != set(DEMO_OFFSETS):
            raise ValueError("A prepared context must account for every configured offset")
        for offset in DEMO_OFFSETS:
            entry = availability.get(offset, {"status": "context_not_prepared", "candidate_id": None})
            if entry["status"] not in {"selected", "outside_first_four", "duplicate_target", "not_available",
                                       "context_not_prepared"}:
                raise ValueError("Unknown demonstration availability status")
            available = entry["status"] in {"selected", "outside_first_four", "duplicate_target"}
            if available != (entry["candidate_id"] is not None):
                raise ValueError("Offset status disagrees with candidate availability")
            if entry["candidate_id"] is not None:
                matching = [row for row in rows if row["candidate_id"] == entry["candidate_id"]]
                if len(matching) != 1 or offset not in matching[0]["demo_offsets"]:
                    raise ValueError("Demonstration availability does not match candidate aliases")
            results["offset_availability"].append({**meta, "k": offset,
                "status": entry["status"], "candidate_id": entry["candidate_id"]})
        selected_offsets = {}
        selected_candidates = {}
        for slot in context.get("slots", []):
            if slot.get("slot") not in DEMO_SLOTS + BFS_SLOTS:
                raise ValueError("Unknown slot in frozen context metadata")
            selected_offsets[slot["slot"]] = slot.get("selected_offset")
            selected_candidates[slot["slot"]] = slot.get("candidate_id")
            results["slot_selection"].append({**meta, **slot})
        for candidate in pools["fixed"]:
            if selected_candidates.get(candidate["slot"]) != candidate["candidate_id"]:
                raise ValueError("Selected context slot and candidate disagree")
            if candidate["candidate_origin"] == "demonstration":
                selected = selected_offsets.get(candidate["slot"])
                if selected not in candidate["demo_offsets"]:
                    raise ValueError("Selected demonstration offset absent from candidate aliases")
        for limit in LIMITS:
            rank_rows, target_rows = common_demo_ranks(rows, attempts, limit)
            for result in rank_rows:
                if not rows:
                    result.update(status="no_candidates", common_success_targets=None, distinct_d=None)
                results["common_ranks"].append({**meta, **result})
            results["common_targets"].extend({**meta, **target} for target in target_rows)
            for row in rows:
                if (row["candidate_origin"] in {"demonstration", "generator"} or row["in_fixed_pool"]
                        or row.get("also_generated", False)):
                    for controller in CONTROLLERS:
                        saved = attempts.get((controller, row["candidate_id"]))
                        outcome = prefix(saved, limit)
                        action_count = min(limit, saved["attempted_actions"]) if saved else None
                        no_ops = (sum(a == b for a, b in zip(saved["state_hashes"][:action_count],
                                                             saved["state_hashes"][1:action_count + 1]))
                                  if saved else None)
                        results["candidate_outcomes"].append({**meta, "controller": controller, "L": limit,
                            "candidate_id": row["candidate_id"], "candidate_origin": row["candidate_origin"],
                            "pool_variant": variant, "in_fixed_pool": row["in_fixed_pool"],
                            "in_generator_pool": row.get("in_generator_pool", False),
                            "also_generated": row.get("also_generated", False),
                            "generator_ranks": row.get("generator_ranks", []),
                            "demo_offsets": row["demo_offsets"], "source_depth": row.get("bfs_depth"),
                            "bfs_pushes": row.get("bfs_pushes"), "generator_rank": row.get("generator_rank"),
                            "attempted_actions": action_count, "no_op_count": no_ops, **outcome})
                if row["candidate_origin"] != "demonstration":
                    continue
                for k, pair_id in zip(row["demo_offsets"], row["demo_pair_ids"]):
                    for controller in CONTROLLERS:
                        outcome = prefix(attempts.get((controller, row["candidate_id"])), limit)
                        results["demo_offsets"].append({**meta, "candidate_id": row["candidate_id"],
                            "pair_id": pair_id, "k": k, "controller": controller, "L": limit, **outcome})
            for controller in CONTROLLERS:
                profile_by_pool: dict[str, dict[str, Any]] = {}
                labels_by_pool: dict[str, dict[str, dict[str, Any]]] = {}
                for pool in POOLS:
                    selection = pools.get(pool, [])
                    info, labels = profile(selection, attempts, controller, limit, pool=pool, variant=variant)
                    if pool.startswith("generator_") and pool not in pools:
                        info["status"] = ProfileStatus.UNAVAILABLE
                    if not rows:
                        info["status"] = ProfileStatus.NO_CANDIDATES
                    profile_by_pool[pool], labels_by_pool[pool] = info, labels
                    results["profiles"].append({**meta, **info})
                    for candidate in selection:
                        cid = candidate["candidate_id"]
                        outcome = prefix(attempts.get((controller, cid)), limit)
                        label = labels.get(cid, {})
                        results["profile_labels"].append({**meta, "controller": controller, "L": limit,
                            "pool": pool, "pool_status": info["status"], "candidate_id": cid,
                            "candidate_origin": candidate["candidate_origin"], "slot": candidate["slot"],
                            "selected_demo_offset": selected_offsets.get(candidate["slot"]),
                            "demo_offsets": candidate["demo_offsets"], "bfs_depth": candidate.get("bfs_depth"),
                            "bfs_pushes": candidate.get("bfs_pushes"), **outcome,
                            "weight": label.get("weight"), "u": label.get("u"),
                            "d_over_median": label.get("d_over_median")})
                for other in ("reduced", "demo_only", "bfs_only"):
                    comparisons = ("fixed", other)
                    info_a, info_b = (profile_by_pool[p] for p in comparisons)
                    labels_a, labels_b = (labels_by_pool[p] for p in comparisons)
                    shared = sorted(set(labels_a) & set(labels_b))
                    status = "complete" if info_a["status"] == info_b["status"] == "complete" else "unavailable"
                    diffs = [abs(labels_a[cid]["u"] - labels_b[cid]["u"]) for cid in shared] if status == "complete" else []
                    results["pool_sensitivity"].append({**meta, "controller": controller, "L": limit,
                        "pool_variant": variant,
                        "reference": comparisons[0], "comparison": comparisons[1], "status": status,
                        "shared_success_targets": len(diffs) if status == "complete" else None,
                        "mean_abs_delta_u": sum(diffs) / len(diffs) if diffs else None,
                        "max_abs_delta_u": max(diffs) if diffs else None})
                info_a, info_b = profile_by_pool["generator_base"], profile_by_pool["generator_augmented"]
                labels_a, labels_b = labels_by_pool["generator_base"], labels_by_pool["generator_augmented"]
                status = "complete" if info_a["status"] == info_b["status"] == "complete" else "unavailable"
                shared = sorted(set(labels_a) & set(labels_b))
                diffs = [abs(labels_a[cid]["u"] - labels_b[cid]["u"]) for cid in shared] if status == "complete" else []
                results["pool_sensitivity"].append({**meta, "controller": controller, "L": limit,
                    "pool_variant": variant,
                    "reference": "generator_base", "comparison": "generator_augmented", "status": status,
                    "shared_success_targets": len(diffs) if status == "complete" else None,
                    "mean_abs_delta_u": sum(diffs) / len(diffs) if diffs else None,
                    "max_abs_delta_u": max(diffs) if diffs else None})
    profile_keys: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for label in results["profile_labels"]:
        if label["pool"] == Pool.FIXED and label["candidate_origin"] == "bfs":
            profile_keys[(label["context_id"], label["controller"], label["L"])].append(label)
    for info in results["profiles"]:
        if info["pool"] != Pool.FIXED:
            continue
        key = (info["context_id"], info["controller"], info["L"])
        selected = profile_keys.get(key, [])
        successes = [row for row in selected if row["p"] == 1] if info["status"] == "complete" else []
        depths = [row["bfs_depth"] for row in successes]
        durations = [row["d"] for row in successes]
        depth_u = midpoint(depths, [1] * len(depths)) if depths else []
        duration_u = midpoint(durations, [1] * len(durations)) if durations else []
        differences = [abs(a-b) for a, b in zip(depth_u, duration_u)]
        counts = rank_counts(depths, durations)
        counts["source_depth_tie_only"] = counts.pop("k_tie_only")
        metadata = {**context_metadata(info), "controller": info["controller"], "L": info["L"],
                    "pool_variant": info["pool_variant"]}
        results["bfs_rank_contexts"].append({**metadata, "status": info["status"],
            "bfs_targets": len(selected), "successful_bfs_targets": len(successes)
            if info["status"] == "complete" else None, "mean_abs_rank_difference":
            sum(differences) / len(differences) if differences else None, **counts})
        for row, du, ru in zip(successes, depth_u, duration_u):
            results["bfs_rank_targets"].append({**metadata, "candidate_id": row["candidate_id"],
                "source_depth": row["bfs_depth"], "bfs_pushes": row["bfs_pushes"], "d": row["d"],
                "u_fixed_pool": row["u"], "u_depth_on_bfs_successes": du,
                "u_duration_on_bfs_successes": ru, "abs_rank_difference": abs(du-ru)})
    fixed_labels = {(row["context_id"], row["controller"], row["L"], row["candidate_id"]): row
                    for row in results["profile_labels"] if row["pool"] == Pool.FIXED}
    for row in results["candidate_outcomes"]:
        key = (row["context_id"], row["controller"], row["L"], row["candidate_id"])
        label = fixed_labels.get(key, {})
        row["u_fixed_pool"] = label.get("u")
        row["d_over_median"] = label.get("d_over_median")
    return results


def groups(rows: list[dict[str, Any]], fields: tuple[str, ...]):
    """Give both overall-family and source-distribution views without pooling families."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in fields)
        grouped[("family", row["family"], "all", *key)].append(row)
        grouped[("source", row["family"], row["source_family"], *key)].append(row)
    for (scope, family, source, *keys), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        yield {"scope": scope, "family": family, "source_family": source,
               **dict(zip(fields, keys))}, group


def with_interval(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return cluster_mean([(row["layout_id"], row[field]) for row in rows if row[field] is not None])


def summarise(result: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    summaries: dict[str, list[dict[str, Any]]] = {key: [] for key in (
        "demo_reachability", "demo_paired", "rank_summary", "rank_paired", "pool_summary", "sensitivity_summary",
        "slot_summary", "offset_coverage", "candidate_summary", "bfs_depth_summary", "bfs_rank_summary",
    )}
    for meta, rows in groups(result["demo_offsets"], ("controller", "L", "k")):
        completed = [row for row in rows if row["status"] == "complete"]
        stats = with_interval(completed, "p")
        missing = len(rows) - len(completed)
        hits = sum(row["p"] for row in completed)
        summaries["demo_reachability"].append({**meta, "planned_contexts": len(rows),
            "completed_contexts": len(completed), "missing_contexts": missing, "reached_contexts": hits,
            "observed_rate": stats["mean"], "rate": stats["mean"] if not missing else None,
            "ci_low": stats["ci_low"] if not missing else None, "ci_high": stats["ci_high"] if not missing else None,
            "lower_bound": hits / len(rows), "upper_bound": (hits + missing) / len(rows),
            "n_layouts": stats["n_layouts"]})
    for meta, rows in groups(result["demo_offsets"], ("L", "k")):
        by_key = {(row["context_id"], row["controller"]): row for row in rows}
        if len(by_key) != len(rows):
            raise ValueError("Duplicate demonstration offset in a context")
        contexts = sorted({row["context_id"] for row in rows})
        paired = []
        for context in contexts:
            a, b = (by_key.get((context, model)) for model in CONTROLLERS)
            if a and b and a["status"] == b["status"] == "complete":
                paired.append({"layout_id": a["layout_id"], "difference": a["p"] - b["p"]})
        stats = with_interval(paired, "difference")
        missing = len(contexts) - len(paired)
        summaries["demo_paired"].append({**meta, "difference": "cllp8_minus_cllp32",
            "planned_contexts": len(contexts), "paired_contexts": len(paired), "missing_pairs": missing,
            "observed_difference": stats["mean"], "paired_difference": stats["mean"] if not missing else None,
            "ci_low": stats["ci_low"] if not missing else None, "ci_high": stats["ci_high"] if not missing else None,
            "n_layouts": stats["n_layouts"]})
    for meta, rows in groups(result["common_ranks"], ("controller", "L")):
        complete = [row for row in rows if row["status"] == "complete"]
        error = with_interval(complete, "mean_abs_u_difference")
        inversion = with_interval(complete, "inversion_rate")
        summaries["rank_summary"].append({**meta, "planned_contexts": len(rows),
            "complete_contexts": len(complete), "incomplete_contexts": len(rows) - len(complete),
            "contexts_with_common_success": sum(row["common_success_targets"] > 0 for row in complete),
            "contexts_with_two_distinct_d": sum(row["distinct_d"] >= 2 for row in complete),
            "common_targets": sum(row["common_success_targets"] for row in complete),
            "mean_abs_u_difference": error["mean"], "error_ci_low": error["ci_low"], "error_ci_high": error["ci_high"],
            "error_contexts": error["n_contexts"], "error_layouts": error["n_layouts"],
            "mean_inversion_rate": inversion["mean"], "inversion_ci_low": inversion["ci_low"],
            "inversion_ci_high": inversion["ci_high"], "inversion_contexts": inversion["n_contexts"],
            **{field: sum(row[field] for row in complete) for field in (
                "pairwise", "strict_comparable", "concordant", "inversions", "k_tie_only", "d_tie_only", "both_tied")}})
    for meta, rows in groups(result["common_ranks"], ("L",)):
        by_key = {(row["context_id"], row["controller"]): row for row in rows}
        paired = []
        for context in sorted({row["context_id"] for row in rows}):
            a, b = (by_key.get((context, model)) for model in CONTROLLERS)
            if a and b and a["mean_abs_u_difference"] is not None and b["mean_abs_u_difference"] is not None:
                paired.append({"layout_id": a["layout_id"],
                               "difference": a["mean_abs_u_difference"] - b["mean_abs_u_difference"]})
        stats = with_interval(paired, "difference")
        summaries["rank_paired"].append({**meta, "difference": "cllp8_minus_cllp32",
            "planned_contexts": len(rows) // len(CONTROLLERS), "paired_defined_contexts": len(paired),
            "n_layouts": stats["n_layouts"], "mean_difference": stats["mean"],
            "ci_low": stats["ci_low"], "ci_high": stats["ci_high"]})
    for meta, rows in groups(result["profiles"], ("controller", "L", "pool", "pool_variant")):
        complete = [row for row in rows if row["status"] == "complete"]
        informative = [{**row, "value": int(not row["fewer_than_two_distinct_d"])} for row in complete]
        stats = with_interval(informative, "value")
        duration = with_interval(complete, "mean_success_d")
        summaries["pool_summary"].append({**meta, "planned_contexts": len(rows),
            "complete_contexts": len(complete), "incomplete_contexts": sum(row["status"] == "incomplete" for row in rows),
            "underfilled_contexts": sum(row["status"] == "underfilled" for row in rows),
            "unavailable_contexts": sum(row["status"] in {"unavailable", "no_candidates"} for row in rows),
            "all_fail_contexts": sum(row["all_fail"] is True for row in complete),
            "singleton_success_contexts": sum(row["singleton_success"] is True for row in complete),
            "multiple_all_tied_contexts": sum(row["all_tied_multiple"] is True for row in complete),
            "informative_contexts": sum(row["distinct_successful_d"] >= 2 for row in complete),
            "informative_fraction": stats["mean"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
            "n_layouts": stats["n_layouts"], "mean_success_d_equal_context": duration["mean"],
            "duration_contexts": duration["n_contexts"]})
    for meta, rows in groups(result["pool_sensitivity"], ("controller", "L", "reference", "comparison", "pool_variant")):
        complete = [row for row in rows if row["status"] == "complete"]
        stats = with_interval(complete, "mean_abs_delta_u")
        summaries["sensitivity_summary"].append({**meta, "planned_contexts": len(rows),
            "complete_pool_contexts": len(complete), "contexts_with_shared_success": stats["n_contexts"],
            "shared_success_targets": sum(row["shared_success_targets"] for row in complete),
            "n_layouts": stats["n_layouts"], "mean_abs_delta_u": stats["mean"],
            "ci_low": stats["ci_low"], "ci_high": stats["ci_high"]})
    for meta, rows in groups(result["slot_selection"], ("slot", "status", "selected_offset")):
        summaries["slot_summary"].append({**meta, "contexts": len(rows),
                                         "layouts": len({row["layout_id"] for row in rows})})
    for meta, rows in groups(result["offset_availability"], ("k",)):
        summaries["offset_coverage"].append({**meta, "frozen_contexts": len(rows),
            "available_offsets": sum(row["status"] in {"selected", "outside_first_four", "duplicate_target"}
                                     for row in rows),
            "unavailable_offsets": sum(row["status"] == "not_available" for row in rows),
            "unprepared_contexts": sum(row["status"] == "context_not_prepared" for row in rows)})
    for meta, rows in groups(result["candidate_outcomes"], ("controller", "L", "candidate_origin", "pool_variant")):
        complete = [row for row in rows if row["status"] == "complete"]
        context_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in complete:
            context_data[row["context_id"]].append(row)
        per_context = [{"layout_id": items[0]["layout_id"], "success": sum(row["p"] for row in items)/len(items),
                        "actions": sum(row["attempted_actions"] for row in items)/len(items),
                        "no_ops": sum(row["no_op_count"] for row in items)/len(items)}
                       for items in context_data.values()]
        success = with_interval(per_context, "success")
        actions = with_interval(per_context, "actions")
        no_ops = with_interval(per_context, "no_ops")
        hits = sum(row["p"] for row in complete)
        summaries["candidate_summary"].append({**meta, "planned_attempts": len(rows),
            "completed_attempts": len(complete), "missing_attempts": len(rows)-len(complete), "successes": hits,
            "success_lower_bound": hits/len(rows), "success_upper_bound": (hits+len(rows)-len(complete))/len(rows),
            "observed_mean_context_success": success["mean"], "success_ci_low": success["ci_low"],
            "success_ci_high": success["ci_high"], "observed_contexts": success["n_contexts"],
            "observed_layouts": success["n_layouts"], "mean_actions_per_attempt_equal_context": actions["mean"],
            "mean_no_ops_per_attempt_equal_context": no_ops["mean"]})
    bfs = [row for row in result["candidate_outcomes"] if row["candidate_origin"] == "bfs"]
    for meta, rows in groups(bfs, ("controller", "L", "pool_variant", "source_depth")):
        complete = [row for row in rows if row["status"] == "complete"]
        context_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in complete:
            context_data[row["context_id"]].append(row)
        averaged = []
        for items in context_data.values():
            successes = [item for item in items if item["p"] == 1]
            durations = [item["d"] for item in successes]
            scales = [item["u_fixed_pool"] for item in successes if item["u_fixed_pool"] is not None]
            averaged.append({"layout_id": items[0]["layout_id"],
                "success": sum(item["p"] for item in items)/len(items),
                "duration": sum(durations)/len(durations) if durations else None,
                "scale": sum(scales)/len(scales) if scales else None})
        success, duration, scale = (with_interval(averaged, field) for field in ("success", "duration", "scale"))
        summaries["bfs_depth_summary"].append({**meta, "planned_attempts": len(rows),
            "completed_attempts": len(complete), "missing_attempts": len(rows)-len(complete),
            "successes": sum(row["p"] for row in complete), "mean_context_success": success["mean"],
            "success_ci_low": success["ci_low"], "success_ci_high": success["ci_high"],
            "success_contexts": success["n_contexts"], "mean_context_success_d": duration["mean"],
            "duration_contexts": duration["n_contexts"], "mean_context_u_fixed": scale["mean"],
            "scale_contexts": scale["n_contexts"]})
    for meta, rows in groups(result["bfs_rank_contexts"], ("controller", "L", "pool_variant")):
        complete = [row for row in rows if row["status"] == "complete"]
        error = with_interval(complete, "mean_abs_rank_difference")
        inversion = with_interval(complete, "inversion_rate")
        summaries["bfs_rank_summary"].append({**meta, "planned_contexts": len(rows),
            "complete_contexts": len(complete), "contexts_with_bfs_success": error["n_contexts"],
            "success_layouts": error["n_layouts"], "mean_abs_rank_difference": error["mean"],
            "ci_low": error["ci_low"], "ci_high": error["ci_high"],
            "mean_context_inversion_rate": inversion["mean"], "inversion_contexts": inversion["n_contexts"],
            **{field: sum(row[field] for row in complete) for field in (
                "pairwise", "strict_comparable", "concordant", "inversions", "source_depth_tie_only",
                "d_tie_only", "both_tied")}})
    return summaries


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    from io import StringIO

    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = StringIO(newline="")
    if keys:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})
    atomic_text(path, stream.getvalue())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows))


def figures(summary: dict[str, list[dict[str, Any]]], output: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    colours = {"cllp8": "#21618c", "cllp32": "#bd5626"}
    families = ("12x12x4", "12x12x10")

    def save(fig, name):
        for ext in ("pdf", "svg", "png"):
            target = output / f"{name}.{ext}"
            temp = output / f".{name}.{ext}.{os.getpid()}.tmp"
            try:
                fig.savefig(temp, format=ext, dpi=180, bbox_inches="tight")
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(8.1, 4.7), sharex=True, sharey=True, layout="constrained")
    offsets = (1, 2, 4, 8, 16, 32)
    for r, family in enumerate(families):
        for c, limit in enumerate(LIMITS):
            ax = axes[r, c]
            drawn = False
            for model in CONTROLLERS:
                subset = [item for item in summary["demo_reachability"] if item["scope"] == "family"
                          and item["family"] == family and item["controller"] == model and item["L"] == limit
                          and item["rate"] is not None]
                for item in sorted(subset, key=lambda row: row["k"]):
                    if item["k"] not in offsets:
                        continue
                    drawn = True
                    x = offsets.index(item["k"]) + (-0.07 if model == "cllp8" else 0.07)
                    y = item["rate"]
                    low = item["ci_low"] if item["ci_low"] is not None else y
                    high = item["ci_high"] if item["ci_high"] is not None else y
                    ax.errorbar(x, y, yerr=[[max(0, y-low)], [max(0, high-y)]], marker="o", markersize=3,
                                color=colours[model], capsize=2, linewidth=0.8, fmt="none")
                    ax.plot(x, y, "o", markersize=3, color=colours[model])
            if not drawn:
                ax.text(0.5, 0.5, "No complete attempts", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks(range(len(offsets)), offsets)
            ax.set_xlim(-0.5, len(offsets)-0.5)
            ax.set_ylim(-0.04, 1.04)
            ax.grid(axis="y", alpha=0.2)
            ax.set_title(f"{family.replace('x', ' × ')}  ·  L = {limit}")
            if c == 0:
                ax.set_ylabel("Exact target reached")
            if r == 1:
                ax.set_xlabel("Demonstration offset k (actions)")
    handles = [plt.Line2D([], [], marker="o", color=colours[model], linestyle="none", label=model.upper())
               for model in CONTROLLERS]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    save(fig, "demo_reachability")

    fig, axes = plt.subplots(2, 3, figsize=(8.3, 4.8), sharex=True, layout="constrained")
    for r, family in enumerate(families):
        for c in range(3):
            ax = axes[r, c]
            drawn = False
            data = summary["pool_summary"] if c < 2 else summary["rank_summary"]
            variant = ("demo4_bfs6", "demo4_bfs4")[c] if c < 2 else None
            for model in CONTROLLERS:
                for limit in LIMITS:
                    matches = [item for item in data if item["scope"] == "family" and item["family"] == family
                               and item["controller"] == model and item["L"] == limit
                               and (c == 2 or (item["pool"] == "fixed" and item["pool_variant"] == variant))]
                    if not matches:
                        continue
                    item = matches[0]
                    value = item["informative_fraction"] if c < 2 else item["mean_abs_u_difference"]
                    if value is None:
                        continue
                    drawn = True
                    low = item["ci_low"] if c < 2 else item["error_ci_low"]
                    high = item["ci_high"] if c < 2 else item["error_ci_high"]
                    low, high = (value if low is None else low), (value if high is None else high)
                    x = LIMITS.index(limit) + (-0.06 if model == "cllp8" else 0.06)
                    ax.errorbar(x, value, yerr=[[max(0, value-low)], [max(0, high-value)]], color=colours[model],
                                marker="o", markersize=4, linewidth=0.8, capsize=2, fmt="none")
                    ax.plot(x, value, "o", color=colours[model], markersize=4)
            if not drawn:
                label = "No complete pools" if c < 2 else "No comparable ranks"
                ax.text(0.5, 0.5, label, ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks(range(len(LIMITS)), LIMITS)
            ax.set_xlim(-0.5, len(LIMITS)-0.5)
            ax.set_ylim(-0.04, 1.04)
            ax.grid(axis="y", alpha=0.2)
            ax.set_title(f"{family.replace('x', ' × ')} · " +
                         (("4 demo + 6 BFS", "4 demo + 4 BFS")[c] if c < 2 else "Offset-rank difference"))
            ax.set_ylabel("Fraction with distinct d" if c < 2 else "Mean |u(d) − u(k)|")
            if r == 1:
                ax.set_xlabel("Primitive-action limit L")
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    save(fig, "scale_diagnostics")


def write_report(output: Path, meta: dict[str, Any], summary: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Stage 2: retrospective temporal profiles", "",
        "Both frozen CLLP controllers receive the complete target board. This is a full-state controller "
        "analysis, not predicate-intent execution or a planning experiment.", "",
        f"Frozen planned attempts: {meta['planned_attempts']}; completed: {meta['completed_attempts']}; "
        f"missing: {meta['missing_attempts']}. Missing data are not failures.", "",
        "Each L=8/16/32 profile uses the prefix of the same greedy trajectory and is recalculated from its "
        "own successful targets. With one deterministic attempt per target, p is exactly 0 or 1. "
        "The successful durations receive equal masses, and u is the midpoint of the tied weighted CDF jump. "
        "No-success candidates have d and u undefined. Median is the conventional arithmetic median.", "",
        "Fixed pools use 4 demonstration + 6 BFS targets, falling back to 4 + 4 according to the frozen "
        "candidate manifest. Demo slots choose the first distinct available targets in priority order "
        "k=1,4,16,32,8,2; actual offsets and fallbacks are in slot_selection.csv and slot_summary.csv. "
        "The reduced pool uses the first 2 + 3 (or 2 + 2); single-origin pools are "
        "shown separately. If present, the 4 + 4 + generator pool is compared with precisely the same "
        "eight base targets. Incomplete and underfilled pools receive no scale labels.", "",
        "The 4 + 6 and 4 + 4 reference profiles are summarised separately. Every valid unique generated "
        "target is included in candidate_outcomes, even when it does not enter the scale pool; its "
        "prefix action cost and no-op count are reconstructed from the saved trace.", "",
        "All demonstration offsets are assessed, including aliases of an identical target. The k-versus-u "
        "diagnostic uses only demonstration targets reached by both controllers at that L; both ranks are "
        "recomputed on this identical intersection. The earliest k is used when a target has multiple offsets. "
        "Pairs tied in k or duration are excluded from the strict inversion denominator and reported separately.", "",
        "Aggregate statistics weight contexts equally, and 95% bootstrap intervals resample original "
        "D4-canonical layouts. Contexts with no successful target remain in pool coverage; metrics requiring "
        "a successful target condition on the stated subset. Source families are separately available in CSV. "
        "Previously inspected layouts and differing source distributions make these exploratory results.", "",
        "| Family | Fixed-pool variant | Controller | L | Complete fixed pools | At least 2 durations |",
        "|---|---|---|---:|---:|---:|"]
    for item in summary["pool_summary"]:
        if item["scope"] != "family" or item["pool"] != "fixed":
            continue
        lines.append(f"| {item['family']} | {item['pool_variant']} | {item['controller']} | {item['L']} | "
                     f"{item['complete_contexts']}/{item['planned_contexts']} | "
                     f"{item['informative_contexts']}/{item['complete_contexts']} |")
    lines += ["", "`demo_reachability.*` shows exact-target success by demonstration offset; unavailable "
              "offsets are absent, not failed attempts. `scale_diagnostics.*` shows the fraction of complete "
              "fixed pools with at least two distinct successful durations and the mean rank difference "
              "on the common-success subset. Markers are unsmoothed and intervals are layout bootstraps.", "",
              "`offset_coverage.csv` lists the frozen contexts and the target offsets available or missing "
              "for each family, source and k; missing offsets are excluded from the corresponding target "
              "success denominator and never treated as controller failures.", "",
              "For BFS candidates, `source_depth` is the search depth used to generate a target, whereas "
              "d is the controller's first-hit duration. `bfs_rank_targets.csv` and "
              "`bfs_rank_summary.csv` compare their mid-ranks on the same successful BFS subset and "
              "report strict inversions and ties; they do not reinterpret depth as demonstration offset.", "",
              "These retrospective descriptions do not demonstrate that requesting u changes a generator's "
              "output, that generated targets are useful, or that complete-task planning improves. "
              "They should not be interpreted as stochastic reliability estimates.", "",
              "No model training or fine-tuning was performed in this stage (training steps: 0). "
              "The historical training data and prior exposure of the frozen checkpoints to these boards "
              "are unknown; these results must not be presented as verified transfer to unseen boards.", "",
              "Detailed per-target outcomes, offsets, profiles, paired diagnostics and uncertainty are in "
              "the accompanying CSV/JSON files. The complete input and output hashes are in `analysis_manifest.json`. "]
    atomic_text(output / "report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Frozen stage-two rollout directory")
    parser.add_argument("--contexts", type=Path, required=True, help="Frozen stage-two contexts JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Export directory")
    args = parser.parse_args()
    plan = json.loads((args.run / "plan.json").read_text(encoding="utf-8"))
    contract = json.loads((args.run / "contract.json").read_text(encoding="utf-8"))
    if contract["plan_sha256"] != canonical_sha(plan):
        raise ValueError("Plan hash does not match frozen controller contract")
    candidate_path = Path(contract["candidates_path"])
    if sha_file(candidate_path) != contract["candidates_sha256"]:
        raise ValueError("Candidate file differs from frozen controller contract")
    contract_sha = canonical_sha(contract)
    attempts = read_jsonl(args.run / "rollouts.jsonl")
    if any(row.get("contract_sha256") != contract_sha for row in attempts):
        raise ValueError("A rollout has the wrong frozen controller contract")
    context_rows = read_jsonl(args.contexts)
    data_manifest_path = args.contexts.parent / "data_manifest.json"
    if data_manifest_path.exists():
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
        expected_context_sha = data_manifest.get("output_sha256", {}).get(args.contexts.name)
        if expected_context_sha is None or sha_file(args.contexts) != expected_context_sha:
            raise ValueError("Context file differs from its frozen preparation manifest")
    candidates = read_jsonl(candidate_path)
    maximum = plan.get("max_boards_per_family")
    def included(row):
        return row["split"] == plan["split"] and (maximum is None or row["board_rank"] < maximum)
    candidates = [row for row in candidates if included(row)]
    context_rows = [row for row in context_rows if included(row)]
    validate_candidates(candidates)
    attempt_map = validate_rollouts(candidates, plan, attempts)
    result = analyse(context_rows, candidates, attempt_map)
    summary = summarise(result)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in {**result, **summary}.items():
        write_csv(args.output / f"{name}.csv", rows)
    labels_by_profile: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result["profile_labels"]:
        labels_by_profile[(row["context_id"], row["controller"], row["L"], row["pool"])].append(row)
    profile_objects = [{**row, "candidates": labels_by_profile.get(
        (row["context_id"], row["controller"], row["L"], row["pool"]), [])} for row in result["profiles"]]
    write_jsonl(args.output / "profiles.jsonl", profile_objects)
    write_jsonl(args.output / "candidate_outcomes.jsonl", result["candidate_outcomes"])
    metadata = {"schema_version": 1, "candidates": str(candidate_path.resolve()),
        "candidates_sha256": sha_file(candidate_path), "contexts_path": str(args.contexts.resolve()),
        "contexts_sha256": sha_file(args.contexts), "run": str(args.run.resolve()),
        "plan_sha256": canonical_sha(plan), "contract_sha256": contract_sha,
        "rollouts_sha256": sha_file(args.run / "rollouts.jsonl"), "exporter_sha256": sha_file(Path(__file__)),
        "planned_attempts": len(plan["planned_attempts"]), "completed_attempts": len(attempt_map),
        "missing_attempts": len(plan["planned_attempts"]) - len(attempt_map),
        "contexts_count": len(context_rows), "bootstrap": {"unit": "canonical_layout", "context_weight": "equal",
            "replicates": 2000, "seed": 20261115, "confidence": 0.95},
        "limits": LIMITS, "median": "arithmetic midpoint for even sample sizes",
        "rank_alias": "earliest demonstration offset for a unique target", "command": list(os.sys.argv)}
    figures(summary, args.output)
    write_report(args.output, metadata, summary)
    atomic_text(args.output / "summaries.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    metadata["output_sha256"] = {path.name: sha_file(path) for path in args.output.iterdir()
        if path.is_file() and path.name != "analysis_manifest.json" and not path.name.startswith(".")}
    atomic_text(args.output / "analysis_manifest.json", json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
