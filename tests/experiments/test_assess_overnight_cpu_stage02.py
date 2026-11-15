"""Synthetic checks for stage-two scale profiles and missing-attempt accounting."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.assess_overnight_cpu_stage02 import CONTROLLERS
from scripts.assess_overnight_cpu_stage02 import analyse
from scripts.assess_overnight_cpu_stage02 import canonical_sha
from scripts.assess_overnight_cpu_stage02 import cluster_mean
from scripts.assess_overnight_cpu_stage02 import common_demo_ranks
from scripts.assess_overnight_cpu_stage02 import midpoint
from scripts.assess_overnight_cpu_stage02 import prefix
from scripts.assess_overnight_cpu_stage02 import profile
from scripts.assess_overnight_cpu_stage02 import rank_counts
from scripts.assess_overnight_cpu_stage02 import select_pools
from scripts.assess_overnight_cpu_stage02 import sha_file
from scripts.assess_overnight_cpu_stage02 import summarise
from scripts.assess_overnight_cpu_stage02 import validate_candidates
from scripts.assess_overnight_cpu_stage02 import validate_rollouts


def candidate(name, origin, *, slot=None, offset=None, fixed=True, generator_pool=False):
    return {"candidate_id": name, "context_id": "ctx1", "layout_id": "board1", "family": "12x12x4",
            "source_family": "original", "split": "evaluation", "board_rank": 0, "context_rank": 0,
            "source_hash": "start", "target_hash": name, "candidate_origin": origin,
            "demo_offsets": [offset] if offset else [], "demo_pair_ids": [f"pair-{offset}"] if offset else [],
            "bfs_depth": 1 if origin == "bfs" else None, "bfs_pushes": 0 if origin == "bfs" else None,
            "bfs_actions": [1] if origin == "bfs" else None, "in_fixed_pool": fixed,
            "in_generator_pool": generator_pool, "pool_variant": "demo4_bfs6" if fixed else None, "slot": slot}


def pool():
    from scripts.assess_overnight_cpu_stage02 import BFS_SLOTS
    from scripts.assess_overnight_cpu_stage02 import DEMO_SLOTS

    demos = [candidate(f"demo-{offset}", "demonstration", slot=slot, offset=offset, generator_pool=True)
             for slot, offset in zip(DEMO_SLOTS, (1, 4, 16, 32))]
    bfs = [candidate(f"bfs-{i}", "bfs", slot=slot, generator_pool=i < 4)
           for i, slot in enumerate(BFS_SLOTS)]
    return demos + bfs


def attempt(row, model, duration=None, *, terminal=False):
    count = duration if duration else 5 if terminal else 32
    hit = duration is not None
    return {"status": "complete", "controller": model, "candidate_id": row["candidate_id"],
            **{key: row[key] for key in ("context_id", "layout_id", "family", "split", "source_hash", "target_hash")},
            "first_hit": duration, "exact_hit": hit, "attempted_actions": count, "actions": [1] * count,
            "state_hashes": [row["source_hash"]] + ["intermediate"] * (count-1) +
                            [row["target_hash"] if hit else "last"],
            "stop_reason": "exact_target" if hit else "task_terminal" if terminal else "action_limit"}


def context(rows):
    demo = {offset: row for row in rows for offset in row["demo_offsets"]}
    return {**{key: rows[0][key] for key in ("context_id", "layout_id", "family", "source_family", "split",
                                              "board_rank", "context_rank")},
            "status": "complete",
            "demo_offset_availability": [
                {"offset": offset, "status": "selected" if offset in demo else "not_available",
                 "candidate_id": demo[offset]["candidate_id"] if offset in demo else None}
                for offset in (1, 4, 16, 32, 8, 2)],
            "slots": [{"slot": row["slot"], "candidate_id": row["candidate_id"], "status": "selected",
                       "selected_offset": row["demo_offsets"][0] if row["demo_offsets"] else None}
                      for row in rows if row["in_fixed_pool"]]}


def test_weighted_midpoint_and_ties():
    assert midpoint([2, 2, 4], [1, 3, 2]) == pytest.approx([1/3, 1/3, 5/6])
    with pytest.raises(ValueError):
        midpoint([2], [0])


def test_fixed_greedy_prefix_and_missing_are_distinct():
    row = candidate("target", "bfs", fixed=False)
    assert prefix(None, 8)["p"] is None
    assert prefix(attempt(row, "cllp8", 10), 8)["p"] == 0
    assert prefix(attempt(row, "cllp8", 10), 16)["d"] == 10
    assert prefix(attempt(row, "cllp8", terminal=True), 8)["stop_reason"] == "task_terminal"
    invalid = attempt(row, "cllp8", 10)
    invalid["first_hit"] = 9
    with pytest.raises(ValueError):
        prefix(invalid, 32)


def test_frozen_pool_full_fallback_and_generator():
    rows = pool()
    gen = candidate("gen-1", "generator", fixed=False, generator_pool=True)
    variant, pools = select_pools(rows + [gen])
    assert variant == "demo4_bfs6"
    assert [len(pools[k]) for k in ("fixed", "reduced", "demo_only", "bfs_only")] == [10, 5, 4, 6]
    assert len(pools["generator_base"]) == 8
    assert set(row["candidate_id"] for row in pools["generator_augmented"]) - {
        row["candidate_id"] for row in pools["generator_base"]} == {"gen-1"}
    for row in rows[:8]:
        row["pool_variant"] = "demo4_bfs4"
    variant, pools = select_pools(rows[:8])
    assert variant == "demo4_bfs4"
    assert len(pools["fixed"]) == 8
    assert len(pools["reduced"]) == 4


def test_profiles_singletons_ties_failures_and_incomplete():
    rows = pool()
    attempts = {("cllp8", row["candidate_id"]): attempt(row, "cllp8") for row in rows}
    info, labels = profile(rows, attempts, "cllp8", 32, pool="fixed", variant="demo4_bfs6")
    assert info["all_fail"] and not labels
    attempts[("cllp8", rows[0]["candidate_id"])] = attempt(rows[0], "cllp8", 4)
    info, labels = profile(rows, attempts, "cllp8", 32, pool="fixed", variant="demo4_bfs6")
    assert info["singleton_success"] and labels[rows[0]["candidate_id"]]["u"] == 0.5
    attempts[("cllp8", rows[1]["candidate_id"])] = attempt(rows[1], "cllp8", 4)
    info, labels = profile(rows, attempts, "cllp8", 32, pool="fixed", variant="demo4_bfs6")
    assert info["all_tied_multiple"] and info["fewer_than_two_distinct_d"]
    assert all(value["u"] == 0.5 for value in labels.values())
    attempts.pop(("cllp8", rows[2]["candidate_id"]))
    info, labels = profile(rows, attempts, "cllp8", 32, pool="fixed", variant="demo4_bfs6")
    assert info["status"] == "incomplete" and info["successful_targets"] is None and not labels


def test_strict_inversions_and_each_tie_category():
    counts = rank_counts([1, 2, 2, 3], [3, 2, 2, 2])
    assert counts == {"pairwise": 6, "strict_comparable": 3, "concordant": 0, "inversions": 3,
                      "k_tie_only": 0, "d_tie_only": 2, "both_tied": 1, "inversion_rate": 1.0}
    assert rank_counts([1, 1], [1, 2])["k_tie_only"] == 1


def test_common_rank_uses_exact_intersection_and_aliases():
    rows = pool()[:4]
    rows[0]["demo_offsets"] = [1, 2]
    rows[0]["demo_pair_ids"] = ["pair-1", "pair-2"]
    attempts = {}
    for model in CONTROLLERS:
        for i, row in enumerate(rows):
            duration = (i + 1) if (model == "cllp8" or i != 2) else None
            attempts[(model, row["candidate_id"])] = attempt(row, model, duration)
    rank, targets = common_demo_ranks(rows, attempts, 32)
    assert [row["common_success_targets"] for row in rank] == [3, 3]
    assert {row["candidate_id"] for row in targets} == {"demo-1", "demo-4", "demo-32"}
    assert next(row for row in targets if row["candidate_id"] == "demo-1")["all_demo_offsets"] == [1, 2]
    attempts.pop(("cllp32", rows[0]["candidate_id"]))
    rank, targets = common_demo_ranks(rows, attempts, 32)
    assert all(row["status"] == "incomplete" and row["common_success_targets"] is None for row in rank)
    assert not targets


def test_equal_context_mean_clustered_by_layout():
    stats = cluster_mean([("A", 1), ("A", 1), ("B", 0)], replicates=100)
    assert stats["mean"] == pytest.approx(2/3)
    assert stats["n_contexts"] == 3 and stats["n_layouts"] == 2
    assert cluster_mean([("A", 1)], replicates=10)["ci_low"] is None


def test_plan_validation_missing_is_not_failure():
    rows = pool()
    planned = [{**{key: row[key] for key in ("candidate_id", "context_id", "layout_id", "family", "split",
                                              "source_hash", "target_hash")}, "controller": model}
               for row in rows for model in CONTROLLERS]
    saved = [attempt(rows[0], "cllp8", 4)]
    indexed = validate_rollouts(rows, {"planned_attempts": planned}, saved)
    assert len(indexed) == 1
    assert prefix(indexed.get(("cllp32", rows[0]["candidate_id"])), 32)["p"] is None
    planned.pop()
    with pytest.raises(ValueError):
        validate_rollouts(rows, {"planned_attempts": planned}, saved)


def test_aliases_and_missing_bounds_in_aggregate():
    rows = pool()
    rows[0]["demo_offsets"] = [1, 2]
    rows[0]["demo_pair_ids"] = ["pair-1", "pair-2"]
    attempts = {(model, row["candidate_id"]): attempt(row, model, 1)
                for row in rows for model in CONTROLLERS if not (model == "cllp32" and row is rows[0])}
    result = analyse([context(rows)], rows, attempts)
    aliases = [item for item in result["demo_offsets"] if item["candidate_id"] == rows[0]["candidate_id"]
               and item["controller"] == "cllp8" and item["L"] == 32]
    assert {item["k"] for item in aliases} == {1, 2}
    summary = summarise(result)
    row = next(item for item in summary["demo_reachability"] if item["scope"] == "family" and
               item["k"] == 1 and item["L"] == 32 and item["controller"] == "cllp32")
    assert row["rate"] is None and row["lower_bound"] == 0 and row["upper_bound"] == 1


def test_duplicate_target_or_offset_is_rejected():
    rows = pool()
    rows[1]["target_hash"] = rows[0]["target_hash"]
    with pytest.raises(ValueError):
        validate_candidates(rows)
    rows = pool()
    rows[1]["demo_offsets"] = [1]
    with pytest.raises(ValueError):
        validate_candidates(rows)


@pytest.mark.parametrize("split", ("evaluation", "train"))
def test_complete_synthetic_export_and_figures(tmp_path, split):
    rows = pool()
    for row in rows:
        row["split"] = split
    contexts = [context(rows)]
    candidates = tmp_path / "candidates.jsonl"
    context_path = tmp_path / "contexts.jsonl"
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    context_path.write_text("".join(json.dumps(row) + "\n" for row in contexts), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    plan = {"split": split, "max_boards_per_family": 1,
            "planned_attempts": [{**{key: row[key] for key in
                ("candidate_id", "context_id", "layout_id", "family", "split", "source_hash", "target_hash")},
                "controller": model} for row in rows for model in CONTROLLERS]}
    contract = {"plan_sha256": canonical_sha(plan), "candidates_path": str(candidates),
                "candidates_sha256": sha_file(candidates)}
    (run / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (run / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    attempts = [{**attempt(row, model, i + 1), "contract_sha256": canonical_sha(contract)}
                for i, row in enumerate(rows) for model in CONTROLLERS]
    (run / "rollouts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8")
    output = tmp_path / "analysis"
    command = [sys.executable, "-m", "scripts.assess_overnight_cpu_stage02", "--run", str(run),
               "--contexts", str(context_path), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    assert manifest["planned_attempts"] == manifest["completed_attempts"] == 20
    assert manifest["missing_attempts"] == 0
    for name in ("demo_reachability", "scale_diagnostics"):
        for suffix in ("pdf", "svg", "png"):
            assert (output / f"{name}.{suffix}").stat().st_size > 500
    assert (output / "slot_summary.csv").stat().st_size > 50
    assert "summaries.json" in manifest["output_sha256"]
    profiles = [json.loads(line) for line in (output / "profiles.jsonl").read_text().splitlines()]
    fixed = next(row for row in profiles if row["pool"] == "fixed" and row["L"] == 8)
    assert len(fixed["candidates"]) == 10
    assert any(row["p"] == 0 and row["d"] is None and row["u"] is None for row in fixed["candidates"])
    assert len((output / "candidate_outcomes.jsonl").read_text().splitlines()) == 60


def test_reference_variants_are_never_pooled_and_generator_outside_pool_is_included():
    rows_a = pool()
    rows_b = pool()[:8]
    for row in rows_b:
        row["context_id"] = "ctx2"
        row["layout_id"] = "board2"
        row["candidate_id"] = "other-" + row["candidate_id"]
        row["target_hash"] = row["candidate_id"]
        row["pool_variant"] = "demo4_bfs4"
    gen = candidate("gen-outside", "generator", fixed=False, generator_pool=False)
    rows = rows_a + rows_b + [gen]
    attempts = {(model, row["candidate_id"]): attempt(row, model, 3)
                for row in rows for model in CONTROLLERS if row is not gen}
    result = analyse([context(rows_a), context(rows_b)], rows, attempts)
    summary = summarise(result)
    fixed = [row for row in summary["pool_summary"] if row["scope"] == "family" and row["L"] == 32 and
             row["pool"] == "fixed" and row["controller"] == "cllp8"]
    assert {(row["pool_variant"], row["planned_contexts"]) for row in fixed} == {
        ("demo4_bfs6", 1), ("demo4_bfs4", 1)}
    missing_gen = [row for row in result["candidate_outcomes"] if row["candidate_id"] == "gen-outside"]
    assert len(missing_gen) == 6
    assert all(row["status"] == "missing" and row["p"] is None for row in missing_gen)
    generated = next(row for row in summary["candidate_summary"] if row["scope"] == "family" and
                     row["candidate_origin"] == "generator" and row["L"] == 32 and row["controller"] == "cllp8")
    assert generated["planned_attempts"] == generated["missing_attempts"] == 1


def test_prefix_cost_noops_and_bfs_depth_rank_separate_from_k():
    rows = pool()
    for i, row in enumerate(rows[4:]):
        row["bfs_depth"] = i + 1
    attempts = {(model, row["candidate_id"]): attempt(row, model, 10)
                for row in rows for model in CONTROLLERS}
    result = analyse([context(rows)], rows, attempts)
    entry = next(row for row in result["candidate_outcomes"] if row["candidate_id"] == "bfs-0"
                 and row["controller"] == "cllp8" and row["L"] == 8)
    assert entry["p"] == 0 and entry["attempted_actions"] == 8 and entry["no_op_count"] == 7
    rank = next(row for row in result["bfs_rank_contexts"] if row["controller"] == "cllp8" and row["L"] == 32)
    assert rank["d_tie_only"] == 15 and rank["source_depth_tie_only"] == 0
    assert "k_tie_only" not in rank
    target = next(row for row in result["bfs_rank_targets"] if row["candidate_id"] == "bfs-0"
                  and row["controller"] == "cllp8" and row["L"] == 32)
    assert target["source_depth"] == 1 and target["d"] == 10
    assert "u_fixed_pool" in target and "u_depth_on_bfs_successes" in target


def test_generator_alias_to_bfs_is_measured_without_changing_fixed_pool():
    rows = pool()
    duplicate = candidate("generator-alias-bfs", "bfs", fixed=False)
    duplicate.update(also_generated=True, generator_ranks=[1])
    rows.append(duplicate)
    plan = {"planned_attempts": [{**{key: row[key] for key in (
        "candidate_id", "context_id", "layout_id", "family", "split", "source_hash", "target_hash")},
        "controller": model} for row in rows for model in CONTROLLERS]}
    saved = [attempt(row, model, 4) for row in rows for model in CONTROLLERS]
    indexed = validate_rollouts(rows, plan, saved)
    result = analyse([context(rows)], rows, indexed)
    alias_rows = [row for row in result["candidate_outcomes"] if row["candidate_id"] == duplicate["candidate_id"]]
    assert len(alias_rows) == 6
    assert all(row["candidate_origin"] == "bfs" and row["also_generated"] and row["generator_ranks"] == [1]
               and not row["in_fixed_pool"] for row in alias_rows)
    assert all(row["candidate_id"] != duplicate["candidate_id"] for row in result["profile_labels"])
