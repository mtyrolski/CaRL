"""Replay and assess the paired, frozen-policy CPU experiment.

The planned pair file defines every denominator. A missing or invalid shard is
an infrastructure omission, never a failed policy attempt. All uncertainty is
computed over original canonical layouts after averaging that layout's cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import from_rows
from carl.icl.boards import state_hash
from carl.icl.boards import tiles
from carl.icl.boards import to_rows
from carl.icl.boards import validate_subgoal
from carl.icl.data_sources.sokoban import environment
from carl.icl.evaluation.data import EvaluationConfig
from carl.icl.evaluation.statistics import interval

OFFSETS = (1, 2, 4, 8, 16, 32)
BUDGETS = (1, 2, 4, 8, 16, 32)
ACTION_CAP = 32


class StopReason(StrEnum):
    """Observable stopping rules of a completed rollout."""

    EXACT_TARGET = "exact_target"
    TASK_TERMINAL = "task_terminal"
    ACTION_LIMIT = "action_limit"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects, rejecting ambiguous rows."""
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def canonical_sha256(value: object) -> str:
    """Hash the runner's canonical JSON contract or frozen plan."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash source data and controller files without copying weights to memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_plan(path: Path, split: str, max_contexts: int, max_boards: int) -> list[dict[str, Any]]:
    """Freeze the original requested cases independently of completed shards."""
    all_rows = read_jsonl(path)
    split_by_layout: dict[str, str] = {}
    for row in all_rows:
        layout, source_split = row.get("layout_id"), row.get("split")
        if not isinstance(layout, str) or not layout or not isinstance(source_split, str) or not source_split:
            raise ValueError("Every source pair needs a layout ID and split")
        if split_by_layout.setdefault(layout, source_split) != source_split:
            raise ValueError("A canonical layout appears in more than one split")
    candidate_rows = [row for row in all_rows if row.get("split") == split]
    for row in candidate_rows:
        for key in ("context_rank", "board_rank"):
            if not isinstance(row.get(key), int) or isinstance(row[key], bool) or row[key] < 0:
                raise ValueError(f"Each planned pair needs a nonnegative {key}")
    rows = [row for row in candidate_rows if row["context_rank"] < max_contexts and row["board_rank"] < max_boards]
    if not rows:
        raise ValueError(f"No planned pairs for split {split!r}")
    required = {"pair_id", "layout_id", "family", "source_family", "split", "source", "target",
                "source_hash", "target_hash", "k"}
    ids: set[str] = set()
    layouts: dict[tuple[str, str], str] = {}
    layout_ranks: dict[tuple[str, str], int] = {}
    rank_layouts: dict[tuple[str, int], str] = {}
    contexts: dict[tuple[str, str, int], str] = {}
    context_offsets = set()
    for row in rows:
        if not required <= row.keys() or not isinstance(row["pair_id"], str) or not row["pair_id"]:
            raise ValueError("Planned pair is missing required identity or board fields")
        if (row["pair_id"] in ids or not isinstance(row["k"], int) or isinstance(row["k"], bool)
                or row["k"] not in OFFSETS):
            raise ValueError("Duplicate pair ID or unsupported offset")
        ids.add(row["pair_id"])
        key = (row["family"], row["layout_id"])
        old = layouts.setdefault(key, row["source_family"])
        if old != row["source_family"]:
            raise ValueError("A layout belongs to multiple source families")
        rank = row["board_rank"]
        if (layout_ranks.setdefault(key, rank) != rank
                or rank_layouts.setdefault((row["family"], rank), row["layout_id"]) != row["layout_id"]):
            raise ValueError("Board ranks and layouts do not correspond one-to-one within a family")
        context = row.get("context_id")
        context_key = (row["family"], row["layout_id"], row["context_rank"])
        if not isinstance(context, str) or not context or contexts.setdefault(context_key, context) != context:
            raise ValueError("Context rank does not identify one source context per layout")
        offset_key = (*context_key, row["k"])
        if offset_key in context_offsets:
            raise ValueError("Duplicate offset in a source context")
        context_offsets.add(offset_key)
    return rows


def load_unavailable(path: Path, split: str, max_contexts: int, max_boards: int) -> list[dict[str, Any]]:
    """Keep absent offsets and absent contexts distinct from model failures."""
    if not path.is_file():
        raise ValueError(f"Missing unavailable-offset inventory: {path}")
    result = []
    seen = set()
    required = {"family", "source_family", "split", "layout_id", "board_rank", "context_rank", "k", "reason"}
    for row in read_jsonl(path):
        if not required <= row.keys():
            raise ValueError("Unavailable-offset record lacks identity or reason")
        if row["split"] != split:
            continue
        if (not isinstance(row["board_rank"], int) or isinstance(row["board_rank"], bool)
                or not isinstance(row["context_rank"], int) or isinstance(row["context_rank"], bool)
                or row["board_rank"] < 0 or row["context_rank"] < 0):
            raise ValueError("Unavailable-offset ranks must be nonnegative integers")
        if row["board_rank"] >= max_boards or row["context_rank"] >= max_contexts:
            continue
        if row["k"] is not None and (not isinstance(row["k"], int) or isinstance(row["k"], bool)
                                     or row["k"] not in OFFSETS):
            raise ValueError("Unavailable-offset k is not part of the protocol")
        key = (row["family"], row["layout_id"], row["context_rank"], row["k"])
        if key in seen:
            raise ValueError("Duplicate unavailable-offset record")
        seen.add(key)
        result.append(row)
    return result


def validate_frozen_plan(
    frozen: dict[str, Any], plan: list[dict[str, Any]], models: Sequence[str],
    split: str, max_boards: int, max_contexts: int,
) -> None:
    """Require the runner and exporter to use the exact same scheduled pairs."""
    if (frozen.get("split") != split or frozen.get("layout_limit_per_family") != max_boards
            or frozen.get("contexts_per_layout") != max_contexts):
        raise ValueError("Assessment prefix differs from the runner's frozen plan")
    if not set(models) <= set(frozen.get("model_ids", [])):
        raise ValueError("Requested model is absent from the runner's frozen plan")
    expected = {(model, row["family"], row["layout_id"], row["pair_id"]) for row in plan for model in models}
    actual = []
    for shard in frozen.get("planned_shards", []):
        if shard.get("model_id") in models:
            actual.extend((shard["model_id"], shard["family"], shard["layout_id"], pair_id)
                          for pair_id in shard["pair_ids"])
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Runner's planned shards do not exactly match assessment pairs")


def _player(board: np.ndarray) -> list[int]:
    positions = np.argwhere(np.isin(tiles(board), (5, 6)))
    if len(positions) != 1:
        raise ValueError("A state must have exactly one player")
    return [int(value) for value in positions[0]]


def replay_trial(pair: dict[str, Any], trial: dict[str, Any], env: Any | None = None) -> dict[str, Any]:
    """Recompute actions, exact target, terminal state, no-ops and cycles."""
    source, target = from_rows(pair["source"]), from_rows(pair["target"])
    if (state_hash(source), state_hash(target)) != (pair["source_hash"], pair["target_hash"]):
        raise ValueError("Planned endpoint hashes disagree with their boards")
    if canonical_layout_id(source) != pair["layout_id"]:
        raise ValueError("Planned canonical layout ID disagrees with the source board")
    validate_subgoal(source, target)
    if np.array_equal(source, target):
        raise ValueError("A target must not already hold at the starting state")
    simulator = env or environment(source)
    if simulator.is_solved(source):
        raise ValueError("Source is already terminal")
    actions = trial.get("actions")
    if trial.get("action_cap") != ACTION_CAP or not isinstance(actions, list) or not 1 <= len(actions) <= ACTION_CAP:
        raise ValueError("Completed trial must contain 1..32 actions")
    current = source.copy()
    hashes = [state_hash(current)]
    seen = {hashes[0]: 0}
    noops = 0
    cycles = []
    stop = None
    for index, action in enumerate(actions, 1):
        if stop is not None or not isinstance(action, int) or isinstance(action, bool) or action not in range(4):
            raise ValueError("Invalid action or action after a stopping event")
        before = current
        current = np.asarray(simulator.next_state(before, action))
        digest = state_hash(current)
        hashes.append(digest)
        if np.array_equal(before, current):
            noops += 1
        elif digest in seen:
            cycles.append({"start": seen[digest], "end": index, "length": index - seen[digest]})
        seen.setdefault(digest, index)
        hit = bool(np.array_equal(current, target))
        stop = (StopReason.EXACT_TARGET if hit else StopReason.TASK_TERMINAL if simulator.done else
                StopReason.ACTION_LIMIT if index == ACTION_CAP else None)
    if stop is None:
        raise ValueError("Completed trial ended before a stopping event")
    hit = stop == StopReason.EXACT_TARGET
    solved = bool(simulator.done)
    expected = {"state_hashes": hashes, "no_op_count": noops, "attempted_actions": len(actions),
                "first_hit": len(actions) if hit else None, "exact_hit": hit,
                "stop_reason": stop, "task_solved": solved, "final_state": to_rows(current),
                "final_player": _player(current)}
    for key, value in expected.items():
        if trial.get(key) != value:
            raise ValueError(f"Recorded {key} disagrees with replay")
    return {**expected, "cycles": cycles, "cycle_detected": bool(cycles),
            "first_cycle_length": cycles[0]["length"] if cycles else None,
            "source_player": _player(source), "target_player": _player(target)}


def read_shards(
    plan: list[dict[str, Any]], shards: Path, models: Sequence[str], split: str,
    *, expected_contract: str | None = None,
    on_progress: Callable[[int, int, int, int], None] | None = None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, str]]]:
    """Replay complete board/model shards and retain malformed or missing files."""
    expected: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in plan:
        for model in models:
            expected[(split, row["family"], model, row["layout_id"])].append(row)
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    errors = []
    envs: dict[tuple[str, str], Any] = {}
    processed = 0
    for (wanted_split, family, model, layout), pairs in sorted(expected.items()):
        path = shards / wanted_split / family / model / f"{layout}.json"
        if not path.is_file():
            processed += 1
            if on_progress is not None:
                on_progress(processed, len(expected), len(observed), len(errors))
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            identities = {row["pair_id"]: row for row in pairs}
            header = {"schema_version": 1, "status": "complete", "split": wanted_split,
                      "family": family, "model_id": model, "layout_id": layout,
                      "source_family": pairs[0]["source_family"]}
            if any(value.get(key) != wanted for key, wanted in header.items()):
                raise ValueError("Shard header does not match the frozen plan")
            if not isinstance(value.get("contract_sha256"), str) or not value["contract_sha256"]:
                raise ValueError("Shard is missing its controller contract hash")
            if expected_contract is not None and value["contract_sha256"] != expected_contract:
                raise ValueError("Shard belongs to a different frozen controller contract")
            if set(value.get("planned_pair_ids", [])) != set(identities):
                raise ValueError("Shard planned pair IDs do not match the frozen plan")
            trials = value.get("trials")
            if not isinstance(trials, list) or len(trials) != len(identities):
                raise ValueError("Shard is missing or duplicating trials")
            if {row.get("pair_id") for row in trials} != set(identities):
                raise ValueError("Shard trial IDs do not match the frozen plan")
            env_key = (family, layout)
            if env_key not in envs:
                envs[env_key] = environment(from_rows(pairs[0]["source"]))
            pending = {}
            for trial in trials:
                pair = identities[trial["pair_id"]]
                fields = ("pair_id", "layout_id", "family", "source_family", "split", "k",
                          "source_hash", "target_hash")
                if trial.get("status") != "complete" or trial.get("model_id") != model:
                    raise ValueError("Trial is not complete or has the wrong model")
                if any(trial.get(field) != pair[field] for field in fields):
                    raise ValueError("Trial identity differs from its planned pair")
                replayed = replay_trial(pair, trial, envs[env_key])
                pending[(model, pair["pair_id"])] = {**trial, **replayed, "trace_file": str(path)}
            observed.update(pending)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            errors.append({"model": model, "family": family, "layout_id": layout,
                           "path": str(path), "reason": f"{type(error).__name__}: {error}"})
        processed += 1
        if on_progress is not None:
            on_progress(processed, len(expected), len(observed), len(errors))
    return observed, errors


def _bootstrap(values: Sequence[float], seed: int, replicates: int) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    config = EvaluationConfig(run_dirs=["."], bootstrap_seed=seed, bootstrap_replicates=replicates)
    return interval(np.asarray(values, dtype=float), config)


def _hit(trial: dict[str, Any], budget: int) -> bool:
    return trial["first_hit"] is not None and trial["first_hit"] <= budget


def analyse(
    plan: list[dict[str, Any]], observed: dict[tuple[str, str], dict[str, Any]], models: Sequence[str],
    *, unavailable: Sequence[dict[str, Any]] = (), seed: int = 20261115, replicates: int = 2000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate board-weighted success, uncertainty and paired differences."""
    cases = []
    for pair in plan:
        for model in models:
            trial = observed.get((model, pair["pair_id"]))
            cases.append({"pair_id": pair["pair_id"], "layout_id": pair["layout_id"],
                          "context_id": pair.get("context_id"), "trajectory_id": pair.get("trajectory_id"),
                          "family": pair["family"], "source_family": pair["source_family"], "split": pair["split"],
                          "i": pair.get("i"), "k": pair["k"], "model_id": model,
                          "board_rank": pair.get("board_rank"), "context_rank": pair.get("context_rank"),
                          "source_path": pair.get("source_path"), "source_sha256": pair.get("source_sha256"),
                          "status": "complete" if trial is not None else "missing",
                          **({key: trial[key] for key in ("exact_hit", "first_hit", "attempted_actions", "no_op_count",
                              "stop_reason", "task_solved", "final_state", "final_player", "source_player",
                              "target_player", "cycle_detected", "first_cycle_length", "cycles", "trace_file")}
                             if trial is not None else {})})

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in plan:
        groups[(pair["family"], "all")].append(pair)
        groups[(pair["family"], pair["source_family"])].append(pair)
    unavailable_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unavailable:
        unavailable_groups[(row["family"], "all")].append(row)
        unavailable_groups[(row["family"], row["source_family"])].append(row)
    summary, boards, paired = [], [], []
    for (family, source_family), group in sorted(groups.items()):
        unselected = unavailable_groups.get((family, source_family), [])
        frozen_layouts = {row["layout_id"] for row in group} | {row["layout_id"] for row in unselected}
        for k in OFFSETS:
            eligible = [pair for pair in group if pair["k"] == k]
            absent = [row for row in unselected if row["k"] in (None, k)]
            missing_contexts = sum(row["k"] is None for row in absent)
            by_layout: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for pair in eligible:
                by_layout[pair["layout_id"]].append(pair)
            for budget in BUDGETS:
                for model in models:
                    board_rates = []
                    hits = complete = 0
                    durations = []
                    full = []
                    for layout, requested in sorted(by_layout.items()):
                        trials = [observed.get((model, pair["pair_id"])) for pair in requested]
                        present = [trial for trial in trials if trial is not None]
                        success = sum(_hit(trial, budget) for trial in present)
                        missing = len(requested) - len(present)
                        complete += len(present)
                        hits += success
                        durations.extend(trial["first_hit"] for trial in present if _hit(trial, budget))
                        full.extend(present)
                        lower, upper = success / len(requested), (success + missing) / len(requested)
                        board_rates.append((lower, upper, len(present), len(requested)))
                        boards.append({"family": family, "source_family": source_family, "layout_id": layout,
                                       "model_id": model, "k": k, "budget": budget, "planned": len(requested),
                                       "complete": len(present), "hits": success,
                                       "p_hat": lower if not missing else None, "bound_low": lower,
                                       "bound_high": upper})
                    planned = len(eligible)
                    missing = planned - complete
                    lower = float(np.mean([row[0] for row in board_rates])) if board_rates else None
                    upper = float(np.mean([row[1] for row in board_rates])) if board_rates else None
                    available = [row[0] * row[3] / row[2] for row in board_rates if row[2]]
                    low, high = (_bootstrap([row[0] for row in board_rates], seed, replicates)
                                 if not missing and board_rates else (None, None))
                    summary.append({"family": family, "source_family": source_family, "model_id": model,
                                    "k": k, "budget": budget, "frozen_layouts": len(frozen_layouts),
                                    "eligible_layouts": len(by_layout), "unavailable_target_slots": len(absent),
                                    "missing_contexts": missing_contexts, "planned": planned,
                                    "complete": complete, "missing": missing, "hits": hits,
                                    "nonhits_by_budget_among_completed": complete - hits,
                                    "p_hat": lower if not missing else None,
                                    "available_board_mean": float(np.mean(available)) if available else None,
                                    "observed_layouts": len(available), "bound_low": lower, "bound_high": upper,
                                    "bootstrap_low": low, "bootstrap_high": high,
                                    "mean_success_actions": float(np.mean(durations)) if durations else None,
                                    "median_success_actions": float(np.median(durations)) if durations else None,
                                    "mean_attempted_actions_full_rollout": float(np.mean([t["attempted_actions"] for t in full])) if full else None,
                                    "total_no_ops_full_rollout": sum(t["no_op_count"] for t in full),
                                    "trials_with_nontrivial_cycle_full_rollout": sum(t["cycle_detected"] for t in full),
                                    "failures_at_action_limit_full_rollout": sum(t["stop_reason"] == StopReason.ACTION_LIMIT for t in full),
                                    "failures_at_task_terminal_full_rollout": sum(t["stop_reason"] == StopReason.TASK_TERMINAL for t in full),
                                    "task_solved_by_budget": sum(t["task_solved"] and t["attempted_actions"] <= budget for t in full)})
                for left, right in itertools.combinations(models, 2):
                    differences = []
                    paired_count = 0
                    for requested in by_layout.values():
                        vals = []
                        for pair in requested:
                            one, two = observed.get((left, pair["pair_id"])), observed.get((right, pair["pair_id"]))
                            if one is not None and two is not None:
                                vals.append(int(_hit(one, budget)) - int(_hit(two, budget)))
                        paired_count += len(vals)
                        if len(vals) == len(requested):
                            differences.append(float(np.mean(vals)))
                    all_paired = bool(eligible) and paired_count == len(eligible)
                    low, high = _bootstrap(differences, seed, replicates) if all_paired else (None, None)
                    paired.append({"family": family, "source_family": source_family, "left": left, "right": right,
                                   "k": k, "budget": budget, "planned_pairs": len(eligible),
                                   "paired_pairs": paired_count, "missing_pairs": len(eligible) - paired_count,
                                   "paired_layouts": len(differences),
                                   "difference": float(np.mean(differences)) if all_paired else None,
                                   "bootstrap_low": low, "bootstrap_high": high})
    return cases, summary, boards, paired


def write_csv(path: Path, rows: list[dict[str, Any]], *, fields: Sequence[str] | None = None) -> None:
    if not rows and fields is None:
        return
    columns = list(fields) if fields is not None else list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    os.replace(temporary, path)


def make_plot(summary: list[dict[str, Any]], output: Path, models: Sequence[str]) -> None:
    """Plot measured offsets and measured budget curves, with gaps for missing cells."""
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
                         "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 9})
    family_order = {"12x12x4": 0, "12x12x10": 1}
    families = sorted({row["family"] for row in summary}, key=lambda name: (family_order.get(name, 2), name))
    fig, axes = plt.subplots(len(families), 2, figsize=(10, max(3.5, 3.2 * len(families))), squeeze=False)
    for index, family in enumerate(families):
        for model_index, model in enumerate(models):
            offset = (model_index - (len(models) - 1) / 2) * min(.10, .7 / len(models))
            rows = [r for r in summary if r["family"] == family and r["source_family"] == "all"
                    and r["model_id"] == model]
            for column, chosen in enumerate(([r for r in rows if r["budget"] == 32],
                                               [r for r in rows if r["k"] == 16])):
                key = "k" if column == 0 else "budget"
                chosen.sort(key=lambda row: row[key])
                x = [OFFSETS.index(r[key]) + offset for r in chosen]
                y = [np.nan if r["p_hat"] is None else r["p_hat"] for r in chosen]
                marker = ("o", "s", "^", "D", "v", "P")[model_index % 6]
                line = axes[index, column].plot(x, y, marker=marker, linestyle="none", label=model)[0]
                complete = [r for r in chosen if r["p_hat"] is not None and r["bootstrap_low"] is not None]
                if complete:
                    axes[index, column].errorbar([OFFSETS.index(r[key]) + offset for r in complete],
                        [r["p_hat"] for r in complete],
                        yerr=[[r["p_hat"] - r["bootstrap_low"] for r in complete],
                              [r["bootstrap_high"] - r["p_hat"] for r in complete]], fmt="none",
                        color=line.get_color(), capsize=2, alpha=.6)
                partial = [r for r in chosen if r["missing"] and r["available_board_mean"] is not None]
                if partial:
                    axes[index, column].vlines([OFFSETS.index(r[key]) + offset for r in partial],
                                              [r["bound_low"] for r in partial],
                                              [r["bound_high"] for r in partial], color=line.get_color(),
                                              linestyle="dotted", alpha=.7)
                    axes[index, column].plot([OFFSETS.index(r[key]) + offset for r in partial],
                                             [r["available_board_mean"] for r in partial],
                                             marker=marker, markerfacecolor="white", color=line.get_color(), linestyle="none")
        for ax in axes[index]:
            ax.set_ylim(-.04, 1.05)
            ax.grid(axis="y", alpha=.25)
            ax.set_ylabel("Exact-target success")
            ax.set_xlim(-.5, len(OFFSETS) - .5)
            ax.set_xticks(range(len(OFFSETS)), [str(value) for value in OFFSETS])
        axes[index, 0].set(title=f"{family} · 32-action budget", xlabel="Recorded offset k (primitive actions)")
        axes[index, 1].set(title=f"{family} · k = 16", xlabel="Primitive-action budget")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(models)), frameon=False)
    fig.text(.08, .012, "Markers are shifted within each labelled value for clarity; bars: 95% layout bootstrap.\n"
             "Open points/dotted bounds: incomplete cells. Zero/all-one intervals may collapse; k is not shortest distance.",
             fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .94))
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"frozen_policy_reachability.{suffix}", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--unavailable", type=Path, help="Defaults to unavailable_offsets.jsonl next to pairs")
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="evaluation")
    parser.add_argument("--max-contexts", type=int, choices=(1, 2), default=1,
                        help="Match the runner's predeclared contexts per layout")
    parser.add_argument("--max-boards", type=int, default=100,
                        help="Use the same frozen board-rank prefix per family as the runner")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    started = time.monotonic()
    if len(set(args.models)) != len(args.models) or args.bootstrap_replicates < 100 or args.max_boards < 1:
        parser.error("models must be unique, max boards positive and bootstrap replicates at least 100")
    plan = load_plan(args.pairs, args.split, args.max_contexts, args.max_boards)
    unavailable_path = args.unavailable or args.pairs.parent / "unavailable_offsets.jsonl"
    unavailable = load_unavailable(unavailable_path, args.split, args.max_contexts, args.max_boards)
    args.output.mkdir(parents=True, exist_ok=True)

    def checkpoint(processed: int, expected: int, complete: int, invalid: int) -> None:
        payload = {"state": "replaying", "processed_shards": processed, "expected_shards": expected,
                   "completed_model_trials": complete, "invalid_shards": invalid,
                   "elapsed_seconds": time.monotonic() - started}
        path = args.output / "assessment_status.json"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    checkpoint(0, 0, 0, 0)
    contract_path, frozen_path = args.shards.parent / "contract.json", args.shards.parent / "plan.json"
    expected_contract = None
    model_contracts = None
    if contract_path.is_file() and frozen_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if contract.get("plan_sha256") != canonical_sha256(frozen):
            raise ValueError("Controller contract disagrees with its frozen plan")
        if contract.get("input_sha256") != file_sha256(args.pairs):
            raise ValueError("Controller contract input pairs changed since the run")
        sources = contract.get("source_sha256")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("Controller contract is missing source and checkpoint hashes")
        for path, expected in sources.items():
            if not Path(path).is_file() or file_sha256(Path(path)) != expected:
                raise ValueError(f"Controller source or checkpoint changed: {path}")
        validate_frozen_plan(frozen, plan, args.models, args.split, args.max_boards, args.max_contexts)
        expected_contract = canonical_sha256(contract)
        model_contracts = contract.get("models")
    elif args.shards.exists() and any(args.shards.rglob("*.json")):
        raise ValueError("Cannot assess shards without the saved controller contract and plan")
    observed, errors = read_shards(plan, args.shards, args.models, args.split,
                                   expected_contract=expected_contract, on_progress=checkpoint)
    cases, summary, boards, paired = analyse(plan, observed, args.models, unavailable=unavailable, seed=args.seed,
                                              replicates=args.bootstrap_replicates)
    write_csv(args.output / "cases.csv", cases)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "per_board.csv", boards)
    write_csv(args.output / "paired_differences.csv", paired)
    unavailable_fields = ("family", "source_family", "split", "layout_id", "board_rank", "context_rank", "k", "reason")
    if unavailable:
        write_csv(args.output / "unavailable_offsets.csv", unavailable)
    else:
        write_csv(args.output / "unavailable_offsets.csv", [], fields=unavailable_fields)
    failures = [row for row in cases if row["status"] == "complete" and not row["exact_hit"]]
    case_fields = list(dict.fromkeys(key for row in cases for key in row))
    write_csv(args.output / "failure_locations.csv", failures, fields=case_fields)
    make_plot(summary, args.output, args.models)
    runner_status_path = args.shards.parent / "status.json"
    runner_status = (json.loads(runner_status_path.read_text(encoding="utf-8"))
                     if runner_status_path.is_file() else None)
    missing = len(plan) * len(args.models) - len(observed)
    metadata = {"state": "complete" if not missing and not errors else "partial",
                "split": args.split, "max_contexts": args.max_contexts, "max_boards": args.max_boards,
                "models": args.models, "planned_pairs": len(plan),
                "planned_model_trials": len(plan) * len(args.models), "completed_model_trials": len(observed),
                "missing_model_trials": missing, "invalid_shards": errors,
                "controller_contract_sha256": expected_contract,
                "model_input_contracts": model_contracts,
                "runner_status_at_assessment": runner_status,
                "unavailable_offset_records": len(unavailable),
                "bootstrap_seed": args.seed, "bootstrap_replicates": args.bootstrap_replicates,
                "metrics": "Each layout has equal weight. First hit and action budgets count primitive actions.",
                "cycle_rule": "Return to a previously visited full state after a changed action; immediate no-ops are excluded.",
                "elapsed_seconds": time.monotonic() - started,
                "scope": "Evaluation uses exact full-state targets; an ordinary policy may receive only the current "
                         "board. See model_input_contracts. Source distributions do not isolate box count."}
    for name in ("assessment.json", "assessment_status.json"):
        path = args.output / name
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
