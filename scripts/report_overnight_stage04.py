#!/usr/bin/env python3
"""Report the matched stage-four generator experiment from saved results.

The two new-arm comparisons use the same board contexts and original layouts.
The historical checkpoint is shown as a reference, not a matched treatment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shlex
from collections import Counter
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from carl.icl.intent_assessment import bootstrap
from carl.icl.intent_augmentation import transform_coordinate
from scripts.report_overnight_stage03 import atomic_text
from scripts.report_overnight_stage03 import board_stats
from scripts.report_overnight_stage03 import display_ratio
from scripts.report_overnight_stage03 import mode_diagnostics
from scripts.report_overnight_stage03 import read_json
from scripts.report_overnight_stage03 import read_jsonl
from scripts.report_overnight_stage03 import verified_assessment
from scripts.report_overnight_stage03 import write_csv

Record = dict[str, Any]
METHODS = ("base", "historical_checkpoint220", "narrow", "wide", "wide_aug")
LABELS = {"base": "Base", "historical_checkpoint220": "Old LoRA", "narrow": "Narrow",
          "wide": "Wide", "wide_aug": "Wide + D4"}
SOURCE_LABELS = {"rl_validation_12x12x4": "RL validation", "proposal_gamma2_12x12x10": "gamma2",
                 "synthetic_reverse_12x12x10": "reverse"}
TRANSFORMS = ("identity", "rot90_ccw", "horizontal_reflection")
PAIRS = (("narrow", "wide"), ("wide", "wide_aug"))


def total_variation(first: dict[str, int], second: dict[str, int]) -> float | None:
    """Compare normalised categorical marginals without assuming equal totals."""
    total_a, total_b = sum(first.values()), sum(second.values())
    if not total_a or not total_b:
        return None
    return sum(abs(first.get(key, 0) / total_a - second.get(key, 0) / total_b)
               for key in first.keys() | second.keys()) / 2


def dataset_diagnostics(data: Path) -> dict[str, Any]:
    manifest = read_json(data / "dataset_manifest.json")
    arms = {name: read_jsonl(data / f"{name}.jsonl") for name in ("narrow", "wide")}
    validation = read_jsonl(data / "validation.jsonl")
    for name in ("narrow", "wide", "validation"):
        path = data / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["output_sha256"][path.name]:
            raise ValueError(f"Dataset artifact hash changed: {path}")
    narrow_ids = {row["record_id"] for row in arms["narrow"]}
    wide_ids = {row["record_id"] for row in arms["wide"]}
    if len(narrow_ids) != len(arms["narrow"]) or len(wide_ids) != len(arms["wide"]):
        raise ValueError("Duplicate original training records")
    offsets = {name: dict(Counter(str(row["k"]) for row in records)) for name, records in arms.items()}
    details = manifest["detailed_marginals"]
    differences = {name: {"narrow": details["narrow"][name], "wide": details["wide"][name],
                          "total_variation": total_variation(details["narrow"][name], details["wide"][name])}
                   for name in ("predicate_counts", "conjunction_sizes", "goal_threshold_counts", "coordinate_counts")}
    differences["fragment_action_offsets"] = {"narrow": offsets["narrow"], "wide": offsets["wide"],
                                                "total_variation": total_variation(offsets["narrow"], offsets["wide"])}
    return {"manifest_sha256": hashlib.sha256((data / "dataset_manifest.json").read_bytes()).hexdigest(),
            "training_examples": {name: len(rows) for name, rows in arms.items()},
            "training_layouts": {name: len({row["layout_id"] for row in rows}) for name, rows in arms.items()},
            "common_original_examples": len(narrow_ids & wide_ids), "validation_examples": len(validation),
            "validation_layouts": len({row["layout_id"] for row in validation}),
            "predicate_distributions": manifest["predicate_distributions"], "marginals": differences,
            "train_families": manifest["adaptation_training_families"],
            "original_split": manifest["original_split"], "x10_exposure": manifest["x10_exposure"],
            "input_fields": manifest["input_fields"], "evidence_only_fields": manifest["evidence_only_fields"],
            "narrow_sampling_correction": manifest.get("narrow_sampling_correction"),
            "output_sha256": manifest["output_sha256"], "source_sha256": manifest["source_sha256"]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset_revision(stage: Path, data_dir: Path) -> Record:
    """Verify the recorded, train-only replacement without changing the original freeze."""
    original_dir = stage / "data_v2"
    old_manifest_path = original_dir / "dataset_manifest.json"
    new_manifest_path = data_dir / "dataset_manifest.json"
    freeze_path = stage / "training_freeze.json"
    freeze = read_json(freeze_path)
    old_sha, new_sha = sha256(old_manifest_path), sha256(new_manifest_path)
    if freeze["data_sha256"]["dataset_manifest"] != old_sha:
        raise ValueError("The originally frozen dataset manifest changed")
    old = read_json(old_manifest_path)
    if new_sha == old_sha:
        return {"kind": "original_v2", "original_manifest_sha256": old_sha,
                "selected_manifest_sha256": new_sha, "paths": [str(old_manifest_path)]}

    decision_path = stage / "training_freeze_amendment_narrow.json"
    realized_path = stage / "training_freeze_amendment_narrow_realized.json"
    revised_freeze_path = stage / "training_freeze_v3.json"
    audit_path = stage / "data_v3_independent_audit.json"
    sample_path = stage / "narrow_sampling_preflight_v3.json"
    token_path = stage / "token_preflight_v3.json"
    decision, realized, revised = (read_json(path) for path in (decision_path, realized_path, revised_freeze_path))
    audit, new = read_json(audit_path), read_json(new_manifest_path)
    expected = {
        "prior_training_freeze_sha256": sha256(freeze_path),
        "decision_amendment_sha256": sha256(decision_path),
        "old_dataset_manifest_sha256": old_sha, "new_dataset_manifest_sha256": new_sha,
        "old_narrow_sha256": sha256(original_dir / "dataset_narrow.jsonl"),
        "new_narrow_sha256": sha256(data_dir / "dataset_narrow.jsonl"),
        "wide_v2_sha256": sha256(original_dir / "dataset_wide.jsonl"),
        "wide_v3_sha256": sha256(data_dir / "dataset_wide.jsonl"),
        "validation_v2_sha256": sha256(original_dir / "validation.jsonl"),
        "validation_v3_sha256": sha256(data_dir / "validation.jsonl"),
        "data_v3_audit_sha256": sha256(audit_path),
        "sampling_preflight_sha256": sha256(sample_path), "token_preflight_sha256": sha256(token_path),
    }
    if any(realized.get(key) != value for key, value in expected.items()):
        raise ValueError("Train-only dataset revision does not match its recorded hashes")
    correction = new.get("narrow_sampling_correction", {})
    if (decision.get("previous_training_plan_sha256") != sha256(freeze_path)
            or not realized.get("no_new_test_results_used")
            or not realized.get("narrow_layouts_unchanged")
            or not realized.get("wide_and_validation_unchanged")
            or expected["wide_v2_sha256"] != expected["wide_v3_sha256"]
            or expected["validation_v2_sha256"] != expected["validation_v3_sha256"]
            or old["splits"] != new["splits"]
            or old["arms"]["narrow"]["train_layout_ids"] != new["arms"]["narrow"]["train_layout_ids"]
            or old["arms"]["wide"] != new["arms"]["wide"]
            or correction.get("source_dataset_manifest_sha256") != old_sha
            or correction.get("dataset_wide_sha256_unchanged") != expected["wide_v2_sha256"]
            or audit.get("status") != "pass" or audit.get("dataset_manifest_sha256") != new_sha
            or revised.get("parent_training_freeze_sha256") != sha256(freeze_path)
            or revised.get("realized_amendment_sha256") != sha256(realized_path)
            or revised["data_sha256"]["dataset_manifest"] != new_sha):
        raise ValueError("Dataset revision violates the pre-test train-only amendment")
    return {"kind": "balanced_narrow_v3", "original_manifest_sha256": old_sha,
            "selected_manifest_sha256": new_sha, "decision": decision, "realized": realized,
            "paths": [str(path) for path in (old_manifest_path, decision_path, realized_path,
                                               revised_freeze_path, audit_path, sample_path, token_path)]}


def metric(row: Record, name: str) -> float | None:
    if row.get("output_status") != "complete":
        return None
    if name == "valid":
        return float(row["valid_nontrivial"])
    if name == "bounded_path":
        if not row["valid_nontrivial"]:
            return 0.0
        found = row.get("bounded_path_found")
        return float(found) if isinstance(found, bool) else None
    raise ValueError("Unknown metric")


def paired_differences(rows: list[Record]) -> list[Record]:
    """Compare matched arms per original board, never treating D4 as new boards."""
    groups: dict[tuple[str, str, str, str], dict[str, Record]] = defaultdict(dict)
    for row in rows:
        if row["condition"] not in ("original", "d4"):
            continue
        key = (row["family"], row["source_family"], row["transform"], row["layout_id"])
        if row["method"] in groups[key]:
            raise ValueError("Duplicate method for the same original board/transform")
        groups[key][row["method"]] = row
    strata = sorted({key[:3] for key in groups})
    results = []
    for family, source, transform in strata:
        group = {key[3]: value for key, value in groups.items() if key[:3] == (family, source, transform)}
        for first, second in PAIRS:
            for name in ("valid", "bounded_path"):
                differences = []
                for layout in sorted(group):
                    arms = group[layout]
                    if first not in arms or second not in arms:
                        continue
                    a, b = metric(arms[first], name), metric(arms[second], name)
                    if a is not None and b is not None:
                        differences.append(b - a)
                seed = int.from_bytes(hashlib.sha256(
                    f"stage04\0{family}\0{source}\0{transform}\0{first}\0{second}\0{name}".encode()
                ).digest()[:8], "big")
                ci = bootstrap(differences, random.Random(seed), count=2000)
                results.append({"family": family, "source_family": source, "transform": transform,
                                "first": first, "second": second, "metric": name,
                                "planned_boards": len(group), "paired_boards": len(differences),
                                "difference_second_minus_first": sum(differences) / len(differences) if differences else None,
                                "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None})
    return results


def family_paired_differences(rows: list[Record]) -> list[Record]:
    """Pool sources within each family while preserving one vote per board."""
    source_by_board = {}
    for row in rows:
        if row["condition"] not in ("original", "d4"):
            continue
        key = (row["family"], row["layout_id"])
        previous = source_by_board.setdefault(key, row["source_family"])
        if previous != row["source_family"]:
            raise ValueError("An original board was assigned multiple sources")
    pooled = [{**row, "source_family": "pooled"} for row in rows]
    results = paired_differences(pooled)
    for result in results:
        family, transform = result["family"], result["transform"]
        planned = Counter(source for (item_family, _), source in source_by_board.items() if item_family == family)
        selected = [row for row in rows if row["family"] == family and row["transform"] == transform
                    and row["condition"] in ("original", "d4")]
        groups: dict[str, dict[str, Record]] = defaultdict(dict)
        for row in selected:
            groups[row["layout_id"]][row["method"]] = row
        paired = Counter()
        for layout, methods in groups.items():
            if result["first"] not in methods or result["second"] not in methods:
                continue
            if (metric(methods[result["first"]], result["metric"]) is not None
                    and metric(methods[result["second"]], result["metric"]) is not None):
                paired[source_by_board[(family, layout)]] += 1
        if sum(paired.values()) != result["paired_boards"]:
            raise ValueError("Paired source counts do not match the paired denominator")
        result["planned_source_board_counts"] = dict(sorted(planned.items()))
        result["paired_source_board_counts"] = dict(sorted(paired.items()))
    return results


def figure_data(rows: list[Record]) -> list[Record]:
    output = []
    for family in ("12x12x4", "12x12x10"):
        for method in METHODS:
            for transform in TRANSFORMS:
                selected = [row for row in rows if row["family"] == family and row["method"] == method
                            and row["transform"] == transform and row["condition"] in ("original", "d4")]
                if not selected:
                    raise ValueError(f"Missing planned cases for {family}/{method}/{transform}")
                sources = {source: len({row["layout_id"] for row in selected if row["source_family"] == source})
                           for source in sorted({row["source_family"] for row in selected})}
                for metric_name in ("valid", "bfs"):
                    identity = f"stage04\0{family}\0{method}\0{transform}\0{metric_name}"
                    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
                    result = board_stats(selected, metric_name, seed)
                    output.append({"family": family, "method": method, "transform": transform,
                                   "metric": metric_name, "source_board_counts": sources, **result,
                                   "valid_outputs": sum(row.get("valid_nontrivial") is True for row in selected),
                                   "bounded_paths_found": sum(row.get("bounded_path_found") is True for row in selected)})
    return output


def repetition_diagnostics(rows: list[Record]) -> list[Record]:
    """Include semantically invalid JSON-parseable coordinate requests."""
    groups: dict[tuple[str, str, str], list[Record]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "original" and row["transform"] == "identity":
            groups[(row["family"], row["source_family"], row["method"])].append(row)
    result = []
    for (family, source, method), group in sorted(groups.items()):
        complete = [row for row in group if row["output_status"] == "complete"]
        coordinates, singles = Counter(), 0
        for row in complete:
            parsed = row.get("parsed_intent")
            effects = parsed.get("effects") if isinstance(parsed, dict) else None
            if not isinstance(effects, list):
                continue
            boxes = [item for item in effects if isinstance(item, dict) and item.get("predicate") == "box_at"
                     and isinstance(item.get("row"), int) and not isinstance(item.get("row"), bool)
                     and isinstance(item.get("col"), int) and not isinstance(item.get("col"), bool)]
            singles += len(effects) == 1 and len(boxes) == 1
            coordinates.update({f"{item['row']},{item['col']}" for item in boxes})
        count = max(coordinates.values(), default=0)
        top = min((key for key, value in coordinates.items() if value == count), default=None)
        result.append({"family": family, "source_family": source, "method": method,
                       "requested_outputs": len(group), "completed_outputs": len(complete),
                       "single_box_at_outputs": singles, "most_common_box_coordinate": top,
                       "most_common_box_coordinate_outputs": count,
                       "share_of_completed": count / len(complete) if complete else None})
    return result


def family_repetition(rows: list[Record]) -> list[Record]:
    """Pool the original-orientation outputs with one output per original board."""
    results = []
    for family in ("12x12x4", "12x12x10"):
        for method in METHODS:
            selected = [row for row in rows if row["family"] == family and row["method"] == method
                        and row["condition"] == "original" and row["transform"] == "identity"]
            layouts = [row["layout_id"] for row in selected]
            if not selected or len(layouts) != len(set(layouts)):
                raise ValueError(f"Expected one original output per board for {family}/{method}")
            sources = {source: sum(row["source_family"] == source for row in selected)
                       for source in sorted({row["source_family"] for row in selected})}
            completed_sources = {source: sum(row["source_family"] == source and row["output_status"] == "complete"
                                             for row in selected) for source in sources}
            pooled = [{**row, "source_family": "pooled"} for row in selected]
            item = repetition_diagnostics(pooled)[0]
            completed = item["completed_outputs"]
            results.append({"family": family, "method": method, "planned_source_board_counts": sources,
                            "completed_source_board_counts": completed_sources,
                            "planned_boards": len(selected), "completed_boards": completed,
                            "single_box_at_outputs": item["single_box_at_outputs"],
                            "single_box_at_share": item["single_box_at_outputs"] / completed if completed else None,
                            "most_common_box_coordinate": item["most_common_box_coordinate"],
                            "most_common_box_coordinate_outputs": item["most_common_box_coordinate_outputs"],
                            "most_common_box_coordinate_share": item["share_of_completed"]})
    return results


def predicate_coverage(rows: list[Record]) -> list[Record]:
    """Count predicates per parseable output and separately per valid output."""
    results = []
    for family in ("12x12x4", "12x12x10"):
        for method in METHODS:
            selected = [row for row in rows if row["family"] == family and row["method"] == method
                        and row["condition"] == "original" and row["transform"] == "identity"]
            if not selected or len({row["layout_id"] for row in selected}) != len(selected):
                raise ValueError(f"Expected one original output per board for {family}/{method}")
            completed = [row for row in selected if row["output_status"] == "complete"]
            parsed = [row for row in completed if row.get("json_valid") is True]
            valid = [row for row in parsed if row.get("valid_nontrivial") is True]
            def predicates(row: Record) -> set[str]:
                intent = row.get("parsed_intent")
                effects = intent.get("effects") if isinstance(intent, dict) else None
                return {item["predicate"] for item in effects if isinstance(item, dict)
                        and isinstance(item.get("predicate"), str)} if isinstance(effects, list) else set()
            def is_conjunction(row: Record) -> bool:
                intent = row.get("parsed_intent")
                effects = intent.get("effects") if isinstance(intent, dict) else None
                return isinstance(effects, list) and len(effects) >= 2
            def presence(group: list[Record]) -> dict[str, int]:
                counts = Counter(predicate for row in group for predicate in predicates(row))
                return dict(sorted(counts.items()))
            def together(group: list[Record]) -> int:
                return sum({"box_at", "boxes_on_goals_at_least"} <= predicates(row) for row in group)
            conjunctions = [row for row in parsed if is_conjunction(row)]
            valid_conjunctions = [row for row in valid if is_conjunction(row)]
            results.append({"family": family, "method": method, "requested_outputs": len(selected),
                            "completed_outputs": len(completed), "parseable_outputs": len(parsed),
                            "valid_outputs": len(valid), "predicate_presence_parseable": presence(parsed),
                            "predicate_presence_valid": presence(valid),
                            "box_and_goal_parseable": together(parsed), "box_and_goal_valid": together(valid),
                            "conjunctions_parseable": len(conjunctions), "conjunctions_valid": len(valid_conjunctions),
                            "conjunctions_any_effect_initially_true": sum(
                                row.get("any_effect_initially_true") is True for row in conjunctions),
                            "conjunctions_initial_truth_unknown": sum(
                                row.get("any_effect_initially_true") is None for row in conjunctions)})
    return results


def plot_repetition(data: list[Record], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "serif", "font.size": 8.0, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True, layout="constrained")
    colours = {"single_box_at": "#28608b", "coordinate_mode": "#c47938"}
    for axis, family in zip(axes, ("12x12x4", "12x12x10"), strict=True):
        entries = [next(row for row in data if row["family"] == family and row["method"] == method)
                   for method in METHODS]
        for index, row in enumerate(entries):
            for shift, key, count, colour in (
                (-.18, "single_box_at_share", "single_box_at_outputs", colours["single_box_at"]),
                (.18, "most_common_box_coordinate_share", "most_common_box_coordinate_outputs", colours["coordinate_mode"]),
            ):
                value = row[key]
                if value is None:
                    axis.text(index + shift, .025, "n/a", ha="center", fontsize=6.5)
                    continue
                axis.bar(index + shift, value, .32, color=colour)
                axis.text(index + shift, value + .025, f"{row[count]}/{row['completed_boards']}",
                          ha="center", va="bottom", fontsize=6.2)
        labels = [f"{LABELS[row['method']]}\n{row['completed_boards']}/{row['planned_boards']}\n"
                  f"{row['most_common_box_coordinate'] or '—'}" for row in entries]
        axis.set_xticks(range(len(METHODS)), labels)
        title = "12×12×4 · RL validation" if family == "12x12x4" else "12×12×10 · mixed gamma2 / reverse"
        axis.set_title(title)
        axis.set_ylim(0, 1.15)
        axis.set_yticks((0, .25, .5, .75, 1), ("0%", "25%", "50%", "75%", "100%"))
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=.2)
        axis.set_xlabel("Method / completed ÷ planned boards / most common coordinate", fontsize=7)
    axes[0].set_ylabel("Share of completed outputs")
    figure.legend(handles=[Patch(color=colours["single_box_at"], label="Single box_at intent"),
                           Patch(color=colours["coordinate_mode"], label="Most common box coordinate")],
                  loc="outside upper center", ncol=2, frameon=False)
    for extension in ("pdf", "svg", "png"):
        path = output / f"stage_04_repetition.{extension}"
        temporary = output / f".stage_04_repetition.{extension}.{os.getpid()}.tmp"
        try:
            figure.savefig(temporary, format=extension, dpi=180, bbox_inches="tight")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(figure)


def plot_summary(data: list[Record], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "serif", "font.size": 8.0, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), sharey=True, layout="constrained")
    colors = {"identity": "#21618c", "rot90_ccw": "#47834f", "horizontal_reflection": "#865a9b"}
    names = {"identity": "Original", "rot90_ccw": "Rotated", "horizontal_reflection": "Reflected"}
    for (metric_name, family), axis in zip(
            ((metric_name, family) for metric_name in ("valid", "bfs") for family in ("12x12x4", "12x12x10")),
            axes.flat, strict=True):
        for index, method in enumerate(METHODS):
            for offset, transform in zip((-.18, 0.0, .18), TRANSFORMS, strict=True):
                row = next(item for item in data if item["family"] == family and item["method"] == method
                           and item["transform"] == transform and item["metric"] == metric_name)
                position = index + offset
                if row["mean"] is None:
                    axis.text(position, .03, "n/a", ha="center", fontsize=7, color=colors[transform])
                    continue
                if row["ci_low"] is not None and row["ci_high"] is not None:
                    axis.vlines(position, row["ci_low"], row["ci_high"], color=colors[transform], linewidth=.9)
                    axis.hlines((row["ci_low"], row["ci_high"]), position - .025, position + .025,
                                color=colors[transform], linewidth=.9)
                axis.plot(position, row["mean"], "o", color=colors[transform], markersize=3.7)
        first = next(item for item in data if item["family"] == family)
        source = ", ".join(f"{SOURCE_LABELS.get(key, key)}: {value}"
                           for key, value in first["source_board_counts"].items())
        axis.set_title(f"{family} · {first['planned_boards']} planned boards\n{source}", fontsize=8)
        axis.set_xticks(range(len(METHODS)), [LABELS[name] for name in METHODS], rotation=24, ha="right")
        axis.set_ylim(-.03, 1.03)
        axis.set_xlim(-.5, len(METHODS) - .5)
        axis.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylabel("Valid unsatisfied intent / known output")
    axes[1, 0].set_ylabel("Bounded BFS path / known output")
    handles = [Line2D([], [], marker="o", linestyle="none", color=colors[key], label=names[key]) for key in TRANSFORMS]
    figure.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    for extension in ("pdf", "svg", "png"):
        path = output / f"stage_04_figure.{extension}"
        temporary = output / f".stage_04_figure.{extension}.{os.getpid()}.tmp"
        try:
            figure.savefig(temporary, format=extension, dpi=180, bbox_inches="tight")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(figure)


def training_curves(arm_dirs: dict[str, Path], data: Path) -> tuple[list[Record], list[Record]]:
    curves, summaries = [], []
    original = {arm: {row["record_id"]: row for row in read_jsonl(data / f"{arm}.jsonl")}
                for arm in ("narrow", "wide")}
    for arm in ("narrow", "wide", "wide_aug"):
        directory = arm_dirs.get(arm)
        if directory is None or not (directory / "training_curve.jsonl").is_file():
            summaries.append({"arm": arm, "state": "not_available", "updates": 0, "best": None})
            continue
        records = read_jsonl(directory / "training_curve.jsonl")
        state = read_json(directory / "status.json") if (directory / "status.json").is_file() else {}
        best = read_json(directory / "best.json") if (directory / "best.json").is_file() else None
        configuration = read_json(directory / "configuration.json") if (directory / "configuration.json").is_file() else {}
        for row in records:
            curves.append({"arm": arm, **{key: row.get(key) for key in
                           ("step", "train_loss", "validation_loss", "model_tokens", "effective_tokens", "supervised_tokens")}})
        sampled = Counter()
        offsets = Counter()
        goal_thresholds = Counter()
        coordinates = Counter()
        source = original["wide" if arm == "wide_aug" else arm]
        for row in records:
            for sample in row.get("samples", []):
                record = source.get(sample["record_id"])
                if record is None or record["layout_id"] != sample["layout_id"]:
                    raise ValueError(f"{arm} sampled an unknown or mismatched training record")
                sampled[record["effect_type"]] += 1
                offsets[str(record["k"])] += 1
                for predicate in record["intent"]["effects"]:
                    if predicate["predicate"] == "boxes_on_goals_at_least":
                        goal_thresholds[str(predicate["count"])] += 1
                    elif predicate["predicate"] in ("box_at", "player_at"):
                        location = transform_coordinate(predicate["row"], predicate["col"],
                                                        len(record["start_observation"]), sample["symmetry"])
                        coordinates[f"{location[0]},{location[1]}"] += 1
        summaries.append({"arm": arm, "state": state.get("status", "unknown"), "updates": len(records),
                          "seed": configuration.get("args", {}).get("seed"),
                          "best": best, "first_train_loss": records[0]["train_loss"] if records else None,
                          "last_train_loss": records[-1]["train_loss"] if records else None,
                          "model_tokens": sum(row.get("model_tokens", 0) for row in records),
                          "effective_tokens": sum(row.get("effective_tokens", 0) for row in records),
                          "supervised_tokens": sum(row.get("supervised_tokens", 0) for row in records),
                          "symmetries": dict(Counter(sample["symmetry"] for row in records for sample in row.get("samples", []))),
                          "sampled_effect_types": dict(sampled), "sampled_offsets": dict(offsets),
                          "sampled_goal_thresholds": dict(goal_thresholds), "sampled_coordinates": dict(coordinates),
                          "sampled_examples": sum(sampled.values())})
    return curves, summaries


def plot_training(curves: list[Record], training: list[Record], output: Path) -> None:
    if not curves:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"narrow": "#a05195", "wide": "#21618c", "wide_aug": "#47834f"}
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharex=True, layout="constrained")
    for arm in ("narrow", "wide", "wide_aug"):
        subset = [row for row in curves if row["arm"] == arm]
        if not subset:
            continue
        axes[0].plot([row["step"] for row in subset], [row["train_loss"] for row in subset],
                     linewidth=.8, color=colors[arm], label=LABELS[arm])
        val = [row for row in subset if row["validation_loss"] is not None]
        if val:
            axes[1].plot([row["step"] for row in val], [row["validation_loss"] for row in val],
                         marker="o", markersize=2.5, linewidth=.8, color=colors[arm])
            metadata = next(row for row in training if row["arm"] == arm)
            best = metadata.get("best") or {}
            chosen = [row for row in val if row["step"] == best.get("step")]
            if chosen:
                axes[1].plot(chosen[0]["step"], chosen[0]["validation_loss"], marker="*", markersize=8,
                             color=colors[arm])
    for axis, title in zip(axes, ("Training loss", "Validation loss"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Optimiser updates")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Response-token cross-entropy")
    axes[0].legend(frameon=False, fontsize=8)
    for extension in ("pdf", "svg", "png"):
        path = output / f"stage_04_training.{extension}"
        temp = output / f".stage_04_training.{extension}.{os.getpid()}.tmp"
        try:
            fig.savefig(temp, format=extension, dpi=180, bbox_inches="tight")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    plt.close(fig)


def recorded_commands(supervisor: dict[str, Any]) -> list[dict[str, str]]:
    """Extract the actual supervised commands, without reconstructing argv."""
    names = ("stage04_prepare_data_v2", "stage04_prepare_data_v3", "stage04_prepare_balanced_narrow",
             "stage04_matched_gpu_benchmark", "stage04_cpu_development_benchmark", "stage04_train_",
             "stage04_evaluation", "stage04_eval_reconcile", "stage04_assess", "stage04_report")
    rows = []
    for stage in supervisor["stages"]:
        if any(stage["name"].startswith(name) for name in names) and stage.get("command"):
            rows.append({"name": stage["name"], "status": str(stage["status"]),
                         "command": shlex.join(stage["command"])})
    return rows


def evaluation_stop_amendment(supervisor: Record) -> Record | None:
    """Describe a watchdog cutoff amendment without implying child argv changed."""
    stages = [stage for stage in supervisor["stages"] if stage["name"] == "stage04_evaluation"]
    if not stages or "stop_amendment" not in stages[0]:
        return None
    if len(stages) != 1:
        raise ValueError("Ambiguous evaluation watchdog history")
    stage = stages[0]
    command = stage["command"]
    def command_float(flag: str) -> float:
        return float(command[command.index(flag) + 1])
    def timestamp(value: float) -> str:
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    before = stage["stop_amendment"]["original"]
    return {"amended_utc": stage["stop_amendment"]["amended_utc"],
            "reason": stage["stop_amendment"]["reason"],
            "original_watchdog_utc": timestamp(before["stop_epoch"]),
            "original_watchdog_epoch": before["stop_epoch"],
            "original_watchdog_monotonic": before["stop_monotonic"],
            "revised_watchdog_utc": timestamp(stage["stop_epoch"]),
            "revised_watchdog_epoch": stage["stop_epoch"],
            "revised_watchdog_monotonic": stage["stop_monotonic"],
            "child_cutoff_utc": timestamp(command_float("--stop-epoch")),
            "child_cutoff_epoch": command_float("--stop-epoch"),
            "child_cutoff_monotonic": command_float("--stop-monotonic")}


def representative_examples(plan: Record, rows: list[Record]) -> list[Record]:
    """Pick the first frozen board in each family, independently of its outcomes."""
    examples = []
    for family in ("12x12x4", "12x12x10"):
        candidates = [context for context in plan["context_manifest"] if context["family"] == family]
        context = min(candidates, key=lambda item: (item["board_rank"], item["context_id"]))
        chosen = [row for row in rows if row["context_id"] == context["context_id"]
                  and row["condition"] == "original" and row["transform"] == "identity"]
        by_method = {row["method"]: row for row in chosen}
        if len(by_method) != len(chosen) or set(by_method) != set(METHODS):
            raise ValueError(f"Incomplete planned examples for {family}")
        if any(row["source_rows"] != context["original_rows"] for row in chosen):
            raise ValueError("Example board differs from the frozen plan")
        examples.append({"family": family, "source_family": context["source_family"],
                         "board_rank": context["board_rank"], "context_id": context["context_id"],
                         "layout_id": context["layout_id"], "board": context["original_rows"],
                         "outputs": [{key: by_method[method].get(key) for key in
                                      ("method", "output_status", "raw_output", "json_valid", "valid_nontrivial",
                                       "bfs_status", "bounded_path_found", "bfs_search")}
                                     for method in METHODS]})
    return examples


def example_lines(examples: list[Record]) -> list[str]:
    lines = ["## Two preselected example boards", "",
             "For each family, the board with the lowest frozen `board_rank` was chosen independently of "
             "the generated outputs. These examples illustrate the outputs; they are not additional "
             "independent results. A BFS path is a solver check, not execution by a controller.", ""]
    for example in examples:
        lines += [f"### {example['family']}, source {example['source_family']}, rank {example['board_rank']}",
                  "", "```text", *example["board"], "```", ""]
        for output in example["outputs"]:
            search = output.get("bfs_search") or {}
            detail = (f"path found in {search.get('witness_depth')} actions, "
                      f"{search.get('expanded_nodes')} nodes expanded" if output["bounded_path_found"] is True
                      else str(output.get("bfs_status") or "not measured"))
            lines += [f"**{output['method']}** — output: {output['output_status']}; JSON valid: "
                      f"{output['json_valid']}; valid unsatisfied intent: {output['valid_nontrivial']}; "
                      f"bounded BFS: {detail}.", ""]
            raw = output["raw_output"] if output["raw_output"] is not None else "[no output saved]"
            longest = max((len(match.group()) for match in re.finditer(r"`+", raw)), default=0)
            fence = "`" * max(3, longest + 1)
            lines += [f"{fence}text", raw, fence, ""]
    return lines


def overview_lines(figure: list[Record], repetitions: list[Record], pairs: list[Record]) -> list[str]:
    """Provide a descriptive, denominator-explicit overview of the frozen primary outputs."""
    lines = ["## Primary results: original orientations", "",
             "Each method was asked once per original board. Invalid responses remain in requested-output "
             "denominators; missing outputs are shown in the completed/planned column. The ×10 family "
             "combines two different sources, so family differences are not isolated effects of box count.", "",
             "| Family | Method | Completed / planned | Valid intent / requested | BFS path / requested | "
             "Single box_at / completed | Most common coordinate / completed |",
             "|---|---|---:|---:|---:|---:|---|"]
    for family in ("12x12x4", "12x12x10"):
        for method in ("narrow", "wide", "wide_aug"):
            value = next(row for row in figure if row["family"] == family and row["method"] == method
                         and row["transform"] == "identity" and row["metric"] == "valid")
            repeat = next(row for row in repetitions if row["family"] == family and row["method"] == method)
            requested, completed = value["planned_outputs"], value["completed_outputs"]
            coordinate = repeat["most_common_box_coordinate"] or "n/a"
            lines.append(f"| {family} | {method} | {completed}/{requested} | "
                         f"{display_ratio(value['valid_outputs'], requested)} | "
                         f"{display_ratio(value['bounded_paths_found'], requested)} | "
                         f"{display_ratio(repeat['single_box_at_outputs'], completed)} | "
                         f"{coordinate}: {display_ratio(repeat['most_common_box_coordinate_outputs'], completed)} |")
    lines += ["", "A bounded BFS path means only that the fixed-budget solver found an action sequence; "
              "it is not evidence that a policy executed the intent or that the puzzle was solved. "
              "The most common coordinate is chosen after observing the outputs and is descriptive.", "",
              "| Family | Comparison | Metric | Paired / planned boards | Paired sources | Difference | 95% board bootstrap |",
              "|---|---|---|---:|---|---:|---:|"]
    for row in pairs:
        if row["transform"] != "identity":
            continue
        value, low, high = row["difference_second_minus_first"], row["ci_low"], row["ci_high"]
        difference = f"{value:+.3f}" if value is not None else "n/a"
        interval = f"[{low:+.3f}, {high:+.3f}]" if low is not None and high is not None else "n/a"
        sources = ", ".join(f"{SOURCE_LABELS.get(source, source)}: {count}"
                            for source, count in row["paired_source_board_counts"].items()) or "n/a"
        lines.append(f"| {row['family']} | {row['second']} − {row['first']} | {row['metric']} | "
                     f"{row['paired_boards']}/{row['planned_boards']} | {sources} | {difference} | {interval} |")
    lines += ["", "The planned ×10 cohort contains gamma2 and reverse boards; source-specific paired "
              "results are reported below. Missing observations can change the paired source mix, as shown "
              "above. BFS errors or missing outputs are excluded from the relevant "
              "paired calculation. Wide versus narrow changes examples as well as the number of layouts. "
              "The single training seed and previously inspected evaluation boards limit interpretation.", ""]
    return lines


def descriptive_findings_lines(coverage: list[Record], repetitions: list[Record], pairs: list[Record]) -> list[str]:
    lines = ["## Descriptive comparison of predicate use", "",
             "Counts below include every JSON-parseable completed output, even when the intent is "
             "semantically invalid. The second predicate column restricts counts to valid non-trivial "
             "intents. A predicate is counted at most once per output; conjunctions contain at least two "
             "requested predicates. Reasons for invalidity can overlap.", "",
             "| Family | Method | Parseable / completed | Valid / requested | "
             "Predicate presence among parseable outputs | Predicate presence among valid outputs | "
             "Conjunctions: parseable / valid | Conjunctions with any initially true predicate |",
             "|---|---|---:|---:|---|---|---:|---:|"]
    for item in coverage:
        lines.append(f"| {item['family']} | {item['method']} | {item['parseable_outputs']}/{item['completed_outputs']} | "
                     f"{item['valid_outputs']}/{item['requested_outputs']} | "
                     f"`{item['predicate_presence_parseable']}` | `{item['predicate_presence_valid']}` | "
                     f"{item['conjunctions_parseable']} / {item['conjunctions_valid']} | "
                     f"{item['conjunctions_any_effect_initially_true']}/{item['conjunctions_parseable']} "
                     f"({item['conjunctions_initial_truth_unknown']} unknown) |")
    lines += [""]
    for family in ("12x12x4", "12x12x10"):
        wide = next(item for item in coverage if item["family"] == family and item["method"] == "wide")
        aug = next(item for item in coverage if item["family"] == family and item["method"] == "wide_aug")
        rep_wide = next(item for item in repetitions if item["family"] == family and item["method"] == "wide")
        rep_aug = next(item for item in repetitions if item["family"] == family and item["method"] == "wide_aug")
        pair = next(item for item in pairs if item["family"] == family and item["transform"] == "identity"
                    and item["first"] == "wide" and item["second"] == "wide_aug" and item["metric"] == "valid")
        box_wide = wide["predicate_presence_parseable"].get("box_at", 0)
        box_aug = aug["predicate_presence_parseable"].get("box_at", 0)
        valid_wide = wide["predicate_presence_valid"].get("box_at", 0)
        valid_aug = aug["predicate_presence_valid"].get("box_at", 0)
        low, high = pair["ci_low"], pair["ci_high"]
        interval = f"[{low:+.3f}, {high:+.3f}]" if low is not None and high is not None else "n/a"
        lines.append(f"For **{family}**, wide → wide+D4 produced a single `box_at` in "
                     f"{rep_wide['single_box_at_outputs']}/{rep_wide['completed_boards']} → "
                     f"{rep_aug['single_box_at_outputs']}/{rep_aug['completed_boards']} completed outputs. "
                     f"Across parseable outputs, `box_at` appeared in {box_wide}/{wide['parseable_outputs']} → "
                     f"{box_aug}/{aug['parseable_outputs']}; within valid outputs it appeared in "
                     f"{valid_wide}/{wide['valid_outputs']} → {valid_aug}/{aug['valid_outputs']}. "
                     f"Outputs containing both `box_at` and `boxes_on_goals_at_least` were "
                     f"{wide['box_and_goal_parseable']}/{wide['parseable_outputs']} → "
                     f"{aug['box_and_goal_parseable']}/{aug['parseable_outputs']}. "
                     f"Wide+D4 produced {aug['conjunctions_parseable']} parseable conjunctions, "
                     f"{aug['conjunctions_valid']} valid and {aug['conjunctions_any_effect_initially_true']} "
                     f"with at least one initially satisfied predicate "
                     f"({aug['conjunctions_initial_truth_unknown']} with unknown initial truth).")
        if (aug["parseable_outputs"] and box_aug == aug["parseable_outputs"]
                and aug["box_and_goal_parseable"] > wide["box_and_goal_parseable"]
                and rep_wide["completed_boards"] and rep_aug["completed_boards"]
                and rep_aug["single_box_at_outputs"] / rep_aug["completed_boards"]
                < rep_wide["single_box_at_outputs"] / rep_wide["completed_boards"]):
            lines.append("Here the reduced single-box share accompanies more conjunctions, while `box_at` "
                         "remains present in every parseable augmented output.")
        lines.append(f"The modal-coordinate count was {rep_wide['most_common_box_coordinate_outputs']}/"
                     f"{rep_wide['completed_boards']} → {rep_aug['most_common_box_coordinate_outputs']}/"
                     f"{rep_aug['completed_boards']}; valid intents were {wide['valid_outputs']}/"
                     f"{wide['requested_outputs']} → {aug['valid_outputs']}/{aug['requested_outputs']}. "
                     f"The paired valid-intent difference (wide+D4 minus wide) has board-bootstrap 95% "
                     f"interval {interval}, based on {pair['paired_boards']}/{pair['planned_boards']} pairs.")
        lines.append("")
    lines += ["An initially satisfied requested predicate makes the intent non-triviality check fail; "
              "it does not by itself show the remaining predicates are unreachable. The mode is selected "
              "after observing outputs. These results use one training seed and an exploratory previously "
              "inspected cohort, so they do not establish a general advantage or disadvantage.", ""]
    return lines


def report_text(data: Record, evaluation: Record, contract: Record, summary: list[Record],
                pairs: list[Record], equivariance: list[Record], repetitions: list[Record],
                modes: list[Record], training: list[Record], commands: list[Record],
                amendment: Record, recovery: Record, revision: Record, stop_amendment: Record | None,
                seed_decision: Record, supplementary: Record, figure: list[Record],
                family_repetitions: list[Record], family_pairs: list[Record], examples: list[Record],
                coverage: list[Record], resources: Record,
                gpu_benchmark: Record, cpu_benchmark: Record, elapsed_seconds: float) -> str:
    lines = ["# Stage 4 — layout diversity and coordinate augmentation", "",
        "This local pilot compares fixed Qwen3-0.6B generators. A valid non-trivial intent requires each "
        "requested effect to be initially false; any preservation conditions must initially hold. "
        "Syntax validity and task usefulness are different questions. No controller, "
        "complete-task planning, online model calls or paid API is measured.", "",
        "All test outputs use the same strict JSON parser and at most one complete outer Markdown fence "
        "is removed for every method. Invalid JSON and semantically invalid intents remain in the requested "
        "output denominator. Missing infrastructure outputs are shown separately and are not behavioural failures.", "",
        "The bounded BFS check uses 16 primitive actions and 2500 expanded nodes, with a 5 s safety timeout. "
        "A path found by BFS is a solver result, not CLLP or Luna success. No path within the limit is unresolved; "
        "easier requested intents can increase the apparent success rate.", "",
        "![Intent validity and bounded BFS paths across methods and orientations](stage_04_figure.png)", "",
        "Whiskers are 95% original-board bootstrap intervals. BFS paths are solver checks, not policy success.", "",
        "The 24 ×4 and 24 ×10 original boards are the previously inspected stage-three evaluation cohort; "
        "their D4 transformations are not independent boards. The 10-box family mixes gamma2 and reverse-generated "
        "sources while the 4-box family is RL validation, so any family difference is source-confounded. "
        "Pretraining exposure is unknown, and these exploratory results do not establish zero-shot transfer.", "",
        f"Additional candidate layouts were available after the primary cohort: "
        f"{supplementary['available_after_primary']['x4']} ×4, "
        f"{supplementary['available_after_primary']['x10_gamma2']} gamma2 ×10 and "
        f"{supplementary['available_after_primary']['x10_reverse']} reverse ×10. The retained 24+24 "
        "cohort was a compute/deadline choice, not a claim that further boards were unavailable. "
        "The existing evaluation path requires three orientations and cached stage-three references; "
        "an expanded block would require additional generations and risk the reporting reserve. "
        f"The supplementary-cohort decision was recorded at `{supplementary['created_utc']}` without "
        "inspecting new test outcomes; see `stage_04/supplementary_cohort_decision.json`.", "",
        "## Training data and provenance", "",
        f"Narrow uses {data['training_examples']['narrow']} checked examples from "
        f"{data['training_layouts']['narrow']} layouts; wide and wide+D4 each use "
        f"the same {data['training_examples']['wide']} original examples from "
        f"{data['training_layouts']['wide']} layouts. Narrow and wide share only "
        f"{data['common_original_examples']} original record IDs, so an A/B difference cannot be attributed "
        "solely to layout count. All adaptation examples are ×4; validation uses "
        f"{data['validation_examples']} checked examples from {data['validation_layouts']} separate layouts.", "",
        "Split assignment was by canonical layout before fragment extraction. Fragments use exact trajectory "
        "differences; future observations and recorded actions are checking evidence, not generator inputs. "
        f"The manifest's permitted inputs are `{data['input_fields']}`. Original upstream split: "
        f"{data['original_split']}. {data['x10_exposure']}", "",
        "The predicate and conjunction-size quotas match; offsets, goal thresholds and coordinates do not. "
        "Total variation below compares the normalised observed marginals; the raw counts are provided to "
        "make the remaining imbalance visible.", "",
        "| Marginal | Narrow counts | Wide counts | Total variation |",
        "|---|---|---|---:|"]
    provenance_index = lines.index("## Training data and provenance")
    lines[provenance_index:provenance_index] = (overview_lines(figure, family_repetitions, family_pairs)
                                               + descriptive_findings_lines(coverage, family_repetitions, family_pairs))
    for key, value in data["marginals"].items():
        if key == "coordinate_counts":
            a, b = value["narrow"], value["wide"]
            narrow = f"{len(a)} coordinates; top {Counter(a).most_common(5)}"
            wide = f"{len(b)} coordinates; top {Counter(b).most_common(5)}"
        else:
            narrow, wide = str(value["narrow"]), str(value["wide"])
        tv = value["total_variation"]
        lines.append(f"| {key} | `{narrow}` | `{wide}` | {tv:.3f} |" if tv is not None else
                     f"| {key} | `{narrow}` | `{wide}` | n/a |")
    lines += ["", f"Dataset manifest SHA256: `{data['manifest_sha256']}`; source dataset SHA256: "
              f"`{data['source_sha256']}`. Full original and combined JSONL hashes are in the dataset manifest.", ""]
    if revision["kind"] == "balanced_narrow_v3":
        decision, realized = revision["decision"], revision["realized"]
        lines += [
            "Before any new evaluation output was generated or inspected, an audit of the original narrow "
            "training logs found 97 walking-only draws among 600, versus 54/600 for wide, despite matching "
            "dataset quotas. The team therefore retained those nine layouts and replaced only the narrow "
            "examples, distributing 28–29 checked fragments per layout and retraining that arm once. "
            "The original narrow adapter and dataset remain saved; wide, wide+D4 and validation bytes did "
            "not change. This adjustment used training trajectories and sampling logs only; it does not "
            "resolve the different examples and remaining marginal imbalances.", "",
            f"The correction decision was recorded at `{decision['created_utc']}` and its completed data "
            f"audit at `{realized['created_utc']}`. The `created_utc` in the revised training freeze was "
            "inherited from the original plan and is not the time of this correction. Decision, realised "
            "hashes, preflights and the independent data audit are retained beside the results.", ""]
    else:
        lines += ["The original narrow dataset was used. The planned train-only replacement did not supply "
                  "a completed selected adapter for this assessment; the original 97/600 versus 54/600 "
                  "walking-only sampling mismatch must be considered in narrow-versus-wide comparisons.", ""]
    if stop_amendment:
        lines += [
            f"After evaluation started, at `{stop_amendment['amended_utc']}`, the supervisor's watchdog "
            f"cutoff was changed from `{stop_amendment['original_watchdog_utc']}` "
            f"(epoch `{stop_amendment['original_watchdog_epoch']}`, monotonic "
            f"`{stop_amendment['original_watchdog_monotonic']}`) to `{stop_amendment['revised_watchdog_utc']}` "
            f"(epoch `{stop_amendment['revised_watchdog_epoch']}`, monotonic "
            f"`{stop_amendment['revised_watchdog_monotonic']}`): {stop_amendment['reason']}. "
            f"The already-running child's own unchanged CLI cutoff was later, `{stop_amendment['child_cutoff_utc']}` "
            f"(epoch `{stop_amendment['child_cutoff_epoch']}`, monotonic "
            f"`{stop_amendment['child_cutoff_monotonic']}`), so the earlier watchdog deadline governs. "
            "The complete supervisor history and rationale are retained in the supervision manifest and "
            "decisions log.", ""]
    lines += [
              "The first data preparation was superseded by data_v2. An independent-audit JSON was overwritten "
              "after the training freeze by an expanded audit; the earlier bytes were subsequently reconstructed "
              "exactly, with the originally frozen SHA256. Both versions and the chronology are retained in "
              "`stage_04/training_freeze_audit_recovery.json`. This was separate from the later, explicitly "
              "recorded narrow-data correction.",
              f"Frozen audit SHA256: `{amendment['frozen_audit_sha256']}`; expanded audit SHA256: "
              f"`{amendment['current_audit_sha256']}`; reconstructed SHA256: "
              f"`{recovery['reconstructed_audit_sha256']}`.", "",
              "The data preparation source was formatted after the executed snapshot: its byte hash changed, "
              "but an independent AST comparison found equivalent code. Reproduction uses the current module "
              "from the repository, not the snapshot copied under analysis.", "",
              "## Fitting and model selection", "",
              f"Hardware: WSL2 on an Intel Core i9-12900HK with "
              f"{resources['logical_affinity_count']} logical CPUs available and "
              f"{resources['memory_total_bytes'] / 10**9:.1f} GB "
              f"({resources['memory_total_bytes'] / 2**30:.1f} GiB) RAM; "
              f"{resources['gpu_observed'].split(';')[0]}. GPU was used for LoRA fitting, "
              "and CPU for generator inference and bounded BFS.", "",
              f"The matched discarded BF16 GPU benchmark used a complete forward/backward/optimizer step: "
              f"{gpu_benchmark['complete_step_seconds']:.3f} s per update, "
              f"{gpu_benchmark['gpu_peak_allocated_bytes'] / 2**30:.2f} GiB peak allocated and "
              f"{gpu_benchmark['gpu_peak_reserved_bytes'] / 2**30:.2f} GiB peak reserved GPU memory. "
              f"The complete training examples and all eight symmetries fitted within "
              f"{gpu_benchmark['token_lengths']['max']} tokens without truncation.", "",
              f"The CPU development timing measured {cpu_benchmark['arms']['wide']['outputs']} outputs each "
              f"for wide ({cpu_benchmark['arms']['wide']['seconds_per_output']:.2f} s/output) and "
              f"wide+D4 ({cpu_benchmark['arms']['wide_aug']['seconds_per_output']:.2f} s/output) while "
              "GPU training was active. Its process exited 1 only at the final print because `flush` was "
              "passed to JSON encoding; 24/24 outputs and a complete timing file had already been saved "
              "and hash-checked. The print was corrected without repeating the measurement. "
              f"Before test inference, the time-based decision was `{seed_decision['decision']}`: "
              f"{seed_decision['reason']}", "",
              "All completed new arms use one shared seed, base revision, rank, optimizer, update count, "
              "padded model-token budget and held-out validation selection. Wide and wide+D4 use the same "
              "underlying example sequence; only valid sampled rotations/reflections differ. Actual non-padding "
              "and response-token counts may differ and are shown below. A single training seed limits any "
        "claim about variability across retraining. Although the total training budget is matched, "
        "the validation-selected checkpoints can come from different update numbers.", "",
              "| Arm | Seed | State | Updates | Validation-selected step | Validation loss | First → last training loss | Padded model tokens | Effective tokens | Supervised tokens |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in training:
        best = row.get("best") or {}
        loss = (f"{row['first_train_loss']:.4f} → {row['last_train_loss']:.4f}"
                if row.get("first_train_loss") is not None else "n/a")
        val = f"{best['validation_loss']:.4f}" if best.get("validation_loss") is not None else "n/a"
        lines.append(f"| {row['arm']} | {row.get('seed') if row.get('seed') is not None else 'n/a'} | "
                     f"{row['state']} | {row['updates']} | {best.get('step', 'n/a')} | {val} | {loss} | "
                     f"{row.get('model_tokens', 0)} | {row.get('effective_tokens', 0)} | {row.get('supervised_tokens', 0)} |")
    if any(row.get("updates") for row in training):
        lines += ["", "![Unsmoothed loss and validation curves](stage_04_training.png)", "",
                  "Stars mark the validation-selected checkpoints; no curves are smoothed.", ""]
    for row in training:
        if row.get("symmetries"):
            lines.append(f"- `{row['arm']}` actual sampled training symmetries: `{row['symmetries']}`.")
    lines += ["", "The sampler shuffles all layouts and visits each once per cycle; a layout's example "
              "pool is reshuffled when exhausted, so examples can recur. The realised category counts "
              "below need not match the quotas of the full 256-record datasets. Coordinates "
              "are counted after the sampled transform.", "",
              "| Arm | Sampled examples | Categories | Fragment offsets | Goal thresholds | Five most common coordinates |",
              "|---|---:|---|---|---|---|"]
    for row in training:
        if row.get("sampled_examples") is None:
            lines.append(f"| {row['arm']} | 0 | n/a | n/a | n/a | n/a |")
            continue
        top = Counter(row["sampled_coordinates"]).most_common(5)
        lines.append(f"| {row['arm']} | {row['sampled_examples']} | `{row['sampled_effect_types']}` | "
                     f"`{row['sampled_offsets']}` | `{row['sampled_goal_thresholds']}` | `{top}` |")
    lines += ["", f"Evaluation state: `{evaluation.get('state')}`; {evaluation.get('completed', 0)}/"
              f"{evaluation.get('planned', 0)} planned outputs recorded, including "
              f"{evaluation.get('baseline_reused', 0)} exact cached base/historical responses. "
              f"The fixed model revision is `{contract['model_revision']}`; new evaluation runs on CPU "
              "with FP32, four threads, batch size 3 and greedy decoding. The historical LoRA is separately "
              "labelled because its training data and schedule are not matched to the new arms.", "",
              "## Generation and bounded feasibility", "",
              "| Family | Source | Method | Orientation | Completed / requested | JSON valid / requested | "
              "Valid intent / requested | BFS paths / requested | No path / valid | Already satisfied | Goal requests | Solver errors |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        if row["condition"] not in ("original", "d4"):
            continue
        requested, completed = int(row["requested_outputs"]), int(row["completed_outputs"])
        valid, paths = int(row["valid_nontrivial"]), int(row["bounded_paths_found"])
        errors = int(row["bfs_infrastructure_errors"])
        lines.append(f"| {row['family']} | {row['source_family']} | {row['method']} | {row['transform']} | "
                     f"{completed}/{requested} | {display_ratio(row['normalized_json_valid'], requested)} | "
                     f"{display_ratio(valid, requested)} | {display_ratio(paths, requested)} | "
                     f"{display_ratio(valid - paths - errors, valid)} | {row['any_effect_initially_true']} | "
                     f"{row['whole_goal_requests']} | {errors} |")
    lines += ["", "When output is missing, requested-output rates are lower bounds; consult the completed "
              "counts rather than treating missing infrastructure outputs as model failures.", "",
              "## Paired differences on original boards", "",
              "Wide minus narrow tests the changed training set, while wide+D4 minus wide tests the added "
              "coordinate transformations. Intervals bootstrap original layouts, within each source and "
              "orientation; repeated orientations are not independent boards. Small homogeneous samples can "
              "produce degenerate empirical intervals.", "",
              "| Family | Source | Orientation | Comparison | Metric | Paired / planned boards | Difference | 95% interval |",
              "|---|---|---|---|---|---:|---:|---:|"]
    for row in pairs:
        value, low, high = row["difference_second_minus_first"], row["ci_low"], row["ci_high"]
        difference = f"{value:+.3f}" if value is not None else "n/a"
        interval = f"[{low:+.3f}, {high:+.3f}]" if low is not None and high is not None else "n/a"
        lines.append(f"| {row['family']} | {row['source_family']} | {row['transform']} | "
                     f"{row['second']} − {row['first']} | {row['metric']} | "
                     f"{row['paired_boards']}/{row['planned_boards']} | {difference} | {interval} |")
    lines += ["", "## Diversity and coordinate repetition", "",
              "Modes and coordinates below include parseable but semantically invalid responses; this "
              "avoids hiding repeated coordinates that happen to hit walls. All counts are original orientations.", "",
              "![Single-box intents and most common box coordinates](stage_04_repetition.png)", "",
              "Each original board contributes at most one completed output. The bars show counts divided "
              "by completed outputs; completed/planned boards and the modal coordinate appear beneath "
              "each method. The modal-coordinate bar includes any parsed `box_at`, including conjunctions; "
              "the coordinate is selected after seeing the outputs, so it is a descriptive mode rather "
              "than a pre-specified hypothesis test. The ×10 panel pools gamma2 and reverse sources, which "
              "remain separate in the table below and are a source-distribution confound. The figure CSV "
              "records planned and completed source counts separately.", "",
              "| Family | Source | Method | Single box_at / completed | Most repeated box coordinate | "
              "Coordinate outputs / completed | Most common structured intent / completed |",
              "|---|---|---|---:|---|---:|---:|"]
    by_key = {(row["family"], row["source_family"], row["method"]): row for row in modes}
    for row in repetitions:
        mode = by_key[(row["family"], row["source_family"], row["method"])]
        lines.append(f"| {row['family']} | {row['source_family']} | {row['method']} | "
                     f"{display_ratio(row['single_box_at_outputs'], row['completed_outputs'])} | "
                     f"{row['most_common_box_coordinate'] or 'n/a'} | "
                     f"{display_ratio(row['most_common_box_coordinate_outputs'], row['completed_outputs'])} | "
                     f"{display_ratio(mode['most_common_count'], mode['completed_outputs'])} |")
    lines += ["", "| Family | Source | Method | Valid original intents | Predicate counts | Conjunction sizes |",
              "|---|---|---|---:|---|---|"]
    for row in summary:
        if row["condition"] == "original" and row["transform"] == "identity":
            lines.append(f"| {row['family']} | {row['source_family']} | {row['method']} | {row['valid_nontrivial']} | "
                         f"`{row['effect_counts_valid']}` | `{row['conjunction_sizes_valid']}` |")
    lines += ["", "## Agreement after inverse D4 mapping", "",
              "Exact structured-intent equality is counted only when both outputs are valid. Multiple "
              "different intents may be equally reasonable, so disagreement does not prove a lack of "
              "board understanding. Natural-language instruction strings are not transformed.", "",
              "| Family | Source | Method | Orientation | Valid pairs / planned | Identical / valid pairs |",
              "|---|---|---|---|---:|---:|"]
    for row in equivariance:
        lines.append(f"| {row['family']} | {row['source_family']} | {row['method']} | {row['transform']} | "
                     f"{row['valid_both']}/{row['planned_contexts']} | "
                     f"{display_ratio(row['same_structured_intent'], row['valid_both'])} |")
    lines += [""] + example_lines(examples)
    lines += ["Full per-output JSONL, prompts, source-separated bootstrap tables, training curves, model "
              "configurations and adapter checkpoints are retained under `stage_04/`. The figure input is "
              "`stage_04_figure_data.csv`; training curves are in `stage_04_training_curve.csv`.", "",
              f"Elapsed stage time at report generation: {elapsed_seconds:.1f} s. The campaign deadline "
              "was not reset. These results do not establish improved complete-task planning or a policy-relative scale effect.", "",
              "## Exact recorded commands", "",
              "The commands below are the actual supervised child argv. Their statuses are a snapshot "
              "taken while this report is generated; final process states are in the supervisor's "
              "`run_manifest.json`. Existing output directories and "
              "absolute clock cutoffs prevent blindly replaying them; use new output paths and a separately "
              "supervised deadline for any new run. Reanalysis can instead reuse the retained JSONL outputs."]
    for command in commands:
        lines += ["", f"**{command['name']}** — {command['status']}", "", "```sh", command["command"], "```"]
    return "\n".join(lines) + "\n"


def parse_arm_dir(value: str) -> tuple[str, Path]:
    arm, separator, path = value.partition("=")
    if not separator or arm not in ("narrow", "wide", "wide_aug") or not path:
        raise ValueError("Use --arm-dir narrow=DIR, wide=DIR or wide_aug=DIR")
    return arm, Path(path).resolve()


def verify_training_artifacts(arm_dirs: dict[str, Path], contract: Record) -> None:
    """Bind evaluated adapters and their sampled training traces to the report."""
    for arm, metadata in contract["adapters"].items():
        directory = arm_dirs.get(arm)
        if directory is None or str(directory) != metadata["arm_directory"]:
            raise ValueError(f"Training directory for evaluated arm {arm} is missing or changed")
        checkpoint = Path(metadata["checkpoint"])
        paths = {"best_sha256": directory / "best.json", "configuration_sha256": directory / "configuration.json",
                 "status_sha256": directory / "status.json", "training_curve_sha256": directory / "training_curve.jsonl",
                 "checkpoint_weights_sha256": checkpoint / "adapter_model.safetensors",
                 "checkpoint_config_sha256": checkpoint / "adapter_config.json"}
        for key, path in paths.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata[key]:
                raise ValueError(f"Training artifact for {arm} differs from evaluated contract: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--assessment", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--arm-dir", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign, stage = args.campaign.resolve(), args.campaign.resolve() / "stage_04"
    output = (args.output or campaign).resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation = read_json(args.evaluation / "status.json")
    contract = read_json(args.evaluation / "contract.json")
    if evaluation.get("state") not in ("complete", "partial", "failed", "stopped"):
        raise ValueError("Generator must stop before final assessment")
    assessment, rows, summary, _, equivariance = verified_assessment(args.assessment)
    key = str((args.evaluation / "predictions.jsonl").resolve())
    if assessment["inputs_sha256"].get(key) != hashlib.sha256((args.evaluation / "predictions.jsonl").read_bytes()).hexdigest():
        raise ValueError("Assessment does not match the stopped generator outputs")
    data = dataset_diagnostics(args.data)
    if data["manifest_sha256"] != contract["data_manifest_sha256"]:
        raise ValueError("Evaluation uses a different dataset manifest")
    freeze = read_json(stage / "training_freeze.json")
    revision = verify_dataset_revision(stage, args.data)
    amendment = read_json(stage / "training_freeze_audit_amendment.json")
    recovery = read_json(stage / "training_freeze_audit_recovery.json")
    seed_path = stage / "seed_decision.json"
    supplementary_path = stage / "supplementary_cohort_decision.json"
    cpu_path = stage / "benchmark_cpu_dev.json"
    gpu_path = stage / "benchmark_matched.json"
    seed_decision, cpu_benchmark, gpu_benchmark = (read_json(path) for path in (seed_path, cpu_path, gpu_path))
    supplementary = read_json(supplementary_path)
    if (supplementary["test_outcomes_inspected"] is not False
            or supplementary["no_training_or_test_decision_based_on_outcomes"] is not True):
        raise ValueError("Supplementary cohort decision was not recorded as outcome-independent")
    if (hashlib.sha256(cpu_path.read_bytes()).hexdigest() != seed_decision["benchmark_sha256"]
            or cpu_benchmark["state"] != "complete" or gpu_benchmark["status"] != "complete"
            or hashlib.sha256((stage / "benchmark_cpu_dev_predictions.jsonl").read_bytes()).hexdigest()
            != cpu_benchmark["predictions_sha256"]
            or hashlib.sha256(Path(seed_decision["benchmark_source_snapshot"]).read_bytes()).hexdigest()
            != cpu_benchmark["script_sha256"]
            or hashlib.sha256(gpu_path.read_bytes()).hexdigest() != freeze["data_sha256"]["matched_benchmark"]):
        raise ValueError("Recorded resource benchmark or seed decision changed")
    if (recovery["reconstructed_audit_sha256"] != amendment["frozen_audit_sha256"]
            or amendment["dataset_manifest_sha256"] != revision["original_manifest_sha256"]):
        raise ValueError("Audit provenance recovery does not match the frozen dataset")
    if hashlib.sha256(Path(recovery["reconstructed_audit"]).read_bytes()).hexdigest() != recovery["reconstructed_audit_sha256"]:
        raise ValueError("Reconstructed frozen audit bytes differ")
    arm_dirs = dict(parse_arm_dir(value) for value in args.arm_dir)
    if len(arm_dirs) != len(args.arm_dir):
        raise ValueError("Duplicate training arm directory")
    verify_training_artifacts(arm_dirs, contract)
    curves, training = training_curves(arm_dirs, args.data)
    pairs = paired_differences(rows)
    family_pairs = family_paired_differences(rows)
    figure = figure_data(rows)
    modes = mode_diagnostics(rows)
    repetition = repetition_diagnostics(rows)
    repetition_family = family_repetition(rows)
    coverage = predicate_coverage(rows)
    plan = read_json(args.evaluation / "plan.json")
    examples = representative_examples(plan, rows)
    supervisor = read_json(campaign / "stage_04_supervision/run_manifest.json")
    resources = read_json(campaign / "campaign_manifest.json")["resources"]
    commands = recorded_commands(supervisor)
    stop_amendment = evaluation_stop_amendment(supervisor)
    clock = read_json(campaign / "stage_04_clock.json")
    import time

    elapsed = max(0.0, time.monotonic() - clock["start_monotonic"])
    for name, records in (("stage_04_figure_data.csv", figure), ("stage_04_paired.csv", pairs),
                          ("stage_04_paired_family.csv", family_pairs),
                          ("stage_04_mode.csv", modes), ("stage_04_repetition.csv", repetition),
                          ("stage_04_repetition_family.csv", repetition_family),
                          ("stage_04_predicate_coverage.csv", coverage),
                          ("stage_04_training_curve.csv", curves)):
        write_csv(output / name, records)
    plot_summary(figure, output)
    plot_repetition(repetition_family, output)
    plot_training(curves, training, output)
    atomic_text(output / "stage_04_resource_snapshot.json", json.dumps({"resources": resources,
        "campaign_manifest_sha256_at_report": sha256(campaign / "campaign_manifest.json")},
        indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    atomic_text(output / "stage_04_examples.json", json.dumps(examples, indent=2, ensure_ascii=False,
                                                                allow_nan=False) + "\n")
    text = report_text(data, evaluation, contract, summary, pairs, equivariance, repetition, modes,
                       training, commands, amendment, recovery, revision, stop_amendment, seed_decision,
                       supplementary, figure, repetition_family, family_pairs, examples,
                       coverage, resources, gpu_benchmark, cpu_benchmark, elapsed)
    atomic_text(output / "stage_04_report.md", text)
    summaries = {"dataset": data, "dataset_revision": revision, "supplementary_cohort_decision": supplementary,
                 "evaluation": evaluation,
                 "watchdog_amendment": stop_amendment, "training": training, "paired": pairs,
                 "paired_family": family_pairs,
                 "figure": figure, "repetition": repetition, "repetition_family": repetition_family,
                 "predicate_coverage": coverage, "mode": modes,
                 "assessment_manifest": assessment, "elapsed_seconds": elapsed}
    atomic_text(output / "stage_04_summary.json", json.dumps(summaries, indent=2, ensure_ascii=False,
                                                               allow_nan=False) + "\n")
    outputs = ["stage_04_report.md", "stage_04_summary.json", "stage_04_figure_data.csv", "stage_04_paired.csv",
               "stage_04_paired_family.csv", "stage_04_examples.json", "stage_04_resource_snapshot.json",
               "stage_04_mode.csv", "stage_04_repetition.csv", "stage_04_repetition_family.csv",
               "stage_04_predicate_coverage.csv",
               "stage_04_training_curve.csv", *[f"stage_04_figure.{ext}" for ext in ("pdf", "svg", "png")],
               *[f"stage_04_repetition.{ext}" for ext in ("pdf", "svg", "png")]]
    if curves:
        outputs.extend(f"stage_04_training.{ext}" for ext in ("pdf", "svg", "png"))
    revision_paths = [Path(path) for path in revision["paths"]]
    manifest = {"schema_version": 1, "outputs_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                                                          for name in outputs},
                "inputs_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                  [args.data / "dataset_manifest.json", args.assessment / "assessment_manifest.json",
                                   args.evaluation / "contract.json", args.evaluation / "status.json",
                                   stage / "training_freeze.json", stage / "training_freeze_audit_recovery.json",
                                   seed_path, supplementary_path, cpu_path, gpu_path, *revision_paths]},
                "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "command": os.sys.argv, "recorded_commands": commands}
    atomic_text(output / "stage_04_report_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(output / "stage_04_report.md"), "planned": evaluation["planned"],
                      "completed": evaluation["completed"], "pairs": len(pairs)}))


if __name__ == "__main__":
    main()
