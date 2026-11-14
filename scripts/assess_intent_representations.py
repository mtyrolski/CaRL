#!/usr/bin/env python3
"""Assess alternative intent serialisations on the same frozen boards.

Formatting is scored using the prespecified parser for each representation;
semantic checks see only the starting board. Optional bounded BFS searches for
a primitive-action sequence and is not an LLM or learned-controller rollout.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import matplotlib
from matplotlib.lines import Line2D

from carl.icl.intent_assessment import board_means
from carl.icl.intent_assessment import bootstrap
from carl.icl.intent_assessment import bounded_witness
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_assessment import read_jsonl
from carl.icl.intent_assessment import test_contexts
from carl.icl.intent_data import validate_prediction
from carl.icl.intent_fence_audit import remove_one_outer_fence
from carl.icl.intent_generator_experiment import parse_json
from scripts.assess_lora_matrix import intent_signature
from scripts.assess_lora_matrix import parse_partial
from scripts.assess_lora_matrix import whole_goal

matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402

Record = dict[str, Any]
Parser = Callable[[str, str], Record]
RNG_SEED = 20261114
ROOT = Path(__file__).resolve().parents[1]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def variant_paths(specifications: list[str]) -> list[tuple[str, Path]]:
    variants: list[tuple[str, Path]] = []
    for item in specifications:
        name, separator, path = item.partition("=")
        if (not separator or not name or Path(name).name != name or name in {".", ".."} or not path
                or any(name == known for known, _ in variants)):
            raise ValueError("Each --variant must give a distinct NAME=PATH")
        variants.append((name, Path(path)))
    return variants


def snapshot_prediction(path: Path, variant: str, destination: Path) -> Record:
    """Capture immutable bytes once, including metadata and complete prompts."""
    if not variant or Path(variant).name != variant or variant in {".", ".."}:
        raise ValueError("Invalid variant name for an input snapshot")
    snapshot = destination / f"{variant}.jsonl"
    files = ((path, snapshot), (path.with_suffix(".run.json"), snapshot.with_suffix(".run.json")),
             (path.with_suffix(".prompts.jsonl"), snapshot.with_suffix(".prompts.jsonl")))
    captured = {}
    for source, target in files:
        if source.exists():
            data = source.read_bytes()
            atomic_bytes(target, data)
            captured[target.name] = {"source": str(source), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        else:
            captured[target.name] = {"source": str(source), "sha256": None, "bytes": None}
    return {"variant": variant, "files": captured}


def read_predictions(path: Path, variant: str, contexts: list[Record], data: Path, ids: Path) -> tuple[dict[str, Record], Record]:
    """Snapshot complete lines and reject duplicate, foreign or mislabelled rows."""
    expected = {row["record_id"]: row["layout_id"] for row in contexts}
    data_bytes = path.read_bytes() if path.exists() else None
    rows, partial = parse_partial(data_bytes) if data_bytes is not None else ([], False)
    result: dict[str, Record] = {}
    for row in rows:
        identifier = row.get("record_id")
        if (not isinstance(identifier, str) or identifier not in expected or identifier in result
                or row.get("layout_id") != expected[identifier] or row.get("variant") != variant
                or row.get("split") != "test"):
            raise ValueError(f"Prediction does not match frozen context/variant: {identifier}")
        if not isinstance(row.get("representation"), str) or not isinstance(row.get("support_set"), str):
            raise ValueError(f"Missing representation or support-set provenance: {identifier}")
        result[identifier] = row
    provenance = {(row["representation"], row["support_set"]) for row in result.values()}
    if len(provenance) > 1:
        raise ValueError(f"Variant mixes different representations or support sets: {path}")
    manifest_path = path.with_suffix(".run.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if rows and manifest is None:
        raise ValueError(f"Prediction rows have no run metadata and cannot be scored: {path}")
    if manifest is not None:
        if manifest.get("data_sha256") != file_hash(data) or manifest.get("test_ids_sha256") != file_hash(ids):
            raise ValueError(f"Prediction metadata has a different dataset or test IDs: {path}")
        if manifest.get("variant") is not None and manifest["variant"] != variant:
            raise ValueError(f"Prediction metadata has a different variant: {path}")
        if provenance:
            representation, support = next(iter(provenance))
            if ((manifest.get("representation") is not None and manifest["representation"] != representation)
                    or (manifest.get("support_set") is not None and manifest["support_set"] != support)):
                raise ValueError(f"Prediction metadata conflicts with row provenance: {path}")
    info: Record = {"path": str(path), "sha256": hashlib.sha256(data_bytes).hexdigest() if data_bytes is not None else None,
                    "partial_final_line": partial, "metadata": manifest}
    return result, info


def validate_matrix_run(plan: Record, method: Record, manifest: Record, prediction_path: Path) -> None:
    """Fail closed if a saved prediction does not match the frozen matrix job."""
    inputs, model, settings = plan["inputs"], plan["model"], plan["settings"]
    expected: Record = {"variant": method["name"], "representation": method["representation"],
                        "support_set": method["support_set"], "model": str(Path(model["path"]).resolve()),
                        "revision": model["revision"], "model_weights_sha256": model["sha256"]["model.safetensors"],
                        "model_config_sha256": model["sha256"]["config.json"],
                        "tokenizer_sha256": model["sha256"]["tokenizer.json"],
                        "data_sha256": inputs["data"]["sha256"], "test_ids_sha256": inputs["test_ids"]["sha256"],
                        "support_data_sha256": inputs["support_data"]["sha256"],
                        "support_manifest_sha256": inputs["support_manifest"]["sha256"],
                        "protocol_sha256": inputs["protocol"]["sha256"],
                        "generator_sha256": inputs["generator"]["sha256"],
                        "support_selection_sha256": inputs["support_selector"]["sha256"], **settings}
    adapter = method.get("adapter")
    expected["adapter"] = str(Path(adapter).resolve()) if adapter is not None else None
    if adapter is None:
        hashes = None
    elif method.get("adapter_pending"):
        frozen_path = Path(plan["run"]) / "matrix_adapter_hashes" / f"{method['name']}.json"
        if not frozen_path.exists():
            raise ValueError(f"Missing frozen adapter hash for {method['name']}")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if str(Path(frozen.get("adapter", "")).resolve()) != expected["adapter"]:
            raise ValueError(f"Frozen adapter path differs for {method['name']}")
        hashes = frozen.get("sha256")
    else:
        hashes = method.get("adapter_sha256")
    if adapter is not None and (not isinstance(hashes, dict)
                                or set(hashes) != {"adapter_model.safetensors", "adapter_config.json"}):
        raise ValueError(f"Incomplete adapter hashes for {method['name']}")
    expected["adapter_weights_sha256"] = hashes["adapter_model.safetensors"] if hashes else None
    expected["adapter_config_sha256"] = hashes["adapter_config.json"] if hashes else None
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Matrix method {method['name']} differs in {key}")
    prompts = prediction_path.with_suffix(".prompts.jsonl")
    if not prompts.exists() or manifest.get("prompts_sha256") != file_hash(prompts):
        raise ValueError(f"Saved prompts differ from generation metadata: {method['name']}")


def board_family(record: Record) -> str:
    board = record["start_observation"]
    count = sum(row.count("$") + row.count("*") for row in board)
    return str(record.get("evaluation_family") or record.get("family") or f"{len(board)}x{len(board[0])}x{count}")


def exposure_group(record: Record) -> str:
    value = record.get("previously_evaluated_in_lora_matrix")
    if value is True:
        return "in_previous_lora_matrix"
    if value is False:
        return "not_in_previous_lora_matrix"
    return "unknown"


def assess_one(record: Record, prediction: Record | None, parser: Parser, *, bfs: bool,
               max_depth: int, max_nodes: int, seconds: float) -> Record:
    board = record["start_observation"]
    family = board_family(record)
    result: Record = {"record_id": record["record_id"], "layout_id": record["layout_id"], "family": family,
                      "previous_lora_exposure": exposure_group(record),
                      "status": "missing" if prediction is None else "evaluated"}
    if prediction is None:
        return result
    if prediction.get("status") in ("infrastructure_error", "generation_error"):
        return {**result, "status": "infrastructure_error", "error": prediction.get("error")}
    raw, representation = prediction.get("raw_output"), prediction.get("representation")
    if not isinstance(raw, str) or not isinstance(representation, str):
        raise ValueError("An evaluated prediction needs raw output and representation")
    try:
        intent = parser(representation, raw)
        syntax_valid, syntax_error = True, None
    except ValueError as error:
        intent, syntax_valid, syntax_error = None, False, str(error)
    if prediction.get("syntax_valid") is not syntax_valid or prediction.get("parsed_intent") != intent:
        raise ValueError(f"Recorded parsing differs from re-parsing raw output: {record['record_id']}")
    raw_json_valid = None
    if representation == "json":
        try:
            parse_json(raw)
            raw_json_valid = True
        except ValueError:
            raw_json_valid = False
    semantic_errors = validate_prediction(intent, board) if syntax_valid else ["invalid representation"]
    valid = bool(syntax_valid and not semantic_errors)
    valid_intent = intent if valid else None
    effects = valid_intent["effects"] if valid_intent is not None else []
    asks_whole_goal = whole_goal(valid_intent, board) if valid_intent is not None else False
    normalised, fence_removed = remove_one_outer_fence(raw)
    try:
        fence_intent = parser(representation, normalised)
        fence_syntax_valid = True
    except ValueError:
        fence_intent, fence_syntax_valid = None, False
    fence_semantic_valid = bool(fence_syntax_valid and not validate_prediction(fence_intent, board))
    fence_whole_goal = bool(fence_intent is not None and fence_semantic_valid and whole_goal(fence_intent, board))
    result.update({"representation": representation, "support_set": prediction["support_set"],
                   "syntax_valid": syntax_valid, "syntax_error": syntax_error,
                   "raw_json_valid": raw_json_valid, "fence_removed": fence_removed,
                   "fence_syntax_valid": fence_syntax_valid, "fence_semantic_valid": fence_semantic_valid,
                   "fence_whole_goal": fence_whole_goal,
                   "fence_valid_not_whole_goal": fence_semantic_valid and not fence_whole_goal,
                   "semantic_valid": valid, "semantic_errors": semantic_errors,
                   "whole_goal": asks_whole_goal,
                   "valid_not_whole_goal": valid and not asks_whole_goal,
                   "whole_goal_only": bool(valid_intent is not None and len(effects) == 1
                                           and asks_whole_goal),
                   "effect_predicates": [item["predicate"] for item in effects],
                   "conjunction_size": len(effects),
                   "preserve_size": len(valid_intent["preserve"]) if valid_intent is not None else None,
                   "intent_signature": intent_signature(valid_intent) if valid_intent is not None else None,
                   "intent": intent, "raw_output": raw, "start_observation": board,
                   "generation_seconds": prediction.get("generation_seconds"),
                   "generated_tokens": prediction.get("generated_tokens"), "solver": None})
    if bfs and valid_intent is not None:
        try:
            result["solver"] = bounded_witness(board, valid_intent, max_depth, max_nodes, seconds)
        except (AssertionError, ValueError) as error:
            result["solver"] = {"status": "verification_error", "witness_found": False, "error": str(error)}
    return result


def measure(rows: list[Record], field: str) -> tuple[float | None, list[float] | None]:
    means = board_means(rows, lambda row: float(row[field]))
    return (sum(means) / len(means), bootstrap(means, random.Random(RNG_SEED))) if means else (None, None)


def summarize(variant: str, rows: list[Record], *, bfs: bool) -> Record:
    measured = [row for row in rows if row["status"] == "evaluated"]
    valid = [row for row in measured if row["semantic_valid"]]
    result: Record = {"variant": variant, "planned": len(rows), "outputs": len(measured),
                      "missing": sum(row["status"] == "missing" for row in rows),
                      "infrastructure_errors": sum(row["status"] in ("infrastructure_error", "rejected_infrastructure")
                                                   for row in rows),
                      "boards_planned": len({row["layout_id"] for row in rows}),
                      "boards_observed": len({row["layout_id"] for row in measured}),
                      "representations": sorted({row["representation"] for row in measured}),
                      "support_sets": sorted({row["support_set"] for row in measured}),
                      "syntax_valid": sum(row["syntax_valid"] for row in measured),
                      "raw_json_valid": (sum(bool(row["raw_json_valid"]) for row in measured)
                                         if measured and all(row["raw_json_valid"] is not None for row in measured)
                                         else None),
                      "fence_removed": sum(row["fence_removed"] for row in measured),
                      "fence_syntax_valid": sum(row["fence_syntax_valid"] for row in measured),
                      "fence_semantic_valid": sum(row["fence_semantic_valid"] for row in measured),
                      "fence_whole_goal": sum(row["fence_whole_goal"] for row in measured),
                      "fence_valid_not_whole_goal": sum(row["fence_valid_not_whole_goal"] for row in measured),
                      "semantic_valid": len(valid),
                      "whole_goal": sum(row["whole_goal"] for row in valid),
                      "whole_goal_only": sum(row["whole_goal_only"] for row in valid),
                      "valid_not_whole_goal": sum(row["valid_not_whole_goal"] for row in measured),
                      "distinct_intents": len({row["intent_signature"] for row in valid}),
                      "effect_predicates": dict(Counter(value for row in valid for value in row["effect_predicates"])),
                      "conjunction_sizes": dict(Counter(str(row["conjunction_size"]) for row in valid)),
                      "preserve_sizes": dict(Counter(str(row["preserve_size"]) for row in valid))}
    for field in ("syntax_valid", "semantic_valid", "whole_goal", "whole_goal_only", "valid_not_whole_goal",
                  "fence_syntax_valid", "fence_semantic_valid", "fence_whole_goal", "fence_valid_not_whole_goal"):
        result[field + "_board_mean"], result[field + "_95_board_bootstrap"] = measure(measured, field)
    for field in ("generation_seconds", "generated_tokens"):
        known = [row[field] for row in measured if isinstance(row.get(field), (int, float))
                 and not isinstance(row[field], bool)]
        result[field] = sum(known) if measured and len(known) == len(measured) else None
        result[field + "_known"] = len(known)
    if bfs:
        attempts = [row for row in valid if row["solver"] is not None]
        found = [row for row in attempts if row["solver"].get("witness_found")]
        errors = [row for row in attempts if row["solver"].get("status") == "verification_error"]
        result.update({"bfs_attempted": len(attempts), "bfs_found": len(found),
                       "bfs_unresolved": len(attempts) - len(found) - len(errors), "bfs_errors": len(errors),
                       "bfs_statuses": dict(Counter(row["solver"]["status"] for row in attempts))})
        pipeline = [{**row, "found": bool(row.get("solver") and row["solver"].get("witness_found"))}
                    for row in measured if row not in errors]
        result["bfs_pipeline_evaluated"] = len(pipeline)
        result["bfs_found_board_mean"], result["bfs_found_95_board_bootstrap"] = measure(pipeline, "found")
    else:
        result.update({"bfs_attempted": None, "bfs_found": None, "bfs_unresolved": None,
                       "bfs_errors": None, "bfs_pipeline_evaluated": None,
                       "bfs_found_board_mean": None, "bfs_found_95_board_bootstrap": None})
    return result


def paired(rows: list[Record], reference: str, candidate: str, field: str, family: str | None = None,
           exposure: str | None = None) -> Record:
    chosen = [row for row in rows if (family is None or row["family"] == family)
              and (exposure is None or row.get("previous_lora_exposure") == exposure)]
    indexed: dict[str, dict[str, Record]] = {reference: {}, candidate: {}}
    for row in chosen:
        variant = row["variant"]
        if variant not in indexed:
            continue
        identifier = row["record_id"]
        if identifier in indexed[variant]:
            raise ValueError("Duplicate context for a comparison variant")
        indexed[variant][identifier] = row
    left, right = indexed[reference], indexed[candidate]
    shared = left.keys() & right.keys()
    for identifier in shared:
        if left[identifier]["layout_id"] != right[identifier]["layout_id"]:
            raise ValueError("The same context refers to different layouts")
    complete = [identifier for identifier in sorted(shared)
                if left[identifier]["status"] == right[identifier]["status"] == "evaluated"]
    pairs = [{"layout_id": left[key]["layout_id"], "delta": int(right[key][field]) - int(left[key][field])}
             for key in complete]
    mean, bounds = measure(pairs, "delta")
    return {"family": family or "all", "previous_lora_exposure": exposure or "all",
            "reference": reference, "candidate": candidate, "metric": field,
            "planned_pairs": len(shared), "paired_contexts": len(complete),
            "paired_boards": len({row["layout_id"] for row in pairs}), "unpaired_contexts": len(shared) - len(complete),
            "reference_positive": sum(left[key][field] for key in complete),
            "candidate_positive": sum(right[key][field] for key in complete),
            "board_mean_delta": mean, "bootstrap_low": bounds[0] if bounds else None,
            "bootstrap_high": bounds[1] if bounds else None,
            "uncertainty": "paired board bootstrap, 2000 resamples" if bounds else "descriptive: fewer than five boards"}


def settings_audit(infos: list[tuple[str, Record]]) -> Record:
    fields = ("model", "revision", "seed", "decoding", "enable_thinking", "max_length", "max_new_tokens", "dtype",
              "model_weights_sha256", "tokenizer_sha256", "support_data_sha256", "support_manifest_sha256",
              "protocol_sha256", "generator_sha256", "support_selection_sha256")
    result = {}
    for field in fields:
        values = {name: (info.get("metadata") or {}).get(field) for name, info in infos}
        known = {json.dumps(value, sort_keys=True) for value in values.values() if value is not None}
        result[field] = {"values": values, "missing": [name for name, value in values.items() if value is None],
                         "mismatch": len(known) > 1}
    return result


def write_csv(path: Path, rows: list[Record], fields: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def plot_summary(output: Path, rows: list[Record], family: str) -> None:
    """Display raw-valid formats and nontrivial intents with board-level intervals."""
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * .75), 4.5), constrained_layout=True)
    for index, row in enumerate(rows):
        if not row["outputs"]:
            ax.text(index, .02, "brak", rotation=90, ha="center", fontsize=8)
            continue
        for offset, field, colour in ((-.24, "syntax_valid", "#8dacc1"), (0, "semantic_valid", "#315c7e"),
                                      (.24, "valid_not_whole_goal", "#c18356")):
            value = row[field + "_board_mean"]
            ax.bar(index + offset, value, width=.22, color=colour)
            interval = row[field + "_95_board_bootstrap"]
            if interval:
                ax.vlines(index + offset, *interval, color="black", linewidth=.8)
            ax.text(index + offset, max(value, interval[1] if interval else value) + .025,
                    f"{row[field]}/{row['outputs']}\n({row['planned']})", ha="center", fontsize=7)
    ax.set_xticks(range(len(rows)), [row["variant"] for row in rows], rotation=35, ha="right", fontsize=8)
    ax.set(ylim=(0, 1.23), ylabel="Średni odsetek na planszę", title=family)
    ax.legend(["Poprawny format i schemat", "Poprawna nietrywialna intencja", "Poprawna, bez pełnego celu"], fontsize=8)
    ax.grid(axis="y", alpha=.2)
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in family)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"validity_{safe}.{extension}", dpi=180)
    plt.close(fig)


def plot_bfs(output: Path, rows: list[Record], family: str) -> None:
    """Show an end-to-end fraction, with explicit search-only denominators."""
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(max(7, len(rows) * .7), 4.5), constrained_layout=True)
    for index, row in enumerate(rows):
        value, interval = row["bfs_found_board_mean"], row["bfs_found_95_board_bootstrap"]
        if value is None:
            ax.text(index, .02, "brak", rotation=90, ha="center", fontsize=8)
            continue
        ax.bar(index, value, width=.55, color="#766090")
        if interval:
            ax.vlines(index, *interval, color="black", linewidth=.8)
        ax.text(index, max(value, interval[1] if interval else value) + .025,
                f"{row['bfs_found']}/{row['bfs_pipeline_evaluated']}\nBFS {row['bfs_found']}/{row['bfs_attempted']}",
                ha="center", fontsize=7)
    ax.set_xticks(range(len(rows)), [row["variant"] for row in rows], rotation=35, ha="right", fontsize=8)
    ax.set(ylim=(0, 1.23), ylabel="Średni odsetek na planszę", title=f"{family}: ograniczone BFS, nie kontroler")
    fig.supxlabel("Mianownik słupka obejmuje ocenione odpowiedzi; BFS sprawdzono tylko na poprawnych intencjach.",
                  fontsize=8)
    ax.grid(axis="y", alpha=.2)
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in family)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"bounded_bfs_{safe}.{extension}", dpi=180)
    plt.close(fig)


CORE_GROUPS = {
    "Bez przykładów": ["base_json_zero", "base_formal_zero", "base_natural_zero"],
    "ICL: dwa przykłady": ["base_json_balanced0", "base_formal_balanced0", "base_natural_balanced0"],
    "LoRA": ["rep_lora_json_zero", "rep_lora_formal_zero", "rep_lora_natural_zero"],
}


def plot_core(output: Path, rows: list[Record]) -> None:
    """Plot prespecified groups; no variant is chosen using its outcome."""
    indexed = {row["variant"]: row for row in rows}
    if not set(indexed) & {name for group in CORE_GROUPS.values() for name in group}:
        return
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    for axis, (title, names) in zip(axes, CORE_GROUPS.items(), strict=True):
        for index, name in enumerate(names):
            row = indexed.get(name)
            if row is None or not row["outputs"]:
                axis.text(index, .02, "brak", rotation=90, ha="center", fontsize=8)
                continue
            for offset, field, colour in ((-.18, "semantic_valid", "#315c7e"),
                                          (.18, "valid_not_whole_goal", "#c18356")):
                value, interval = row[field + "_board_mean"], row[field + "_95_board_bootstrap"]
                axis.bar(index + offset, value, width=.32, color=colour)
                if interval:
                    axis.vlines(index + offset, *interval, color="black", linewidth=.8)
                axis.text(index + offset, max(value, interval[1] if interval else value) + .02,
                          f"{row[field]}/{row['outputs']}\n({row['planned']})", ha="center", fontsize=7)
        axis.set_xticks(range(3), ["JSON", "Formalny", "Angielski"], fontsize=8)
        axis.set(xlim=(-.6, 2.6), ylim=(0, 1.24), title=title)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Średni odsetek na planszę")
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color="#315c7e", label="Poprawna nietrywialna intencja"),
                        Patch(color="#c18356", label="Poprawna, bez pełnego celu")],
               loc="outside upper center", ncols=2, fontsize=8)
    fig.supxlabel("Nad słupkiem: poprawne/ocenione (zaplanowane); 95% bootstrap po planszach.", fontsize=8)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"core_representations.{extension}", dpi=180)
    plt.close(fig)


def plot_contrasts(output: Path, rows: list[Record]) -> None:
    """Show fixed contrasts, keeping each paired board-bootstrap interval."""
    shown = [row for row in rows if row["family"] == row["previous_lora_exposure"] == "all"
             and row["metric"] in ("semantic_valid", "valid_not_whole_goal")]
    if not shown:
        return
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10, max(3, .4 * len(shown) / 2 + 1.5)), constrained_layout=True)
    for ax, metric, title in zip(axes, ("semantic_valid", "valid_not_whole_goal"),
                                 ("Poprawna intencja", "Poprawna, bez pełnego celu")):
        subset = [row for row in shown if row["metric"] == metric]
        for index, row in enumerate(subset):
            value = row["board_mean_delta"]
            if value is None:
                ax.text(0, index, "brak par", ha="center", fontsize=7)
                continue
            ax.scatter([value], [index], color="#315c7e", s=24)
            if row["bootstrap_low"] is not None:
                ax.hlines(index, row["bootstrap_low"], row["bootstrap_high"], color="#315c7e")
            ax.annotate(f"{row['paired_contexts']} kontekstów / {row['paired_boards']} plansz", (value, index),
                        xytext=(4, 3), textcoords="offset points", fontsize=6)
        ax.axvline(0, color="0.4", linewidth=.7)
        ax.set_yticks(range(len(subset)), [f"{r['candidate']} − {r['reference']}" for r in subset], fontsize=7)
        ax.set(xlim=(-1.15, 1.15), title=title, xlabel="Różnica odsetka na planszę")
        ax.grid(axis="x", alpha=.2)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"paired_overall.{extension}", dpi=180)
    plt.close(fig)


def comparison_pairs(variants: list[str], reference: str, requested: list[str]) -> list[tuple[str, str]]:
    """Fix additional contrasts explicitly; never select them from results."""
    candidates = [(reference, name) for name in variants if name != reference]
    for item in requested:
        before, separator, after = item.partition("=")
        pair = before, after
        if not separator or before not in variants or after not in variants or before == after:
            raise ValueError("Each --contrast must name distinct listed variants as REFERENCE=CANDIDATE")
        if pair not in candidates:
            candidates.append(pair)
    return candidates


def flat_summary(row: Record) -> Record:
    result = dict(row)
    for field in ("syntax_valid", "semantic_valid", "whole_goal", "whole_goal_only", "valid_not_whole_goal", "bfs_found",
                  "fence_syntax_valid", "fence_semantic_valid", "fence_whole_goal", "fence_valid_not_whole_goal"):
        bounds = row.get(field + "_95_board_bootstrap")
        result[field + "_bootstrap_low"] = bounds[0] if bounds else None
        result[field + "_bootstrap_high"] = bounds[1] if bounds else None
    return result


def training_seed_summary(rows: list[Record]) -> tuple[list[Record], list[Record]]:
    """Describe fixed training seeds separately from bootstrap over layouts."""
    indexed = {row["variant"]: row for row in rows}
    points: list[Record] = []
    summaries: list[Record] = []
    for representation in ("json", "formal", "natural"):
        for seed in (1729, 2718, 3141):
            suffix = "zero" if seed == 1729 else f"seed{seed}"
            variant = f"rep_lora_{representation}_{suffix}"
            row = indexed.get(variant)
            completed = bool(row and row.get("planned", 0) > 0 and row.get("outputs") == row.get("planned"))
            status = "complete" if completed else "partial" if row and row.get("outputs", 0) else "missing"
            points.append({"representation": representation, "seed": seed, "variant": variant, "status": status,
                           "planned": row.get("planned") if row else None, "outputs": row.get("outputs") if row else None,
                           "boards_observed": row.get("boards_observed") if row else None,
                           "semantic_valid_board_mean": row.get("semantic_valid_board_mean") if row else None,
                           "valid_not_whole_goal_board_mean": row.get("valid_not_whole_goal_board_mean") if row else None})
        for metric in ("semantic_valid", "valid_not_whole_goal"):
            selected = [row for row in points if row["representation"] == representation and row["status"] == "complete"
                        and row[metric + "_board_mean"] is not None]
            values = [row[metric + "_board_mean"] for row in selected]
            summaries.append({"representation": representation, "metric": metric, "seeds_planned": 3,
                              "seeds_complete": len(values), "complete_seeds": [row["seed"] for row in selected],
                              "mean_over_complete_seeds": sum(values) / len(values) if values else None,
                              "min_over_complete_seeds": min(values) if values else None,
                              "max_over_complete_seeds": max(values) if values else None,
                              "interpretation": "descriptive across completed training seeds; not a board bootstrap"})
    return points, summaries


def plot_training_seeds(output: Path, points: list[Record]) -> None:
    """Show individual measured seeds; board uncertainty is plotted separately."""
    if not any(row["status"] in ("complete", "partial") for row in points):
        return
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, constrained_layout=True)
    colours = {1729: "#315c7e", 2718: "#c18356", 3141: "#72965f"}
    offsets = {1729: -.18, 2718: 0, 3141: .18}
    for axis, metric, title in zip(axes, ("semantic_valid", "valid_not_whole_goal"),
                                   ("Poprawna intencja", "Poprawna, bez pełnego celu")):
        for index, representation in enumerate(("json", "formal", "natural")):
            for row in points:
                value = row.get(metric + "_board_mean")
                if row["representation"] != representation or value is None:
                    continue
                axis.scatter(index + offsets[row["seed"]], value, color=colours[row["seed"]],
                             marker="o" if row["status"] == "complete" else "x", s=35)
                axis.annotate(f"{row['outputs']}/{row['planned']}", (index + offsets[row["seed"]], value),
                              xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6)
        axis.set_xticks(range(3), ["JSON", "Formalny", "Kontrolowany angielski"], fontsize=8)
        axis.set(ylim=(-.05, 1.1), title=title, ylabel="Średni odsetek na planszę")
        axis.grid(axis="y", alpha=.2)
    handles = [Line2D([], [], marker="o", linestyle="none", color=colours[seed], label=str(seed))
               for seed in (1729, 2718, 3141)]
    axes[-1].legend(handles=handles, title="Seed treningu", fontsize=8)
    fig.suptitle("Wariacja między seedami treningu — opisowo")
    fig.supxlabel("Kółko: kompletna ocena; krzyżyk: częściowa. Przedziały po planszach pokazano osobno.", fontsize=8)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"training_seed_variation.{extension}", dpi=180)
    plt.close(fig)


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--data", required=True, type=Path)
    argument_parser.add_argument("--test-ids", required=True, type=Path)
    argument_parser.add_argument("--output", required=True, type=Path)
    argument_parser.add_argument("--variant", action="append", default=[], help="NAME=prediction.jsonl")
    argument_parser.add_argument("--matrix-plan", type=Path,
                                 help="Frozen plan; includes every planned variant and validates run metadata")
    argument_parser.add_argument("--input-snapshot", type=Path,
                                 help="Read immutable per-variant files from an earlier assessment snapshot")
    argument_parser.add_argument("--reference", required=True)
    argument_parser.add_argument("--contrast", action="append", default=[], help="Additional REFERENCE=CANDIDATE")
    argument_parser.add_argument("--contrasts-plan", type=Path,
                                 help="Frozen primary/secondary contrast pairs and primary metric")
    argument_parser.add_argument("--protocol", type=Path,
                                 default=Path("carl/icl/intent_representation_experiment.py"),
                                 help="Exact parser source used to generate the predictions")
    argument_parser.add_argument("--bfs", action="store_true")
    argument_parser.add_argument("--max-depth", type=int, default=24)
    argument_parser.add_argument("--max-nodes", type=int, default=5000)
    argument_parser.add_argument("--seconds", type=float, default=.2)
    args = argument_parser.parse_args()
    specification = importlib.util.spec_from_file_location("representation_recorded_protocol", args.protocol)
    if specification is None or specification.loader is None:
        raise ValueError("Could not load the recorded representation protocol")
    protocol = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(protocol)
    parse_representation = protocol.parse_representation
    protocol_hash = file_hash(args.protocol)
    if args.max_depth < 0 or args.max_nodes < 1 or args.seconds <= 0:
        argument_parser.error("BFS depth must be nonnegative; node and time limits must be positive")
    manifest = json.loads(args.test_ids.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != file_hash(args.data):
        raise ValueError("Test IDs are not bound to this dataset")
    contexts = test_contexts(read_jsonl(args.data), args.test_ids)
    if not contexts:
        raise ValueError("No frozen test contexts")
    matrix = None
    if args.matrix_plan:
        checksum = args.matrix_plan.with_suffix(".sha256")
        if not checksum.is_file() or checksum.read_text(encoding="utf-8").strip() != file_hash(args.matrix_plan):
            raise ValueError("Frozen matrix plan checksum is missing or differs")
        matrix = json.loads(args.matrix_plan.read_text(encoding="utf-8"))
    requested = variant_paths(args.variant)
    if matrix is not None:
        inputs = matrix["inputs"]
        if (inputs["data"]["sha256"] != file_hash(args.data) or inputs["test_ids"]["sha256"] != file_hash(args.test_ids)
                or inputs["protocol"]["sha256"] != protocol_hash):
            raise ValueError("Matrix plan differs from the evaluation dataset, IDs or parser")
        variants = [(method["name"], Path(method["output"])) for method in matrix["methods"]]
        if len({name for name, _ in variants}) != len(variants):
            raise ValueError("Matrix plan contains duplicate variants")
        if requested and requested != variants:
            raise ValueError("Explicit variants must match all frozen matrix methods and order")
    else:
        variants = requested
    if not variants:
        raise ValueError("Provide --matrix-plan or at least one --variant")
    if args.reference not in {name for name, _ in variants}:
        raise ValueError("Reference must be a listed variant")
    contrasts_path = args.contrasts_plan
    if contrasts_path is None and args.matrix_plan:
        candidate = args.matrix_plan.parent / "contrasts_plan.json"
        if candidate.exists():
            contrasts_path = candidate
    contrast_plan = json.loads(contrasts_path.read_text(encoding="utf-8")) if contrasts_path else None
    if contrast_plan is not None and contrast_plan.get("status") != "frozen_before_test_predictions":
        raise ValueError("Contrast plan does not record a freeze before test predictions")
    campaign_path = args.matrix_plan.parent / "campaign_state.json" if args.matrix_plan else None
    if contrast_plan is not None and contrasts_path is not None and campaign_path and campaign_path.exists():
        recorded = json.loads(campaign_path.read_text(encoding="utf-8")).get("contrasts_sha256")
        if recorded is not None and recorded != file_hash(contrasts_path):
            raise ValueError("Frozen contrast plan differs from the campaign record")
    planned_roles: dict[tuple[str, str], str] = {}
    for key, role in (("primary_contrasts", "primary"), ("secondary_contrasts", "secondary")):
        for pair in (contrast_plan or {}).get(key, []):
            if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(name, str) for name in pair):
                raise ValueError("Each frozen contrast must be a pair of variant names")
            typed_pair = pair[0], pair[1]
            if typed_pair in planned_roles:
                raise ValueError("Frozen contrast is listed more than once")
            planned_roles[typed_pair] = role
    if contrast_plan is not None and args.contrast:
        supplied = [tuple(item.split("=", 1)) for item in args.contrast]
        if len(supplied) != len(set(supplied)) or set(supplied) != set(planned_roles):
            raise ValueError("Explicit contrasts differ from the preregistered contrast plan")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "assessment.jsonl").exists() or (args.output / "config.json").exists():
        raise FileExistsError("Assessment exists; preserve it and choose a new output directory")
    snapshots = args.output / "input_snapshot"
    if args.input_snapshot is not None:
        snapshot_manifest = json.loads((args.input_snapshot / "manifest.json").read_text(encoding="utf-8"))
        if snapshot_manifest.get("matrix_plan_sha256") != (file_hash(args.matrix_plan) if args.matrix_plan else None):
            raise ValueError("Input snapshot belongs to a different matrix plan")
        if [item.get("variant") for item in snapshot_manifest.get("variants", [])] != [name for name, _ in variants]:
            raise ValueError("Input snapshot variant order differs from the evaluation")
        for entry in snapshot_manifest["variants"]:
            stem = entry["variant"]
            if set(entry.get("files", {})) != {f"{stem}.jsonl", f"{stem}.run.json", f"{stem}.prompts.jsonl"}:
                raise ValueError(f"Input snapshot file listing is incomplete: {stem}")
            for name, info in entry["files"].items():
                if Path(name).name != name or name in {".", ".."}:
                    raise ValueError("Invalid filename in the input snapshot")
                source = args.input_snapshot / name
                if (info["sha256"] is None and source.exists()) or (info["sha256"] is not None
                        and (not source.is_file() or file_hash(source) != info["sha256"])):
                    raise ValueError(f"Input snapshot file changed: {name}")
    captured = [snapshot_prediction((args.input_snapshot / f"{name}.jsonl") if args.input_snapshot else path,
                                    name, snapshots) for name, path in variants]
    atomic_json(snapshots / "manifest.json", {"schema_version": 1, "variants": captured,
                                               "matrix_plan_sha256": file_hash(args.matrix_plan) if args.matrix_plan else None,
                                               "note": "Exact bytes read for this assessment; missing files remain absent."})
    methods = {method["name"]: method for method in matrix["methods"]} if matrix is not None else {}
    loaded = []
    rejected = []
    for name, source_path in variants:
        path = snapshots / f"{name}.jsonl"
        try:
            predictions, info = read_predictions(path, name, contexts, args.data, args.test_ids)
            recorded = (info.get("metadata") or {}).get("protocol_sha256")
            if recorded is not None and recorded != protocol_hash:
                raise ValueError("Parser source hash differs from generation; use --protocol SNAPSHOT")
            if matrix is not None and info.get("metadata") is not None:
                validate_matrix_run(matrix, methods[name], info["metadata"], path)
            for record in contexts:
                assess_one(record, predictions.get(record["record_id"]), parse_representation, bfs=False,
                           max_depth=args.max_depth, max_nodes=args.max_nodes, seconds=args.seconds)
        except (ValueError, KeyError, TypeError, OSError) as error:
            reason = f"{type(error).__name__}: {error}"
            try:
                digest = file_hash(path) if path.is_file() else None
            except OSError:
                digest = None
            info = {"path": str(path), "sha256": digest,
                    "partial_final_line": None, "metadata": None, "rejected_reason": reason}
            predictions = {}
            rejected.append({"variant": name, "path": str(path), "reason": reason})
        info["source_path"] = str((args.input_snapshot / f"{name}.jsonl") if args.input_snapshot else source_path)
        loaded.append((name, predictions, info))
    config: Record = {"data_sha256": file_hash(args.data), "test_ids_sha256": file_hash(args.test_ids),
                      "parser_path": str(args.protocol), "parser_sha256": protocol_hash,
                      "matrix_plan_path": str(args.matrix_plan) if args.matrix_plan else None,
                      "matrix_plan_sha256": file_hash(args.matrix_plan) if args.matrix_plan else None,
                      "contrasts_plan_sha256": file_hash(contrasts_path) if contrasts_path else None,
                      "contrasts_plan": contrast_plan,
                      "input_snapshot_source": str(args.input_snapshot) if args.input_snapshot else None,
                      "input_snapshot_manifest_sha256": file_hash(snapshots / "manifest.json"),
                      "assessment_source_sha256": {
                          str(path.relative_to(ROOT)): file_hash(path) for path in (
                              Path(__file__).resolve(), ROOT / "scripts/assess_lora_matrix.py",
                              ROOT / "carl/icl/intent_assessment.py", ROOT / "carl/icl/intent_data.py",
                              ROOT / "carl/icl/intent_fence_audit.py")},
                      "variants": {name: info for name, _, info in loaded},
                      "settings_audit": settings_audit([(name, info) for name, _, info in loaded]),
                      "bfs": args.bfs, "max_depth": args.max_depth, "max_nodes": args.max_nodes,
                      "seconds": args.seconds, "seed": RNG_SEED, "bootstrap_resamples": 2000,
                      "bfs_interpretation": "Bounded solver, not a controller; cutoff is unresolved.",
                      "format_interpretation": "Strict parser checks representation syntax and common intent schema; no formatting repairs.",
                      "fence_interpretation": "Separate posthoc analysis removes at most one complete plain/json Markdown fence under the same rule for all formats. "
                                              "Whole-goal and valid-not-whole-goal partition semantically valid intents, using all evaluated outputs as denominator.",
                      "exposure_interpretation": "Not in the previous LoRA matrix does not establish lack of prior model exposure."}
    atomic_json(args.output / "config.json", config)
    atomic_json(args.output / "rejected_variants.json", rejected)
    all_rows: list[Record] = []
    all_summaries: list[Record] = []
    family_summaries: list[Record] = []
    exposure_summaries: list[Record] = []
    families = sorted({board_family(row) for row in contexts})
    exposures = sorted({exposure_group(row) for row in contexts})
    with (args.output / "assessment.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        for name, predictions, info in loaded:
            rows = []
            for record in contexts:
                if info.get("rejected_reason"):
                    result = {"variant": name, "record_id": record["record_id"], "layout_id": record["layout_id"],
                              "family": board_family(record), "previous_lora_exposure": exposure_group(record),
                              "status": "rejected_infrastructure", "error": info["rejected_reason"]}
                else:
                    result = {"variant": name, **assess_one(record, predictions.get(record["record_id"]),
                                                            parse_representation, bfs=args.bfs, max_depth=args.max_depth,
                                                            max_nodes=args.max_nodes, seconds=args.seconds)}
                stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                rows.append(result)
            all_rows.extend(rows)
            all_summaries.append({"family": "all", **summarize(name, rows, bfs=args.bfs)})
            family_summaries.extend({"family": family, **summarize(name, [row for row in rows if row["family"] == family],
                                                                    bfs=args.bfs)} for family in families)
            exposure_summaries.extend({"family": family or "all", "previous_lora_exposure": exposure,
                                       **summarize(name, [row for row in rows if row["previous_lora_exposure"] == exposure
                                                         and (family is None or row["family"] == family)], bfs=args.bfs)}
                                      for exposure in exposures for family in (None, *families)
                                      if any(row["previous_lora_exposure"] == exposure
                                             and (family is None or row["family"] == family) for row in rows))
            atomic_json(args.output / "summary.json", {"config": config, "results": all_summaries,
                                                        "family_results": family_summaries,
                                                        "exposure_results": exposure_summaries,
                                                        "rejected_variants": rejected})
            print(json.dumps({"variant": name, "outputs": len(predictions),
                              "semantic_valid": all_summaries[-1]["semantic_valid"]}), flush=True)
    pairs = comparison_pairs([name for name, _ in variants], args.reference,
                             args.contrast + [f"{before}={after}" for before, after in planned_roles])
    contrasts = [{**paired(all_rows, before, after, metric, family, exposure),
                  "contrast_role": planned_roles.get((before, after), "reference_extra"),
                  "metric_protocol": "posthoc_fence" if metric.startswith("fence_") else "strict",
                  "registered_primary_metric": bool(contrast_plan and metric == contrast_plan.get("primary_metric"))}
                 for before, after in pairs
                 for metric in ("syntax_valid", "semantic_valid", "valid_not_whole_goal", "whole_goal_only",
                                "fence_syntax_valid", "fence_semantic_valid", "fence_whole_goal",
                                "fence_valid_not_whole_goal")
                 for family in (None, *families) for exposure in (None, *exposures)]
    seed_points, seed_summaries = training_seed_summary(all_summaries)
    summary = {"config": config, "results": all_summaries, "family_results": family_summaries,
               "exposure_results": exposure_summaries, "paired": contrasts, "rejected_variants": rejected,
               "training_seed_points": seed_points, "training_seed_summary": seed_summaries}
    atomic_json(args.output / "summary.json", summary)
    write_csv(args.output / "summary.csv", [flat_summary(row) for row in all_summaries + family_summaries],
              ["family", "variant", "planned", "outputs", "missing", "infrastructure_errors", "boards_planned",
               "boards_observed", "syntax_valid", "raw_json_valid", "fence_removed", "fence_syntax_valid",
               "fence_semantic_valid", "fence_whole_goal", "fence_valid_not_whole_goal",
               "semantic_valid", "whole_goal", "whole_goal_only", "valid_not_whole_goal",
               "distinct_intents", "bfs_attempted", "bfs_found", "bfs_unresolved", "bfs_errors",
               "bfs_pipeline_evaluated", "generation_seconds", "generation_seconds_known", "generated_tokens",
               "generated_tokens_known", *[field + suffix for field in
                   ("syntax_valid", "semantic_valid", "whole_goal", "whole_goal_only", "valid_not_whole_goal", "bfs_found",
                    "fence_syntax_valid", "fence_semantic_valid", "fence_whole_goal", "fence_valid_not_whole_goal")
                   for suffix in ("_board_mean", "_bootstrap_low", "_bootstrap_high")]])
    if exposure_summaries:
        write_csv(args.output / "exposure.csv", [flat_summary(row) for row in exposure_summaries],
                  ["family", "previous_lora_exposure", "variant", "planned", "outputs", "missing", "boards_planned",
                   "boards_observed", "syntax_valid", "semantic_valid", "valid_not_whole_goal", "whole_goal",
                   "fence_semantic_valid", "fence_whole_goal", "fence_valid_not_whole_goal",
                   "bfs_attempted", "bfs_found", "bfs_unresolved"])
    if contrasts:
        write_csv(args.output / "paired.csv", contrasts, list(contrasts[0]))
        plot_contrasts(args.output, contrasts)
    write_csv(args.output / "training_seed_points.csv", seed_points, list(seed_points[0]))
    plot_training_seeds(args.output, seed_points)
    plot_summary(args.output, all_summaries, "Wszystkie plansze")
    atomic_json(args.output / "core_methods.json", CORE_GROUPS)
    plot_core(args.output, all_summaries)
    if args.bfs:
        plot_bfs(args.output, all_summaries, "Wszystkie plansze")
    for family in families:
        plot_summary(args.output, [row for row in family_summaries if row["family"] == family], family)
        if args.bfs:
            plot_bfs(args.output, [row for row in family_summaries if row["family"] == family], family)
    for exposure in exposures:
        subset = [row for row in exposure_summaries if row["family"] == "all"
                  and row["previous_lora_exposure"] == exposure]
        if subset:
            plot_summary(args.output, subset, f"Poprzednia macierz LoRA: {exposure}")


if __name__ == "__main__":
    main()
