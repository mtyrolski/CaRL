#!/usr/bin/env python3
"""Compare frozen Sokoban generator outputs, including interrupted variants.

Strict JSON and one-fence removal are separate analyses. Optional bounded BFS
checks feasibility only; it is not a language-model or learned controller.
"""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.patches import Patch

from carl.icl.intent_assessment import board_means
from carl.icl.intent_assessment import bootstrap
from carl.icl.intent_assessment import bounded_witness
from carl.icl.intent_assessment import check_raw_prediction
from carl.icl.intent_assessment import file_hash
from carl.icl.intent_assessment import read_jsonl
from carl.icl.intent_assessment import test_contexts
from carl.icl.intent_data import validate_prediction
from carl.icl.intent_fence_audit import inspect_output
from carl.icl.intent_generator_experiment import parse_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def variants(values: list[str]) -> list[tuple[str, Path]]:
    result = []
    for item in values:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or any(name == old for old, _ in result):
            raise ValueError("Each --variant must have a distinct NAME=PATH")
        result.append((name, Path(path)))
    return result


def parse_partial(data: bytes) -> tuple[list[dict[str, Any]], bool]:
    partial = False
    if data and not data.endswith(b"\n"):
        before, _, tail = data.rpartition(b"\n")
        try:
            json.loads(tail)
        except (UnicodeDecodeError, ValueError):
            data, partial = before + (b"\n" if before else b""), True
    return [json.loads(line) for line in data.splitlines() if line.strip()], partial


def read_partial(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read complete JSONL lines; leave an interrupted final line untouched."""
    return parse_partial(path.read_bytes()) if path.exists() else ([], False)


def equivalent_test_ids(canonical: Path, equivalents: list[Path], data: Path) -> dict[str, str]:
    """Accept alternate manifests only when their ordered IDs are identical."""
    base = json.loads(canonical.read_text(encoding="utf-8"))
    identifiers = base.get("record_ids")
    if (not isinstance(identifiers, list) or not all(isinstance(item, str) for item in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError("Frozen test IDs must be a distinct ordered list")
    hashes = {}
    for path in (canonical, *equivalents):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("record_ids") != identifiers:
            raise ValueError(f"Test ID manifest has a different selection or order: {path}")
        dataset_hash = item.get("dataset_sha256")
        if dataset_hash is not None and dataset_hash != file_hash(data):
            raise ValueError(f"Test ID manifest has a different dataset hash: {path}")
        hashes[str(path)] = file_hash(path)
    return hashes


def load_predictions(path: Path, contexts: list[dict[str, Any]], data: Path,
                     accepted_ids_hashes: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {row["record_id"]: row["layout_id"] for row in contexts}
    snapshot = path.read_bytes() if path.exists() else None
    rows, partial = parse_partial(snapshot) if snapshot is not None else ([], False)
    predictions: dict[str, Any] = {}
    for row in rows:
        identifier = row.get("record_id")
        if (not isinstance(identifier, str) or identifier not in expected or identifier in predictions
                or row.get("layout_id") != expected[identifier]
                or row.get("split") != "test"):
            raise ValueError(f"Prediction conflicts with frozen test contexts: {identifier}")
        check_raw_prediction(row)
        predictions[identifier] = row
    summary = path.with_suffix(".summary.json")
    run = path.with_suffix(".run.json")
    metadata_path = summary if summary.exists() else run if run.exists() else None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path else None
    if (metadata_path == summary and metadata is not None and metadata.get("n", 0) > len(rows) and run.exists()):
        # Generation completed after this process captured its consistent byte snapshot.
        metadata_path = run
        metadata = json.loads(run.read_text(encoding="utf-8"))
    if metadata is not None:
        if metadata.get("data_sha256") != file_hash(data) or metadata.get("test_ids_sha256") not in accepted_ids_hashes:
            raise ValueError(f"Prediction metadata does not match frozen input: {path}")
        if metadata_path == summary and (metadata.get("status") != "complete" or metadata.get("n") != len(rows)):
            raise ValueError(f"Completed prediction summary disagrees with rows: {path}")
    return predictions, {"path": str(path), "sha256": hashlib.sha256(snapshot).hexdigest() if snapshot is not None else None,
                         "partial_final_line": partial, "metadata": metadata}


def intent_signature(intent: dict[str, Any]) -> str:
    """Ignore optional prose and predicate order when counting distinct intents."""
    return json.dumps({key: sorted(json.dumps(item, sort_keys=True, separators=(",", ":"))
                                   for item in intent[key]) for key in ("effects", "preserve")}, sort_keys=True)


def whole_goal(intent: dict[str, Any], board: list[str]) -> bool:
    boxes = sum(row.count("$") + row.count("*") for row in board)
    return any(effect.get("predicate") == "boxes_on_goals_at_least" and effect.get("count", -1) >= boxes
               for effect in intent["effects"])


def assess(record: dict[str, Any], prediction: dict[str, Any] | None, *, bfs: bool,
           max_depth: int, max_nodes: int, seconds: float) -> dict[str, Any]:
    board = record["start_observation"]
    fallback_family = f"{len(board)}x{len(board[0])}x{sum(row.count('$') + row.count('*') for row in board)}"
    result: dict[str, Any] = {"record_id": record["record_id"], "layout_id": record["layout_id"],
                              "family": record.get("evaluation_family") or record.get("family") or fallback_family,
                              "status": "missing" if prediction is None else "evaluated"}
    if prediction is None:
        return result
    raw = prediction["raw_output"]
    try:
        strict_intent = parse_json(raw)
        strict_json = True
    except ValueError:
        strict_intent, strict_json = None, False
    strict_errors = validate_prediction(strict_intent, record["start_observation"]) if strict_json else ["invalid JSON"]
    relaxed = inspect_output(raw, record["start_observation"])
    valid_intent = strict_intent if strict_json and not strict_errors and isinstance(strict_intent, dict) else None
    valid = valid_intent is not None
    strict_effects = valid_intent["effects"] if valid_intent is not None else []
    recovered = relaxed["parsed"]
    fence_intent = recovered if relaxed["valid_nontrivial"] and isinstance(recovered, dict) else None
    fence_effects = fence_intent["effects"] if fence_intent is not None else []
    result.update({"strict_json": strict_json, "strict_valid": bool(valid), "strict_errors": strict_errors,
                   "fence_removed": relaxed["fence_removed"], "fence_json": relaxed["json_valid"],
                   "fence_valid": relaxed["valid_nontrivial"],
                   "strict_whole_goal": whole_goal(valid_intent, record["start_observation"]) if valid_intent is not None else False,
                   "fence_whole_goal": whole_goal(fence_intent, record["start_observation"]) if fence_intent is not None else False,
                   "strict_whole_goal_only": bool(valid_intent is not None and len(strict_effects) == 1
                                                  and whole_goal(valid_intent, record["start_observation"])),
                   "fence_whole_goal_only": bool(fence_intent is not None and len(fence_effects) == 1
                                                 and whole_goal(fence_intent, record["start_observation"])),
                   "strict_intent_signature": intent_signature(valid_intent) if valid_intent is not None else None,
                   "fence_intent_signature": intent_signature(fence_intent) if fence_intent is not None else None,
                   "strict_predicates": [item["predicate"] for item in strict_effects],
                   "fence_predicates": [item["predicate"] for item in fence_effects],
                   "strict_conjunction_size": len(strict_effects), "fence_conjunction_size": len(fence_effects),
                   "generation_seconds": prediction.get("generation_seconds"),
                   "generated_tokens": prediction.get("generated_tokens"), "raw_output": raw,
                   "strict_intent": strict_intent, "fence_intent": fence_intent, "start_observation": board,
                   "demonstration_later_observation": record.get("later_observation"),
                   "solver": None})
    if bfs and valid_intent is not None:
        try:
            result["solver"] = bounded_witness(record["start_observation"], valid_intent, max_depth, max_nodes, seconds)
        except (AssertionError, ValueError) as error:
            result["solver"] = {"status": "verification_error", "witness_found": False, "error": str(error)}
    return result


def summarize(name: str, rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    measured = [row for row in rows if row["status"] == "evaluated"]
    valid = [row for row in measured if row["strict_valid"]]
    fence_valid = [row for row in measured if row["fence_valid"]]
    found = [row for row in valid if row["solver"] and row["solver"].get("witness_found")]
    solver_errors = [row for row in valid if row["solver"] and row["solver"].get("status") == "verification_error"]
    solver_assessed = [row for row in measured if row not in solver_errors]
    known_seconds = [row["generation_seconds"] for row in measured if isinstance(row.get("generation_seconds"), (int, float))
                     and not isinstance(row["generation_seconds"], bool)]
    known_tokens = [row["generated_tokens"] for row in measured if isinstance(row.get("generated_tokens"), int)
                    and not isinstance(row["generated_tokens"], bool)]
    summary: dict[str, Any] = {"variant": name, "planned": len(rows), "outputs": len(measured),
                               "missing": len(rows) - len(measured), "boards_planned": len({r['layout_id'] for r in rows}),
                               "boards_observed": len({r['layout_id'] for r in measured}),
                               "strict_json": sum(row["strict_json"] for row in measured),
                               "strict_valid": len(valid), "fence_removed": sum(row["fence_removed"] for row in measured),
                               "fence_json": sum(row["fence_json"] for row in measured),
                               "fence_valid": sum(row["fence_valid"] for row in measured),
                               "strict_whole_goal": sum(row["strict_whole_goal"] for row in valid),
                               "fence_whole_goal": sum(row["fence_whole_goal"] for row in fence_valid),
                               "strict_whole_goal_only": sum(row["strict_whole_goal_only"] for row in valid),
                               "fence_whole_goal_only": sum(row["fence_whole_goal_only"] for row in fence_valid),
                               "strict_distinct_intents": len({row["strict_intent_signature"] for row in valid}),
                               "fence_distinct_intents": len({row["fence_intent_signature"] for row in fence_valid}),
                               "strict_predicate_counts": dict(Counter(value for row in valid for value in row["strict_predicates"])),
                               "fence_predicate_counts": dict(Counter(value for row in fence_valid for value in row["fence_predicates"])),
                               "strict_conjunction_sizes": dict(Counter(str(row["strict_conjunction_size"]) for row in valid)),
                               "fence_conjunction_sizes": dict(Counter(str(row["fence_conjunction_size"]) for row in fence_valid)),
                               "generation_seconds": sum(known_seconds) if measured and len(known_seconds) == len(measured) else None,
                               "generation_seconds_known": len(known_seconds),
                               "generated_tokens": sum(known_tokens) if measured and len(known_tokens) == len(measured) else None,
                               "generated_tokens_known": len(known_tokens),
                               "bfs_enabled": config["bfs"], "bfs_found": len(found) if config["bfs"] else None,
                               "bfs_attempted": sum(row["solver"] is not None for row in valid) if config["bfs"] else None,
                               "pipeline_evaluated_for_bfs": len(solver_assessed) if config["bfs"] else None,
                               "bfs_unresolved": len(valid) - len(found) - len(solver_errors) if config["bfs"] else None,
                               "bfs_errors": len(solver_errors),
                               "bfs_statuses": dict(Counter(row["solver"]["status"] for row in valid if row["solver"])),
                               "bfs_expanded_nodes": sum(row["solver"].get("expanded_nodes", 0) for row in valid if row["solver"]),
                               "bfs_seconds": sum(row["solver"].get("elapsed_seconds", 0) for row in valid if row["solver"])}
    for field in ("strict_json", "strict_valid", "fence_json", "fence_valid", "strict_whole_goal", "fence_whole_goal",
                  "strict_whole_goal_only", "fence_whole_goal_only"):
        means = board_means(measured, lambda row: int(row[field]))
        summary[field + "_board_mean"] = sum(means) / len(means) if means else None
        summary[field + "_95_board_bootstrap"] = bootstrap(means, random.Random(20261114))
    if config["bfs"]:
        means = board_means(solver_assessed, lambda row: int(bool(row.get("solver") and row["solver"].get("witness_found"))))
        summary["bfs_found_board_mean"] = sum(means) / len(means) if means else None
        summary["bfs_found_95_board_bootstrap"] = bootstrap(means, random.Random(20261114))
    return summary


def paired(rows: list[dict[str, Any]], reference: str, candidate: str) -> dict[str, Any]:
    before = {r["record_id"]: r for r in rows if r["variant"] == reference and r["status"] == "evaluated"}
    after = {r["record_id"]: r for r in rows if r["variant"] == candidate and r["status"] == "evaluated"}
    joined = [{"layout_id": before[key]["layout_id"],
               "delta": int(after[key]["strict_valid"]) - int(before[key]["strict_valid"])}
              for key in sorted(before.keys() & after.keys())]
    means = board_means(joined, lambda row: row["delta"])
    return {"reference": reference, "candidate": candidate, "contexts": len(joined), "boards": len(means),
            "strict_valid_delta": sum(means) / len(means) if means else None,
            "strict_valid_delta_95_board_bootstrap": bootstrap(means, random.Random(20261114))}


def settings_audit(loaded: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Expose mismatches and unrecorded settings before interpreting comparisons."""
    fields = ("model", "revision", "seed", "decoding", "enable_thinking", "max_length", "max_new_tokens", "dtype", "prompt_sha256")
    result = {}
    for field in fields:
        values = {name: (info["metadata"] or {}).get(field) for name, _, info in loaded}
        known = {json.dumps(value, sort_keys=True) for value in values.values() if value is not None}
        result[field] = {"values": values, "missing": [name for name, value in values.items() if value is None],
                         "mismatch": len(known) > 1}
    return result


def plot(output: Path, summaries: list[dict[str, Any]], name: str = "overall") -> None:
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(max(7, len(summaries) * 1.5), 4), constrained_layout=True)
    for x, row in enumerate(summaries):
        if not row["outputs"]:
            ax.text(x, .02, "brak", rotation=90, ha="center")
            continue
        for offset, field, colour in ((-.19, "strict_valid", "#436c8c"),
                                      (.19, "fence_valid", "#c18356")):
            value = row[field + "_board_mean"]
            if value is None:
                ax.text(x + offset, .02, "brak", rotation=90, ha="center")
                continue
            ax.bar(x + offset, value, width=.34, color=colour)
            interval = row[field + "_95_board_bootstrap"]
            if interval:
                ax.vlines(x + offset, *interval, color="black", linewidth=.8)
            ax.text(x + offset, max(value, interval[1] if interval else value) + .025,
                    f"{row[field]}/{row['outputs']}\n(plan {row['planned']})", ha="center", fontsize=7)
    ax.set_xticks(range(len(summaries)), [row["variant"] for row in summaries], rotation=20, ha="right")
    ax.set(xlim=(-.65, len(summaries) - .35), ylim=(0, 1.2),
           ylabel="Średni odsetek poprawnych intencji na planszę")
    ax.set_title(name)
    ax.legend(handles=[Patch(color="#436c8c", label="Ścisły JSON"),
                       Patch(color="#c18356", label="Po usunięciu jednej otoczki")], fontsize=8)
    ax.grid(axis="y", alpha=.2)
    for ext in ("png", "pdf", "svg"):
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
        fig.savefig(output / f"validity_{safe_name}.{ext}", dpi=180)
    plt.close(fig)


def plot_bfs(output: Path, summaries: list[dict[str, Any]], name: str = "overall") -> None:
    """Show pipeline success per evaluated output; verification errors are excluded."""
    plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(max(7, len(summaries) * 1.5), 4), constrained_layout=True)
    for x, row in enumerate(summaries):
        value = row.get("bfs_found_board_mean")
        if value is None:
            ax.text(x, .02, "brak", rotation=90, ha="center")
            continue
        ax.bar(x, value, width=.6, color="#80609a")
        interval = row["bfs_found_95_board_bootstrap"]
        if interval:
            ax.vlines(x, *interval, color="black", linewidth=.8)
        ax.text(x, max(value, interval[1] if interval else value) + .025,
                f"{row['bfs_found']}/{row['pipeline_evaluated_for_bfs']} odp.\n"
                f"BFS: {row['bfs_found']}/{row['bfs_attempted']}", ha="center", fontsize=7)
    ax.set_xticks(range(len(summaries)), [row["variant"] for row in summaries], rotation=20, ha="right")
    ax.set(xlim=(-.65, len(summaries) - .35), ylim=(0, 1.2),
           ylabel="Średni odsetek na planszę", title=f"{name}: ograniczone BFS, nie kontroler")
    fig.supxlabel("Mianownik słupka: ocenione odpowiedzi (także błędne); BFS uruchomiono tylko dla poprawnych intencji.",
                  fontsize=8)
    ax.grid(axis="y", alpha=.2)
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"bounded_bfs_{safe_name}.{ext}", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--equivalent-test-ids", action="append", type=Path, default=[],
                        help="Alternate ID manifest, accepted only if the ordered selection is identical")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="NAME=prediction.jsonl; repeat per method")
    parser.add_argument("--bfs", action="store_true")
    parser.add_argument("--max-depth", type=int, default=24)
    parser.add_argument("--max-nodes", type=int, default=10000)
    parser.add_argument("--seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_depth < 0 or args.max_nodes < 1 or args.seconds <= 0:
        parser.error("BFS limits must be nonnegative depth and positive nodes/seconds")
    frozen_dataset = json.loads(args.test_ids.read_text(encoding="utf-8")).get("dataset_sha256")
    equivalent_hashes = equivalent_test_ids(args.test_ids, args.equivalent_test_ids, args.data)
    contexts = test_contexts(read_jsonl(args.data), args.test_ids)
    if not contexts:
        raise ValueError("The frozen selection is empty")
    inputs = variants(args.variant)
    loaded = [(name, *load_predictions(path, contexts, args.data, set(equivalent_hashes.values()))) for name, path in inputs]
    config = {"data_sha256": file_hash(args.data), "test_ids_sha256": file_hash(args.test_ids),
              "test_ids_dataset_hash_recorded": frozen_dataset is not None,
              "equivalent_test_ids_sha256": equivalent_hashes,
              "bfs": args.bfs, "max_depth": args.max_depth, "max_nodes": args.max_nodes, "seconds": args.seconds,
              "variants": {name: info for name, _, info in loaded},
              "settings_audit": settings_audit(loaded),
              "interpretation": "BFS finds a bounded action sequence; it does not measure any learned or LLM controller.",
              "fence_analysis": "Post-hoc removal of one complete outer Markdown fence; raw outputs are retained."}
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists() or (args.output / "assessment.jsonl").exists():
        raise FileExistsError("Assessment already exists; preserve it and use a new output directory")
    atomic_json(args.output / "config.json", config)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    family_summaries: list[dict[str, Any]] = []
    output_rows = args.output / "assessment.jsonl"
    with output_rows.open("x", encoding="utf-8", buffering=1) as stream:
        for name, predictions, _ in loaded:
            rows = []
            for record in contexts:
                row = {"variant": name, **assess(record, predictions.get(record["record_id"]), bfs=args.bfs,
                                                  max_depth=args.max_depth, max_nodes=args.max_nodes, seconds=args.seconds)}
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                rows.append(row)
            all_rows.extend(rows)
            summaries.append(summarize(name, rows, config))
            family_summaries.extend({"family": family, **summarize(name, [row for row in rows if row["family"] == family], config)}
                                    for family in sorted({row["family"] for row in rows}))
            summary = {"results": summaries, "family_results": family_summaries,
                       "paired_strict": [paired(all_rows, summaries[0]["variant"], row["variant"])
                                         for row in summaries[1:]],
                       "paired_strict_by_family": [
                           {"family": family, **paired([r for r in all_rows if r["family"] == family],
                                                      summaries[0]["variant"], row["variant"])}
                           for row in summaries[1:] for family in sorted({r["family"] for r in all_rows})],
                       "config": config}
            atomic_json(args.output / "summary.json", summary)
            print(json.dumps({"variant": name, "outputs": len(predictions), "strict_valid": summaries[-1]["strict_valid"]}), flush=True)
    flat_fields = ["family", "variant", "planned", "outputs", "missing", "boards_planned", "boards_observed", "strict_json",
                   "strict_valid", "fence_removed", "fence_json", "fence_valid", "strict_whole_goal", "fence_whole_goal",
                   "strict_whole_goal_only", "fence_whole_goal_only",
                   "strict_distinct_intents", "fence_distinct_intents",
                   "bfs_found", "bfs_attempted", "pipeline_evaluated_for_bfs", "bfs_unresolved", "bfs_errors",
                   "bfs_expanded_nodes", "bfs_seconds",
                   "generation_seconds", "generation_seconds_known", "generated_tokens", "generated_tokens_known"]
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{"family": "all", **row} for row in summaries] + family_summaries)
    plot(args.output, summaries)
    if args.bfs:
        plot_bfs(args.output, summaries)
    for family in sorted({row["family"] for row in family_summaries}):
        plot(args.output, [row for row in family_summaries if row["family"] == family], family)
        if args.bfs:
            plot_bfs(args.output, [row for row in family_summaries if row["family"] == family], family)


if __name__ == "__main__":
    main()
