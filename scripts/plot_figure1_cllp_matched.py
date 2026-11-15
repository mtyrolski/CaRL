#!/usr/bin/env python3
"""Plot paired full-state reachability from a verified Figure 1 pilot and CLLPs.

The CLLPs are greedy trained policies, whereas the earlier language-model
controllers were sampled and received 32 examples. The shared action cap does
not imply equal computation. Incomplete or mismatched inputs stop the export.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import NormalDist
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.switch_backend("Agg")
ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "analysis/icl_reachability/figure1_proposal_hard_recovery_v5"
OFFSETS = (1, 2, 4, 8, 12, 16)
BOARD_COUNT = 20
ACTION_CAP = 32


class Model(StrEnum):
    """Policies measured on the same full-state target pairs."""

    LUNA = "Luna"
    TERRA = "Terra"
    CLLP_8 = "CLLP-8"
    CLLP_32 = "CLLP-32"


@dataclass(frozen=True, slots=True)
class Pair:
    """Frozen source and target, identified without passing demonstrations to CLLP."""

    pair_id: str
    layout_id: str
    context_id: str
    source_hash: str
    target_hash: str
    k: int


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects: {path}")
    return rows


def wilson(hits: int, n: int) -> tuple[float, float]:
    """Return a pointwise two-sided 95% Wilson score interval."""
    if n <= 0 or not 0 <= hits <= n:
        raise ValueError("Invalid hit count")
    z = NormalDist().inv_cdf(0.975)
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return min(p, max(0.0, centre - radius)), max(p, min(1.0, centre + radius))


def load_baseline(
    pairs_path: Path, trials_path: Path, summary_path: Path, manifest_path: Path, verification_path: Path
) -> tuple[dict[str, Pair], dict[Model, dict[str, bool]]]:
    """Recover the exact 20 × six pairs and verify the earlier trial summary."""
    manifest = read_json(manifest_path)
    study = manifest["study"]
    if manifest["dataset_artifacts"]["core_pairs.json"] != sha256(pairs_path):
        raise ValueError("Pair dataset differs from the pilot manifest")
    if (study["core_offsets"] != list(OFFSETS) or study["core_repetitions"] != 1
            or study["action_cap"] != ACTION_CAP or study["few_shot"] != 32
            or study["policy_temperature"] != 1.0):
        raise ValueError("Pilot protocol differs from the declared comparison")
    verification = read_json(verification_path)["12"]
    if (verification.get("mode") != "live" or verification.get("counts", {}).get("complete") != 240
            or verification.get("verified_files") != 240):
        raise ValueError("Complete replay-verified 12 × 12 pilot export required")

    jobs: dict[str, dict[str, Any]] = {}
    for job in manifest["jobs"]:
        if job["key"] in jobs:
            raise ValueError("Duplicate trial key in pilot manifest")
        jobs[job["key"]] = job

    lookup: dict[tuple[str, int], Pair] = {}
    for raw in read_json(pairs_path)["pairs"]:
        pair = Pair(raw["pair_id"], raw["layout_id"], raw["context_id"], raw["source_hash"],
                    raw["target_hash"], int(raw["k"]))
        key = (pair.context_id, pair.k)
        if key in lookup or pair.source_hash == pair.target_hash:
            raise ValueError("Duplicate context/offset or already satisfied target")
        lookup[key] = pair

    baseline: dict[Model, dict[str, bool]] = {Model.LUNA: {}, Model.TERRA: {}}
    pairs: dict[str, Pair] = {}
    arms = {"luna": Model.LUNA, "terra_6_1": Model.TERRA}
    model_ids = {"luna": "gpt-6-luna", "terra_6_1": "gpt-6.1-terra"}
    for row in read_csv(trials_path):
        if row["family"] != "12":
            continue
        if row["arm"] not in arms or row["status"] != "complete" or row["hit"] not in {"0", "1"}:
            raise ValueError("A pilot trial is missing, unresolved or unrecognised")
        key = (row["context"], int(row["k"]))
        if key not in lookup:
            raise ValueError(f"Pilot context/offset is absent from pair data: {key}")
        pair = lookup[key]
        if pair.layout_id != row["layout"]:
            raise ValueError(f"Pilot layout differs from pair data: {pair.pair_id}")
        job = jobs.get(row["trial_key"])
        expected = {"arm": row["arm"], "context": pair.context_id, "layout": pair.layout_id,
                    "pair": pair.pair_id, "source_hash": pair.source_hash, "target_hash": pair.target_hash,
                    "k": pair.k, "model": model_ids[row["arm"]], "repetition": 0, "group": "main"}
        if not isinstance(job, dict) or any(job.get(name) != value for name, value in expected.items()):
            raise ValueError(f"Pilot manifest identity mismatch: {pair.pair_id}")
        trace_path = Path(row["trace_path"])
        if not trace_path.is_absolute():
            trace_path = ROOT / trace_path
        trace = read_json(trace_path)
        if trace.get("identity") != job or trace.get("status") != "complete":
            raise ValueError(f"Pilot trace identity or status mismatch: {pair.pair_id}")
        model = arms[row["arm"]]
        if pair.pair_id in baseline[model] or (pair.pair_id in pairs and pair != pairs[pair.pair_id]):
            raise ValueError(f"Duplicate or conflicting pilot pair: {pair.pair_id}")
        attempted = int(row["attempted_actions"])
        hit = row["hit"] == "1"
        if not 0 <= attempted <= ACTION_CAP or (hit and row["first_hit"] != str(attempted)):
            raise ValueError(f"Invalid pilot duration: {pair.pair_id}")
        if not hit and row["first_hit"]:
            raise ValueError(f"Failed pilot trial has a first hit: {pair.pair_id}")
        if (trace.get("exact_hit") is not hit or trace.get("attempted_actions") != attempted
                or trace.get("first_hit") != (attempted if hit else None)
                or (trace.get("endpoint_hash") == pair.target_hash) != hit):
            raise ValueError(f"Pilot trace outcome disagrees with its CSV: {pair.pair_id}")
        pairs[pair.pair_id] = pair
        baseline[model][pair.pair_id] = hit

    _validate_cohort(pairs)
    for model, results in baseline.items():
        if set(results) != set(pairs):
            raise ValueError(f"Missing paired pilot trials: {model}")
    summary_rows = [row for row in read_csv(summary_path) if row["family"] == "12"]
    source_summary = {(row["arm"], int(row["k"])): row for row in summary_rows}
    if len(source_summary) != len(arms) * len(OFFSETS) or len(summary_rows) != len(source_summary):
        raise ValueError("Pilot summary has duplicate or missing cells")
    for arm, model in arms.items():
        for k in OFFSETS:
            matching = [pair_id for pair_id, pair in pairs.items() if pair.k == k]
            row = source_summary.get((arm, k), {})
            if (row.get("status") != "COMPLETE" or int(row.get("resolved", -1)) != BOARD_COUNT
                    or int(row.get("planned", -1)) != BOARD_COUNT
                    or int(row.get("hits", -1)) != sum(baseline[model][key] for key in matching)):
                raise ValueError(f"Pilot summary disagrees with its raw trials: {arm}, k={k}")
    return pairs, baseline


def _validate_cohort(pairs: dict[str, Pair]) -> None:
    if len(pairs) != BOARD_COUNT * len(OFFSETS):
        raise ValueError("Expected exactly 120 distinct paired targets")
    contexts: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs.values():
        contexts[pair.context_id].append(pair)
    if len(contexts) != BOARD_COUNT or len({p.layout_id for p in pairs.values()}) != BOARD_COUNT:
        raise ValueError("Expected 20 distinct layouts with one context each")
    for items in contexts.values():
        if (sorted(pair.k for pair in items) != list(OFFSETS)
                or len({pair.layout_id for pair in items}) != 1
                or len({pair.source_hash for pair in items}) != 1):
            raise ValueError("Every context needs one layout, one start and all six offsets")


def load_cllp(path: Path, model: Model, pairs: dict[str, Pair]) -> dict[str, bool]:
    """Check every CLLP trace against the frozen pair identifiers and action cap."""
    results: dict[str, bool] = {}
    for row in read_jsonl(path):
        pair_id = row.get("pair_id")
        pair = pairs.get(pair_id)
        if row.get("model") != model or pair is None or pair_id in results:
            raise ValueError(f"Unexpected, duplicate or wrong-model CLLP trial: {pair_id}")
        for key in ("layout_id", "source_hash", "target_hash", "k"):
            if row.get(key) != getattr(pair, key):
                raise ValueError(f"Mismatched {key} for {pair_id}")
        actions, hashes = row.get("actions"), row.get("state_hashes")
        hit, first_hit = row.get("exact_hit"), row.get("first_hit")
        if (row.get("action_cap") != ACTION_CAP or not isinstance(actions, list)
                or not isinstance(hashes, list) or not isinstance(hit, bool)
                or len(hashes) != len(actions) + 1 or len(actions) > ACTION_CAP
                or not all(isinstance(action, int) and not isinstance(action, bool) and 0 <= action <= 3
                           for action in actions)
                or row.get("attempted_actions") != len(actions)
                or hashes[0] != pair.source_hash):
            raise ValueError(f"Invalid CLLP trace or action budget: {pair_id}")
        no_ops = sum(a == b for a, b in zip(hashes, hashes[1:]))
        if row.get("no_op_count") != no_ops:
            raise ValueError(f"Incorrect no-op count: {pair_id}")
        if hit:
            if (not isinstance(first_hit, int) or isinstance(first_hit, bool) or first_hit != len(actions)
                    or hashes[-1] != pair.target_hash or pair.target_hash in hashes[:-1]
                    or row.get("stop_reason") != "exact_target"):
                raise ValueError(f"Incorrect CLLP first hit: {pair_id}")
        elif (first_hit is not None or pair.target_hash in hashes
              or row.get("stop_reason") not in {"task_terminal", "action_limit"}
              or (row.get("stop_reason") == "action_limit" and len(actions) != ACTION_CAP)):
            raise ValueError(f"Invalid non-hit CLLP outcome: {pair_id}")
        results[pair_id] = hit
    missing = set(pairs) - set(results)
    if missing:
        raise ValueError(f"Missing {len(missing)} {model} trials")
    return results


def validate_cllp_artifacts(path: Path, model: Model, pairs_path: Path) -> list[Path]:
    """Require complete runs and recheck saved pair/checkpoint hashes."""
    status_path = path.parent / "status.json"
    provenance_path = path.parent / "provenance.json"
    status, provenance = read_json(status_path), read_json(provenance_path)
    count = BOARD_COUNT * len(OFFSETS)
    if (status.get("state") != "complete" or status.get("requested") != count
            or status.get("completed") != count):
        raise ValueError(f"Incomplete {model} run")
    if (provenance.get("model") != model or provenance.get("action_cap") != ACTION_CAP
            or provenance.get("token_length") != 291 or "greedy" not in provenance.get("policy", "")):
        raise ValueError(f"Incorrect controller provenance: {model}")
    checkpoint = Path(provenance["checkpoint"])
    hashes = provenance["sha256"]
    for source in (pairs_path.resolve(), checkpoint / "config.json", checkpoint / "pytorch_model.bin"):
        if hashes.get(str(source)) != sha256(source):
            raise ValueError(f"Input hash changed since {model} run: {source}")
    return [status_path, provenance_path]


def summarise(pairs: dict[str, Pair], results: dict[Model, dict[str, bool]]) -> list[dict[str, Any]]:
    rows = []
    for model in Model:
        if set(results[model]) != set(pairs):
            raise ValueError(f"Unpaired results for {model}")
        for k in OFFSETS:
            keys = [key for key, pair in pairs.items() if pair.k == k]
            hits = sum(results[model][key] for key in keys)
            low, high = wilson(hits, len(keys))
            rows.append({"model": model, "k": k, "hits": hits, "boards": len(keys),
                         "success_fraction": hits / len(keys), "wilson_low": low, "wilson_high": high})
    return rows


def paired_differences(
    pairs: dict[str, Pair], results: dict[Model, dict[str, bool]], *, seed: int = 20261115, samples: int = 10000
) -> list[dict[str, Any]]:
    """Paired bootstrap over layouts; one layout contributes one pair per k."""
    rng = random.Random(seed)
    rows = []
    for method in (Model.CLLP_8, Model.CLLP_32):
        for comparator in (Model.LUNA, Model.TERRA):
            for k in OFFSETS:
                keys = sorted(key for key, pair in pairs.items() if pair.k == k)
                values = [int(results[method][key]) - int(results[comparator][key]) for key in keys]
                boots = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
                rows.append({"method": method, "comparator": comparator, "k": k, "paired_boards": len(keys),
                             "difference": sum(values) / len(values), "bootstrap_low": boots[int(.025 * samples)],
                             "bootstrap_high": boots[int(.975 * samples)], "bootstrap_seed": seed,
                             "bootstrap_replicates": samples})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "STIXGeneral"],
                         "font.size": 9, "mathtext.fontset": "stix", "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    styles = {
        Model.LUNA: ("#176b8b", "o", "-", "Luna · 32 examples · sampled"),
        Model.TERRA: ("#c6632e", "s", "--", "Terra · 32 examples · sampled"),
        Model.CLLP_8: ("#287c70", "^", "-.", "CLLP-8 · trained · greedy"),
        Model.CLLP_32: ("#8264a1", "D", ":", "CLLP-32 · trained · greedy"),
    }
    for model, (color, marker, line, label) in styles.items():
        data = [row for row in rows if row["model"] == model]
        x = [row["k"] for row in data]
        y = [row["success_fraction"] for row in data]
        error = [[row["success_fraction"] - row["wilson_low"] for row in data],
                 [row["wilson_high"] - row["success_fraction"] for row in data]]
        ax.errorbar(x, y, yerr=error, color=color, marker=marker, linestyle=line, linewidth=1.4,
                    markersize=4.5, capsize=2.5, elinewidth=.7, label=label)
    ax.set(xlim=(.4, 16.6), ylim=(-.035, 1.07), xticks=OFFSETS,
           xlabel="Steps ahead in the demonstration, k", ylabel="Exact-state reachability")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color="#e0e0e0", linewidth=.6)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name in {"bottom", "left"})
        spine.set_color("#aaaaaa")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.31), ncol=2, frameon=False, fontsize=8.1,
              columnspacing=1.2, handlelength=2.1)
    fig.suptitle("Sokoban 12 × 12 · 10 boxes", y=.97, fontsize=12, fontweight="bold")
    fig.text(.14, .045, "Same 20 layouts; 1 attempt per target; 32-action limit. Error bars: 95% Wilson.\n"
             "Language models sample actions at T=1; CLLPs use greedy actions. Compute differs.",
             ha="left", va="bottom", fontsize=7.6, color="#444444")
    fig.subplots_adjust(left=.14, right=.98, top=.72, bottom=.23)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"figure1_cllp_matched.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "data/icl/reachability_proposal_hard_12_v1/core_pairs.json")
    parser.add_argument("--pilot-trials", type=Path, default=PILOT / "pilot20_export/trials.csv")
    parser.add_argument("--pilot-summary", type=Path, default=PILOT / "pilot20_export/summary_by_k_model.csv")
    parser.add_argument("--pilot-verification", type=Path, default=PILOT / "pilot20_export/verification.json")
    parser.add_argument("--pilot-manifest", type=Path, default=PILOT / "campaign/study_12/manifest.json")
    parser.add_argument("--cllp8", type=Path, required=True, help="CLLP-8 trials.jsonl")
    parser.add_argument("--cllp32", type=Path, required=True, help="CLLP-32 trials.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    pairs, results = load_baseline(args.pairs, args.pilot_trials, args.pilot_summary,
                                   args.pilot_manifest, args.pilot_verification)
    artifact_paths = []
    for model, path in ((Model.CLLP_8, args.cllp8), (Model.CLLP_32, args.cllp32)):
        artifact_paths.extend(validate_cllp_artifacts(path, model, args.pairs))
        results[model] = load_cllp(path, model, pairs)
    summary = summarise(pairs, results)
    diffs = paired_differences(pairs, results)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_differences.csv", diffs)
    paired = [{"pair_id": key, "layout_id": pair.layout_id, "context_id": pair.context_id,
               "source_hash": pair.source_hash, "target_hash": pair.target_hash, "k": pair.k,
               **{str(model): results[model][key] for model in Model}} for key, pair in sorted(pairs.items())]
    write_csv(output / "paired_cases.csv", paired)
    inputs = [args.pairs, args.pilot_trials, args.pilot_summary, args.pilot_manifest, args.pilot_verification,
              args.cllp8, args.cllp32]
    inputs.extend(artifact_paths)
    for row in read_csv(args.pilot_trials):
        if row["family"] == "12":
            trace = Path(row["trace_path"])
            inputs.append(trace if trace.is_absolute() else ROOT / trace)
    metadata = {"schema_version": 1, "boards": BOARD_COUNT, "offsets": OFFSETS, "action_cap": ACTION_CAP,
                "pilot": "32 examples, sampled action distribution at policy temperature 1; one attempt",
                "cllp": "trained full-state controller, greedy argmax, no in-context examples; one attempt",
                "uncertainty": "Pointwise Wilson intervals; paired bootstrap resamples whole layouts.",
                "limitations": ["Different computation and decoding protocols.", "Previously examined cohort.",
                                "k counts recorded demonstration actions, not shortest distance.",
                                "Exact-state targets are not predicate-only intents."],
                "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs]}
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot(summary, output)
    print(f"Exported {len(paired)} paired targets and {len(summary)} cells to {output}")


if __name__ == "__main__":
    main()
