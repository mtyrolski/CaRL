#!/usr/bin/env python3
"""Inspect a frozen training sampler before running a replacement adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

from carl.icl.intent_generator_experiment import SYSTEM
from carl.icl.intent_generator_experiment import board_balanced
from scripts.audit_overnight_stage04_training import recompute_tokens


def sha(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_train(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if any(row.get("split") not in ("train", "validation") for row in rows):
        raise ValueError("probe accepts training and validation splits only")
    training = [row for row in rows if row["split"] == "train"]
    if not training or len({row["record_id"] for row in training}) != len(training):
        raise ValueError("training record IDs must be unique and nonempty")
    return training


def independent_sampler(records: list[dict[str, Any]], seed: int) -> Iterator[dict[str, Any]]:
    """Reconstruct the saved board-balanced schedule without importing its RNG."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["layout_id"])].append(record)
    rng = random.Random(seed)
    remaining: dict[str, list[dict[str, Any]]] = {key: [] for key in groups}
    while groups:
        boards = list(groups)
        rng.shuffle(boards)
        for board in boards:
            if not remaining[board]:
                remaining[board] = rng.sample(groups[board], len(groups[board]))
            yield remaining[board].pop()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_board = Counter(row["layout_id"] for row in records)
    effect_types = Counter(str(row.get("effect_type", "unknown")) for row in records)
    lengths = Counter(str(len(row["intent"]["effects"])) for row in records)
    thresholds = Counter(str(item["count"]) for row in records for item in row["intent"]["effects"]
                         if item["predicate"] == "boxes_on_goals_at_least")
    predicates = Counter(item["predicate"] for row in records for item in row["intent"]["effects"])
    return {"records": len(records), "unique_record_ids": len({row["record_id"] for row in records}),
            "layouts": len(by_board), "per_layout": dict(sorted(by_board.items())),
            "effect_types": dict(sorted(effect_types.items())), "conjunction_sizes": dict(sorted(lengths.items())),
            "goal_thresholds": dict(sorted(thresholds.items())), "effect_predicates": dict(sorted(predicates.items()))}


def compare_counts(a: dict[str, int], b: dict[str, int], n: int) -> dict[str, Any]:
    keys = sorted(set(a) | set(b))
    differences = {key: a.get(key, 0) - b.get(key, 0) for key in keys}
    return {"narrow_minus_wide": differences,
            "half_l1_per_sample": sum(abs(value) for value in differences.values()) / (2 * n)}


def probe(narrow_path: Path, wide_path: Path, *, steps: int, accumulation: int, seed: int,
          reference_curve: Path | None = None, pad_to_length: int = 280,
          max_length: int = 512, tokenizer: Any = None) -> dict[str, Any]:
    if steps < 1 or accumulation < 1 or pad_to_length < 1 or pad_to_length > max_length:
        raise ValueError("steps and accumulation must be positive")
    narrow, wide = read_train(narrow_path), read_train(wide_path)
    if len(narrow) != len(wide):
        raise ValueError("arms must have equal numbers of original training examples")
    n = steps * accumulation
    groups = {}
    sequences = {}
    for name, records in (("narrow", narrow), ("wide", wide)):
        sequence = list(islice(board_balanced(records, seed), n))
        independent = list(islice(independent_sampler(records, seed), n))
        ids = [row["record_id"] for row in sequence]
        if ids != [row["record_id"] for row in independent]:
            raise ValueError("independent reconstruction differs from the training sampler")
        groups[name] = {"pool": summarize(records), "sampled": summarize(sequence),
                        "sequence_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
        if tokenizer is not None:
            pool_tokens = [recompute_tokens(tokenizer, SYSTEM, row, "identity", max_length) for row in records]
            sequence_tokens = [recompute_tokens(tokenizer, SYSTEM, row, "identity", max_length) for row in sequence]
            if any(effective > pad_to_length for effective, _ in pool_tokens):
                raise ValueError("a complete training example exceeds the frozen padding budget")
            groups[name]["tokens"] = {"complete_pool_max": max(effective for effective, _ in pool_tokens),
                                      "sampled_effective_total": sum(effective for effective, _ in sequence_tokens),
                                      "sampled_supervised_total": sum(answer for _, answer in sequence_tokens),
                                      "planned_model_total": n * pad_to_length, "truncated": 0}
        sequences[name] = ids
    if reference_curve is not None:
        with reference_curve.open(encoding="utf-8") as source:
            curve = [json.loads(line) for line in source if line.strip()]
        actual = [sample["record_id"] for update in curve for sample in update["samples"]]
        if actual != sequences["wide"]:
            raise ValueError("simulated wide sequence differs from the completed wide training curve")
        if tokenizer is not None and (sum(update["effective_tokens"] for update in curve)
                                      != groups["wide"]["tokens"]["sampled_effective_total"]
                                      or sum(update["supervised_tokens"] for update in curve)
                                      != groups["wide"]["tokens"]["sampled_supervised_total"]
                                      or sum(update["model_tokens"] for update in curve) != n * pad_to_length):
            raise ValueError("simulated wide token counts differ from the completed training curve")
    else:
        actual = None
    pool_equal = {key: groups["narrow"]["pool"][key] == groups["wide"]["pool"][key]
                  for key in ("effect_types", "conjunction_sizes", "effect_predicates")}
    if not all(pool_equal.values()):
        raise ValueError("original pool predicate/conjunction marginals differ")
    comparisons = {key: compare_counts(groups["narrow"]["sampled"][key], groups["wide"]["sampled"][key], n)
                   for key in ("effect_types", "conjunction_sizes", "goal_thresholds", "effect_predicates")}
    return {"schema_version": 1, "purpose": "training-only deterministic preflight; no evaluation outputs read",
            "seed": seed, "steps": steps, "accumulation": accumulation, "samples_per_arm": n,
            "pad_to_length": pad_to_length, "max_length": max_length, "planned_model_tokens_per_arm": n * pad_to_length,
            "data_sha256": {"narrow": sha(narrow_path), "wide": sha(wide_path)},
            "reference_curve_sha256": sha(reference_curve) if reference_curve else None,
            "wide_sequence_matches_completed_curve": actual is not None,
            "pool_marginals_match": pool_equal,
            "shared_pool_record_ids": len({row["record_id"] for row in narrow} & {row["record_id"] for row in wide}),
            "arms": groups, "sampled_comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--narrow", type=Path, required=True)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--reference-curve", type=Path)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pad-to-length", type=int, default=280)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    result = probe(args.narrow, args.wide, steps=args.steps, accumulation=args.accumulation, seed=args.seed,
                   reference_curve=args.reference_curve, pad_to_length=args.pad_to_length,
                   max_length=args.max_length, tokenizer=tokenizer)
    result["tokenizer_sha256"] = sha(args.model / "tokenizer.json")
    result["probe_source_sha256"] = sha(Path(__file__))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"state": "complete", "output": str(output), "comparisons": result["sampled_comparisons"]}),
          flush=True)


if __name__ == "__main__":
    main()
