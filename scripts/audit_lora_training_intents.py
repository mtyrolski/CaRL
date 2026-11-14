#!/usr/bin/env python3
"""Describe LoRA output diversity and exact matches to checked training labels.

This post-hoc audit uses current test boards, saved model outputs and training
labels. Matching an intent to a training label does not establish why it was
generated, nor whether the intent can be achieved by a controller.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Any

from carl.icl.intent_generator_experiment import board_balanced
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records
from scripts.audit_intent_diversity import Mode
from scripts.audit_intent_diversity import audit as diversity_audit
from scripts.audit_intent_diversity import canonical
from scripts.audit_intent_diversity import write_csv
from scripts.audit_matched_representation_training import training_inputs

ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIONS = ("json", "formal", "natural")
Record = dict[str, Any]


def repository_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def label_index(records: list[Record]) -> dict[str, list[Record]]:
    """Index the complete conjunction, retaining every distinct training record."""
    result: dict[str, list[Record]] = defaultdict(list)
    for row in records:
        if (row.get("split") != "train" or row.get("validation", {}).get("accepted") is not True
                or row.get("intent", {}).get("instruction") is not None):
            raise ValueError("Only checked training labels with null instructions are eligible")
        result[canonical(row["intent"])[0]].append(row)
    return dict(result)


def enrich(row: Record, context: Record, pool: dict[str, list[Record]], used: dict[str, list[Record]]) -> Record:
    """Compare a saved proposal with labels, using no later test observation."""
    board = context["start_observation"]
    boxes = sum(line.count("$") + line.count("*") for line in board)
    if len(board) != 12 or any(len(line) != 12 for line in board) or boxes not in (4, 10):
        raise ValueError("Expected a 12x12 test board with 4 or 10 boxes")
    if row["layout_id"] != context["layout_id"] or row["record_id"] != context["record_id"]:
        raise ValueError("Proposal and frozen current board differ")
    signature = row["canonical_signature"]
    candidates = pool.get(signature, []) if signature is not None else []
    sampled = used.get(signature, []) if signature is not None else []
    return {"variant": row["variant"], "mode": row["mode"], "record_id": row["record_id"],
            "layout_id": row["layout_id"], "family": row["family"], "box_count": boxes,
            "parseable": row["parseable"], "semantic_valid": row["semantic_valid"],
            "validation_errors": row["validation_errors"], "canonical_signature": signature,
            "canonical_intent": row["canonical_intent"],
            "training_pool_match": bool(candidates) if signature is not None else None,
            "reconstructed_sample_match": bool(sampled) if signature is not None else None,
            "matching_train_record_ids": sorted(item["record_id"] for item in candidates),
            "matching_train_layout_ids": sorted({item["layout_id"] for item in candidates})}


def describe(variant: str, mode: str, group_type: str, group: str,
             rows: list[Record]) -> tuple[Record, list[Record]]:
    """Retain denominators and current-board validity for each exact intent."""
    parsed = [row for row in rows if row["parseable"]]
    groups: dict[str, list[Record]] = defaultdict(list)
    for row in parsed:
        groups[row["canonical_signature"]].append(row)
    frequencies = []
    for signature, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        first = members[0]
        valid = sum(row["semantic_valid"] is True for row in members)
        reasons = Counter(error for row in members for error in (row["validation_errors"] or []))
        frequencies.append({"variant": variant, "mode": mode, "group_type": group_type, "group": group,
                            "canonical_intent": first["canonical_intent"], "canonical_signature": signature,
                            "occurrences": len(members), "independent_boards": len({r["layout_id"] for r in members}),
                            "valid_on_board": valid, "invalid_on_board": len(members) - valid,
                            "training_pool_match": first["training_pool_match"],
                            "reconstructed_sample_match": first["reconstructed_sample_match"],
                            "matching_train_record_ids": first["matching_train_record_ids"],
                            "matching_train_layout_ids": first["matching_train_layout_ids"],
                            "validation_error_counts": dict(reasons)})
    matched = [row for row in parsed if row["training_pool_match"] is True]
    sampled = [row for row in parsed if row["reconstructed_sample_match"] is True]
    summary = {"variant": variant, "mode": mode, "group_type": group_type, "group": group,
               "outputs": len(rows), "independent_boards": len({row["layout_id"] for row in rows}),
               "parseable": len(parsed), "invalid_format": len(rows) - len(parsed),
               "valid_on_board": sum(row["semantic_valid"] is True for row in parsed),
               "invalid_on_board": sum(row["semantic_valid"] is False for row in parsed),
               "distinct_intents": len(groups), "training_pool_match": len(matched),
               "training_pool_match_valid_on_board": sum(row["semantic_valid"] is True for row in matched),
               "reconstructed_sample_match": len(sampled),
               "reconstructed_sample_match_valid_on_board": sum(row["semantic_valid"] is True for row in sampled),
               "most_frequent_occurrences": frequencies[0]["occurrences"] if frequencies else 0}
    return summary, frequencies


def audit(run: Path) -> tuple[Record, list[Record], list[Record], list[Path]]:
    """Validate the matched training and complete generator outputs before counting."""
    run = run.resolve()
    matrix_path = run / "matrix_plan.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    data_path = Path(matrix["inputs"]["data"]["path"])
    training_path = Path(matrix["inputs"]["support_data"]["path"])
    test = read_records(data_path, "test")
    contexts = {row["record_id"]: {key: row[key] for key in ("record_id", "layout_id", "start_observation")}
                for row in test}
    source = [row for row in read_records(training_path) if row.get("split") in ("train", "validation")]
    matched_path = run / "matched_training_audit.json"
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    if (matched.get("status") != "passed" or matched.get("primary", {}).get("status") != "passed"
            or matched.get("source_dataset_sha256") != file_sha256(training_path)
            or matched.get("test_dataset_sha256") != file_sha256(data_path)):
        raise ValueError("Matched training audit does not bind these datasets")
    prior = diversity_audit(run)
    required = {f"rep_lora_{name}_zero" for name in REPRESENTATIONS}
    if not required <= set(prior["sources"]):
        raise ValueError("All three matched LoRA variants must have completed and verified predictions")
    paths = [Path(__file__).resolve(), ROOT / "scripts/audit_intent_diversity.py",
             ROOT / "scripts/audit_matched_representation_training.py",
             ROOT / "carl/icl/intent_generator_experiment.py", ROOT / "carl/icl/intent_representation_training.py",
             ROOT / "carl/icl/intent_representation_experiment.py", matrix_path, matrix_path.with_suffix(".sha256"),
             training_path, data_path, Path(matrix["inputs"]["test_ids"]["path"]), matched_path]
    expected_sequence = matched["primary"]["reconstructed_first_400_train_ids"]
    training = None
    sample_ids = None
    for name in REPRESENTATIONS:
        directory = run / f"adapter_repr_{name}_seed1729"
        config_path, status_path = directory / "configuration.json", directory / "status.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        train, _ = training_inputs(config, source, list(contexts.values()))
        if (config.get("representation") != name or config.get("data_sha256") != file_sha256(training_path)
                or config.get("seed") != 1729 or config.get("max_steps") != 100
                or config.get("accumulation") != 4 or status.get("status") != "finished"
                or status.get("completed_steps") != 100
                or not isinstance(status.get("last_adapter"), str)
                or repository_path(status["last_adapter"]) != (directory / "checkpoint-last-100").resolve()):
            raise ValueError(f"Matched LoRA training is incomplete or has different settings: {name}")
        for key, path in (("training_code_sha256", ROOT / "carl/icl/intent_representation_training.py"),
                          ("initialiser_code_sha256", ROOT / "carl/icl/intent_generator_experiment.py"),
                          ("protocol_sha256", ROOT / "carl/icl/intent_representation_experiment.py")):
            if config.get(key) != file_sha256(path):
                raise ValueError(f"Training source differs from the recorded configuration: {key}")
        reconstructed = [row["record_id"] for row in islice(board_balanced(train, config["seed"]), 400)]
        if reconstructed != expected_sequence or (training is not None and train != training):
            raise ValueError("Training inputs or reconstructed sample order differ between variants")
        training, sample_ids = train, set(reconstructed)
        paths.extend((config_path, status_path))
        jobs = [item for item in matrix["methods"] if item["name"] == f"rep_lora_{name}_zero"]
        if (len(jobs) != 1 or jobs[0].get("representation") != name or jobs[0].get("support_set") != "zero"
                or not isinstance(jobs[0].get("adapter"), str)
                or repository_path(jobs[0]["adapter"]) != (directory / "checkpoint-last-100").resolve()):
            raise ValueError(f"Generator output does not use the matching zero-shot LoRA checkpoint: {name}")
        prediction = Path(jobs[0]["output"])
        paths.extend((prediction, prediction.with_suffix(".run.json"), prediction.with_suffix(".summary.json"),
                      prediction.with_suffix(".prompts.jsonl")))
        source_hashes = prior["sources"][jobs[0]["name"]]
        for field, path in (("predictions_sha256", prediction), ("run_sha256", prediction.with_suffix(".run.json")),
                            ("summary_sha256", prediction.with_suffix(".summary.json")),
                            ("prompts_sha256", prediction.with_suffix(".prompts.jsonl"))):
            if source_hashes[field] != file_sha256(path):
                raise ValueError(f"Generator input changed during the audit: {name}/{field}")
    if training is None or sample_ids is None:
        raise ValueError("No verified training labels")
    pool = label_index(training)
    sampled = label_index([row for row in training if row["record_id"] in sample_ids])
    records = [enrich(row, contexts[row["record_id"]], pool, sampled)
               for row in prior["records"] if row["variant"] in required]
    summaries, frequencies = [], []
    for name in REPRESENTATIONS:
        variant = f"rep_lora_{name}_zero"
        for mode in Mode:
            selected = [row for row in records if row["variant"] == variant and row["mode"] == mode]
            if len(selected) != 128 or len({row["layout_id"] for row in selected}) != 128:
                raise ValueError(f"Expected 128 independent outputs: {variant}/{mode}")
            cohorts = [("all", "all", selected)]
            cohorts.extend(("box_count", f"12x12x{count}", [row for row in selected if row["box_count"] == count])
                           for count in (4, 10))
            cohorts.extend(("family", family, [row for row in selected if row["family"] == family])
                           for family in sorted({row["family"] for row in selected}))
            for group_type, group, rows in cohorts:
                summary, counts = describe(variant, mode, group_type, group, rows)
                summaries.append(summary)
                frequencies.extend(counts)
    hashes = {str(path.resolve()): file_sha256(path) for path in paths}
    metadata = {"schema_version": 1, "analysis_kind": "posthoc_descriptive", "summaries": summaries,
                "training_pool": {"records": len(training), "layouts": len({row["layout_id"] for row in training}),
                                  "distinct_labels": len(pool), "reconstructed_sample_count": len(expected_sequence),
                                  "reconstructed_unique_records": len(sample_ids),
                                  "reconstructed_distinct_labels": len(sampled)},
                "interpretation": ["This is an exploratory description; exact label identity does not establish cause.",
                                   "Training-pool and reconstructed-sample matches are distinct from ICL: all three "
                                   "methods use zero demonstrations, while their system prompts contain syntax examples.",
                                   "The 400 sample IDs are reconstructed from the saved seed and sampler, not a logged step trace.",
                                   "Validity uses the current board and does not establish controller achievement or feasibility.",
                                   "x10 combines two separately reported source families; family and box count are confounded.",
                                   "Strict and one-fence-removal results are separate; no future test state enters these counts."],
                "verified_generator_sources": {name: prior["sources"][name] for name in sorted(required)},
                "input_sha256": hashes}
    return metadata, frequencies, records, paths


def write_output(output: Path, summary: Record, frequencies: list[Record], records: list[Record], paths: list[Path]) -> None:
    if output.exists():
        raise FileExistsError("Choose a fresh output directory for each audit")
    if {str(path.resolve()): file_sha256(path) for path in paths} != summary["input_sha256"]:
        raise ValueError("An audit input changed while it was being read")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        (temporary / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temporary / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                                                 encoding="utf-8")
        write_csv(temporary / "summary.csv", summary["summaries"])
        write_csv(temporary / "frequencies.csv", frequencies)
        provenance = {"schema_version": 1, "analysis_kind": "posthoc_descriptive", "working_directory": str(Path.cwd()),
                      "command_argv": [sys.executable, *sys.argv], "input_sha256": summary["input_sha256"],
                      "output_sha256": {path.name: file_sha256(path) for path in sorted(temporary.iterdir())}}
        (temporary / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output or run / "lora_training_intent_audit"
    summary, frequencies, records, paths = audit(run)
    write_output(output, summary, frequencies, records, paths)
    print(json.dumps({"output": str(output), "summaries": len(summary["summaries"]), "records": len(records)}))


if __name__ == "__main__":
    main()
