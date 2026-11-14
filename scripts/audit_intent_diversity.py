#!/usr/bin/env python3
"""Audit repeated intent proposals against fixed ICL support examples.

This descriptive, post-hoc audit reads only completed prediction files. Exact
matching is measured after canonicalising conjunction order; it does not by
itself establish why a model produced a particular intent.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from datetime import timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from carl.icl.intent_data import validate_prediction
from carl.icl.intent_fence_audit import remove_one_outer_fence
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_generator_experiment import read_records
from carl.icl.intent_representation_experiment import SupportSet
from carl.icl.intent_representation_experiment import load_supports
from carl.icl.intent_representation_experiment import messages_for
from carl.icl.intent_representation_experiment import parse_representation


class Mode(StrEnum):
    STRICT = "strict"
    POSTHOC_FENCE = "posthoc_fence"


def canonical(intent: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Keep all predicates but ignore their order within each conjunction."""
    normalised = {group: sorted(intent[group], key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
                  for group in ("effects", "preserve")}
    return json.dumps(normalised, ensure_ascii=False, sort_keys=True, separators=(",", ":")), normalised


def parse_one(prediction: dict[str, Any], context: dict[str, Any], mode: Mode,
              support_signatures: list[str], shown_support_signatures: list[str] | None = None) -> dict[str, Any]:
    """Reparse original output and separately evaluate its normalized form."""
    raw, representation = prediction["raw_output"], prediction["representation"]
    if not isinstance(raw, str):
        raise ValueError("Prediction has no raw response")
    try:
        strict = parse_representation(representation, raw)
    except ValueError:
        strict = None
    if prediction.get("syntax_valid") is not (strict is not None) or prediction.get("parsed_intent") != strict:
        raise ValueError("Saved strict parse differs from original output")
    text, removed = (remove_one_outer_fence(raw) if mode == Mode.POSTHOC_FENCE else (raw, False))
    try:
        intent = parse_representation(representation, text)
    except ValueError:
        intent = None
    result: dict[str, Any] = {
        "record_id": context["record_id"], "layout_id": context["layout_id"],
        "family": context.get("evaluation_family") or context.get("family"),
        "previously_evaluated_in_lora_matrix": context.get("previously_evaluated_in_lora_matrix"),
        "mode": mode, "fence_removed": removed, "parseable": intent is not None,
        "semantic_valid": None, "validation_errors": None, "canonical_signature": None,
        "canonical_intent": None, "matches_balanced0_support": [],
        "matches_shown_support": [], "matches_shown_support_any": False if shown_support_signatures else None,
    }
    if intent is not None:
        key, value = canonical(intent)
        errors = validate_prediction(intent, context["start_observation"])
        result.update(semantic_valid=not errors, validation_errors=errors, canonical_signature=key,
                      canonical_intent=value,
                      matches_balanced0_support=[index for index, signature in enumerate(support_signatures)
                                                 if signature == key])
        if shown_support_signatures:
            matches = [index for index, signature in enumerate(shown_support_signatures) if signature == key]
            result.update(matches_shown_support=matches, matches_shown_support_any=bool(matches))
    return result


def summaries(variant: str, support_set: str, rows: list[dict[str, Any]], support_count: int) -> tuple[
        dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Describe frequencies on every observed board, including invalid intents."""
    parsed = [row for row in rows if row["parseable"]]
    valid = [row for row in parsed if row["semantic_valid"] is True]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in parsed:
        groups.setdefault(row["canonical_signature"], []).append(row)
    frequency = []
    for key, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        valid_members = [row for row in members if row["semantic_valid"] is True]
        family_counts = Counter(str(row["family"]) for row in members)
        family_valid = Counter(str(row["family"]) for row in valid_members)
        reasons = Counter(error for row in members for error in (row["validation_errors"] or []))
        frequency.append({"variant": variant, "mode": rows[0]["mode"], "support_set": support_set,
                          "canonical_intent": members[0]["canonical_intent"], "canonical_signature": key,
                          "occurrences": len(members), "independent_boards": len({r["layout_id"] for r in members}),
                          "valid_on_board": len(valid_members), "invalid_on_board": len(members) - len(valid_members),
                          "matches_balanced0_support": members[0]["matches_balanced0_support"],
                          "matches_shown_support": members[0]["matches_shown_support"],
                          "family_counts": dict(family_counts), "valid_by_family": dict(family_valid),
                          "validation_error_counts": dict(reasons)})
    matches = []
    for index in range(support_count):
        subset = [row for row in parsed if index in row["matches_balanced0_support"]]
        for family in ("all", *sorted({str(row["family"]) for row in rows})):
            selected = [row for row in subset if family == "all" or row["family"] == family]
            family_output = [row for row in rows if family == "all" or row["family"] == family]
            family_parsed = [row for row in parsed if family == "all" or row["family"] == family]
            good = sum(row["semantic_valid"] is True for row in selected)
            matches.append({"variant": variant, "mode": rows[0]["mode"], "actual_support_set": support_set,
                            "balanced0_was_shown": support_set == SupportSet.BALANCED0,
                            "support_index": index, "family": family, "outputs": len(family_output),
                            "parseable": len(family_parsed), "matches": len(selected), "valid_on_board": good,
                            "invalid_on_board": len(selected) - good})
    match_rows = [row for row in parsed if row["matches_balanced0_support"]]
    summary = {"variant": variant, "mode": rows[0]["mode"], "support_set": support_set,
               "balanced0_was_shown": support_set == SupportSet.BALANCED0,
               "outputs": len(rows), "independent_boards": len({row["layout_id"] for row in rows}),
               "parseable": len(parsed), "valid": len(valid), "invalid_format": len(rows) - len(parsed),
               "invalid_on_board": len(parsed) - len(valid), "distinct_intents": len(groups),
               "fences_removed": sum(row["fence_removed"] for row in rows),
               "match_support_any": len(match_rows),
               "match_support_any_valid": sum(row["semantic_valid"] is True for row in match_rows),
               "mode_occurrences": frequency[0]["occurrences"] if frequency else 0,
               "mode_valid_on_board": frequency[0]["valid_on_board"] if frequency else 0}
    for index in range(support_count):
        summary[f"match_support_{index}"] = sum(index in row["matches_balanced0_support"] for row in parsed)
    return summary, frequency, matches


def shown_summaries(variant: str, support_set: str, rows: list[dict[str, Any]],
                    shown: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Count exact matches with the demonstrations actually supplied to this method."""
    if not shown:
        return {"shown_support_count": 0, "shown_support_match_any": None,
                "shown_support_match_any_valid": None}, []
    matched = [row for row in rows if row["matches_shown_support_any"] is True]
    summary = {"shown_support_count": len(shown), "shown_support_match_any": len(matched),
               "shown_support_match_any_valid": sum(row["semantic_valid"] is True for row in matched)}
    output = []
    for index, example in enumerate(shown):
        for family in ("all", *sorted({str(row["family"]) for row in rows})):
            family_rows = [row for row in rows if family == "all" or row["family"] == family]
            hits = [row for row in family_rows if index in row["matches_shown_support"]]
            valid = sum(row["semantic_valid"] is True for row in hits)
            output.append({"variant": variant, "mode": rows[0]["mode"], "actual_support_set": support_set,
                           "support_index": index, "support_record_id": example["record_id"],
                           "support_canonical_signature": example["canonical_signature"], "family": family,
                           "outputs": len(family_rows), "parseable": sum(row["parseable"] for row in family_rows),
                           "matches": len(hits), "valid_on_board": valid, "invalid_on_board": len(hits) - valid})
    return summary, output


def add_balanced0_visibility(matches: list[dict[str, Any]], balanced0: list[dict[str, Any]],
                             shown: list[dict[str, Any]]) -> None:
    """Separate the full-set label from whether an individual example or intent was shown."""
    identifiers = {item["record_id"] for item in shown}
    signatures = {item["canonical_signature"] for item in shown}
    for match in matches:
        example = balanced0[match["support_index"]]
        match.update(balanced0_support_record_id=example["record_id"],
                     balanced0_support_example_was_shown=example["record_id"] in identifiers,
                     balanced0_support_intent_was_shown=example["canonical_signature"] in signatures)


def verify_prompts(path: Path, expected_sha256: str, ids: list[str], contexts: dict[str, dict[str, Any]],
                   support: list[dict[str, Any]], representation: str) -> None:
    """Check the actual saved messages, not only the declared support-set name."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256 or not content.endswith(b"\n"):
        raise ValueError("Saved prompts differ from generation metadata or have an incomplete line")
    prompts = [json.loads(line) for line in content.splitlines() if line.strip()]
    if [row.get("record_id") for row in prompts] != ids:
        raise ValueError("Saved prompts differ from the frozen context order")
    for row in prompts:
        record = contexts[row["record_id"]]
        if (row.get("layout_id") != record["layout_id"]
                or row.get("messages") != messages_for(record, support, representation)):
            raise ValueError("Saved prompt messages differ from the specified support examples or board")


def verify_run(plan: dict[str, Any], job: dict[str, Any], meta: dict[str, Any], prompt_path: Path) -> None:
    """Check model, exact representation, source hashes and decoding settings."""
    expected = {"variant": job["name"], "representation": job["representation"], "support_set": job["support_set"],
                "model": plan["model"]["path"], "revision": plan["model"]["revision"],
                "model_weights_sha256": plan["model"]["sha256"]["model.safetensors"],
                "model_config_sha256": plan["model"]["sha256"]["config.json"],
                "tokenizer_sha256": plan["model"]["sha256"]["tokenizer.json"],
                "data_sha256": plan["inputs"]["data"]["sha256"],
                "test_ids_sha256": plan["inputs"]["test_ids"]["sha256"],
                "support_data_sha256": plan["inputs"]["support_data"]["sha256"],
                "support_manifest_sha256": plan["inputs"]["support_manifest"]["sha256"],
                "protocol_sha256": plan["inputs"]["protocol"]["sha256"],
                "generator_sha256": plan["inputs"]["generator"]["sha256"],
                "support_selection_sha256": plan["inputs"]["support_selector"]["sha256"], **plan["settings"]}
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved generation settings differ from the frozen matrix")
    if not prompt_path.is_file() or meta.get("prompts_sha256") != file_sha256(prompt_path):
        raise ValueError("Saved prompts differ from the generation metadata")
    adapter = job.get("adapter")
    if meta.get("adapter") != adapter:
        raise ValueError("Saved generation used a different adapter")
    if adapter is None:
        expected_hashes = None
    elif job.get("adapter_pending"):
        recorded = Path(plan["run"]) / "matrix_adapter_hashes" / f"{job['name']}.json"
        bound = json.loads(recorded.read_text(encoding="utf-8"))
        if bound.get("adapter") != adapter:
            raise ValueError("Pending adapter path differs from its frozen hash")
        expected_hashes = bound["sha256"]
    else:
        expected_hashes = job["adapter_sha256"]
    for output_key, filename in (("adapter_weights_sha256", "adapter_model.safetensors"),
                                 ("adapter_config_sha256", "adapter_config.json")):
        if meta.get(output_key) != (expected_hashes[filename] if expected_hashes else None):
            raise ValueError("Saved generation used different adapter weights or configuration")


def audit(run: Path) -> dict[str, Any]:
    """Snapshot all completed methods and compute strict/post-hoc distributions."""
    started_utc = datetime.now(timezone.utc).isoformat()
    run = run.resolve()
    plan_path = run / "matrix_plan.json"
    checksum = plan_path.with_suffix(".sha256")
    if checksum.read_text(encoding="utf-8").strip() != file_sha256(plan_path):
        raise ValueError("Frozen matrix plan checksum differs")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("run") != str(run) or plan.get("schema_version") != 2:
        raise ValueError("Unexpected matrix run or schema")
    for name in ("data", "test_ids", "support_data", "support_manifest", "sampling_plan", "independent_audit",
                 "protocol", "generator", "support_selector", "intent_checker"):
        item = plan["inputs"][name]
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    independent = json.loads(Path(plan["inputs"]["independent_audit"]["path"]).read_text(encoding="utf-8"))
    if (independent.get("status") != "passed"
            or independent.get("dataset_sha256") != plan["inputs"]["data"]["sha256"]
            or independent.get("test_ids_sha256") != plan["inputs"]["test_ids"]["sha256"]
            or independent["symmetry"]["independent_d4_classes_per_split"]["test"] != 128
            or independent["symmetry"]["duplicate_test_layouts"] != 0
            or independent["symmetry"]["train_validation_test_pairwise_overlaps"] != 0):
        raise ValueError("Independent audit does not establish test-layout separation")
    data = read_records(plan["inputs"]["data"]["path"], "test")
    by_id = {row["record_id"]: row for row in data}
    frozen = json.loads(Path(plan["inputs"]["test_ids"]["path"]).read_text(encoding="utf-8"))
    ids = frozen["record_ids"]
    if (len(ids) != 128 or len(set(ids)) != len(ids) or len(by_id) != len(data)
            or any(identifier not in by_id for identifier in ids)
            or len({by_id[identifier]["layout_id"] for identifier in ids}) != len(ids)):
        raise ValueError("Expected 128 distinct frozen test boards")
    support_data = Path(plan["inputs"]["support_data"]["path"])
    support_manifest = Path(plan["inputs"]["support_manifest"]["path"])
    banks = {name: load_supports(support_data, support_manifest, name) for name in SupportSet}
    support = banks[SupportSet.BALANCED0]
    if len(support) != 2:
        raise ValueError("Expected exactly two balanced0 support examples")
    support_signatures = [canonical(row["intent"])[0] for row in support]
    if len(set(support_signatures)) != len(support_signatures):
        raise ValueError("The two support intents are semantically identical")
    support_info = [{"index": index, "record_id": row["record_id"], "layout_id": row["layout_id"],
                     "intent": row["intent"], "canonical_signature": support_signatures[index]}
                    for index, row in enumerate(support)]
    support_sets = {name.value: [{"index": index, "record_id": row["record_id"], "layout_id": row["layout_id"],
                                  "intent": row["intent"], "canonical_signature": canonical(row["intent"])[0]}
                                 for index, row in enumerate(bank)] for name, bank in banks.items()}
    summaries_out, frequencies_out, matches_out, shown_matches_out, cases_out, excluded, sources = [], [], [], [], [], [], {}
    for job in plan["methods"]:
        name = job["name"]
        path = Path(job["output"])
        summary_path, metadata_path = path.with_suffix(".summary.json"), path.with_suffix(".run.json")
        if not summary_path.is_file():
            excluded.append({"variant": name, "reason": "missing_or_in_progress_summary"})
            continue
        try:
            method_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if method_summary.get("status") != "complete" or method_summary.get("n") != len(ids):
                raise ValueError("Method is not complete on all frozen contexts")
            content = path.read_bytes()
            if not content.endswith(b"\n"):
                raise ValueError("Completed predictions end in an incomplete line")
            predictions = [json.loads(line) for line in content.splitlines() if line.strip()]
            if [row.get("record_id") for row in predictions] != ids:
                raise ValueError("Completed predictions have a different frozen order")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            prompt_path = path.with_suffix(".prompts.jsonl")
            verify_run(plan, job, metadata, prompt_path)
            selected_support = banks[SupportSet(job["support_set"])]
            shown_info = support_sets[job["support_set"]]
            shown_signatures = [example["canonical_signature"] for example in shown_info]
            verify_prompts(prompt_path, metadata["prompts_sha256"], ids, by_id,
                           selected_support, job["representation"])
            method_cases = []
            for prediction in predictions:
                record = by_id[prediction["record_id"]]
                if (prediction.get("layout_id") != record["layout_id"] or prediction.get("variant") != name
                        or prediction.get("split") != "test" or prediction.get("representation") != job["representation"]
                        or prediction.get("support_set") != job["support_set"]):
                    raise ValueError("Prediction row differs from the frozen method")
                method_cases.extend({"variant": name, **parse_one(prediction, record, mode, support_signatures,
                                                                  shown_signatures)}
                                    for mode in Mode)
            method_summaries, method_frequencies, method_matches, method_shown_matches = [], [], [], []
            for mode in Mode:
                selected = [row for row in method_cases if row["mode"] == mode]
                summary, frequencies, matches = summaries(name, job["support_set"], selected, len(support))
                shown_summary, shown_matches = shown_summaries(name, job["support_set"], selected, shown_info)
                add_balanced0_visibility(matches, support_info, shown_info)
                method_summaries.append({**summary, **shown_summary})
                method_frequencies.extend(frequencies)
                method_matches.extend(matches)
                method_shown_matches.extend(shown_matches)
            source = {"predictions_sha256": hashlib.sha256(content).hexdigest(),
                      "run_sha256": file_sha256(metadata_path), "summary_sha256": file_sha256(summary_path),
                      "prompts_sha256": file_sha256(prompt_path)}
            summaries_out.extend(method_summaries)
            frequencies_out.extend(method_frequencies)
            matches_out.extend(method_matches)
            shown_matches_out.extend(method_shown_matches)
            cases_out.extend(method_cases)
            sources[name] = source
        except (OSError, ValueError, KeyError, TypeError) as error:
            excluded.append({"variant": name, "reason": f"{type(error).__name__}: {error}"})
    return {
        "schema_version": 1, "started_utc": started_utc,
        "scope": "descriptive post-hoc snapshot of complete 128-board methods",
        "matrix_plan_sha256": file_sha256(plan_path), "dataset_sha256": plan["inputs"]["data"]["sha256"],
        "test_ids_sha256": plan["inputs"]["test_ids"]["sha256"],
        "independent_audit_sha256": plan["inputs"]["independent_audit"]["sha256"],
        "sampling_plan_sha256": plan["inputs"]["sampling_plan"]["sha256"],
        "intent_checker_sha256": plan["inputs"]["intent_checker"]["sha256"],
        "diversity_script_sha256": file_sha256(Path(__file__)),
        "fence_normalisation_sha256": file_sha256(Path(remove_one_outer_fence.__code__.co_filename)),
        "support_manifest_sha256": plan["inputs"]["support_manifest"]["sha256"],
        "support_data_sha256": plan["inputs"]["support_data"]["sha256"],
        "support_examples": support_info, "shown_support_sets": support_sets, "sources": sources, "excluded_methods": excluded,
        "summaries": summaries_out, "frequencies": frequencies_out, "support_matches": matches_out,
        "shown_support_matches": shown_matches_out,
        "records": cases_out,
        "interpretation": ["Strict and one-fence-removal results are distinct protocols; the latter is post-hoc.",
                           "Canonical matching compares all effect and preservation predicates, ignoring array order.",
                           "Matches to balanced0 are observed identity, not proof of causal copying; methods with another "
                           "support set did not necessarily see these demonstrations.",
                           "balanced0_was_shown identifies the complete support set; per-example fields identify shared examples. "
                           "Shown-support matches refer to the individually verified prompt messages; zero-shot has no such rate.",
                           "For each parsed intent, validity is checked independently against the current board only; "
                           "it does not establish controller achievement or bounded feasibility.",
                           "All counted units are independent canonical layouts; unavailable or partial methods are excluded, "
                           "not scored as failures."]}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def write_report(destination: Path, result: dict[str, Any]) -> None:
    """Create a new immutable snapshot; never overwrite an earlier audit."""
    if destination.exists():
        raise FileExistsError("Choose a new directory for each diversity snapshot")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        summary = {key: value for key, value in result.items() if key != "records"}
        (temporary / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temporary / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                                         for row in result["records"]), encoding="utf-8")
        write_csv(temporary / "summary.csv", result["summaries"])
        write_csv(temporary / "frequencies.csv", result["frequencies"])
        write_csv(temporary / "support_matches.csv", result["support_matches"])
        write_csv(temporary / "shown_support_matches.csv", result["shown_support_matches"])
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output or run / "diversity_audit"
    result = audit(run)
    write_report(output, result)
    print(json.dumps({"output": str(output), "completed_methods": len(result["sources"]),
                      "excluded_methods": len(result["excluded_methods"])}))


if __name__ == "__main__":
    main()
