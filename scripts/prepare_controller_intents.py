#!/usr/bin/env python3
"""Prepare fixed-policy inputs from generated intents without leaking demonstrations."""

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from carl.icl.intent_data import validate_prediction
from carl.icl.intent_generator_experiment import file_sha256
from carl.icl.intent_representation_experiment import parse_representation
from scripts.run_representation_matrix import verify_plan


def verify_generator_config(metadata: dict[str, Any], plan: dict[str, Any], variant: str,
                            predictions: Path) -> None:
    """Bind generated inputs to the whole frozen matrix, not just board IDs."""
    jobs = [job for job in plan["methods"] if job["name"] == variant]
    if len(jobs) != 1:
        raise ValueError("Variant is not unique in the frozen matrix")
    job = jobs[0]
    if Path(job["output"]).resolve() != predictions.resolve():
        raise ValueError("Prediction path differs from the frozen matrix")
    expected = {"variant": variant, "representation": job["representation"], "support_set": job["support_set"],
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
                "support_selection_sha256": plan["inputs"]["support_selector"]["sha256"],
                "adapter": job["adapter"], **plan["settings"]}
    if job["adapter"] is None:
        expected.update(adapter_weights_sha256=None, adapter_config_sha256=None)
    elif not job["adapter_pending"]:
        expected.update(adapter_weights_sha256=job["adapter_sha256"]["adapter_model.safetensors"],
                        adapter_config_sha256=job["adapter_sha256"]["adapter_config.json"])
    else:
        frozen = Path(plan["run"]) / "matrix_adapter_hashes" / f"{variant}.json"
        adapter = json.loads(frozen.read_text(encoding="utf-8"))
        if adapter["adapter"] != job["adapter"]:
            raise ValueError("Frozen adapter path differs from the matrix")
        expected.update(adapter_weights_sha256=adapter["sha256"]["adapter_model.safetensors"],
                        adapter_config_sha256=adapter["sha256"]["adapter_config.json"])
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Generator configuration differs from the frozen matrix")
    prompt_file = predictions.with_suffix(".prompts.jsonl")
    if metadata.get("prompts_sha256") != file_sha256(prompt_file):
        raise ValueError("The saved generator prompts differ from their manifest")


def prepare(records: list[dict[str, Any]], ids: list[str], predictions: list[dict[str, Any]],
            variant: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test = [r for r in records if r.get("split") == "test"]
    by_id = {r["record_id"]: r for r in test}
    if len(by_id) != len(test):
        raise ValueError("Dataset test IDs must be unique")
    if len(ids) != len(set(ids)) or any(identifier not in by_id for identifier in ids):
        raise ValueError("Frozen test IDs must be unique and present")
    outputs: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        identifier = prediction.get("record_id")
        if (not isinstance(identifier, str) or identifier not in by_id or identifier in outputs
                or prediction.get("variant") != variant
                or prediction.get("layout_id") != by_id[identifier]["layout_id"]):
            raise ValueError("Prediction has duplicate, foreign or mismatched provenance")
        outputs[identifier] = prediction
    controller_records, decisions = [], []
    for identifier in ids:
        source = by_id[identifier]
        prediction = outputs.get(identifier)
        item = {"record_id": identifier, "layout_id": source["layout_id"],
                "family": source.get("evaluation_family", source.get("family")), "variant": variant}
        if prediction is None:
            decisions.append({**item, "status": "missing_generation", "errors": []})
            continue
        try:
            intent = parse_representation(prediction["representation"], prediction["raw_output"])
        except (KeyError, ValueError) as error:
            if prediction.get("syntax_valid") is not False or prediction.get("parsed_intent") is not None:
                raise ValueError("Recorded parse status differs from the raw response") from error
            decisions.append({**item, "status": "invalid_representation", "errors": [str(error)]})
            continue
        if prediction.get("syntax_valid") is not True or prediction.get("parsed_intent") != intent:
            raise ValueError("Recorded intent differs from independent parsing of the raw response")
        errors = validate_prediction(intent, source["start_observation"])
        if errors:
            decisions.append({**item, "status": "invalid_intent", "errors": errors})
            continue
        controller_records.append({**item, "split": "test", "start_observation": source["start_observation"],
                                   "task_goal": source["task_goal"], "intent": intent})
        decisions.append({**item, "status": "ready", "errors": []})
    return controller_records, decisions


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--matrix-plan", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.test_ids.read_text(encoding="utf-8"))
    if frozen.get("dataset_sha256") != file_sha256(args.data):
        raise ValueError("Frozen controller IDs refer to a different dataset")
    if args.matrix_plan.with_suffix(".sha256").read_text(encoding="utf-8").strip() != file_sha256(args.matrix_plan):
        raise ValueError("Frozen matrix plan checksum differs")
    matrix = json.loads(args.matrix_plan.read_text(encoding="utf-8"))
    verify_plan(args.matrix_plan.resolve().parent, matrix)
    all_ids = json.loads(Path(matrix["inputs"]["test_ids"]["path"]).read_text(encoding="utf-8"))["record_ids"]
    if not set(frozen["record_ids"]).issubset(all_ids):
        raise ValueError("Controller contexts are not a subset of the frozen matrix")
    metadata = json.loads(args.predictions.with_suffix(".run.json").read_text(encoding="utf-8"))
    if matrix["inputs"]["data"]["sha256"] != file_sha256(args.data):
        raise ValueError("Controller dataset differs from the frozen matrix")
    verify_generator_config(metadata, matrix, args.variant, args.predictions)
    inputs, decisions = prepare(read_jsonl(args.data), frozen["record_ids"], read_jsonl(args.predictions), args.variant)
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh controller input directory")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent))
    try:
        atomic_text(temporary / "records.jsonl", jsonl(inputs))
        atomic_text(temporary / "eligibility.jsonl", jsonl(decisions))
        manifest = {"schema_version": 1, "variant": args.variant,
                    "counts": dict(Counter(row["status"] for row in decisions)), "planned": len(decisions),
                    "data_sha256": file_sha256(args.data), "test_ids_sha256": file_sha256(args.test_ids),
                    "predictions_sha256": file_sha256(args.predictions),
                    "matrix_plan_sha256": file_sha256(args.matrix_plan),
                    "prediction_run_sha256": file_sha256(args.predictions.with_suffix(".run.json")),
                    "prompts_sha256": file_sha256(args.predictions.with_suffix(".prompts.jsonl")),
                    "records_sha256": file_sha256(temporary / "records.jsonl"),
                    "eligibility_sha256": file_sha256(temporary / "eligibility.jsonl"),
                    "input_contract": "Only generated intent, current board and task; no future board or actions."}
        atomic_text(temporary / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        os.rename(temporary, args.output_dir)
    except Exception:
        shutil.rmtree(temporary)
        raise
    print(json.dumps(manifest["counts"]))


if __name__ == "__main__":
    main()
