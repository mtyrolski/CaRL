"""Bind an exploratory pilot to verified development and eight-shot smoke evidence.

The receipt authorises neither money nor an unbounded run. Each live run also
requires a first-20 allowlist and its own transactional campaign ledger.
"""
import argparse
import json
from pathlib import Path

from carl.icl.reachability_data import canonical_bytes
from scripts.reachability_runner import engine

VERSION = "figure1-exploratory-pilot-evidence-v1"
MODELS = {"luna": "gpt-6-luna", "terra_6_1": "gpt-6.1-terra"}


def _digest(value):
    return engine.sha(canonical_bytes(value))


def _development(path: Path, dataset_sha: str) -> dict:
    dev = engine.read(path)
    digest = dev.get("evidence_sha256")
    if (dev.get("schema") != "figure1-dev-readiness-evidence-v1"
            or digest != _digest({k: v for k, v in dev.items() if k != "evidence_sha256"})
            or dev.get("ready_for_exploratory_gate_review") is not True or dev.get("errors") != []
            or dev.get("development_core_planned") != 96 or dev.get("replayed", 0) < 96
            or dev.get("model_profiles") != MODELS or dev.get("candidate_layouts") != 20
            or dev.get("candidate_pairs") != 120 or dev.get("dataset_manifest_sha256") != dataset_sha
            or dev.get("verification", {}).get("mode") != "live"):
        raise ValueError("Development evidence does not pass the exploratory pilot gate")
    # The receipt's snapshot was computed from the 99 saved v3 trial rows; it
    # must still describe those exact observations, not merely a stale report.
    root = path.parent / "figure1_pilot_20261104_v3" / "study"
    manifest = engine.read(root / "manifest.json")
    if manifest.get("contract_id") != dev.get("contract_id"):
        raise ValueError("Development manifest differs from evidence")
    records = []
    for job in manifest["jobs"]:
        trial = root / "trials" / (job["key"] + ".json")
        if trial.exists():
            row = engine.read(trial)
            if row.get("identity") != job:
                raise ValueError("Development trial identity differs from evidence")
            records.append({"key": job["key"], "sha256": _digest(row)})
    if _digest(records) != dev.get("trial_snapshot_sha256"):
        raise ValueError("Development trial snapshot differs from evidence")
    return dev


def _smoke(config: Path, dataset_sha: str) -> dict:
    study = engine.load_study(config)
    if (study.mode != "live" or study.n_main != 0 or study.n_pilot != 1
            or study.few_shot != 8 or study.core_repetitions != 1
            or study.reference_contexts != 0 or study.core_offsets != [1]
            or study.action_cap not in (4, 32)
            or {arm: profile.model_id for arm, profile in study.arms.items()} != MODELS
            or engine.load_config(study.experiment_config).policy_demo_selection != "diverse_training_v2"):
        raise ValueError("Eight-shot smoke contract is not the required paired development smoke")
    manifest, output = engine.checked(study)
    if manifest.get("dataset_manifest_sha256") != dataset_sha:
        raise ValueError("Eight-shot smoke dataset differs from pilot dataset")
    jobs = manifest["jobs"]
    if (len(jobs) != 2 or {job["arm"] for job in jobs} != set(MODELS)
            or len({job["pair"] for job in jobs}) != 1 or any(job["group"] != "pilot" for job in jobs)):
        raise ValueError("Eight-shot smoke does not contain a shared paired goal")
    verification = engine.verify(study)
    if verification.get("verified_files") != 2 or verification.get("counts") != {"complete": 2}:
        raise ValueError("Eight-shot smoke is incomplete or has not been replayed")
    rows = []
    for job in jobs:
        row = engine.read(output / "trials" / (job["key"] + ".json"))
        if (row.get("status") != "complete" or not any(
                request.get("returned_model") == MODELS[job["arm"]]
                and request.get("reason") is None
                and isinstance(request.get("probabilities"), list)
                and len(request["probabilities"]) == 4 for request in row.get("requests", []))):
            raise ValueError("Eight-shot smoke lacks a usable verified model distribution")
        rows.append({"key": job["key"], "sha256": _digest(row)})
    return {"contract_id": manifest["contract_id"], "manifest_sha256": engine.sha((output / "manifest.json").read_bytes()),
            "trial_snapshot_sha256": _digest(rows), "verified_files": verification["verified_files"]}


def build(dev_path: Path, smoke_config: Path) -> dict:
    smoke = engine.load_study(smoke_config)
    dataset_sha = engine.sha((engine.absolute(smoke.dataset_dir) / "manifest.json").read_bytes())
    dev = _development(dev_path, dataset_sha)
    smoke_evidence = _smoke(smoke_config, dataset_sha)
    value = {"schema": VERSION, "dataset_manifest_sha256": dataset_sha,
             "development_path": str(dev_path.resolve()), "development_sha256": engine.sha(dev_path.read_bytes()),
             "development_contract_id": dev["contract_id"], "smoke_config": str(smoke_config.resolve()),
             "smoke_config_sha256": engine.sha(smoke_config.read_bytes()), "smoke": smoke_evidence}
    value["evidence_sha256"] = _digest(value)
    return value


def validate_evidence(study) -> None:
    path = engine.absolute(study.exploratory_pilot_evidence)
    raw = path.read_bytes()
    proof = json.loads(raw)
    if (engine.sha(raw) != study.exploratory_pilot_evidence_sha256
            or proof.get("schema") != VERSION
            or proof.get("evidence_sha256") != _digest({k: v for k, v in proof.items() if k != "evidence_sha256"})
            or proof.get("dataset_manifest_sha256") != engine.sha((engine.absolute(study.dataset_dir) / "manifest.json").read_bytes())):
        raise ValueError("Exploratory pilot evidence or dataset hash differs")
    dev_path, smoke_config = Path(proof["development_path"]), Path(proof["smoke_config"])
    if (engine.sha(dev_path.read_bytes()) != proof.get("development_sha256")
            or engine.sha(smoke_config.read_bytes()) != proof.get("smoke_config_sha256")):
        raise ValueError("Exploratory development or smoke evidence changed")
    smoke_study = engine.load_study(smoke_config)
    # The 8- and 32-shot files have different names. Compare the resolved
    # configuration rather than filenames/bytes; Study.few_shot is the only
    # intentional policy difference and overrides the file's default.
    smoke_config_semantics = engine.load_config(smoke_study.experiment_config).model_dump(
        mode="json", exclude={"name", "few_shot"})
    study_config_semantics = engine.load_config(study.experiment_config).model_dump(
        mode="json", exclude={"name", "few_shot"})
    if (smoke_config_semantics != study_config_semantics
            or {k: v.model_dump(mode="json") for k, v in smoke_study.arms.items()}
            != {k: v.model_dump(mode="json") for k, v in study.arms.items()}
            or smoke_study.policy_temperature != study.policy_temperature
            or smoke_study.format_retries != study.format_retries
            or smoke_study.transport_retries != study.transport_retries):
        raise ValueError("Exploratory policy, profiles or retry contract differs from eight-shot smoke")
    dev = _development(dev_path, proof["dataset_manifest_sha256"])
    if dev["contract_id"] != proof.get("development_contract_id") or _smoke(smoke_config, proof["dataset_manifest_sha256"]) != proof.get("smoke"):
        raise ValueError("Exploratory development or smoke snapshot changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-evidence", type=Path, required=True)
    parser.add_argument("--smoke-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    proof = build(args.dev_evidence, args.smoke_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(proof, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": engine.sha(args.output.read_bytes())}))


if __name__ == "__main__":
    main()
