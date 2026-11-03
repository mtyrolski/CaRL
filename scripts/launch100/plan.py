"""Validate portable measured assets and freeze a reproducible launch without API calls."""

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import tiktoken
from dotenv import dotenv_values

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows, state_hash
from carl.icl.config import ROOT, ExperimentConfig, Limits
from carl.icl.datasets import _public_instances, environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.types import Instance, JSONObject, TrainingData
from scripts.goal_probe.components import MechanicsExample, TaskProgressComponents
from scripts.launch100.retry import HTTP_RETRY_POLICY

ASSETS = ROOT / "configs/icl/checkpoints/launch100/assets"
OUTPUT = ROOT / "analysis/luna_icl/launch100_nodes500"
METHODS = {"relative_u": ("relative_u", [0.85, 0.5], "u", "pilot_relative"),
           "absolute_d": ("absolute_d", [8.0, 4.0], "d", "pilot_absolute"),
           "fixed_k": ("fixed", [8.0, 4.0], "k", "pilot_fixed"),
           "verbal_relative": ("relative_u", [0.85, 0.5], "u", "pilot_fixed")}
LIMITS = Limits(max_requests=24000, max_input_tokens=24000, max_total_input_tokens=576000000,
                max_output_tokens=98304000, wall_seconds=21600, timeout_seconds=90)


def digest(value: object) -> str:
    """Hash canonical JSON for job identities and resume checks."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path: Path) -> str:
    """Hash exact artifact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Bundle:
    """Validated public inputs; archival provenance remains byte-preserved."""

    path: Path
    config: ExperimentConfig
    training: TrainingData
    instances: list[Instance]
    demonstrations: list[MechanicsExample]
    manifest: JSONObject


def load_assets(path: Path = ASSETS) -> Bundle:
    """Read only public boards and training assets; never load held-out witnesses."""
    path = path.resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != "carl-launch100-assets-v1":
        raise ValueError("Unsupported launch assets")
    for name, item in manifest["files"].items():
        artifact = (path / name).resolve()
        if not artifact.is_relative_to(path) or sha256(artifact) != item["sha256"]:
            raise ValueError(f"Launch asset mismatch: {name}")
    config = ExperimentConfig.model_validate(json.loads((path / "config.json").read_text()))
    config = config.model_copy(update={"data_dir": str(path / "data"),
                                      "generator_checkpoint": str(path / "checkpoint/generator.json")})
    training_path = path / "data/training.json"
    training = cast(TrainingData, json.loads(training_path.read_text()))
    values = json.loads((path / "data/value_examples.json").read_text())
    if (values["training_sha256"] != sha256(training_path)
            or values["source_sha256"] != training.get("provenance", {})["source_sha256"]):
        raise ValueError("Value examples differ from frozen training")
    training["value_examples"] = values["examples"]
    instances = _public_instances(path / "data/real_12x12x4/pilot100.json")
    rows = json.loads((path / "demonstrations.json").read_text())["demonstrations"]
    demos = [MechanicsExample(id=row["id"], state=row["state"], subgoal=row["subgoal"]) for row in rows]
    layouts = {canonical_layout_id(from_rows(row["state"])) for row in instances}
    train_layouts = {canonical_layout_id(from_rows(row["state"])) for row in training["contexts"]}
    train_layouts.update(canonical_layout_id(from_rows(row["state"])) for row in demos)
    if len(instances) != 100 or len(layouts) != 100 or layouts & train_layouts or len(demos) != 6:
        raise ValueError("Expected 100 distinct train-disjoint boards and six progress demonstrations")
    return Bundle(path, config, training, instances, demos, manifest)


def configuration(bundle: Bundle, method: str, action_budget: int) -> ExperimentConfig:
    """Carry forward scales; 500 counts expanded queue entries, not actions."""
    if not 1 <= action_budget <= 4000:
        raise ValueError("Action budget must be in 1..4000; node budget remains 500")
    name, scales, _, _ = METHODS[method]
    return ExperimentConfig.model_validate({**bundle.config.model_dump(), "name": f"launch100_{method}",
        "method": name, "scales": scales, "max_expansions": 500, "action_budget": action_budget,
        "repeat_seeds": [11], "instances": 100, "split": "pilot100", "family": "real_12x12x4",
        "generator_example_limit": 26, "num_subgoals": 1, "limits": LIMITS.model_dump()})


def component(bundle: Bundle, config: ExperimentConfig, api: ExperimentAPI, state: Instance) -> TaskProgressComponents:
    """Use the measured intervention, including all six unlabelled demonstrations."""
    label = {"relative_u": "u", "absolute_d": "d", "fixed": "k"}[config.method]
    return TaskProgressComponents(config, api, environment(from_rows(state["state"])),
                                  bundle.training, bundle.demonstrations, scale_label=label)


def build_plan(bundle: Bundle, methods: list[str], action_budget: int = 4000) -> JSONObject:
    """Preflight actual prompts and all source identities without constructing an API client."""
    if not methods or len(methods) != len(set(methods)) or any(name not in METHODS for name in methods):
        raise ValueError("Select distinct supported methods")
    configs = {name: configuration(bundle, name, action_budget) for name in methods}
    offline = cast(ExperimentAPI, SimpleNamespace(provider="replay"))
    encoder = tiktoken.get_encoding("o200k_base")
    prompts = []
    for name, config in configs.items():
        if name == "verbal_relative":
            continue
        initial = bundle.instances[0]
        model = component(bundle, config, offline, initial)
        if model.fingerprint() != bundle.manifest["policy_fingerprint"]:
            raise ValueError("Frozen policy fingerprint differs")
        if model.generator_checkpoint is None or len(model.generator_checkpoint.examples) != 26:
            raise ValueError("Expected exactly 26 calibrated training demonstrations")
        heldout = {canonical_layout_id(from_rows(row["state"])) for row in bundle.instances}
        if any(canonical_layout_id(from_rows(list(row.state))) in heldout for row in model.generator_checkpoint.examples):
            raise ValueError("Generator training overlaps the launch cohort")
        for scale in config.scales:
            payload = model.generator_payload(from_rows(initial["state"]), scale, resolve_embeddings=False)
            tokens = len(encoder.encode(json.dumps(payload), disallowed_special=()))
            if tokens > config.limits.max_input_tokens:
                raise ValueError("Generator prompt exceeds token limit")
            prompts.append({"method": name, "scale": scale, "input_tokens_estimate": tokens, "payload": payload})
    verbal = None
    if "verbal_relative" in methods:
        from scripts.verbal_icl.launch import load_bundle

        verbal = load_bundle(bundle.path / "verbal.json").contract
    paths = [*sorted((ROOT / "carl").rglob("*.py")), *sorted(Path(__file__).parent.glob("*.py")),
             ROOT / "scripts/goal_probe/components.py", ROOT / "configs/icl/campaign_budget.json", ROOT / "uv.lock",
             ROOT / "configs/icl/checkpoints/launch100/cost_reference.json"]
    if verbal is not None:
        paths.extend(sorted((ROOT / "scripts/verbal_icl").glob("*.py")))
    contract = {"version": 1, "provider": "openai", "methods": methods, "seed": 11,
                "node_budget": 500, "action_budget": action_budget, "action_cap": 8,
                "node_definition": "expanded non-stale frontier entries, including scale and primitive fallback entries",
                "cohort": [{"id": row["id"], "state_hash": state_hash(from_rows(row["state"]))} for row in bundle.instances],
                "configs": {name: config.model_dump(mode="json") for name, config in configs.items()},
                "asset_manifest_sha256": sha256(bundle.path / "manifest.json"),
                "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
                "policy_fingerprint": bundle.manifest["policy_fingerprint"], "verbal": verbal,
                "scales_note": "u=.85,.5 and d=8,4 carried forward, not retuned for task-progress prompt",
                "fixed_note": "fixed8,4 with cap=k and enumerated one-step primitive fallback; not native trained k841",
                "usd_note": "--usd-cap is an absolute shared campaign ceiling, not fresh budget or a cost estimate"}
    contract["wall_limit_note"] = "21600 seconds per process invocation; logical request/token limits persist across resume"
    contract["retry_policy"] = HTTP_RETRY_POLICY
    contract["retry_note"] = "SDK automatic retries disabled; one explicit HTTP429/5xx retry plus configured JSON retry"
    return {**contract, "freeze_id": digest(contract), "prompt_previews": prompts,
            "jobs": 100 * len(methods), "assets": str(bundle.path)}


def ledger_status() -> JSONObject:
    """Read existing reservations without creating or changing the global ledger."""
    prices = json.loads((ROOT / "configs/icl/campaign_budget.json").read_text())
    path = ROOT / prices["ledger_path"]
    used: dict[str, float] = {}
    effective = float(prices["limit_usd"])
    unknown = 0.0
    if path.exists():
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            used = {stage: amount / 1e9 for stage, amount in db.execute(
                "SELECT stage,SUM(COALESCE(charged_nanos,reserved_nanos)) FROM attempts GROUP BY stage")}
            row = db.execute("SELECT value FROM metadata WHERE key='effective_limit_nanos'").fetchone()
            if row is not None:
                effective = min(effective, int(row[0]) / 1e9)
            unknown = db.execute("SELECT COALESCE(SUM(reserved_nanos),0) FROM attempts WHERE status!='settled'").fetchone()[0] / 1e9
    for value in (os.environ.get("CARL_CAMPAIGN_MAX_USD"), dotenv_values(ROOT / ".env").get("CARL_CAMPAIGN_MAX_USD")):
        if value is not None:
            cap = float(value)
            if not math.isfinite(cap) or cap < 0:
                raise ValueError("Campaign environment ceiling must be finite and nonnegative")
            effective = min(effective, cap)
    present = path.exists()
    return {"ledger_present": present, "effective_ceiling_usd": effective,
            "committed_usd": sum(used.values()) if present else None,
            "remaining_usd": max(0, effective - sum(used.values())) if present else None,
            "unsettled_reserved_usd": unknown if present else None,
            "stages": {name: {"limit_usd": float(cap), "committed_usd": used.get(name, 0) if present else None,
                                "remaining_usd": max(0, float(cap) - used.get(name, 0)) if present else None}
                       for name, cap in prices["stage_limits_usd"].items()}}


def prepare(plan: JSONObject, output: Path) -> None:
    """Freeze immutable resolved inputs before execution; never overwrite a different plan."""
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Launch protocol differs; choose a new output directory")
    if not path.exists():
        write_json(path, plan)
