"""Frozen, resumable reachability trials with an explicitly offline transport."""
import csv
import hashlib
import json
import math
import os
import time
from decimal import Decimal
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, model_validator

from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash
from carl.icl.calibration import weighted_percentiles
from carl.icl.components import ICLComponents
from carl.icl.config import ExperimentConfig, load_config
from carl.icl.datasets import environment
from carl.icl.reachability_data import _unbundle, canonical_bytes, policy_query, verify_dataset
from scripts.reachability_runner.ledger import LimitExceeded, OfflineLedger

ROOT = Path(__file__).resolve().parents[2]
VERSION = "reachability-trials-v2"
STAGES = ("pilot", "main", "calibration_A", "calibration_B")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def absolute(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    currency: str
    input_per_million: str
    output_per_million: str
    simulated: bool


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1)
    provider: str
    endpoint: str
    reasoning: str
    response_format: str
    max_output_tokens: int = Field(ge=32)
    text_temperature: float | None
    pricing: Pricing | None
    request_options: dict[str, Any]
    allowed_response_model_ids: list[str] | None = None
    mock_probabilities: list[float] | None = None
    mock_outputs: list[str] | None = None
    mock_transport_failure: bool = False
    mock_timeout: bool = False

    @model_validator(mode="after")
    def check_contract(self):
        if self.endpoint != "responses" or self.response_format not in {"text", "json_schema"}:
            raise ValueError("Unsupported endpoint or response format")
        if self.provider not in {"mock", "openai"}:
            raise ValueError("Unsupported provider")
        supported = {"service_tier", "text", "reasoning", "tools", "stream", "store", "include"}
        if set(self.request_options) != supported:
            raise ValueError("Profile must declare exactly the supported Responses request options")
        if self.request_options.get("reasoning", {}).get("effort") != self.reasoning:
            raise ValueError("Profile reasoning must match explicit request options")
        if any(key in self.request_options for key in ("model", "input", "max_output_tokens", "metadata", "temperature", "top_p")):
            raise ValueError("Request options cannot override required policy fields")
        if self.text_temperature is not None:
            raise ValueError("Text-generation temperature is not implemented for this Responses contract")
        if self.provider == "mock":
            probs = self.mock_probabilities
            if probs is None or len(probs) != 4 or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs) or sum(probs) <= 0:
                raise ValueError("Mock profile requires an explicit valid action distribution")
            if self.pricing is None or not self.pricing.simulated or self.pricing.input_per_million != "0" or self.pricing.output_per_million != "0":
                raise ValueError("Mock pricing must be explicitly simulated and zero")
        elif not self.allowed_response_model_ids or any(not item for item in self.allowed_response_model_ids):
            raise ValueError("Live profiles must explicitly list permitted resolved model IDs")
        elif any(item != self.model_id and not item.startswith(self.model_id + "-")
                 for item in self.allowed_response_model_ids):
            raise ValueError("Resolved model allowlist cannot substitute a different model family")
        return self


class Study(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_dir: str
    output_dir: str
    cache_dir: str
    mode: str
    replay_source: str | None
    experiment_config: str
    arms: dict[str, Profile]
    n_main: int = Field(ge=0, le=100)
    n_pilot: int = Field(ge=0)
    core_offsets: list[int]
    core_repetitions: int = Field(ge=1, le=5)
    reference_contexts: int = Field(ge=0, le=8)
    reference_targets: int = Field(ge=0, le=8)
    reference_repetitions_per_group: int = Field(ge=1, le=10)
    development_reference_manifest: str | None = None
    live_campaign_config: str | None = None
    exploratory_pilot_evidence: str | None = None
    exploratory_pilot_evidence_sha256: str | None = None
    credential_source: str = "environment"
    live_timeout_seconds: float = Field(default=30.0, gt=0)
    live_wall_seconds: float = Field(default=3600.0, gt=0)
    action_cap: int = Field(ge=1)
    few_shot: int = Field(ge=0)
    policy_temperature: float = Field(gt=0)
    format_retries: int = Field(ge=0, le=1)
    transport_retries: int = Field(ge=0, le=0)
    limits: dict[str, Any]

    @model_validator(mode="after")
    def check_study(self):
        if self.mode not in {"mock", "replay", "live"}:
            raise ValueError("Mode must be mock, replay or live")
        if self.credential_source not in {"environment", "dotenv"}:
            raise ValueError("Credential source must be explicitly environment or dotenv")
        if self.mode == "replay" and self.replay_source not in {"mock", "live"}:
            raise ValueError("Replay must explicitly select mock or live cache")
        if self.mode != "replay" and self.replay_source is not None:
            raise ValueError("replay_source is valid only in replay mode")
        if set(self.arms) != {"luna", "terra_6_1"}:
            raise ValueError("Exactly the two frozen dataset arms are required")
        if len({p.response_format for p in self.arms.values()}) != 1:
            raise ValueError("Both arms must use the same response format and policy instruction")
        expected_provider = "mock" if self.mode == "mock" or self.replay_source == "mock" else "openai"
        if any(p.provider != expected_provider for p in self.arms.values()):
            raise ValueError("Profile provider does not match selected mode")
        if not self.core_offsets or len(set(self.core_offsets)) != len(self.core_offsets) or any(k not in (1, 2, 4, 8, 12, 16) for k in self.core_offsets):
            raise ValueError("Offsets must be distinct members of the frozen K")
        if set(self.limits.get("arm_usd", {})) != set(self.arms) or set(self.limits.get("stage_usd", {})) != set(STAGES):
            raise ValueError("Explicit limits are required for every arm and stage")
        if any(not isinstance(self.limits.get(key), int) or isinstance(self.limits.get(key), bool) or self.limits[key] < 0
               for key in ("logical_calls", "physical_calls")):
            raise ValueError("Logical and physical call limits must be nonnegative integers")
        if self.reference_contexts and not self.reference_targets:
            raise ValueError("Reference target count is required")
        if self.limits.get("token_limit") is not None:
            limit = self.limits["token_limit"]
            reservation = self.limits.get("mock_token_reservation")
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise ValueError("Offline token limit must be a nonnegative integer")
            if self.mode == "mock" and (not isinstance(reservation, int) or isinstance(reservation, bool) or reservation <= 0):
                raise ValueError("A positive simulated token reservation is required with a mock token limit")
        return self


def load_study(path: Path) -> Study:
    return Study.model_validate_json(path.read_text(encoding="utf-8"))


def data_inputs(study: Study) -> tuple[dict, dict, list, list, dict]:
    root = absolute(study.dataset_dir)
    manifest = read(root / "manifest.json")
    for name, expected in manifest["artifacts_sha256"].items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError(f"Dataset artifact hash mismatch: {name}")
    core, reference = read(root / "core_pairs.json"), read(root / "calibration_pairs.json")
    if any(doc["experiment_id"] != manifest["experiment_id"] for doc in (core, reference)):
        raise ValueError("Dataset experiment IDs do not match")
    boards = read(root / "states.json")["states"]
    if study.development_reference_manifest is not None:
        dev = read(absolute(study.development_reference_manifest))
        if (dev.get("schema") != "figure1-development-diagnostic-v1" or dev.get("split") != "pilot"
                or dev.get("source_dataset_id") != manifest["experiment_id"]
                or dev.get("source_bundle_sha256") != manifest["artifacts_sha256"]["private/source_bundle.tar.gz"]
                or dev.get("selected_targets") != len(dev.get("pairs", []))):
            raise ValueError("Development diagnostic does not belong to the frozen pilot dataset")
        pilots = {r["context_id"]: r for r in core["pairs"] if r["split"] == "pilot"}
        main_layouts = {r["layout_id"] for r in core["pairs"] if r["split"] != "pilot"}
        dev_archive = _unbundle((root / "private/source_bundle.tar.gz").read_bytes())
        pilot_instances = json.loads(dev_archive["source/real_12x12x4/validation.json"])["instances"]
        pilot_witnesses = json.loads(dev_archive["source/private_witnesses/real_12x12x4/validation.json"])
        initial = {row["id"]: row["state"] for row in pilot_instances}
        witness_actions = {row["id"]: row["actions"] for row in pilot_witnesses}
        checked_trajectories = {}
        new_reference = []
        for pair in dev["pairs"]:
            original = pilots.get(pair["context_id"])
            if (original is None or pair["layout_id"] in main_layouts
                    or any(pair[field] != original[field] for field in ("layout_id", "source_hash", "trajectory_id", "i"))
                    or pair["source_hash"] == pair["target_hash"] or not 1 <= pair["k"] <= 16):
                raise ValueError("Development diagnostic context is not disjoint and anchored")
            trajectory = pair["trajectory_id"]
            if trajectory not in checked_trajectories:
                state = from_rows(initial[trajectory])
                env = environment(state)
                states = [state]
                for action in witness_actions[trajectory]:
                    if env.is_solved(state):
                        raise ValueError("Development witness continues after terminal")
                    next_state = np.asarray(env.next_state(state, action))
                    if np.array_equal(state, next_state):
                        raise ValueError("Development witness includes a no-op")
                    state = next_state
                    states.append(state)
                if not env.is_solved(state):
                    raise ValueError("Development witness does not solve task")
                checked_trajectories[trajectory] = states
            states = checked_trajectories[trajectory]
            if (pair["i"] + pair["k"] >= len(states)
                    or state_hash(states[pair["i"]]) != pair["source_hash"]
                    or state_hash(states[pair["i"] + pair["k"]]) != pair["target_hash"]):
                raise ValueError("Development diagnostic target differs from witness replay")
            groups = {group: {arm: [] for arm in study.arms} for group in ("A", "B")}
            for trial in pair["trials"]:
                arm = {"gpt-6-luna": "luna", "gpt-6.1-terra": "terra_6_1"}.get(trial["model"])
                if arm is None or trial["group"] not in groups:
                    raise ValueError("Development diagnostic model/group differs from profile")
                groups[trial["group"]][arm].append({"trial": trial["repetition"], "trial_id": trial["trial_id"],
                                                    "action_seed": trial["action_seed"]})
            new_reference.append({**pair, "family": dev["family"], "groups": groups})
        if len({(p["context_id"], p["target_hash"]) for p in dev["pairs"]}) != len(dev["pairs"]):
            raise ValueError("Development diagnostic contains duplicate target states")
        if any(len(pair["groups"][group][arm]) < study.reference_repetitions_per_group
               for pair in new_reference for group in ("A", "B") for arm in study.arms):
            raise ValueError("Development diagnostic lacks planned repetitions")
        dev_states = dev["states_by_hash"]
        for key, value in dev_states.items():
            if state_hash(from_rows(value)) != key:
                raise ValueError("Development diagnostic state hash mismatch")
        if any(pair[field] not in dev_states for pair in dev["pairs"] for field in ("source_hash", "target_hash")):
            raise ValueError("Development diagnostic state is missing")
        boards.update(dev_states)
        reference["pairs"] = new_reference
        manifest = {**manifest, "calibration_contexts": list(dict.fromkeys(pair["context_id"] for pair in dev["pairs"]))}
    source = _unbundle((root / "private/source_bundle.tar.gz").read_bytes())
    training = json.loads(source["source/training.json"])
    return manifest, training, core["pairs"], reference["pairs"], boards


def _content_sources() -> dict[str, str]:
    files = list((ROOT / "carl/icl").rglob("*.py")) + list((ROOT / "carl/environment/sokoban").rglob("*.py"))
    files += list((ROOT / "scripts/reachability_runner").rglob("*.py"))
    files += [p for p in (ROOT / "scripts/figure1_pilot").glob("*.py") if p.name in {
        "budget.py", "transport.py", "block_order.py", "exact_token_preflight.py", "exploratory_evidence.py"}]
    files += [ROOT / "carl/utils/aliases.py", ROOT / "uv.lock", ROOT / "pyproject.toml"]
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted(files)}


def _package_versions() -> dict:
    result = {}
    for name in ("numpy", "pydantic", "filelock", "openai", "tiktoken", "gym", "gymnasium"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    import sys
    result["python"] = sys.version.split()[0]
    return result


def resolved_policy(study: Study, profile: Profile) -> ExperimentConfig:
    base = load_config(study.experiment_config)
    if base.generator_checkpoint or base.locked_policy_fingerprint:
        raise ValueError("A model-locked generator or policy cannot be reused")
    calibration = base.calibration.model_copy(update={"action_cap": study.action_cap})
    return ExperimentConfig.model_validate({**base.model_dump(), "model": profile.model_id, "method": "direct",
        "few_shot": study.few_shot, "policy_temperature": study.policy_temperature,
        "policy_reasoning_effort": profile.reasoning, "policy_response_format": profile.response_format,
        "policy_tokens": profile.max_output_tokens, "calibration": calibration.model_dump()})


def _job(study: Study, data_manifest: dict, contract_id: str, pair: dict, arm: str,
         stage: str, repetition: int, original_seed: int) -> dict:
    profile = study.arms[arm]
    identity = {"experiment": contract_id, "arm": arm, "group": stage, "family": pair["family"],
                "layout": pair["layout_id"], "context": pair["context_id"], "pair": pair["pair_id"],
                "model": profile.model_id, "repetition": repetition}
    key = sha(canonical_bytes(identity))
    seed = int(sha(canonical_bytes([VERSION, "action_rng", data_manifest["seed"], identity]))[:16], 16)
    return {**identity, "key": key, "seed": seed, "dataset_trial_seed": original_seed,
            "k": pair["k"], "source_hash": pair["source_hash"], "target_hash": pair["target_hash"]}


def jobs(study: Study, data_manifest: dict, core: list, reference: list, contract_id: str) -> list[dict]:
    from scripts.figure1_pilot.block_order import order_jobs

    if study.n_main > len(data_manifest["n100"]):
        raise ValueError("Requested main cohort exceeds frozen data")
    selected = set(data_manifest["n100"][:study.n_main])
    available_pilot = sorted({r["context_id"] for r in core if r["split"] == "pilot"})
    if study.n_pilot > len(available_pilot):
        raise ValueError("Requested pilot cohort exceeds frozen data")
    selected_pilot = set(available_pilot[:study.n_pilot])
    refs = set(data_manifest["calibration_contexts"][:study.reference_contexts])
    if study.reference_contexts > len(data_manifest["calibration_contexts"]):
        raise ValueError("Requested reference contexts exceed frozen data")
    result = []
    for pair in core:
        stage = "pilot" if pair["split"] == "pilot" else "main"
        if pair["k"] not in study.core_offsets or pair["context_id"] not in (selected_pilot if stage == "pilot" else selected):
            continue
        for arm in study.arms:
            for item in pair["trials"][arm][:study.core_repetitions]:
                result.append(_job(study, data_manifest, contract_id, pair, arm, stage, item["trial"], item["action_seed"]))
    counts: Counter = Counter()
    for pair in reference:
        context = pair["context_id"]
        if context not in refs or counts[context] >= study.reference_targets:
            continue
        counts[context] += 1
        for group in ("A", "B"):
            stage = f"calibration_{group}"
            for arm in study.arms:
                for item in pair["groups"][group][arm][:study.reference_repetitions_per_group]:
                    result.append(_job(study, data_manifest, contract_id, pair, arm, stage, item["trial"], item["action_seed"]))
    if len({row["key"] for row in result}) != len(result) or len({row["seed"] for row in result}) != len(result):
        raise ValueError("Trial ID or RNG seed collision")
    reference_targets = {context: counts[context] for context in refs}
    return order_jobs(result, main_contexts=data_manifest["n100"][:study.n_main],
                      pilot_contexts=available_pilot[:study.n_pilot],
                      reference_contexts=data_manifest["calibration_contexts"][:study.reference_contexts],
                      core_offsets=study.core_offsets, arms=tuple(study.arms), core_repetitions=study.core_repetitions,
                      reference_repetitions=study.reference_repetitions_per_group,
                      reference_targets_by_context=reference_targets)


def contract(study: Study) -> dict:
    from scripts.figure1_pilot.block_order import ORDER_VERSION

    dataset, training, core, reference, _boards = data_inputs(study)
    configs = {arm: resolved_policy(study, profile).model_dump(mode="json") for arm, profile in study.arms.items()}
    value = {"version": VERSION, "study": study.model_dump(mode="json"),
             "dataset_manifest_sha256": sha((absolute(study.dataset_dir) / "manifest.json").read_bytes()),
             "dataset_artifacts": dataset["artifacts_sha256"], "dataset_experiment": dataset["experiment_id"],
             "training_sha256": sha(canonical_bytes(training)), "resolved_configs": configs,
             "code_sha256": _content_sources(), "packages": _package_versions(),
             "action_map": ["UP", "DOWN", "LEFT", "RIGHT"], "masking": "none",
             "status_contract": ["exact_hit", "miss_cap", "solved_before_target", "invalid_model_response", "unresolved"],
             "rng": "PCG64 seeded by SHA256(version,dataset seed,full trial identity)",
             "ordering_version": ORDER_VERSION,
             "randomness_note": "Independent action streams conditional on cached model distributions, not independent model generations."}
    if study.development_reference_manifest is not None:
        value["development_reference_sha256"] = sha(absolute(study.development_reference_manifest).read_bytes())
    if study.live_campaign_config is not None:
        value["live_campaign_config_sha256"] = sha(absolute(study.live_campaign_config).read_bytes())
    if study.exploratory_pilot_evidence is not None:
        value["exploratory_pilot_evidence_sha256"] = sha(absolute(study.exploratory_pilot_evidence).read_bytes())
    # Location and execution mode are frozen in the manifest but are not a new
    # scientific observation. Replay therefore retains the original trial IDs.
    identity = {**value, "study": study.model_dump(mode="json", exclude={"mode", "replay_source", "output_dir", "cache_dir"})}
    value["contract_id"] = sha(canonical_bytes(identity))
    value["jobs"] = jobs(study, dataset, core, reference, value["contract_id"])
    return value


def plan(study: Study) -> dict:
    manifest = contract(study)
    counts = Counter((row["group"], row["arm"]) for row in manifest["jobs"])
    by_stage = {stage: {arm: counts[(stage, arm)] for arm in study.arms} for stage in STAGES}
    ceilings = {arm: sum(counts[(stage, arm)] for stage in STAGES) * study.action_cap * (study.format_retries + 1)
                for arm in study.arms}
    result = {"mode": study.mode, "dataset_classification": "exploratory", "per_stage_arm_trials": by_stage,
            "total_trials": len(manifest["jobs"]), "action_cap": study.action_cap,
            "max_action_attempts": len(manifest["jobs"]) * study.action_cap,
            "max_policy_requests_by_arm": ceilings,
            "max_cost_scenario_usd": {arm: "0" if p.provider == "mock" else None for arm, p in study.arms.items()},
            "note": "Worst-case request counts include format retry; live cost is unknown until reviewed pricing and limits."}
    if study.mode == "live" and study.live_campaign_config is not None:
        config = _live_configuration(study)
        quotes = {}
        for arm, profile in study.arms.items():
            price = config["models"][profile.model_id]
            upper = (Decimal(str(price["cache_write_12h_usd_per_million"])) * 2 * price["max_input_tokens"]
                     + Decimal(str(price["output_usd_per_million"])) * Decimal("1.5") * profile.max_output_tokens)
            amount = upper * Decimal(str(price["regional_multiplier"])) / Decimal(1_000_000)
            quotes[arm] = str(amount * ceilings[arm])
        result["conservative_total_reservation_quote_usd"] = quotes
        result["campaign_limit_usd"] = config["limit_usd"]
        result["quote_note"] = "Full-context quote per possible request; conservative committed exposure, not predicted spend."
    return result


def _exploratory_pilot_gate(study: Study) -> None:
    """Allow only the preregistered exploratory N100/R1 pilot contract.

    This is not the final-cohort gate. Live execution is separately restricted
    in ``run`` to an explicit allowlist drawn from the first 20 contexts.
    """
    message = "Final live cohort is blocked; exploratory pilot evidence is missing or invalid"
    if (study.n_main != 100 or study.n_pilot != 0 or study.reference_contexts != 0
            or study.core_offsets != [1, 2, 4, 8, 12, 16] or study.core_repetitions != 1
            or study.action_cap != 32 or study.few_shot not in (8, 32)
            or not study.exploratory_pilot_evidence or not study.exploratory_pilot_evidence_sha256):
        raise ValueError(message)
    from scripts.figure1_pilot.exploratory_evidence import validate_evidence

    validate_evidence(study)
    if load_config(study.experiment_config).policy_demo_selection != "diverse_training_v2":
        raise ValueError("Exploratory pilot requires the frozen diverse training demonstrations")


def _live_configuration(study: Study) -> dict:
    """Fail closed unless an explicit Figure 1 campaign and budget exist."""
    if study.n_main:
        _exploratory_pilot_gate(study)
    if study.reference_contexts and not study.development_reference_manifest:
        raise ValueError("Live pilot diagnostics require an explicitly disjoint development reference manifest")
    if study.live_campaign_config is None:
        raise ValueError("stage 5 requires a separately authorised Figure 1 budget configuration")
    config = read(absolute(study.live_campaign_config))
    if (not str(config.get("campaign_id", "")).startswith("figure1_")
            or not config.get("authorization_reference") or Decimal(str(config.get("limit_usd", "0"))) <= 0
            or Decimal(str(study.limits.get("total_usd", "0"))) <= 0):
        raise ValueError("stage 5 requires a positive explicitly authorised Figure 1 budget")
    if set(config.get("models", {})) != {p.model_id for p in study.arms.values()}:
        raise ValueError("Campaign prices must match exactly the requested models")
    if (not isinstance(study.limits.get("token_limit"), int)
            or isinstance(study.limits["token_limit"], bool) or study.limits["token_limit"] <= 0
            or study.limits["logical_calls"] <= 0 or study.limits["physical_calls"] <= 0):
        raise ValueError("Live studies require explicit positive logical, physical and token limits")
    if set(config.get("model_limits_usd", {})) != set(config["models"]):
        raise ValueError("Campaign requires an explicit limit for each requested model")
    used_stages = set()
    if study.n_pilot:
        used_stages.add("pilot")
    if study.n_main:
        used_stages.add("main")
    if study.reference_contexts:
        used_stages.update(("calibration_A", "calibration_B"))
    if any(Decimal(str(config.get("stage_limits_usd", {}).get(stage, "0"))) <= 0
           or Decimal(str(study.limits["stage_usd"][stage])) <= 0 for stage in used_stages):
        raise ValueError("Every planned stage requires an explicit positive campaign and study limit")
    for arm, profile in study.arms.items():
        price = config["models"][profile.model_id]
        if (profile.model_id != {"luna": "gpt-6-luna", "terra_6_1": "gpt-6.1-terra"}[arm]
                or profile.provider != "openai" or profile.pricing is None or profile.pricing.simulated
                or profile.pricing.currency != "USD"
                or Decimal(profile.pricing.input_per_million) != Decimal(str(price["input_usd_per_million"]))
                or Decimal(profile.pricing.output_per_million) != Decimal(str(price["output_usd_per_million"]))
                or price.get("max_input_tokens", 0) < 1_050_000
                or price.get("max_output_tokens", 0) < profile.max_output_tokens
                or Decimal(str(study.limits["arm_usd"][arm])) <= 0
                or Decimal(str(config["model_limits_usd"][profile.model_id])) <= 0):
            raise ValueError("Both arms require explicit compatible model prices and positive limits")
    return config


def prepare(study: Study) -> dict:
    if study.mode == "live":
        _live_configuration(study)
    verify_dataset(absolute(study.dataset_dir))
    manifest = contract(study)
    output = absolute(study.output_dir)
    file = output / "manifest.json"
    if file.exists():
        if read(file) != manifest:
            raise ValueError("Contract differs from existing freeze; choose a new output")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is nonempty without manifest")
        output.mkdir(parents=True, exist_ok=True)
        write_json(file, manifest)
        OfflineLedger(output / "ledger.sqlite3", study.limits, create=True)
    return manifest


def checked(study: Study) -> tuple[dict, Path]:
    output = absolute(study.output_dir)
    manifest = read(output / "manifest.json")
    if manifest != contract(study):
        raise ValueError("Frozen contract changed; resume is refused")
    if not (output / "ledger.sqlite3").exists():
        raise ValueError("Study ledger missing; resume would reset accounting")
    return manifest, output

class Unresolved(RuntimeError):
    """No resolved policy result; a trial remains pending."""


class InvalidModelResponse(RuntimeError):
    """A completed model response failed policy validation after bounded retries."""


def _replay_live_cache_valid(study: Study, profile: Profile, request: dict, cached: dict, key: str) -> bool:
    """Validate live cache provenance before replay samples an action, read-only."""
    import sqlite3
    from scripts.figure1_pilot.budget import _payload_hash
    from scripts.figure1_pilot.transport import response_digest

    if not study.live_campaign_config:
        return False
    try:
        config_path = absolute(study.live_campaign_config)
        config = read(config_path)
        path = absolute(config["ledger_path"])
        if not path.exists():
            return False
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            meta = db.execute("SELECT value FROM metadata WHERE key='config_hash'").fetchone()
            row = db.execute("SELECT a.model,a.request_hash,a.status,a.usage_json,a.charged_nanos,"
                             "c.payload_sha256,c.response_sha256 FROM attempts a JOIN figure1_contracts c "
                             "ON c.attempt_id=a.id WHERE a.id=?", (cached.get("campaign_reservation_id"),)).fetchone()
        return bool(meta and meta[0] == sha(config_path.read_bytes()) and row and row[0] == profile.model_id
                    and row[1] == key and row[2] == "settled" and row[4] is not None
                    and json.loads(row[3]) == cached.get("usage") and row[5] == _payload_hash(request["payload"])
                    and row[6] == response_digest(cached) and cached.get("requested_model") == profile.model_id
                    and cached.get("model") in (profile.allowed_response_model_ids or ()))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        return False


def _decode_probabilities(text: str, temperature: float) -> np.ndarray:
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        raise InvalidModelResponse("invalid_json") from None
    if not isinstance(data, dict) or not isinstance(data.get("probabilities"), list):
        raise InvalidModelResponse("invalid_vector")
    raw = data["probabilities"]
    if len(raw) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 or x > 1 for x in raw):
        raise InvalidModelResponse("invalid_vector")
    values = np.asarray(raw, dtype=float)
    if values.sum() <= 0:
        raise InvalidModelResponse("invalid_vector")
    values = (values / values.max()) ** (1 / temperature)
    return values / values.sum()


def _payload(study: Study, profile: Profile, training: dict, source: np.ndarray, target: np.ndarray) -> dict:
    config = resolved_policy(study, profile)
    # Only payload construction: no client call, generator or value evaluation.
    fake_api = type("PayloadOnly", (), {"provider": "mock", "metrics": {}})()
    component = ICLComponents(config, fake_api, environment(source), training)
    payload = component.policy_payload(source, target)
    # The experiment profile is authoritative for provider-specific request options.
    base_format = payload["text"]["format"]
    for name in ("service_tier", "text", "reasoning", "tools", "stream", "store", "include"):
        payload.pop(name, None)
    payload.update(profile.request_options)
    payload["text"] = {**profile.request_options.get("text", {}), "format": base_format}
    return payload


def _cache_call(study: Study, profile: Profile, ledger: OfflineLedger, job: dict, attempt: int,
                payload: dict, slot: int, *, live: dict | None = None) -> tuple[dict, dict]:
    profile_hash = sha(canonical_bytes(profile.model_dump(mode="json")))
    request = {"profile_sha256": profile_hash, "endpoint": profile.endpoint, "payload": payload, "slot": slot}
    key = sha(canonical_bytes(request))
    cache_kind = "mock" if study.mode == "mock" or study.replay_source == "mock" else "live"
    cache_file = absolute(study.cache_dir) / cache_kind / f"{key}.json"
    if study.mode == "live":
        if live is None:
            raise Unresolved("live_guard_missing")
        from scripts.figure1_pilot.transport import LiveTransportError, call_responses

        remaining = live["deadline"] - time.monotonic()
        if remaining <= 0:
            raise Unresolved("wall_limit")
        try:
            return call_responses(request=request, cache_file=cache_file, study_ledger=ledger,
                                  campaign_ledger=live["ledger"], job=job, attempt=attempt,
                                  client=live["client"], timeout_seconds=min(study.live_timeout_seconds, remaining),
                                  allowed_response_models=tuple(profile.allowed_response_model_ids or ()))
        except LiveTransportError as exc:
            raise Unresolved(exc.reason_code) from None
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache_file) + ".lock", timeout=30):
        try:
            exists = cache_file.exists()
            existing = read(cache_file) if exists else None
            if exists and not isinstance(existing, dict):
                raise ValueError("Cache entry must be an object")
            if (existing is not None and existing.get("status") == "completed"
                    and (not isinstance(existing.get("output_text"), str)
                         or not isinstance(existing.get("model"), str)
                         or not isinstance(existing.get("usage"), dict))):
                raise ValueError("Completed cache entry is incomplete")
        except (ValueError, UnicodeError, OSError):
            # Corruption is not a cache miss. Preserve the evidence and do
            # not allow another physical request without reconciliation.
            reservation = ledger.reserve(job=job["key"], attempt=attempt, stage=job["group"], arm=job["arm"],
                                         request_hash=key, slot=slot, cache_hit=True, reserve_usd="0")
            ledger.finish(reservation, status="cache_error", charged_usd="0", usage={},
                          reason="corrupt_cache_requires_reconciliation")
            raise Unresolved("corrupt_cache_requires_reconciliation") from None
        if existing is not None and existing.get("request") != request:
            raise Unresolved("cache_request_mismatch")
        if existing is not None and existing.get("status") != "completed":
            reservation = ledger.reserve(job=job["key"], attempt=attempt, stage=job["group"], arm=job["arm"],
                                         request_hash=key, slot=slot, cache_hit=True, reserve_usd="0")
            ledger.finish(reservation, status="cache_error", charged_usd="0", usage={},
                          reason="cached_error_requires_reconciliation")
            raise Unresolved("cached_error_requires_reconciliation")
        if existing is not None and study.replay_source == "live" and not _replay_live_cache_valid(study, profile, request, existing, key):
            reservation = ledger.reserve(job=job["key"], attempt=attempt, stage=job["group"], arm=job["arm"],
                                         request_hash=key, slot=slot, cache_hit=True, reserve_usd="0")
            ledger.finish(reservation, status="cache_error", charged_usd="0", usage={},
                          reason="cached_campaign_mismatch_requires_reconciliation")
            raise Unresolved("cached_campaign_mismatch_requires_reconciliation")
        is_hit = existing is not None
        replay_miss = study.mode == "replay" and not is_hit
        reservation = ledger.reserve(job=job["key"], attempt=attempt, stage=job["group"], arm=job["arm"],
                                     request_hash=key, slot=slot, cache_hit=is_hit or replay_miss, reserve_usd="0",
                                     reserve_tokens=study.limits.get("mock_token_reservation", 0)
                                     if study.limits.get("token_limit") is not None and study.mode == "mock" else 0)
        if replay_miss:
            ledger.finish(reservation, status="replay_miss", charged_usd="0", usage={}, reason="replay_cache_miss")
            raise Unresolved("replay_cache_miss")
        if is_hit:
            ledger.finish(reservation, status="cache_hit", charged_usd="0", usage=existing.get("usage", {}))
            trace = {"request_hash": key, "slot": slot, "cache_hit": True, "ledger_id": reservation}
            if study.replay_source == "live":
                trace["campaign_reservation_id"] = existing.get("campaign_reservation_id")
            return existing, trace
        if study.mode != "mock":
            # No fresh provider call can bypass the existing campaign budget.
            ledger.finish(reservation, status="unknown", charged_usd=None, usage={}, reason="live_disabled")
            raise Unresolved("live_disabled")
        # An interrupted request remains a blocking cache entry. Never issue a
        # second physical request just because the first response is missing.
        write_json(cache_file, {"request": request, "status": "pending", "ledger_id": reservation})
        try:
            if profile.mock_timeout:
                raise TimeoutError("Explicitly simulated timeout")
            if profile.mock_transport_failure:
                raise OSError("Explicitly simulated transport failure")
            if profile.mock_outputs is not None:
                if not profile.mock_outputs:
                    raise ValueError("Empty scripted mock output list")
                output = profile.mock_outputs[min(slot, len(profile.mock_outputs) - 1)]
            else:
                output = json.dumps({"probabilities": profile.mock_probabilities})
            response = {"request": request, "status": "completed", "output_text": output,
                        "model": profile.model_id, "usage": {"input_tokens": None, "output_tokens": None,
                        "provider_cached_input_tokens": None}, "provider": "mock"}
            write_json(cache_file, response)
            ledger.finish(reservation, status="settled", charged_usd="0", usage=response["usage"])
            return response, {"request_hash": key, "slot": slot, "cache_hit": False, "ledger_id": reservation}
        except Exception as exc:
            reason = "timeout_billing_unknown" if isinstance(exc, TimeoutError) else "transport_or_interruption_unknown"
            write_json(cache_file, {"request": request, "status": "error", "ledger_id": reservation,
                                    "reason": reason})
            ledger.finish(reservation, status="unknown", charged_usd=None, usage={}, reason=reason)
            raise Unresolved(reason) from None


def _policy(study: Study, profile: Profile, ledger: OfflineLedger, job: dict, attempt: int,
            training: dict, source: np.ndarray, target: np.ndarray, history: list[dict], step_index: int,
            *, live: dict | None = None) -> np.ndarray:
    payload = _payload(study, profile, training, source, target)
    last_reason = "invalid_json"
    for slot in range(study.format_retries + 1):
        response, trace = _cache_call(study, profile, ledger, job, attempt, payload, slot, live=live)
        trace.update({"returned_model": response.get("model"), "usage": response.get("usage"), "reason": None,
                      "step": step_index, "response_sha256": sha(canonical_bytes(response))})
        try:
            probabilities = _decode_probabilities(response["output_text"], study.policy_temperature)
        except InvalidModelResponse as exc:
            last_reason = str(exc)
            trace["reason"] = last_reason
            history.append(trace)
            continue
        trace["probabilities"] = probabilities.tolist()
        history.append(trace)
        return probabilities
    raise InvalidModelResponse(last_reason)


def _trial_file(output: Path, job: dict) -> Path:
    return output / "trials" / f"{job['key']}.json"


def _empty_result(job: dict) -> dict:
    return {"identity": job, "status": "running", "steps": [], "requests": [], "attempts": [],
            "invalid_output_count": 0, "invalid_action_count": 0, "no_op_count": 0}


def _replay_prefix(row: dict, source: np.ndarray, target: np.ndarray, cap: int) -> tuple[np.ndarray, np.random.Generator]:
    env = environment(source)
    rng = np.random.default_rng(row["identity"]["seed"])
    current = source.copy()
    for index, step in enumerate(row["steps"]):
        if index >= cap or np.array_equal(current, target) or env.is_solved(current):
            raise ValueError("Recorded action continues after a stopping condition")
        probs = np.asarray(step["probabilities"], dtype=float)
        if probs.shape != (4,) or not np.isfinite(probs).all() or np.any(probs < 0) or not np.isclose(probs.sum(), 1):
            raise ValueError("Recorded policy distribution invalid")
        action = int(rng.choice(4, p=probs))
        if action != step["action"] or state_hash(current) != step["before_hash"]:
            raise ValueError("Recorded action or source hash disagrees with seeded replay")
        previous = current
        current = np.asarray(env.next_state(current, action))
        if state_hash(current) != step["after_hash"]:
            raise ValueError("Recorded successor hash disagrees with simulator")
        if step["changed"] != (not np.array_equal(previous, current)):
            raise ValueError("Recorded no-op flag disagrees with simulator")
    return current, rng


def _complete(row: dict, env: Any, current: np.ndarray, target: np.ndarray, cap: int) -> bool:
    reached, solved = np.array_equal(current, target), bool(env.is_solved(current))
    if not reached and not solved and len(row["steps"]) < cap:
        return False
    row.update(status="complete", exact_hit=bool(reached), task_solved=solved,
               solved_before_target=bool(solved and not reached),
               first_hit=len(row["steps"]) if reached else None, attempted_actions=len(row["steps"]),
               no_op_count=sum(not step["changed"] for step in row["steps"]),
               stop_reason="exact_hit" if reached else "solved_before_target" if solved else "miss_cap",
               endpoint_hash=state_hash(current))
    return True


def _live_client(study: Study):
    """Select exactly one configured credential source without logging secrets."""
    if study.credential_source == "dotenv":
        from dotenv import dotenv_values
        key = dotenv_values(ROOT / ".env").get("OPENAI_API_KEY")
    else:
        key = os.environ.get("OPENAI_API_KEY")
    if not key or not str(key).strip():
        raise ValueError("Configured credential source contains no OPENAI_API_KEY")
    from openai import OpenAI
    return OpenAI(api_key=str(key).strip(), base_url="https://api.openai.com/v1",
                  timeout=study.live_timeout_seconds, max_retries=0)


def run(study: Study, *, resume: bool = False, max_trials: int | None = None,
        interrupt_after_actions: int | None = None, allowed_job_keys: set[str] | None = None) -> dict:
    manifest, output = checked(study)
    if study.mode == "live" and study.n_main:
        # The N=100 contract fixes identities and streams for a later extension,
        # but this authorised pilot must never execute beyond its first 20.
        dataset = read(absolute(study.dataset_dir) / "manifest.json")
        contexts = set(dataset["n100"][:20])
        permitted = {job["key"] for job in manifest["jobs"]
                     if job["group"] == "main" and job["context"] in contexts}
        if not allowed_job_keys or not allowed_job_keys <= permitted:
            raise ValueError("Exploratory live execution requires an explicit first-20 job allowlist")
    ledger = OfflineLedger(output / "ledger.sqlite3", study.limits)
    _data, training, core, reference, boards = data_inputs(study)
    pairs = {row["pair_id"]: row for row in [*core, *reference]}
    existing = list((output / "trials").glob("*.json")) if (output / "trials").exists() else []
    if existing and not resume:
        raise ValueError("Existing trials require --resume")
    live = None
    if study.mode == "live":
        _live_configuration(study)
        from scripts.figure1_pilot.budget import Figure1Ledger
        assert study.live_campaign_config is not None
        live = {"ledger": Figure1Ledger(absolute(study.live_campaign_config)),
                "client": _live_client(study),
                "deadline": time.monotonic() + study.live_wall_seconds}
    executed, total_actions = 0, 0
    try:
        with FileLock(str(output / ".run.lock"), timeout=0):
            if allowed_job_keys is not None:
                if not allowed_job_keys <= {job["key"] for job in manifest["jobs"]}:
                    raise ValueError("Allowed job keys contain jobs outside the frozen manifest")
            for job in manifest["jobs"]:
                if allowed_job_keys is not None and job["key"] not in allowed_job_keys:
                    continue
                file = _trial_file(output, job)
                row = read(file) if file.exists() else _empty_result(job)
                if row["identity"] != job:
                    raise ValueError("Trial identity differs from frozen job")
                if row["status"] in {"complete", "invalid_model_response"}:
                    continue
                if max_trials is not None and executed >= max_trials:
                    break
                executed += 1
                source_target = policy_query(pairs[job["pair"]], boards)
                source, target = from_rows(source_target["state"]), from_rows(source_target["subgoal"])
                current, rng = _replay_prefix(row, source, target, study.action_cap)
                env = environment(source)
                row["attempts"].append({"attempt": len(row["attempts"]) + 1, "status": "running",
                                        "start_step": len(row["steps"]), "reason": None})
                for old in row["attempts"][:-1]:
                    if old["status"] == "running":
                        old.update(status="unresolved", reason="process_interrupted")
                attempt = row["attempts"][-1]["attempt"]
                row["status"] = "running"
                row.pop("reason", None)
                write_json(file, row)
                try:
                    while not _complete(row, env, current, target, study.action_cap):
                        request_start = len(row["requests"])
                        try:
                            probs = _policy(study, study.arms[job["arm"]], ledger, job, attempt,
                                            training, current, target, row["requests"], len(row["steps"]), live=live)
                        finally:
                            # Preserve request IDs even when a decision or budget fails.
                            row["invalid_output_count"] += sum(bool(r.get("reason")) for r in row["requests"][request_start:])
                            write_json(file, row)
                        action = int(rng.choice(4, p=probs))
                        next_state = np.asarray(env.next_state(current, action))
                        row["steps"].append({"action": action, "probabilities": probs.tolist(),
                            "request_hash": row["requests"][-1]["request_hash"],
                            "before_hash": state_hash(current), "after_hash": state_hash(next_state),
                            "changed": not np.array_equal(current, next_state)})
                        current = next_state
                        total_actions += 1
                        write_json(file, row)
                        if interrupt_after_actions is not None and total_actions >= interrupt_after_actions:
                            raise Unresolved("simulated_interruption")
                    row["attempts"][-1]["status"] = "complete"
                except InvalidModelResponse as exc:
                    row.update(status="invalid_model_response", exact_hit=False, task_solved=bool(env.is_solved(current)),
                               solved_before_target=False, first_hit=None, attempted_actions=len(row["steps"]),
                               no_op_count=sum(not step["changed"] for step in row["steps"]),
                               stop_reason="invalid_model_response", endpoint_hash=state_hash(current), reason=str(exc))
                    row["attempts"][-1].update(status="invalid_model_response", reason=str(exc))
                except (Unresolved, LimitExceeded) as exc:
                    row.update(status="unresolved", reason=str(exc), attempted_actions=len(row["steps"]),
                               endpoint_hash=state_hash(current), no_op_count=sum(not step["changed"] for step in row["steps"]))
                    row["attempts"][-1].update(status="unresolved", reason=str(exc))
                    write_json(file, row)
                    break
                write_json(file, row)
    except Timeout as exc:
        raise RuntimeError("Study is already running") from exc
    return status(study)


def status(study: Study) -> dict:
    manifest, output = checked(study)
    counts = Counter()
    for job in manifest["jobs"]:
        file = _trial_file(output, job)
        counts[read(file)["status"] if file.exists() else "missing"] += 1
    scope = OfflineLedger(output / "ledger.sqlite3", study.limits).summary()
    result = {"mode": study.mode, "contract_id": manifest["contract_id"], "planned": len(manifest["jobs"]),
              "counts": dict(counts), "ledger": scope}
    if study.mode == "live":
        scope["new_request_reservations"] = scope.pop("mock_new_calls")
        scope.pop("provider_physical_requests", None)
        scope.pop("provider_prompt_cached_input_tokens", None)
        scope.pop("actual_provider_cost_usd", None)
        scope["note"] = "Study scope only; the separately recorded campaign is the billing authority."
        result["campaign"] = _campaign_summary_readonly(study)
    return result


def _campaign_summary_readonly(study: Study) -> dict:
    """Inspect live campaign accounting without creating or changing a ledger."""
    import sqlite3

    if not study.live_campaign_config:
        return {"status": "missing_campaign_configuration"}
    config = read(absolute(study.live_campaign_config))
    path = absolute(config["ledger_path"])
    if not path.exists():
        return {"status": "not_created"}
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        saved = db.execute("SELECT value FROM metadata WHERE key='config_hash'").fetchone()
        if saved is None or saved[0] != sha(absolute(study.live_campaign_config).read_bytes()):
            raise ValueError("Campaign ledger does not match frozen configuration")
        records = db.execute("SELECT model,stage,status,COUNT(*),SUM(COALESCE(charged_nanos,0)),"
                             "SUM(CASE WHEN charged_nanos IS NULL THEN reserved_nanos ELSE 0 END) "
                             "FROM attempts GROUP BY model,stage,status ORDER BY model,stage,status").fetchall()
        usage_rows = db.execute("SELECT model,usage_json FROM attempts WHERE status='settled'").fetchall()
        count_rows = db.execute("SELECT status,COUNT(*),SUM(COALESCE(charged_nanos,0)),"
                                "SUM(CASE WHEN charged_nanos IS NULL THEN reserved_nanos ELSE 0 END) "
                                "FROM attempts WHERE endpoint='responses.input_tokens' GROUP BY status").fetchall()
    usage = {}
    for model, raw in usage_rows:
        data = json.loads(raw)
        entry = usage.setdefault(model, {"input_tokens": 0, "output_tokens": 0, "provider_cached_input_tokens": 0})
        entry["input_tokens"] += data.get("input_tokens", 0) or 0
        entry["output_tokens"] += data.get("output_tokens", 0) or 0
        cached = (data.get("input_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            entry["provider_cached_input_tokens"] = None
        elif entry["provider_cached_input_tokens"] is not None:
            entry["provider_cached_input_tokens"] += cached
    return {"status": "present", "by_model_stage_status": [
        {"model": model, "stage": stage, "status": state, "calls": count,
         "accounted_usd": charged / 1e9, "reserved_unsettled_usd": reserved / 1e9}
        for model, stage, state, count, charged, reserved in records], "settled_usage": usage,
        "input_token_counter": [{"status": state, "calls": count, "assumed_usd": assumed / 1e9,
                                  "reserved_unsettled_usd": reserved / 1e9}
                                 for state, count, assumed, reserved in count_rows],
        "cost_note": "Generation accounting is conservative when cache-write usage is unavailable. "
                     "Input-token counting is separate and its assumed charge is not observed provider billing; "
                     "counter pricing is unverified. Neither is an invoice."}


def verify(study: Study) -> dict:
    """Independently replay every recorded action and validate cached policy vectors."""
    from scripts.reachability_runner.independent_verify import check_action_rng, replay_actions

    manifest, output = checked(study)
    _data, training, core, reference, boards = data_inputs(study)
    pairs = {row["pair_id"]: row for row in [*core, *reference]}
    import sqlite3
    with sqlite3.connect((output / "ledger.sqlite3").resolve().as_uri() + "?mode=ro", uri=True) as db:
        ledger_rows = {r[0]: r[1:] for r in db.execute(
            "SELECT id,job,request_hash,slot,status,usage_json,cache_hit,attempt,stage,arm,reserved_nanos,charged_nanos FROM calls")}
    campaign_rows = None
    input_count_verification = None
    if study.mode == "live" or study.replay_source == "live":
        if not study.live_campaign_config:
            raise ValueError("Live-response verification requires its campaign ledger")
        campaign_config = read(absolute(study.live_campaign_config))
        campaign_path = absolute(campaign_config["ledger_path"])
        if not campaign_path.exists():
            raise ValueError("Live campaign ledger is missing; refusing to reset accounting")
        if campaign_config.get("input_token_reservation") is not None:
            from scripts.figure1_pilot.budget import Figure1Ledger
            input_count_verification = Figure1Ledger.verify_input_counts_readonly(campaign_config, campaign_path)
        with sqlite3.connect(campaign_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            saved = db.execute("SELECT value FROM metadata WHERE key='config_hash'").fetchone()
            if saved is None or saved[0] != sha(absolute(study.live_campaign_config).read_bytes()):
                raise ValueError("Campaign ledger does not match frozen configuration")
            campaign_rows = {r[0]: r[1:] for r in db.execute(
                "SELECT a.id,a.model,a.request_hash,a.status,a.usage_json,c.payload_sha256,a.reserved_nanos,a.charged_nanos,c.response_sha256 "
                "FROM attempts a JOIN figure1_contracts c ON c.attempt_id=a.id")}
    counts = Counter()
    expected_names = {job["key"] + ".json" for job in manifest["jobs"]}
    actual_names = {path.name for path in (output / "trials").glob("*.json")}
    if actual_names - expected_names:
        raise ValueError("Unexpected trial identifier or file")
    for job in manifest["jobs"]:
        file = _trial_file(output, job)
        if not file.exists():
            counts["missing"] += 1
            continue
        row = read(file)
        if row["identity"] != job:
            raise ValueError("Trial identity differs from frozen manifest")
        query = policy_query(pairs[job["pair"]], boards)
        source, target = from_rows(query["state"]), from_rows(query["subgoal"])
        if state_hash(source) != job["source_hash"] or state_hash(target) != job["target_hash"]:
            raise ValueError("Manifest state hash differs from dataset")
        replay = replay_actions(source, target, row["steps"], study.action_cap)
        check_action_rng(row["steps"], job["seed"])
        if row.get("endpoint_hash") is not None and row["endpoint_hash"] != replay["endpoint_hash"]:
            raise ValueError("Recorded endpoint differs from replay")
        if row.get("attempted_actions") is not None and row["attempted_actions"] != len(row["steps"]):
            raise ValueError("Attempted actions differ from recorded prefix")
        if row.get("no_op_count") is not None and row["no_op_count"] != sum(not s["changed"] for s in row["steps"]):
            raise ValueError("No-op count differs from recorded prefix")
        profile = study.arms[job["arm"]]
        profile_hash = sha(canonical_bytes(profile.model_dump(mode="json")))
        cache_kind = "mock" if study.mode == "mock" or study.replay_source == "mock" else "live"
        replay_states = replay["states"]
        for trace in row["requests"]:
            path = absolute(study.cache_dir) / cache_kind / f"{trace['request_hash']}.json"
            cached = read(path)
            if (cached.get("status") != "completed" or sha(canonical_bytes(cached["request"])) != trace["request_hash"]
                    or cached["request"]["profile_sha256"] != profile_hash
                    or cached["request"]["endpoint"] != profile.endpoint
                    or cached["request"]["slot"] != trace["slot"]
                    or trace.get("response_sha256") != sha(canonical_bytes(cached))
                    or trace["step"] >= len(replay_states)
                    or cached["request"]["payload"] != _payload(study, profile, training, replay_states[trace["step"]], target)
                    or trace.get("returned_model") != cached.get("model")
                    or trace.get("usage") != cached.get("usage")):
                raise ValueError("Recorded response cache differs from request")
            ledger_entry = ledger_rows.get(trace["ledger_id"])
            if ledger_entry is None or ledger_entry[:3] != (job["key"], trace["request_hash"], trace["slot"]):
                raise ValueError("Recorded request is absent from the study ledger")
            if (ledger_entry[3] != ("cache_hit" if trace["cache_hit"] else "settled")
                    or json.loads(ledger_entry[4]) != trace.get("usage")
                    or bool(ledger_entry[5]) != trace["cache_hit"]
                    or ledger_entry[7:9] != (job["group"], job["arm"])):
                raise ValueError("Ledger usage, status or identity differs from request trace")
            if campaign_rows is not None:
                from scripts.figure1_pilot.budget import _payload_hash
                from scripts.figure1_pilot.transport import response_digest

                campaign_id = cached.get("campaign_reservation_id")
                saved = campaign_rows.get(campaign_id)
                if (not campaign_id or trace.get("campaign_reservation_id") != campaign_id or saved is None
                        or saved[0] != profile.model_id or saved[1] != trace["request_hash"] or saved[2] != "settled"
                        or json.loads(saved[3]) != cached["usage"]
                        or saved[4] != _payload_hash(cached["request"]["payload"])
                        or saved[7] != response_digest(cached)
                        or cached.get("requested_model") != profile.model_id
                        or cached.get("model") not in (profile.allowed_response_model_ids or ())
                        or saved[6] is None or not 0 <= saved[6] <= saved[5]
                        or ledger_entry[10] is None
                        or (trace["cache_hit"] and (ledger_entry[9] != 0 or ledger_entry[10] != 0))
                        or (not trace["cache_hit"] and (ledger_entry[9] != saved[5] or ledger_entry[10] != saved[6]))):
                    raise ValueError("Campaign, resolved model or usage differs from response cache")
            try:
                vector = _decode_probabilities(cached["output_text"], study.policy_temperature)
            except InvalidModelResponse as exc:
                if trace.get("reason") != str(exc):
                    raise ValueError("Invalid response reason differs from replay") from exc
            else:
                if trace.get("reason") is not None or not np.allclose(vector, trace.get("probabilities")):
                    raise ValueError("Recorded distribution differs from response")
        for index, step in enumerate(row["steps"]):
            matches = [t for t in row["requests"] if t.get("step") == index
                       and t.get("request_hash") == step.get("request_hash") and t.get("reason") is None]
            if not matches or not np.allclose(step["probabilities"], matches[-1]["probabilities"]):
                raise ValueError("Step has no matching recorded policy distribution")
        status_name = row["status"]
        if status_name == "complete":
            if replay["stop_reason"] is None:
                raise ValueError("Completed trial has no valid stop condition")
            for field in ("exact_hit", "task_solved", "solved_before_target", "first_hit", "attempted_actions", "no_op_count", "stop_reason", "endpoint_hash"):
                if row.get(field) != replay[field]:
                    raise ValueError(f"Trial result differs from simulator: {field}")
        elif status_name == "invalid_model_response":
            if row.get("stop_reason") != status_name or row.get("first_hit") is not None or replay["stop_reason"] is not None:
                raise ValueError("Invalid response status is inconsistent")
            if row.get("exact_hit") is not False or row.get("task_solved") != replay["task_solved"] or row.get("solved_before_target") is not False:
                raise ValueError("Invalid response outcome differs from simulator")
            last_attempt = row["attempts"][-1]["attempt"] if row["attempts"] else None
            final_requests = [trace for trace in row["requests"] if trace.get("step") == len(row["steps"])
                              and ledger_rows[trace["ledger_id"]][6] == last_attempt]
            if ([trace["slot"] for trace in final_requests] != list(range(study.format_retries + 1))
                    or any(trace.get("reason") not in {"invalid_json", "invalid_vector"} for trace in final_requests)):
                raise ValueError("Invalid response lacks recorded failure evidence")
        elif status_name not in {"running", "unresolved"}:
            raise ValueError("Unknown trial status")
        counts[status_name] += 1
    return {"verified_files": sum(count for key, count in counts.items() if key != "missing"),
            "counts": dict(counts), "mode": study.mode,
            "input_count_verification": input_count_verification,
            "note": "Replay verifies saved observations and cache; it does not create additional trials."}


def _layout(row: dict) -> str:
    return row.get("job", {}).get("layout", "__single_unspecified_layout__")


def _metric_row(rows: list[dict], *, arm: str, stage: str, k: int) -> dict:
    """Complete-cohort estimand and missing-data bounds, with equal layout weight."""
    hits = sum(row.get("stop_reason") == "exact_hit" and row.get("status") == "complete" for row in rows)
    misses = sum(row.get("stop_reason") in {"miss_cap", "solved_before_target"} and row.get("status") == "complete" for row in rows)
    invalid = sum(row.get("status") == "invalid_model_response" for row in rows)
    times = [row["first_hit"] for row in rows if row.get("status") == "complete" and row.get("exact_hit")]
    total, resolved = len(rows), hits + misses + invalid
    unresolved = total - resolved
    layouts: dict[str, list] = {}
    for row in rows:
        layouts.setdefault(_layout(row), []).append(row)
    lower, upper, estimates = [], [], []
    for group in layouts.values():
        h = sum(_score(row) == 1 for row in group)
        q = sum(_score(row) is None for row in group)
        lower.append(h / len(group))
        upper.append((h + q) / len(group))
        if not q:
            estimates.append(h / len(group))
    complete = total > 0 and unresolved == 0
    return {"arm": arm, "stage": stage, "k": k, "status": "COMPLETE" if complete else "PARTIAL",
            "planned": total, "planned_layouts": len(layouts),
            "resolved_layouts": len(estimates), "hits": hits, "valid_misses": misses,
            "invalid_model_response": invalid, "unresolved": unresolved, "coverage": resolved / total if total else None,
            "operational_p": float(np.mean(estimates)) if complete else None,
            "resolved_p_descriptive": hits / resolved if resolved else None,
            "valid_rollout_p": hits / (hits + misses) if hits + misses else None,
            "mean_first_hit_given_hit": float(np.mean(times)) if times else None,
            "format_failure_rate": invalid / resolved if resolved else None,
            "lower_bound": float(np.mean(lower)) if lower else None,
            "upper_bound": float(np.mean(upper)) if upper else None,
            "bound_unit": "equal-layout mean; unresolved trials range from failure to success",
            "bootstrap_95_low": None, "bootstrap_95_high": None,
            "zero_hit_warning": bool(complete and hits == 0)}


def _score(row: dict) -> int | None:
    if row.get("status") == "invalid_model_response":
        return 0
    if row.get("status") == "complete":
        return int(row["exact_hit"])
    return None


def paired_metrics(rows: list[dict], study: Study, contract_id: str) -> list[dict]:
    """Bootstrap whole layouts once per stage, jointly across k, arms and repetitions.

    Intervals are withheld for a partial stage, a mismatched layout grid, or fewer
    than two layouts. Bounds use equal layout weights and do not impute missing data.
    """
    result = []
    for stage in ("pilot", "main"):
        stage_rows = [row for row in rows if row["job"]["group"] == stage]
        if not stage_rows:
            continue
        all_layouts = sorted({_layout(row) for row in stage_rows})
        stage_complete = all(_score(row) is not None for row in stage_rows)
        stage_complete = stage_complete and all(
            {_layout(row) for row in stage_rows if row["job"]["k"] == k} == set(all_layouts)
            for k in study.core_offsets
        )
        stage_pairs: dict[tuple, set] = {}
        for row in stage_rows:
            job = row["job"]
            key = (job["k"], job["layout"], job["context"], job["pair"], job["repetition"])
            stage_pairs.setdefault(key, set()).add(job["arm"])
        stage_complete = stage_complete and all(arms == {"luna", "terra_6_1"} for arms in stage_pairs.values())
        indices = None
        if stage_complete and len(all_layouts) >= 2:
            seed = int(sha(canonical_bytes([contract_id, "paired_layout_bootstrap_v2", stage]))[:16], 16)
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(all_layouts), size=(10000, len(all_layouts)))
        for k in study.core_offsets:
            index: dict[tuple, dict] = {}
            for row in stage_rows:
                job = row["job"]
                if job["k"] == k:
                    key = (job["layout"], job["context"], job["pair"], job["repetition"])
                    index.setdefault(key, {})[job["arm"]] = _score(row)
            if not index:
                continue
            bounds: dict[str, list] = {}
            complete: dict[str, list] = {}
            per_arm: dict[str, dict[str, list]] = {"luna": {}, "terra_6_1": {}}
            for key, arms in index.items():
                layout = key[0]
                a, b = arms.get("luna"), arms.get("terra_6_1")
                bounds.setdefault(layout, []).append(((b if b is not None else 0) - (a if a is not None else 1),
                                                       (b if b is not None else 1) - (a if a is not None else 0)))
                if a is not None and b is not None:
                    complete.setdefault(layout, []).append(b - a)
                for arm, value in (("luna", a), ("terra_6_1", b)):
                    if value is not None:
                        per_arm[arm].setdefault(layout, []).append(value)
            complete_pairs = sum(len(values) for values in complete.values())
            fully_complete = complete_pairs == len(index)
            descriptive = float(np.mean([np.mean(v) for v in complete.values()])) if complete else None
            record = {"stage": stage, "k": k, "status": "COMPLETE" if fully_complete else "PARTIAL",
                "planned_pairs": len(index), "resolved_pairs": complete_pairs,
                "planned_layouts": len(bounds), "resolved_layouts": sum(
                    len(complete.get(layout, [])) == len(values) for layout, values in bounds.items()),
                "pair_coverage": complete_pairs / len(index),
                "delta_terra_minus_luna": descriptive if fully_complete else None,
                "resolved_delta_descriptive": descriptive,
                "bootstrap_95_low": None, "bootstrap_95_high": None,
                "full_cohort_lower_bound": float(np.mean([np.mean([x[0] for x in values]) for values in bounds.values()])),
                "full_cohort_upper_bound": float(np.mean([np.mean([x[1] for x in values]) for values in bounds.values()])),
                "bound_unit": "equal-layout mean of paired trial difference bounds",
                "bootstrap_repetitions": 10000, "bootstrap_unit": "layout; shared across k and arms",
                "luna_bootstrap_95_low": None, "luna_bootstrap_95_high": None,
                "terra_6_1_bootstrap_95_low": None, "terra_6_1_bootstrap_95_high": None}
            if indices is not None and fully_complete and set(bounds) == set(all_layouts):
                values = np.array([np.mean(complete[layout]) for layout in all_layouts])
                record["bootstrap_95_low"], record["bootstrap_95_high"] = (
                    float(x) for x in np.quantile(values[indices].mean(axis=1), [.025, .975]))
                for arm in ("luna", "terra_6_1"):
                    values = np.array([np.mean(per_arm[arm][layout]) for layout in all_layouts])
                    record[f"{arm}_bootstrap_95_low"], record[f"{arm}_bootstrap_95_high"] = (
                        float(x) for x in np.quantile(values[indices].mean(axis=1), [.025, .975]))
            result.append(record)
    return result


def calibration_metrics(rows: list[dict]) -> list[dict]:
    """Keep zero-hit and partial targets, and label only complete reference pools."""
    result = []
    grouped: dict[tuple, list] = {}
    for row in rows:
        job = row["job"]
        if job["group"] in {"calibration_A", "calibration_B"}:
            grouped.setdefault((job["group"], job["arm"], job["context"]), []).append(row)
    for (stage, arm, context), group in sorted(grouped.items()):
        targets = []
        for pair in sorted({row["job"]["pair"] for row in group}):
            trials = [row for row in group if row["job"]["pair"] == pair]
            complete = all(_score(row) is not None for row in trials)
            times = [row["first_hit"] for row in trials if _score(row) == 1]
            targets.append({"arm": arm, "group": stage, "context": context, "pair": pair,
                "planned": len(trials), "hits": len(times), "complete": complete,
                "invalid_model_response": sum(row["status"] == "invalid_model_response" for row in trials),
                "p": len(times) / len(trials) if complete and trials else None,
                "d": float(np.mean(times)) if times else None, "weight": None, "u": None,
                "cdf_before": None, "cdf_at": None, "context_status": None, "exclusion_reason": None})
        context_complete = all(target["complete"] for target in targets)
        for target in targets:
            target["context_status"] = "COMPLETE" if context_complete else "PARTIAL"
            target["exclusion_reason"] = "partial_reference_pool" if not context_complete else (
                "zero_exact_hits" if not target["hits"] else None)
        reached = [target for target in targets if target["hits"]]
        if context_complete and reached:
            weights, labels = weighted_percentiles([target["d"] for target in reached], [target["p"] for target in reached])
            for item, weight, label in zip(reached, weights, labels, strict=True):
                lower = sum(float(w) for other, w in zip(reached, weights, strict=True) if other["d"] < item["d"])
                at = sum(float(w) for other, w in zip(reached, weights, strict=True) if other["d"] <= item["d"])
                item.update(weight=float(weight), u=float(label), cdf_before=lower, cdf_at=at)
        result.extend(targets)
    return result


def export(study: Study) -> dict:
    """Export complete-cohort estimates, bounds, paired layout intervals and calibration."""
    verification = verify(study)
    manifest, output = checked(study)
    rows = []
    for job in manifest["jobs"]:
        file = _trial_file(output, job)
        row = read(file) if file.exists() else {"status": "missing"}
        rows.append({"job": job, **row})
    metrics = []
    for stage in ("pilot", "main"):
        for arm in study.arms:
            for k in study.core_offsets:
                group = [row for row in rows if row["job"]["group"] == stage and row["job"]["arm"] == arm and row["job"]["k"] == k]
                if group:
                    metrics.append(_metric_row(group, arm=arm, stage=stage, k=k))
    cal = calibration_metrics(rows)
    paired = paired_metrics(rows, study, manifest["contract_id"])
    for item in metrics:
        matched = next((p for p in paired if p["stage"] == item["stage"] and p["k"] == item["k"]), None)
        if matched:
            item["bootstrap_95_low"] = matched[f"{item['arm']}_bootstrap_95_low"]
            item["bootstrap_95_high"] = matched[f"{item['arm']}_bootstrap_95_high"]
    def write_csv(file: Path, items: list[dict], fallback: list[str]) -> None:
        with file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]) if items else fallback)
            writer.writeheader()
            writer.writerows(items)
    write_csv(output / "core_metrics.csv", metrics, ["arm", "stage", "k"])
    write_csv(output / "calibration_metrics.csv", cal, ["arm", "group", "context", "pair"])
    write_csv(output / "paired_metrics.csv", paired, ["stage", "k"])
    summary = {"mode": study.mode, "scientific_observation": study.mode == "live",
               "verification": verification, "status": status(study), "core": metrics, "paired": paired, "calibration": cal,
               "note": "Mock is not a model result. Replay reconstructs the same trial identities, not new observations. "
                       "Action sampling is conditional on cached policy distributions. Bounds are not confidence intervals. "
                       "An all-zero bootstrap does not prove zero population probability."}
    write_json(output / "export.json", summary)
    return summary
