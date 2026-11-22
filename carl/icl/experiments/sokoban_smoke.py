"""Prepare, preview and evaluate a small matched Sokoban predicate sweep offline."""

import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows
from carl.icl.config import ROOT, ExperimentConfig, Limits, PredicateICLSettings, PredicatePlanningSettings, Settings, load_config
from carl.icl.evaluation.matched import MatchedSweep
from carl.icl.intent_annotations import DA_VERSION, prepare_annotations
from carl.icl.intent_domain import DOMAIN_VERSION, TASK_GOAL, decode_intent
from carl.icl.intents.generation.data import file_sha256, read_records
from carl.icl.planning.intent_search import SearchMode
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.predicate_icl import MockResponses, PredicateCondition, PredicateQuery
from carl.icl.types import JSONObject

DEFAULT_CONFIG = Path("configs/icl/quickstart/sokoban_sweep.json")
MODES = (SearchMode.DIRECT, SearchMode.ONE_STEP, SearchMode.FULL_SEARCH)
Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]*$")]
Seed = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class Profile(Settings):
    """Resolve an existing config and assert its expected example count."""

    config: Path
    shots: Literal[0, 4]


class Budget(Settings):
    """Set the planner action cap for all arms in one comparison."""

    id: Name
    action_budget: PositiveInt


class SmokeConfig(Settings):
    """Inputs for an offline demonstration with explicit held-out queries."""

    schema_version: Literal[1]
    demonstrations: Path
    test_queries: Path
    default_profile: Name
    profiles: dict[Name, Profile]
    annotation_seed: Seed
    seeds: list[Seed] = Field(min_length=1)
    budgets: list[Budget] = Field(min_length=2)
    limits: Limits
    max_expansions: PositiveInt
    rollout_cap: PositiveInt
    mock: MockResponses

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        """Reject ambiguous arm definitions and incomplete fixture responses."""
        if self.default_profile not in self.profiles:
            raise ValueError("default_profile must name an existing profile")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("repeat seeds must be unique")
        if len({budget.id for budget in self.budgets}) != len(self.budgets):
            raise ValueError("budget IDs must be unique")
        if len({budget.action_budget for budget in self.budgets}) != len(self.budgets):
            raise ValueError("action budgets must be distinct")
        if None in (self.mock.policy, self.mock.generator, self.mock.value):
            raise ValueError("offline inference requires policy, generator and value responses")
        return self


def _read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _query(rows: list[str], split: Literal["test", "validation"], group: str) -> PredicateQuery:
    return PredicateQuery(observation={"rows": rows}, task_goal=TASK_GOAL, split=split, task_group=group)


@dataclass(frozen=True)
class FixtureInputs:
    """Checked sources and separated inference queries, with no test labels."""

    demonstrations: Path
    test_queries: Path
    base_config: ExperimentConfig
    test_rows: dict[str, list[str]]

    @classmethod
    def load(cls, config: SmokeConfig, config_path: Path, profile: str) -> Self:
        base = config_path.parent
        demonstrations = base / config.demonstrations
        test_queries = base / config.test_queries
        selected = config.profiles[profile]
        experiment = load_config((base / selected.config).resolve())
        if experiment.predicate_icl is not None and (
            experiment.predicate_icl.scale_arm != "unscaled"
            or experiment.predicate_icl.local_roles is not None
            or (experiment.predicate_icl.policy_shots, experiment.predicate_icl.generator_shots) != (selected.shots, selected.shots)
        ):
            raise ValueError("smoke profiles require the specified shots, unscaled checked data and API roles")
        rows = TypeAdapter(dict[Name, list[str]]).validate_python(_read(test_queries))
        layouts = {_layout(rows) for rows in rows.values()}
        if len(layouts) < 2 or len(layouts) != len(rows):
            raise ValueError("the fixture needs two or more distinct test layouts")
        return cls(demonstrations, test_queries, experiment, rows)


def _layout(rows: Sequence[str]) -> str:
    return canonical_layout_id(from_rows(list(rows)))


@dataclass(frozen=True)
class PreparedData:
    """Recorded predicate inputs and their split identity."""

    data: Path
    validation_query: Path
    test_queries: dict[str, Path]
    validation_id: str
    mock: Path
    intent: Path
    split_fingerprint: str


def _prepare(config: SmokeConfig, source: FixtureInputs, output: Path) -> PreparedData:
    annotations = output / "annotations"
    audit = prepare_annotations(source.demonstrations, annotations, max_offset=1, seed=config.annotation_seed)
    data = annotations / "D_A.jsonl"
    checked = read_records(data)
    validation = [row for row in checked if row["split"] == "validation"]
    if len(validation) != 1 or audit["accepted"] != audit["attempted"] or any(row["split"] == "test" for row in checked):
        raise ValueError("smoke demonstrations need verified train and exactly one validation example")
    anchor = validation[0]
    validation_query = output / "validation-query.json"
    group = anchor.get("task_group") or anchor["source_id"]
    write_json(validation_query, _query(anchor["start_observation"], "validation", group).model_dump(mode="json"))
    test_queries = {}
    test_layouts = []
    for identifier, rows in sorted(source.test_rows.items()):
        test_queries[identifier] = output / "queries" / f"{identifier}.json"
        write_json(test_queries[identifier], _query(rows, "test", identifier).model_dump(mode="json"))
        test_layouts.append(_layout(rows))
    known_layouts = {row["layout_id"] for row in checked}
    known_groups = {row.get("task_group") or row["source_id"] for row in checked}
    if known_layouts.intersection(test_layouts) or known_groups.intersection(test_queries):
        raise ValueError("test queries overlap a training or validation layout or group")
    mock = output / "mock-responses.json"
    write_json(mock, config.mock.model_dump(mode="json"))
    intent = output / "preview-intent.json"
    assert config.mock.generator is not None
    write_json(intent, json.loads(config.mock.generator))
    split_fingerprint = contract_hash({
        "annotations": [{key: row[key] for key in ("record_id", "source_id", "layout_id", "split")} for row in checked],
        "tests": {identifier: _read(path) for identifier, path in test_queries.items()},
    })
    return PreparedData(data, validation_query, test_queries, anchor["record_id"], mock, intent, split_fingerprint)


def _condition(
    config: SmokeConfig, source: FixtureInputs, data: PreparedData, *,
    profile: str, budget: Budget, seed: int, output: Path,
) -> ExperimentConfig:
    experiment = source.base_config.model_dump(mode="json")
    shot_count = config.profiles[profile].shots
    settings = PredicateICLSettings.model_validate({
        **experiment.get("predicate_icl", {}), "scale_arm": "unscaled", "validation_id": data.validation_id,
        "policy_shots": shot_count, "generator_shots": shot_count, "context_limit": 8192,
        "planning": PredicatePlanningSettings(scales=(None,), rollout_cap=config.rollout_cap).model_dump(mode="json"),
    })
    return ExperimentConfig.model_validate({
        **experiment, "predicate_icl": settings.model_dump(mode="json"), "seed": seed,
        "action_budget": budget.action_budget, "max_expansions": config.max_expansions,
        "limits": config.limits.model_dump(mode="json"), "cache_dir": str(output / "cache"),
        "output_dir": str(output),
    })


@dataclass(frozen=True)
class Condition:
    """Bind one sweep condition to the production predicate runner."""

    predicate: PredicateCondition
    data: PreparedData

    @classmethod
    def freeze(cls, config: ExperimentConfig, prepared: PreparedData, path: Path, *, support_seed: int) -> Self:
        write_json(path / "experiment.json", config.model_dump(mode="json"))
        predicate = PredicateCondition(config, prepared.data, path / "support.json")
        predicate.freeze(support_seed=support_seed)
        return cls(predicate, prepared)

    def preview(self, query: Path, output: Path) -> None:
        result = self.predicate.preview(
            PredicateQuery.model_validate(_read(query)), decode_intent(_read(self.data.intent)),
        )
        write_json(output, result)

    def infer(self, query: Path, output: Path, *, mode: SearchMode | None = None, receipt: Path | None = None) -> JSONObject:
        result = self.predicate.infer(
            PredicateQuery.model_validate(_read(query)), mode="mock", planning_mode=mode,
            mock=MockResponses.model_validate(_read(self.data.mock)), validation_result=receipt,
        )
        write_json(output, result)
        return result


def _profile_demo(condition: Condition, output: Path) -> None:
    query = next(iter(condition.data.test_queries.values()))
    condition.preview(query, output / "preview.json")
    receipt = output / "validation.json"
    validation = condition.infer(condition.data.validation_query, receipt)
    held_out = condition.infer(query, output / "held-out.json", receipt=receipt)
    if any(trial["status"] != "achieved" or trial["metrics"]["physical_requests"] for trial in (validation, held_out)):
        raise ValueError("mock profile must achieve its fixture intent without physical requests")


def _limits(experiment: ExperimentConfig) -> JSONObject:
    predicate = experiment.predicate_icl
    assert predicate is not None and predicate.planning is not None
    api = experiment.limits.model_dump(mode="json")
    wall_seconds = api.pop("wall_seconds")
    return {"action_budget": experiment.action_budget, "wall_seconds": wall_seconds, "max_expansions": experiment.max_expansions,
            "api": api, **predicate.planning.model_dump(mode="json", exclude={"value_checkpoint"})}


@dataclass
class SweepOutput:
    """Accumulate exact run identities across budgets and repeats."""

    budgets: dict[str, JSONObject]
    instances: dict[str, JSONObject]
    trials: list[JSONObject]

    def record(
        self, *, result: JSONObject, budget: Budget, limits: JSONObject, source_id: str,
        seed: int, mode: SearchMode, artifact: str,
    ) -> None:
        policy = result["components"]["policy"]
        initial = result["nodes"][0]["state"]["snapshot"]
        if any(result["planning"][name] != limit for name, limit in limits.items() if name != "api"):
            raise ValueError("planned limits differ from the settings recorded by the planner")
        identity = {"id": f"{source_id}-seed-{seed}", "task_id": source_id, "group": _layout(initial["rows"]), "seed": seed,
                    "policy_fingerprint": policy["fingerprint"], "initial_fingerprint": contract_hash(initial)}
        if self.instances.setdefault(identity["id"], identity) != identity:
            raise ValueError("test identity, initial state or policy changed across planning arms")
        if self.budgets.setdefault(budget.id, limits) != limits:
            raise ValueError("actual limits changed within a configured budget")
        accounting = result["accounting"]
        if accounting["physical_requests"]:
            raise ValueError("offline sweep unexpectedly made a physical request")
        self.trials.append({
            "budget_id": budget.id, "instance_id": identity["id"], "task_id": source_id, "seed": seed, "mode": mode.value,
            "policy_fingerprint": identity["policy_fingerprint"], "initial_fingerprint": identity["initial_fingerprint"],
            "status": result["status"], "solved": result["solved"], "finish_reason": result["finish_reason"],
            "elapsed_seconds": accounting["wall_seconds"], "cost_usd": 0.0, "limits": limits,
            "accounting": accounting, "artifact": artifact,
        })


def _run_budget(
    sweep: SweepOutput, experiment: ExperimentConfig, prepared: PreparedData, budget: Budget, output: Path,
    *, support_seed: int,
) -> None:
    trial_path = output / "runs" / budget.id / f"seed-{experiment.seed}"
    condition = Condition.freeze(experiment, prepared, trial_path, support_seed=support_seed)
    limits = _limits(experiment)
    for mode in MODES:
        mode_path = trial_path / mode.value
        receipt = mode_path / "validation.json"
        if not condition.infer(prepared.validation_query, receipt, mode=mode)["solved"]:
            raise ValueError(f"validation failed for {budget.id}, seed {experiment.seed}, mode {mode}")
        for source_id, query in prepared.test_queries.items():
            artifact = mode_path / f"{source_id}.json"
            result = condition.infer(query, artifact, mode=mode, receipt=receipt)
            sweep.record(result=result, budget=budget, limits=limits, source_id=source_id, seed=experiment.seed,
                         mode=mode, artifact=artifact.relative_to(output).as_posix())


def run(config_path: Path, output: Path, *, profile: str | None = None, prepare_only: bool = False) -> Path:
    """Publish a fresh reproducible run, protecting previous outputs from overrides."""
    config_path = config_path.resolve()
    config = SmokeConfig.model_validate(_read(config_path))
    selected = profile or config.default_profile
    if selected not in config.profiles:
        raise ValueError(f"unknown smoke profile: {selected}")
    source = FixtureInputs.load(config, config_path, selected)
    support_seed = source.base_config.seed
    output = output.resolve()
    try:
        output.mkdir(parents=True)
    except FileExistsError as error:
        raise ValueError(f"output already exists: {output}; select a fresh output for a rerun or config override") from error
    frozen = {
        "design": config.model_dump(mode="json"), "selected_profile": selected,
        "input_hashes": {name: file_sha256(path) for name, path in (
            ("demonstrations", source.demonstrations), ("test_queries", source.test_queries), ("uv.lock", ROOT / "uv.lock"),
        )},
        "base_experiment": source.base_config.model_dump(mode="json"), "prepare_only": prepare_only,
        "support_seed": support_seed,
    }
    write_json(output / "frozen-config.json", {**frozen, "fingerprint": contract_hash(frozen)})
    prepared = _prepare(config, source, output)
    profile_path = output / "profile"
    demo = _condition(config, source, prepared, profile=selected, budget=max(config.budgets, key=lambda item: item.action_budget),
                      seed=config.seeds[0], output=profile_path)
    _profile_demo(Condition.freeze(demo, prepared, profile_path, support_seed=support_seed), profile_path)
    if prepare_only:
        return profile_path / "held-out.json"
    sweep = SweepOutput({}, {}, [])
    for budget in config.budgets:
        for seed in config.seeds:
            run_path = output / "runs" / budget.id / f"seed-{seed}"
            experiment = _condition(config, source, prepared, profile=selected, budget=budget, seed=seed, output=run_path)
            _run_budget(sweep, experiment, prepared, budget, output, support_seed=support_seed)
    artifact = {
        "schema_version": 1, "domain": "sokoban", "execution_mode": "mock", "modes": [mode.value for mode in MODES],
        "budgets": [{"id": identifier, "limits": limits} for identifier, limits in sweep.budgets.items()],
        "instances": list(sweep.instances.values()), "trials": sweep.trials, "split_fingerprint": prepared.split_fingerprint,
        "offline_compute_seconds": None, "adaptation_seconds": None, "config": frozen,
        "versions": {"python": platform.python_version(), "annotations": DA_VERSION, "domain": DOMAIN_VERSION,
                     **{package: version(package) for package in ("carl", "pydantic", "openai", "tiktoken")}},
    }
    path = output / "sweep.json"
    MatchedSweep.model_validate(artifact)
    write_json(path, artifact)
    return path
