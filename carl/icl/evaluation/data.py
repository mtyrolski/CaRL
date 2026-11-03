"""Validated run discovery with explicit comparison boundaries and provenance."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from carl.icl.config import ROOT, ExperimentConfig, Settings
from carl.icl.types import JSONObject, Provider


class EvaluationConfig(Settings):
    """Local input roots and reproducible evaluation/export settings."""

    run_dirs: list[str] = Field(min_length=1)
    output_dir: str = "analysis/luna_icl/evaluation"
    bootstrap_replicates: int = Field(default=2000, ge=100, le=100000)
    bootstrap_seed: int = Field(default=0, ge=0)
    budget_points: int = Field(default=65, ge=2, le=1000)
    comparison_methods: tuple[str, str] = ("relative", "fixed")
    comparison_configs: list[tuple[str, str]] = Field(default_factory=list)
    max_board_viewers: int = Field(default=4, ge=0, le=40)
    max_viewer_attempts: int = Field(default=20, ge=1, le=200)
    static_formats: list[Literal["png", "svg", "pdf"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_comparisons(self) -> "EvaluationConfig":
        """Reject self-comparisons and repeated requested contrasts."""
        if self.comparison_methods[0] == self.comparison_methods[1]:
            raise ValueError("Comparison methods must differ")
        if any(left == right for left, right in self.comparison_configs):
            raise ValueError("A condition cannot be compared with itself")
        if len(set(self.comparison_configs)) != len(self.comparison_configs):
            raise ValueError("Duplicate explicit comparisons")
        return self


class RunRecord(Settings):
    """The fields exported by the all-Luna runner; token counters retain their meaning."""

    config: str
    method: str
    family: str
    split: str
    instance: str
    seed: int
    provider: Provider
    tuning_status: str
    solved: bool
    finish_reason: str
    attempted_actions: int = Field(ge=0)
    solution_actions: int | None = Field(ge=0)
    expansions: int = Field(ge=0)
    seconds: float = Field(ge=0)
    requests: int = Field(ge=0)
    response_requests: int = Field(ge=0)
    embedding_requests: int = Field(ge=0)
    cache_hits: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(ge=0)
    cache_write_tokens: int = Field(ge=0)
    api_seconds: float = Field(ge=0)
    logical_requests: int = Field(ge=0)
    logical_input_tokens: int = Field(ge=0)
    logical_output_tokens: int = Field(ge=0)
    generator_calls: int = Field(ge=0)
    policy_calls: int = Field(ge=0)
    value_calls: int = Field(ge=0)

    @model_validator(mode="after")
    def check_outcome(self) -> "RunRecord":
        """Reject contradictory completion flags and fabricated failure lengths."""
        if self.solved != (self.finish_reason == "solved"):
            raise ValueError("Solved flag and finish reason disagree")
        if self.solved != (self.solution_actions is not None):
            raise ValueError("Only solved runs may have a solution length")
        if (
            self.solution_actions is not None
            and self.solution_actions > self.attempted_actions
        ):
            raise ValueError("Solution length exceeds all attempted actions")
        return self


@dataclass(frozen=True)
class SavedRun:
    """One outcome plus the resolved settings and saved execution details."""

    record: RunRecord
    settings: ExperimentConfig
    directory: Path
    detail: JSONObject

    @property
    def cohort(self) -> str:
        """Separate every non-matched setting, provider and frozen policy contract."""
        ignored = {"name", "method", "scales", "fixed_cap", "output_dir", "cache_dir"}
        contract = {
            "config": self.settings.model_dump(exclude=ignored),
            "provider": self.record.provider,
            "policy": self.detail["frozen_policy"],
            "prompt_profile_recorded": self.detail.get(
                "_prompt_profile_recorded", False
            ),
        }
        return hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest()[:12]

    @property
    def label(self) -> str:
        """Expose differences that must not disappear into a pooled success rate."""
        difficulty = "one-move fixture" if self.settings.easy else "full initial state"
        return (
            f"{self.record.family} · {self.record.provider} · {difficulty} · "
            f"{self.settings.action_budget} actions · {self.settings.model} · {self.record.split}"
        )


def resolve_path(value: str) -> Path:
    """Keep path semantics consistent with inference configs, independent of cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    """Load strict evaluation settings; relative paths resolve from the repository."""
    return EvaluationConfig.model_validate_json(
        resolve_path(str(path)).read_text(encoding="utf-8")
    )


def discover_conditions(roots: list[str]) -> list[Path]:
    """Read leaf conditions, never both a matrix's aggregate CSV and its children."""
    conditions: set[Path] = set()
    for value in roots:
        root = resolve_path(value)
        if not root.is_dir():
            raise ValueError(f"Run directory does not exist: {root}")
        matches = {
            path.parent
            for path in root.rglob("results.json")
            if (path.parent / "config.json").is_file()
        }
        if not matches:
            raise ValueError(f"No condition results.json/config.json pairs in {root}")
        conditions.update(matches)
    return sorted(conditions)


def _validate_detail(run: SavedRun) -> None:
    row, detail, settings = run.record, run.detail, run.settings
    required = {
        "solved",
        "attempted_actions",
        "finish_reason",
        "instance",
        "action_path",
        "frozen_policy",
    }
    if missing := required - detail.keys():
        raise ValueError(f"Missing detail fields for {row.instance}: {sorted(missing)}")
    expected = (
        settings.name,
        settings.method,
        settings.family,
        settings.split,
        settings.tuning_status,
    )
    actual = (row.config, row.method, row.family, row.split, row.tuning_status)
    if actual != expected or row.seed not in settings.repeat_seeds:
        raise ValueError(
            f"Resolved configuration disagrees with outcome: {run.directory}"
        )
    if row.attempted_actions > settings.action_budget:
        raise ValueError("Recorded search exceeded its action budget")
    for field in ("solved", "attempted_actions", "finish_reason"):
        if detail[field] != getattr(row, field):
            raise ValueError(f"Outcome/detail mismatch for {row.instance}: {field}")
    if detail["instance"]["id"] != row.instance or not detail.get("frozen_policy"):
        raise ValueError("Missing or mismatched instance/policy identity")
    if row.solved and len(detail["action_path"]) != row.solution_actions:
        raise ValueError("Stored action path length disagrees with the solved outcome")


def load_runs(config: EvaluationConfig) -> list[SavedRun]:
    """Reject duplicate logical trials instead of silently treating reruns as new samples."""
    runs: list[SavedRun] = []
    seen: set[tuple[str, str, str, int]] = set()
    for directory in discover_conditions(config.run_dirs):
        raw_config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        settings = ExperimentConfig.model_validate(raw_config)
        records = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        if not records:
            raise ValueError(f"Condition has no outcome records: {directory}")
        for value in records:
            row = RunRecord.model_validate(value)
            # Never allow a stored identifier to escape the condition directory.
            detail_path = directory / f"{row.instance}_seed{row.seed}.json"
            if detail_path.resolve().parent != directory:
                raise ValueError("Invalid instance identifier in saved outcome")
            detail = json.loads(detail_path.read_text(encoding="utf-8"))
            detail["_prompt_profile_recorded"] = "prompts" in raw_config
            run = SavedRun(row, settings, directory, detail)
            _validate_detail(run)
            key = (run.cohort, row.config, row.instance, row.seed)
            if key in seen:
                raise ValueError(
                    f"Duplicate logical trial {key}; select one execution directory"
                )
            seen.add(key)
            runs.append(run)
    if not runs:
        raise ValueError("No completed outcome records in the selected directories")
    return runs
