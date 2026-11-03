"""Validated settings for the all-Luna experiment and optional generator snapshots."""
import json
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from carl.icl.prompt_defaults import (
    DEFAULT_ABSOLUTE,
    DEFAULT_FIXED,
    DEFAULT_GENERATOR,
    DEFAULT_POLICY,
    DEFAULT_RELATIVE,
    DEFAULT_VALUE,
    DEFAULT_VALUE_REMAINING,
)
from carl.icl.types import JSONObject, PathSetting

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_MODEL: Final[str] = "gpt-6-luna"
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"


class Settings(BaseModel):
    """Reject misspelled keys and non-finite experimental settings."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PromptSettings(Settings):
    """Frozen component instructions, saved with each experiment condition."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    policy: str = Field(default=DEFAULT_POLICY, min_length=1)
    value: str = Field(default=DEFAULT_VALUE, min_length=1)
    value_remaining: str = Field(default=DEFAULT_VALUE_REMAINING, min_length=1)
    generator: str = Field(default=DEFAULT_GENERATOR, min_length=1)
    fixed: str = Field(default=DEFAULT_FIXED, min_length=1)
    relative: str = Field(default=DEFAULT_RELATIVE, min_length=1)
    absolute: str = Field(default=DEFAULT_ABSOLUTE, min_length=1)
    ratio_d_over_ac: str = Field(
        default="ratio_d_over_ac is mean successful first-hit time divided by the training context weighted median; "
                "infer this normalized scale from examples without calculating it on the query.", min_length=1)
    normalized_offset: str = Field(
        default="normalized_offset is the trajectory action offset divided by the fixed denominator shown below; "
                "infer the label from examples without accessing a future query trajectory.", min_length=1)


class Limits(Settings):
    """Per-run request, token and time ceilings, including logical cache use."""

    max_requests: int = Field(default=128, ge=1)
    max_input_tokens: int = Field(default=16000, ge=1)
    max_total_input_tokens: int = Field(default=1000000, ge=1)
    max_output_tokens: int = Field(default=16384, ge=1)
    wall_seconds: float = Field(default=600, gt=0)
    timeout_seconds: float = Field(default=60, gt=0)


class Calibration(Settings):
    """Independent reference population, rollout and calibration budget settings."""

    reference_window: int = Field(default=16, ge=1)
    candidates: int = Field(default=8, ge=1)
    sampled_targets: int | None = Field(default=None, ge=1)
    trials: int = Field(default=4, ge=1)
    action_cap: int = Field(default=16, ge=1)
    limits: Limits = Field(
        default_factory=lambda: Limits(
            max_requests=4096,
            max_total_input_tokens=16000000,
            max_output_tokens=262144,
            wall_seconds=7200,
        )
    )


class Tuning(Settings):
    """Equal-size validation grids for the two main comparison conditions."""

    relative_scales: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.9, 0.5), (0.75, 0.25), (0.95, 0.65)]
    )
    fixed_scales: list[tuple[int, int]] = Field(
        default_factory=lambda: [(8, 4), (12, 4), (16, 8)]
    )

    @model_validator(mode="after")
    def check_grids(self) -> "Tuning":
        """Require comparable trial counts and meaningful, distinct scale pairs."""
        if not self.relative_scales or len(self.relative_scales) != len(
            self.fixed_scales
        ):
            raise ValueError(
                "Tuning grids must be nonempty and have equal numbers of candidates"
            )
        if any(not 0 < low < high < 1 for high, low in self.relative_scales):
            raise ValueError("Relative tuning pairs must satisfy 0 < low < high < 1")
        if any(not 0 < low < high for high, low in self.fixed_scales):
            raise ValueError("Fixed tuning pairs must satisfy 0 < low < high")
        if len(set(self.relative_scales)) != len(self.relative_scales) or len(
            set(self.fixed_scales)
        ) != len(self.fixed_scales):
            raise ValueError("Tuning grids must not contain duplicate pairs")
        return self


class ExperimentConfig(Settings):
    """A complete reproducible inference condition with separate calibration limits."""

    name: str = "luna_relative"
    model: str = DEFAULT_MODEL
    reasoning_effort: Literal["none", "low", "medium", "high"] = "none"
    policy_reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    json_max_retries: int = Field(default=0, ge=0, le=1)
    generator_reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    value_reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    response_format: Literal["text", "json_schema"] = "text"
    policy_response_format: Literal["text", "json_schema"] | None = None
    generator_response_format: Literal["text", "json_schema"] | None = None
    value_response_format: Literal["text", "json_schema"] | None = None
    generator_layout_schema: bool = False
    generator_max_attempts: int = Field(default=1, ge=1, le=5)
    value_target: Literal["return", "remaining_actions"] = "return"
    value_examples: Literal["initial", "trajectory"] = "initial"
    value_board_features: bool = False
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    method: Literal["relative", "relative_u", "fixed", "absolute", "absolute_d", "ratio_d_over_ac", "normalized_offset", "primitive", "direct"] = "relative"
    offset_denominator: float = Field(default=16.0, gt=0)
    scales: list[float] = Field(default_factory=lambda: [0.9, 0.5])
    fixed_cap: Literal["horizon", "shared"] = "horizon"
    action_budget: int = Field(default=256, ge=1)
    max_expansions: int | None = Field(default=128, ge=1)
    num_subgoals: int = Field(default=1, ge=1, le=8)
    few_shot: int = Field(default=4, ge=0)
    generator_checkpoint: str | None = Field(default=None, min_length=1)
    generator_example_limit: int | None = Field(default=None, ge=0)
    selection: Literal["direct", "embeddings"] = "direct"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    discount: float = Field(default=0.99, gt=0, lt=1)
    policy_temperature: float = Field(default=1.0, gt=0)
    locked_policy_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    policy_coordinate_features: bool = False
    seed: int = 20261029
    repeat_seeds: list[int] = Field(default_factory=lambda: [11, 22, 33])
    generator_tokens: int = Field(default=4096, ge=128)
    policy_tokens: int = Field(default=128, ge=32)
    value_tokens: int = Field(default=64, ge=16)
    limits: Limits = Field(default_factory=Limits)
    calibration: Calibration = Field(default_factory=Calibration)
    tuning: Tuning = Field(default_factory=Tuning)
    trajectory_source: str = "rl-data/validation/sokoban/offline/12-12-4/sokoban_12_12_4_trajectories_part_1.pkl"
    data_dir: str = "data/icl"
    cache_dir: str = "analysis/luna_icl/cache"
    output_dir: str = "analysis/luna_icl/runs"
    train_boards: int = Field(default=8, ge=1)
    validation_boards: int = Field(default=4, ge=1)
    test_boards: int = Field(default=8, ge=1)
    family: str = "real_12x12x4"
    split: Literal["validation", "test", "component_validation", "scale_validation", "pilot100", "holdout1000"] = "test"
    instances: int = Field(default=4, ge=1)
    easy: bool = False
    tuning_status: Literal[
        "pilot_unvalidated", "validation_selected"
    ] = "pilot_unvalidated"

    @model_validator(mode="after")
    def check_scales(self) -> "ExperimentConfig":
        """Validate scale semantics and the corresponding execution cap."""
        if self.generator_example_limit is not None and self.generator_checkpoint is None:
            raise ValueError("generator_example_limit requires a generator_checkpoint")
        if self.generator_layout_schema and (
            self.generator_response_format or self.response_format
        ) != "json_schema":
            raise ValueError("Layout-constrained generation requires json_schema output")
        if self.method in {"relative", "relative_u", "fixed", "absolute", "absolute_d", "ratio_d_over_ac", "normalized_offset"}:
            if (
                len(self.scales) != 2
                or len(set(self.scales)) != 2
                or any(x <= 0 for x in self.scales)
            ):
                raise ValueError(
                    "Hierarchical conditions require two distinct positive scales"
                )
            if self.method in {"relative", "relative_u"} and any(x >= 1 for x in self.scales):
                raise ValueError(
                    "Relative percentiles must lie strictly between zero and one"
                )
            if self.method == "fixed" and any(not x.is_integer() for x in self.scales):
                raise ValueError(
                    "Fixed horizons must be integer primitive-action counts"
                )
            if (
                self.method == "fixed"
                and self.fixed_cap == "horizon"
                and max(self.scales) > self.calibration.action_cap
            ):
                raise ValueError(
                    "Fixed horizons cannot exceed the configured maximum controller cap"
                )
        return self

    def path(self, field: PathSetting) -> Path:
        """Resolve an explicit path setting relative to the repository root."""
        path = Path(getattr(self, field)).expanduser()
        return path if path.is_absolute() else ROOT / path


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a validated JSON config with cycle-checked recursive inheritance."""

    def merge(base: JSONObject, overrides: JSONObject) -> JSONObject:
        for key, value in overrides.items():
            base[key] = (
                merge(base[key], value)
                if isinstance(value, dict) and isinstance(base.get(key), dict)
                else value
            )
        return base

    def read(source: Path, seen: set[Path]) -> JSONObject:
        source = source.resolve()
        if source in seen:
            raise ValueError("Circular experiment config inheritance")
        content = json.loads(source.read_text(encoding="utf-8"))
        parent = content.pop("extends", None)
        base = read(source.parent / parent, seen | {source}) if parent else {}
        return merge(base, content)

    path = Path(path)
    return ExperimentConfig.model_validate(
        read(path if path.is_absolute() else ROOT / path, set())
    )
