"""Immutable generator checkpoint records with explicit calibration labels."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from carl.icl.config import ModelSettings, PromptSettings
from carl.icl.types import Example, ResponseFormat

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class FrozenRecord(BaseModel):
    """Freeze nested records and reject unknown fields and nonfinite numbers."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SourceArtifact(FrozenRecord):
    """A source file's recorded location and exact byte digest."""

    path: str = Field(min_length=1)
    sha256: Digest


class GeneratorDefaults(FrozenRecord):
    """Recommended settings; explicit experimental prompt variants remain allowed."""

    prompts: PromptSettings
    reasoning_effort: Literal["none", "low", "medium", "high"]
    response_format: ResponseFormat
    layout_schema: bool
    max_output_tokens: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    num_subgoals: int = Field(ge=1)
    offset_denominator: float = Field(gt=0)


class CalibratedExample(FrozenRecord):
    """One observed, successfully reached target; several may share a context."""

    context: str = Field(min_length=1)
    target_index: int = Field(ge=0)
    state: tuple[str, ...] = Field(min_length=1)
    subgoal: tuple[str, ...] = Field(min_length=1)
    actions: tuple[int, ...] = Field(min_length=1)
    k: int = Field(ge=1)
    u: float = Field(gt=0, lt=1)
    d: float = Field(gt=0)
    weight: float = Field(gt=0, le=1)
    a_c: float = Field(gt=0)
    ratio_d_over_ac: float = Field(gt=0)
    normalized_offset: float = Field(gt=0)
    hits: int = Field(ge=1)
    trials: int = Field(ge=1)
    success_rate: float = Field(gt=0, le=1)
    p_hat: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_witness(self) -> "CalibratedExample":
        """Require an action witness and consistent successful-trial accounting."""
        if len(self.actions) != self.k or any(action not in range(4) for action in self.actions):
            raise ValueError("Checkpoint witness actions must match k and use actions 0..3")
        if self.hits > self.trials or self.success_rate != self.hits / self.trials or self.p_hat != self.success_rate:
            raise ValueError("Checkpoint hit counts and success rates disagree")
        return self

    def as_example(self) -> Example:
        """Return a fresh mutable request record without exposing checkpoint storage."""
        return Example(
            context=self.context, target_index=self.target_index, state=list(self.state),
            subgoal=list(self.subgoal), actions=list(self.actions), k=self.k, u=self.u, d=self.d,
            weight=self.weight, ratio_d_over_ac=self.ratio_d_over_ac,
            normalized_offset=self.normalized_offset, hits=self.hits, success_rate=self.success_rate,
        )


class GeneratorCheckpoint(FrozenRecord):
    """Generator-only examples; policy training and policy few-shot count are separate."""

    version: Literal["carl-generator-checkpoint-v1", "carl-generator-checkpoint-v2"] = "carl-generator-checkpoint-v1"
    model: str = Field(min_length=1)
    generator_transport: ModelSettings | None = Field(default=None, exclude_if=lambda value: value is None)
    provider: Literal["openai", "mock"]
    policy_fingerprint: Digest
    policy_few_shot: int = Field(ge=0)
    action_cap: int = Field(ge=1)
    calibration_signature: Digest
    calibration_result_digest: Digest
    calibration_evidence: SourceArtifact
    source_hashes: tuple[SourceArtifact, ...] = Field(min_length=1)
    defaults: GeneratorDefaults
    actual_example_count: int = Field(ge=1)
    examples: tuple[CalibratedExample, ...] = Field(min_length=1)
    sha256: Digest

    @model_validator(mode="after")
    def validate_examples(self) -> "GeneratorCheckpoint":
        """Reject count drift and repeated targets while allowing repeated contexts."""
        if (self.version == "carl-generator-checkpoint-v2") != (self.generator_transport is not None):
            raise ValueError("Generator checkpoint version and transport disagree")
        if self.generator_transport is not None and self.generator_transport.model != self.model:
            raise ValueError("Generator checkpoint model and transport disagree")
        if self.actual_example_count != len(self.examples):
            raise ValueError("Checkpoint actual_example_count differs from stored examples")
        ids = [(example.context, example.target_index) for example in self.examples]
        if len(set(ids)) != len(ids):
            raise ValueError("Checkpoint repeats a context/target identity")
        if any(example.d > self.action_cap for example in self.examples):
            raise ValueError("Checkpoint first-hit mean exceeds the calibrated action cap")
        return self
