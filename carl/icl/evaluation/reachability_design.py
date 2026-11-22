"""Freeze exact future-state queries on disjoint Sokoban layout families."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from carl.icl.boards import canonical_layout_id, from_rows, state_hash, tiles, to_rows
from carl.icl.config import Settings
from carl.icl.data_sources.sokoban import reverse_board, verify_solution
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import Board, JSONObject, TrainingContext, TrainingData

Positive = Annotated[int, Field(strict=True, ge=1)]
Action = Annotated[int, Field(strict=True, ge=0, le=3)]


class ReachabilityModel(Settings):
    """One controller condition and its explicitly selected offsets."""

    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    kind: Literal["mock", "api", "local"]
    config: Path | None = None
    training_receipt: Path | None = None
    offsets: list[Positive] | None = Field(default=None, min_length=1)
    offline_compute_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    adaptation_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ReachabilityStudy(Settings):
    """Configuration of the disjoint 12x12 transfer pilot."""

    schema_version: Literal[1]
    data: Literal["seeded_fixture", "trajectory_files"]
    seed: int = Field(strict=True, ge=0)
    offsets: list[Positive] = Field(min_length=1)
    trials_per_point: Positive = 100
    reserves_per_point: int = Field(default=0, strict=True, ge=0)
    training: Path | None = None
    test: Path | None = None
    models: list[ReachabilityModel] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_conditions(self) -> "ReachabilityStudy":
        """Require an unambiguous set of offsets, conditions and data sources."""
        if len(set(self.offsets)) != len(self.offsets) or len({m.id for m in self.models}) != len(self.models):
            raise ValueError("Offsets and model IDs must be unique")
        if self.data == "trajectory_files" and (self.training is None or self.test is None):
            raise ValueError("Provide explicit training and test trajectory files")
        if self.data == "seeded_fixture" and (self.training is not None or self.test is not None):
            raise ValueError("The seeded fixture creates its own trajectory files")
        for model in self.models:
            offsets = model.offsets or self.offsets
            if len(set(offsets)) != len(offsets) or set(offsets) - set(self.offsets):
                raise ValueError(f"{model.id}: offsets must be a unique subset of study offsets")
            if model.config is None:
                raise ValueError(f"{model.id}: supply the controller config")
            if (model.training_receipt is not None) != (model.kind == "local"):
                raise ValueError(f"{model.id}: only local checkpoints require a training receipt")
            if self.data == "seeded_fixture" and model.kind != "mock":
                raise ValueError("The fixture is restricted to mock controllers")
        return self


class Witness(Settings):
    """Verifiable source data. Actions never enter a test-time model request."""

    id: str = Field(min_length=1)
    board: list[str] = Field(min_length=12, max_length=12)
    actions: list[Action] = Field(min_length=1)


class WitnessFile(Settings):
    """A named split containing complete legal Sokoban demonstrations."""

    schema_version: Literal[1]
    split: Literal["train", "test"]
    trajectories: list[Witness] = Field(min_length=1)


@dataclass(frozen=True)
class Trajectory:
    """A validated witness and its canonical layout group."""

    witness: Witness
    layout: str
    states: list[Board]


class ReachabilityCase(Settings):
    """A frozen pair sampled at offset k with one layout per point."""

    trajectory: str
    layout: str
    index: int = Field(ge=0)
    k: Positive
    seed: int = Field(ge=0)
    source: list[str]
    target: list[str]
    source_sha256: str
    target_sha256: str


class TrainingReceipt(Settings):
    """Bind a declared local adaptation to its exact training split and weights."""

    training_split_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


def fixture(study: ReachabilityStudy, split: Literal["train", "test"]) -> WitnessFile:
    """Build deterministic small 12x12 witnesses using the existing reverse generator."""
    train = split == "train"
    number = 4 if train else study.trials_per_point + study.reserves_per_point
    split_seed = np.random.SeedSequence([study.seed, int(train)])
    records = []
    for i, seed in enumerate(split_seed.generate_state(number)):
        board, actions, _ = reverse_board(12, 4 if train else 10, int(seed), pulls=max(16, max(study.offsets) + 2))
        records.append(Witness(id=f"{split}-{i}", board=to_rows(board), actions=actions))
    return WitnessFile(schema_version=1, split=split, trajectories=records)


def validated_trajectories(data: WitnessFile, split: Literal["train", "test"]) -> list[Trajectory]:
    """Verify family, action witnesses and independence units before sampling."""
    if data.split != split:
        raise ValueError(f"Expected the {split} split")
    expected_boxes = 4 if split == "train" else 10
    ids: set[str] = set()
    layouts: set[str] = set()
    trajectories = []
    for witness in data.trajectories:
        board = from_rows(witness.board)
        if board.shape[:2] != (12, 12) or np.isin(tiles(board), [3, 4]).sum() != expected_boxes:
            raise ValueError(f"{split}/{witness.id}: expected a 12x12 board with {expected_boxes} boxes")
        layout = canonical_layout_id(board)
        if witness.id in ids or layout in layouts:
            raise ValueError(f"{split}/{witness.id}: duplicate ID or symmetric layout")
        ids.add(witness.id)
        layouts.add(layout)
        trajectories.append(Trajectory(witness, layout, verify_solution(board, witness.actions)))
    return trajectories


def design(study: ReachabilityStudy, train: WitnessFile, test: WitnessFile) -> tuple[TrainingData, JSONObject]:
    """Build verified support and seed-matched test pairs, keeping witness paths private."""
    training, testing = validated_trajectories(train, "train"), validated_trajectories(test, "test")
    if {t.layout for t in training} & {t.layout for t in testing}:
        raise ValueError("Training and test share canonical layouts")
    fingerprints = {item.split: contract_hash(item.model_dump(mode="json")) for item in (train, test)}
    examples: list[TrainingContext] = [
        {"id": t.witness.id, "state": t.witness.board, "remaining_actions": len(t.witness.actions),
         "calibration_indices": [], "candidates": [
             {"k": k, "subgoal": to_rows(t.states[k]), "actions": t.witness.actions[:k]}
             for k in range(1, min(len(t.states), max(8, max(study.offsets)) + 1))]}
        for t in training
    ]
    wanted = study.trials_per_point + study.reserves_per_point
    cases: dict[int, list[ReachabilityCase]] = {}
    used_offsets = {k for model in study.models for k in (model.offsets or study.offsets)}
    for k in sorted(used_offsets):
        rng = np.random.default_rng(np.random.SeedSequence([study.seed, k]))
        candidates = [_sample_case(testing[int(i)], k, rng) for i in rng.permutation(len(testing))]
        cases[k] = [case for case in candidates if case is not None][:wanted]
        if len(cases[k]) < wanted:
            raise ValueError(f"k={k}: need {wanted} distinct eligible test layouts, found {len(cases[k])}")
    return {"contexts": examples, "provenance": {"train_sha256": fingerprints["train"]}}, {
        "data_provenance": study.data, "split_fingerprints": fingerprints,
        "train_layouts": sorted(t.layout for t in training), "test_layouts": sorted(t.layout for t in testing),
        "cases": {str(k): [case.model_dump(mode="json") for case in rows] for k, rows in cases.items()},
    }


def _sample_case(trajectory: Trajectory, k: int, rng: np.random.Generator) -> ReachabilityCase | None:
    states = trajectory.states
    eligible = [i for i in range(len(states) - k) if not np.array_equal(states[i], states[i + k])]
    if not eligible:
        return None
    index = int(rng.choice(eligible))
    source, target = states[index], states[index + k]
    return ReachabilityCase(trajectory=trajectory.witness.id, layout=trajectory.layout, index=index, k=k,
                            seed=int(rng.integers(2**32)), source=to_rows(source), target=to_rows(target),
                            source_sha256=state_hash(source), target_sha256=state_hash(target))
