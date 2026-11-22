"""Fit and freeze a discounted continuation heuristic from solved training demonstrations."""

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field

from carl.icl.api import write_json
from carl.icl.boards import canonical_layout_id, from_rows, to_rows
from carl.icl.config import Settings
from carl.icl.data_sources.sokoban import verify_solution
from carl.icl.intent_annotations import _load_traces
from carl.icl.intent_domain import TASK_GOAL, SokobanObservation
from carl.icl.intents.generation.data import DataSplit
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject


def _features(observation: SokobanObservation) -> tuple[float, float, float]:
    """Use visible boxes and unmet goals with a dimension-normalized distance."""
    rows = observation.rows
    boxes, goals = [], []
    covered = 0
    for row, line in enumerate(rows):
        for col, tile in enumerate(line):
            if tile == "$":
                boxes.append((row, col))
            elif tile in ".+":
                goals.append((row, col))
            elif tile == "*":
                covered += 1
    unmet = len(goals) / max(len(goals) + covered, 1)
    width = len(rows[0]) + len(rows) - 2
    distance = sum(
        min(abs(row - r) + abs(col - c) for r, c in goals) for row, col in boxes
    ) / (len(boxes) * max(width, 1)) if goals and boxes else 0.0
    return 1.0, unmet, distance


class FittedIntentValue(Settings):
    """A portable ridge fit bound to train-only provenance and a discount factor."""

    model_config = ConfigDict(frozen=True)

    version: Literal["sokoban-demonstration-value-v1"] = "sokoban-demonstration-value-v1"
    discount: float = Field(gt=0, lt=1, strict=True)
    coefficients: tuple[float, float, float]
    data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    train_trajectories: int = Field(ge=1, strict=True)
    train_examples: int = Field(ge=2, strict=True)
    train_layout_ids: tuple[str, ...] = Field(min_length=1)
    train_task_groups: tuple[str, ...] = Field(min_length=1)

    def __call__(self, observation: SokobanObservation, task_goal: str) -> float:
        """Predict the demonstration-policy continuation from the visible board."""
        if task_goal != TASK_GOAL:
            raise ValueError("Fitted Sokoban value only supports the all-boxes-on-goals task")
        value = sum(feature * coefficient for feature, coefficient in zip(
            _features(observation), self.coefficients, strict=True,
        ))
        return min(max(value, 0.0), 1.0)

    def assert_held_out(
        self, observation: SokobanObservation, task_group: str | None, *, split: Literal["validation", "test"] = "test",
    ) -> None:
        """Reject validation or test queries whose layout or group entered the fit."""
        if split not in ("validation", "test") or observation.partial or not task_group or not task_group.strip():
            raise ValueError("Fitted value evaluation needs a complete held-out board and task group")
        layout = canonical_layout_id(from_rows(list(observation.rows)))
        if layout in self.train_layout_ids or task_group in self.train_task_groups:
            raise ValueError("Query overlaps the fitted value's training layout or task group")

    def identity(self) -> JSONObject:
        """Describe the frozen coefficients and the demonstration return contract."""
        payload = self.model_dump(mode="json")
        return {**payload, "fingerprint": contract_hash(payload)}

    def save(self, path: Path) -> None:
        """Publish the frozen model and fingerprint as one artifact."""
        write_json(path, self.identity())


def fit_value(source: Path, *, discount: float) -> FittedIntentValue:
    """Fit CaRL's discount ** remaining_actions labels on verified train traces."""
    if type(discount) not in (int, float) or not 0 < discount < 1:
        raise ValueError("The demonstration discount must be in (0,1)")
    payload = source.read_bytes()
    traces, _ = _load_traces(payload)
    training = [trace for trace in traces if trace.demonstration.split is DataSplit.TRAIN]
    if not training:
        raise ValueError("Fitted value needs at least one training demonstration")
    inputs: list[tuple[float, float, float]] = []
    returns: list[float] = []
    groups = set()
    for trace in training:
        demo = trace.demonstration
        groups.add(demo.task_group or demo.source_id)
        try:
            states = verify_solution(trace.snapshots[0].restore(), list(demo.actions))
        except ValueError as exc:
            raise ValueError(f"Training value trajectory {demo.source_id}: {exc}") from exc
        for position, state in enumerate(states):
            inputs.append(_features(SokobanObservation(tuple(to_rows(state)))))
            returns.append(discount ** (len(states) - 1 - position))
    matrix = np.asarray(inputs)
    penalty = np.diag([0.0, 1e-6, 1e-6])
    coefficients = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ np.asarray(returns))
    return FittedIntentValue(
        discount=discount, coefficients=(float(coefficients[0]), float(coefficients[1]), float(coefficients[2])),
        data_sha256=hashlib.sha256(payload).hexdigest(), train_trajectories=len(training), train_examples=len(returns),
        train_layout_ids=tuple(sorted({trace.layout_id for trace in training})), train_task_groups=tuple(sorted(groups)),
    )


def load_value(path: Path, *, discount: float) -> FittedIntentValue:
    """Load an intact value artifact with the planning run's exact discount."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected a fitted value object")
    fingerprint = payload.pop("fingerprint", None)
    model = FittedIntentValue.model_validate(payload)
    if fingerprint != model.identity()["fingerprint"]:
        raise ValueError("Fitted value fingerprint does not match its contents")
    if model.discount != discount:
        raise ValueError("Fitted value discount does not match the planning config")
    return model
