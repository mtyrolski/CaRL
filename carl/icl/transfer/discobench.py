"""Record a DiscoBench source pin and run a separate synthetic interface pilot."""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from carl.icl.api import BudgetExceeded, write_json
from carl.icl.luna_intent_rollout import IntentRollout, RolloutStatus
from carl.icl.planning.intent_execution import DomainExecutionOutcome, DomainExecutionState
from carl.icl.planning.intent_search import (
    IntentSearchRuntime, IntentSearchSettings, SearchMode, search_domain_intents,
)
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject


DISCOGEN_REPOSITORY = "https://github.com/AlexGoldie/discogen"
DISCOGEN_REVISION = "98f98a4a6c2bc5a5db01e7d3e0dffbbce651f488"
DISCOBENCH_TASK = "BayesianOptimisation_acq_fn"
DISCOBENCH_CONFIG = f"discogen/discobench_configs/{DISCOBENCH_TASK}.yaml"
DISCOBENCH_BLOB = "68a85dc864ac273d827c21f413c281ed235bbb93"
SYNTHETIC_PILOT = "synthetic_black_box_query_v1"
GOAL = "Observe an objective score of zero."
POINTS = range(4)


def public_task_pin() -> JSONObject:
    """Identify the public experiment separately from its synthetic interface smoke."""
    return {
        "repository": DISCOGEN_REPOSITORY,
        "revision": DISCOGEN_REVISION,
        "config": DISCOBENCH_CONFIG,
        "config_git_blob": DISCOBENCH_BLOB,
        "source_package_version": "1.3.0",
        "license": "MIT",
        "requires_python": ">=3.12,<4.0",
        "task": DISCOBENCH_TASK,
        "public_config_url": f"{DISCOGEN_REPOSITORY}/blob/{DISCOGEN_REVISION}/{DISCOBENCH_CONFIG}",
        "small_meta_train_instance": "Ackley1d",
        "small_meta_test_instance": "Ackley2d",
    }


@dataclass(frozen=True)
class Sample:
    """One observation produced by querying the black-box objective."""

    point: int
    score: int


@dataclass(frozen=True)
class Observation:
    """The model-visible history of measured points."""

    samples: tuple[Sample, ...]

    @property
    def best_score(self) -> int | None:
        """Return the best score observed so far."""
        return min((sample.score for sample in self.samples), default=None)


@dataclass(frozen=True)
class Snapshot:
    """Restorable state carrying the private objective parameter."""

    optimum: int
    samples: tuple[Sample, ...] = ()

    def step(self, point: int) -> "Snapshot":
        """Sample a point, leaving repeated queries as charged no-ops."""
        if isinstance(point, bool) or not isinstance(point, int) or point not in POINTS:
            raise ValueError("Synthetic query must identify a point on the finite grid")
        if any(sample.point == point for sample in self.samples):
            return self
        result = Sample(point, (point - self.optimum) ** 2)
        return Snapshot(self.optimum, (*self.samples, result))


@dataclass(frozen=True)
class Intent:
    """Collect a given number of distinct samples while preserving the best score."""

    samples_at_least: int
    preserve_best_at_most: int | None


@dataclass(frozen=True)
class QueryAttempt:
    """The controller's immutable action memory for one objective query."""

    action: int
    no_op: bool


State: TypeAlias = DomainExecutionState[Snapshot, QueryAttempt]
Outcome: TypeAlias = DomainExecutionOutcome[Snapshot, QueryAttempt]


@dataclass(frozen=True)
class SyntheticQueryExecutor:
    """Adapt a finite query process to the same search contract as Craftax."""

    charge: Callable[[str], None]

    @staticmethod
    def initial_state(snapshot: Snapshot, memory: Sequence[QueryAttempt]) -> State:
        """Use the immutable initial snapshot and historical queries."""
        return DomainExecutionState(snapshot, tuple(memory))

    @staticmethod
    def observation(state: State) -> Observation:
        """Expose queries and their measured scores only."""
        return Observation(state.snapshot.samples)

    @staticmethod
    def task_solved(state: State) -> bool:
        """Check the objective using the outcome of actual queries."""
        return any(sample.score == 0 for sample in state.snapshot.samples)

    @staticmethod
    def terminal(state: State) -> bool:
        """Stop after finding an optimum or exhausting the grid."""
        return SyntheticQueryExecutor.task_solved(state) or len(state.snapshot.samples) == len(POINTS)

    @staticmethod
    def validate_intent(observation: Observation, intent: Intent) -> Intent:
        """Require an unsatisfied effect and an already satisfied preservation term."""
        if not len(observation.samples) < intent.samples_at_least <= len(POINTS):
            raise ValueError("The intent must request a new, attainable sample count")
        if intent.preserve_best_at_most is not None:
            best = observation.best_score
            if best is None or best > intent.preserve_best_at_most:
                raise ValueError("The preservation threshold must hold before execution")
        return intent

    @staticmethod
    def state_key(state: State) -> State:
        """Distinguish snapshots and controller memories in deduplication."""
        return state

    @staticmethod
    def snapshot_record(snapshot: Snapshot) -> JSONObject:
        """Record snapshot identity and query results without publishing the private objective."""
        samples = [[sample.point, sample.score] for sample in snapshot.samples]
        return {"snapshot_id": contract_hash([snapshot.optimum, samples]), "samples": samples}

    @staticmethod
    def _intent_holds(observation: Observation, intent: Intent | None) -> tuple[bool, bool]:
        if intent is None:
            return False, True
        best = observation.best_score
        preserved = intent.preserve_best_at_most is None or (
            best is not None and best <= intent.preserve_best_at_most
        )
        return len(observation.samples) >= intent.samples_at_least, preserved

    def execute(
        self, state: State, intent: Intent | None, goal: str, *, seed: int, cap: int,
        before_action: Callable[[], None] | None = None,
    ) -> Outcome:
        """Exhaust unseen points in a fixed order with the same mock in every mode."""
        if goal != GOAL:
            raise ValueError("Unknown synthetic objective")
        del seed  # The mock picks the next unseen point deterministically.
        actions, rewards, steps = [], [], []
        model_calls = 0
        error = None
        if self.task_solved(state):
            status = RolloutStatus.TASK_SOLVED
        elif self.terminal(state):
            status = RolloutStatus.EPISODE_TERMINATED
        else:
            status = RolloutStatus.ACTION_LIMIT
        while len(actions) < cap and status == RolloutStatus.ACTION_LIMIT:
            observation = self.observation(state)
            point = next(point for point in POINTS if all(sample.point != point for sample in observation.samples))
            try:
                self.charge("policy")
                model_calls += 1
                if before_action is not None:
                    before_action()
            except BudgetExceeded as exc:
                status, error = RolloutStatus.BUDGET_EXHAUSTED, str(exc)
                break
            next_snapshot = state.snapshot.step(point)
            no_op = next_snapshot == state.snapshot
            state = DomainExecutionState(next_snapshot, (*state.memory, QueryAttempt(point, no_op)))
            solved = self.task_solved(state)
            reached, preserved = self._intent_holds(self.observation(state), intent)
            reward = float(solved)
            actions.append(point)
            rewards.append(reward)
            steps.append({
                "action": point, "no_op": no_op, "simulator_no_op": no_op, "intent_holds": reached,
                "preservation_holds": preserved, "task_solved": solved, "reward": reward,
                "terminated": self.terminal(state), "truncated": False,
            })
            if not preserved:
                status = RolloutStatus.PRESERVATION_VIOLATED
            elif reached:
                status = RolloutStatus.ACHIEVED
            elif solved:
                status = RolloutStatus.TASK_SOLVED
            elif self.terminal(state):
                status = RolloutStatus.EPISODE_TERMINATED
        rollout = IntentRollout(
            status, status == RolloutStatus.ACHIEVED, self.task_solved(state), actions, model_calls,
            [f"{sample.point}={sample.score}" for sample in self.observation(state).samples], steps, error,
        )
        return DomainExecutionOutcome(state, rollout, tuple(rewards))


def run_synthetic_smoke(
    mode: SearchMode | str = "one-step", *, action_budget: int = 16, request_budget: int = 64,
) -> JSONObject:
    """Run a deterministic interface case with explicit provenance and logical mock costs."""
    if isinstance(request_budget, bool) or not isinstance(request_budget, int) or request_budget < 0:
        raise ValueError("Synthetic request budget must be a nonnegative integer")
    metrics = {key: 0 for key in ("logical_requests", "physical_requests", "policy_calls", "generator_calls", "value_calls")}

    def charge(role: str) -> None:
        if metrics["logical_requests"] >= request_budget:
            raise BudgetExceeded("Synthetic logical request budget exhausted")
        metrics["logical_requests"] += 1
        metrics[f"{role}_calls"] += 1

    def propose(observation: Observation, _goal: str, scale: float | None) -> list[Intent]:
        charge("generator")
        batch = 2 if scale is not None and scale >= 0.5 else 1
        return [Intent(min(len(observation.samples) + batch, len(POINTS)), observation.best_score)]

    def value(observation: Observation, _goal: str) -> float:
        charge("value")
        best = observation.best_score
        return 0.0 if best is None else 1.0 / (1.0 + best)

    settings = IntentSearchSettings(scales=(0.8, 0.3), action_budget=action_budget, rollout_cap=3, seed=17)
    runtime = IntentSearchRuntime(SyntheticQueryExecutor(charge), propose, value, lambda: metrics)
    result = search_domain_intents(Snapshot(2), GOAL, mode=mode, settings=settings, runtime=runtime)
    return {
        "provenance": "synthetic_contract_smoke",
        "pilot": SYNTHETIC_PILOT,
        "backend": "in_process_mock",
        "public_task": public_task_pin(),
        "benchmark_score": None,
        "request_budget": request_budget,
        "result": result.as_dict(),
    }


def main() -> int:
    """Print the pinned task or save one offline synthetic result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("public-pin", "synthetic-smoke"))
    parser.add_argument("--mode", choices=("direct", "one-step", "full-search"), default="one-step")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = public_task_pin() if args.operation == "public-pin" else run_synthetic_smoke(args.mode)
    if args.output is None:
        print(json.dumps(payload, indent=2, allow_nan=False))
    else:
        write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
