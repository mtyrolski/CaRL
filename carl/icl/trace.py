"""Record a controller rollout without changing its sampling or stopping rules."""
from dataclasses import dataclass

import numpy as np

from carl.icl.search import Attempt, rollout
from carl.icl.types import Board, BoardEnvironment, Probabilities, RolloutPolicy
from carl.utils.aliases import State


@dataclass(frozen=True)
class RolloutStep:
    """The distribution and simulator transition for one attempted action."""

    state: Board
    probabilities: Probabilities
    action: int
    successor: Board

    @property
    def changed(self) -> bool:
        """Blocked moves are retained and count toward the cap."""
        return not np.array_equal(self.state, self.successor)


class RecordingPolicy:
    """Observe each policy distribution without invoking it a second time."""

    def __init__(self, policy: RolloutPolicy) -> None:
        self.policy = policy
        self.latest: Probabilities | None = None

    def probabilities(self, state: Board, subgoal: Board) -> Probabilities:
        """Forward the distribution used by the shared stochastic controller."""
        probabilities = self.policy.probabilities(state, subgoal)
        self.latest = probabilities.copy()
        return probabilities


class RecordingEnvironment:
    """Record only transitions requested by the shared rollout."""

    def __init__(self, env: BoardEnvironment, policy: RecordingPolicy) -> None:
        self.env = env
        self.policy = policy
        self.steps: list[RolloutStep] = []

    def next_state(self, state: State, action: int, /) -> State:
        """Apply one simulator action and attach the corresponding policy output."""
        probabilities = self.policy.latest
        if probabilities is None:
            raise RuntimeError(
                "Rollout transition has no associated policy distribution"
            )
        before = np.asarray(state).copy()
        successor = self.env.next_state(state, action)
        self.steps.append(
            RolloutStep(
                before,
                probabilities,
                action,
                np.asarray(successor).copy(),
            )
        )
        self.policy.latest = None
        return successor

    def is_solved(self, state: State, /) -> bool:
        """Use the original environment's terminal predicate."""
        return self.env.is_solved(state)


def trace_rollout(
    env: BoardEnvironment,
    policy: RolloutPolicy,
    state: Board,
    target: Board,
    *,
    cap: int,
    rng: np.random.Generator,
) -> tuple[Attempt, list[RolloutStep]]:
    """Run search.rollout and return exactly the transitions that it executed."""
    recorded_policy = RecordingPolicy(policy)
    recorded_env = RecordingEnvironment(env, recorded_policy)
    attempt = rollout(recorded_env, recorded_policy, state, target, cap=cap, rng=rng)
    return attempt, recorded_env.steps
