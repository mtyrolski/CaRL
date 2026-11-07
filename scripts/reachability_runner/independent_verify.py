"""Independent action replay; does not call the runner's stop or replay helpers."""
from typing import Any

import numpy as np

from carl.icl.boards import state_hash
from carl.icl.datasets import environment


def replay_actions(source: np.ndarray, target: np.ndarray, steps: list[dict], cap: int) -> dict[str, Any]:
    """Recompute every state, first hit, terminal and stop from recorded actions."""
    if cap < 0:
        raise ValueError("Negative action cap")
    env = environment(source)
    current = source.copy()
    states = [current.copy()]
    hit_at = 0 if np.array_equal(current, target) else None
    solved = bool(env.is_solved(current))
    noops = 0
    for index, step in enumerate(steps):
        if hit_at is not None or solved or index >= cap:
            raise ValueError("Action after first hit, task terminal or cap")
        if state_hash(current) != step.get("before_hash"):
            raise ValueError("Source hash mismatch in independent replay")
        action = step.get("action")
        if isinstance(action, bool) or not isinstance(action, int) or action not in range(4):
            raise ValueError("Action outside the declared four-action map")
        successor = np.asarray(env.next_state(current, action))
        changed = not np.array_equal(current, successor)
        if not changed:
            noops += 1
        if state_hash(successor) != step.get("after_hash") or step.get("changed") is not changed:
            raise ValueError("Successor hash or no-op mismatch in independent replay")
        current = successor
        states.append(current.copy())
        if np.array_equal(current, target):
            hit_at = index + 1
        solved = bool(env.is_solved(current))
    reason = ("exact_hit" if hit_at is not None else "solved_before_target" if solved
              else "miss_cap" if len(steps) == cap else None)
    return {"states": states, "exact_hit": hit_at is not None, "task_solved": solved,
            "solved_before_target": solved and hit_at is None, "first_hit": hit_at,
            "attempted_actions": len(steps), "no_op_count": noops,
            "stop_reason": reason, "endpoint_hash": state_hash(current)}


def check_action_rng(steps: list[dict], seed: int) -> None:
    """Separately audit categorical draws from each recorded frozen vector."""
    rng = np.random.default_rng(seed)
    for step in steps:
        weights = np.asarray(step.get("probabilities"), dtype=float)
        if (weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1)):
            raise ValueError("Invalid recorded action distribution")
        if int(rng.choice(4, p=weights)) != step.get("action"):
            raise ValueError("Action differs from saved RNG stream")
