"""Validate saved calibration identities, seeded trials and exact trajectory outcomes."""

import hashlib

import numpy as np

from carl.icl.boards import from_rows, to_rows
from carl.icl.datasets import environment
from carl.icl.types import JSONObject

REPETITIONS = 2
SEED = 20261102
CAP = 8


def trial_seed(context_id: str, index: int, trial: int) -> int:
    """Common deterministic action-sampling streams independent of worker order."""
    return int(hashlib.sha256(f"{SEED}:{context_id}:{index}:{trial}".encode()).hexdigest()[:8], 16)


def verify_target(context: JSONObject, index: int, row: JSONObject, policy_fingerprint: str) -> None:
    """Fail closed on incomplete, stale, or internally inconsistent saved trials."""
    candidate = context["candidates"][index]
    expected = {"context": context["id"], "target_index": index, "k": candidate["k"],
                "source": context["state"], "target": candidate["subgoal"],
                "policy_fingerprint": policy_fingerprint}
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved calibration target identity or policy differs")
    if not row.get("complete") or row.get("error") is not None or len(row["records"]) != REPETITIONS:
        raise ValueError("Saved target does not contain two complete trials")
    for trial, record in enumerate(row["records"]):
        if (record["seed"] != trial_seed(context["id"], index, trial)
                or record["trial"] != trial or record["cap"] != CAP or record["error"] is not None):
            raise ValueError("Saved trial seed, index, cap or error differs")
        source, target = from_rows(context["state"]), from_rows(candidate["subgoal"])
        env = environment(source)
        current = source
        first_hit = None
        if not 0 < len(record["steps"]) <= CAP:
            raise ValueError("Saved nontrivial target must have 1..L recorded actions")
        for position, step in enumerate(record["steps"], start=1):
            if first_hit is not None or env.is_solved(current):
                raise ValueError("Saved rollout continued beyond its stopping condition")
            action = step["action_index"]
            probabilities = np.asarray(step["probabilities"], dtype=float)
            if (action not in range(4) or probabilities.shape != (4,) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1)):
                raise ValueError("Invalid saved policy distribution or action")
            current = np.asarray(env.next_state(current, action), dtype=np.float32)
            if to_rows(current) != step["endpoint"]:
                raise ValueError("Saved policy transition does not replay")
            if np.array_equal(current, target):
                first_hit = position
        if (record["first_hit"] != first_hit or record["exact_hit"] != (first_hit is not None)
                or record["task_solved"] != env.is_solved(current) or record["actions"] != len(record["steps"])
                or record["actual_endpoint"] != to_rows(current) or not record["replay_verified"]):
            raise ValueError("Saved hit, action count or endpoint disagrees with replay")
