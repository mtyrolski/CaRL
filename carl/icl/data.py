"""Export horizon-labelled demonstrations and disjoint evaluation layouts."""

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import level_key, to_rows, validate_subgoal


def prepare_data(config: dict[str, Any], env: Any) -> dict[str, Any]:
    """Export verified demonstration/evaluation layouts for the legacy hybrid runner."""
    source = Path(config["trajectories_path"])
    raw = joblib.load(source)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    rng = np.random.default_rng(config["seed"])
    required = config["demonstration_levels"] + config["instances"]
    horizons = config["generator_k_list"]
    selected, seen = [], set()
    replayed = 0
    for index in rng.permutation(len(trajectories)):
        trajectory = np.asarray(trajectories[index])
        if len(trajectory) <= max(horizons):
            continue
        key = level_key(trajectory[0])
        if key in seen or env.is_solved(trajectory[0]) or not env.is_solved(trajectory[-1]):
            continue
        for before, after in zip(trajectory[:-1], trajectory[1:]):
            validate_subgoal(before, after)
            if not any(
                np.array_equal(env.next_state(before, action), after) for action in range(4)
            ):
                raise ValueError(f"Non-replayable transition in trajectory {index}")
            replayed += 1
        seen.add(key)
        selected.append((int(index), trajectory))
        if len(selected) == required:
            break
    if len(selected) != required:
        raise ValueError(f"Need {required} distinct solved layouts, found {len(selected)}")
    demonstrations = []
    for index, trajectory in selected[: config["demonstration_levels"]]:
        for k in horizons:
            demonstrations.append(
                {
                    "state": to_rows(trajectory[0]),
                    "subgoal": to_rows(trajectory[k]),
                    "k": k,
                    "trajectory_index": index,
                    "level_key": level_key(trajectory[0]),
                }
            )
    instances = [
        {"id": f"trajectory_{index}", "state": to_rows(trajectory[0])}
        for index, trajectory in selected[config["demonstration_levels"] :]
    ]
    easy_instances = []
    for index, trajectory in selected[config["demonstration_levels"] :]:
        last_unsolved = next(state for state in reversed(trajectory) if not env.is_solved(state))
        if not any(env.is_solved(env.next_state(last_unsolved, action)) for action in range(4)):
            raise ValueError("Expected a replayable final transition for the easy fixture")
        easy_instances.append(
            {"id": f"trajectory_{index}_one_move", "state": to_rows(last_unsolved)}
        )
    provenance = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": config["seed"],
        "replayed_transitions": replayed,
        "horizon_semantics": "primitive trajectory offset; not policy-relative hit time",
        "split": "disjoint wall/goal layouts within one offline trajectory shard",
    }
    write_json(
        Path(config["demonstrations_path"]),
        {"provenance": provenance, "demonstrations": demonstrations},
    )
    write_json(Path(config["instances_path"]), {"provenance": provenance, "instances": instances})
    write_json(
        Path(config["easy_instances_path"]),
        {
            "provenance": provenance,
            "instances": easy_instances,
            "difficulty": "one primitive action from solved; an interface diagnostic, not a planning benchmark",
        },
    )
    return provenance
