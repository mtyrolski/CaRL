"""Create a new disjoint test split from verified existing offline trajectories.

This only reads the user's existing trajectory shard; it trains nothing and
runs no model or legacy data generator. Existing prepared data is never overwritten.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

from carl.icl.api import write_json
from carl.icl.boards import from_rows, level_key, to_rows
from carl.icl.config import load_config
from carl.icl.datasets import dataset_signature, environment, verify_solution
from carl.icl.types import Board


def verified_trajectory(states: np.ndarray) -> tuple[list[Board], list[int]]:
    """Recover actual primitive moves and reject invalid or unsolved paths."""
    env = environment(states[0])
    actions: list[int] = []
    for before, after in zip(states[:-1], states[1:]):
        if env.is_solved(before):
            break
        if np.array_equal(before, after):
            raise ValueError("Trajectory contains a no-op")
        choices = [action for action in range(4) if np.array_equal(env.next_state(before, action), after)]
        if not choices:
            raise ValueError("Trajectory contains an invalid transition")
        actions.append(choices[0])
    verified = verify_solution(states[0], actions)
    return verified, actions


def prepare(output: Path, count: int, seed: int, config_path: str, exclude: tuple[Path, ...] = ()) -> list[str]:
    """Sample new wall/goal layouts distinct from every existing prepared split."""
    if count < 1:
        raise ValueError("count must be positive")
    config = load_config(config_path)
    root = config.path("data_dir")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}")
    training_path = root / "training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    provenance = training.get("provenance", {})
    source = config.path("trajectory_source")
    if provenance.get("signature") != dataset_signature(config):
        raise ValueError("Prepared training data does not match the configuration")
    if provenance.get("source_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Trajectory shard differs from prepared data")
    family = "real_12x12x4"
    validation_path = root / family / "validation.json"
    original_test_path = root / family / "test.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    original_test = json.loads(original_test_path.read_text(encoding="utf-8"))
    seen = {level_key(from_rows(context["state"])) for context in training["contexts"]}
    seen.update(item["level_key"] for item in validation["instances"])
    seen.update(item["level_key"] for item in original_test["instances"])
    for directory in exclude:
        previous = json.loads((directory / family / "test.json").read_text(encoding="utf-8"))
        seen.update(item["level_key"] for item in previous["instances"])
    raw = joblib.load(source)
    trajectories = list(raw.values()) if isinstance(raw, dict) else list(raw)
    records = []
    for index in np.random.default_rng(seed).permutation(len(trajectories)):
        candidate = np.asarray(trajectories[int(index)])
        if len(candidate) <= config.calibration.reference_window:
            continue
        key = level_key(candidate[0])
        if key in seen:
            continue
        try:
            states, actions = verified_trajectory(candidate)
        except ValueError:
            continue
        if len(actions) < config.calibration.reference_window:
            continue
        identifier = f"trajectory_{int(index)}"
        records.append((identifier, states, actions, key))
        seen.add(key)
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError(f"Only found {len(records)} verified disjoint trajectories")
    output.mkdir(parents=True, exist_ok=True)
    (output / family).mkdir()
    (output / "private_witnesses" / family).mkdir(parents=True)
    shutil.copy2(training_path, output / "training.json")
    value_path = root / "value_examples.json"
    if value_path.exists():
        shutil.copy2(value_path, output / "value_examples.json")
    shutil.copy2(validation_path, output / family / "validation.json")
    for name in ("validation_easy.json",):
        if (root / family / name).exists():
            shutil.copy2(root / family / name, output / family / name)
    validation_witnesses = root / "private_witnesses" / family / "validation.json"
    if validation_witnesses.exists():
        shutil.copy2(validation_witnesses, output / "private_witnesses" / family / "validation.json")
    public = [{"id": identifier, "state": to_rows(states[0]), "level_key": key}
              for identifier, states, _actions, key in records]
    easy = [{"id": identifier + "_one_move", "state": to_rows(states[-2]), "level_key": key}
            for identifier, states, _actions, key in records]
    witnesses = [{"id": identifier, "actions": actions} for identifier, _states, actions, _key in records]
    write_json(output / family / "test.json", {"provenance": provenance, "instances": public})
    write_json(output / family / "test_easy.json", {"provenance": provenance, "instances": easy})
    write_json(output / "private_witnesses" / family / "test.json", witnesses)
    write_json(output / "fresh_holdout_manifest.json", {
        "seed": seed, "count": count, "source_sha256": provenance["source_sha256"],
        "excluded_existing_train_validation_test": True, "ids": [record[0] for record in records],
        "additional_excluded_holdouts": [str(path) for path in exclude],
    })
    return [record[0] for record in records]


def main() -> int:
    """Parse the destination and print the selected trajectory identifiers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/icl/experiments/tuned_fixed.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="Additional prepared holdout root to exclude; may be repeated")
    args = parser.parse_args()
    excluded = tuple(path.expanduser().resolve() for path in args.exclude)
    print("Prepared fresh holdout:", prepare(args.output.expanduser().resolve(), args.count, args.seed, args.config, excluded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
