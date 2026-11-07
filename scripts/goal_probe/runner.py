"""Freeze or execute twelve paired, targeted goal-placement component queries."""

import argparse
import hashlib
import json
import os
from typing import cast

import tiktoken
from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.config import ROOT, ExperimentConfig, Limits, load_config
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import credentials
from carl.icl.components import ICLComponents
from carl.icl.types import JSONObject, Provider
from scripts.goal_probe.components import MechanicsExample, TaskProgressComponents
from scripts.goal_probe.data import OUTPUT, Query
from scripts.goal_probe.explicit import GoalAPI, run_case

BASE_CONFIG = ROOT / "configs/icl/experiments/luna_checkpoint26_retry2.json"
GOAL_CONFIG = ROOT / "configs/icl/prompts/generator_goal_progress.json"


def inputs() -> tuple[dict[str, ExperimentConfig], list[MechanicsExample], list[Query]]:
    """The public validation file contains no future states or witness actions."""
    config = load_config(BASE_CONFIG)
    limits = Limits(max_requests=32, max_input_tokens=24000, max_total_input_tokens=768000,
                    max_output_tokens=65536, wall_seconds=900, timeout_seconds=90)
    base = config.model_copy(update={"limits": limits})
    goal = base.model_copy(update={"prompts": load_config(GOAL_CONFIG).prompts})
    saved = json.loads((OUTPUT / "data_training_demonstrations.json").read_text())["demonstrations"]
    demonstrations = [MechanicsExample(id=row["id"], state=row["state"], subgoal=row["subgoal"])
                      for row in saved]
    queries = cast(list[Query], json.loads((OUTPUT / "data_validation_queries.json").read_text())["instances"])
    return {"baseline26": base, "goal_examples": goal}, demonstrations, queries


def prepare_manifest(provider: Provider) -> JSONObject:
    """Save effective prompts separately from audit witnesses; no API is used."""
    configs, demonstrations, queries = inputs()
    training, _ = load_data(configs["baseline26"])
    payloads: list[JSONObject] = []
    fingerprints = set()
    encoder = tiktoken.get_encoding("o200k_base")
    for query in queries:
        for variant, config in configs.items():
            state = from_rows(query["state"])
            api = GoalAPI(config, "replay", "preview")
            component = (TaskProgressComponents(config, api, environment(state), training, demonstrations)
                         if variant == "goal_examples" else ICLComponents(config, api, environment(state), training))
            payload = component.generator_payload(state, 0.85, resolve_embeddings=False)
            tokens = len(encoder.encode(json.dumps(payload), disallowed_special=()))
            if tokens > config.limits.max_input_tokens:
                raise ValueError("Progress demonstrations exceed the input limit")
            fingerprints.add(component.fingerprint())
            payloads.append({"query_id": query["id"], "variant": variant, "payload": payload,
                             "serialized_input_token_estimate": tokens})
    if len(fingerprints) != 1:
        raise ValueError("The paired study changes CLLP")
    paths = [BASE_CONFIG, GOAL_CONFIG, OUTPUT / "data_training_demonstrations.json",
             OUTPUT / "data_validation_queries.json", configs["baseline26"].path("generator_checkpoint"),
             configs["baseline26"].path("data_dir") / "training.json",
             *sorted((ROOT / "carl/icl").rglob("*.py")),
             *(ROOT / "scripts/goal_probe" / name for name in ("runner.py", "explicit.py", "components.py", "data.py")),
             ROOT / "scripts/generator_probe/cases.py"]
    return {"provider": provider, "model": configs["baseline26"].model,
            "expected_queries": len(payloads), "validation_layouts": len(queries),
            "u": 0.85, "L": 8, "seed": 11, "stage": "reserve", "stage_cap_usd": 1.5,
            "purpose": "targeted goal-placement diagnostic; selected near known progress, not full-board solve rate",
            "intervention": "goal-progress instructions plus six unlabelled training examples; combined factor",
            "configs": {name: config.model_dump(mode="json") for name, config in configs.items()},
            "policy_fingerprint": next(iter(fingerprints)), "payloads": payloads,
            "hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("replay", "openai"), default="replay")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    provider = cast(Provider, args.provider)
    manifest = prepare_manifest(provider)
    if not (args.prepare or args.execute):
        print(json.dumps({key: value for key, value in manifest.items()
                          if key not in {"hashes", "payloads", "configs"}}, indent=2))
        return
    directory = OUTPUT / "explicit" / provider
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / "execution.lock"), timeout=0):
        path = directory / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Prepared protocol changed; preserve prior observations")
        if not path.exists():
            if not args.prepare:
                raise ValueError("Prepare immutable inputs before execution")
            write_json(path, manifest)
        if args.prepare:
            print(f"Saved inspectable prompts: {path}")
            return
        os.environ["CARL_CAMPAIGN_STAGE"] = "reserve"
        credentials(provider)
        configs, demonstrations, queries = inputs()
        training, _ = load_data(configs["baseline26"])
        for index, query in enumerate(queries):
            for variant, config in configs.items():
                destination = directory / f"case_{index:02d}_{variant}.json"
                if destination.exists():
                    row = json.loads(destination.read_text())
                    if not row.get("complete"):
                        raise ValueError(f"Interrupted case requires inspection: {destination}")
                else:
                    row = run_case(config, training, query, demonstrations, variant, provider, destination)
                print(f"{query['id']} {variant}: hit={row.get('reached_any')} "
                      f"goal_progress={row.get('reached_goal_progress')} error={row['error']}", flush=True)
                if row["error"] is not None:
                    return


if __name__ == "__main__":
    main()
