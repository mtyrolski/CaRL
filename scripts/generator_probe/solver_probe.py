"""One full validation solve with the measured ICL checkpoint and durable trace."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash
from carl.icl.config import ROOT, load_config
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import components_for_run, credentials
from carl.icl.luna import LunaComponents
from carl.icl.search import search
from carl.icl.search_tree import expansion_figure, tree_figure
from scripts.experiment_support.trace import DurableEnvironment, DurableTree, verify_actions
from scripts.generator_probe.cases import CapturedAPI
from scripts.generator_probe.report import costs
from scripts.luna_generator_probe import OUTPUT
from scripts.experiment_support.trace import verify_tree

CONFIG = ROOT / "configs/icl/experiments/luna_checkpoint26_retry2.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(CONFIG)
    training, instances = load_data(config)
    instance = instances[0]
    initial = from_rows(instance["state"])
    reference = components_for_run(config, "replay", training, initial)
    paths = [CONFIG, config.path("generator_checkpoint"), Path(__file__).resolve(),
             ROOT / "scripts/experiment_support/trace.py",
             ROOT / "scripts/generator_probe/cases.py", *sorted((ROOT / "carl/icl").rglob("*.py")),
             config.path("data_dir") / "training.json", config.path("data_dir") / "value_examples.json",
             config.path("data_dir") / config.family / "scale_validation.json"]
    manifest = {"provider": "openai", "model": config.model, "board_id": instance["id"],
                "state_hash": state_hash(initial), "split": config.split, "family": config.family,
                "seed": 11, "B": config.action_budget, "policy_fingerprint": reference.fingerprint(),
                "config": config.model_dump(mode="json"), "stage": "reserve", "stage_cap_usd": 1.5,
                "purpose": "one exploratory full validation solve; never a pilot or held-out observation",
                "hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    if not args.prepare and not args.execute:
        print(json.dumps({key: manifest[key] for key in (
            "board_id", "split", "family", "seed", "B", "stage_cap_usd", "purpose")}, indent=2))
        return
    output = config.path("output_dir")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / "probe.lock"), timeout=0):
        path = output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Full-solver protocol changed")
        if not path.exists():
            if not args.prepare:
                raise ValueError("Prepare solver probe first")
            write_json(path, manifest)
        if args.prepare:
            print(f"Prepared {path}")
            return
        if (output / "case.json").exists():
            print("Saved case exists; no inference repeated.")
            return
        trace = output / "trace.jsonl"
        if trace.exists():
            raise ValueError("Interrupted trace exists; inspect it before any new solve")
        os.environ["CARL_CAMPAIGN_STAGE"] = "reserve"
        credentials("openai")
        api = CapturedAPI(config, "openai", "full_solver_validation")
        tree = DurableTree(trace)
        env = DurableEnvironment(environment(initial), tree)
        components = LunaComponents(config, api, env, training)
        try:
            result = search(env, components, initial, config, 11, observer=tree)
        finally:
            write_json(output / "calls.json", api.calls)
            if api.client is not None:
                api.client.close()
        verification = verify_tree(tree, initial, config.action_budget)
        events = [json.loads(line) for line in trace.read_text().splitlines()]
        verify_actions(tree, initial, events)
        case = {"provider": "openai", "instance": instance, "config": config.model_dump(mode="json"),
                "outcome": asdict(result), "verification": verification, "metrics": api.metrics,
                "diagnostics": components.diagnostics, "cost": costs(), "tree_summary": tree.summary(),
                "tree": {"nodes": [asdict(node) for node in tree.nodes],
                         "expansions": [asdict(entry) for entry in tree.expansions]}}
        write_json(output / "case.json", case)
        tree_figure(tree).write_html(output / "search_tree.html", include_plotlyjs=True)
        expansion_figure(tree).write_html(output / "expansions.html", include_plotlyjs=True)
        print(json.dumps({"outcome": asdict(result), "verification": verification,
                          "tree": tree.summary()}, indent=2))


if __name__ == "__main__":
    main()
