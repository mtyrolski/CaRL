"""One 32-action full-board diagnostic with the goal-placement demonstrations."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.config import ROOT, ExperimentConfig, Limits
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import credentials
from carl.icl.search import search
from carl.icl.search_tree import expansion_figure, tree_figure
from scripts.campaign_scale_stress import DurableEnvironment, DurableTree, verify_actions
from scripts.goal_probe.components import TaskProgressComponents
from scripts.goal_probe.data import OUTPUT, goals
from scripts.goal_probe.explicit import GoalAPI
from scripts.goal_probe.runner import inputs
from scripts.tune_campaign_scales import verify_tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    configs, demonstrations, _ = inputs()
    config = ExperimentConfig.model_validate({**configs["goal_examples"].model_dump(),
        "action_budget": 32, "max_expansions": 32,
        "limits": Limits(max_requests=128, max_input_tokens=24000, max_total_input_tokens=3000000,
                         max_output_tokens=131072, wall_seconds=900, timeout_seconds=90).model_dump()})
    training, instances = load_data(config)
    instance = instances[0]
    directory = OUTPUT / "full_board_32"
    paths = [Path(__file__).resolve(), ROOT / "scripts/goal_probe/components.py",
             ROOT / "scripts/goal_probe/explicit.py", ROOT / "scripts/goal_probe/runner.py",
             ROOT / "scripts/campaign_scale_stress.py", ROOT / "scripts/tune_campaign_scales.py",
             config.path("generator_checkpoint"), config.path("data_dir") / "training.json",
             config.path("data_dir") / "value_examples.json",
             config.path("data_dir") / config.family / "scale_validation.json",
             OUTPUT / "data_training_demonstrations.json", *sorted((ROOT / "carl/icl").rglob("*.py"))]
    manifest = {"purpose": "one exploratory full validation board, not a matched solve-rate comparison",
                "config": config.model_dump(mode="json"), "instance": instance,
                "demonstrations": demonstrations, "provider": "openai", "seed": 11,
                "hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    if not (args.prepare or args.execute):
        print(json.dumps({"board_id": instance["id"], "B": 32, "seed": 11,
                          "purpose": manifest["purpose"], "stage": "reserve"}, indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / "execution.lock"), timeout=0):
        path = directory / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Full-board protocol changed")
        if not path.exists():
            if not args.prepare:
                raise ValueError("Prepare the diagnostic before execution")
            write_json(path, manifest)
        if args.prepare:
            print(f"Prepared {path}")
            return
        if (directory / "case.json").exists():
            print("Completed record already exists; no API calls")
            return
        trace = directory / "trace.jsonl"
        if trace.exists():
            raise ValueError("Inspect interrupted trace before any new execution")
        os.environ["CARL_CAMPAIGN_STAGE"] = "reserve"
        credentials("openai")
        state = from_rows(instance["state"])
        api = GoalAPI(config, "openai", "full_board_32")
        tree = DurableTree(trace)
        env = DurableEnvironment(environment(state), tree)
        components = TaskProgressComponents(config, api, env, training, demonstrations)
        try:
            outcome = search(env, components, state, config, 11, observer=tree)
        finally:
            write_json(directory / "calls.json", api.calls)
            if api.client is not None:
                api.client.close()
        verification = verify_tree(tree, state, config.action_budget)
        verify_actions(tree, state, [json.loads(line) for line in trace.read_text().splitlines()])
        result = {"instance": instance, "config": config.model_dump(mode="json"),
                  "outcome": asdict(outcome), "verification": verification,
                  "metrics": api.metrics, "diagnostics": components.diagnostics,
                  "tree_summary": tree.summary(),
                  "max_endpoint_goals": max(goals(node.state) for node in tree.nodes),
                  "tree": {"nodes": [asdict(node) for node in tree.nodes],
                           "expansions": [asdict(entry) for entry in tree.expansions]}}
        write_json(directory / "case.json", result)
        tree_figure(tree).write_html(directory / "search_tree.html", include_plotlyjs=True)
        expansion_figure(tree).write_html(directory / "expansions.html", include_plotlyjs=True)
        print(json.dumps({"solved": outcome.solved, "finish_reason": outcome.finish_reason,
                          "actions": outcome.attempted_actions, "max_endpoint_goals": result["max_endpoint_goals"]}))


if __name__ == "__main__":
    main()
