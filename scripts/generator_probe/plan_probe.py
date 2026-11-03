"""Four paired validation requests with a visible plan before the explicit board."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import cast

from filelock import FileLock

from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows, to_rows
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import credentials
from carl.icl.luna import LunaComponents
from carl.icl.types import Board, JSONObject, Provider
from scripts.generator_probe.cases import CapturedAPI, classify, execute_target
from scripts.generator_probe.report import costs
from scripts.generator_probe.visible_plan import recorded_plan, visible_plan_payload
from scripts.luna_generator_probe import OUTPUT


class VisiblePlanComponents(LunaComponents):
    """Change only generator payloads; the controller never receives the action plan."""

    def generator_payload(self, state: Board, scale: float, resolve_embeddings: bool = True) -> JSONObject:
        payload = super().generator_payload(state, scale, resolve_embeddings)
        examples = self.generator_examples(state, scale, resolve_embeddings)
        return visible_plan_payload(payload, examples, action_cap=self.config.calibration.action_cap)


def run(source: JSONObject, baseline: JSONObject, provider: Provider, destination: Path) -> JSONObject:
    config = ExperimentConfig.model_validate(baseline["config"])
    training, _ = load_data(config)
    state = from_rows(source["state"])
    api = CapturedAPI(config, provider, f"visible_plan:{source['id']}")
    components = VisiblePlanComponents(config, api, environment(state), training)
    result: JSONObject = {"source_id": source["id"], "board_id": source["board_id"],
                          "source": source["state"], "scale": baseline["scale"],
                          "baseline_reached": baseline["reached_any"], "reached_any": False,
                          "error": None, "complete": False, "proposals": [], "provider": provider}
    write_json(destination, result)
    try:
        proposals = components.propose(state, baseline["scale"])
        result["classification"] = classify(api.calls, state, config.generator_layout_schema)
        result["generator_calls"] = list(api.calls)
        result["planned_actions"] = [recorded_plan(call.get("response", {}).get("output_text", ""))
                                     for call in api.calls]
        write_json(destination, result)
        for target in proposals:
            proposal: JSONObject = {"target": to_rows(target)}
            result["proposals"].append(proposal)

            def persist(progress: JSONObject) -> None:
                proposal["progress"] = progress
                write_json(destination, result)

            attempt = execute_target(components, state, target, baseline["sampling_seed"], persist)
            proposal["rollout"] = attempt
            result["reached_any"] |= attempt["exact_hit"]
            result["error"] = attempt["error"]
            if result["error"] is not None:
                break
    except (BudgetExceeded, RequestFailed) as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    result.update(complete=result["error"] is None, calls=api.calls,
                  diagnostics=components.diagnostics, metrics=api.metrics)
    write_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("replay", "openai"), default="replay")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    provider = cast(Provider, args.provider)
    sources = json.loads((OUTPUT / "validation_sources_30.json").read_text())["instances"]
    selected = [source for source in sources if source["position"] == 0][:4]
    baseline_dir = OUTPUT / "expanded_comparison/openai"
    paths = [baseline_dir / f"{source['id']}_few_shot_26.json" for source in selected]
    baselines = [json.loads(path.read_text()) for path in paths]
    if any(not result.get("complete") for result in baselines):
        raise ValueError("All four baseline requests must be complete")
    inputs = [*paths, Path(__file__).resolve(), ROOT / "scripts/generator_probe/visible_plan.py",
              ROOT / "scripts/generator_probe/cases.py", *sorted((ROOT / "carl/icl").rglob("*.py"))]
    manifest = {"provider": provider, "cases": [source["id"] for source in selected],
                "selection": "first four validation layouts at position0, independent of their outcomes",
                "factor": "visible plan plus training witness actions; not a pure instruction-only change",
                "policy": "unchanged; planned_actions never executed or supplied to policy",
                "stage": "reserve", "stage_cap_usd": 1.5,
                "hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}}
    if not args.prepare and not args.execute:
        print(json.dumps({key: manifest[key] for key in ("cases", "factor", "policy", "stage_cap_usd")}, indent=2))
        return
    directory = OUTPUT / "visible_plan" / provider
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / "probe.lock"), timeout=0):
        path = directory / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Visible-plan protocol changed")
        if not path.exists():
            if not args.prepare:
                raise ValueError("Prepare the visible-plan probe first")
            write_json(path, manifest)
        if args.prepare:
            print(f"Prepared {path}")
            return
        os.environ["CARL_CAMPAIGN_STAGE"] = "reserve"
        credentials(provider)
        rows = []
        for source, baseline in zip(selected, baselines, strict=True):
            destination = directory / f"{source['id']}.json"
            if destination.exists():
                result = json.loads(destination.read_text())
                if not result.get("complete"):
                    raise ValueError("Interrupted visible-plan case needs reconciliation")
            else:
                result = run(source, baseline, provider, destination)
            rows.append(result)
            write_json(directory / "summary.json", {
                "observed": len(rows), "expected": len(selected), "reached": sum(row["reached_any"] for row in rows),
                "baseline_reached": sum(row["baseline_reached"] for row in rows),
                "copies": sum(row["diagnostics"]["self_copies"] for row in rows),
                "invalid": sum(row["diagnostics"]["invalid_proposals"] for row in rows), "cost": costs()})
            print(f"{source['id']}: visible_plan={result['reached_any']}, baseline={baseline['reached_any']}", flush=True)
            if result["error"] is not None:
                return


if __name__ == "__main__":
    main()
