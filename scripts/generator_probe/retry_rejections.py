"""Replay rejected validation replies, then allow one explicit static-feedback retry."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import cast

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.config import ROOT, ExperimentConfig
from carl.icl.datasets import load_data
from carl.icl.experiment import credentials
from carl.icl.types import Instance, JSONObject, Provider
from scripts.generator_probe.cases import run_case
from scripts.generator_probe.report import costs
from scripts.luna_generator_probe import OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("replay", "openai"), default="replay")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    provider = cast(Provider, args.provider)
    baseline = OUTPUT / "expanded_comparison/openai"
    paths = sorted(baseline.glob("*_few_shot_26.json"))
    originals: list[JSONObject] = [json.loads(path.read_text()) for path in paths]
    if len(originals) != 30 or any(not row.get("complete") or row["error"] for row in originals):
        raise ValueError("Need all thirty completed original observations")
    rejected = [row for row in originals if not row["proposals"]]
    configs = {row["board_id"]: ExperimentConfig.model_validate(row["config"]).model_copy(
        update={"generator_max_attempts": 2}) for row in rejected}
    frozen_paths = [*paths, Path(__file__).resolve(), ROOT / "scripts/generator_probe/cases.py",
                    *sorted((ROOT / "carl/icl").rglob("*.py"))]
    manifest: JSONObject = {
        "provider": provider, "selection": "all static rejections among the completed26-shot validation cases",
        "expected": len(rejected), "original_cases": len(originals),
        "original_reached": sum(row["reached_any"] for row in originals),
        "interpretation": "post-hoc repair diagnostic using cached original first replies; not an independent retest",
        "queries": [{"source_id": row["board_id"], "u": row["scale"],
                     "config": configs[row["board_id"]].model_dump(mode="json")} for row in rejected],
        "stage": "reserve", "stage_cap_usd": 1.5,
        "hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_paths},
    }
    if not args.prepare and not args.execute:
        print(json.dumps({key: manifest[key] for key in (
            "selection", "expected", "original_cases", "original_reached", "interpretation")}, indent=2))
        return
    output = OUTPUT / "static_retry" / provider
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / "probe.lock"), timeout=0):
        path = output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Retry protocol changed")
        if not path.exists():
            if not args.prepare:
                raise ValueError("Prepare retry diagnostic before execution")
            write_json(path, manifest)
        if args.prepare:
            print(f"Prepared {path}")
            return
        os.environ["CARL_CAMPAIGN_STAGE"] = "reserve"
        credentials(provider)
        rows: list[JSONObject] = []
        for original in rejected:
            config = configs[original["board_id"]]
            training, _ = load_data(config)
            instance = Instance(id=original["board_id"], state=original["source"], level_key=original["board_id"])
            destination = output / f"{original['board_id']}.json"
            if destination.exists():
                row = json.loads(destination.read_text())
                if not row.get("complete"):
                    raise ValueError("Interrupted retry case needs reconciliation")
            else:
                row = run_case(config, training, {}, instance, "few_shot_26_static_retry",
                               original["scale"], provider, destination)
            rows.append(row)
            recovered = sum(item["reached_any"] for item in rows)
            write_json(output / "summary.json", {
                "expected": len(rejected), "observed": len(rows), "recovered": recovered,
                "original_reached": manifest["original_reached"], "cohort_size": 30,
                "revised_replay_first_hits": manifest["original_reached"] + recovered,
                "new_endpoint_actions": sum(proposal["rollout"]["actions"] for item in rows
                                            for proposal in item["proposals"]),
                "cost": costs(), "interpretation": manifest["interpretation"],
            })
            print(f"{original['board_id']}: recovered={row['reached_any']} error={row['error']}", flush=True)
            if row["error"] is not None:
                return


if __name__ == "__main__":
    main()
