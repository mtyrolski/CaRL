"""Plan or run a small matched generator test on real offline validation layouts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Final, cast

from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash
from carl.icl.config import ROOT, ExperimentConfig, Limits, load_config
from carl.icl.datasets import load_data
from carl.icl.experiment import calibration_path, components_for_run, credentials
from carl.icl.types import JSONObject, Provider
from scripts.generator_probe.cases import run_case
from scripts.generator_probe.report import summarize

BASE: Final = ROOT / "configs/icl/campaign_data/calibration_json_retry_v1.json"
PROMPT: Final = ROOT / "configs/icl/prompts/generator_planned_board.json"
OUTPUT: Final = ROOT / "analysis/luna_icl/campaign_100/generator_probe_v2"
VARIANTS: Final = ("baseline", "planned_board", "planned_board_low")
SCALES: Final = (0.5, 0.85)


def configurations() -> dict[str, ExperimentConfig]:
    """Change generator instructions/effort only; policy and calibration stay frozen."""
    baseline = load_config(BASE).model_copy(update={
        "method": "relative_u", "scales": list(reversed(SCALES)), "family": "real_12x12x4",
        "split": "component_validation", "instances": 2, "num_subgoals": 1,
        "generator_max_attempts": 1,
        "limits": Limits(max_requests=20, max_input_tokens=12000, max_total_input_tokens=240000,
                         max_output_tokens=65536, wall_seconds=900, timeout_seconds=90),
    })
    prompt = load_config(PROMPT).prompts.generator
    planned = baseline.model_copy(update={"prompts": baseline.prompts.model_copy(update={"generator": prompt})})
    return {"baseline": baseline, "planned_board": planned,
            "planned_board_low": planned.model_copy(update={"generator_reasoning_effort": "low"})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Save immutable inputs; offline")
    mode.add_argument("--execute", action="store_true", help="Execute/resume the prepared cases")
    parser.add_argument("--provider", choices=("replay", "openai"), default="replay")
    args = parser.parse_args()
    provider = cast(Provider, args.provider)
    configs = configurations()
    base = configs["baseline"]
    training, instances = load_data(base)
    reference = components_for_run(base, "replay", training, from_rows(instances[0]["state"]))
    if reference.calibrated is None:
        raise ValueError("Missing real frozen calibration")
    paths = [BASE, PROMPT, Path(__file__).resolve(), calibration_path(base, reference),
             base.path("data_dir") / "training.json",
             base.path("data_dir") / base.family / "component_validation.json",
             *sorted((ROOT / "carl/icl").rglob("*.py")),
             *sorted((ROOT / "scripts/generator_probe").glob("*.py"))]
    manifest: JSONObject = {
        "purpose": "exploratory generator diagnostic; no pilot/holdout inference", "provider": provider,
        "variants": {key: value.model_dump(mode="json") for key, value in configs.items()},
        "selected_boards": [{"board_id": item["id"], "state_hash": state_hash(from_rows(item["state"]))}
                            for item in instances],
        "scales": SCALES, "expected_queries": len(instances) * len(SCALES) * len(VARIANTS),
        "policy_fingerprint": reference.fingerprint(), "stage": "reserve", "stage_cap_usd": 1.5,
        "action_cap": base.calibration.action_cap, "max_physical_attempts": 240,
        "train_provenance": training.get("provenance"),
        "selected_example_ids": reference.calibrated["selected_example_ids"][:base.few_shot],
        "hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    # Normalize tuples exactly as the persisted JSON representation.
    manifest = json.loads(json.dumps(manifest))
    if not args.prepare and not args.execute:
        print(json.dumps({key: manifest[key] for key in (
            "purpose", "selected_boards", "scales", "expected_queries", "stage_cap_usd", "action_cap")}, indent=2))
        return
    output = OUTPUT / provider
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(OUTPUT / "probe.lock"), timeout=0):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Probe inputs changed; preserve results and use a new protocol")
        elif args.prepare:
            write_json(manifest_path, manifest)
        else:
            raise ValueError("Run --prepare with this provider before --execute")
        if args.prepare:
            print(f"Prepared {manifest_path}")
            return
        os.environ["CARL_CAMPAIGN_STAGE"] = "reserve"
        credentials(provider)
        rows: list[JSONObject] = []
        for instance in instances:
            for scale in SCALES:
                for variant in VARIANTS:
                    path = output / f"{instance['id']}_{scale}_{variant}.json"
                    if path.exists():
                        row = json.loads(path.read_text())
                    else:
                        row = run_case(configs[variant], training, reference.calibrated, instance,
                                       variant, scale, provider)
                        write_json(path, row)
                    rows.append(row)
                    summarize(rows, manifest["expected_queries"], output)
                    print(f"{instance['id']} u={scale} {variant}: reached={row['reached_any']} "
                          f"accepted={len(row['proposals'])} error={row['error']}", flush=True)
                    if row["error"] is not None:
                        return


if __name__ == "__main__":
    main()
