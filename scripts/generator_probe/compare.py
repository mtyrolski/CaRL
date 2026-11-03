"""Compare small and expanded generator contexts on thirty offline source states."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, cast

import tiktoken
from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows
from carl.icl.config import ROOT, Limits
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import credentials
from carl.icl.luna import LunaComponents
from carl.icl.types import Instance, JSONObject, Provider
from scripts.generator_probe.cases import CapturedAPI, run_case
from scripts.generator_probe.report import summarize
from scripts.luna_generator_probe import configurations

DIRECTORY: Final = ROOT / "analysis/luna_icl/campaign_100/generator_probe_v2"
SOURCES: Final = DIRECTORY / "validation_sources_30.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--provider", choices=("replay", "openai"), default="replay")
    parser.add_argument("--variant", choices=("planned_board", "planned_board_low"), default="planned_board_low")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    provider = cast(Provider, args.provider)
    checkpoint = args.checkpoint.resolve()
    saved_checkpoint = json.loads(checkpoint.read_text())
    count = saved_checkpoint["actual_example_count"]
    if count <= 4:
        raise ValueError("Expanded comparison needs more than four measured successful demonstrations")
    base = configurations()[args.variant].model_copy(update={
        "generator_checkpoint": str(checkpoint),
        "limits": Limits(max_requests=20, max_input_tokens=24000, max_total_input_tokens=480000,
                         max_output_tokens=65536, wall_seconds=1200, timeout_seconds=90),
    })
    configs = {f"few_shot_{limit}": base.model_copy(update={"generator_example_limit": limit})
               for limit in (4, count)}
    training, _ = load_data(base)
    sources = json.loads(SOURCES.read_text())
    original_ids = sorted({item["board_id"] for item in sources["instances"]})
    jobs: list[tuple[Instance, str, float]] = []
    prepared_payloads: list[JSONObject] = []
    encoding = tiktoken.get_encoding("o200k_base")
    maximum_tokens = 0
    for item in sources["instances"]:
        instance = Instance(id=item["id"], state=item["state"], level_key=item["board_id"])
        scale = (0.5, 0.85)[(original_ids.index(item["board_id"]) + item["position"]) % 2]
        for name, config in configs.items():
            api = CapturedAPI(config, "replay", "preview")
            component = LunaComponents(config, api, environment(from_rows(instance["state"])), training)
            payload = component.generator_payload(from_rows(instance["state"]), scale, resolve_embeddings=False)
            tokens = len(encoding.encode(json.dumps(payload), disallowed_special=()))
            if tokens > config.limits.max_input_tokens:
                raise ValueError("Expanded generator prompt exceeds the input token cap")
            maximum_tokens = max(maximum_tokens, tokens)
            prepared_payloads.append({"source_id": instance["id"], "variant": name, "payload": payload,
                                      "input_token_estimate": tokens})
            jobs.append((instance, name, scale))
    paths = [checkpoint, SOURCES, Path(__file__).resolve(), ROOT / "scripts/luna_generator_probe.py",
             *sorted((ROOT / "carl/icl").rglob("*.py")),
             ROOT / "scripts/generator_probe/cases.py", ROOT / "scripts/generator_probe/report.py"]
    manifest: JSONObject = {
        "provider": provider, "queries": len(jobs), "states": len(sources["instances"]),
        "layouts": len(original_ids), "demonstration_counts": [4, count], "prompt_variant": args.variant,
        "max_input_token_estimate": maximum_tokens, "stage": "reserve", "stage_cap_usd": 1.5,
        "max_rollout_actions": len(jobs) * 8,
        "configs": {name: config.model_dump(mode="json") for name, config in configs.items()},
        "schedule": [{"source_id": instance["id"], "variant": name, "u": scale} for instance, name, scale in jobs],
        "hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "interpretation": "30 source states on10 layouts, paired4-vs-all few-shot; dependent within-layout observations",
    }
    if not args.prepare and not args.execute:
        print(json.dumps({key: manifest[key] for key in (
            "queries", "states", "layouts", "demonstration_counts", "max_input_token_estimate",
            "stage_cap_usd", "max_rollout_actions", "interpretation")}, indent=2))
        return
    output = DIRECTORY / "expanded_comparison" / provider
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(DIRECTORY / "probe.lock"), timeout=0):
        path = output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Comparison protocol changed; preserve the saved run")
        if not path.exists():
            if not args.prepare:
                raise ValueError("Prepare comparison before execution")
            write_json(path, manifest)
            write_json(output / "generator_inputs.json", prepared_payloads)
        if args.prepare:
            print(f"Prepared {path}; complete inspectable prompts in generator_inputs.json")
            return
        os.environ["CARL_CAMPAIGN_STAGE"] = "reserve"
        credentials(provider)

        def run(job: tuple[Instance, str, float]) -> JSONObject:
            instance, name, scale = job
            destination = output / f"{instance['id']}_{name}.json"
            if destination.exists():
                saved = json.loads(destination.read_text())
                if not saved.get("complete"):
                    raise ValueError(f"Interrupted case requires reconciliation: {destination}")
                return saved
            result = run_case(configs[name], training, {}, instance, name, scale, provider, destination)
            write_json(destination, result)
            return result

        rows: list[JSONObject] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(jobs), args.workers):
                futures = [pool.submit(run, job) for job in jobs[start:start + args.workers]]
                results = [future.result() for future in futures]
                rows.extend(results)
                summarize(rows, len(jobs), output)
                for row in results:
                    print(f"{row['board_id']} {row['variant']}: reached={row['reached_any']} "
                          f"accepted={len(row['proposals'])} error={row['error']}", flush=True)
                if any(row["error"] is not None for row in results):
                    return


if __name__ == "__main__":
    main()
