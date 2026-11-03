"""Measure new generator demonstrations with the unchanged production CLLP."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, cast

import numpy as np
from filelock import FileLock

from carl.icl.api import write_json
from carl.icl.boards import from_rows, state_hash
from carl.icl.calibration import calibration_result_digest, weighted_median, weighted_percentiles
from carl.icl.config import ROOT, Limits
from carl.icl.datasets import environment, load_data
from carl.icl.experiment import credentials
from carl.icl.luna import LunaComponents
from carl.icl.types import JSONObject, Provider, TrainingData
from scripts.generator_probe.cases import CapturedAPI, execute_target
from scripts.generator_probe.integrity import trial_seed, verify_target
from scripts.generator_probe.report import costs
from scripts.luna_generator_probe import configurations

DIRECTORY: Final = ROOT / "analysis/luna_icl/campaign_100/generator_probe_v2"
DATASET: Final = DIRECTORY / "training_candidates.json"
REPETITIONS: Final = 2
SEED: Final = 20261102


def digest(value: JSONObject) -> str:
    """Identify a persisted protocol without depending on formatting."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def target_run(
    context: JSONObject, index: int, training: TrainingData, provider: Provider, output: Path,
) -> JSONObject:
    """Resume a whole saved target; never silently repeat an interrupted rollout."""
    candidate = context["candidates"][index]
    path = output / f"{context['id']}_{index}.json"
    source, target = from_rows(context["state"]), from_rows(candidate["subgoal"])
    if np.array_equal(source, target):
        raise ValueError("Calibration cannot include the source state as a target")
    config = configurations()["baseline"].model_copy(update={
        "limits": Limits(max_requests=32, max_input_tokens=12000, max_total_input_tokens=384000,
                         max_output_tokens=65536, wall_seconds=1200, timeout_seconds=90),
    })
    api = CapturedAPI(config, provider, f"calibration:{context['id']}:{index}")
    components = LunaComponents(config, api, environment(source), training)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("complete"):
            verify_target(context, index, saved, components.fingerprint())
        return saved
    row: JSONObject = {"context": context["id"], "target_index": index, "k": candidate["k"],
                       "source": context["state"], "target": candidate["subgoal"],
                       "policy_fingerprint": components.fingerprint(), "records": [], "error": None,
                       "complete": False}
    write_json(path, row)
    for trial in range(REPETITIONS):
        seed = trial_seed(context["id"], index, trial)

        def persist(progress: JSONObject) -> None:
            row["progress"] = {"trial": trial, "seed": seed, **progress}
            write_json(path, row)

        row["progress"] = {"trial": trial, "seed": seed, "actions": 0, "steps": []}
        write_json(path, row)
        record = execute_target(components, source, target, seed, persist)
        record.update(trial=trial, seed=seed, cap=config.calibration.action_cap)
        row["records"].append(record)
        row["calls"] = api.calls
        row["metrics"] = dict(api.metrics)
        row["error"] = record["error"]
        # An unfinished target is explicit and will block automatic resume.
        row["complete"] = len(row["records"]) == REPETITIONS and row["error"] is None
        write_json(path, row)
        if row["error"] is not None:
            break
    if row["complete"]:
        verify_target(context, index, row, components.fingerprint())
    return row


def aggregate(contexts: list[JSONObject], rows: list[JSONObject], manifest: JSONObject) -> JSONObject:
    """Calculate midpoint u only for targets actually hit by the frozen controller."""
    targets: list[JSONObject] = []
    by_context: dict[str, list[JSONObject]] = {}
    source_by_id = {context["id"]: context for context in contexts}
    complete = len(rows) == sum(len(context["candidates"]) for context in contexts)
    for row in rows:
        context = source_by_id[row["context"]]
        candidate = context["candidates"][row["target_index"]]
        records = row["records"]
        times = [record["first_hit"] for record in records if record["exact_hit"]]
        valid = row["complete"]
        complete &= valid
        target: JSONObject = {**candidate, "state": context["state"], "context": context["id"],
                              "target_index": row["target_index"], "trials": len(records),
                              "hits": len(times), "p_hat": len(times) / REPETITIONS if valid else None,
                              "success_rate": len(times) / REPETITIONS if valid else None,
                              "d": float(np.mean(times)) if times and valid else None,
                              "target_hash": state_hash(from_rows(candidate["subgoal"])),
                              "excluded_reason": "incomplete" if not valid else None if times else "zero_exact_hits"}
        targets.append(target)
        if times and valid:
            by_context.setdefault(context["id"], []).append(target)
    context_rows: list[JSONObject] = []
    for context in contexts:
        retained = by_context.get(context["id"], [])
        if not retained:
            context_rows.append({"context": context["id"], "retained": 0, "distinct_d": 0, "a_c": None})
            continue
        times, hits = [row["d"] for row in retained], [row["hits"] for row in retained]
        weights, percentiles = weighted_percentiles(times, hits)
        median = weighted_median(times, [float(count) for count in hits])
        for row, weight, percentile in zip(retained, weights, percentiles, strict=True):
            row.update(weight=float(weight), u=float(percentile), a_c=median,
                       ratio_d_over_ac=row["d"] / median, normalized_offset=row["k"] / 16,
                       cdf_before=sum(float(w) for other, w in zip(retained, weights, strict=True)
                                      if other["d"] < row["d"]),
                       cdf_at=sum(float(w) for other, w in zip(retained, weights, strict=True)
                                  if other["d"] <= row["d"]))
        context_rows.append({"context": context["id"], "retained": len(retained),
                             "distinct_d": len(set(times)), "a_c": median})
    # Interleave contexts so every prefix covers as many layouts as possible.
    ordered = [group[index] for index in range(3) for group in by_context.values() if index < len(group)]
    summary: JSONObject = {"status": "completed" if complete else "partial", "provider": manifest["provider"],
                           "signature": digest(manifest), "policy_fingerprint": manifest["policy_fingerprint"],
                           "config": {"trials": REPETITIONS, "action_cap": 8, "reference_window": 8},
                           "examples": ordered, "targets": targets, "contexts": context_rows,
                           "selected_example_ids": [{"context": row["context"], "target_index": row["target_index"]}
                                                    for row in ordered],
                           "randomness_note": "Independent seeded action draws from cached policy distributions; "
                           "no new independent provider draw is claimed for replay. The reference population "
                           "is three distinct offline targets per context: prefer k=1,2,4, then fill from "
                           "k=3,5,6,7,8 after excluding source copies and duplicates; not all future states.",
                           "cost": costs()}
    summary["result_digest"] = calibration_result_digest(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("replay", "openai"), default="replay")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    provider = cast(Provider, args.provider)
    dataset = json.loads(DATASET.read_text())
    contexts = dataset["contexts"]
    config = configurations()["baseline"]
    training, _ = load_data(config)
    source = from_rows(contexts[0]["state"])
    reference = LunaComponents(config, CapturedAPI(config, "replay", "plan"), environment(source), training)
    paths = [DATASET, Path(__file__).resolve(), *sorted((ROOT / "carl/icl").rglob("*.py")),
             ROOT / "scripts/luna_generator_probe.py", ROOT / "scripts/generator_probe/cases.py",
             ROOT / "scripts/generator_probe/integrity.py"]
    manifest = {"provider": provider, "policy_fingerprint": reference.fingerprint(), "R": REPETITIONS,
                "L": 8, "targets": len(contexts) * 3, "layouts": len(contexts), "seed": SEED,
                "stage": "reserve", "stage_cap_usd": 1.5, "max_actions": len(contexts) * 3 * REPETITIONS * 8,
                "max_physical_attempts": len(contexts) * 3 * 32,
                "selection": "prefer offsets1,2,4; replace source copies/duplicates with3,5,6,7,8; keep actual k",
                "policy_config": config.model_dump(mode="json"),
                "hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    if not args.prepare and not args.execute:
        print(json.dumps({key: manifest[key] for key in ("provider", "R", "L", "targets", "layouts", "max_actions", "stage_cap_usd")}, indent=2))
        return
    output = DIRECTORY / "expanded_calibration" / provider
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(DIRECTORY / "probe.lock"), timeout=0):
        path = output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Calibration protocol changed")
        if not path.exists():
            if not args.prepare:
                raise ValueError("Prepare the calibration before execution")
            write_json(path, manifest)
        if args.prepare:
            print(f"Prepared {path}")
            return
        os.environ["CARL_CAMPAIGN_STAGE"] = "reserve"
        credentials(provider)
        jobs = [(context, index) for context in contexts for index in range(len(context["candidates"]))]
        rows: list[JSONObject] = []
        # Submit bounded pairs; errors prevent scheduling any later batch.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(jobs), args.workers):
                batch = [pool.submit(target_run, context, index, training, provider, output)
                         for context, index in jobs[start:start + args.workers]]
                results = [future.result() for future in batch]
                rows.extend(results)
                summary = aggregate(contexts, rows, manifest)
                write_json(output / "calibration.json", summary)
                for row in results:
                    print(f"{row['context']} k={row['k']}: "
                          f"{sum(record['exact_hit'] for record in row['records'])}/{REPETITIONS} "
                          f"complete={row['complete']}", flush=True)
                if any(not row["complete"] for row in results):
                    return


if __name__ == "__main__":
    main()
