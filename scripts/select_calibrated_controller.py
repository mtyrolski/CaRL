#!/usr/bin/env python3
"""Select one frozen local controller using only a fixed calibration set."""

import argparse
import json
import math
import os
from collections import Counter
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from carl.icl.local_intent_policy import sha256_file


def score(records: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute success first, then no-op rate, balancing by layout."""
    by_id = {row["record_id"]: row for row in records}
    if not records or len(by_id) != len(records):
        raise ValueError("Calibration records require distinct IDs")
    if [row["record_id"] for row in results] != [row["record_id"] for row in records]:
        raise ValueError("Calibration results must cover the same ordered records")
    by_layout: dict[str, list[bool]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    noops = actions = calls = fences = 0
    for row in results:
        result = row["result"]
        status = result["status"]
        statuses[status] += 1
        if status != "api_error":
            by_layout[by_id[row["record_id"]]["layout_id"]].append(bool(result["achieved"]))
        noops += sum(step.get("no_op") is True for step in result["steps"])
        actions += len(result["actions"])
        calls += result["model_calls"]
        fences += sum(answer["fence_removed"] is True for answer in row["raw_policy_responses"])
    layout_rates = {layout: Fraction(sum(values), len(values)) for layout, values in by_layout.items() if values}
    mean = sum(layout_rates.values(), Fraction()) / len(layout_rates) if layout_rates else None
    return {"records": len(records), "layouts": len(layout_rates), "statuses": dict(statuses),
            "layout_success": {key: float(value) for key, value in layout_rates.items()},
            "board_mean_success": float(mean) if mean is not None else None,
            "successes": sum(sum(values) for values in by_layout.values()),
            "completed_for_success": sum(len(values) for values in by_layout.values()),
            "no_ops": noops, "primitive_actions": actions, "no_op_fraction": noops / actions if actions else None,
            "model_calls": calls, "fences_removed": fences}


def choose(scores: dict[str, dict[str, Any]]) -> str:
    """Maximise board-mean success, minimise no-ops, then prefer smaller model."""
    eligible = [(name, value) for name, value in scores.items() if value["board_mean_success"] is not None]
    if not eligible:
        raise ValueError("No candidate has a measured calibration outcome")
    sizes = {"05b": 0, "15b": 1}
    if any(name not in sizes for name, _ in eligible):
        raise ValueError("Controller size is unknown")
    return max(eligible, key=lambda item: (item[1]["board_mean_success"],
                                          -(item[1]["no_op_fraction"] if item[1]["no_op_fraction"] is not None
                                            else math.inf), -sizes[item[0]]))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    paths = {"05b": args.small, "15b": args.large}
    results = {name: [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
               for name, path in paths.items()}
    scores = {name: score(records, rows) for name, rows in results.items()}
    selected = choose(scores)
    manifests = {name: json.loads(path.with_suffix(".manifest.json").read_text()) for name, path in paths.items()}
    configs = {name: value["config"] for name, value in manifests.items()}
    comparable = ("device", "max_new_tokens", "max_context_tokens", "action_cap")
    if any(configs["05b"][key] != configs["15b"][key] for key in comparable):
        raise ValueError("Controller candidates used different rollout budgets or decoding settings")
    output = {"schema_version": 1, "selection": "max board-mean success; then min no-op fraction; then smaller model",
              "calibration_records_sha256": sha256_file(args.records),
              "result_sha256": {name: sha256_file(path) for name, path in paths.items()},
              "scores": scores, "selected": selected, "selected_config": configs[selected],
              "selected_configuration_sha256": manifests[selected]["configuration_sha256"],
              "parser": manifests[selected]["action_parser"],
              "limitations": "Selection used only four calibration layouts; do not infer generalisation."}
    text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output.exists() and args.output.read_text() != text:
        raise FileExistsError("Refusing to overwrite a different frozen policy selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"Selected {selected} based on {len(records)} calibration intents")


if __name__ == "__main__":
    main()
