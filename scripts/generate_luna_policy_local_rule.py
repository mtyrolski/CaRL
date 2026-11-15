#!/usr/bin/env python3
"""Measure a current-board local rule in two fixed proposal slots per board."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from carl.icl.intent_assessment import local_rule_with_stats
from carl.icl.policy_relative_lora import assess_output
from carl.icl.policy_relative_lora import test_contexts

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(contexts_path: Path, output: Path) -> dict[str, Any]:
    """Run both deterministic slots independently, with no model/API calls."""
    contexts = test_contexts(contexts_path)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Will not overwrite local-rule measurements: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".local-rule-", dir=output.parent))
    try:
        totals: Counter[str] = Counter()
        statuses: Counter[str] = Counter()
        per_slot = []
        for slot in range(2):
            rows = []
            for context in contexts:
                # Deliberately repeat the complete rule computation per slot;
                # do not give this arm extra proposals through deduplication.
                intent, work = local_rule_with_stats(context["start_observation"])
                raw = json.dumps(intent, ensure_ascii=False, separators=(",", ":"))
                assessment = assess_output(raw, context["start_observation"], normalize=True)
                statuses[assessment["status"]] += 1
                for key in ("walk_graphs", "walk_nodes_expanded", "walk_edges_examined", "push_checks",
                            "legal_pushes", "goal_distance_checks", "elapsed_seconds"):
                    totals[key] += work[key]
                rows.append({**{key: context[key] for key in ("record_id", "context_id", "layout_id", "family")},
                             "arm": "R", "requested_scale": None, "slot": slot, "raw_output": raw,
                             "generated_tokens": None, "generation_seconds": None, "normalization_applied": False,
                             "local_rule_work": work, **assessment})
            path = temporary / f"R_slot{slot}.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
                            encoding="utf-8")
            per_slot.append({"slot": slot, "file": path.name, "records": len(rows), "sha256": digest(path)})
        result = {
            "schema_version": 1,
            "arm": "R",
            "method": "current-board local rule: graph traversal to select an accessible push",
            "input_fields": ["start_observation"],
            "input_sha256": digest(contexts_path),
            "implementation_sha256": digest(ROOT / "carl/icl/intent_assessment.py"),
            "runner_sha256": digest(Path(__file__)),
            "slots_per_context": 2,
            "slots": per_slot,
            "predictions": sum(statuses.values()),
            "statuses": dict(statuses),
            "local_rule_work_totals": dict(totals),
            "cost_convention": "Each slot runs the rule independently; tested edges include blocked directions.",
            "model_calls": 0,
        }
        (temporary / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8")
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory for both slots")
    args = parser.parse_args()
    print(json.dumps(generate(args.contexts, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
