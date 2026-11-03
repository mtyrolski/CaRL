"""Expose the cost implications of a much larger budget without pretending to measure it."""

import json

from carl.icl.config import ROOT
from carl.icl.types import JSONObject


def forecast(plan: JSONObject) -> JSONObject:
    """Scale one observed explicit run by its low-level work, with sampling limits attached."""
    path = ROOT / "configs/icl/checkpoints/launch100/cost_reference.json"
    measured = json.loads(path.read_text())
    explicit_methods = [name for name in plan["methods"] if name != "verbal_relative"]
    jobs = len(plan["cohort"]) * len(explicit_methods)
    action_factor = plan["node_budget"] / measured["attempted_actions"]
    metrics = measured["metrics"]
    scenarios = []
    for name, factor in (("same_usage_per_low_level_node", action_factor),):
        scenarios.append({"assumption": name, "jobs": jobs,
                          "mixed_cache_usd": jobs * factor * measured["settled_usd"],
                          "logical_tokens": int(jobs * factor * (metrics["logical_input_tokens"]
                                                                + metrics["logical_output_tokens"])),
                          "provider_requests": int(jobs * factor * metrics["requests"]),
                          "serial_hours": jobs * factor * measured["seconds"] / 3600})
    return {"reference": str(path.relative_to(ROOT)), "measured_boards": 1,
            "measured_cost_usd": measured["settled_usd"], "concurrency": 1,
            "explicit_methods": explicit_methods, "scenarios": scenarios,
            "limitations": measured["limitations"],
            "interpretation": "This is an extrapolation, not an upper bound. "
                              "The ledger stops execution at the lower remaining campaign or stage limit. "
                              "The full 100-board comparison is prepared, not guaranteed to complete within that cap. "
                              "No verbal whole-search cost forecast is supported by existing observations."}
