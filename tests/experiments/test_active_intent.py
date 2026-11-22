"""Check native intent lifetimes and prevent hindsight measurement leakage."""

import json

from carl.icl.experiments.intent_pilot_hindsight import _full_prefix
from carl.icl.intent_domain import TASK_GOAL, SokobanSnapshot, decode_intent
from carl.icl.planning.active_intent import IntentBudget, IntentSession
from carl.icl.planning.local_direct import SokobanDirect
from carl.icl.types import JSONObject


BOARD = SokobanSnapshot(("#######", "#@$.-##", "#-----#", "#-$.-##", "#-----#", "#-*---#", "#######"))
KEEP = {"predicate": "box_at", "row": 5, "col": 2}
TWO_GOALS = {
    "effects": [{"predicate": "box_at", "row": 1, "col": 3}, {"predicate": "box_at", "row": 3, "col": 3}],
    "preserve": [KEEP],
}


def _response(*actions: str) -> JSONObject:
    return {"response": json.dumps({"actions": actions, "intent": "a controller note"})}


def _session(h: JSONObject, budget: IntentBudget) -> IntentSession:
    return IntentSession(SokobanDirect(BOARD, decode_intent(h)), TASK_GOAL, budget)


def test_timeout_keeps_live_checkpoint_and_charges_speculation() -> None:
    session = _session(TWO_GOALS, IntentBudget(24, 24, 24, 60))
    queries = []

    def decide(messages: list[dict[str, str]]) -> JSONObject:
        queries.append(json.loads(messages[-1]["content"]))
        return _response("RIGHT", "DOWN") if len(queries) == 1 else _response("UP")

    timeout = session.attempt("timeout", TWO_GOALS, decide)
    assert timeout["status"] == "intent_timeout" and timeout["executed_primitives"] == 12
    assert timeout["first_hit"] is None and timeout["decisions"][0]["discarded_suffix"] == ["DOWN"]
    assert all(query["assigned_intent"] == timeout["requested_h"] for query in queries)
    assert queries[1]["observation"]["rows"][1][3] == "*"
    assert queries[1]["previous_intent"] == "a controller note"
    assert timeout["states"][0]["state_id"] != timeout["continuation"]["state_id"]

    player_target = {"effects": [{"predicate": "player_at", "row": 2, "col": 2}], "preserve": [KEEP]}
    replanned = session.attempt("replan", player_target, lambda _messages: _response("DOWN"))
    assert replanned["status"] == "intent_reached" and replanned["first_hit"] == 1
    assert replanned["states"][0] == timeout["continuation"]
    assert replanned["budget_before"] == timeout["budget_after"]

    queries.clear()
    branch = _session(TWO_GOALS, IntentBudget(24, 24, 24, 60))
    speculative = branch.attempt("branch", TWO_GOALS, decide, speculative=True)
    assert speculative["status"] == "intent_timeout" and speculative["continuation"] == speculative["states"][0]
    assert branch.domain.snapshot == BOARD and not branch.history
    assert branch.budget.actions == 12 and branch.budget.calls == 12


def test_early_budget_and_failed_preservation_leave_real_state() -> None:
    limited = _session(TWO_GOALS, IntentBudget(1, 4, 20, 60))
    stopped = limited.attempt("global-cap", TWO_GOALS, lambda _messages: _response("RIGHT"))
    assert stopped["status"] == "budget_cutoff" and stopped["budget_reason"] == "actions"
    assert stopped["executed_primitives"] == 1 and stopped["first_hit"] is None

    h = {"effects": [TWO_GOALS["effects"][0]], "preserve": [{"predicate": "box_at", "row": 1, "col": 2}]}
    violated = _session(h, IntentBudget(12, 12, 12, 60))
    result = violated.attempt("violation", h, lambda _messages: _response("RIGHT"))
    assert result["status"] == "preservation_failed" and result["first_hit"] is None
    assert result["steps"][0]["effects"] is True and result["steps"][0]["preservation"] is False
    assert result["continuation"]["observation"][1][3] == "*"


def test_craftax_hindsight_checks_preservation_after_the_first_hit() -> None:
    intent = {"effects": [{"predicate": "inventory_at_least", "item": "stone", "count": 2}],
              "preserve": [{"predicate": "tool_tier_at_least", "tool": "pickaxe", "tier": 1}]}
    frames = [{"inventory": {"stone": stone, "pickaxe": pickaxe}} for stone, pickaxe in ((1, 1), (2, 1), (2, 0))]
    decision = _full_prefix("craftax", intent, frames, [{"native_ticks": 1}] * 2)
    assert decision.status == "rejected" and "state 2" in decision.errors[0]
    uncertain = _full_prefix("craftax", intent, frames[:2], [{"native_ticks": 2}])
    assert uncertain.status == "unknown"
