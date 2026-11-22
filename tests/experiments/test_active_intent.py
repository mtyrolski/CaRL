"""Check native intent lifetimes and matched generator prompts."""

import json

import pytest

from carl.icl.intent_domain import TASK_GOAL, SokobanSnapshot, decode_intent
from carl.icl.planning.active_intent import IntentBudget, IntentSession
from carl.icl.planning.local_direct import SokobanDirect
from carl.icl.planning.paired_generator import messages, proposals
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
    checkpoint = branch.domain.checkpoint()
    speculative = branch.attempt("branch", TWO_GOALS, decide, speculative=True)
    assert speculative["status"] == "intent_timeout" and speculative["continuation"] == speculative["states"][0]
    assert branch.domain.checkpoint() == checkpoint and not branch.history
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


def test_directed_invalid_response_is_a_finished_attempt_with_checkpoint() -> None:
    session = _session(TWO_GOALS, IntentBudget(12, 12, 12, 60))
    checkpoints: list[JSONObject] = []
    result = session.attempt(
        "bad-action", TWO_GOALS, lambda _messages: _response(), stop_on_invalid=True, checkpoint=checkpoints.append,
    )
    assert result["status"] == "invalid_response" and result["first_hit"] is None
    assert len(result["decisions"]) == 1 and result["executed_primitives"] == 0
    assert checkpoints[0]["status"] == "running" and checkpoints[0]["requested_h"] == result["requested_h"]


def test_paired_generator_keeps_support_and_diversity_without_scale_proxies() -> None:
    effects = [{"effects": [{"predicate": "player_at", "row": target, "col": 1}], "preserve": [], "instruction": None}
               for target in (2, 3)]
    options = [{"id": tag, "intent": intent} for tag, intent in zip(("A", "B"), effects, strict=True)]
    support = [{"observation": list(BOARD.rows), "task": TASK_GOAL, "intent": h, "measured_u": u,
                "selection_weight": 0.5, "duration": 314, "demonstration_offset": 159}
               for u, h in zip((0.25, 0.75), effects, strict=True)]
    left = messages("sokoban", "unlabelled", list(BOARD.rows), TASK_GOAL, support, options)
    right = messages("sokoban", "relative", list(BOARD.rows), TASK_GOAL, support, options)
    baseline, relative = (json.loads(wire[1]["content"]) for wire in (left, right))
    assert relative.pop("requested_u") == [0.25, 0.75]
    assert [example.pop("u") for example in relative["examples"]] == [0.25, 0.75]
    assert baseline == relative
    assert all(set(example) == {"observation", "task", "intent"} for example in baseline["examples"])
    assert "scale" not in left[0]["content"].lower()
    with pytest.raises(ValueError, match="distinct measured scale"):
        messages("sokoban", "relative", list(BOARD.rows), TASK_GOAL, support[:1], options)

    for arm in ("relative", "unlabelled"):
        pair = proposals("sokoban", arm, '{"choices":["A","B"]}', list(BOARD.rows), options)
        hit_actions = []
        for slot in pair["slots"]:
            assert slot["status"] == "valid"
            session = _session(slot["intent"], IntentBudget(12, 12, 12, 60))
            result = session.attempt("fixture", slot["intent"], lambda _messages: _response("DOWN", "DOWN"))
            hit_actions.append(result["first_hit"])
        assert hit_actions == [1, 2]

    first = {**effects[0], "instruction": "move down"}
    second = {**effects[0], "instruction": "achieve the same tile with different wording"}
    duplicate = proposals("sokoban", "relative", json.dumps([first, second]), list(BOARD.rows), options)
    assert [slot["status"] for slot in duplicate["slots"]] == ["valid", "duplicate"]
    assert duplicate["distinct_effects"] == 1 and duplicate["within_overlap"]
