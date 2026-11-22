"""The fixed policy records its original answer and never sees the future."""

from scripts.run_fixed_local_controller import evaluate_record


def record():
    return {"record_id": "one", "start_observation": ["#######", "#@$-.-#", "#-$-.-#", "#-----#",
                                                         "#-----#", "#-----#", "#######"],
            "intent": {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": [],
                       "instruction": None}, "task_goal": "put boxes on goals", "later_observation": "SECRET"}


def test_one_fence_removed_symmetrically_and_original_preserved():
    prompts = []

    def policy(prompt):
        prompts.append(prompt)
        return '```json\n{"action":"RIGHT"}\n```'

    result = evaluate_record(record(), policy, 2)
    assert result["result"]["achieved"] is True
    assert result["raw_policy_responses"] == [{"raw": '```json\n{"action":"RIGHT"}\n```',
                                                "normalised": '{"action":"RIGHT"}', "fence_removed": True}]
    assert "SECRET" not in prompts[0]


def test_other_prose_remains_invalid_and_is_not_silently_repaired():
    result = evaluate_record(record(), lambda _: 'Go right: {"action":"RIGHT"}', 2)
    assert result["result"]["status"] == "invalid_response"
    assert result["result"]["actions"] == []
    assert result["raw_policy_responses"][0]["fence_removed"] is False
