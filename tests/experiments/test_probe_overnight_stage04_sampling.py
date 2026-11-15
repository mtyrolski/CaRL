"""The training-only sampler probe must agree with saved sample sequences."""

import json
from itertools import islice

import pytest

from scripts.probe_overnight_stage04_sampling import independent_sampler
from scripts.probe_overnight_stage04_sampling import probe


def row(identifier, board, effect, split="train"):
    predicates = [{"predicate": "box_at", "row": 2, "col": 3}]
    if effect == "goal":
        predicates.append({"predicate": "boxes_on_goals_at_least", "count": 2})
    return {"record_id": identifier, "layout_id": board, "split": split, "effect_type": effect,
            "task_goal": "Put every box on a goal.",
            "start_observation": ["#####", "#@--#", "#-$-#", "#-.-#", "#####"],
            "intent": {"effects": predicates, "preserve": [], "instruction": None}}


class CharTokenizer:
    eos_token = "!"

    def apply_chat_template(self, messages, **_kwargs):
        return [ord(char) for char in json.dumps(messages)]

    def encode(self, text, **_kwargs):
        return [ord(char) for char in text]


def test_probe_reconstructs_schedule_and_rejects_a_changed_curve(tmp_path):
    narrow = [row("a", "a0", "box"), row("b", "a0", "goal"), row("c", "a1", "box")]
    wide = [row("d", "b0", "box"), row("e", "b1", "goal"), row("f", "b2", "box")]
    a, b, curve_path = (tmp_path / name for name in ("a.jsonl", "b.jsonl", "curve.jsonl"))
    for path, records in ((a, narrow), (b, wide)):
        path.write_text("\n".join(json.dumps(item) for item in records) + "\n")
    expected = list(islice(independent_sampler(wide, 1729), 8))
    curve = [{"step": i // 2 + 1, "samples": [{"record_id": item["record_id"]} for item in expected[i:i+2]]}
             for i in range(0, 8, 2)]
    curve_path.write_text("\n".join(json.dumps(item) for item in curve) + "\n")
    result = probe(a, b, steps=4, accumulation=2, seed=1729, reference_curve=curve_path)
    assert result["wide_sequence_matches_completed_curve"] and result["samples_per_arm"] == 8
    assert result["arms"]["narrow"]["pool"]["layouts"] == 2
    assert result["arms"]["wide"]["pool"]["layouts"] == 3
    assert all(result["pool_marginals_match"].values())
    tokens = probe(a, b, steps=4, accumulation=2, seed=1729, pad_to_length=5000,
                   max_length=5000, tokenizer=CharTokenizer())
    assert tokens["planned_model_tokens_per_arm"] == 40000
    assert tokens["arms"]["narrow"]["tokens"]["truncated"] == 0
    assert tokens["arms"]["wide"]["tokens"]["sampled_effective_total"] < 40000
    curve[0]["samples"][0]["record_id"] = "wrong"
    curve_path.write_text("\n".join(json.dumps(item) for item in curve) + "\n")
    with pytest.raises(ValueError, match="differs from the completed"):
        probe(a, b, steps=4, accumulation=2, seed=1729, reference_curve=curve_path)
