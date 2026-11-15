"""Local, model-free tests of stage-three transforms and durable predictions."""
from __future__ import annotations

import contextlib
import json

import numpy as np
import pytest
import torch

from carl.icl.boards import canonical_layout_id
from carl.icl.boards import state_hash
from carl.icl.boards import to_rows
from carl.icl.data_sources.sokoban import environment
from scripts import overnight_stage03_generate as runner
from scripts.run_overnight_cpu_policy import atomic_json
from scripts.run_overnight_cpu_policy import digest


def board_pair():
    cells = np.ones((12, 12), dtype=int)
    cells[[0, -1], :] = 0
    cells[:, [0, -1]] = 0
    cells[2, 2:5] = 3
    cells[5, 3], cells[5, 4], cells[5, 5] = 5, 4, 2
    source = np.eye(7, dtype=np.float32)[cells]
    target = np.asarray(environment(source).next_state(source, 3))
    return {"source": to_rows(source), "target": to_rows(target), "source_hash": state_hash(source),
            "target_hash": state_hash(target), "layout_id": canonical_layout_id(source),
            "pair_id": "pair", "k": 1, "demonstration_actions": [3]}


def test_coordinates_actions_and_replay_match_the_real_environment():
    pair = board_pair()
    for transform in runner.TRANSFORMS:
        rows = runner.transform_rows(pair["source"], transform)
        for row in range(12):
            for col in range(12):
                mapped = runner.map_coordinate(row, col, transform)
                assert rows[mapped[0]][mapped[1]] == pair["source"][row][col]
                assert runner.inverse_coordinate(*mapped, transform) == (row, col)
        actions = [0, 1, 2, 3]
        assert runner.transform_actions(runner.transform_actions(actions, transform), transform, inverse=True) == actions
        assert runner.replay_transform(pair, transform)["actions_replayed"] == 1


def test_inverse_intent_maps_arrays_and_flags_untranslated_text():
    intent = {"effects": [{"predicate": "box_at", "row": 3, "col": 4}],
              "preserve": [{"predicate": "player_at", "row": 5, "col": 6}], "instruction": "move left"}
    restored, untranslated = runner.inverse_intent(intent, "rot90_ccw")
    assert restored["effects"] == [{"predicate": "box_at", "row": 4, "col": 8}]
    assert restored["preserve"] == [{"predicate": "player_at", "row": 6, "col": 6}]
    assert restored["instruction"] == "move left" and untranslated


def test_duplicate_json_keys_rejected_and_fence_rule_symmetric():
    source = board_pair()["source"]
    case = {"source_rows": source, "transform": "identity"}
    duplicate = '{"effects":[],"effects":[],"preserve":[]}'
    result = runner.output_record(case, duplicate, 1, 1.0)
    assert not result["strict_json_valid"] and not result["json_valid"]
    valid = '{"effects":[{"predicate":"box_at","row":5,"col":5}],"preserve":[],"instruction":null}'
    wrapped = runner.output_record(case, "```json\n" + valid + "\n```", 1, 1.0)
    assert not wrapped["strict_json_valid"] and wrapped["json_valid"] and wrapped["valid_nontrivial"]


def test_selection_balances_evaluation_sources_and_prefers_unexposed_development():
    rows = []
    for family, source, rank in (("12x12x4", "x4", 0), ("12x12x4", "x4", 1),
                                 ("12x12x10", "gamma", 0), ("12x12x10", "gamma", 1),
                                 ("12x12x10", "reverse", 2), ("12x12x10", "reverse", 3)):
        pair = board_pair()
        pair.update(context_id=f"{family}-{rank}", layout_id=f"layout-{family}-{rank}", family=family,
                    source_family=source, split="evaluation", board_rank=rank, context_rank=0)
        rows.append(pair)
    chosen = runner.select_contexts(rows, "evaluation", 2, set())
    assert sum(row["family"] == "12x12x4" for row in chosen) == 2
    assert {row["source_family"] for row in chosen if row["family"] == "12x12x10"} == {"gamma", "reverse"}
    with pytest.raises(ValueError, match="must be even"):
        runner.select_contexts(rows, "evaluation", 1, set())
    with pytest.raises(ValueError, match="overlaps historical adapter training"):
        runner.select_contexts(rows, "evaluation", 2, set(), {"layout-12x12x4-0"})
    dev = [{**row, "split": "development"} for row in rows]
    chosen = runner.select_contexts(dev, "development", 1, set(), {"layout-12x12x4-0"})
    assert next(row for row in chosen if row["family"] == "12x12x4")["board_rank"] == 1


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, **_kwargs):
        rendered = json.dumps(messages, sort_keys=True)
        return list(range(len(rendered) // 10)) if tokenize else rendered


def test_plan_has_complete_cartesian_grid_and_no_teacher_data_in_prompts():
    pair = board_pair()
    pair.update(context_id="context", family="12x12x4", source_family="x4", split="development", board_rank=0,
                adapter_training_exposure=False, reference_pair={**pair, "secret_teacher": "NEVER_IN_THE_PROMPT"})
    supports = [{"record_id": f"support-{i}", "layout_id": f"support-layout-{i}", "split": "train",
                 "validation": {"accepted": True}, "start_observation": pair["source"], "task_goal": runner.GOAL,
                 "intent": {"effects": [{"predicate": "box_at", "row": 5, "col": 5}], "preserve": [],
                            "instruction": None}} for i in range(2)]
    plan, prompts = runner.build_plan([pair], supports, FakeTokenizer(), "new-cohort", 1024, 128)
    assert len(plan["cases"]) == len(prompts) == 9 and len(plan["replay_checks"]) == 3
    assert len({row["case_id"] for row in plan["cases"]}) == 9
    assert plan["context_manifest"][0]["expected_methods"] == ["base", "in_context", "checkpoint220"]
    assert plan["context_manifest"][0]["expected_transforms"] == ["identity", "rot90_ccw", "horizontal_reflection"]
    assert "NEVER_IN_THE_PROMPT" not in json.dumps(prompts)
    assert len(plan["support_intents"]) == 2


def test_atomic_resume_preserves_torn_tail_and_rejects_divergent_index(tmp_path):
    case = {"case_id": "case", "source_rows": board_pair()["source"], "transform": "identity", "method": "base"}
    row = {**runner.output_record(case, "invalid", 2, 1.0), "contract_sha256": "contract"}
    atomic_json(tmp_path / "cases/case.json", row)
    (tmp_path / "predictions.jsonl").write_bytes(json.dumps(row).encode() + b"\n{" )
    restored = runner.reconcile(tmp_path, [case], "contract")
    assert restored["case"] == row and len(list(tmp_path.glob("torn_tail_*.bin"))) == 1
    assert json.loads((tmp_path / "predictions.jsonl").read_text()) == row
    (tmp_path / "cases/case.json").unlink()
    with pytest.raises(ValueError, match="differs from its durable case files"):
        runner.reconcile(tmp_path, [case], "contract")


def test_frozen_inputs_resume_only_on_exact_contract(tmp_path):
    plan, prompts = {"cases": []}, [{"case_id": "x"}]
    contract = {"plan_sha256": digest(plan), "prompts_sha256": digest(prompts)}
    atomic_json(tmp_path / "contract.json", contract)
    runner.persist_frozen(tmp_path, plan, prompts, contract, resume=True)
    assert json.loads((tmp_path / "plan.json").read_text()) == plan
    (tmp_path / "plan.json").unlink()
    with pytest.raises(ValueError, match="disagree with the contract"):
        runner.persist_frozen(tmp_path, {"cases": [1]}, prompts, contract, resume=True)


def test_model_batch_switches_adapter_and_counts_until_eos():
    class BatchTokenizer:
        pad_token_id, eos_token_id = 0, 9

        def __call__(self, prompts, **_kwargs):
            return {"input_ids": torch.tensor([[0, 5], [5, 5]][:len(prompts)]),
                    "attention_mask": torch.tensor([[0, 1], [1, 1]][:len(prompts)])}

        def decode(self, tokens, **_kwargs):
            return ",".join(str(token) for token in tokens if token != 9)

    class Model:
        disabled = False
        observed = []

        @contextlib.contextmanager
        def disable_adapter(self):
            self.disabled = True
            yield
            self.disabled = False

        def generate(self, input_ids, **_kwargs):
            self.observed.append(self.disabled)
            return torch.cat([input_ids, torch.tensor([[7, 9, 0], [8, 9, 0]][:len(input_ids)])], dim=1)

    model, tokenizer = Model(), BatchTokenizer()
    result, _ = runner.model_batch(torch, tokenizer, model, ["a", "b"], adapted=False,
                                   max_new_tokens=3, max_length=10)
    assert result == [("7", 2), ("8", 2)] and model.observed == [True]
    runner.model_batch(torch, tokenizer, model, ["a", "b"], adapted=True, max_new_tokens=3, max_length=10)
    assert model.observed == [True, False]
