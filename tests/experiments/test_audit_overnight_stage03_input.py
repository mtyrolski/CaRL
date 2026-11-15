"""Synthetic checks for the stage-three input-only audit."""

import copy
import hashlib
import itertools
import json

import pytest

from carl.icl.intent_generator_experiment import prompt_messages
from carl.icl.intent_in_context import in_context_messages
from scripts.audit_overnight_stage03_input import digest
from scripts.audit_overnight_stage03_input import inspect
from scripts.audit_overnight_stage03_input import prompts_match_contract
from scripts.audit_overnight_stage03_input import transform_rows


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, **_kwargs):
        rendered = json.dumps(messages, sort_keys=True)
        return list(rendered.encode()) if tokenize else rendered

    def __call__(self, rendered, **_kwargs):
        return {"input_ids": list(rendered.encode())}


@pytest.fixture
def sample():
    board = ["############", "#@-$-------#"] + ["#----------#"] * 9 + ["############"]
    methods = ["base", "in_context", "checkpoint220"]
    transforms = ["identity", "rot90_ccw", "horizontal_reflection"]
    context = {
        "cohort_id": "fixture", "context_id": "context", "layout_id": "layout", "family": "12x12x4",
        "source_family": "source", "split": "evaluation", "board_rank": 0, "adapter_training_exposure": False,
        "original_rows": board, "expected_methods": methods, "expected_transforms": transforms,
    }
    intent = {"effects": [{"predicate": "box_at", "row": 1, "col": 4}], "preserve": [], "instruction": None}
    support = [{"record_id": f"id-{i}", "layout_id": f"support-{i}", "intent": intent,
                "start_observation": board, "task_goal": "Put every box on a goal.",
                "split": "train", "validation": {"accepted": True}} for i in range(2)]
    tokenizer = FakeTokenizer()
    cases, prompts = [], []
    for method, transform in itertools.product(methods, transforms):
        identifier = digest({"cohort_id": "fixture", "context_id": "context", "method": method,
                             "transform": transform})
        source = transform_rows(board, transform)
        record = {"layout_id": "layout", "start_observation": source, "task_goal": "Put every box on a goal."}
        messages = in_context_messages(record, support) if method == "in_context" else prompt_messages(record)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_sha256 = hashlib.sha256(rendered.encode()).hexdigest()
        case = {key: context[key] for key in ("cohort_id", "context_id", "layout_id", "family", "source_family",
                                               "split", "board_rank", "adapter_training_exposure", "original_rows")}
        case.update(case_id=identifier, method=method, transform=transform, source_rows=source,
                    prompt_sha256=prompt_sha256, prompt_tokens=len(rendered.encode()))
        cases.append(case)
        prompts.append({"case_id": identifier, "messages": messages, "rendered_prompt": rendered,
                        "prompt_sha256": prompt_sha256})
    plan = {"cohort_id": "fixture", "context_manifest": [context], "cases": cases,
            "replay_checks": [{"context_id": "context", "transform": name} for name in transforms],
            "support_intents": [{key: row[key] for key in ("record_id", "layout_id", "intent")} for row in support]}
    freeze = {"contexts": [copy.deepcopy(context)], "source_counts": {"source": 1}, "methods": methods, "transforms": transforms,
              "pairs_sha256": "pairs"}
    contract = {"plan_sha256": digest(plan), "files_sha256": {"pairs": "pairs"}, "max_length": 10000,
                "max_new_tokens": 128}
    return plan, freeze, contract, prompts, tokenizer, support


def test_complete_grid_and_tokens_pass(sample):
    result = inspect(*sample)
    assert result["state"] == "passed"
    assert result["planned_cases"] == result["prompt_count"] == result["grid_size"] == 9
    assert result["token_id_mismatches"] == result["case_metadata_mismatches"] == 0
    assert result["evaluation_outputs_read"] is False
    assert result["model_weights_loaded"] is False


def test_rejects_changed_frozen_metadata_or_unexpected_prompt(sample):
    plan, freeze, contract, prompts, tokenizer, support = copy.deepcopy(sample)
    freeze["contexts"][0]["layout_id"] = "other"
    prompts[0]["messages"][-1]["content"] += " hidden future state"
    result = inspect(plan, freeze, contract, prompts, tokenizer, support)
    assert result["state"] == "failed"
    assert result["frozen_order_matches"] is False
    assert result["prompt_or_hash_mismatches"] == 1


def test_rejects_changed_tokens_and_case_metadata(sample):
    plan, freeze, contract, prompts, tokenizer, support = copy.deepcopy(sample)
    plan["cases"][0]["family"] = "12x12x10"
    contract["plan_sha256"] = digest(plan)

    class MismatchedTokenizer(FakeTokenizer):
        def __call__(self, rendered, **_kwargs):
            return {"input_ids": super().__call__(rendered)["input_ids"] + [0]}

    result = inspect(plan, freeze, contract, prompts, MismatchedTokenizer(), support)
    assert result["state"] == "failed"
    assert result["token_id_mismatches"] == 9
    assert result["case_metadata_mismatches"] == 1


def test_independently_detects_hidden_training_overlap(sample):
    result = inspect(*sample, known_train_layouts={"layout"})
    assert result["state"] == "failed"
    assert result["known_adapter_training_layouts"] == 1
    assert result["historical_train_registry_checked"] is True


def test_contract_hashes_canonical_prompts_not_jsonl_bytes(sample):
    prompts = sample[3]
    contract = {"prompts_sha256": digest(prompts)}
    jsonl_bytes = "".join(json.dumps(row) + "\n" for row in prompts).encode()
    assert hashlib.sha256(jsonl_bytes).hexdigest() != contract["prompts_sha256"]
    assert prompts_match_contract(prompts, contract)
    assert not prompts_match_contract(prompts[:-1], contract)
