"""Check frozen local policy decoding and the controller input contract."""

import json
from typing import Any

import pytest
import torch

from carl.icl.local_intent_policy import LocalIntentPolicy
from carl.icl.local_intent_policy import PolicyConfig
from carl.icl.local_intent_policy import config_hash
from carl.icl.local_intent_policy import run_record
from carl.icl.luna_intent_rollout import RolloutStatus


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 7

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.options = kwargs
        return torch.tensor([[1, 2, 3]])

    def decode(self, tokens, *, skip_special_tokens):
        assert tokens.tolist() == [4, 5]
        assert skip_special_tokens
        return '{"action":"RIGHT"}'


class FakeModel:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))
        self.is_eval = False
        self.options = None

    def eval(self):
        self.is_eval = True

    def parameters(self):
        return [self.parameter]

    def generate(self, **kwargs):
        assert not torch.is_grad_enabled()
        self.options = kwargs
        return torch.cat((kwargs["input_ids"], torch.tensor([[4, 5]])), dim=1)


def config(**overrides: Any) -> PolicyConfig:
    values: dict[str, Any] = {"model": "local", "revision": "revision", "device": "cpu"} | overrides
    return PolicyConfig(**values)


def test_local_policy_uses_frozen_weights_and_greedy_generation():
    tokenizer, model = FakeTokenizer(), FakeModel()
    policy = LocalIntentPolicy(tokenizer, model, torch, config())
    assert policy("current state and intent") == '{"action":"RIGHT"}'
    assert model.is_eval and not model.parameter.requires_grad
    assert model.options is not None
    assert model.options["do_sample"] is False
    assert model.options["pad_token_id"] == 7
    assert tokenizer.messages == [{"role": "user", "content": "current state and intent"}]


def test_oversize_prompt_is_rejected_without_truncating():
    model = FakeModel()
    policy = LocalIntentPolicy(FakeTokenizer(), model, torch, config(max_new_tokens=2, max_context_tokens=4))
    with pytest.raises(ValueError, match="no truncation"):
        policy("too long")
    assert model.options is None


def test_rollout_hides_future_and_uses_real_environment():
    record = {"record_id": "one", "start_observation": ["#######", "#@$-.-#", "#-$-.-#", "#-----#",
                                                            "#-----#", "#-----#", "#######"],
              "intent": {"effects": [{"predicate": "box_at", "row": 1, "col": 3}], "preserve": [],
                         "instruction": "PRIVATE-INSTRUCTION"},
              "task_goal": "put boxes on goals", "later_observation": "PRIVATE-FUTURE",
              "fragment_actions": "PRIVATE-ACTIONS", "k": 1234567}
    prompts = []

    def decide(prompt):
        prompts.append(prompt)
        return '{"action":"RIGHT"}'

    result = run_record(record, decide, 2)
    assert result["result"]["status"] == RolloutStatus.ACHIEVED
    assert result["result"]["actions"] == [3]
    assert "PRIVATE" not in prompts[0] and "1234567" not in prompts[0]
    question = json.loads(prompts[0].split("\n\n")[-1])
    assert set(question) == {"state", "state_coordinates", "intent", "task_goal", "attempt_history"}


def test_config_hash_changes_with_policy_budget():
    assert config_hash(config(action_cap=2)) != config_hash(config(action_cap=3))
