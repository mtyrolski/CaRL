"""Model selection must not change the controller or reuse another policy's identity."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from carl.icl.api import request_cache_digest
from carl.icl.boards import from_rows, to_rows
from carl.icl.components import ICLComponents
from carl.icl.config import ExperimentConfig
from carl.icl.datasets import environment
from carl.icl.luna import LunaComponents


class RecordingClient:
    """Minimal client implementing the component inference contract."""

    provider = "mock"

    def __init__(self):
        self.metrics = {"cache_hits": 0}
        self.requests = []

    def call(self, endpoint, payload, mock=None):
        self.requests.append((endpoint, payload))
        return {"output_text": json.dumps({"probabilities": [0.0, 0.0, 0.25, 0.75]})}

    def call_json_retry(self, endpoint, payload, mock=None):
        return self.call(endpoint, payload, mock)


def test_legacy_import_is_the_same_model_neutral_component():
    assert LunaComponents is ICLComponents
    assert ExperimentConfig().model == "gpt-6-luna"


def test_model_and_policy_settings_change_fingerprints_and_cache_keys():
    board = from_rows(["#####", "#---#", "#@$.#", "#---#", "#####"])
    env = environment(board)
    target = env.next_state(board, 3)
    training = {"contexts": [{"id": "train", "state": to_rows(board), "remaining_actions": 1,
                             "candidates": [{"k": 1, "actions": [3], "subgoal": to_rows(target)}],
                             "calibration_indices": [0]}]}
    models = ["model-luna", "model-terra", "model-astra"]
    clients = [RecordingClient() for _ in models]
    configs = [ExperimentConfig(model=model, few_shot=1) for model in models]
    configs[1] = ExperimentConfig(model=models[1], few_shot=1, policy_reasoning_effort="medium",
                                  policy_response_format="text", policy_tokens=4096)
    components = [ICLComponents(config, client, env, training) for config, client in zip(configs, clients, strict=True)]
    payloads = [component.policy_payload(board, target) for component in components]
    assert [payload["model"] for payload in payloads] == models
    assert payloads[1]["reasoning"]["effort"] == "medium"
    assert payloads[1]["text"]["format"]["type"] == "text"
    assert payloads[1]["max_output_tokens"] == 4096
    assert len({component.fingerprint() for component in components}) == 3
    assert len({request_cache_digest({"endpoint": "responses", "payload": payload}) for payload in payloads}) == 3
    default_settings = ICLComponents(ExperimentConfig(model=models[1], few_shot=1), RecordingClient(), env, training)
    assert default_settings.fingerprint() != components[1].fingerprint()
    assert request_cache_digest({"endpoint": "responses", "payload": default_settings.policy_payload(board, target)}) != (
        request_cache_digest({"endpoint": "responses", "payload": payloads[1]})
    )
    assert all(config.prompts.policy == configs[0].prompts.policy for config in configs)
    assert all(config.few_shot == configs[0].few_shot for config in configs)
    assert all(config.policy_temperature == configs[0].policy_temperature for config in configs)
    for component in components:
        np.testing.assert_allclose(component.probabilities(board, target), [0, 0, .25, .75])
    assert all(len(client.requests) == 1 for client in clients)


def test_policy_token_budget_is_validated():
    with pytest.raises(ValidationError):
        ExperimentConfig(policy_tokens=1)
