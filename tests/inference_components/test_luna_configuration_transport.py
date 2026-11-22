"""Configuration and transport behaviour of ICL components."""

import json
from pathlib import Path

import numpy as np
import pytest

from carl.icl.api import BudgetExceeded
from carl.icl.api import RequestFailed
from carl.icl.boards import level_key
from carl.icl.config import ExperimentConfig
from carl.icl.config import Limits
from carl.icl.config import load_config
from carl.icl.datasets import environment
from carl.icl.datasets import reverse_board
from carl.icl.datasets import verify_solution
from carl.icl.experiment import credentials
from carl.icl.experiment import matrix_configs
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna import LunaComponents
from tests.inference_components._luna_experiment_fixtures import sokoban_board as sokoban_board
from tests.inference_components._luna_experiment_fixtures import training_for


def test_verified_32x32x10_reverse_board():
    board, actions, metadata = reverse_board(32, 10, 123, pulls=20)
    states = verify_solution(board, actions)
    assert board.shape == (32, 32, 7)
    assert int(np.isin(board.argmax(-1), [3, 4]).sum()) == 10
    assert len(actions) > 20 and metadata["boxes_off_goal"] >= 5
    assert all(level_key(s) == level_key(board) for s in states)


def test_config_nested_inheritance_and_matched_matrix(tmp_path):
    (tmp_path / "base.json").write_text(
        json.dumps({"calibration": {"limits": {"max_requests": 55, "timeout_seconds": 8}}})
    )
    (tmp_path / "child.json").write_text(
        json.dumps({"extends": "base.json", "calibration": {"limits": {"max_requests": 12}}})
    )
    config = load_config(tmp_path / "child.json")
    assert config.calibration.limits.timeout_seconds == 8
    assert config.calibration.limits.max_requests == 12
    for name in ("matrix_pilot.json", "matrix_smoke.json"):
        matrix_configs(Path("configs/icl/experiments") / name)
    with pytest.raises(ValueError):
        ExperimentConfig(method="relative", scales=[1, 4])


def test_prompt_config_overrides_merge_and_reject_unknown_fields(tmp_path):
    (tmp_path / "base.json").write_text(
        json.dumps({"prompts": {"policy": "First policy", "value": "Shared value"}}),
        encoding="utf-8",
    )
    (tmp_path / "child.json").write_text(
        json.dumps({"extends": "base.json", "prompts": {"policy": "Updated policy"}}),
        encoding="utf-8",
    )
    config = load_config(tmp_path / "child.json")
    assert config.prompts.policy == "Updated policy"
    assert config.prompts.value == "Shared value"
    assert config.prompts.generator
    with pytest.raises(ValueError):
        ExperimentConfig.model_validate({"prompts": {"not_a_component": "bad"}})


def test_placeholder_key_fails_before_network(monkeypatch):
    monkeypatch.setattr("carl.icl.experiment.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("OPENAI_API_KEY", "replace-with-your-openai-api-key")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        credentials("openai")
    credentials("mock")


@pytest.mark.parametrize(
    "grid",
    [
        {"relative_scales": []},
        {"relative_scales": [[0.8, 0.2]]},
        {"relative_scales": [[1.0, 0.5], [0.8, 0.2], [0.7, 0.3]]},
        {"fixed_scales": [[8, 4], [8, 4], [16, 8]]},
        {"fixed_scales": [[4, 8], [12, 4], [16, 8]]},
    ],
)
def test_tuning_rejects_unmatched_or_invalid_grids(grid):
    with pytest.raises(ValueError):
        ExperimentConfig(tuning=grid)


def test_tuning_grids_can_be_configured_with_inheritance(tmp_path):
    base = {"tuning": {"relative_scales": [[0.8, 0.3]], "fixed_scales": [[6, 2]]}}
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "experiment.json").write_text(
        json.dumps({"extends": "base.json"}), encoding="utf-8"
    )
    config = load_config(tmp_path / "experiment.json")
    assert config.tuning.relative_scales == [(0.8, 0.3)]
    assert config.tuning.fixed_scales == [(6, 2)]


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_real_sdk_all_three_roles_and_incomplete_usage_accounting(tmp_path, board):
    httpx = pytest.importorskip("httpx")
    OpenAI = pytest.importorskip("openai").OpenAI
    config = ExperimentConfig(method="fixed", scales=[2, 1], few_shot=1)
    training = training_for(board)
    captured = []
    incomplete = False

    def handle(request):
        payload = json.loads(request.content)
        captured.append(payload)
        role = payload["metadata"]["component"]
        answers = dict(
            policy=dict(probabilities=[0, 0, 0, 1]),
            value=dict(value=0.5),
            generator=dict(
                subgoals=[dict(rows=training["contexts"][0]["candidates"][0]["subgoal"])]
            ),
        )
        response = dict(
            id="resp_test",
            object="response",
            created_at=0,
            model="gpt-6-luna",
            status="incomplete" if incomplete else "completed",
            output=[
                dict(
                    type="message",
                    id="msg_test",
                    role="assistant",
                    status="completed",
                    content=[
                        dict(type="output_text", text=json.dumps(answers[role]), annotations=[])
                    ],
                ),
            ],
            usage=dict(
                input_tokens=100,
                output_tokens=10,
                total_tokens=110,
                input_tokens_details=dict(cached_tokens=20),
                output_tokens_details=dict(reasoning_tokens=0),
            ),
        )
        event = {
            "type": "response.incomplete" if incomplete else "response.completed",
            "sequence_number": 0,
            "response": response,
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"event: {event['type']}\ndata: {json.dumps(event)}\n\n",
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = OpenAI(api_key="unit-test-not-a-key", http_client=transport, max_retries=0)
        api = ExperimentAPI(tmp_path, "openai", config.limits, client)
        components = LunaComponents(config, api, environment(board), training)
        np.testing.assert_array_equal(components.probabilities(board, board), [0, 0, 0, 1])
        assert components.value(board) == 0.5
        assert len(components.propose(board, 1)) == 1
        assert {p["metadata"]["component"] for p in captured} == {"policy", "value", "generator"}
        assert all(p["stream"] and p["tools"] == [] for p in captured)
        assert api.metrics["output_tokens"] == 30
        completed = [json.loads(path.read_text()) for path in (tmp_path / "openai").glob("*.json")]
        assert len(completed) == 3
        assert all(record["response"]["output"][0]["content"][0]["text"]
                   == record["output_text"] for record in completed)
        assert all(record["response"]["usage"]["input_tokens"] == 100 for record in completed)
        incomplete = True
        other = np.asarray(environment(board).next_state(board, 0), dtype=np.float32)
        with pytest.raises(RequestFailed):
            components.value(other)
        assert api.metrics["output_tokens"] == 40  # Incomplete responses can still be billed.
        records = [json.loads(path.read_text()) for path in (tmp_path / "openai").glob("*.json")]
        failed = next(record for record in records if record["status"] == "error")
        assert failed["response"]["status"] == "incomplete"


@pytest.mark.usefixtures("deterministic_token_estimates")
def test_total_input_budget_charges_cached_calls(tmp_path):
    payload = dict(max_output_tokens=16, metadata=dict(component="value"))
    first = ExperimentAPI(tmp_path, "mock", Limits())
    first.call("responses", payload, mock=lambda: dict(output_text='{"value":0.5}', usage={}))
    tokens = int(first.metrics["logical_input_tokens"])
    api = ExperimentAPI(tmp_path, "mock", Limits(max_total_input_tokens=tokens))
    api.call("responses", payload)
    with pytest.raises(BudgetExceeded, match="Total estimated input"):
        api.call("responses", payload)
    assert api.metrics["requests"] == 0 and api.metrics["cache_hits"] == 1
