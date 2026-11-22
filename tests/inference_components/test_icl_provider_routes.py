"""Exercise configured ICL endpoints with the SDK and ledger entirely offline."""
import json
from pathlib import Path
import sys

import httpx
from openai import OpenAI
import pytest
from tiktoken.registry import ENCODINGS

from carl.icl.api import BudgetExceeded, RequestFailed, write_json
from carl.icl.boards import from_rows, level_key
from carl.icl.campaign_budget import CampaignLedger
from carl.icl.components import ICLComponents
from carl.icl.config import ExperimentConfig, Limits, ModelSettings, load_config
from carl.icl.datasets import dataset_signature, environment
from carl.icl.experiment import create_inference_api
from scripts import run_hybrid_icl_search, run_icl_search

CONFIGS = Path("configs/icl/experiments")
START = ["#####", "#---#", "#@$.#", "#---#", "#####"]
TARGET = ["#####", "#---#", "#-@*#", "#---#", "#####"]
DEMO = ["#####", "#@#-#", "#-$.#", "#---#", "#####"]
DEMO_NEXT = ["#####", "#-#-#", "#@$.#", "#---#", "#####"]
DEMO_TARGET = ["#####", "#-#-#", "#-@*#", "#---#", "#####"]
VALIDATION = ["#####", "#-###", "#@$.#", "#---#", "#####"]

ANSWERS = {
    "policy": {"probabilities": [0, 0, 0, 1]},
    "generator": {"subgoals": [{"rows": TARGET}]},
    "value": {"value": 0.5},
}

pytestmark = pytest.mark.usefixtures("deterministic_token_estimates")


@pytest.fixture
def route_ledger(tmp_path, monkeypatch):
    price = {
        "input_usd_per_million": "1",
        "output_usd_per_million": "2",
        "max_input_tokens": 16000,
        "max_output_tokens": 4096,
    }
    path = tmp_path / "budget.json"
    write_json(path, {
        "campaign_id": "offline-icl-routes",
        "limit_usd": "1",
        "stage_limits_usd": {"test": "1"},
        "ledger_path": str(tmp_path / "budget.sqlite3"),
        "models": {},
        "routes": {
            "openai/responses/gpt-6-luna": price,
            "openai/responses/shared-model": price,
            "openai/embeddings/shared-embedding": {"input_usd_per_million": "1", "max_input_tokens": 128},
            "mistral/chat_completions/ministral-8b-2512": price,
            "mistral/chat_completions/shared-model": price,
        },
    })
    monkeypatch.delenv("CARL_CAMPAIGN_MAX_USD", raising=False)
    monkeypatch.setenv("CARL_CAMPAIGN_STAGE", "test")
    monkeypatch.setattr("carl.icl.campaign_budget.dotenv_values", lambda _path: {})
    monkeypatch.setattr("carl.icl.api._campaign_ledger", lambda: CampaignLedger(path))
    return CampaignLedger(path)


def _response(payload, answer):
    response = {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": payload["model"],
        "status": "completed",
        "output": [{
            "type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps(answer), "annotations": []}],
        }],
        "usage": {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27},
    }
    if payload.get("stream"):
        event = {"type": "response.completed", "sequence_number": 0, "response": response}
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text=f"event: response.completed\ndata: {json.dumps(event)}\n\n",
        )
    return httpx.Response(200, json=response)


def _chat(payload, answer, *, finish="stop", refusal=None):
    return httpx.Response(200, json={
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": payload["model"],
        "choices": [{
            "index": 0, "finish_reason": finish,
            "message": {"role": "assistant", "content": json.dumps(answer), "refusal": refusal},
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27},
    })


def _prepare_cli_config(tmp_path, example):
    path = tmp_path / "experiment.json"
    write_json(path, {
        "extends": str((CONFIGS / example).resolve()),
        "scales": [1, 2], "easy": False, "few_shot": 1, "instances": 1, "generator_tokens": 128,
        "data_dir": str(tmp_path / "data"), "cache_dir": str(tmp_path / "cache"),
        "output_dir": str(tmp_path / "output"),
    })
    config = load_config(path)
    root = config.path("data_dir")
    write_json(root / "training.json", {
        "provenance": {"signature": dataset_signature(config)},
        "contexts": [{
            "id": "demo", "state": DEMO, "remaining_actions": 2, "calibration_indices": [0, 1],
            "candidates": [
                {"k": 1, "actions": [1], "subgoal": DEMO_NEXT},
                {"k": 2, "actions": [1, 3], "subgoal": DEMO_TARGET},
            ],
        }],
    })
    for split, board in (("validation", VALIDATION), ("test", START)):
        write_json(root / config.family / f"{split}.json", {
            "instances": [{"id": split, "state": board, "level_key": level_key(from_rows(board))}],
        })
    return path


@pytest.mark.parametrize("example, chat_count", [
    ("smoke_fixed.json", 0),
    ("provider_openai.example.json", 0),
    ("provider_ministral.example.json", 1),
])
def test_cli_dispatches_both_config_examples_then_replays(tmp_path, monkeypatch, route_ledger, example, chat_count):
    path = _prepare_cli_config(tmp_path, example)
    seen = []

    def handle(request):
        payload = json.loads(request.content)
        seen.append(request.url.host)
        if request.url.host == "api.mistral.ai":
            assert request.url.path == "/openai/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer fixture-mistral"
            assert payload["model"] == "ministral-8b-2512"
            assert payload["max_tokens"] == 128 and payload["temperature"] == 0
            assert payload["response_format"] == {"type": "json_object"}
            assert payload["messages"][0]["role"] == "system"
            assert "JSON" in payload["messages"][0]["content"]
            assert set(payload).isdisjoint({"reasoning", "input", "metadata", "stream"})
            return _chat(payload, ANSWERS["generator"])
        assert request.url.host == "api.openai.com"
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer fixture-openai"
        return _response(payload, ANSWERS[payload["metadata"]["component"]])

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        monkeypatch.setattr("openai.OpenAI", lambda **options: OpenAI(http_client=http, **options))
        monkeypatch.setattr("carl.icl.experiment.load_dotenv", lambda *_a, **_kw: None)
        monkeypatch.setenv("OPENAI_API_KEY", "fixture-openai")
        monkeypatch.setenv("MISTRAL_API_KEY", "fixture-mistral")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://must-not-be-used.invalid/v1")
        live = tmp_path / "live"
        monkeypatch.setattr(sys, "argv", ["icl", "run", "--config", str(path), "--execute", "--output", str(live)])
        assert run_icl_search.main() == 0
    live_row, = json.loads((live / "results.json").read_text())
    assert live_row["solved"] and live_row["attempted_actions"] == 1
    assert live_row["logical_requests"] == live_row["physical_requests"] == 3
    assert live_row["chat_completion_requests"] == seen.count("api.mistral.ai") == chat_count
    assert len(seen) == 3
    attempts = route_ledger.summary()["attempts"]
    assert sum(row["count"] for row in attempts) == 3
    assert all(row["status"] == "settled" for row in attempts)
    assert sum(row["committed_usd"] for row in attempts) == pytest.approx(0.000102)

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("MISTRAL_API_KEY")
    monkeypatch.setattr("openai.OpenAI", lambda **_kw: pytest.fail("replay constructed an SDK client"))
    replay = tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", ["icl", "run", "--config", str(path), "--replay", "--output", str(replay)])
    assert run_icl_search.main() == 0
    row, = json.loads((replay / "results.json").read_text())
    assert row["solved"] and row["cache_hits"] == row["logical_requests"] == 3
    assert row["physical_requests"] == row["input_tokens"] == 0
    assert route_ledger.summary()["attempts"] == attempts


PAYLOAD = {"model": "shared-model", "input": [{"role": "user", "content": "Reply in JSON"}],
           "max_output_tokens": 16, "metadata": {"component": "value"}}


def test_overlapping_content_and_model_use_distinct_provider_caches(tmp_path, route_ledger, monkeypatch):
    native = ModelSettings(model="shared-model")
    chat = ModelSettings(model="shared-model", provider="mistral")
    config = ExperimentConfig(cache_dir=str(tmp_path / "cache"), limits=Limits(max_requests=2))

    def handle(request):
        payload = json.loads(request.content)
        if request.url.host == "api.mistral.ai":
            return _chat(payload, {"value": 0.7})
        return _response(payload, {"value": 0.1})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        def factory(transport, timeout):
            url = (transport or native).base_url
            return OpenAI(api_key="fake", base_url=url, http_client=http, timeout=timeout, max_retries=0)

        live = create_inference_api(config, "openai", client_factory=factory)
        # Write the historical cache format using the original unscoped call.
        old = live.call("responses", PAYLOAD)
        assert live.call("responses", PAYLOAD, transport=native) == old
        new = create_inference_api(config, "openai", client_factory=factory).call("responses", PAYLOAD, transport=chat)
        assert new["output_text"] != old["output_text"]
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "missing-vocabulary"))
    monkeypatch.delitem(ENCODINGS, "o200k_base", raising=False)
    monkeypatch.setattr("tiktoken.get_encoding", lambda _name: pytest.fail("replay requested vocabulary"))
    replay = create_inference_api(config, "replay", client_factory=lambda *_a: pytest.fail("replay dispatched"))
    assert replay.call("responses", PAYLOAD, transport=chat) == new
    assert replay.call("responses", PAYLOAD, transport=native) == old
    assert replay.metrics["cache_hits"] == replay.metrics["logical_requests"] == 2
    assert replay.metrics["logical_token_byte_fallback"] == 1
    with pytest.raises(BudgetExceeded, match="Logical API request"):
        replay.call("responses", PAYLOAD, transport=chat)
    fresh = create_inference_api(config, "replay")
    with pytest.raises(RequestFailed, match="No completed cached"):
        fresh.call("responses", PAYLOAD, transport=chat.model_copy(update={"revision": "other"}))
    assert len(list(config.path("cache_dir").rglob("*.json"))) == 2


@pytest.mark.parametrize("failure, expected_state", [("length", "settled"), ("refusal", "settled"), ("timeout", "unknown")])
def test_chat_failures_are_redacted_and_bill_only_once(
    tmp_path, route_ledger, failure, expected_state, caplog,
):
    transport = ModelSettings(model="shared-model", provider="mistral")
    config = ExperimentConfig(cache_dir=str(tmp_path / "cache"))
    calls = []

    def handle(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("secret-in-provider-error", request=request)
        payload = json.loads(request.content)
        return _chat(payload, {"value": 1}, finish="length" if failure == "length" else "stop",
                     refusal="secret-in-provider-error" if failure == "refusal" else None)

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = OpenAI(api_key="fake", base_url=transport.base_url, http_client=http, max_retries=0)
        api = create_inference_api(config, "openai", client=client, transport=transport)
        with pytest.raises(RequestFailed) as caught:
            api.call("responses", PAYLOAD)
        assert "secret-in-provider-error" not in str(caught.value) + caplog.text
        with pytest.raises(RequestFailed, match="reconciliation"):
            api.call("responses", PAYLOAD)
    assert len(calls) == api.metrics["physical_requests"] == 1
    row, = route_ledger.summary()["attempts"]
    assert row["status"] == expected_state
    if failure == "timeout":
        cached, = config.path("cache_dir").rglob("*.json")
        assert "secret-in-provider-error" not in cached.read_text()


@pytest.mark.parametrize("provider", ["openai", "mistral"])
def test_campaign_limit_prevents_dispatch_on_either_route(tmp_path, monkeypatch, route_ledger, provider):
    monkeypatch.setenv("CARL_CAMPAIGN_MAX_USD", "0")
    settings = ModelSettings(model="shared-model", provider=provider)
    config = ExperimentConfig(cache_dir=str(tmp_path / "cache"))
    with httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("unexpected SDK call"))) as http:
        client = OpenAI(api_key="fake", base_url=settings.base_url, http_client=http, max_retries=0)
        api = create_inference_api(config, "openai", client=client, transport=settings)
        with pytest.raises(BudgetExceeded, match="budget would be exceeded"):
            api.call("responses", PAYLOAD)
    assert api.metrics["logical_requests"] == 1
    assert api.metrics["physical_requests"] == 0
    assert route_ledger.summary()["attempts"] == []


def test_capability_rejection_precedes_credentials_and_sdk(tmp_path):
    route = ModelSettings.model_validate({
        "model": "shared-model", "provider": "mistral", "capabilities": {"json_schema": False},
    })
    api = create_inference_api(
        ExperimentConfig(cache_dir=str(tmp_path)), "openai", transport=route,
        client_factory=lambda *_a: pytest.fail("invalid request constructed a client"),
    )
    with pytest.raises(ValueError, match="does not support embeddings"):
        api.call("embeddings", {"model": "shared-model", "input": ["board"]})
    with pytest.raises(ValueError, match="does not support the requested reasoning"):
        api.call("responses", {**PAYLOAD, "reasoning": {"effort": "low"}})
    with pytest.raises(ValueError, match="does not support the requested structured output"):
        api.call("responses", {**PAYLOAD, "text": {"format": {"type": "json_schema"}}})
    assert api.metrics["physical_requests"] == 0


def test_hybrid_cli_previews_explicit_generator_without_a_ledger(tmp_path, monkeypatch):
    raw = json.loads((CONFIGS.parent / "sokoban_search.json").read_text())
    selected = load_config(CONFIGS / "provider_ministral.example.json").explicit_role("generator")
    assert selected is not None
    raw.update(
        roles={"generator": selected.model_dump(mode="json")},
        board_size=[5, 5], num_boxes=1, generator_k_list=[1], instances=1, torch_threads=1,
        demonstrations_path=str(tmp_path / "demonstrations.json"),
        instances_path=str(tmp_path / "instances.json"),
        cache_dir=str(tmp_path / "cache"), output=str(tmp_path / "output"),
    )
    write_json(tmp_path / "demonstrations.json", {
        "demonstrations": [{"state": DEMO_NEXT, "subgoal": DEMO_TARGET, "k": 1}],
    })
    write_json(tmp_path / "instances.json", {"instances": [{"id": "test", "state": START}]})
    config = tmp_path / "hybrid.json"
    write_json(config, raw)
    monkeypatch.setattr("carl.icl.api._campaign_ledger", lambda: pytest.fail("preview opened a ledger"))
    monkeypatch.setattr("openai.OpenAI", lambda **_kw: pytest.fail("preview constructed an SDK client"))
    monkeypatch.setattr(run_hybrid_icl_search.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(run_hybrid_icl_search.torch, "manual_seed", lambda _: None)
    monkeypatch.setattr(run_hybrid_icl_search.np.random, "seed", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["icl", "--config", str(config)])
    assert run_hybrid_icl_search.main() == 0
    path, = (tmp_path / "output").glob("*/prepared_requests.json")
    preview = json.loads(path.read_text())
    payload = preview["requests"][0]["payload"]
    assert payload["model"] == selected.model and payload["temperature"] == 0
    assert payload["text"]["format"] == {"type": "json_object"}
    assert not (tmp_path / "cache").exists()


def test_native_embeddings_are_priced_and_replayable(tmp_path, route_ledger):
    transport = ModelSettings(model="shared-model")
    config = ExperimentConfig(cache_dir=str(tmp_path / "cache"))
    payload = {"model": "shared-embedding", "input": ["first", "second"], "encoding_format": "float"}

    def handle(request):
        assert request.url.path == "/v1/embeddings"
        assert json.loads(request.content) == payload
        return httpx.Response(200, json={
            "object": "list", "model": payload["model"],
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
            ],
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        })

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = OpenAI(api_key="fake", base_url=transport.base_url, http_client=http, max_retries=0)
        live = create_inference_api(config, "openai", client=client, transport=transport)
        result = live.call("embeddings", payload)
    assert result["vectors"] == [[1.0, 0.0], [0.0, 1.0]]
    assert live.metrics["physical_requests"] == 1 and live.metrics["input_tokens"] == 8
    attempts = route_ledger.summary()["attempts"]
    assert [(row["status"], row["committed_usd"]) for row in attempts] == [("settled", 0.000008)]
    replay = create_inference_api(config, "replay")
    assert replay.call("embeddings", payload) == result
    assert replay.metrics["logical_requests"] == replay.metrics["cache_hits"] == 1
    assert replay.metrics["physical_requests"] == 0
    assert route_ledger.summary()["attempts"] == attempts


def test_chat_text_fallback_retries_invalid_json_in_an_isolated_slot(tmp_path, route_ledger):
    config = ExperimentConfig.model_validate({
        "cache_dir": str(tmp_path / "cache"), "few_shot": 0, "json_max_retries": 1,
        "roles": {"value": {"provider": "mistral", "model": "shared-model", "response_format": "text"}},
    })
    transport = config.role("value")
    board = from_rows(START)
    seen = []

    def handle(request):
        payload = json.loads(request.content)
        assert "response_format" not in payload
        assert "Return only JSON matching this schema:" in payload["messages"][0]["content"]
        seen.append(payload)
        return _chat(payload, "unexpected text" if len(seen) == 1 else {"value": 0.7})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        client = OpenAI(api_key="fake", base_url=transport.base_url, http_client=http, max_retries=0)
        live = create_inference_api(config, "openai", client=client)
        components = ICLComponents(config, live, environment(board), {"contexts": []})
        assert components.value(board) == 0.7
    assert len(seen) == 2 and seen[0] == seen[1]
    assert live.metrics["physical_requests"] == live.metrics["logical_requests"] == 2
    records = [json.loads(path.read_text()) for path in config.path("cache_dir").rglob("*.json")]
    assert {row.get("cache_attempt_slot", 0) for row in records} == {0, 1}
    replay = create_inference_api(config, "replay")
    components = ICLComponents(config, replay, environment(board), {"contexts": []})
    assert components.value(board) == 0.7
    assert replay.metrics["physical_requests"] == 0
    assert replay.metrics["logical_requests"] == replay.metrics["cache_hits"] == 2


@pytest.mark.parametrize("command", ["preview", "matrix"])
def test_cache_dir_override_reaches_experiment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str) -> None:
    config = ExperimentConfig(method="fixed", scales=[2, 1])
    seen = []
    monkeypatch.setattr(run_icl_search, "load_config", lambda _path: config)
    monkeypatch.setattr(run_icl_search, "matrix_configs", lambda _path: [config, config])
    monkeypatch.setattr(run_icl_search, "preview", lambda item, _provider: seen.append(item) or tmp_path)
    monkeypatch.setattr(run_icl_search, "run_matrix", lambda items, _provider, _directory: seen.extend(items))
    monkeypatch.setattr(sys, "argv", ["run_icl_search", command, "--replay", "--output", str(tmp_path),
                                      "--cache-dir", str(tmp_path / "restored")])
    assert run_icl_search.main() == 0
    assert len(seen) == (1 if command == "preview" else 2)
    assert all(item.cache_dir == str(tmp_path / "restored") for item in seen)
