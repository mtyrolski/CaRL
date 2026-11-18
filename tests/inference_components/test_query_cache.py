"""Durability and reuse of complete local inference records."""
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carl.icl.api import BudgetExceeded, CachedOpenAI, RequestFailed


def fake_response() -> SimpleNamespace:
    """Mimic an SDK response with extra provider data in its full output."""
    raw = {
        "id": "resp_test", "model": "luna", "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
        "extra_provider_field": {"detail": "preserved"},
    }
    return SimpleNamespace(
        status="completed", output_text="answer", output=[], model="luna", id="resp_test",
        usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 2}),
        model_dump=lambda **_kwargs: raw,
    )


def test_full_request_and_output_survive_backup_and_restart(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    payload: dict[str, Any] = {
        "model": "luna", "input": [{"role": "user", "content": "zażółć"}],
        "temperature": 0.2,
    }
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: fake_response()))
    record = CachedOpenAI(cache, "openai", client=client).call("responses", payload)
    assert record["request"] == {"endpoint": "responses", "payload": payload}
    assert record["response"]["extra_provider_field"] == {"detail": "preserved"}
    assert record["response"]["output"][0]["content"][0]["text"] == "answer"

    restored = tmp_path / "restored"
    shutil.copytree(cache, restored)
    replay = CachedOpenAI(restored, "replay")
    assert replay.call("responses", payload) == record
    assert replay.metrics["cache_hits"] == 1
    assert replay.client is None
    for changed in ({**payload, "model": "other"}, {**payload, "temperature": 0.3},
                    {**payload, "input": "different"}):
        with pytest.raises(RequestFailed, match="No completed cached"):
            replay.call("responses", changed)


def test_existing_cache_records_remain_replayable(tmp_path: Path) -> None:
    payload = {"model": "luna", "input": "board"}
    api = CachedOpenAI(tmp_path, "mock")
    record = api.call("responses", payload, mock=lambda: {"output_text": "old", "usage": {}})
    path = next((tmp_path / "mock").glob("*.json"))
    record.pop("response")  # Previous cache schema stored only normalized fields.
    path.write_text(json.dumps(record), encoding="utf-8")
    assert CachedOpenAI(tmp_path, "mock").call("responses", payload) == record


def test_replay_miss_does_not_create_cache_files(tmp_path: Path) -> None:
    cache = tmp_path / "missing"
    with pytest.raises(RequestFailed, match="No completed cached"):
        CachedOpenAI(cache, "replay").call("responses", {"model": "luna", "input": "board"})
    assert not cache.exists()


def test_embeddings_preserve_full_sdk_response(tmp_path: Path) -> None:
    import httpx
    from openai import OpenAI

    payload = {"model": "embedding-model", "input": ["board"]}
    response = {
        "object": "list", "model": "embedding-model", "provider_extra": "kept",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=response))) as http:
        client = OpenAI(api_key="test-only", http_client=http, max_retries=0)
        saved = CachedOpenAI(tmp_path, "openai", client=client).call("embeddings", payload)
    assert saved["response"]["provider_extra"] == "kept"
    assert saved["response"]["data"][0]["embedding"] == [0.5, 0.25]
    assert CachedOpenAI(tmp_path, "replay").call("embeddings", payload) == saved


def test_concurrent_calls_share_one_provider_request(tmp_path: Path) -> None:
    payload = {"model": "luna", "input": "board"}
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def create(**_kwargs: Any) -> SimpleNamespace:
        calls.append(1)
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError("test failed to release provider")
        return fake_response()

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    clients = [CachedOpenAI(tmp_path, "openai", client=client) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(clients[0].call, "responses", payload)
        assert entered.wait(timeout=5)
        second = executor.submit(clients[1].call, "responses", payload)
        release.set()
        assert first.result(timeout=10) == second.result(timeout=10)
    assert len(calls) == 1
    assert sum(api.metrics["cache_hits"] for api in clients) == 1


def test_waiting_for_cache_respects_run_deadline(tmp_path: Path) -> None:
    from filelock import FileLock

    payload = {"model": "luna", "input": "board"}
    api = CachedOpenAI(tmp_path, "mock", max_wall_seconds=0.1)
    request = {"endpoint": "responses", "payload": payload}
    import hashlib
    digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    api.cache_dir.mkdir(parents=True)
    with FileLock(str(api.cache_dir / f"{digest}.json.lock")):
        with pytest.raises(BudgetExceeded, match="waiting for cache"):
            api.call("responses", payload, mock=lambda: {"output_text": "unused"})


def test_corrupt_cache_does_not_trigger_a_live_request(tmp_path: Path) -> None:
    payload = {"model": "luna", "input": "board"}
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kwargs: fake_response()))
    api = CachedOpenAI(tmp_path, "openai", client=client)
    api.call("responses", payload)
    path = next((tmp_path / "openai").glob("*.json"))
    path.write_text("{", encoding="utf-8")
    fresh_client = CachedOpenAI(tmp_path, "openai", client=client)
    with pytest.raises(RequestFailed, match="Cannot read cache record"):
        fresh_client.call("responses", payload)
    assert fresh_client.metrics["requests"] == 0
