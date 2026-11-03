"""Bounded HTTP retry and pending-recovery slot selection without provider calls."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from carl.icl.api import RequestFailed, _CACHE_ATTEMPT_SLOT, request_cache_digest
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.types import JSONObject
from scripts.launch100.retry import call_with_transport_retry, delay_seconds, has_completed_response


def api_at(path: Path, provider: str = 'openai') -> ExperimentAPI:
    return cast(ExperimentAPI, SimpleNamespace(cache_dir=path, provider=provider, ledger=None))


def save(path: Path, request: JSONObject, status: str, http: int | None = None) -> None:
    (path / f'{request_cache_digest(request)}.json').write_text(json.dumps(
        {'request': request, 'status': status, 'http_status': http}))


@pytest.mark.parametrize('http', [429, 500, 503, 599])
@pytest.mark.parametrize('base_slot', [0, 1])
def test_transient_retry_separate_slot_and_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   http: int, base_slot: int) -> None:
    monkeypatch.setattr('scripts.launch100.retry.time.sleep', lambda _seconds: None)
    api, payload = api_at(tmp_path), {'model': 'gpt-6-luna'}
    request = {'endpoint': 'responses', 'payload': payload}
    slots: list[int] = []
    def call(endpoint: str, actual: JSONObject, _mock: object = None) -> JSONObject:
        assert endpoint == 'responses' and actual is payload
        slots.append(_CACHE_ATTEMPT_SLOT.get())
        if len(slots) == 1:
            save(tmp_path, request, 'error', http)
            assert not has_completed_response(api, endpoint, actual)
            raise RequestFailed('recorded provider HTTP error')
        save(tmp_path, request, 'completed')
        return {'output_text': 'ok'}
    token = _CACHE_ATTEMPT_SLOT.set(base_slot)
    try:
        assert call_with_transport_retry(api, 'responses', payload, call=call)['output_text'] == 'ok'
        assert slots == [base_slot, base_slot + 2]
        assert _CACHE_ATTEMPT_SLOT.get() == base_slot
        assert has_completed_response(api, 'responses', payload)
    finally:
        _CACHE_ATTEMPT_SLOT.reset(token)


@pytest.mark.parametrize('http', [None, 400, 401, 403, 408])
def test_auth_timeout_and_other_errors_never_retry(tmp_path: Path, http: int | None) -> None:
    api, payload = api_at(tmp_path), {'model': 'gpt-6-luna'}
    calls = []
    def call(endpoint: str, actual: JSONObject, _mock: object = None) -> JSONObject:
        calls.append(endpoint)
        save(tmp_path, {'endpoint': endpoint, 'payload': actual}, 'error', http)
        raise RequestFailed('provider error')
    with pytest.raises(RequestFailed):
        call_with_transport_retry(api, 'responses', payload, call=call)
    assert len(calls) == 1 and not has_completed_response(api, 'responses', payload)


def test_failed_retry_stops_after_two_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('scripts.launch100.retry.time.sleep', lambda _seconds: None)
    api, payload = api_at(tmp_path), {'model': 'gpt-6-luna'}
    calls = []
    def call(endpoint: str, actual: JSONObject, _mock: object = None) -> JSONObject:
        calls.append(_CACHE_ATTEMPT_SLOT.get())
        save(tmp_path, {'endpoint': endpoint, 'payload': actual}, 'error', 503)
        raise RequestFailed('provider error')
    with pytest.raises(RequestFailed):
        call_with_transport_retry(api, 'responses', payload, call=call)
    assert calls == [0, 2] and not has_completed_response(api, 'responses', payload)


def test_retry_after_is_bounded() -> None:
    assert delay_seconds({}) == 2.0
    assert delay_seconds({'headers': {'Retry-After': '300'}}) == 30.0
    assert delay_seconds({'retry_after_seconds': float('nan')}) == 2.0


def test_pending_successful_retry_replays_without_sleep_or_new_attempt(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_sleep(_seconds: float) -> None:
        raise AssertionError('Replay must not back off or call a provider')
    monkeypatch.setattr('scripts.launch100.retry.time.sleep', forbidden_sleep)
    api, payload = api_at(tmp_path, 'replay'), {'model': 'gpt-6-luna'}
    request = {'endpoint': 'responses', 'payload': payload}
    save(tmp_path, request, 'error', 429)
    token = _CACHE_ATTEMPT_SLOT.set(2)
    try:
        save(tmp_path, request, 'completed')
    finally:
        _CACHE_ATTEMPT_SLOT.reset(token)
    calls = []
    def replay(endpoint: str, actual: JSONObject, _mock: object = None) -> JSONObject:
        assert api.provider == 'replay' and endpoint == 'responses' and actual is payload
        slot = _CACHE_ATTEMPT_SLOT.get()
        calls.append(slot)
        if slot == 0:
            raise RequestFailed('Cached provider error')
        assert slot == 2
        return {'output_text': 'saved response'}
    assert has_completed_response(api, 'responses', payload)
    assert call_with_transport_retry(api, 'responses', payload, call=replay)['output_text'] == 'saved response'
    assert calls == [0, 2]
