"""One explicit HTTP-transient retry in a separate durable cache/ledger slot."""
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carl.icl.api import RequestFailed, _CACHE_ATTEMPT_SLOT, request_cache_digest
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.types import JSONObject

HTTP_RETRY_POLICY = {
    'max_retries': 1, 'statuses': ['429', '500-599'], 'base_delay_seconds': 2.0,
    'maximum_retry_after_seconds': 30.0, 'slots': 'original+2',
    'auth_or_unknown_transport': 'never retry', 'unknown_reservation': 'retain',
    'pending_recovery': 'completed base or eligible completed retry cache required; never new network',
}
Call = Callable[[str, JSONObject, Callable[[], JSONObject] | None], JSONObject]


def cache_record(cache_dir: Path, request: JSONObject, slot: int) -> JSONObject | None:
    """Read only the matching immutable cache record; never alter failed evidence."""
    token = _CACHE_ATTEMPT_SLOT.set(slot)
    try:
        path = cache_dir / f'{request_cache_digest(request)}.json'
    finally:
        _CACHE_ATTEMPT_SLOT.reset(token)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) and record.get('request') == request else None


def transient(record: JSONObject | None) -> bool:
    """Only a recorded HTTP response establishes eligibility; timeout text does not."""
    if record is None or record.get('status') != 'error':
        return False
    status = record.get('http_status')
    return isinstance(status, int) and not isinstance(status, bool) and (status == 429 or 500 <= status <= 599)


def has_completed_response(api: ExperimentAPI, endpoint: str, payload: JSONObject) -> bool:
    """Preflight pending recovery across the original and its one authorized retry slot."""
    request = {'endpoint': endpoint, 'payload': payload}
    slot = _CACHE_ATTEMPT_SLOT.get()
    primary = cache_record(api.cache_dir, request, slot)
    if primary is not None and primary.get('status') == 'completed':
        return True
    if slot not in (0, 1) or not transient(primary):
        return False
    retry = cache_record(api.cache_dir, request, slot + 2)
    return retry is not None and retry.get('status') == 'completed'


def delay_seconds(record: JSONObject) -> float:
    """Use a bounded numeric Retry-After when recorded, otherwise two seconds."""
    # The current core cache omits response headers, so normally the fallback applies.
    value: Any = record.get('retry_after_seconds')
    headers = record.get('headers')
    if value is None and isinstance(headers, dict):
        value = headers.get('retry-after', headers.get('Retry-After'))
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 2.0
    return min(max(parsed, 0.0), 30.0) if math.isfinite(parsed) else 2.0


def call_with_transport_retry(api: ExperimentAPI, endpoint: str, payload: JSONObject,
                              mock: Callable[[], JSONObject] | None = None, *, call: Call) -> JSONObject:
    """Call the normal metered client twice at most, preserving failed/unknown records.

    The callback must bypass this helper and the outer response journal. Each call
    still passes through ExperimentAPI/CachedOpenAI, including separate ledger
    reservations. Replay mode never sleeps or permits a fresh provider request.
    """
    slot = _CACHE_ATTEMPT_SLOT.get()
    try:
        return call(endpoint, payload, mock)
    except RequestFailed:
        if api.provider not in {'openai', 'replay'} or slot not in (0, 1):
            raise
        request = {'endpoint': endpoint, 'payload': payload}
        failed = cache_record(api.cache_dir, request, slot)
        if not transient(failed):
            raise
        assert failed is not None
        if api.provider == 'openai':
            time.sleep(delay_seconds(failed))
        token = _CACHE_ATTEMPT_SLOT.set(slot + 2)
        try:
            if api.ledger is not None:
                api.ledger.event(api.stage, 'http_transient_retry', endpoint,
                                 request_cache_digest(request), api.budget_tag)
            return call(endpoint, payload, mock)
        finally:
            _CACHE_ATTEMPT_SLOT.reset(token)
