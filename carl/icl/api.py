"""Bounded Responses/Embeddings calls with separate mock and live caches.

Limits apply to one client/run, including failed attempts and both endpoints.
This synchronous client is deliberately single-worker; it is not a shared budget
coordinator. Cached replay never instantiates the OpenAI SDK or requires a key.
"""
import hashlib
import json
import os
import time
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from uuid import uuid4

from filelock import FileLock, Timeout

from carl.icl.campaign_budget import CampaignLedger, CampaignLimitExceeded, campaign_stage
from carl.icl.responses import collect_response


# The slot is local cache metadata, never part of the payload sent to OpenAI.
# Context-local storage keeps concurrent calls from sharing retry state.
_CACHE_ATTEMPT_SLOT: ContextVar[int] = ContextVar("carl_cache_attempt_slot", default=0)


def request_cache_digest(request: dict[str, Any]) -> str:
    """Identify the full request and, when present, its local retry slot."""
    slot = _CACHE_ATTEMPT_SLOT.get()
    key = request if slot == 0 else {"request": request, "cache_attempt_slot": slot}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


class BudgetExceeded(RuntimeError):
    """An action-independent inference resource ceiling has been exhausted."""


class RequestFailed(RuntimeError):
    """A sanitized transport or model-output failure."""


def _campaign_ledger() -> CampaignLedger:
    """Bind every live/replay client to the same durable campaign ledger."""
    return CampaignLedger()


def write_json(path: Path, data: Any) -> None:
    """Atomically replace a UTF-8 JSON artifact, rejecting non-finite numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _full_response(response: Any) -> dict[str, Any] | None:
    """Serialize all fields returned by the SDK, including unknown provider fields."""
    dump = getattr(response, "model_dump", None)
    if not callable(dump):
        return None
    result = dump(mode="json", exclude_unset=True)
    if not isinstance(result, dict):
        raise TypeError("SDK response must serialize to a JSON object")
    return result


class CachedOpenAI:
    """Bound and cache physical requests while keeping mock and live data separate."""

    def __init__(
        self,
        cache_dir: str | Path,
        provider: str = "replay",
        *,
        max_requests: int = 12,
        timeout_seconds: float = 60,
        max_wall_seconds: float = 300,
        client: Any = None,
    ) -> None:
        if provider not in {"openai", "replay", "mock"}:
            raise ValueError("provider must be openai, replay or mock")
        if max_requests < 1 or timeout_seconds <= 0 or max_wall_seconds <= 0:
            raise ValueError("Request, timeout and wall-time limits must be positive")
        self.provider = provider
        self.ledger = _campaign_ledger() if provider in {"openai", "replay"} else None
        self.stage = campaign_stage()
        self.budget_tag = os.environ.get("CARL_CAMPAIGN_TAG", "")
        self.cache_dir = Path(cache_dir) / ("mock" if provider == "mock" else "openai")
        self.max_requests = max_requests
        self.timeout_seconds = timeout_seconds
        self.max_wall_seconds = max_wall_seconds
        self.client = client
        self.started = time.perf_counter()
        self.metrics: dict[str, int | float] = {
            "requests": 0,
            "response_requests": 0,
            "embedding_requests": 0,
            "cache_hits": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "api_seconds": 0.0,
        }

    def _client(self) -> Any:
        if self.client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RequestFailed("Set OPENAI_API_KEY in the environment for --execute")
            from openai import OpenAI

            self.client = OpenAI(timeout=self.timeout_seconds, max_retries=0)
        return self.client

    def call(
        self,
        endpoint: str,
        payload: dict[str, Any],
        mock: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a completed cached response or perform one bounded provider attempt."""
        if time.perf_counter() - self.started >= self.max_wall_seconds:
            raise BudgetExceeded("Run wall-time limit reached")
        if endpoint not in {"responses", "embeddings"}:
            raise ValueError("Unsupported API endpoint")
        request = {"endpoint": endpoint, "payload": payload}
        slot = _CACHE_ATTEMPT_SLOT.get()
        digest = request_cache_digest(request)
        if self.ledger is not None:
            self.ledger.event(self.stage, "logical_request", endpoint, digest, self.budget_tag)
            if slot:
                self.ledger.event(self.stage, "json_retry", endpoint, digest, self.budget_tag)
        path = self.cache_dir / f"{digest}.json"
        if self.provider == "replay":
            # Reads are safe without a lock because writers publish by atomic rename.
            return self._call_record(request, mock, digest, path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path) + ".lock")
        remaining = self.max_wall_seconds - (time.perf_counter() - self.started)
        if remaining <= 0:
            raise BudgetExceeded("Run wall-time limit reached")
        try:
            lock.acquire(timeout=remaining)
        except Timeout as exc:
            raise BudgetExceeded("Run wall-time limit reached while waiting for cache") from exc
        try:
            return self._call_record(request, mock, digest, path)
        finally:
            lock.release()

    def _call_record(
        self,
        request: dict[str, Any],
        mock: Callable[[], dict[str, Any]] | None,
        digest: str,
        path: Path,
    ) -> dict[str, Any]:
        endpoint = request["endpoint"]
        payload = request["payload"]
        if time.perf_counter() - self.started >= self.max_wall_seconds:
            raise BudgetExceeded("Run wall-time limit reached")
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RequestFailed(f"Cannot read cache record {path}") from exc
            if not isinstance(cached, dict):
                raise RequestFailed(f"Invalid cache record {path}")
            if cached.get("request") != request:
                raise RequestFailed(f"Cache request mismatch in {path}")
            if cached.get("status") == "completed":
                self.metrics["cache_hits"] += 1
                if self.ledger is not None:
                    kind = "local_replay_hit" if self.provider == "replay" else "live_cache_hit"
                    self.ledger.event(self.stage, kind, endpoint, digest, self.budget_tag)
                return cached
            if cached.get("status") == "error":
                # A failed physical request can have unknown provider billing.
                # Reusing its slot would both hide the record and risk a second
                # charge without explicit reconciliation.
                raise RequestFailed(f"Cached provider error requires reconciliation: {digest}")
        if self.provider == "replay":
            if self.ledger is not None:
                self.ledger.event(self.stage, "replay_miss", endpoint, digest, self.budget_tag)
            raise RequestFailed(f"No completed cached {endpoint} response for {digest}")
        if self.metrics["requests"] >= self.max_requests:
            raise BudgetExceeded("Run API request limit reached")
        # Initialise before counting an attempt so a missing local key isn't billed as a request.
        client = self._client() if self.provider == "openai" else None
        reservation = None
        if self.provider == "openai":
            if getattr(client, "max_retries", 0) != 0:
                raise BudgetExceeded("The campaign requires SDK automatic retries to be disabled")
            assert self.ledger is not None
            try:
                reservation = self.ledger.reserve(self.stage, endpoint, payload, digest, self.budget_tag)
            except CampaignLimitExceeded as exc:
                raise BudgetExceeded(str(exc)) from exc
        self.metrics["requests"] += 1
        self.metrics["response_requests" if endpoint == "responses" else "embedding_requests"] += 1
        start = time.perf_counter()
        record: dict[str, Any] = {
            "request": request,
            "provider": self.provider,
            "request_hash": digest,
        }
        if _CACHE_ATTEMPT_SLOT.get():
            record["cache_attempt_slot"] = _CACHE_ATTEMPT_SLOT.get()
        try:
            if self.provider == "mock":
                if mock is None:
                    raise RequestFailed("This operation has no offline mock")
                result = mock()
                record["response"] = result.copy()
            elif endpoint == "responses":
                assert client is not None
                response = client.responses.create(**payload)
                if payload.get("stream"):
                    response = collect_response(
                        response,
                        deadline=self.started + self.max_wall_seconds,
                        on_terminal=lambda terminal: record.update(
                            usage=terminal.usage.model_dump() if terminal.usage else {},
                            model=terminal.model,
                            response=_full_response(terminal),
                        ),
                    )
                record["response"] = _full_response(response)
                record["usage"] = response.usage.model_dump() if response.usage else {}
                record["model"] = response.model
                refused = any(
                    part.type == "refusal"
                    for item in response.output
                    if item.type == "message"
                    for part in item.content
                )
                if response.status != "completed" or refused or not response.output_text:
                    raise RequestFailed("Response incomplete, refused or empty")
                result = {
                    "output_text": response.output_text,
                    "model": response.model,
                    "response_id": response.id,
                    "usage": response.usage.model_dump() if response.usage else {},
                }
            else:
                assert client is not None
                response = client.embeddings.create(**payload)
                record["response"] = _full_response(response)
                result = {
                    "vectors": [
                        item.embedding
                        for item in sorted(response.data, key=lambda item: item.index)
                    ],
                    "model": response.model,
                    "usage": response.usage.model_dump(),
                }
            record.update(result, status="completed")
        except Exception as exc:
            # Provider exception text may contain request details or credentials.
            record.update(
                status="error",
                error_type=type(exc).__name__,
                http_status=getattr(exc, "status_code", None),
            )
            record["seconds"] = time.perf_counter() - start
            self._account(record)
            if reservation is not None and self.ledger is not None:
                if not self.ledger.settle(reservation, record.get("usage", {}), payload):
                    self.ledger.mark_unknown(reservation)
                self.ledger.event(self.stage, "provider_error", endpoint, digest, self.budget_tag)
            write_json(path, record)
            raise RequestFailed(
                f"{endpoint} failed ({record['error_type']}, HTTP {record['http_status']}); see cache record"
            ) from None
        record["seconds"] = time.perf_counter() - start
        self._account(record)
        if reservation is not None and self.ledger is not None:
            if not self.ledger.settle(reservation, record.get("usage", {}), payload):
                self.ledger.mark_unknown(reservation)
        write_json(path, record)
        return record

    def _account(self, record: dict) -> None:
        self.metrics["api_seconds"] += record["seconds"]
        usage = record.get("usage", {})
        details = usage.get("input_tokens_details") or {}
        self.metrics["input_tokens"] += (
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        )
        self.metrics["output_tokens"] += usage.get("output_tokens", 0) or 0
        self.metrics["cached_input_tokens"] += details.get("cached_tokens", 0) or 0
        self.metrics["cache_write_tokens"] += details.get("cache_write_tokens", 0) or 0
