"""Bounded inference with endpoint-specific live and replay records.

Limits apply to one client/run, including failed attempts and both endpoints.
This synchronous client is deliberately single-worker; it is not a shared budget
coordinator. Cached replay never instantiates the OpenAI SDK or requires a key.
"""
import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from filelock import FileLock, Timeout

from carl.icl.campaign_budget import CampaignLedger, CampaignLimitExceeded, campaign_stage
from carl.icl.config import DEFAULT_OPENAI_BASE_URL
from carl.icl.responses import chat_output_text, chat_payload, collect_response, is_token_count, normalize_usage

if TYPE_CHECKING:
    from carl.icl.config import ModelSettings

ClientFactory = Callable[["ModelSettings | None", float], Any]
OPENAI_KEY_VARIABLE = "OPENAI_API_KEY"


# The slot is local cache metadata, never part of the payload sent to OpenAI.
# Context-local storage keeps concurrent calls from sharing retry state.
_CACHE_ATTEMPT_SLOT: ContextVar[int] = ContextVar("carl_cache_attempt_slot", default=0)


def request_cache_digest(request: dict[str, Any]) -> str:
    """Identify the full request and, when present, its local retry slot."""
    slot = _CACHE_ATTEMPT_SLOT.get()
    key = request if slot == 0 else {"request": request, "cache_attempt_slot": slot}
    return hashlib.sha256(json.dumps(key, sort_keys=True, allow_nan=False).encode()).hexdigest()


class BudgetExceeded(RuntimeError):
    """An action-independent inference resource ceiling has been exhausted."""


class RequestFailed(RuntimeError):
    """A sanitized transport or model-output failure."""


def _campaign_ledger() -> CampaignLedger:
    """Bind every live/replay client to the same durable campaign ledger."""
    return CampaignLedger()


def create_client(transport: "ModelSettings | None", timeout_seconds: float) -> Any:
    """Construct a zero-retry SDK client using the selected endpoint's credential."""
    from openai import OpenAI

    provider = transport.provider if transport is not None else "openai"
    variable = transport.api_key_env if transport is not None else OPENAI_KEY_VARIABLE
    key = os.environ.get(variable, "").strip()
    if not key or any(marker in key.lower() for marker in ("replace-with", "your-api", "placeholder")):
        raise RequestFailed(f"Set {variable} in the environment for --execute")

    # SDK defaults otherwise copy OpenAI organization/project identifiers to the gateway.
    account = None if provider == "openai" else ""
    base_url = transport.base_url if transport is not None else DEFAULT_OPENAI_BASE_URL
    try:
        return OpenAI(
            api_key=key, base_url=base_url, timeout=timeout_seconds, max_retries=0,
            organization=account, project=account,
        )
    except Exception:
        raise RequestFailed(f"Cannot initialize SDK client for {provider}") from None


def _transport_identity(transport: "ModelSettings | None") -> dict[str, Any] | None:
    """Keep the unversioned default Responses route in its historical namespace."""
    if transport is None:
        return None
    if (
        transport.provider == "openai" and transport.api == "responses"
        and transport.revision is None and transport.api_key_env == OPENAI_KEY_VARIABLE
    ):
        return None
    return {
        "provider": transport.provider,
        "api": transport.api,
        "base_url": transport.base_url,
        "revision": transport.revision,
    }


def _wire_request(
    endpoint: str, payload: dict[str, Any], transport: "ModelSettings | None"
) -> tuple[str, dict[str, Any]]:
    """Reject unsupported capabilities and translate the canonical request."""
    if endpoint not in ("responses", "embeddings"):
        raise ValueError("Unsupported API endpoint")
    if transport is None:
        return endpoint, payload
    capabilities = transport.capabilities
    if endpoint == "embeddings":
        if transport.api != "responses" or not capabilities.embeddings:
            raise ValueError("This transport does not support embeddings")
        return endpoint, payload
    if payload.get("model") != transport.model:
        raise ValueError("Request model differs from the configured transport model")
    if (payload.get("reasoning") or {}).get("effort", "none") != "none" and not capabilities.reasoning:
        raise ValueError("This transport does not support the requested reasoning")
    response_format = ((payload.get("text") or {}).get("format") or {}).get("type", "text")
    if (response_format not in ("text", "json_schema", "json_object")
            or response_format == "json_schema" and not capabilities.json_schema
            or response_format == "json_object" and not capabilities.json_object):
        raise ValueError("This transport does not support the requested structured output")
    if payload.get("stream") and not capabilities.streaming:
        raise ValueError("This transport does not support streaming")
    if transport.api == "chat_completions":
        return "chat_completions", chat_payload(payload)
    return endpoint, payload


@dataclass(frozen=True)
class _PreparedRequest:
    """A request in its component and wire representations."""

    endpoint: str
    payload: dict[str, Any]
    transport: "ModelSettings | None"
    record_fields: dict[str, Any]


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
    json.dumps(result, allow_nan=False)
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
        transport: "ModelSettings | None" = None,
        client_factory: ClientFactory | None = None,
        preview_only: bool = False,
    ) -> None:
        if provider not in {"openai", "replay", "mock"}:
            raise ValueError("provider must be openai, replay or mock")
        if max_requests < 1 or any(
            not math.isfinite(value) or value <= 0 for value in (timeout_seconds, max_wall_seconds)
        ):
            raise ValueError("Request, timeout and wall-time limits must be positive")
        if client is not None and client_factory is not None:
            raise ValueError("Supply a client or a client factory")
        self.provider = provider
        self.preview_only = preview_only
        self.ledger = _campaign_ledger() if provider in {"openai", "replay"} and not preview_only else None
        self.stage = campaign_stage()
        self.budget_tag = os.environ.get("CARL_CAMPAIGN_TAG", "")
        self.cache_root = Path(cache_dir)
        self.cache_dir = self.cache_root / ("mock" if provider == "mock" else "openai")
        self.max_requests = max_requests
        self.timeout_seconds = timeout_seconds
        self.max_wall_seconds = max_wall_seconds
        self.client = client
        self._client_injected = client is not None
        self.transport = transport
        self.client_factory = client_factory or create_client
        self._clients: dict[tuple[str, str], Any] = {}
        self.started = time.perf_counter()
        self.metrics: dict[str, int | float] = {
            "requests": 0,
            "physical_requests": 0,
            "logical_requests": 0,
            "response_requests": 0,
            "chat_completion_requests": 0,
            "embedding_requests": 0,
            "cache_hits": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "api_seconds": 0.0,
        }

    def _client(self, transport: "ModelSettings | None" = None) -> Any:
        if self._client_injected:
            return self.client
        provider = transport.provider if transport is not None else "openai"
        variable = transport.api_key_env if transport is not None else OPENAI_KEY_VARIABLE
        route = (provider, variable)
        if route not in self._clients:
            self._clients[route] = self.client_factory(transport, self.timeout_seconds)
        if route == ("openai", OPENAI_KEY_VARIABLE):
            self.client = self._clients[route]
        return self._clients[route]

    def call(
        self,
        endpoint: str,
        payload: dict[str, Any],
        mock: Callable[[], dict[str, Any]] | None = None,
        *,
        transport: "ModelSettings | None" = None,
    ) -> dict[str, Any]:
        """Return a completed cached response or perform one bounded provider attempt."""
        if self.preview_only:
            raise RequestFailed("A preview client cannot perform inference")
        if time.perf_counter() - self.started >= self.max_wall_seconds:
            raise BudgetExceeded("Run wall-time limit reached")
        transport = self.transport if transport is None else transport
        wire_endpoint, wire_payload = _wire_request(endpoint, payload, transport)
        identity = _transport_identity(transport)
        request = {"endpoint": endpoint, "payload": payload}
        record_fields: dict[str, Any] = {"request": request}
        cache_dir = self.cache_dir
        key: dict[str, Any] = request
        if identity is not None:
            namespace = "mock" if self.provider == "mock" else "provider"
            cache_dir = self.cache_root / "v2" / namespace / identity["provider"] / wire_endpoint
            record_fields.update(transport=identity, wire_endpoint=wire_endpoint, wire_payload=wire_payload)
            key = record_fields
        prepared = _PreparedRequest(wire_endpoint, wire_payload, transport, record_fields)
        slot = _CACHE_ATTEMPT_SLOT.get()
        digest = request_cache_digest(key)
        self.metrics["logical_requests"] += 1
        if self.ledger is not None:
            self.ledger.event(self.stage, "logical_request", wire_endpoint, digest, self.budget_tag)
            if slot:
                self.ledger.event(self.stage, "json_retry", wire_endpoint, digest, self.budget_tag)
        path = cache_dir / f"{digest}.json"
        if self.provider == "replay":
            # Reads are safe without a lock because writers publish by atomic rename.
            return self._call_record(prepared, mock, digest, path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path) + ".lock")
        remaining = self.max_wall_seconds - (time.perf_counter() - self.started)
        if remaining <= 0:
            raise BudgetExceeded("Run wall-time limit reached")
        try:
            lock.acquire(timeout=remaining)
        except Timeout as exc:
            raise BudgetExceeded("Run wall-time limit reached while waiting for cache") from exc
        try:
            return self._call_record(prepared, mock, digest, path)
        finally:
            lock.release()

    def _cached_record(self, prepared: _PreparedRequest, digest: str, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise RequestFailed(f"Cannot read cache record {path}") from None
        if not isinstance(cached, dict):
            raise RequestFailed(f"Invalid cache record {path}")
        mode = "mock" if self.provider == "mock" else "openai"
        if (cached.get("request_hash", digest) != digest
                or cached.get("provider", mode) != mode
                or cached.get("transport") != prepared.record_fields.get("transport")
                or any(cached.get(key) != value for key, value in prepared.record_fields.items())):
            raise RequestFailed(f"Cache request mismatch in {path}")
        if cached.get("status") != "completed":
            # A failed or incomplete record might already have incurred charges.
            raise RequestFailed(f"Cached provider error requires reconciliation: {digest}")
        result_key = "vectors" if prepared.endpoint == "embeddings" else "output_text"
        result_type = list if result_key == "vectors" else str
        if not isinstance(cached.get(result_key), result_type):
            raise RequestFailed(f"Invalid cached output in {path}")
        self.metrics["cache_hits"] += 1
        if self.ledger is not None:
            kind = "local_replay_hit" if self.provider == "replay" else "live_cache_hit"
            self.ledger.event(self.stage, kind, prepared.endpoint, digest, self.budget_tag)
        return cached

    def _reserve(self, prepared: _PreparedRequest, digest: str, deadline: float) -> tuple[Any, str]:
        transport = prepared.transport
        client = self._client() if transport is None else self._client(transport)
        if getattr(client, "max_retries", 0) != 0:
            raise BudgetExceeded("The campaign requires SDK automatic retries to be disabled")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise BudgetExceeded("Run wall-time limit reached before provider dispatch")
        assert self.ledger is not None
        scope: dict[str, Any] = {}
        if transport is not None:
            scope["provider"] = transport.provider
        if isinstance(self.ledger, CampaignLedger):
            scope["protect_request"] = True
        try:
            reservation = self.ledger.reserve(
                self.stage, prepared.endpoint, prepared.payload, digest, self.budget_tag, **scope
            )
        except CampaignLimitExceeded as exc:
            raise BudgetExceeded(str(exc)) from exc
        return client, reservation

    def _call_record(
        self, prepared: _PreparedRequest, mock: Callable[[], dict[str, Any]] | None, digest: str, path: Path
    ) -> dict[str, Any]:
        deadline = self.started + self.max_wall_seconds
        if time.perf_counter() >= deadline:
            raise BudgetExceeded("Run wall-time limit reached")
        cached = self._cached_record(prepared, digest, path)
        if cached is not None:
            return cached
        endpoint, payload = prepared.endpoint, prepared.payload
        if self.provider == "replay":
            if self.ledger is not None:
                self.ledger.event(self.stage, "replay_miss", endpoint, digest, self.budget_tag)
            raise RequestFailed(f"No completed cached {endpoint} response for {digest}")
        if self.metrics["requests"] >= self.max_requests:
            raise BudgetExceeded("Run API request limit reached")
        # Initialise before counting an attempt so a missing local key isn't billed as a request.
        client, reservation = self._reserve(prepared, digest, deadline) if self.provider == "openai" else (None, None)
        start = time.perf_counter()
        record: dict[str, Any] = {
            **prepared.record_fields,
            "provider": self.provider,
            "request_hash": digest,
        }
        if _CACHE_ATTEMPT_SLOT.get():
            record["cache_attempt_slot"] = _CACHE_ATTEMPT_SLOT.get()
        try:
            if start >= deadline:
                raise TimeoutError("Run wall-time limit reached before provider dispatch")
            self.metrics["requests"] += 1
            counter = {
                "responses": "response_requests",
                "chat_completions": "chat_completion_requests",
                "embeddings": "embedding_requests",
            }[endpoint]
            self.metrics[counter] += 1
            if self.provider == "mock":
                if mock is None:
                    raise RequestFailed("This operation has no offline mock")
                result = mock()
                record["response"] = result.copy()
            else:
                assert client is not None
                if callable(getattr(client, "with_options", None)):
                    client = client.with_options(timeout=min(self.timeout_seconds, deadline - start), max_retries=0)
                self.metrics["physical_requests"] += 1
                result = self._dispatch(client, endpoint, payload, record, deadline)
                self._check_output_cap(endpoint, payload, record["usage"])
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Provider response exceeded the run deadline")
            record.update(result, status="completed")
        except BaseException as exc:
            # Provider exception text may contain request details or credentials.
            http_status = getattr(exc, "status_code", None)
            record.update(
                status="error",
                error_type=type(exc).__name__,
                http_status=http_status if isinstance(http_status, int) and 100 <= http_status <= 599 else None,
            )
            record["seconds"] = time.perf_counter() - start
            self._account(record)
            if reservation is not None and self.ledger is not None:
                if not self.ledger.settle(reservation, record.get("usage", {}), payload):
                    self.ledger.mark_unknown(reservation)
                self.ledger.event(self.stage, "provider_error", endpoint, digest, self.budget_tag)
            write_json(path, record)
            if not isinstance(exc, Exception):
                raise
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

    @staticmethod
    def _check_output_cap(endpoint: str, payload: dict[str, Any], usage: dict[str, Any]) -> None:
        cap = payload.get("max_tokens" if endpoint == "chat_completions" else "max_output_tokens")
        actual = usage.get("output_tokens")
        if is_token_count(cap) and is_token_count(actual) and actual > cap:
            raise RequestFailed("Provider usage exceeded the requested output token limit")

    @staticmethod
    def _dispatch(
        client: Any, endpoint: str, payload: dict[str, Any], record: dict[str, Any], deadline: float
    ) -> dict[str, Any]:
        def capture(response: Any) -> None:
            usage = response.usage.model_dump() if response.usage else {}
            record.update(usage=normalize_usage(usage, endpoint), model=response.model, response=_full_response(response))

        if endpoint == "responses":
            response = client.responses.create(**payload)
            if payload.get("stream"):
                response = collect_response(response, deadline=deadline, on_terminal=capture)
            capture(response)
            refused = any(
                part.type == "refusal"
                for item in response.output if item.type == "message"
                for part in item.content
            )
            if response.status != "completed" or refused or not isinstance(response.output_text, str) or not response.output_text.strip():
                raise RequestFailed("Response incomplete, refused or empty")
            return {"output_text": response.output_text, "response_id": response.id}
        if endpoint == "chat_completions":
            response = client.chat.completions.create(**payload)
            capture(response)
            return {"output_text": chat_output_text(record["response"] or {}), "response_id": response.id}
        response = client.embeddings.create(**payload)
        capture(response)
        items = sorted(response.data, key=lambda item: item.index)
        size = 1 if isinstance(payload["input"], str) else len(payload["input"])
        if (not items or any(not is_token_count(item.index) for item in items)
                or [item.index for item in items] != list(range(size))):
            raise RequestFailed("Invalid embedding response indices")
        vectors = [item.embedding for item in items]
        if any(
            not isinstance(vector, list) or not vector or len(vector) != len(vectors[0])
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise RequestFailed("Invalid embedding response vectors")
        return {"vectors": vectors}

    def _account(self, record: dict) -> None:
        self.metrics["api_seconds"] += record["seconds"]
        usage = record.get("usage", {})
        if not isinstance(usage, dict):
            return
        details = usage.get("input_tokens_details")
        details = details if isinstance(details, dict) else {}
        for key, count in (
            ("input_tokens", usage.get("input_tokens", usage.get("prompt_tokens"))),
            ("output_tokens", usage.get("output_tokens")),
            ("cached_input_tokens", details.get("cached_tokens")),
            ("cache_write_tokens", details.get("cache_write_tokens")),
        ):
            if is_token_count(count):
                self.metrics[key] += count
