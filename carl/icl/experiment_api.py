"""Charge cached inference to the same experimental budgets as fresh inference."""
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tiktoken
from tiktoken.registry import ENCODINGS

from carl.icl.api import BudgetExceeded, CachedOpenAI, ClientFactory, RequestFailed, _CACHE_ATTEMPT_SLOT
from carl.icl.config import Limits
from carl.icl.responses import is_token_count

if TYPE_CHECKING:
    from carl.icl.config import ModelSettings

TOKENIZER = "o200k_base"
TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
TOKENIZER_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


class _ByteEncoding:
    """Conservative token count when the local BPE vocabulary is unavailable."""

    def encode(self, text: str, **_kwargs: object) -> bytes:
        """Return UTF-8 bytes, an upper bound for BPE tokenization."""
        return text.encode("utf-8")


def local_token_encoding() -> tiktoken.Encoding | _ByteEncoding:
    """Use the pinned local vocabulary or count bytes without fetching it."""
    if TOKENIZER in ENCODINGS:
        return tiktoken.get_encoding(TOKENIZER)
    cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get(
        "DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache"),
    ))
    if cache_dir:
        path = Path(cache_dir) / hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()
        try:
            with path.open("rb") as stream:
                valid = hashlib.file_digest(stream, "sha256").hexdigest() == TOKENIZER_SHA256
        except OSError:
            valid = False
        if valid:
            return tiktoken.get_encoding(TOKENIZER)
    return _ByteEncoding()


class ExperimentAPI(CachedOpenAI):
    """Charge cache hits and physical requests against the same experimental limits."""

    def __init__(
        self, cache_dir: str | Path, provider: str, limits: Limits, client: Any = None,
        *, encoding: Any = None, transport: "ModelSettings | None" = None,
        client_factory: ClientFactory | None = None, preview_only: bool = False,
    ) -> None:
        super().__init__(
            cache_dir,
            provider,
            max_requests=limits.max_requests,
            timeout_seconds=limits.timeout_seconds,
            max_wall_seconds=limits.wall_seconds,
            client=client,
            transport=transport,
            client_factory=client_factory,
            preview_only=preview_only,
        )
        self.limits = limits
        self.encoding = encoding
        if self.encoding is None and not preview_only:
            self.encoding = local_token_encoding()
        self.metrics.update(
            logical_token_byte_fallback=int(isinstance(self.encoding, _ByteEncoding)),
            logical_input_tokens=0,
            logical_output_tokens=0,
            generator_calls=0,
            policy_calls=0,
            value_calls=0,
        )

    def call_json_retry(
        self,
        endpoint: str,
        payload: dict[str, Any],
        mock: Callable[[], dict[str, Any]] | None = None,
        *,
        transport: "ModelSettings | None" = None,
    ) -> dict[str, Any]:
        """Use a separate cache slot while sending the original, unchanged payload.

        Route through ``self.call`` so subclass counters, logical limits and the
        shared ledger apply to the retry exactly as they do to the first call.
        """
        token = _CACHE_ATTEMPT_SLOT.set(1)
        try:
            if transport is None:
                return self.call(endpoint, payload, mock)
            return self.call(endpoint, payload, mock, transport=transport)
        finally:
            _CACHE_ATTEMPT_SLOT.reset(token)

    def call(
        self,
        endpoint: str,
        payload: dict[str, Any],
        mock: Callable[[], dict[str, Any]] | None = None,
        *,
        transport: "ModelSettings | None" = None,
    ) -> dict[str, Any]:
        """Reserve response capacity and charge logical usage, including cached inference."""
        if self.preview_only:
            raise RequestFailed("A preview client cannot perform inference")
        prompt_tokens = len(
            self.encoding.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=())
        )
        if prompt_tokens > self.limits.max_input_tokens:
            raise BudgetExceeded("Per-request input token estimate exceeded")
        if (
            self.metrics["logical_input_tokens"] + prompt_tokens
            > self.limits.max_total_input_tokens
        ):
            raise BudgetExceeded("Total estimated input token budget exhausted (cache hits count)")
        if self.metrics["logical_requests"] >= self.limits.max_requests:
            raise BudgetExceeded("Logical API request budget exhausted (cache hits count)")
        remaining = self.limits.max_output_tokens - self.metrics["logical_output_tokens"]
        output_cap = payload["max_output_tokens"] if endpoint == "responses" else 0
        if not is_token_count(output_cap):
            raise ValueError("Invalid completion token limit")
        if remaining < output_cap:
            # Reserve a full configured completion; keep cache keys stable across runs.
            raise BudgetExceeded("Insufficient remaining output tokens for a full request")
        self.metrics["logical_input_tokens"] += prompt_tokens
        role = payload.get("metadata", {}).get("component")
        if role in {"generator", "policy", "value"}:
            self.metrics[f"{role}_calls"] += 1
        try:
            result = super().call(endpoint, payload, mock, transport=transport)
        except BaseException:
            self.metrics["logical_output_tokens"] += output_cap
            raise
        usage = result.get("usage")
        output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
        if not is_token_count(output_tokens):
            output_tokens = len(
                self.encoding.encode(result.get("output_text", ""), disallowed_special=())
            )
        self.metrics["logical_output_tokens"] += output_tokens
        return result
