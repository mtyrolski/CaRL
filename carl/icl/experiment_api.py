"""Charge cached inference to the same experimental budgets as fresh inference."""
import json
from collections.abc import Callable
from typing import Any
from pathlib import Path

import tiktoken

from carl.icl.api import BudgetExceeded, CachedOpenAI, _CACHE_ATTEMPT_SLOT
from carl.icl.config import Limits


class ExperimentAPI(CachedOpenAI):
    """Charge cache hits and physical requests against the same experimental limits."""

    def __init__(
        self, cache_dir: str | Path, provider: str, limits: Limits, client: Any = None,
        *, encoding: Any = None,
    ) -> None:
        super().__init__(
            cache_dir,
            provider,
            max_requests=limits.max_requests,
            timeout_seconds=limits.timeout_seconds,
            max_wall_seconds=limits.wall_seconds,
            client=client,
        )
        self.limits = limits
        self.encoding = encoding if encoding is not None else tiktoken.get_encoding("o200k_base")
        self.metrics.update(
            logical_requests=0,
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
    ) -> dict[str, Any]:
        """Use a separate cache slot while sending the original, unchanged payload.

        Route through ``self.call`` so subclass counters, logical limits and the
        shared ledger apply to the retry exactly as they do to the first call.
        """
        token = _CACHE_ATTEMPT_SLOT.set(1)
        try:
            return self.call(endpoint, payload, mock)
        finally:
            _CACHE_ATTEMPT_SLOT.reset(token)

    def call(
        self,
        endpoint: str,
        payload: dict[str, Any],
        mock: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Reserve response capacity and charge logical usage, including cached inference."""
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
        if endpoint == "responses" and remaining < payload["max_output_tokens"]:
            # Reserve a full configured completion; keep cache keys stable across runs.
            raise BudgetExceeded("Insufficient remaining output tokens for a full request")
        self.metrics["logical_requests"] += 1
        self.metrics["logical_input_tokens"] += prompt_tokens
        role = payload.get("metadata", {}).get("component")
        if role in {"generator", "policy", "value"}:
            self.metrics[f"{role}_calls"] += 1
        try:
            result = super().call(endpoint, payload, mock)
        except Exception:
            if endpoint == "responses":
                self.metrics["logical_output_tokens"] += payload["max_output_tokens"]
            raise
        usage = result.get("usage", {})
        output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = len(
                self.encoding.encode(result.get("output_text", ""), disallowed_special=())
            )
        self.metrics["logical_output_tokens"] += output_tokens
        return result
