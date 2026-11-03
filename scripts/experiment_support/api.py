"""Shared logical generator and token budgets."""
import json
from typing import Any

from carl.icl.api import BudgetExceeded
from carl.icl.experiment_api import ExperimentAPI


class CappedAPI(ExperimentAPI):
    """Limit logical generator requests as well as normal per-run API limits."""

    def __init__(self, *args: Any, max_generator_calls: int, max_total_tokens: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_generator_calls = max_generator_calls
        self.max_total_tokens = max_total_tokens

    def call(self, endpoint: str, payload: dict[str, Any], mock: Any = None) -> dict[str, Any]:
        if payload.get('metadata', {}).get('component') == 'generator' and self.metrics['generator_calls'] >= self.max_generator_calls:
            raise BudgetExceeded('Logical generator request limit exhausted')
        prompt_tokens = len(self.encoding.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=()))
        reserve = payload.get('max_output_tokens', 0) if endpoint == 'responses' else 0
        used = self.metrics['logical_input_tokens'] + self.metrics['logical_output_tokens']
        if used + prompt_tokens + reserve > self.max_total_tokens:
            raise BudgetExceeded('Combined logical input/output token limit exhausted')
        try:
            result = super().call(endpoint, payload, mock)
        except Exception:
            self._verify_total()
            raise
        self._verify_total()
        return result

    def _verify_total(self) -> None:
        if self.metrics['logical_input_tokens'] + self.metrics['logical_output_tokens'] > self.max_total_tokens:
            raise BudgetExceeded('Combined logical token limit exceeded after provider response')
