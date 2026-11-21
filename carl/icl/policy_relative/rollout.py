"""Run isolated predicate attempts and replay fixed controller responses."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from carl.icl.config import ModelSettings
from carl.icl.data_sources.sokoban import environment
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.intent_domain import Intent, SokobanObservation, SokobanSnapshot
from carl.icl.luna_action_format import action_text_format
from carl.icl.luna_intent_rollout import (
    ACTION_PARSER_VERSION, MEASUREMENT_PROMPT_VERSION, ActionAttempt, ObservationPolicy, rollout,
)
from carl.icl.responses import is_token_count, text_message
from carl.icl.types import Board

__all__ = ["ACTION_PARSER_VERSION", "MEASUREMENT_PROMPT_VERSION", "ReplayIntentPolicy", "run_measurement_attempt"]

PolicyFactory: TypeAlias = Callable[[int], Callable[[str], str] | ObservationPolicy]


@dataclass(frozen=True)
class ReplayIntentPolicy:
    """A controller whose requests are resolved only against the stage 02 cache."""

    api: ExperimentAPI
    model_settings: ModelSettings
    output_tokens: int
    seed: int

    def __post_init__(self) -> None:
        if self.api.provider != "replay":
            raise ValueError("Intent measurements support only offline replay for API controllers")
        if not is_token_count(self.output_tokens) or not self.output_tokens:
            raise ValueError("output_tokens must be a positive integer")
        if not is_token_count(self.seed):
            raise ValueError("seed must be a nonnegative integer")

    @property
    def metrics(self) -> dict[str, int | float]:
        return self.api.metrics

    def payload(self, prompt: str) -> dict[str, Any]:
        """Build a reproducible cache key for one step in this seeded attempt."""
        settings = self.model_settings
        output_format = (
            action_text_format() if settings.response_format == "json_schema"
            else {"format": {"type": settings.response_format}}
        )
        payload = {
            "model": settings.model,
            "input": [text_message("user", prompt)],
            "metadata": {"component": "policy", "attempt_seed": str(self.seed)},
            "max_output_tokens": self.output_tokens,
            "text": output_format,
            "stream": settings.capabilities.streaming,
            "store": False,
            "tools": [],
        }
        if settings.capabilities.reasoning:
            payload["reasoning"] = {"effort": settings.reasoning_effort}
        if settings.temperature is not None:
            payload["temperature"] = settings.temperature
        return payload

    def __call__(self, prompt: str) -> str:
        result = self.api.call("responses", self.payload(prompt), transport=self.model_settings)
        return result["output_text"]


def run_measurement_attempt(
    snapshot: SokobanSnapshot,
    intent: Intent | dict[str, Any],
    task_goal: str,
    *,
    cap: int,
    seed: int,
    initial_memory: Sequence[ActionAttempt],
    policy_factory: PolicyFactory,
    observe: Callable[[Board], SokobanObservation] | None = None,
) -> dict[str, Any]:
    """Restore the simulator and construct a fresh, seeded controller per trial."""
    if not is_token_count(seed):
        raise ValueError("seed must be a nonnegative integer")
    state = snapshot.restore()
    policy = policy_factory(seed)
    before = dict(getattr(policy, "metrics", {}))
    result = rollout(
        environment(state), state, dict(intent), policy, cap=cap, task_goal=task_goal,
        initial_memory=initial_memory, observe=observe, measured=True,
    )
    metrics = getattr(policy, "metrics", {})
    return {
        "status": result.status.value,
        "primitive_actions_attempted": len(result.actions),
        "first_hit_actions": result.first_hit_actions,
        "task_solved": result.task_solved,
        "model_calls": result.model_calls,
        "metrics": {
            key: value if key == "logical_token_byte_fallback" else value - before.get(key, 0)
            for key, value in metrics.items()
        },
        "actions": result.actions,
        "steps": result.steps,
        "endpoint": result.endpoint,
        "error": result.error,
    }
