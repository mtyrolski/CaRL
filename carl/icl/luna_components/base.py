"""Shared configuration, payload construction, and response decoding."""
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from carl.icl.api import RequestFailed
from carl.icl.checkpoints.io import load_generator_checkpoint
from carl.icl.checkpoints.types import GeneratorCheckpoint
from carl.icl.config import ExperimentConfig
from carl.icl.experiment_api import ExperimentAPI
from carl.icl.luna_components.formatting import POLICY_COORDINATE_INSTRUCTIONS, POLICY_COORDINATE_VERSION
from carl.icl.responses import RESPONSE_CONTRACT_VERSION, request_options, text_message
from carl.icl.types import BoardEnvironment, ComponentRole, Example, JSONObject, TrainingData


class LunaBase(ABC):
    """Common state and behavior for the component-specific implementations."""

    def __init__(
        self,
        config: ExperimentConfig,
        api: ExperimentAPI,
        env: BoardEnvironment,
        training: TrainingData,
        *,
        calibrated: JSONObject | None = None,
    ) -> None:
        self.config, self.api, self.env = config, api, env
        self.training, self.calibrated = training, calibrated
        self.raw_examples: list[Example] = [
            Example(**candidate, state=context["state"], context=context["id"])
            for context in training["contexts"]
            for candidate in context["candidates"]
        ]
        self.diagnostics = {
            "invalid_proposals": 0,
            "self_copies": 0,
            "duplicate_proposals": 0,
        }
        if config.locked_policy_fingerprint is not None and self.fingerprint() != config.locked_policy_fingerprint:
            raise ValueError("Frozen policy fingerprint mismatch: model, prompt, examples or sampling changed")
        self.generator_checkpoint: GeneratorCheckpoint | None = None
        if config.generator_checkpoint is not None:
            self.generator_checkpoint = load_generator_checkpoint(
                config.path("generator_checkpoint"), model=config.model, policy_fingerprint=self.fingerprint(),
                action_cap=config.calibration.action_cap, provider=api.provider,
            )
    def fingerprint(self) -> str:
        # Excludes test family/size, method, run seed and search scales: frozen transfer.
        """Hash the frozen policy contract without test settings or search scales."""
        contract = {
            "model": self.config.model,
            "reasoning": self.config.policy_reasoning_effort or self.config.reasoning_effort,
            "response_contract": RESPONSE_CONTRACT_VERSION,
            "response_format": self.config.policy_response_format or self.config.response_format,
            "prompt": self.config.prompts.policy,
            "policy_tokens": self.config.policy_tokens,
            "temperature": self.config.policy_temperature,
            "few_shot": self.config.few_shot,
            "training": {key: value for key, value in self.training.items() if key != "value_examples"},
            "policy_examples": self.policy_examples(),
            "action_cap": self.config.calibration.action_cap,
            "sampling": "categorical-unmasked",
            "stopping": "exact-target-or-task-terminal",
        }
        if self.config.policy_coordinate_features:
            contract["policy_coordinates"] = {
                "version": POLICY_COORDINATE_VERSION,
                "instructions": POLICY_COORDINATE_INSTRUCTIONS,
            }
        if self.config.json_max_retries:
            contract["json_retry"] = {"version": "isolated-cache-slot-v1", "max_retries": self.config.json_max_retries}
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    def _payload(
        self,
        role: ComponentRole,
        prompt: str,
        examples: list[tuple[JSONObject, JSONObject]],
        query: JSONObject,
        *,
        schema: JSONObject,
    ) -> JSONObject:
        response_format = getattr(self.config, f"{role}_response_format", None) or self.config.response_format
        reasoning_effort = getattr(self.config, f"{role}_reasoning_effort", None) or self.config.reasoning_effort
        if response_format == "text":
            prompt += "\nReturn only JSON matching this schema: " + json.dumps(schema)
        messages: list[JSONObject] = [text_message("developer", prompt)]
        for question, answer in examples:
            messages.extend(
                [
                    text_message("user", json.dumps(question)),
                    text_message("assistant", json.dumps(answer)),
                ]
            )
        messages.append(text_message("user", json.dumps(query)))
        output_format: JSONObject = {"type": "text"}
        if response_format == "json_schema":
            output_format = {
                "type": "json_schema",
                "name": f"sokoban_{role}",
                "strict": True,
                "schema": schema,
            }
        payload = {
            **request_options(reasoning_effort, output_format),
            "model": self.config.model,
            "input": messages,
            "max_output_tokens": getattr(self.config, f"{role}_tokens"),
            "metadata": {"component": role},
        }
        return payload
    def _json(
        self, payload: JSONObject, mock_value: Callable[[], JSONObject]
    ) -> JSONObject:
        def mock() -> JSONObject:
            return {
                "output_text": json.dumps(mock_value()),
                "model": "mock_all_components",
                "usage": {},
            }

        for attempt in range(self.config.json_max_retries + 1):
            result = (self.api.call("responses", payload, mock=mock) if attempt == 0 else
                      self.api.call_json_retry("responses", payload, mock=mock))
            try:
                value = json.loads(result["output_text"])
                if not isinstance(value, dict):
                    raise ValueError("Expected an object")
                return value
            except (ValueError, TypeError):
                if attempt == self.config.json_max_retries:
                    raise RequestFailed("Invalid JSON from model component") from None
        raise AssertionError("Unreachable retry loop")
    def _diverse(self, examples: list[Example], count: int) -> list[Example]:
        selected, seen = [], set()
        for example in examples:
            if example["context"] not in seen:
                group = [
                    item for item in examples if item["context"] == example["context"]
                ]
                weights = np.array(
                    [item.get("weight", 1.0) for item in group], dtype=float
                )
                context_seed = int(
                    hashlib.sha256(example["context"].encode()).hexdigest()[:8], 16
                )
                rng = np.random.default_rng(
                    np.random.SeedSequence([self.config.seed, context_seed])
                )
                selected.append(
                    group[int(rng.choice(len(group), p=weights / weights.sum()))]
                )
                seen.add(example["context"])
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(
                f"Need {count} distinct few-shot contexts; found {len(selected)}"
            )
        return selected

    @abstractmethod
    def policy_examples(self) -> list[tuple[JSONObject, JSONObject]]:
        """Provided by the policy component."""
        raise NotImplementedError
