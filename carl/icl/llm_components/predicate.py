"""Frozen prompted Sokoban action and intent roles sharing the experiment API."""

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

from carl.icl.api import BudgetExceeded
from carl.icl.config import ExperimentConfig, ModelSettings
from carl.icl.experiment_api import ExperimentAPI, local_token_encoding
from carl.icl.intent_domain import (
    DOMAIN_VERSION, Action, Intent, SokobanObservation, decode_intent, render_instruction, validate_intent_proposal,
)
from carl.icl.llm_components.formatting import schema_object
from carl.icl.local_models import LocalRoleModel
from carl.icl.luna_action_format import action_text_format
from carl.icl.luna_intent_rollout import ACTION_PARSER_VERSION, Action as RolloutAction, ActionAttempt, _action
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.predicate_prompts import (
    GENERATOR_PROMPT_VERSION, PREDICATE_GENERATOR_PROMPT as PREDICATE_GENERATOR_PROMPT, SCALE_INSTRUCTIONS,
    generator_instruction, generator_query,
)
from carl.icl.responses import RESPONSE_CONTRACT_VERSION, is_token_count, text_message
from carl.icl.types import JSONObject

PredicateRole: TypeAlias = Literal["policy", "generator"]
PromptExample: TypeAlias = tuple[JSONObject, JSONObject]
MockText: TypeAlias = Callable[[], str]

POLICY_PROMPT_VERSION: Final = "sokoban-predicate-policy-v1"
INTENT_PARSER_VERSION: Final = "sokoban-observable-intent-json-v1"

PREDICATE_POLICY_PROMPT: Final = (
    "You control a Sokoban player. Choose one primitive action toward satisfying the given instruction. "
    "Make all required effects true together while keeping every preservation condition true. "
    "Coordinates are zero-based row and column. # is wall, - is floor, . is goal, $ is box, "
    "* is box on goal, @ is player, + is player on goal and ? is an unseen cell. "
    "A player can push one box into a free square but cannot pull or push two boxes. "
    "attempt_history lists earlier actions, and no_op means the visible observation did not change. "
    'Respond with exactly one JSON object with an action key and one of "UP", "DOWN", "LEFT" or "RIGHT".'
)
def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _intent_text_format() -> JSONObject:
    position = schema_object({
        "predicate": {"type": "string", "enum": ["box_at", "player_at"]},
        "row": {"type": "integer"},
        "col": {"type": "integer"},
    })
    count = schema_object({
        "predicate": {"type": "string", "enum": ["boxes_on_goals_at_least"]},
        "count": {"type": "integer"},
    })
    clauses = {"type": "array", "items": {"anyOf": [position, count]}}
    return {"format": {
        "type": "json_schema", "name": "sokoban_intent_v1", "strict": True,
        "schema": schema_object({"effects": clauses, "preserve": clauses, "instruction": {"type": "null"}}),
    }}


@dataclass(frozen=True)
class _Role:
    name: PredicateRole
    model: ModelSettings | None
    local: LocalRoleModel | None
    prompt: str
    examples: tuple[tuple[str, str], ...]
    output_tokens: int
    contract: JSONObject


def _role(
    config: ExperimentConfig,
    name: PredicateRole,
    prompt: str,
    examples: Sequence[PromptExample],
    *,
    support_fingerprint: str | None,
    context_limit: int,
    scale_arm: str | None = None,
    local: LocalRoleModel | None = None,
) -> _Role:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{name} prompt must be nonempty")
    if scale_arm is not None:
        prompt = generator_instruction(scale_arm, prompt)
    query_fields = {"observation", "instruction", "attempt_history"} if name == "policy" else {
        "observation", "task_goal", "scale",
    }
    answer_fields = {"action"} if name == "policy" else {"effects", "preserve", "instruction"}
    if any(set(question) != query_fields or set(answer) != answer_fields for question, answer in examples):
        raise ValueError(f"Unexpected fields in {name} support example")
    frozen = tuple((_json(question), _json(answer)) for question, answer in examples)
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    if local is None:
        model = config.role(name)
        model_identity = model.identity()
    else:
        model = None
        if local.role != name:
            raise ValueError(f"Local model was bound for {local.role}, expected {name}")
        local.assert_prompt(prompt_sha256)
        model_identity = local.identity()

    output_tokens = config.policy_tokens if name == "policy" else config.generator_tokens
    examples_sha256 = contract_hash(frozen)
    contract = {
        "role": name,
        "domain_version": DOMAIN_VERSION,
        "response_contract_version": RESPONSE_CONTRACT_VERSION,
        "prompt_version": POLICY_PROMPT_VERSION if name == "policy" else GENERATOR_PROMPT_VERSION,
        "parser_version": ACTION_PARSER_VERSION if name == "policy" else INTENT_PARSER_VERSION,
        "prompt_sha256": prompt_sha256,
        "examples_sha256": examples_sha256,
        "support_fingerprint": support_fingerprint or examples_sha256,
        "shots": len(frozen),
        "model": model_identity,
        "output_tokens": output_tokens,
        "context_limit": context_limit,
    }
    if scale_arm is not None:
        contract["scale_arm"] = scale_arm
        contract["json_max_retries"] = config.json_max_retries
    contract["fingerprint"] = contract_hash(contract)
    return _Role(name, model, local, prompt, frozen, output_tokens, contract)


def _request(role: _Role, query: JSONObject, attempt_seed: int | None = None) -> JSONObject:
    messages = [text_message("developer", role.prompt)]
    for question, answer in role.examples:
        messages.extend((text_message("user", question), text_message("assistant", answer)))
    messages.append(text_message("user", _json(query)))
    model = role.model
    metadata = {"component": role.name, "component_fingerprint": role.contract["fingerprint"]}
    if attempt_seed is not None:
        if not is_token_count(attempt_seed):
            raise ValueError("Attempt seed must be a nonnegative integer")
        metadata["attempt_seed"] = str(attempt_seed)
    text = {"format": {"type": model.response_format if model is not None else "text"}}
    if model is not None and model.response_format == "json_schema":
        text = action_text_format() if role.name == "policy" else _intent_text_format()
    payload = {
        "model": model.model if model is not None else "local",
        "input": messages,
        "metadata": metadata,
        "max_output_tokens": role.output_tokens,
        "text": text,
        "stream": model.capabilities.streaming if model is not None else False,
        "store": False,
        "tools": [],
    }
    if model is not None and model.capabilities.reasoning:
        payload["reasoning"] = {"effort": model.reasoning_effort}
    if model is not None and model.temperature is not None:
        payload["temperature"] = model.temperature
    return payload


@dataclass(frozen=True)
class PredicateComponents:
    """Action and proposal generation with independent prompts and shared budgets."""

    api: ExperimentAPI
    _policy: _Role
    _generator: _Role
    context_limit: int
    scale_arm: str
    _local_metrics: dict[str, int] = field(default_factory=lambda: {
        "local_requests": 0, "local_policy_calls": 0, "local_generator_calls": 0,
        "local_input_tokens": 0, "local_output_tokens": 0,
    })
    _local_started: float = field(default_factory=time.monotonic)

    @property
    def metrics(self) -> dict[str, int | float]:
        """Report local compute separately from logical and physical API requests."""
        local = self._local_metrics if self._policy.local is not None or self._generator.local is not None else {}
        return {**self.api.metrics, **local}

    def is_local(self, role: PredicateRole) -> bool:
        """Report whether a role uses local weights."""
        return self._role(role).local is not None

    def _role(self, role: PredicateRole) -> _Role:
        if role == "policy":
            return self._policy
        if role == "generator":
            return self._generator
        raise ValueError(f"Unknown predicate component role: {role}")

    def contract(self, role: PredicateRole) -> JSONObject:
        """Return the role identity for experiment records and calibration."""
        return json.loads(_json(self._role(role).contract))

    def assert_local_query(
        self, observation: SokobanObservation, *, split: Literal["validation", "test"], task_group: str | None,
        roles: Sequence[PredicateRole] = ("policy", "generator"),
    ) -> None:
        """Check a query against the local checkpoint's training and validation splits."""
        for name in roles:
            local = self._role(name).local
            if local is not None:
                local.assert_query(observation, split=split, task_group=task_group)

    def budget_counts(self, payload: JSONObject) -> dict[str, int]:
        """Estimate the complete canonical request and reserve its output capacity."""
        local = self._role(payload["metadata"]["component"]).local
        if local is None:
            encoding = self.api.encoding if self.api.encoding is not None else local_token_encoding()
            input_tokens = len(encoding.encode(json.dumps(payload, ensure_ascii=False), disallowed_special=()))
            context_limit = self.context_limit
        else:
            input_tokens = local.input_tokens(payload["input"])
            context_limit = local.context_limit(self.context_limit)
        return {
            "input_tokens": input_tokens,
            "output_tokens": payload["max_output_tokens"],
            "context_limit": context_limit,
        }

    def _checked(self, payload: JSONObject) -> JSONObject:
        counts = self.budget_counts(payload)
        if counts["input_tokens"] + counts["output_tokens"] > counts["context_limit"]:
            raise BudgetExceeded(f"Predicate {payload['metadata']['component']} context window exceeded")
        return payload

    def _policy_request(
        self, observation: SokobanObservation, intent: Intent,
        memory: Sequence[ActionAttempt], attempt_seed: int | None,
    ) -> JSONObject:
        if any(not isinstance(item, ActionAttempt) or not isinstance(item.action, RolloutAction)
               or not isinstance(item.no_op, bool) for item in memory):
            raise ValueError("Memory must contain typed actions with a boolean no-op flag")
        return _request(self._policy, {
            "observation": list(observation.rows),
            "instruction": render_instruction(intent),
            "attempt_history": [{"action": item.action.value, "no_op": item.no_op} for item in memory],
        }, attempt_seed)

    def policy_payload(
        self, observation: SokobanObservation, intent: Intent, memory: Sequence[ActionAttempt] = (),
        *, attempt_seed: int | None = None,
    ) -> JSONObject:
        """Build π(x, ℓ(h)) from the observation, instruction and attempt memory."""
        return self._checked(self._policy_request(observation, intent, memory, attempt_seed))

    def generator_payload(self, observation: SokobanObservation, task_goal: str, scale: float | None) -> JSONObject:
        """Build G(x, g, u) from the current observation, task and requested scale."""
        query = generator_query(observation.rows, task_goal, scale, self.scale_arm)
        return self._checked(_request(self._generator, query))

    def _call(self, payload: JSONObject, role: _Role, mock: MockText | None = None, *, retry: bool = False) -> str:
        if role.local is not None:
            return self._call_local(payload, role, mock)

        def mock_response() -> JSONObject:
            assert mock is not None
            return {"output_text": mock(), "model": "mock_predicate_components", "usage": {}}

        call = self.api.call_json_retry if retry else self.api.call
        result = call("responses", payload, mock=mock_response if mock is not None else None, transport=role.model)
        return result["output_text"]

    def _call_local(self, payload: JSONObject, role: _Role, mock: MockText | None) -> str:
        if self.api.preview_only:
            raise ValueError("A preview client cannot perform local inference")
        if mock is not None:
            raise ValueError(f"Mock text cannot replace the local {role.name} model")
        local = role.local
        assert local is not None
        metrics, limits = self._local_metrics, self.api.limits
        if time.monotonic() - self._local_started >= limits.wall_seconds:
            raise BudgetExceeded("Local inference wall-time budget exhausted")
        if metrics["local_requests"] >= limits.max_requests:
            raise BudgetExceeded("Local inference request budget exhausted")
        input_count = local.input_tokens(payload["input"])
        if input_count > limits.max_input_tokens or metrics["local_input_tokens"] + input_count > limits.max_total_input_tokens:
            raise BudgetExceeded("Local inference input token budget exhausted")
        if metrics["local_output_tokens"] + role.output_tokens > limits.max_output_tokens:
            raise BudgetExceeded("Insufficient remaining local output tokens for a full request")
        metrics["local_requests"] += 1
        metrics[f"local_{role.name}_calls"] += 1
        metrics["local_input_tokens"] += input_count
        try:
            output, _, output_count = local.complete(
                payload["input"], output_tokens=role.output_tokens, context_limit=self.context_limit,
            )
        except Exception:
            metrics["local_output_tokens"] += role.output_tokens
            raise
        metrics["local_output_tokens"] += output_count
        return output

    def action(
        self, observation: SokobanObservation, intent: Intent, memory: Sequence[ActionAttempt] = (),
        *, mock: MockText | None = None,
    ) -> Action:
        """Call the configured policy endpoint and parse its one-action text response."""
        payload = self.policy_payload(observation, intent, memory)
        return Action(_action(self._call(payload, self._policy, mock)))

    def propose(
        self, observation: SokobanObservation, task_goal: str, scale: float | None, *, mock: MockText | None = None,
    ) -> Intent:
        """Parse and check the generator's proposal against the visible starting board."""
        payload = self.generator_payload(observation, task_goal, scale)
        raw = self._call(payload, self._generator, mock)
        try:
            intent = decode_intent(raw)
        except ValueError:
            if not self._generator.contract["json_max_retries"]:
                raise
            intent = decode_intent(self._call(payload, self._generator, mock, retry=True))
        problems = validate_intent_proposal(intent, observation)
        if problems:
            raise ValueError("Invalid generated intent: " + ", ".join(problems))
        return intent

    def policy_factory(self, seed: int, *, mock: Callable[[JSONObject], str] | None = None) -> "PredicatePolicy":
        """Construct a new controller for one seeded measurement attempt."""
        if not is_token_count(seed):
            raise ValueError("Attempt seed must be a nonnegative integer")
        return PredicatePolicy(self, seed, mock)


@dataclass(frozen=True)
class PredicatePolicy:
    """Typed stage 04 rollout adapter for the same frozen action role."""

    components: PredicateComponents
    seed: int
    mock: Callable[[JSONObject], str] | None

    @property
    def metrics(self) -> dict[str, int | float]:
        """Expose the stage 02 counters for each trial's usage delta."""
        return self.components.metrics

    def render_action(
        self, observation: SokobanObservation, intent: Intent, memory: Sequence[ActionAttempt],
    ) -> str:
        """Provide the exact rendered messages for the rollout trace."""
        payload = self.components._policy_request(observation, intent, memory, self.seed)
        return _json(payload["input"])

    def choose_action(
        self, observation: SokobanObservation, intent: Intent, memory: Sequence[ActionAttempt],
    ) -> str:
        """Return raw role output for the stage 04 action parser and rollout checker."""
        payload = self.components.policy_payload(observation, intent, memory, attempt_seed=self.seed)
        mock = self.mock
        return self.components._call(payload, self.components._policy, None if mock is None else lambda: mock(payload))


def create_predicate_components(
    config: ExperimentConfig,
    api: ExperimentAPI,
    *,
    context_limit: int,
    policy_prompt: str = PREDICATE_POLICY_PROMPT,
    generator_prompt: str = PREDICATE_GENERATOR_PROMPT,
    policy_examples: Sequence[PromptExample] = (),
    generator_examples: Sequence[PromptExample] = (),
    policy_support_fingerprint: str | None = None,
    generator_support_fingerprint: str | None = None,
    scale_arm: str = "relative_u",
    local_roles: Mapping[PredicateRole, LocalRoleModel] | None = None,
) -> PredicateComponents:
    """Instantiate two configured roles with copied, immutable rendered support."""
    if not is_token_count(context_limit) or context_limit == 0:
        raise ValueError("Context limit must be a positive integer")
    if scale_arm not in SCALE_INSTRUCTIONS:
        raise ValueError(f"Unknown predicate scale arm: {scale_arm}")
    local = local_roles or {}
    if any(role not in ("policy", "generator") for role in local):
        raise ValueError("Only the predicate policy and generator support local weights")
    policy = _role(
        config, "policy", policy_prompt, policy_examples,
        support_fingerprint=policy_support_fingerprint, context_limit=context_limit, local=local.get("policy"),
    )
    generator = _role(
        config, "generator", generator_prompt, generator_examples,
        support_fingerprint=generator_support_fingerprint, context_limit=context_limit, scale_arm=scale_arm,
        local=local.get("generator"),
    )
    return PredicateComponents(api, policy, generator, context_limit, scale_arm)
