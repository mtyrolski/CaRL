"""Frozen prompted Sokoban action and intent roles sharing the experiment API."""

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias, cast

from carl.icl.api import BudgetExceeded, RequestFailed
from carl.icl.config import ExperimentConfig, ModelSettings
from carl.icl.experiment_api import ExperimentAPI, local_token_encoding
from carl.icl.intent_domain import (
    DOMAIN_VERSION, Action, Intent, SokobanObservation, decode_intent, render_instruction, validate_intent_proposal,
)
from carl.icl.llm_components.formatting import schema_object
from carl.icl.local_models import LocalRole, LocalRoleModel
from carl.icl.luna_action_format import action_text_format
from carl.icl.luna_intent_rollout import ACTION_PARSER_VERSION, Action as RolloutAction, ActionAttempt, _action
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.predicate_prompts import (
    GENERATOR_PROMPT_VERSION, PREDICATE_GENERATOR_PROMPT as PREDICATE_GENERATOR_PROMPT, SCALE_INSTRUCTIONS,
    generator_instruction, generator_query,
)
from carl.icl.responses import RESPONSE_CONTRACT_VERSION, is_token_count, text_message
from carl.icl.types import JSONObject

PredicateRole: TypeAlias = Literal["policy", "generator", "value"]
PromptExample: TypeAlias = tuple[JSONObject, JSONObject]
MockText: TypeAlias = Callable[[], str]

POLICY_PROMPT_VERSION: Final = "sokoban-predicate-policy-v1"
INTENT_PARSER_VERSION: Final = "sokoban-observable-intent-json-v1"
VALUE_PROMPT_VERSION: Final = "sokoban-observable-value-v1"


class InvalidProposal(ValueError):
    """The generator returned an invalid or already satisfied intent."""

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
    discount: float | None = None,
) -> _Role:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{name} prompt must be nonempty")
    if scale_arm is not None:
        prompt = generator_instruction(scale_arm, prompt)
    query_fields = {
        "policy": {"observation", "instruction", "attempt_history"},
        "generator": {"observation", "task_goal", "scale"},
        "value": {"observation", "task_goal", "discount"},
    }[name]
    answer_fields = {"policy": {"action"}, "generator": {"effects", "preserve", "instruction"}, "value": {"value"}}[name]
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

    output_tokens = {"policy": config.policy_tokens, "generator": config.generator_tokens, "value": config.value_tokens}[name]
    examples_sha256 = contract_hash(frozen)
    contract: JSONObject = {
        "role": name,
        "domain_version": DOMAIN_VERSION,
        "response_contract_version": RESPONSE_CONTRACT_VERSION,
        "prompt_version": {"policy": POLICY_PROMPT_VERSION, "generator": GENERATOR_PROMPT_VERSION,
                           "value": VALUE_PROMPT_VERSION}[name],
        "parser_version": {"policy": ACTION_PARSER_VERSION, "generator": INTENT_PARSER_VERSION,
                           "value": VALUE_PROMPT_VERSION}[name],
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
    if discount is not None:
        contract["discount"] = discount
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
    text: JSONObject = {"format": {"type": model.response_format if model is not None else "text"}}
    if model is not None and model.response_format == "json_schema":
        if role.name == "value":
            text = {"format": {"type": "json_schema", "name": "sokoban_value_v1", "strict": True, "schema": schema_object({
                "value": {"type": "number", "minimum": 0, "maximum": 1},
            })}}
        else:
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
    """Action, proposal and value generation with independent prompts and shared budgets."""

    api: ExperimentAPI
    _policy: _Role
    _generator: _Role
    _value: _Role
    context_limit: int
    scale_arm: str
    _local_metrics: dict[str, int] = field(default_factory=lambda: {
        "local_requests": 0, "local_policy_calls": 0, "local_generator_calls": 0,
        "local_input_tokens": 0, "local_output_tokens": 0, "fitted_value_calls": 0,
    })

    @property
    def metrics(self) -> dict[str, int | float]:
        """Report local compute separately from logical and physical API requests."""
        return {**self.api.metrics, **self._local_metrics}

    def is_local(self, role: PredicateRole) -> bool:
        """Report whether a role uses local weights."""
        return self._role(role).local is not None

    def _role(self, role: PredicateRole) -> _Role:
        if role == "policy":
            return self._policy
        if role == "generator":
            return self._generator
        if role == "value":
            return self._value
        raise ValueError(f"Unknown predicate component role: {role}")

    def contract(self, role: PredicateRole) -> JSONObject:
        """Return the role identity for experiment records and calibration."""
        return cast(JSONObject, json.loads(_json(self._role(role).contract)))

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

    def _instruction_request(
        self, observation: SokobanObservation, instruction: str,
        memory: Sequence[ActionAttempt], attempt_seed: int | None,
    ) -> JSONObject:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("The policy instruction must be a nonempty string")
        if any(not isinstance(item, ActionAttempt) or not isinstance(item.action, RolloutAction)
               or not isinstance(item.no_op, bool) for item in memory):
            raise ValueError("Memory must contain typed actions with a boolean no-op flag")
        return _request(self._policy, {
            "observation": list(observation.rows),
            "instruction": instruction,
            "attempt_history": [{"action": item.action.value, "no_op": item.no_op} for item in memory],
        }, attempt_seed)

    def _policy_request(
        self, observation: SokobanObservation, intent: Intent,
        memory: Sequence[ActionAttempt], attempt_seed: int | None,
    ) -> JSONObject:
        return self._instruction_request(observation, render_instruction(intent), memory, attempt_seed)

    def policy_payload(
        self, observation: SokobanObservation, intent: Intent, memory: Sequence[ActionAttempt] = (),
        *, attempt_seed: int | None = None,
    ) -> JSONObject:
        """Build π(x, ℓ(h)) from the observation, instruction and attempt memory."""
        return self._checked(self._policy_request(observation, intent, memory, attempt_seed))

    def task_policy_payload(
        self, observation: SokobanObservation, task_goal: str, memory: Sequence[ActionAttempt] = (),
        *, attempt_seed: int | None = None,
    ) -> JSONObject:
        """Use g as the instruction for the same frozen action model and parser."""
        return self._checked(self._instruction_request(observation, task_goal, memory, attempt_seed))

    def generator_payload(
        self, observation: SokobanObservation, task_goal: str, scale: float | None, *, proposal_index: int = 0,
    ) -> JSONObject:
        """Build G(x, g, u) from the current observation, task and requested scale."""
        if not is_token_count(proposal_index):
            raise ValueError("Proposal index must be a nonnegative integer")
        query = generator_query(observation.rows, task_goal, scale, self.scale_arm)
        payload = _request(self._generator, query)
        if proposal_index:
            payload["metadata"]["proposal_index"] = str(proposal_index)
        return self._checked(payload)

    def value_payload(self, observation: SokobanObservation, task_goal: str) -> JSONObject:
        """Build V(x, g) with only the visible observation and configured discount."""
        if not isinstance(task_goal, str) or not task_goal.strip():
            raise ValueError("Task goal must be nonempty")
        return self._checked(_request(self._value, {
            "observation": list(observation.rows), "task_goal": task_goal,
            "discount": self._value.contract["discount"],
        }))

    def _request_budget(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        limits, remote, local = self.api.limits, self.api.metrics, self._local_metrics
        if self.api.preview_only:
            raise RequestFailed("A preview client cannot perform inference")
        if time.perf_counter() - self.api.started >= limits.wall_seconds:
            raise BudgetExceeded("Shared inference wall-time budget exhausted")
        if remote["logical_requests"] + local["local_requests"] + local["fitted_value_calls"] >= limits.max_requests:
            raise BudgetExceeded("Shared logical inference request budget exhausted")
        if input_tokens > limits.max_input_tokens or (
            remote["logical_input_tokens"] + local["local_input_tokens"] + input_tokens > limits.max_total_input_tokens
        ):
            raise BudgetExceeded("Shared inference input token budget exhausted")
        if remote["logical_output_tokens"] + local["local_output_tokens"] + output_tokens > limits.max_output_tokens:
            raise BudgetExceeded("Insufficient remaining shared output tokens for a full request")

    def record_fitted_value_call(self) -> None:
        """Charge a fitted predictor call to the same logical request and time limits."""
        self._request_budget()
        self._local_metrics["fitted_value_calls"] += 1

    def _call(self, payload: JSONObject, role: _Role, mock: MockText | None = None, *, retry: bool = False) -> str:
        counts = self.budget_counts(payload)
        self._request_budget(counts["input_tokens"], role.output_tokens)
        if role.local is not None:
            return self._call_local(payload, role, mock, counts["input_tokens"])

        def mock_response() -> JSONObject:
            assert mock is not None
            return {"output_text": mock(), "model": "mock_predicate_components", "usage": {}}

        call = self.api.call_json_retry if retry else self.api.call
        result = call("responses", payload, mock=mock_response if mock is not None else None, transport=role.model)
        return cast(str, result["output_text"])

    def _call_local(self, payload: JSONObject, role: _Role, mock: MockText | None, input_count: int) -> str:
        if mock is not None:
            raise ValueError(f"Mock text cannot replace the local {role.name} model")
        local = role.local
        assert local is not None
        metrics = self._local_metrics
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
        proposal_index: int = 0,
    ) -> Intent:
        """Parse and check the generator's proposal against the visible starting board."""
        payload = self.generator_payload(observation, task_goal, scale, proposal_index=proposal_index)
        raw = self._call(payload, self._generator, mock)
        try:
            intent = decode_intent(raw)
        except ValueError as exc:
            if not self._generator.contract["json_max_retries"]:
                raise InvalidProposal(str(exc)) from exc
            raw = self._call(payload, self._generator, mock, retry=True)
            try:
                intent = decode_intent(raw)
            except ValueError as retry_error:
                raise InvalidProposal(str(retry_error)) from retry_error
        problems = validate_intent_proposal(intent, observation)
        if problems:
            raise InvalidProposal("Invalid generated intent: " + ", ".join(problems))
        return intent

    def value(self, observation: SokobanObservation, task_goal: str, *, mock: MockText | None = None) -> float:
        """Parse a finite unit-interval continuation value through the stage-02 transport."""
        payload = self.value_payload(observation, task_goal)
        for attempt in range(self._value.contract["json_max_retries"] + 1):
            raw = self._call(payload, self._value, mock, retry=bool(attempt))
            try:
                result = json.loads(raw, object_pairs_hook=lambda pairs: pairs)
            except (TypeError, ValueError):
                continue
            if (isinstance(result, list) and len(result) == 1 and isinstance(result[0], tuple)
                    and result[0][0] == "value"):
                value = result[0][1]
                if type(value) in (float, int) and 0 <= value <= 1:
                    return float(value)
        raise RequestFailed("Value model must return exactly one finite value in [0,1]")

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
        return self._choose(self.components.policy_payload(observation, intent, memory, attempt_seed=self.seed))

    def render_task_action(
        self, observation: SokobanObservation, task_goal: str, memory: Sequence[ActionAttempt],
    ) -> str:
        """Render the same policy prompt with the task goal as the instruction."""
        payload = self.components._instruction_request(observation, task_goal, memory, self.seed)
        return _json(payload["input"])

    def choose_task_action(
        self, observation: SokobanObservation, task_goal: str, memory: Sequence[ActionAttempt],
    ) -> str:
        """Route direct actions through the same model and accounting as intent actions."""
        return self._choose(self.components.task_policy_payload(observation, task_goal, memory, attempt_seed=self.seed))

    def _choose(self, payload: JSONObject) -> str:
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
    local_roles: Mapping[LocalRole, LocalRoleModel] | None = None,
) -> PredicateComponents:
    """Instantiate configured roles with copied, immutable rendered support."""
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
    value = _role(config, "value", config.prompts.value + (
        '\nUse the visible observation, task_goal and discount. ? is an unseen cell. '
        'Respond with exactly one JSON object of the form {"value":0.5}.'
    ), (), support_fingerprint=None, context_limit=context_limit, discount=config.discount)
    return PredicateComponents(api, policy, generator, value, context_limit, scale_arm)
