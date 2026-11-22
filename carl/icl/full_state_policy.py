"""Local CLLP pilots that condition actions on the current and target boards."""

import json
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import torch
from pydantic import Field, TypeAdapter
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification

from carl.environment.sokoban.env import SokobanEnv
from carl.icl.boards import from_rows, state_hash, to_rows, validate_subgoal
from carl.icl.config import ROOT, Settings
from carl.icl.data_sources.sokoban import environment
from carl.icl.intents.generation.backend import require_peft
from carl.icl.intents.generation.checkpoint import base_snapshot_identity
from carl.icl.intents.generation.data import file_sha256
from carl.icl.luna_intent_rollout import Action
from carl.icl.policy_relative.labels import contract_hash
from carl.icl.types import JSONObject
from carl.inference_components.conditional_low_level_policy import (
    ConditionalLowLevelPolicy, TransformerConditionalLowLevelPolicy,
)
from carl.inference_components.validator import BasicValidator
from carl.utils.aliases import State


class BertPolicySettings(Settings):
    """Load an exact four-action BERT CLLP checkpoint trained by CaRL."""

    family: Literal["bert"]
    checkpoint: str = Field(min_length=1)
    device: Literal["cpu", "cuda"] = "cpu"


class QwenPolicySettings(Settings):
    """Pin the historical full-state coordinate prompt and rank-eight adapter."""

    family: Literal["qwen3"]
    model: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    adapter: str = Field(min_length=1)
    adapter_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "bfloat16"] = "float32"
    context_limit: int = Field(default=512, ge=1)


FullStateSettings = Annotated[BertPolicySettings | QwenPolicySettings, Field(discriminator="family")]


class FullStatePair(Settings):
    """One reachability query with an explicit primitive-action budget."""

    source: list[str]
    target: list[str]
    action_cap: int = Field(ge=1, strict=True)


def load_settings(path: Path) -> FullStateSettings:
    """Read the model-specific local full-state config."""
    return TypeAdapter(FullStateSettings).validate_json(path.read_text(encoding="utf-8"))


def _load_bert(path: str) -> BertForSequenceClassification:
    return cast(BertForSequenceClassification, BertForSequenceClassification.from_pretrained(
        path, local_files_only=True, use_safetensors=True,
    ))


def _coordinate_prompt(env: SokobanEnv, source: State, target: State) -> str:
    a, b = env.get_agent_position(source), env.get_agent_position(target)
    return (
        "Sokoban: # wall, - floor, . goal, $ box, * box on goal, @ player, + player on goal. "
        "Rows increase down and columns increase right. Choose exactly one action: UP, DOWN, LEFT or RIGHT.\n"
        "Current board:\n" + "\n".join(to_rows(source)) + "\nTarget board:\n" + "\n".join(to_rows(target))
        + f"\nPlayer now: row {a[0]}, column {a[1]}. Target player: row {b[0]}, column {b[1]}.\n"
        + "Choose the next action."
    )


class QwenFullStatePolicy(ConditionalLowLevelPolicy):
    """Adapt the pilot's four Qwen direction logits to the existing CLLP interface."""

    def __init__(self, settings: QwenPolicySettings, env: SokobanEnv) -> None:
        base = ROOT / Path(settings.model).expanduser()
        self.adapter = ROOT / Path(settings.adapter).expanduser()
        super().__init__(AutoModelForCausalLM.from_pretrained, str(base), env)
        self.settings = settings
        self.sokoban = env
        self.device = torch.device(settings.device)
        self.model: nn.Module | None = None
        self.tokenizer = None
        self.direction_ids: Tensor | None = None

    @cached_property
    def _artifact_identity(self) -> JSONObject:
        base = Path(self.path_to_conditional_low_level_policy_weights)
        adapter = {
            "config_sha256": file_sha256(self.adapter / "adapter_config.json"),
            "weights_sha256": file_sha256(self.adapter / "adapter_model.safetensors"),
        }
        if adapter != {"config_sha256": self.settings.adapter_config_sha256,
                       "weights_sha256": self.settings.adapter_weights_sha256}:
            raise ValueError("full-state adapter files differ from the configured hashes")
        declaration = json.loads((self.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if declaration["peft_type"] != "LORA" or declaration["task_type"] != "CAUSAL_LM":
            raise ValueError("expected the full-state policy's causal LoRA adapter")
        return {
            "role": "full_state_policy", "family": "qwen3", "base": base_snapshot_identity(base, self.settings.revision),
            "adapter": adapter, "prompt": "sokoban-full-state-coordinate-v1", "decoder": "argmax_four_direction_tokens",
            "context_limit": self.settings.context_limit, "device": str(self.device), "dtype": self.settings.dtype,
        }

    def identity(self) -> JSONObject:
        """Check exact legacy PEFT files and bind them to the full-state protocol."""
        return deepcopy(self._artifact_identity)

    def construct_network(self) -> None:
        """Load the pinned local base, tokenizer and frozen LoRA weights."""
        self.identity()
        peft = require_peft()
        path = self.path_to_conditional_low_level_policy_weights
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        tokens = [tokenizer.encode(action.value, add_special_tokens=False) for action in Action]
        if any(len(token) != 1 for token in tokens) or len({token[0] for token in tokens}) != len(Action):
            raise ValueError("full-state policy requires four distinct single-token actions")
        base = cast(nn.Module, AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=getattr(torch, self.settings.dtype),
        ))
        base.to(self.device)
        self.model = peft.PeftModel.from_pretrained(base, self.adapter, is_trainable=False, local_files_only=True).eval()
        self.tokenizer = tokenizer
        self.direction_ids = torch.tensor([token[0] for token in tokens], device=self.device)

    def get_network(self) -> nn.Module:
        """Return the instantiated full-state model for the CaRL validator."""
        if self.model is None:
            raise ValueError("full-state policy network has not been loaded")
        return self.model

    def get_action(self, state: State, state_after_k: State) -> Tensor:
        """Score each legal direction using the complete current and target states."""
        assert self.tokenizer is not None and self.direction_ids is not None
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": _coordinate_prompt(self.sokoban, state, state_after_k)}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
        ).to(self.device)
        if ids.shape[1] > self.settings.context_limit:
            raise ValueError("full-state prompt exceeds the configured context limit")
        with torch.inference_mode():
            logits = self.get_network()(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            return logits[0, -1].index_select(-1, self.direction_ids).float()


def load_policy(settings: FullStateSettings, env: SokobanEnv) -> tuple[ConditionalLowLevelPolicy, JSONObject]:
    """Resolve the exact local CLLP artifact without loading its network."""
    if settings.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable for the requested full-state policy")
    policy: ConditionalLowLevelPolicy
    if isinstance(settings, QwenPolicySettings):
        qwen = QwenFullStatePolicy(settings, env)
        identity = qwen.identity()
        policy = qwen
    else:
        checkpoint = ROOT / Path(settings.checkpoint).expanduser()
        if not checkpoint.name.startswith("checkpoint-"):
            raise ValueError("select an exact BERT checkpoint-N directory")
        identity = {
            "role": "full_state_policy", "family": "bert", "device": settings.device,
            "decoder": "argmax_cllp_logits", "prompt": "sokoban-cllp-board-pair-tokens-v1",
            "config_sha256": file_sha256(checkpoint / "config.json"),
            "weights_sha256": file_sha256(checkpoint / "model.safetensors"),
        }
        policy = TransformerConditionalLowLevelPolicy(_load_bert, str(checkpoint), env)
        policy.device = torch.device(settings.device)
    return policy, identity


def evaluate(settings: FullStateSettings, pair: FullStatePair) -> JSONObject:
    """Evaluate an exact local checkpoint with CaRL's reachability validator."""
    state, target = from_rows(pair.source), from_rows(pair.target)
    validate_subgoal(state, target)
    env = environment(state)
    policy, identity = load_policy(settings, env)
    information = {"policy": identity, "policy_fingerprint": contract_hash(identity), "source_sha256": state_hash(state),
                   "target_sha256": state_hash(target), "action_cap": pair.action_cap}
    if np.array_equal(state, target):
        return {**information, "achieved": True, "actions": []}
    validator = BasicValidator(env, policy, budget_for_achieving_subgoal=pair.action_cap)
    validator.construct_network()
    result = validator.is_valid(state, target, steps_limit=pair.action_cap)
    return {**information, "achieved": result.is_valid, "actions": result.path}
