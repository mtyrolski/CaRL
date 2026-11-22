"""Load pinned local model roles without importing PEFT until inference needs it."""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

from carl.icl.api import BudgetExceeded
from carl.icl.boards import canonical_layout_id, from_rows
from carl.icl.intent_domain import SokobanObservation
from carl.icl.intents.generation.backend import require_peft
from carl.icl.intents.generation.checkpoint import (
    ADAPTER_MANIFEST, base_snapshot_identity, verify_adapter_artifact, verify_validation_selection,
)
from carl.icl.intents.generation.data import file_sha256
from carl.icl.responses import is_token_count
from carl.icl.types import JSONObject

if TYPE_CHECKING:
    from carl.icl.config import LocalModelSettings

LocalRole: TypeAlias = Literal["policy", "generator"]


def _local_chat(messages: Sequence[JSONObject]) -> list[dict[str, str]]:
    return [
        {"role": "system" if message["role"] == "developer" else message["role"],
         "content": "".join(part["text"] for part in message["content"])}
        for message in messages
    ]


def _identity(settings: "LocalModelSettings", role: LocalRole, base: Path, adapter: Path | None) -> JSONObject:
    manifest = None if adapter is None else verify_adapter_artifact(adapter, base=base, revision=settings.revision, role=role)
    adapter_identity = None
    if adapter is not None and manifest is not None:
        adapter_identity = {
            "manifest_sha256": file_sha256(adapter / ADAPTER_MANIFEST),
            "weights_sha256": manifest["files"]["adapter_model.safetensors"],
            "chat_template_sha256": manifest["chat_template_sha256"],
            "prompt_sha256": manifest["prompt"]["prompt_sha256"],
            "checkpoint": manifest["checkpoint"],
        }
    return {
        "backend": "local_transformers", "role": role, "revision": settings.revision,
        "base": base_snapshot_identity(base, settings.revision) if manifest is None else manifest["base"],
        "adapter": adapter_identity, "device": settings.device, "dtype": settings.dtype,
        "decoding": {"sampling": "greedy", "enable_thinking": False},
    }


@dataclass(frozen=True)
class LocalRoleModel:
    """A verified weights and tokenizer binding for a single predicate role."""

    settings: "LocalModelSettings"
    role: LocalRole
    base: Path
    adapter: Path | None
    _frozen_identity: str

    def identity(self) -> JSONObject:
        """Describe the artifact and decoding without initializing a model."""
        return cast(JSONObject, json.loads(self._frozen_identity))

    def assert_prompt(self, prompt_sha256: str) -> None:
        """Require the checkpoint's training prompt before using an adapter."""
        adapter = self.identity()["adapter"]
        if adapter is not None and adapter["prompt_sha256"] != prompt_sha256:
            raise ValueError(f"Local {self.role} adapter prompt does not match the configured role")

    def assert_query(
        self, observation: SokobanObservation, *, split: Literal["validation", "test"], task_group: str | None,
    ) -> None:
        """Check local fitting splits before evaluating a visible board with an adapter."""
        if split not in ("validation", "test"):
            raise ValueError("Local adapter query split must be validation or test")
        if self.adapter is None:
            return
        if observation.partial or not isinstance(task_group, str) or not task_group.strip():
            raise ValueError("Local adapter evaluation requires a full visible board and a declared task group")
        contract = self.identity()["adapter"]
        manifest_bytes = (self.adapter / ADAPTER_MANIFEST).read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != contract["manifest_sha256"]:
            raise ValueError("Local adapter manifest changed after binding")
        manifest = json.loads(manifest_bytes)
        training = manifest["training"]
        if split == "test":
            verify_validation_selection(self.adapter, manifest)
        layout = canonical_layout_id(from_rows(list(observation.rows)))
        for pool in ("train", "validation") if split == "test" else ("train",):
            layouts = training.get(f"{pool}_layout_ids")
            groups = training.get(f"{pool}_task_groups")
            if not isinstance(layouts, list) or not isinstance(groups, list):
                raise ValueError(f"Local {self.role} adapter is missing its {pool} split identities")
            if layout in layouts or task_group in groups:
                raise ValueError(f"Query overlaps the local {self.role} adapter {pool} layout or task group")

    def _check_frozen(self) -> None:
        if _identity(self.settings, self.role, self.base, self.adapter) != self.identity():
            raise ValueError("Local model or adapter changed after binding")

    @cached_property
    def _tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        source = self.base if self.adapter is None else self.adapter / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            raise ValueError("Local tokenizer must provide a chat template")
        if self.adapter is not None and hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != (
            self.identity()["adapter"]["chat_template_sha256"]
        ):
            raise ValueError("Local adapter tokenizer chat template differs from training")
        return tokenizer

    @cached_property
    def _stack(self) -> tuple[Any, Any]:
        self._check_frozen()
        peft = require_peft() if self.adapter is not None else None
        import torch
        from transformers import AutoModelForCausalLM

        device = self.settings.device
        dtype = getattr(torch, self.settings.dtype)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for the configured local model")
        if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The configured CUDA device does not support bfloat16")
        if device == "cpu" and dtype == torch.float16:
            raise ValueError("float16 CPU execution is unavailable for the configured local model")
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.base, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=dtype,
        )
        model = model.to(device)
        if peft is not None:
            model = peft.PeftModel.from_pretrained(model, self.adapter, is_trainable=False, local_files_only=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return torch, model

    @cached_property
    def _base_context_limit(self) -> int | None:
        config = json.loads((self.base / "config.json").read_text(encoding="utf-8"))
        value = config.get("max_position_embeddings", config.get("n_positions"))
        return value if is_token_count(value) and value else None

    def context_limit(self, requested: int) -> int:
        """Combine the experiment and configured model position limits."""
        return min(requested, self._base_context_limit or requested)

    def _encode(self, messages: Sequence[dict[str, str]]) -> list[int]:
        ids = self._tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        return cast(list[int], ids)

    def input_tokens(self, messages: Sequence[JSONObject]) -> int:
        """Count the full role prompt with the pinned tokenizer and template."""
        return len(self._encode(_local_chat(messages)))

    def complete_chat(
        self, messages: Sequence[dict[str, str]], *, output_tokens: int, context_limit: int,
        on_payload: Callable[[JSONObject], None] | None = None,
    ) -> tuple[str, int, int]:
        """Generate locally and return the completion and exact input/output token counts."""
        if not is_token_count(output_tokens) or not output_tokens or not is_token_count(context_limit) or not context_limit:
            raise ValueError("Local input and output token caps must be positive integers")
        tokens = self._encode(messages)
        if len(tokens) + output_tokens > self.context_limit(context_limit):
            raise BudgetExceeded(f"Local {self.role} context window exceeded")
        cold_load = "_stack" not in self.__dict__
        load_started = time.perf_counter()
        torch, model = self._stack
        load_seconds = time.perf_counter() - load_started if cold_load else 0.0
        ids = torch.tensor([tokens], device=self.settings.device)
        tokenizer = self._tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if on_payload is not None:
            rendered = tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            actual_ids = ids[0].tolist()
            if tokenizer(rendered, add_special_tokens=False)["input_ids"] != actual_ids:
                raise ValueError("Rendered local chat does not match the actual input tensor")
            with_thinking = tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            parameter = next(model.parameters())
            on_payload({
                "rendered": rendered, "input_ids": actual_ids, "enable_thinking": False,
                "template_changes_with_thinking": rendered != with_thinking,
                "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "dtype": str(parameter.dtype), "device": str(parameter.device),
                "cold_load": cold_load, "load_seconds": load_seconds,
                "max_new_tokens": output_tokens, "do_sample": False, "use_cache": True,
            })
        with torch.inference_mode():
            output = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=output_tokens,
                do_sample=False, pad_token_id=pad_id, use_cache=True,
            )[0, ids.shape[1]:]
        return tokenizer.decode(output, skip_special_tokens=True).strip(), len(tokens), len(output)

    def complete(
        self, messages: Sequence[JSONObject], *, output_tokens: int, context_limit: int,
    ) -> tuple[str, int, int]:
        """Reuse the predicate wire messages and route them through the native chat template."""
        return self.complete_chat(_local_chat(messages), output_tokens=output_tokens, context_limit=context_limit)


def bind_local_role(
    settings: "LocalModelSettings", role: LocalRole, *, prompt_sha256: str | None = None,
) -> LocalRoleModel:
    """Pin revision, role, content hashes and decoding before any inference."""
    if role not in ("policy", "generator"):
        raise ValueError(f"Unsupported local model role: {role}")
    base = Path(settings.model).expanduser().resolve()
    adapter = Path(settings.adapter).expanduser().resolve() if settings.adapter is not None else None
    identity = json.dumps(_identity(settings, role, base, adapter), sort_keys=True, allow_nan=False)
    model = LocalRoleModel(settings, role, base, adapter, identity)
    if prompt_sha256 is not None:
        model.assert_prompt(prompt_sha256)
    return model
