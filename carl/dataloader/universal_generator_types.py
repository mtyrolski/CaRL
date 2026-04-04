from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from carl.utils.aliases import State


@dataclass
class UniversalTeacherAnnotation:
    """Offline teacher annotation for universal propositional generator training."""

    current_state: State
    proposition_state: State
    validator_accept: bool
    validator_reject: bool
    reached: bool
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversalGeneratorBatch:
    """Unified batch for raw / finetune / contrastive universal-generator recipes."""

    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    labels_attention_mask: Tensor
    recipe: str

    def to_hf_dict(self) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }

    @classmethod
    def from_xy(
        cls,
        x: Tensor,
        y: Tensor,
        recipe: str,
        pad_token_id: int = 1,
    ) -> "UniversalGeneratorBatch":
        return cls(
            input_ids=x,
            labels=y,
            attention_mask=(x != pad_token_id).long(),
            labels_attention_mask=(y != pad_token_id).long(),
            recipe=recipe,
        )
