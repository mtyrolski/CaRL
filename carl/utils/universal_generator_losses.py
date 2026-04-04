from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean pool sequence hidden states using an attention mask."""
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden_states shape (B,L,D), got {tuple(hidden_states.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected attention_mask shape (B,L), got {tuple(attention_mask.shape)}")
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


def in_batch_infonce(anchor_embeddings: Tensor, positive_embeddings: Tensor, temperature: float = 0.1) -> Tensor:
    """InfoNCE with in-batch negatives using cosine-normalized embeddings."""
    if anchor_embeddings.ndim != 2 or positive_embeddings.ndim != 2:
        raise ValueError("Embeddings must have shape (B, D)")
    if anchor_embeddings.shape != positive_embeddings.shape:
        raise ValueError("Anchor and positive embeddings must have identical shape")
    if anchor_embeddings.shape[0] == 0:
        raise ValueError("Batch size must be > 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    anchor = F.normalize(anchor_embeddings, p=2, dim=-1)
    positive = F.normalize(positive_embeddings, p=2, dim=-1)
    logits = anchor @ positive.T
    logits = logits / temperature
    targets = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, targets)


def verifier_consistency_placeholder(device: torch.device | None = None) -> Tensor:
    """Optional loss placeholder until verifier labels are available in the batch."""
    return torch.zeros((), device=device)
