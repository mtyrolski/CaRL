from __future__ import annotations

import torch

from carl.utils.universal_generator_losses import in_batch_infonce
from carl.utils.universal_generator_losses import masked_mean_pool


def test_masked_mean_pool_shapes_and_values():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[2.0, 0.0], [4.0, 2.0], [6.0, 4.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = masked_mean_pool(hidden, mask)
    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 3.0]))
    assert torch.allclose(pooled[1], torch.tensor([4.0, 2.0]))


def test_in_batch_infonce_is_finite_and_backpropagates():
    anchor = torch.randn(4, 8, requires_grad=True)
    positive = anchor.detach().clone() + 0.01 * torch.randn(4, 8)
    loss = in_batch_infonce(anchor, positive, temperature=0.2)
    assert torch.isfinite(loss)
    loss.backward()
    assert anchor.grad is not None
    assert torch.isfinite(anchor.grad).all()
