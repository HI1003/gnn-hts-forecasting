"""Loss computation for model training."""

from __future__ import annotations

import torch


def compute_loss(
    y_hat: torch.Tensor,
    y_target: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    if y_hat.shape != y_target.shape:
        raise ValueError(f"Prediction shape {tuple(y_hat.shape)} != target shape {tuple(y_target.shape)}")
    return criterion(y_hat, y_target)
