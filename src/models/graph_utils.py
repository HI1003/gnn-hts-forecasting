"""Graph batching helpers."""

from __future__ import annotations

import torch


def batch_edge_index(edge_index: torch.Tensor, batch_size: int, node_num: int) -> torch.Tensor:
    """Repeat a single-graph edge index for independent batched snapshots."""
    if batch_size == 1:
        return edge_index
    offsets = torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype) * node_num
    expanded = edge_index.unsqueeze(0) + offsets.view(-1, 1, 1)
    return expanded.permute(1, 0, 2).reshape(2, -1)
