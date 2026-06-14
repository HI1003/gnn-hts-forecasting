"""Bottom-up reconciliation strategies used by both TGLP and TALP."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _aggregate_from_bottom(sum_matrix: torch.Tensor, bottom_values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nb,sbh->snh", sum_matrix, bottom_values)


def _map_all_nodes_to_bottom(
    mapping: nn.Module,
    base_predictions: torch.Tensor,
    node_num: int,
    num_bottom_nodes: int,
) -> torch.Tensor:
    batch_size, _, horizon = base_predictions.shape
    flat_pred = base_predictions.permute(0, 2, 1).reshape(batch_size * horizon, node_num)
    mapped_bottom = mapping(flat_pred).reshape(batch_size, horizon, num_bottom_nodes)
    return mapped_bottom.permute(0, 2, 1)


def _build_bottom_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 128,
    layers: int = 2,
    dropout: float = 0.1,
    decay: float = 0.5,
) -> nn.Sequential:
    if layers < 1:
        raise ValueError(f"layers must be >= 1, got {layers}")
    if hidden_dim < 1:
        raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    if not (0.0 < decay <= 1.0):
        raise ValueError(f"decay must be in (0, 1], got {decay}")

    widths = []
    width = float(hidden_dim)
    for _ in range(layers):
        widths.append(max(8, int(round(width))))
        width = max(8.0, width * decay)

    modules: list[nn.Module] = []
    in_dim = input_dim
    for width in widths:
        modules.extend([nn.Linear(in_dim, width), nn.LayerNorm(width), nn.ReLU(), nn.Dropout(dropout)])
        in_dim = width
    modules.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*modules)


class BaseBottomUpReconciliation(nn.Module):
    """Common hierarchy bookkeeping for bottom-up reconciliation."""

    def __init__(self, node_num: int, sum_matrix: np.ndarray):
        super().__init__()
        self.node_num = int(node_num)
        sum_tensor = torch.as_tensor(sum_matrix, dtype=torch.float32)
        if sum_tensor.ndim != 2:
            raise ValueError(f"sum_matrix must be 2-D, got shape {tuple(sum_tensor.shape)}")
        if sum_tensor.shape[0] != self.node_num:
            raise ValueError(f"sum_matrix rows {sum_tensor.shape[0]} != node_num {self.node_num}")

        self.register_buffer("sum_matrix", sum_tensor)
        self.num_bottom_nodes = int(sum_tensor.shape[1])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes
        if self.num_bottom_nodes <= 0 or self.bottom_start_idx < 0:
            raise ValueError(
                f"Invalid bottom-up dimensions: node_num={self.node_num}, "
                f"bottom_nodes={self.num_bottom_nodes}"
            )

    def _base_bottom(self, base_predictions: torch.Tensor) -> torch.Tensor:
        return base_predictions[:, self.bottom_start_idx :, :]

    def _to_original_scale(self, backbone: nn.Module, bottom_predictions: torch.Tensor) -> torch.Tensor:
        return backbone.inverse_log(backbone.denormalize(bottom_predictions))

    def _aggregate(self, bottom_values: torch.Tensor) -> torch.Tensor:
        return _aggregate_from_bottom(self.sum_matrix, bottom_values)


class BUReconciliation(BaseBottomUpReconciliation):
    """BU: directly aggregate bottom-level base forecasts."""

    def forward(self, base_predictions: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
        bottom_predictions = self._base_bottom(base_predictions)
        bottom_values = self._to_original_scale(backbone, bottom_predictions)
        return self._aggregate(bottom_values)


class BULReconciliation(BaseBottomUpReconciliation):
    """BUL: linearly map all base forecasts to bottom-level forecasts."""

    def __init__(self, node_num: int, sum_matrix: np.ndarray):
        super().__init__(node_num=node_num, sum_matrix=sum_matrix)
        self.bottom_mapping = nn.Linear(self.node_num, self.num_bottom_nodes)
        self.residual_ratio = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, base_predictions: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
        base_bottom = self._base_bottom(base_predictions)
        mapped_bottom = _map_all_nodes_to_bottom(
            self.bottom_mapping,
            base_predictions,
            self.node_num,
            self.num_bottom_nodes,
        )
        ratio = torch.sigmoid(self.residual_ratio)
        bottom_predictions = ratio * base_bottom + (1.0 - ratio) * mapped_bottom
        bottom_values = self._to_original_scale(backbone, bottom_predictions)
        return self._aggregate(bottom_values)


class BUNReconciliation(BaseBottomUpReconciliation):
    """BUN: nonlinearly map all base forecasts to bottom-level forecasts."""

    def __init__(
        self,
        node_num: int,
        sum_matrix: np.ndarray,
        mlp_hidden_dim: int = 128,
        mlp_layers: int = 2,
        mlp_dropout: float = 0.1,
        mlp_decay: float = 0.5,
    ):
        super().__init__(node_num=node_num, sum_matrix=sum_matrix)
        self.bottom_mapping = _build_bottom_mlp(
            input_dim=self.node_num,
            output_dim=self.num_bottom_nodes,
            hidden_dim=mlp_hidden_dim,
            layers=mlp_layers,
            dropout=mlp_dropout,
            decay=mlp_decay,
        )
        self.residual_ratio = nn.Parameter(torch.tensor(0.6), requires_grad=True)

    def forward(self, base_predictions: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
        base_bottom = self._base_bottom(base_predictions)
        mapped_bottom = _map_all_nodes_to_bottom(
            self.bottom_mapping,
            base_predictions,
            self.node_num,
            self.num_bottom_nodes,
        )
        sparsity_score = (base_bottom.abs() < 1e-3).float().mean(dim=(1, 2), keepdim=True)
        ratio = torch.sigmoid(self.residual_ratio)
        adaptive_ratio = ratio + (1.0 - ratio) * sparsity_score
        adaptive_ratio = torch.clamp(adaptive_ratio, min=0.05, max=0.95)
        bottom_predictions = adaptive_ratio * base_bottom + (1.0 - adaptive_ratio) * mapped_bottom

        bottom_log = torch.clamp(backbone.denormalize(bottom_predictions), min=0.0, max=20.0)
        bottom_values = backbone.inverse_log(bottom_log)
        return self._aggregate(bottom_values)


class ReconciledForecastModel(nn.Module):
    """Compose a TGLP/TALP backbone with one reconciliation strategy."""

    def __init__(self, backbone: nn.Module, reconciliation: BaseBottomUpReconciliation):
        super().__init__()
        self.backbone = backbone
        self.reconciliation = reconciliation

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.denormalize(x)

    def inverse_log(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.inverse_log(x)

    def transform_target(self, y: torch.Tensor) -> torch.Tensor:
        return self.backbone.transform_target(y)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        base_predictions = self.backbone(x, edge_index)
        return self.reconciliation(base_predictions, self.backbone)
