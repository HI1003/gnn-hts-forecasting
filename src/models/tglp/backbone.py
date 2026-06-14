"""TGLP backbone: GCNConv + GRU + projection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from src.models.graph_utils import batch_edge_index


class TGLP(nn.Module):
    """Generate normalized log-space base forecasts using GCNConv and GRU."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        gnn_layers: int = 1,
        gru_layers: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.node_num = int(node_num)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.gnn_layers = int(gnn_layers)
        self.gru_layers = int(gru_layers if gru_layers is not None else num_layers)
        self.num_layers = self.gru_layers
        self.global_min = float(global_min)
        self.global_max = float(global_max)

        if self.node_num < 1:
            raise ValueError(f"node_num must be positive, got {self.node_num}")
        if self.gnn_layers < 1:
            raise ValueError(f"gnn_layers must be >= 1, got {self.gnn_layers}")
        if self.gru_layers < 1:
            raise ValueError(f"gru_layers must be >= 1, got {self.gru_layers}")

        self.gcn_layers = nn.ModuleList()
        in_channels = self.input_dim
        for _ in range(self.gnn_layers):
            self.gcn_layers.append(GCNConv(in_channels=in_channels, out_channels=self.hidden_dim))
            in_channels = self.hidden_dim

        self.spatial_norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.gnn_layers)])
        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.gru_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.hidden_dim, self.output_dim)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.global_max - self.global_min) + self.global_min

    def inverse_log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - 1.0

    def transform_target(self, y: torch.Tensor) -> torch.Tensor:
        return self.inverse_log(self.denormalize(y))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, node_num, feature_dim, timesteps = x.shape
        if node_num != self.node_num:
            raise ValueError(f"Expected {self.node_num} nodes, got {node_num}")
        if feature_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {feature_dim}")

        batched_edge_index = batch_edge_index(edge_index, batch_size, self.node_num)
        h0 = torch.zeros(self.gru_layers, batch_size * node_num, self.hidden_dim, device=x.device)

        temporal_features = []
        for t in range(timesteps):
            step_out = x[:, :, :, t].reshape(batch_size * node_num, feature_dim)
            for layer_idx, gcn in enumerate(self.gcn_layers):
                step_out = gcn(step_out, batched_edge_index)
                step_out = self.spatial_norms[layer_idx](step_out)
                step_out = self.dropout(step_out)
            temporal_features.append(step_out)

        temporal_features = torch.stack(temporal_features, dim=1)
        gru_out, _ = self.gru(temporal_features, h0)
        final_hidden = gru_out[:, -1, :].view(batch_size, node_num, self.hidden_dim)
        return self.projection(final_hidden)
