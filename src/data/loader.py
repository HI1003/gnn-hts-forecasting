from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

from src.data.paths import (
    adjacency_path,
    dataset_csv_path,
    dataset_name,
    hierarchy_level_names,
    hierarchy_level_sizes,
    load_dataset_config,
    node_values_path,
    normalization_params_path,
    resolve_project_root,
    sum_matrix_path,
    total_nodes,
)


class LoadDatasetLoader(object):
    """Dataset loader backed by configs/ and datasets/."""

    def __init__(
        self,
        dataset_source: str | Path | dict[str, Any],
        use_promotion: bool = False,
        input_dim: int = 1,
        adj_file: str = "adj_hierarchy.npy",
        dataset_name_override: str | None = None,
        config_source: str | Path | dict[str, Any] | None = None,
    ):
        super(LoadDatasetLoader, self).__init__()

        self.project_root = self._resolve_project_root(dataset_source)
        resolved_config_source = self._resolve_config_source(
            dataset_source=dataset_source,
            dataset_name_override=dataset_name_override,
            config_source=config_source,
        )
        self.dataset_config = load_dataset_config(
            resolved_config_source,
            project_root=self.project_root,
        )
        self.dataset_name = dataset_name(self.dataset_config)
        self.use_promotion = use_promotion
        self.expected_input_dim = input_dim
        self.adj_file = Path(adj_file).name

        self.level_names = hierarchy_level_names(self.dataset_config)
        self.hierarchy_level_sizes = hierarchy_level_sizes(self.dataset_config)
        self.config_total_nodes = total_nodes(self.dataset_config)

        self.normalization_params_file = normalization_params_path(
            self.dataset_config,
            project_root=self.project_root,
        )
        self.sum_matrix_file = sum_matrix_path(
            self.dataset_config,
            project_root=self.project_root,
        )
        self.adjacency_file = adjacency_path(
            self.dataset_config,
            self.adj_file,
            project_root=self.project_root,
        )
        self.value_file = node_values_path(
            self.dataset_config,
            self.use_promotion,
            project_root=self.project_root,
        )
        self.active_csv_path = dataset_csv_path(
            self.dataset_config,
            self.use_promotion,
            project_root=self.project_root,
        )
        self.reference_csv_path = dataset_csv_path(
            self.dataset_config,
            False,
            project_root=self.project_root,
        )

        if not self.normalization_params_file.exists():
            raise FileNotFoundError(f"Cannot find normalization parameters: {self.normalization_params_file}")
        norm_params = np.load(self.normalization_params_file, allow_pickle=True).item()
        if "global_min" not in norm_params or "global_max" not in norm_params:
            raise ValueError(f"Invalid normalization parameters: {self.normalization_params_file}")
        self.global_min = float(norm_params["global_min"])
        self.global_max = float(norm_params["global_max"])
        if self.global_max <= self.global_min:
            raise ValueError(
                f"Invalid normalization range: global_min={self.global_min}, "
                f"global_max={self.global_max}."
            )

        self.A = None
        self.X = None
        self.edges = None
        self.edge_weights = None
        self.features = None
        self.targets = None
        self.time_index = None
        self.node_names = None
        self.target_time_index = None
        self.target_time_index_full = None

        self._read_dataset()

        self.sum_matrix = pd.read_csv(
            self.sum_matrix_file,
            header=None,
        ).to_numpy().astype(np.float32)

        self.num_total_nodes = int(self.X.shape[0])
        if self.sum_matrix.ndim != 2 or self.sum_matrix.shape[0] != self.num_total_nodes:
            raise ValueError(
                f"Summing matrix shape {self.sum_matrix.shape} is incompatible with "
                f"{self.num_total_nodes} loaded nodes."
            )
        self.num_bottom_nodes = int(self.sum_matrix.shape[1])
        self.bottom_start_idx = self.num_total_nodes - self.num_bottom_nodes
        self.num_mid_nodes = self.bottom_start_idx - 1
        if self.num_bottom_nodes <= 0 or self.bottom_start_idx <= 0:
            raise ValueError(
                f"Invalid hierarchy dimensions: total_nodes={self.num_total_nodes}, "
                f"bottom_nodes={self.num_bottom_nodes}."
            )

        if self.config_total_nodes and self.config_total_nodes != self.num_total_nodes:
            raise ValueError(
                f"Config total_nodes={self.config_total_nodes} "
                f"but loaded data has {self.num_total_nodes} nodes."
            )
        if self.hierarchy_level_sizes:
            if sum(self.hierarchy_level_sizes) != self.num_total_nodes:
                raise ValueError(
                    f"hierarchy level sizes {self.hierarchy_level_sizes} "
                    f"do not sum to {self.num_total_nodes}."
                )
        else:
            self.hierarchy_level_sizes = [1, self.num_mid_nodes, self.num_bottom_nodes]

    @staticmethod
    def _resolve_project_root(dataset_source: str | Path | dict[str, Any]) -> Path:
        if isinstance(dataset_source, dict):
            return resolve_project_root()
        if isinstance(dataset_source, (str, Path)):
            return resolve_project_root(dataset_source)
        return resolve_project_root()

    @staticmethod
    def _resolve_config_source(
        dataset_source: str | Path | dict[str, Any],
        dataset_name_override: str | None,
        config_source: str | Path | dict[str, Any] | None,
    ) -> str | Path | dict[str, Any]:
        if config_source is not None:
            return config_source
        if isinstance(dataset_source, dict):
            return dataset_source
        if isinstance(dataset_source, (str, Path)):
            source_path = Path(dataset_source)
            if source_path.suffix.lower() in {".yaml", ".yml"}:
                return source_path
        if dataset_name_override:
            return dataset_name_override
        return "italian"

    def _read_dataset(self):
        for label, path in {
            "normalization parameters": self.normalization_params_file,
            "node values": self.value_file,
            "adjacency matrix": self.adjacency_file,
            "sum matrix": self.sum_matrix_file,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"Cannot find {label}: {path}")

        adjacency = np.load(self.adjacency_file).astype(np.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {adjacency.shape}.")
        self.A = torch.from_numpy(adjacency)

        self.X = torch.from_numpy(
            np.load(self.value_file).astype(np.float32).transpose((1, 2, 0))
        )
        num_nodes, feature_dim, total_steps = self.X.shape
        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Adjacency shape {adjacency.shape} is incompatible with {num_nodes} loaded nodes."
            )
        if feature_dim != self.expected_input_dim:
            raise ValueError(
                f"[Feature dim mismatch] Data feature dim F={feature_dim} "
                f"but expected input_dim={self.expected_input_dim}."
            )

        if not self.active_csv_path.exists():
            raise FileNotFoundError(f"Cannot find original CSV file: {self.active_csv_path}")

        active_df = pd.read_csv(self.active_csv_path)
        time_col = active_df.columns[0]
        time_index_raw = active_df[time_col].values
        if len(time_index_raw) != total_steps:
            raise ValueError(
                f"CSV time length {len(time_index_raw)} != X time length T={total_steps}."
            )
        self.time_index = pd.to_datetime(time_index_raw)

        self.node_names = self._load_node_names(num_nodes)

        logging.info(
            "Loaded %s dataset | A shape=%s | X shape=%s | time steps=%s | nodes=%s",
            self.dataset_name,
            tuple(self.A.shape),
            tuple(self.X.shape),
            len(self.time_index),
            len(self.node_names),
        )

    def _load_node_names(self, expected_nodes: int) -> list[str]:
        if self.reference_csv_path.exists():
            ref_columns = list(pd.read_csv(self.reference_csv_path, nrows=0).columns[1:])
            if len(ref_columns) == expected_nodes:
                return ref_columns

        active_columns = list(pd.read_csv(self.active_csv_path, nrows=0).columns[1:])
        if len(active_columns) == expected_nodes:
            return active_columns

        filtered = [name for name in active_columns if not self._looks_like_promo(name)]
        if len(filtered) >= expected_nodes:
            return filtered[:expected_nodes]
        raise ValueError(
            f"Cannot infer {expected_nodes} node names from {self.active_csv_path}. "
            f"Columns found: {len(active_columns)}"
        )

    @staticmethod
    def _looks_like_promo(col: str) -> bool:
        value = str(col).lower()
        promo_tokens = [
            "promo",
            "promotion",
            "discount",
            "deal",
            "event",
            "is_promo",
            "onpromo",
            "on_promo",
            "snap",
        ]
        return any(token in value for token in promo_tokens)

    def denormalize_data(self, normalized_data: np.ndarray) -> np.ndarray:
        return normalized_data * (self.global_max - self.global_min) + self.global_min

    def inverse_log_transform(self, log_data: np.ndarray) -> np.ndarray:
        return np.exp(log_data) - 1.0

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        self.edges = edge_indices.numpy()
        self.edge_weights = values.numpy()

    def _generate_task(
        self,
        num_timesteps_in: int = 7,
        num_timesteps_out: int = 1,
    ):
        total_timesteps = self.X.shape[2]
        if num_timesteps_in < 1:
            raise ValueError(f"num_timesteps_in must be >= 1, got {num_timesteps_in}")
        if num_timesteps_out < 1:
            raise ValueError(f"num_timesteps_out must be >= 1, got {num_timesteps_out}")
        if total_timesteps < num_timesteps_in + num_timesteps_out:
            raise ValueError(
                f"Not enough time steps: T={total_timesteps}, "
                f"num_timesteps_in={num_timesteps_in}, num_timesteps_out={num_timesteps_out}."
            )
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(total_timesteps - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        if self.time_index is None:
            raise RuntimeError("time_index is None. Check CSV loading.")

        self.target_time_index = [self.time_index[i + num_timesteps_in] for (i, _) in indices]
        self.target_time_index_full = [
            list(self.time_index[i + num_timesteps_in:j]) for i, j in indices
        ]
        self.features = [self.X[:, :, i:i + num_timesteps_in].numpy() for i, _ in indices]
        self.targets = [self.X[:, 0, i + num_timesteps_in:j].numpy() for i, j in indices]

    def get_dataset(
        self,
        num_timesteps_in: int = 7,
        num_timesteps_out: int = 1,
    ) -> StaticGraphTemporalSignal:
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        return StaticGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets)
