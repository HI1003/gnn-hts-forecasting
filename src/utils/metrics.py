"""Metrics reported by the paper: RMSE and MASE."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


def compute_rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mase(
    y_true,
    y_pred,
    num_timesteps_in: int = 7,
    epsilon: float = 1e-8,
    history_len: int | None = None,
) -> float:
    if history_len is not None:
        num_timesteps_in = int(history_len)

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    if y_true.ndim == 3 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
        y_pred = y_pred[..., 0]

    total_steps = y_true.shape[0]
    if total_steps <= num_timesteps_in:
        return float("nan")

    y_true_flat = y_true.reshape(total_steps, -1)
    y_pred_flat = y_pred.reshape(total_steps, -1)
    model_errors = np.abs(y_true_flat[num_timesteps_in:, :] - y_pred_flat[num_timesteps_in:, :])
    naive_errors = np.abs(y_true_flat[num_timesteps_in:, :] - y_true_flat[num_timesteps_in - 1 : -1, :])
    return float(model_errors.mean() / (naive_errors.mean() + epsilon))


def _resolve_level_slices(config: dict, node_count: int):
    sizes = config.get("hierarchy_level_sizes")
    if sizes:
        sizes = [int(v) for v in sizes]
        if sum(sizes) != node_count:
            raise ValueError(f"hierarchy_level_sizes sum {sum(sizes)} != node_count {node_count}")
        level_names = config.get("hierarchy_level_names", ["Total", "Middle", "Bottom"])
        slices = []
        start = 0
        for name, size in zip(level_names, sizes):
            slices.append((name, list(range(start, start + size))))
            start += size
        return slices

    bottom_start = int(config.get("bottom_start_idx", 1))
    num_mid = int(config.get("num_mid_nodes", max(bottom_start - 1, 0)))
    return [
        ("Total", [0]),
        ("Middle", list(range(1, 1 + num_mid))),
        ("Bottom", list(range(1 + num_mid, node_count))),
    ]


def calculate_level_metrics(predictions: np.ndarray, true_values: np.ndarray, config: dict) -> pd.DataFrame:
    """Save and return RMSE/MASE by hierarchy level."""
    rows: list[dict[str, float | str]] = []
    history_len = int(config.get("num_timesteps_in", 7))

    rows.append(
        {
            "Level": "All",
            "RMSE": compute_rmse(true_values, predictions),
            "MASE": compute_mase(true_values, predictions, num_timesteps_in=history_len),
        }
    )

    for level_name, level_indices in _resolve_level_slices(config, predictions.shape[1]):
        if not level_indices:
            continue
        level_pred = predictions[:, level_indices, :]
        level_true = true_values[:, level_indices, :]
        rmse = compute_rmse(level_true, level_pred)
        mase = compute_mase(level_true, level_pred, num_timesteps_in=history_len)
        rows.append({"Level": level_name, "RMSE": rmse, "MASE": mase})
        logging.info("%s level - RMSE: %.4f, MASE: %.4f", level_name, rmse, mase)

    level_metrics_df = pd.DataFrame(rows, columns=["Level", "RMSE", "MASE"])
    timestamp = config["timestamp"]
    metrics_path = f"{config['output_dir']}/level_metrics_{config['model_name']}_{timestamp}.csv"
    level_metrics_df.to_csv(metrics_path, index=False)
    logging.info("Level metrics saved to %s", metrics_path)
    return level_metrics_df
