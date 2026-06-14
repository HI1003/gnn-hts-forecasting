"""I/O utilities: save predictions, plot results, save model info."""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.utils.metrics import compute_mase, compute_rmse


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def to_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return [to_serializable(v) for v in obj]
    return obj


def save_predictions(predictions: np.ndarray, true_values: np.ndarray, config: dict):
    """Save wide-form prediction/ground-truth CSVs and per-node error statistics."""
    try:
        if predictions.ndim != 3:
            raise ValueError(f"predictions expected 3-dim [T, N, H], got {predictions.shape}")
        if true_values.ndim != 3:
            raise ValueError(f"true_values expected 3-dim [T, N, H], got {true_values.shape}")

        T, N, H = predictions.shape

        if H == 1:
            if 'time_index' not in config:
                raise KeyError("config['time_index'] is required.")
            time_index_raw = config['time_index']
            if len(time_index_raw) != T:
                raise ValueError(
                    f"time_index length {len(time_index_raw)} != T={T}."
                )
            row_index = pd.DatetimeIndex(pd.to_datetime(time_index_raw), name='Time')
        else:
            if 'time_index_full' not in config:
                raise KeyError("config['time_index_full'] is required for multi-step.")
            time_index_full = config['time_index_full']
            if len(time_index_full) != T:
                raise ValueError(
                    f"time_index_full length {len(time_index_full)} != T={T}."
                )
            origin_time_raw = config.get('time_index')
            if origin_time_raw is None or len(origin_time_raw) != T:
                origin_time_raw = [seq[0] if len(seq) > 0 else pd.NaT for seq in time_index_full]

            origin_times, horizon_ids, target_times = [], [], []
            for sample_idx, seq in enumerate(time_index_full):
                if len(seq) != H:
                    raise ValueError(
                        f"time_index_full[{sample_idx}] length {len(seq)} != H={H}."
                    )
                for h_idx, target_ts in enumerate(seq, start=1):
                    origin_times.append(pd.to_datetime(origin_time_raw[sample_idx]))
                    horizon_ids.append(h_idx)
                    target_times.append(pd.to_datetime(target_ts))

            row_index = pd.MultiIndex.from_arrays(
                [origin_times, horizon_ids, target_times],
                names=['OriginTime', 'Horizon', 'TargetTime'],
            )

        if 'node_names' not in config:
            raise KeyError("config['node_names'] is required.")
        node_names = config['node_names']
        if len(node_names) != N:
            raise ValueError(f"node_names length {len(node_names)} != N={N}.")
        col_names = list(node_names)

        preds_2d = predictions.transpose(0, 2, 1).reshape(T * H, N)
        trues_2d = true_values.transpose(0, 2, 1).reshape(T * H, N)

        df_pred = pd.DataFrame(preds_2d, index=row_index, columns=col_names)
        df_true = pd.DataFrame(trues_2d, index=row_index, columns=col_names)

        timestamp = config['timestamp']
        out_dir = config['output_dir']
        os.makedirs(out_dir, exist_ok=True)

        pred_path = os.path.join(out_dir, f"predictions_wide_{config['model_name']}_{timestamp}.csv")
        true_path = os.path.join(out_dir, f"ground_truth_wide_{config['model_name']}_{timestamp}.csv")

        df_pred.to_csv(pred_path)
        df_true.to_csv(true_path)

        mase_list = []
        rmse_list = []
        history_len = int(config.get("num_timesteps_in", 7))
        for j in range(N):
            series_true = true_values[:, j, :]
            series_pred = predictions[:, j, :]
            rmse_list.append(compute_rmse(series_true, series_pred))
            mase_list.append(compute_mase(series_true, series_pred, num_timesteps_in=history_len))

        node_stats = pd.DataFrame({
            'Node': col_names,
            'RMSE': rmse_list,
            'MASE': mase_list,
        })
        stats_path = os.path.join(out_dir, f"node_stats_{config['model_name']}_{timestamp}.csv")
        node_stats.to_csv(stats_path, index=False)

        logging.info(f"Predictions (wide) saved to {pred_path}")
        logging.info(f"Ground truth (wide) saved to {true_path}")
        logging.info(f"Node statistics saved to {stats_path}")

    except Exception as e:
        logging.error(f"Error saving predictions: {str(e)}")
        raise


def plot_predictions(predictions: np.ndarray, true_values: np.ndarray, node_idx: int, config: dict):
    """Plot predicted vs true values for a given node."""
    plt.figure(figsize=(15, 8))

    if predictions.ndim != 3 or true_values.ndim != 3:
        raise ValueError("plot_predictions expects [T, N, H] arrays.")

    horizon_idx = int(config.get("plot_horizon_index", 0))
    horizon_idx = max(0, min(horizon_idx, predictions.shape[2] - 1))

    T = predictions.shape[0]
    if predictions.shape[2] == 1 and 'time_index' in config and len(config['time_index']) == T:
        x_axis = pd.to_datetime(config['time_index'])
    elif 'time_index_full' in config and len(config['time_index_full']) == T:
        x_axis = pd.to_datetime([seq[horizon_idx] for seq in config['time_index_full']])
    else:
        x_axis = np.arange(T)

    plt.subplot(2, 1, 1)
    plt.plot(x_axis, true_values[:, node_idx, horizon_idx], label='True')
    plt.plot(x_axis, predictions[:, node_idx, horizon_idx], label='Predicted')
    plt.title(f"Node {node_idx}: Predicted vs True (h={horizon_idx + 1})")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    errors = predictions[:, node_idx, horizon_idx] - true_values[:, node_idx, horizon_idx]
    plt.plot(x_axis, errors, label='Error')
    plt.axhline(y=0, linestyle='--')
    plt.title(f"Prediction Error for Node {node_idx}")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    timestamp = config['timestamp']
    plt.savefig(f"{config['output_dir']}/node_{node_idx}_predictions_{config['model_name']}_{timestamp}.png")
    plt.close()


def save_model_info(
    model: torch.nn.Module,
    config: dict,
    metrics: dict,
    level_metrics_df: pd.DataFrame,
    training_results: dict,
):
    """Save model architecture, parameter counts, config, and metrics to JSON."""
    total_params, trainable_params = count_parameters(model)

    config_serialized = {k: to_serializable(v) for k, v in config.items()}
    metrics_serialized = {k: to_serializable(v) for k, v in (metrics or {}).items()}
    training_serialized = {k: to_serializable(v) for k, v in (training_results or {}).items()}

    if isinstance(level_metrics_df, pd.DataFrame):
        level_metrics_serialized = level_metrics_df.to_dict(orient='records')
    else:
        level_metrics_serialized = to_serializable(level_metrics_df)

    info = {
        "model_name": config.get("model_name", ""),
        "timestamp": config.get("timestamp", ""),
        "device": str(next(model.parameters()).device),
        "num_parameters_total": total_params,
        "num_parameters_trainable": trainable_params,
        "config": config_serialized,
        "metrics": metrics_serialized,
        "level_metrics": level_metrics_serialized,
        "training_results": training_serialized,
    }

    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    info_path = os.path.join(out_dir, f"model_info_{config['model_name']}_{config['timestamp']}.json")

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    logging.info(f"Model info saved to {info_path}")
