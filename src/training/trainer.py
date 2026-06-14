"""Training and evaluation loops for hierarchical forecasting models."""

from __future__ import annotations

import copy
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.training.loss import compute_loss
from src.utils.metrics import compute_mase, compute_rmse


def _resolve_device(device: torch.device | None = None) -> torch.device:
    return device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: torch.nn.Module,
    train_dataset,
    val_dataset,
    static_edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    config: dict,
    device: torch.device | None = None,
):
    device = _resolve_device(device)
    train_size = len(train_dataset.features)
    val_size = len(val_dataset.features)
    batch_size = int(config.get("batch_size", 1))
    if train_size <= 0:
        raise ValueError("Training dataset is empty.")
    if val_size <= 0:
        raise ValueError("Validation dataset is empty.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    logging.info("Training set size: %s", train_size)
    logging.info("Validation set size: %s", val_size)

    train_start_time = time.time()
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0

        indices = np.random.permutation(train_size)
        for start in tqdm(
            range(0, train_size, batch_size),
            desc=f"Epoch {epoch + 1}/{config['epochs']} - Training",
        ):
            batch_idx = indices[start : start + batch_size]
            x_np = np.stack([train_dataset.features[i] for i in batch_idx], axis=0)
            y_np = np.stack([train_dataset.targets[i] for i in batch_idx], axis=0)
            x = torch.from_numpy(x_np).float().to(device)
            y = torch.from_numpy(y_np).float().to(device)

            optimizer.zero_grad()
            y_transformed = model.transform_target(y)
            y_hat = model(x, static_edge_index)
            loss = compute_loss(y_hat, y_transformed, criterion)
            if not torch.isfinite(loss):
                logging.warning("Non-finite loss detected in epoch %s; skipping batch.", epoch + 1)
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.shape[0]

        train_loss = train_loss / train_size
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for start in tqdm(
                range(0, val_size, batch_size),
                desc=f"Epoch {epoch + 1}/{config['epochs']} - Validation",
            ):
                batch_idx = slice(start, start + batch_size)
                x_np = np.stack(val_dataset.features[batch_idx], axis=0)
                y_np = np.stack(val_dataset.targets[batch_idx], axis=0)
                x = torch.from_numpy(x_np).float().to(device)
                y = torch.from_numpy(y_np).float().to(device)

                y_transformed = model.transform_target(y)
                y_hat = model(x, static_edge_index)
                loss = compute_loss(y_hat, y_transformed, criterion)
                val_loss += loss.item() * x.shape[0]

        val_loss = val_loss / val_size
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if torch.isfinite(torch.tensor(val_loss)) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        logging.info(
            "Epoch %s/%s - Train Loss: %.6f, Val Loss: %.6f, LR: %.6f",
            epoch + 1,
            config["epochs"],
            train_loss,
            val_loss,
            optimizer.param_groups[0]["lr"],
        )

        if patience_counter >= config["patience"]:
            logging.info("Early stopping triggered at epoch %s", epoch + 1)
            break

    train_time = time.time() - train_start_time
    logging.info("Total training time: %.2f seconds", train_time)

    os.makedirs(config["output_dir"], exist_ok=True)
    timestamp = config.get("timestamp", "")
    model_path = os.path.join(config["output_dir"], f"best_model_{config['model_name']}_{timestamp}.pth")
    torch.save(best_model_state, model_path)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{config['output_dir']}/loss_curve_{config['model_name']}_{timestamp}.png", dpi=300)
    plt.close()

    return train_losses, val_losses, train_time


def evaluate_model(
    model: torch.nn.Module,
    test_dataset,
    static_edge_index: torch.Tensor,
    device: torch.device | None = None,
    config: dict | None = None,
):
    device = _resolve_device(device)
    model.eval()
    if len(test_dataset.features) <= 0:
        raise ValueError("Test dataset is empty.")

    predictions, true_values = [], []
    with torch.no_grad():
        for snapshot in tqdm(test_dataset, desc="Testing"):
            x = torch.FloatTensor(snapshot.x).unsqueeze(0).to(device)
            y = torch.FloatTensor(snapshot.y).unsqueeze(0).to(device)

            y_transformed = model.transform_target(y)
            y_hat = model(x, static_edge_index)

            predictions.append(y_hat.cpu().numpy())
            true_values.append(y_transformed.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    history_len = int(config.get("num_timesteps_in", 7)) if config else 7
    metrics = {
        "rmse": compute_rmse(true_values, predictions),
        "mase": compute_mase(true_values, predictions, num_timesteps_in=history_len),
    }

    for metric_name, value in metrics.items():
        logging.info("%s: %.4f", metric_name.upper(), value)

    return predictions, true_values, metrics
