#!/usr/bin/env python3
"""Train one TGLP/TALP configuration from configs/*.yaml."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric_temporal.signal import temporal_signal_split


def _get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    if current_file.parent.name == "scripts":
        return current_file.parent.parent
    return current_file.parent


PROJECT_ROOT = _get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._output_paths import methods_dir
from src.data.loader import LoadDatasetLoader
from src.data.paths import dataset_name, hierarchy_level_names, hierarchy_level_sizes, load_dataset_config
from src.models.reconciliation import (
    BUReconciliation,
    BULReconciliation,
    BUNReconciliation,
    ReconciledForecastModel,
)
from src.models.talp import TALP
from src.models.tglp import TGLP
from src.training.io import plot_predictions, save_model_info, save_predictions, to_serializable
from src.training.trainer import evaluate_model, train_model
from src.utils.metrics import calculate_level_metrics
from src.utils.seed import set_seed


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CHOICES = ("tglp", "talp")
RECONCILE_CHOICES = ("bu", "bul", "bun")
GRAPH_DIR_MAP = {
    "adj_hierarchy.npy": "hierarchy",
    "adj_static_hybrid_cosine.npy": "hybrid_cosine",
    "adj_static_similarity_cosine.npy": "similarity_cosine",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Dataset config path or config name under configs/.")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--reconcile", required=True, choices=RECONCILE_CHOICES)
    parser.add_argument(
        "--scenario",
        choices=("with", "without"),
        default=None,
        help="Promotion scenario. Defaults to the first scenario listed in the config.",
    )
    parser.add_argument("--adj-file", default="adj_hierarchy.npy")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8, help="TALP only.")
    parser.add_argument("--dropout", type=float, default=0.2, help="TALP only.")
    parser.add_argument("--bun-mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--bun-mlp-layers", type=int, default=2)
    parser.add_argument("--bun-mlp-dropout", type=float, default=0.1)
    parser.add_argument("--bun-mlp-decay", type=float, default=0.5)
    parser.add_argument("--num-timesteps-in", type=int, default=None)
    parser.add_argument("--num-timesteps-out", type=int, default=None)
    parser.add_argument(
        "--plot-nodes",
        type=int,
        default=0,
        help="How many nodes to plot after evaluation. Use -1 for all, 0 to skip.",
    )
    parser.add_argument("--output-dir", default=None, help="Optional explicit run output directory.")
    return parser.parse_args()


def _resolve_scenario(config: dict, scenario_arg: str | None) -> tuple[str, bool]:
    scenarios = [str(item) for item in config.get("data", {}).get("scenarios", ["with", "without"])]
    scenario = scenario_arg or scenarios[0]
    if scenario not in scenarios:
        raise ValueError(f"Scenario {scenario!r} is not defined in config: {scenarios}")
    return scenario, scenario == "with"


def _internal_model_name(model: str, reconcile: str) -> str:
    prefixes = {"tglp": "GCN-GRU-LP", "talp": "GAT-GRU-LP"}
    return f"{prefixes[model]}-{reconcile.upper()}"


def _graph_output_name(adj_file: str) -> str:
    return GRAPH_DIR_MAP.get(Path(adj_file).name, Path(adj_file).stem)


def _select_nodes_to_plot(total_nodes: int, plot_nodes: int) -> range:
    if plot_nodes == 0:
        return range(0)
    if plot_nodes < 0:
        return range(total_nodes)
    return range(min(total_nodes, plot_nodes))


def _build_model(config: dict, loader: LoadDatasetLoader) -> torch.nn.Module:
    backbone_args = (
        config["node_num"],
        config["input_dim"],
        config["hidden_dim"],
        config["output_dim"],
        config["num_layers"],
        loader.global_min,
        loader.global_max,
    )
    if config["model_family"] == "tglp":
        backbone = TGLP(
            *backbone_args,
            gnn_layers=config["gnn_layers"],
            gru_layers=config["gru_layers"],
            dropout=config["dropout"],
        )
    elif config["model_family"] == "talp":
        backbone = TALP(
            *backbone_args,
            heads=config["heads"],
            dropout=config["dropout"],
            gnn_layers=config["gnn_layers"],
            gru_layers=config["gru_layers"],
        )
    else:
        raise ValueError(f"Unsupported model_family: {config['model_family']}")

    if config["reconcile"] == "bu":
        reconciliation = BUReconciliation(config["node_num"], loader.sum_matrix)
    elif config["reconcile"] == "bul":
        reconciliation = BULReconciliation(config["node_num"], loader.sum_matrix)
    elif config["reconcile"] == "bun":
        reconciliation = BUNReconciliation(
            config["node_num"],
            loader.sum_matrix,
            mlp_hidden_dim=config["bun_mlp_hidden_dim"],
            mlp_layers=config["bun_mlp_layers"],
            mlp_dropout=config["bun_mlp_dropout"],
            mlp_decay=config["bun_mlp_decay"],
        )
    else:
        raise ValueError(f"Unsupported reconciliation strategy: {config['reconcile']}")

    return ReconciledForecastModel(backbone, reconciliation).to(DEVICE)


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2, ensure_ascii=False)


def main() -> None:
    args = _parse_args()

    dataset_config = load_dataset_config(args.config, project_root=PROJECT_ROOT)
    dataset_label = dataset_name(dataset_config)
    scenario, use_promotion = _resolve_scenario(dataset_config, args.scenario)
    internal_model_name = _internal_model_name(args.model, args.reconcile)

    training_defaults = dataset_config.get("training", {})
    num_timesteps_in = args.num_timesteps_in or int(training_defaults.get("default_num_timesteps_in", 7))
    num_timesteps_out = args.num_timesteps_out or int(training_defaults.get("default_num_timesteps_out", 1))
    epochs = args.epochs or int(training_defaults.get("default_epochs", 150))
    patience = args.patience or int(training_defaults.get("default_patience", 20))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_name = _graph_output_name(args.adj_file)
    base_output_dir = methods_dir(
        PROJECT_ROOT,
        args.model.upper(),
        use_promotion,
        graph_name=graph_name,
        dataset_name=dataset_label,
    )
    default_run_dir = base_output_dir / f"{internal_model_name}_{timestamp}"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_run_dir

    config = {
        "config_path": dataset_config.get("__config_path__"),
        "dataset_name": dataset_label,
        "model_family": args.model,
        "reconcile": args.reconcile,
        "model_name": internal_model_name,
        "scenario": scenario,
        "use_promotion": use_promotion,
        "output_dir": str(output_dir),
        "adj_file": Path(args.adj_file).name,
        "node_num": int(dataset_config["hierarchy"]["total_nodes"]),
        "input_dim": 2 if use_promotion else 1,
        "hidden_dim": args.hidden_dim,
        "output_dim": num_timesteps_out,
        "num_layers": args.gru_layers,
        "gnn_layers": args.gnn_layers,
        "gru_layers": args.gru_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "bun_mlp_hidden_dim": args.bun_mlp_hidden_dim,
        "bun_mlp_layers": args.bun_mlp_layers,
        "bun_mlp_dropout": args.bun_mlp_dropout,
        "bun_mlp_decay": args.bun_mlp_decay,
        "batch_size": args.batch_size,
        "epochs": epochs,
        "lr": args.lr,
        "patience": patience,
        "seed": args.seed,
        "timestamp": timestamp,
        "num_timesteps_in": num_timesteps_in,
        "num_timesteps_out": num_timesteps_out,
        "plot_horizon_index": max(0, num_timesteps_out - 1),
        "hierarchy_level_names": hierarchy_level_names(dataset_config),
        "hierarchy_level_sizes": hierarchy_level_sizes(dataset_config),
    }

    logging.info("Training %s on %s (%s) | device=%s", config["model_name"], dataset_label, scenario, DEVICE)

    set_seed(config["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = LoadDatasetLoader(
        PROJECT_ROOT,
        use_promotion=use_promotion,
        input_dim=config["input_dim"],
        adj_file=config["adj_file"],
        config_source=dataset_config,
    )

    config["num_total_nodes"] = int(loader.num_total_nodes)
    config["num_bottom_nodes"] = int(loader.num_bottom_nodes)
    config["bottom_start_idx"] = int(loader.bottom_start_idx)
    config["num_mid_nodes"] = int(loader.num_mid_nodes)

    if config["node_num"] != config["num_total_nodes"]:
        logging.warning(
            "Config node_num=%s != loaded nodes=%s, auto-align.",
            config["node_num"],
            config["num_total_nodes"],
        )
        config["node_num"] = config["num_total_nodes"]

    _save_json(output_dir / f"config_{config['model_name']}_{timestamp}.json", config)

    dataset = loader.get_dataset(num_timesteps_in=num_timesteps_in, num_timesteps_out=num_timesteps_out)
    total_snapshots = len(dataset.features)
    first_train_len = int(total_snapshots * 0.8)

    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
    train_dataset, val_dataset = temporal_signal_split(train_dataset, train_ratio=0.8)

    if loader.target_time_index is None:
        raise RuntimeError("loader.target_time_index is empty after dataset generation.")

    test_time_index = loader.target_time_index[first_train_len:]
    if len(test_time_index) != len(test_dataset.features):
        raise ValueError(f"test_time_index length {len(test_time_index)} != test samples {len(test_dataset.features)}")
    config["time_index"] = list(test_time_index)
    config["node_names"] = loader.node_names

    if loader.target_time_index_full is not None:
        test_time_index_full = loader.target_time_index_full[first_train_len:]
        if len(test_time_index_full) == len(test_dataset.features):
            config["time_index_full"] = test_time_index_full

    static_edge_index = torch.tensor(loader.edges, dtype=torch.long).to(DEVICE)
    model = _build_model(config, loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.SmoothL1Loss()

    train_losses, val_losses, train_time = train_model(
        model,
        train_dataset,
        val_dataset,
        static_edge_index,
        optimizer,
        criterion,
        config,
        DEVICE,
    )

    training_results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_time": train_time,
    }
    _save_json(output_dir / f"training_results_{config['model_name']}_{timestamp}.json", training_results)

    best_model_path = output_dir / f"best_model_{config['model_name']}_{timestamp}.pth"
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    except Exception as exc:
        logging.warning("Could not reload best checkpoint from %s: %s", best_model_path, exc)

    predictions, true_values, metrics = evaluate_model(model, test_dataset, static_edge_index, DEVICE, config)
    _save_json(output_dir / f"metrics_{config['model_name']}_{timestamp}.json", metrics)

    level_metrics = calculate_level_metrics(predictions, true_values, config)
    save_predictions(predictions, true_values, config)

    for node_idx in _select_nodes_to_plot(predictions.shape[1], args.plot_nodes):
        plot_predictions(predictions, true_values, node_idx, config)

    save_model_info(model, config, metrics, level_metrics, training_results)
    logging.info("Training and evaluation completed. Outputs: %s", output_dir)


if __name__ == "__main__":
    main()
