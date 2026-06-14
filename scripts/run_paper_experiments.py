#!/usr/bin/env python3
"""Run the public paper experiment grid: TGLP/TALP x BU/BUL/BUN."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_single.py"

MODEL_CHOICES = ("tglp", "talp")
RECONCILE_CHOICES = ("bu", "bul", "bun")
SCENARIO_CHOICES = ("with", "without")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/italian.yaml", "configs/walmart.yaml"],
        help="Dataset config paths or config names.",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--reconciles", nargs="+", choices=RECONCILE_CHOICES, default=list(RECONCILE_CHOICES))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_CHOICES, default=list(SCENARIO_CHOICES))
    parser.add_argument("--adj-file", default="adj_hierarchy.npy")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=1)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bun-mlp-hidden-dim", type=int, default=128)
    parser.add_argument("--bun-mlp-layers", type=int, default=2)
    parser.add_argument("--bun-mlp-dropout", type=float, default=0.1)
    parser.add_argument("--bun-mlp-decay", type=float, default=0.5)
    parser.add_argument("--num-timesteps-in", type=int, default=None)
    parser.add_argument("--num-timesteps-out", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def _optional_arg(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def _build_command(
    args: argparse.Namespace,
    config: str,
    model: str,
    reconcile: str,
    scenario: str,
    seed: int,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        config,
        "--model",
        model,
        "--reconcile",
        reconcile,
        "--scenario",
        scenario,
        "--adj-file",
        args.adj_file,
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(seed),
        "--hidden-dim",
        str(args.hidden_dim),
        "--gnn-layers",
        str(args.gnn_layers),
        "--gru-layers",
        str(args.gru_layers),
        "--heads",
        str(args.heads),
        "--dropout",
        str(args.dropout),
        "--bun-mlp-hidden-dim",
        str(args.bun_mlp_hidden_dim),
        "--bun-mlp-layers",
        str(args.bun_mlp_layers),
        "--bun-mlp-dropout",
        str(args.bun_mlp_dropout),
        "--bun-mlp-decay",
        str(args.bun_mlp_decay),
    ]
    _optional_arg(command, "--epochs", args.epochs)
    _optional_arg(command, "--patience", args.patience)
    _optional_arg(command, "--num-timesteps-in", args.num_timesteps_in)
    _optional_arg(command, "--num-timesteps-out", args.num_timesteps_out)
    return command


def main() -> None:
    args = _parse_args()
    combinations = itertools.product(args.configs, args.models, args.reconciles, args.scenarios, args.seeds)

    for config, model, reconcile, scenario, seed in combinations:
        command = _build_command(args, config, model, reconcile, scenario, seed)
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
