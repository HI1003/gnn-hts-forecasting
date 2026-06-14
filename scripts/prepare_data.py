#!/usr/bin/env python3
"""
Preprocess raw CSV data into .npy files consumed by the models.

Pipeline (per dataset):
  1. Read the CSV, separate quantity columns and promotion columns.
  2. Apply log1p transform:  x_log = log(1 + x)   (to BOTH quantities and promotions)
  3. Global min-max normalization:  x_norm = (x_log - global_min) / (global_max - global_min)
     where global_min and global_max are computed over quantity nodes across all time steps,
     and the SAME (global_min, global_max) is reused for the promotion channel so that both
     channels live in the same log-normalized space consumed by the loader / denormalizer.
  4. Save:
     - node_values_without_promotion_log.npy  shape (T, N, 1)
     - node_values_with_promotion_log.npy     shape (T, N, 2)  (channel 0 = qty, channel 1 = promo)
     - normalization_params.npy               dict {global_min, global_max}

Usage:
  python scripts/prepare_data.py --dataset italian
  python scripts/prepare_data.py --dataset walmart
  python scripts/prepare_data.py --dataset all
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
DATASETS = {
    "italian": {
        "raw_dir": PROJECT_ROOT / "datasets" / "italian" / "raw",
        "out_dir": PROJECT_ROOT / "datasets" / "italian" / "processed",
        "csv_without": "Italian_dataset_without_promotion.csv",
        "csv_with": "Italian_dataset_with_promotion.csv",
        "num_nodes": 123,
        "promo_prefix": "PROMO_",
    },
    "walmart": {
        "raw_dir": PROJECT_ROOT / "datasets" / "walmart" / "raw",
        "out_dir": PROJECT_ROOT / "datasets" / "walmart" / "processed",
        "csv_without": "Walmart_dataset_without_promotion.csv",
        "csv_with": "Walmart_dataset_with_promotion.csv",
        "num_nodes": 44,
        "promo_prefix": "_snap",
    },
}


def _is_promo_col(col: str, promo_prefix: str) -> bool:
    """Identify promotion/snap indicator columns."""
    s = col.lower()
    if promo_prefix.lower().startswith("_"):
        # suffix style: e.g. "total_snap", "CA_snap"
        return s.endswith(promo_prefix.lower())
    else:
        # prefix style: e.g. "PROMO_QTY", "PROMO_B1"
        return s.startswith(promo_prefix.lower())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")
    return pd.read_csv(path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_numeric_array(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if values.min() < 0:
        raise ValueError(f"{name} contains negative values, which are invalid for log1p.")


def _canonical_quantity_col(col: str) -> str:
    if col == "QTY":
        return "total"
    if col.startswith("QTY_"):
        return col[len("QTY_"):]
    return col


def prepare_dataset(name: str) -> None:
    cfg = DATASETS[name]
    raw_dir: Path = cfg["raw_dir"]
    out_dir: Path = cfg["out_dir"]
    num_nodes: int = cfg["num_nodes"]
    promo_prefix: str = cfg["promo_prefix"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Read CSVs
    # ------------------------------------------------------------------
    df_without = _read_csv(raw_dir / cfg["csv_without"])
    df_with = _read_csv(raw_dir / cfg["csv_with"])

    _require(len(df_without) == len(df_with), f"{name}: raw CSV row counts differ.")
    without_dates = df_without.iloc[:, 0].astype(str).tolist()
    with_dates = df_with.iloc[:, 0].astype(str).tolist()
    _require(without_dates == with_dates, f"{name}: raw CSV date columns are not aligned.")

    # Separate quantity columns (exclude date column and promo columns)
    all_cols_with = list(df_with.columns[1:])  # skip date
    qty_cols = [c for c in all_cols_with if not _is_promo_col(c, promo_prefix)]
    promo_cols = [c for c in all_cols_with if _is_promo_col(c, promo_prefix)]

    _require(
        len(qty_cols) == num_nodes,
        f"{name}: expected {num_nodes} quantity columns, got {len(qty_cols)}: {qty_cols[:5]}...",
    )
    _require(
        len(promo_cols) == num_nodes,
        f"{name}: expected {num_nodes} promo columns, got {len(promo_cols)}: {promo_cols[:5]}...",
    )

    # Quantity values: from the "without" CSV (all columns except date)
    qty_without_cols = list(df_without.columns[1:])
    _require(
        len(qty_without_cols) == num_nodes,
        f"{name}: expected {num_nodes} quantity columns in without-promotion CSV, "
        f"got {len(qty_without_cols)}.",
    )
    _require(
        qty_without_cols == [_canonical_quantity_col(col) for col in qty_cols],
        f"{name}: quantity columns differ between with- and without-promotion CSVs.",
    )
    qty_values = df_without[qty_without_cols].values.astype(np.float64)  # (T, N)

    # Promo values: from the "with" CSV
    promo_values = df_with[promo_cols].values.astype(np.float64)  # (T, N)
    _validate_numeric_array(f"{name} quantity values", qty_values)
    _validate_numeric_array(f"{name} promotion values", promo_values)

    T = qty_values.shape[0]
    logger.info(f"[{name}] T={T}, N={num_nodes}, promo_cols={len(promo_cols)}")

    # ------------------------------------------------------------------
    # 2. Log transform on quantities
    # ------------------------------------------------------------------
    qty_log = np.log1p(qty_values)  # (T, N)

    # ------------------------------------------------------------------
    # 3. Global min-max normalization
    # ------------------------------------------------------------------
    global_min = float(qty_log.min())
    global_max = float(qty_log.max())
    _require(global_max > global_min, f"{name}: quantity values have zero normalization range.")
    logger.info(f"[{name}] global_min={global_min:.6f}, global_max={global_max:.6f}")

    qty_norm = (qty_log - global_min) / (global_max - global_min)  # (T, N), range [0, 1]

    # Promotion channel: same log+normalize pipeline, sharing the quantity-derived
    # (global_min, global_max). Keeps both channels in one log-normalized space so the
    # loader's shared denormalizer (src/data/loader.py) stays consistent.
    promo_log = np.log1p(promo_values)
    promo_norm = (promo_log - global_min) / (global_max - global_min)

    # ------------------------------------------------------------------
    # 4. Assemble and save
    # ------------------------------------------------------------------
    # Without promotion: (T, N, 1)
    node_values_without = qty_norm[:, :, np.newaxis]

    # With promotion: (T, N, 2); channel 0 = normalized qty, channel 1 = normalized promo.
    node_values_with = np.stack([qty_norm, promo_norm], axis=-1)

    np.save(out_dir / "node_values_without_promotion_log.npy", node_values_without)
    np.save(out_dir / "node_values_with_promotion_log.npy", node_values_with)
    np.save(out_dir / "normalization_params.npy", {
        "global_min": global_min,
        "global_max": global_max,
    })

    logger.info(
        f"[{name}] Saved to {out_dir}/:\n"
        f"  node_values_without_promotion_log.npy  shape={node_values_without.shape}\n"
        f"  node_values_with_promotion_log.npy     shape={node_values_with.shape}\n"
        f"  normalization_params.npy               {{global_min={global_min}, global_max={global_max}}}"
    )


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw CSV data into .npy files.")
    parser.add_argument(
        "--dataset",
        choices=["italian", "walmart", "all"],
        default="all",
        help="Which dataset to preprocess (default: all)",
    )
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        logger.info(f"Processing {name}...")
        prepare_dataset(name)
    logger.info("Done.")


if __name__ == "__main__":
    main()
