# A Graph Neural Network-based Framework for Hierarchical Time Series Forecasting in Retail

This repository contains the implementation for a graph neural network-based framework for hierarchical time series forecasting in retail. It includes the TGLP and TALP models, learnable reconciliation variants, dataset preprocessing scripts, and experiment entry points for the proposed methods.

The public release focuses on the models and reconciliation strategies reported in the paper:

```text
Backbones:       TGLP, TALP
Reconciliation:  BU, BUL, BUN
Metrics:         RMSE, MASE
```

`TGLP` is implemented as `GCNConv + GRU + projection`; `TALP` is implemented as
`GATv2Conv + GRU + projection`. The `BU`, `BUL`, and `BUN` reconciliation
strategies are implemented once and can be attached to either backbone.

## Repository Structure

```text
gnn-hts-forecasting/
|-- configs/
|   |-- italian.yaml
|   `-- walmart.yaml
|-- datasets/
|   |-- italian/
|   |   |-- raw/
|   |   `-- graph/
|   `-- walmart/
|       |-- raw/
|       `-- graph/
|-- scripts/
|   |-- prepare_data.py
|   |-- train_single.py
|   `-- run_paper_experiments.py
|-- src/
|   |-- data/
|   |-- models/
|   |   |-- tglp/
|   |   |-- talp/
|   |   `-- reconciliation.py
|   |-- training/
|   `-- utils/
|-- pyproject.toml
|-- LICENSE
`-- README.md
```

## Installation

Create a Python environment and install the package in editable mode:

```bash
pip install -e .
```

The project requires Python 3.10 or later. PyTorch, PyTorch Geometric, and torch-geometric-temporal are listed in `pyproject.toml`; install the PyTorch/PyG builds that match your CUDA environment if you use GPU acceleration.

## Data Preparation

Raw CSV files are not redistributed in this repository. The paper cites the original data sources; please download the datasets from those sources and place the CSV files under `datasets/{italian,walmart}/raw/` before preprocessing. Hierarchy graph files are included under `datasets/{italian,walmart}/graph/`. Preprocessed tensors are intentionally not committed. Generate them before training:

```bash
python scripts/prepare_data.py --dataset all
```

This creates:

```text
datasets/italian/processed/
datasets/walmart/processed/
```

## Quick Start

Train TGLP with bottom-up direct reconciliation on the Italian dataset:

```bash
python scripts/train_single.py --config configs/italian.yaml --model tglp --reconcile bu
```

Train TALP with bottom-up nonlinear reconciliation on the Walmart dataset:

```bash
python scripts/train_single.py --config configs/walmart.yaml --model talp --reconcile bun
```

Run the main experiment scripts:

```bash
python scripts/run_paper_experiments.py
```

The paper reports neural-model means and standard deviations over five seeds.
For a five-seed run, pass the seeds explicitly, for example:

```bash
python scripts/run_paper_experiments.py --seeds 42 43 44 45 46
```

The quick-start hyperparameters are runnable defaults. The numerical tables in
the paper use tuned hyperparameters selected during the experimental workflow,
so exact table reproduction requires using the corresponding tuned settings and
the same multi-seed protocol.

Outputs are written under `output/curated/{Dataset}/{with_promotion,without_promotion}/models/methods/`.
Each run saves the best checkpoint, training-loss curve, predictions, ground truth, per-node RMSE/MASE, level RMSE/MASE, and model metadata.

This public repository contains the core proposed model implementations and
training/evaluation pipeline. Third-party baseline implementations, Optuna
search logs, statistical-significance scripts, sensitivity/convergence plotting,
and efficiency-trade-off figure generation are not included in this code release.
Please refer to the corresponding baseline papers and official repositories when
reproducing those comparisons.

## Datasets

The Italian dataset uses a 3-level hierarchy with 123 nodes: Total, Brand, and Item. The Walmart dataset uses a 4-level hierarchy with 44 nodes: Total, State, Store, and Category.

The raw CSV files are intentionally not included to avoid redistribution or licensing issues. The original data sources are cited in the paper; users who need to reproduce the experiments should download the data from those sources and prepare the files with the names expected by the configuration files.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{hu2026gnn_hts,
  title   = {A Graph Neural Network-based Framework for Hierarchical Time Series Forecasting in Retail},
  author  = {Hu, Guoping and Giurcaneanu, Ciprian Doru and Chang, Qian and Yu, Yunjun},
  journal = {Knowledge-Based Systems},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
