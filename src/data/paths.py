from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

VALUES_WITH_PROMOTION = "node_values_with_promotion_log.npy"
VALUES_WITHOUT_PROMOTION = "node_values_without_promotion_log.npy"
NORMALIZATION_PARAMS = "normalization_params.npy"
SUM_MATRIX = "sum_matrix.csv"


def resolve_project_root(base_path: str | Path | None = None) -> Path:
    if base_path is None:
        return PROJECT_ROOT

    path = Path(base_path).resolve()
    if path.is_file():
        path = path.parent

    if (path / "configs").exists() and (path / "datasets").exists():
        return path
    if path.name in {"scripts", "src"} and (path.parent / "configs").exists() and (path.parent / "datasets").exists():
        return path.parent
    if path.parent.name == "datasets" and (path.parent.parent / "configs").exists():
        return path.parent.parent
    if path.name in {"raw", "processed", "graph"} and path.parent.parent.name == "datasets":
        return path.parent.parent.parent
    if path.name in {"configs", "datasets"}:
        return path.parent
    return path


def _config_path_from_name(name: str, project_root: Path) -> Path:
    candidate = Path(name)
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        candidate = Path("configs") / f"{name.lower()}.yaml"
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def load_dataset_config(
    config_source: str | Path | dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    if isinstance(config_source, dict):
        config = dict(config_source)
        config.setdefault("__config_path__", None)
        return config

    root = resolve_project_root(project_root)
    config_path = _config_path_from_name(str(config_source), root)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid dataset config format: {config_path}")

    config["__config_path__"] = str(config_path)
    return config


def dataset_name(config_source: str | Path | dict[str, Any], project_root: str | Path | None = None) -> str:
    config = load_dataset_config(config_source, project_root=project_root)
    return str(config.get("dataset_name", "Italian"))


def dataset_key(config_source: str | Path | dict[str, Any], project_root: str | Path | None = None) -> str:
    return dataset_name(config_source, project_root=project_root).strip().lower()


def dataset_root(
    config_source: str | Path | dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    config = load_dataset_config(config_source, project_root=project_root)
    root = resolve_project_root(project_root)
    data_dir = Path(config["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    return data_dir.resolve()


def raw_dir(config_source: str | Path | dict[str, Any], project_root: str | Path | None = None) -> Path:
    return dataset_root(config_source, project_root=project_root) / "raw"


def processed_dir(config_source: str | Path | dict[str, Any], project_root: str | Path | None = None) -> Path:
    return dataset_root(config_source, project_root=project_root) / "processed"


def graph_dir(config_source: str | Path | dict[str, Any], project_root: str | Path | None = None) -> Path:
    return dataset_root(config_source, project_root=project_root) / "graph"


def dataset_csv_path(
    config_source: str | Path | dict[str, Any],
    use_promotion: bool,
    project_root: str | Path | None = None,
) -> Path:
    config = load_dataset_config(config_source, project_root=project_root)
    filename = str(config["data"]["csv_pattern"]).format(
        scenario="with" if use_promotion else "without"
    )
    return raw_dir(config, project_root=project_root) / filename


def node_values_path(
    config_source: str | Path | dict[str, Any],
    use_promotion: bool,
    project_root: str | Path | None = None,
) -> Path:
    filename = VALUES_WITH_PROMOTION if use_promotion else VALUES_WITHOUT_PROMOTION
    return processed_dir(config_source, project_root=project_root) / filename


def normalization_params_path(
    config_source: str | Path | dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    return processed_dir(config_source, project_root=project_root) / NORMALIZATION_PARAMS


def sum_matrix_path(
    config_source: str | Path | dict[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    return graph_dir(config_source, project_root=project_root) / SUM_MATRIX


def adjacency_path(
    config_source: str | Path | dict[str, Any],
    adj_file: str,
    project_root: str | Path | None = None,
) -> Path:
    return graph_dir(config_source, project_root=project_root) / Path(adj_file).name


def hierarchy_level_names(
    config_source: str | Path | dict[str, Any],
    project_root: str | Path | None = None,
) -> list[str]:
    config = load_dataset_config(config_source, project_root=project_root)
    return list(config.get("hierarchy", {}).get("level_names", []))


def hierarchy_level_sizes(
    config_source: str | Path | dict[str, Any],
    project_root: str | Path | None = None,
) -> list[int]:
    config = load_dataset_config(config_source, project_root=project_root)
    return [int(size) for size in config.get("hierarchy", {}).get("level_sizes", [])]


def total_nodes(
    config_source: str | Path | dict[str, Any],
    project_root: str | Path | None = None,
) -> int:
    config = load_dataset_config(config_source, project_root=project_root)
    return int(config.get("hierarchy", {}).get("total_nodes", 0))


def required_dataset_paths(
    config_source: str | Path | dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    return {
        "raw_without_promotion_csv": dataset_csv_path(config_source, False, project_root=project_root),
        "raw_with_promotion_csv": dataset_csv_path(config_source, True, project_root=project_root),
        VALUES_WITHOUT_PROMOTION: node_values_path(config_source, False, project_root=project_root),
        VALUES_WITH_PROMOTION: node_values_path(config_source, True, project_root=project_root),
        NORMALIZATION_PARAMS: normalization_params_path(config_source, project_root=project_root),
        SUM_MATRIX: sum_matrix_path(config_source, project_root=project_root),
        "adj_hierarchy.npy": adjacency_path(config_source, "adj_hierarchy.npy", project_root=project_root),
        "adj_static_similarity_cosine.npy": adjacency_path(
            config_source,
            "adj_static_similarity_cosine.npy",
            project_root=project_root,
        ),
        "adj_static_hybrid_cosine.npy": adjacency_path(
            config_source,
            "adj_static_hybrid_cosine.npy",
            project_root=project_root,
        ),
    }
