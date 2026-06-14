from __future__ import annotations

from pathlib import Path


def _dataset_label(dataset_name: str) -> str:
    normalized = str(dataset_name).strip()
    if not normalized:
        return "Italian"
    return normalized[0].upper() + normalized[1:]


def curated_root(project_root: Path, dataset_name: str = "Italian") -> Path:
    return project_root / "output" / "curated" / _dataset_label(dataset_name)


def scenario_name(use_promotion: bool) -> str:
    return "with_promotion" if use_promotion else "without_promotion"


def methods_dir(
    project_root: Path,
    model_name: str,
    use_promotion: bool,
    graph_name: str = "hierarchy",
    dataset_name: str = "Italian",
) -> Path:
    return (
        curated_root(project_root, dataset_name=dataset_name)
        / scenario_name(use_promotion)
        / "models"
        / "methods"
        / model_name
        / graph_name
    )
