"""Fast repository checks that do not require the external NEO CSV."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd
import pytest

from neos_pipeline import load_dataset, resolve_data_path, run_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_missing_dataset_has_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="NEO data file not found"):
        resolve_data_path(missing)


def test_notebook_code_cells_parse_and_have_no_stale_outputs() -> None:
    notebook_path = REPO_ROOT / "Project_NEOs.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            ast.parse(source)
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None


def test_required_project_files_exist() -> None:
    for relative_path in ("README.md", "requirements.txt", "neos_pipeline.py", "ENGINEERING_REVIEW.md"):
        assert (REPO_ROOT / relative_path).is_file()


def _write_fixture(path: Path, *, target: list[object] | None = None) -> Path:
    rows = 24
    frame = pd.DataFrame(
        {
            "absolute_magnitude": [float(index) for index in range(rows)],
            "estimated_diameter_min": [index / 10 for index in range(rows)],
            "estimated_diameter_max": [index / 5 for index in range(rows)],
            "relative_velocity": [1000 + index for index in range(rows)],
            "miss_distance": [10000 + 10 * index for index in range(rows)],
            "is_hazardous": target or [index % 2 for index in range(rows)],
            "neo_id": [f"neo-{index}" for index in range(rows)],
        }
    )
    frame.to_csv(path, index=False)
    return path


def test_loader_rejects_missing_target_and_non_finite_values(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "neo.csv")
    frame = pd.read_csv(path).drop(columns="is_hazardous")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)

    path = _write_fixture(tmp_path / "neo_inf.csv")
    frame = pd.read_csv(path)
    frame["relative_velocity"] = frame["relative_velocity"].astype(float)
    frame.loc[0, "relative_velocity"] = float("inf")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        load_dataset(path)


def test_loader_rejects_missing_target_values(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "neo_missing_target.csv", target=[0, 1] + [None] * 22)
    with pytest.raises(ValueError, match="missing values"):
        load_dataset(path)


def test_experiment_validates_split_and_returns_metrics(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path / "neo.csv")
    with pytest.raises(ValueError, match="strictly between"):
        run_experiment(path, test_size=1)

    result = run_experiment(path, test_size=0.25, random_state=7)
    assert result["random_state"] == 7
    assert set(result["metrics"]) == {"logistic_regression", "random_forest"}
    expected_metrics = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"}
    for metrics in result["metrics"].values():
        assert set(metrics) == expected_metrics
        assert all(0 <= value <= 1 for value in metrics.values())
