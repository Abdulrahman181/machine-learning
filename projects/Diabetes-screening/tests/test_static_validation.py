"""Dependency-light checks for the repository's portable entry points."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from diabetes_contracts import FEATURES, resolve_model_path, validate_model_contract, validate_prediction

ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_parse() -> None:
    for path in ROOT.rglob("*.py"):
        if ".git" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_streamlit_entrypoint_has_no_machine_specific_path() -> None:
    source = (ROOT / "diabetes.py").read_text(encoding="utf-8")
    assert "C:\\\\Users" not in source
    assert "DIABETES_MODEL_PATH" in source
    assert "FEATURES" in source


def test_notebook_is_valid_json_and_has_no_stale_outputs() -> None:
    notebook = json.loads((ROOT / "diabetes.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        assert "\\n" not in source
        ast.parse(source, filename=f"diabetes.ipynb:cell-{index}")
        assert not cell.get("outputs") and cell.get("execution_count") is None


def test_model_path_is_independent_of_process_cwd(tmp_path: Path) -> None:
    assert resolve_model_path(None, tmp_path) == (tmp_path / "model.pkl").resolve()
    assert resolve_model_path("relative/model.pkl", tmp_path) == (tmp_path / "relative/model.pkl").resolve()


class CompatibleModel:
    n_features_in_ = len(FEATURES)
    feature_names_in_ = FEATURES

    def predict(self, _frame: object) -> list[int]:
        return [1]


def test_model_contract_and_prediction_validation() -> None:
    assert validate_model_contract(CompatibleModel()) is not None
    assert validate_prediction([1]) == 1


def test_prediction_validation_rejects_ambiguous_outputs() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        validate_prediction([0, 1])
    with pytest.raises(ValueError, match="invalid class"):
        validate_prediction([2])
