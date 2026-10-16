from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = next(ROOT.glob("*.ipynb"))


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_code_parses_and_has_no_stale_outputs() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] >= 4
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        assert not cell.get("outputs"), "committed notebook output should be empty"
        source = "".join(cell.get("source", []))
        source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("%")
        )
        ast.parse(source)


def test_data_loading_is_portable_and_guarded() -> None:
    notebook = _notebook()
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "IMDB_DATA_PATH" in source
    assert "FileNotFoundError" in source
    assert "D:\\IMDb_All_Genres_etf_clean1.csv" not in source
    assert "LEAKAGE_COLUMNS" in source
    assert "random_state=42" in source
