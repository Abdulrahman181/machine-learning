"""Fast repository checks plus a small deterministic execution fixture."""

import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "vehicles-EDA.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_valid_and_code_compiles() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] >= 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_notebook_has_no_stale_outputs_or_execution_counts() -> None:
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_data_loading_is_portable_and_data_gated() -> None:
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    )
    assert "/kaggle/" not in source
    assert "USED_CARS_DATA" in source
    assert "FileNotFoundError" in source
    assert "synthetic" not in source.lower()
    assert "format=\"mixed\"" in source
    assert "Rows without an ID are retained" in source


def test_notebook_executes_with_malformed_values_without_stale_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise all cells with malformed numeric input, without claiming EDA results."""
    data_path = tmp_path / "vehicles.csv"
    columns = [
        "id",
        "price",
        "year",
        "condition",
        "title_status",
        "type",
        "odometer",
        "fuel",
        "posting_date",
    ]
    rows = [
        ["1", "12000", "2018.5", "good", "clean", "sedan", "50000", "gas", "2024-01-01"],
        ["1", "13000", "2020", "good", "clean", "sedan", "51000", "gas", "2024-01-02"],
        ["", "bad", "2021", "excellent", "clean", "truck", "-5", "diesel", "not-a-date"],
        ["", "9000", "1890", "", "clean", "suv", "70000", "gas", "2024-01-03"],
    ]
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)

    monkeypatch.setenv("USED_CARS_DATA", str(data_path))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace: dict[str, object] = {}
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
            if "plt" in namespace:
                namespace["plt"].show = lambda *args, **kwargs: None

    # Duplicate non-missing IDs are removed, while rows without IDs remain.
    assert len(namespace["analysis_df"]) == 3
    assert list(namespace["year_counts"].index) == [1890, 2021]
    namespace["plt"].close("all")


def test_required_runtime_dependencies_are_declared() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("pandas", "numpy", "matplotlib", "seaborn", "jupyter"):
        assert package in requirements
