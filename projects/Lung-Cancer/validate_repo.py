"""Static repository validation for the lung-cancer notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOK = REPO_ROOT / "Lung Cancer.ipynb"
FORBIDDEN_PATH_MARKERS = ("/home/", "/content/", "\\users\\", "j:\\", "e:\\")
REQUIRED_NOTEBOOK_FILES = ("README.md", "requirements.txt", "lung_cancer.py")


def _cell_source(cell: dict[str, Any], index: int, errors: list[str]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list) and all(isinstance(part, str) for part in source):
        return "".join(source)
    if isinstance(source, str):
        return source
    errors.append(f"Cell {index} has a malformed source field")
    return ""


def validate_notebook(path: Path = NOTEBOOK) -> list[str]:
    """Return actionable static errors for notebook syntax, state, and paths."""
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [f"Notebook not found: {path}"]
    except json.JSONDecodeError as exc:
        return [f"Invalid notebook JSON: {exc}"]

    if not isinstance(notebook, dict):
        return ["Notebook root must be a JSON object"]
    if notebook.get("nbformat") != 4:
        errors.append(f"Unsupported notebook format: {notebook.get('nbformat')!r}")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return errors + ["Notebook does not contain a cells list"]

    all_source: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"Cell {index} is not an object")
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            errors.append(f"Cell {index} has an invalid cell_type: {cell_type!r}")
        source = _cell_source(cell, index, errors)
        all_source.append(source)
        if cell.get("outputs"):
            errors.append(f"Cell {index} contains stale outputs")
        if cell.get("execution_count") is not None:
            errors.append(f"Cell {index} contains a stale execution count")
        if cell_type == "code":
            try:
                ast.parse(source, filename=f"{path.name}:cell-{index}")
            except SyntaxError as exc:
                errors.append(f"Cell {index} has syntax error: {exc}")
        lowered = source.lower()
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in lowered:
                errors.append(f"Cell {index} contains a local absolute path marker: {marker}")

    source_text = "\n".join(all_source)
    if "FileNotFoundError" not in source_text or "LUNG_CANCER_DATA" not in source_text:
        errors.append("Notebook is missing the explicit data-missing guard")
    if "validate_feature_values" not in source_text:
        errors.append("Notebook is missing prediction feature validation")
    if "load_artifact" not in source_text:
        errors.append("Notebook is missing artifact reload validation")
    return errors


def validate_required_files() -> list[str]:
    """Check files required for the documented repository workflow."""
    return [
        f"Required file missing: {name}"
        for name in REQUIRED_NOTEBOOK_FILES
        if not (REPO_ROOT / name).is_file()
    ]


def main() -> int:
    errors = validate_required_files() + validate_notebook()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Validated {NOTEBOOK.name}: syntax, clean outputs, schema guards, and paths OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
