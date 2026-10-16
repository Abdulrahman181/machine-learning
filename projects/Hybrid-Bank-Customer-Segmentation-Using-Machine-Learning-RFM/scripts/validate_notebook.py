"""Static validation for every repository notebook; does not require the dataset."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Bank_Customer_Segmentation.ipynb"
FORBIDDEN_PATH_MARKERS = ("/content/", "C:\\", "\\\\content\\")


def notebook_paths(root: Path = ROOT) -> list[Path]:
    """Return repository notebooks while excluding VCS/tool caches."""
    return sorted(
        path
        for path in root.rglob("*.ipynb")
        if ".git" not in path.parts and ".ipynb_checkpoints" not in path.parts
    )


def validate_notebook(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot read valid notebook JSON: {exc}"]

    if notebook.get("nbformat") != 4:
        errors.append("expected nbformat 4")
    if not isinstance(notebook.get("cells"), list):
        return errors + ["expected a list of notebook cells"]

    cell_ids: set[str] = set()
    for index, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict):
            errors.append(f"cell {index} is not an object")
            continue
        cell_id = cell.get("id")
        if cell_id is not None:
            if cell_id in cell_ids:
                errors.append(f"cell {index} duplicates id {cell_id!r}")
            cell_ids.add(cell_id)
        if cell.get("cell_type") not in {"code", "markdown", "raw"}:
            errors.append(f"cell {index} has invalid cell_type")
            continue
        source = cell.get("source", [])
        if not isinstance(source, (list, str)):
            errors.append(f"cell {index} has invalid source type")
            source_text = ""
        else:
            source_text = "".join(source) if isinstance(source, list) else source
        if cell.get("cell_type") != "code":
            continue
        try:
            ast.parse(source_text, filename=f"{path}:cell-{index}")
        except SyntaxError as exc:
            errors.append(f"cell {index} has syntax error: {exc}")
        if cell.get("outputs"):
            errors.append(f"cell {index} contains saved outputs")
        if cell.get("execution_count") is not None:
            errors.append(f"cell {index} contains an execution count")
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in source_text:
                errors.append(f"cell {index} contains non-portable path marker {marker!r}")

    return errors


def validate_all_notebooks(root: Path = ROOT) -> list[str]:
    notebooks = notebook_paths(root)
    if not notebooks:
        return ["no notebooks found"]
    errors: list[str] = []
    for path in notebooks:
        errors.extend(f"{path.relative_to(root)}: {error}" for error in validate_notebook(path))
    return errors


def main() -> int:
    errors = validate_all_notebooks()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"OK: {len(notebook_paths())} notebook(s) passed static validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
