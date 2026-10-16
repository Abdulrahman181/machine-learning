"""Dependency-free static validation for the repository's executable notebook."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "cardiac-surgery-and-percutaneous (1).ipynb"


def validate_notebook(path: Path = NOTEBOOK) -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read valid notebook JSON: {exc}"]

    if notebook.get("nbformat") != 4:
        errors.append("notebook must use nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("notebook must contain a non-empty cells list")
        return errors

    code_source: list[str] = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            code_source.append(source)
            try:
                ast.parse(source, filename=f"{path.name}:cell-{index}")
            except SyntaxError as exc:
                errors.append(f"cell {index} has syntax error: {exc}")
            if cell.get("outputs"):
                errors.append(f"cell {index} contains stale outputs")
            if cell.get("execution_count") is not None:
                errors.append(f"cell {index} has a stale execution count")

    joined = "\n".join(code_source)
    for required in (
        "CARDIAC_SURGERY_DATA",
        "prepare_data",
        "ColumnTransformer",
        "classification_report",
        "StratifiedGroupKFold",
    ):
        if required not in joined:
            errors.append(f"notebook is missing expected workflow component: {required}")
    for removed in ("tensorflow", "keras", "xgboost", "LabelEncoder"):
        if removed in joined:
            errors.append(f"notebook contains removed/unnecessary dependency or unsafe pattern: {removed}")
    return errors


def main() -> int:
    errors = validate_notebook()
    if errors:
        print("Notebook validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Notebook validation passed: {NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
