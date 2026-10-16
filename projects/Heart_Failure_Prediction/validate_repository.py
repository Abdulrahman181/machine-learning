#!/usr/bin/env python3
"""Static validation for the Heart Failure Prediction repository.

This validator deliberately does not import the optional ML stack or execute the
notebook, so it remains useful when the dataset and native dependencies are absent.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "Heart_Failure_Prediction.ipynb"
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"/content/|/Users/|/home/[^/]+/"),
    re.compile(r"[A-Za-z]:[\\/]"),
)


def validate_notebook() -> list[str]:
    errors: list[str] = []
    try:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read valid notebook JSON: {exc}"]

    if notebook.get("nbformat") != 4:
        errors.append("notebook must use nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("notebook must contain cells")
        return errors

    code_sources: list[str] = []
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            code_sources.append(source)
            try:
                ast.parse(source)
            except SyntaxError as exc:
                errors.append(f"cell {index} has invalid Python: {exc}")
            if cell.get("outputs"):
                errors.append(f"cell {index} contains stale outputs")
            if cell.get("execution_count") is not None:
                errors.append(f"cell {index} has a non-empty execution count")

    source = "\n".join(code_sources)
    required_fragments = (
        "HEART_FAILURE_DATA",
        "stratify=y",
        "ColumnTransformer",
        "Pipeline",
        'handle_unknown="ignore"',
        "random_state=RANDOM_STATE",
    )
    for fragment in required_fragments:
        if fragment not in source:
            errors.append(f"notebook is missing required reproducibility marker: {fragment}")
    for pattern in ABSOLUTE_PATH_PATTERNS:
        if pattern.search(source):
            errors.append(f"notebook contains a non-portable absolute path matching {pattern.pattern}")
    return errors


def validate_python_files() -> list[str]:
    errors: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if ".git" in path.parts or path.name.startswith("."):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f"{path.relative_to(ROOT)} is not valid Python: {exc}")
    return errors


def main() -> int:
    errors = validate_notebook() + validate_python_files()
    if errors:
        print("STATIC VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("STATIC VALIDATION PASSED: notebook JSON, AST, outputs, paths, and Python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
