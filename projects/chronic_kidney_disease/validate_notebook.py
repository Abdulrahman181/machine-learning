"""Static checks for the CKD notebook; does not require the dataset or ML dependencies."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).with_name("Chronical_kidney_disease.ipynb")


def main() -> int:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise AssertionError("Notebook must use nbformat 4")

    if notebook.get("nbformat_minor", 0) < 5:
        raise AssertionError("Notebook must use a current nbformat 4 minor version")

    code_sources: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=f"{NOTEBOOK.name}:cell-{index}")
            code_sources.append(source)
            if cell.get("outputs"):
                raise AssertionError(f"Cell {index} contains stale outputs")
            if cell.get("execution_count") is not None:
                raise AssertionError(f"Cell {index} has a stale execution count")
        elif cell.get("cell_type") == "markdown":
            markdown = "".join(cell.get("source", []))
            if any(line.strip().startswith('"') for line in markdown.splitlines()):
                raise AssertionError(f"Cell {index} contains malformed quoted markdown")

    source = "\n".join(code_sources)
    required_tokens = {
        "CKD_DATA_DIR",
        "FileNotFoundError",
        "train_test_split",
        "stratify=y",
        "random_state=SEED",
        "ColumnTransformer",
        "SimpleImputer",
        "OneHotEncoder",
        "cross_validate",
        "roc_auc",
        "balanced_accuracy",
        "predict_proba",
        "FEATURE_COLUMNS",
        "duplicate columns after alias mapping",
        "Unsupported or missing target labels",
        "load_model_artifact",
        "artifact_version",
    }
    missing = sorted(token for token in required_tokens if token not in source)
    if missing:
        raise AssertionError(f"Notebook is missing required safeguards: {missing}")

    forbidden_patterns = [
        r"/home/ubuntu/",
        r"[A-Za-z]:\\",
        r"SelectKBest",
        r"\.sample\(",
        r"accuracy\s*=\s*0\.\d+",
    ]
    for pattern in forbidden_patterns:
        if re.search(pattern, source):
            raise AssertionError(f"Non-reproducible or unsafe pattern found: {pattern}")

    print(f"Static validation passed: {NOTEBOOK.name}; {len(code_sources)} code cells; no outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
