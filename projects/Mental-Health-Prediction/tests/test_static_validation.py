"""Dependency-light static checks for the repository notebook."""

from __future__ import annotations

import json
import re
import ast
import csv
import os
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError:  # Keep static checks runnable before runtime dependencies are installed.
    np = None
    pd = None

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Mental Health Prediction.ipynb"


class NotebookStaticValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code_cells = [
            cell
            for cell in cls.notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        ]
        cls.code = "\n\n".join(
            "".join(cell.get("source", [])) for cell in cls.code_cells
        )

    def test_notebook_schema_and_code_compile(self) -> None:
        self.assertEqual(self.notebook.get("nbformat"), 4)
        self.assertGreater(len(self.code_cells), 0)
        for index, cell in enumerate(self.code_cells):
            self.assertIsNone(
                cell.get("execution_count"), f"cell {index} has stale execution count"
            )
            self.assertEqual(
                cell.get("outputs", []), [], msg=f"cell {index} has stale notebook output"
            )
            compile("".join(cell.get("source", [])), f"<cell {index}>", "exec")

    def test_data_path_is_portable_and_actionable(self) -> None:
        self.assertIn("MENTAL_HEALTH_DATA", self.code)
        self.assertIn('Path("data") / "data.csv"', self.code)
        self.assertIn("FileNotFoundError", self.code)
        self.assertNotRegex(self.code, re.compile(r"(?:[A-Za-z]:[\\/]|/home/|/Users/|/mnt/)"))
        self.assertIn("astype(object).where", self.code)
        self.assertIn("keep_empty_features=True", self.code)

    def test_split_and_preprocessing_are_leakage_safe(self) -> None:
        self.assertIn("stratify=y", self.code)
        self.assertIn("random_state=RANDOM_STATE", self.code)
        self.assertIn("make_preprocessor()", self.code)
        self.assertIn('Pipeline(\n        steps=[("preprocess", make_preprocessor())', self.code)
        self.assertIn('OneHotEncoder(handle_unknown="ignore"', self.code)

    def test_metrics_cover_imbalance_and_probability_quality(self) -> None:
        for metric in (
            "accuracy_score",
            "balanced_accuracy_score",
            "precision_score",
            "recall_score",
            "f1_score",
            "roc_auc_score",
        ):
            self.assertIn(metric, self.code)

    def _loader_namespace(self):
        """Load only notebook definitions, without triggering the external data lookup."""
        if pd is None or np is None:
            self.skipTest("pandas and numpy are required for runtime smoke tests")
        loader_cell = self.code_cells[1]
        tree = ast.parse("".join(loader_cell.get("source", [])))
        namespace = {
            "csv": csv,
            "os": os,
            "Path": Path,
            "np": np,
            "pd": pd,
            "TARGET": "treatment",
            "DROP_COLUMNS": {"Country", "state", "Timestamp", "comments"},
        }
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<loader>", "exec"), namespace)
        return namespace

    def test_loader_normalizes_nullable_strings_and_rejects_empty_features(self) -> None:
        namespace = self._loader_namespace()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "survey.csv"
            pd.DataFrame(
                {
                    "treatment": [" Yes ", "No", "yes", " NO "],
                    "Age": [17, 25, 40, 101],
                    "category": [" a ", None, "b", " a"],
                    "all_missing": [None, None, None, None],
                    "comments": ["private", "private", "private", "private"],
                }
            ).to_csv(path, index=False)
            cleaned = namespace["load_and_clean"](path)

        self.assertNotIn("all_missing", cleaned.columns)
        self.assertNotIn("comments", cleaned.columns)
        self.assertEqual(cleaned["treatment"].tolist(), [1, 0, 1, 0])
        self.assertTrue(pd.isna(cleaned.loc[1, "category"]))
        self.assertTrue(pd.isna(cleaned.loc[0, "Age"]))
        self.assertTrue(pd.isna(cleaned.loc[3, "Age"]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "only_target.csv"
            pd.DataFrame({"treatment": ["Yes", "No"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "at least one"):
                namespace["load_and_clean"](path)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate_columns.csv"
            path.write_text("treatment,feature,feature\nYes,1,2\nNo,3,4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate CSV"):
                namespace["load_and_clean"](path)

    def test_pipeline_smoke_handles_missing_categories(self) -> None:
        if pd is None or np is None:
            self.skipTest("pandas and numpy are required for runtime smoke tests")
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is required for the pipeline smoke test")

        dataset = pd.DataFrame(
            {
                "treatment": np.tile([0, 1], 20),
                "Age": np.arange(20, 60, dtype=float),
                "category": ["a", None, "b", "a"] * 10,
            }
        )
        namespace = {
            "dataset": dataset,
            "TARGET": "treatment",
            "TEST_SIZE": 0.25,
            "RANDOM_STATE": 42,
            "np": np,
            "pd": pd,
        }
        exec("".join(self.code_cells[2].get("source", [])), namespace)
        exec("".join(self.code_cells[3].get("source", [])), namespace)
        self.assertEqual(set(namespace["metrics"].index), {"logistic_regression", "random_forest"})
        self.assertEqual(len(namespace["metrics"]), 2)


if __name__ == "__main__":
    unittest.main()
