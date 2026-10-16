import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lung_cancer import (
    DataContractError,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_artifact,
    normalize_columns,
    save_artifact,
    split_dataset,
    validate_dataset,
    validate_feature_values,
)
from validate_repo import NOTEBOOK, validate_notebook, validate_required_files


def make_frame(rows_per_class=7):
    rows = []
    for label, target in (("NO", 0), ("YES", 1)):
        for index in range(rows_per_class):
            row = {column: (index + target) % 2 for column in FEATURE_COLUMNS[1:]}
            row["GENDER"] = "M" if index % 2 == 0 else "F"
            row[TARGET_COLUMN] = label
            rows.append(row)
    return pd.DataFrame(rows)


class RepositoryValidationTests(unittest.TestCase):
    def test_notebook_passes_static_validation(self):
        self.assertEqual(validate_notebook(NOTEBOOK), [])
        self.assertEqual(validate_required_files(), [])

    def test_notebook_has_no_outputs_or_execution_counts(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            self.assertFalse(cell.get("outputs"), "Notebook outputs must stay clean")
            self.assertIsNone(cell.get("execution_count"))

    def test_required_project_files_exist(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("README.md", "requirements.txt", ".gitignore", "lung_cancer.py"):
            self.assertTrue((root / name).is_file(), name)

    def test_normalization_rejects_ambiguous_columns(self):
        frame = pd.DataFrame([[1, 2]], columns=[" age ", "AGE"])
        with self.assertRaises(DataContractError):
            normalize_columns(frame)

    def test_dataset_contract_normalizes_and_encodes(self):
        frame = make_frame()
        frame = frame.rename(columns={"FATIGUE": " FATIGUE "})
        normalized, X, y = validate_dataset(frame, min_class_count=5)
        self.assertEqual(list(X.columns), list(FEATURE_COLUMNS))
        self.assertEqual(set(y), {0, 1})
        self.assertEqual(normalized["FATIGUE"].dtype.kind, "i")

    def test_dataset_contract_rejects_unknown_missing_and_nonfinite_values(self):
        frame = make_frame()
        frame.loc[0, "GENDER"] = "X"
        with self.assertRaises(DataContractError):
            validate_dataset(frame)

        frame = make_frame()
        frame.loc[0, "AGE"] = np.nan
        with self.assertRaises(DataContractError):
            validate_dataset(frame)

        frame = make_frame()
        frame["AGE"] = frame["AGE"].astype(float)
        frame.loc[0, "AGE"] = np.inf
        with self.assertRaises(DataContractError):
            validate_dataset(frame)

    def test_split_is_stratified_and_requires_cv_capacity(self):
        _, X, y = validate_dataset(make_frame(), min_class_count=2)
        first = split_dataset(X, y, random_state=49, cv_splits=5)
        second = split_dataset(X, y, random_state=49, cv_splits=5)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right) if isinstance(left, pd.DataFrame) else pd.testing.assert_series_equal(left, right)
        self.assertEqual(set(first[2]), {0, 1})
        self.assertEqual(set(first[3]), {0, 1})

        _, X_small, y_small = validate_dataset(make_frame(rows_per_class=5))
        with self.assertRaises(DataContractError):
            split_dataset(X_small, y_small, cv_splits=5)

    def test_prediction_contract_rejects_missing_unknown_and_nonfinite_values(self):
        values = {column: 0 for column in FEATURE_COLUMNS}
        values["GENDER"] = "M"
        row = validate_feature_values(values)
        self.assertEqual(row.shape, (1, len(FEATURE_COLUMNS)))
        for bad in ("missing", "extra", "nonfinite"):
            candidate = values.copy()
            if bad == "missing":
                candidate.pop("AGE")
            elif bad == "extra":
                candidate["EXTRA"] = 1
            else:
                candidate["AGE"] = np.inf
            with self.assertRaises(DataContractError):
                validate_feature_values(candidate)

    def test_artifact_round_trip_and_schema_guard(self):
        _, X, y = validate_dataset(make_frame())
        model = LogisticRegression(solver="liblinear", random_state=49).fit(X, y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            saved = save_artifact(model, path)
            loaded = load_artifact(saved)
            self.assertEqual(loaded["feature_columns"], list(FEATURE_COLUMNS))
            self.assertEqual(int(loaded["model"].predict(X.iloc[[0]])[0]), int(model.predict(X.iloc[[0]])[0]))

            bad_path = Path(directory) / "bad.pkl"
            bad = loaded.copy()
            bad["feature_columns"] = ["WRONG"]
            joblib.dump(bad, bad_path)
            with self.assertRaises(DataContractError):
                load_artifact(bad_path)

    def test_empty_csv_is_reported_as_a_contract_error(self):
        from lung_cancer import load_dataset

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(DataContractError):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
