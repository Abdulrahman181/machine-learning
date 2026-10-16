from __future__ import annotations

import unittest

import pandas as pd

from heart_failure_schema import REQUIRED_COLUMNS, validate_dataset
from validate_repository import validate_notebook, validate_python_files


def sample_frame(rows: int = 12) -> pd.DataFrame:
    data = {
        column: [0] * rows for column in REQUIRED_COLUMNS
    }
    data.update(
        {
            "Age": list(range(40, 40 + rows)),
            "RestingBP": [120] * rows,
            "Cholesterol": [200] * rows,
            "FastingBS": [0] * rows,
            "MaxHR": [150] * rows,
            "Oldpeak": [0.5] * rows,
            "Sex": ["M"] * rows,
            "ChestPainType": ["ATA"] * rows,
            "RestingECG": ["Normal"] * rows,
            "ExerciseAngina": ["N"] * rows,
            "ST_Slope": ["Up"] * rows,
            "HeartDisease": [index % 2 for index in range(rows)],
        }
    )
    return pd.DataFrame(data)


class RepositoryTests(unittest.TestCase):
    def test_validates_and_normalizes_numeric_strings(self) -> None:
        frame = sample_frame()
        frame["Age"] = frame["Age"].astype(str)
        normalized = validate_dataset(frame)
        self.assertEqual(normalized["Age"].dtype, "float64")
        self.assertEqual(normalized["HeartDisease"].dtype, "int8")

    def test_rejects_fractional_target_instead_of_truncating(self) -> None:
        frame = sample_frame()
        frame["HeartDisease"] = frame["HeartDisease"].astype(float)
        frame.loc[0, "HeartDisease"] = 0.5
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            validate_dataset(frame)

    def test_rejects_malformed_numeric_feature(self) -> None:
        frame = sample_frame()
        frame["Age"] = frame["Age"].astype(object)
        frame.loc[0, "Age"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Age"):
            validate_dataset(frame)

    def test_static_repository_checks_pass(self) -> None:
        self.assertEqual(validate_notebook(), [])
        self.assertEqual(validate_python_files(), [])


if __name__ == "__main__":
    unittest.main()
