"""Schema and quality checks for the heart-failure classification dataset."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

TARGET_COLUMN = "HeartDisease"
NUMERIC_COLUMNS = (
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
)
CATEGORICAL_COLUMNS = (
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
)
REQUIRED_COLUMNS = frozenset((*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, TARGET_COLUMN))


def _invalid_column_values(values: pd.Series) -> pd.Series:
    """Return a mask for non-null values that cannot be parsed as numbers."""
    parsed = pd.to_numeric(values, errors="coerce")
    return values.notna() & parsed.isna()


def _format_columns(columns: Iterable[str]) -> str:
    return ", ".join(sorted(columns))


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a dataset without fitting any model state.

    Numeric feature values may be numbers or numeric strings, and missing feature
    values are retained for the pipeline imputers. Non-numeric non-null values,
    duplicate columns, missing required columns, and malformed binary targets are
    rejected instead of being silently reinterpreted.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dataset must be a pandas DataFrame.")
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].unique().tolist()
        raise ValueError(f"Dataset contains duplicate column names: {duplicates}")

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {_format_columns(missing_columns)}")
    if frame.empty:
        raise ValueError("Dataset is empty; provide a non-empty heart.csv file.")

    normalized = frame.copy()
    target_values = pd.to_numeric(normalized[TARGET_COLUMN], errors="coerce")
    invalid_target = (
        normalized[TARGET_COLUMN].isna()
        | target_values.isna()
        | target_values.mod(1).ne(0)
        | ~target_values.isin((0, 1))
    )
    if invalid_target.any():
        examples = normalized.loc[invalid_target, TARGET_COLUMN].head(3).tolist()
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} must contain only binary 0/1 values; "
            f"invalid examples: {examples}"
        )
    normalized[TARGET_COLUMN] = target_values.astype("int8")

    for column in NUMERIC_COLUMNS:
        if _invalid_column_values(normalized[column]).any():
            raise ValueError(f"Numeric column {column!r} contains non-numeric non-null values.")
        parsed = pd.to_numeric(normalized[column], errors="coerce")
        if parsed.notna().sum() == 0:
            raise ValueError(f"Numeric column {column!r} contains no usable values.")
        normalized[column] = parsed.astype("float64")

    for column in CATEGORICAL_COLUMNS:
        if normalized[column].notna().sum() == 0:
            raise ValueError(f"Categorical column {column!r} contains no usable values.")
        # A stable string representation prevents mixed int/string categories
        # from failing inside OneHotEncoder while preserving missing values.
        normalized[column] = normalized[column].astype("string")

    class_counts = normalized[TARGET_COLUMN].value_counts()
    if len(class_counts) != 2:
        raise ValueError("Target column must contain both classes 0 and 1.")
    if class_counts.min() < 4:
        raise ValueError(
            "Each target class needs at least 4 rows for the stratified train/validation/test split."
        )
    return normalized
