"""Reusable validation and artifact helpers for the lung-cancer experiment.

The module deliberately contains no fallback data and no fabricated model results.
Joblib artifacts must be treated as trusted local files because deserialization can
execute arbitrary Python code.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 49
TARGET_COLUMN = "LUNG_CANCER"
FEATURE_COLUMNS = (
    "GENDER",
    "AGE",
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
)
REQUIRED_COLUMNS = FEATURE_COLUMNS + (TARGET_COLUMN,)
NUMERIC_FEATURE_COLUMNS = FEATURE_COLUMNS[1:]
_ARTIFACT_VERSION = 1


class DataContractError(ValueError):
    """Raised when an input frame or prediction row violates the data contract."""


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical column names and reject ambiguous duplicates."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    normalized = frame.copy()
    names = (
        pd.Index(normalized.columns)
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "_", regex=True)
    )
    if any(not name for name in names):
        raise DataContractError("Column names must not be empty after normalization")
    duplicates = names[names.duplicated()].unique().tolist()
    if duplicates:
        raise DataContractError(f"Duplicate column names after normalization: {duplicates}")
    normalized.columns = names
    return normalized


def _coerce_numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Coerce numeric columns in place while preserving a useful contract error."""
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"{column} must contain numeric values") from exc
    values = frame[list(columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataContractError("Numeric feature columns must contain only finite values")


def validate_dataset(frame: pd.DataFrame, *, min_class_count: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Validate and encode a CSV frame, returning ``(normalized, X, y)``.

    Extra columns are retained in the normalized frame but are never used as model
    features. Missing values, unknown categories, duplicate canonical names, and
    non-finite numeric values are rejected rather than imputed silently.
    """
    dataset = normalize_columns(frame)
    missing = sorted(set(REQUIRED_COLUMNS) - set(dataset.columns))
    if missing:
        raise DataContractError(f"Dataset is missing required columns: {missing}")
    if dataset.empty:
        raise DataContractError("Dataset must contain at least one row")
    required = dataset.loc[:, list(REQUIRED_COLUMNS)]
    if required.isna().any().any():
        counts = required.isna().sum()
        found = counts[counts > 0].to_dict()
        raise DataContractError(f"Dataset contains missing values: {found}")

    dataset["GENDER"] = dataset["GENDER"].astype(str).str.strip().str.upper()
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(str).str.strip().str.upper()
    if not dataset["GENDER"].isin({"M", "F"}).all():
        raise DataContractError("GENDER must contain only M or F")
    if not dataset[TARGET_COLUMN].isin({"YES", "NO"}).all():
        raise DataContractError("LUNG_CANCER must contain only YES or NO")
    dataset["GENDER"] = dataset["GENDER"].map({"M": 0, "F": 1}).astype(int)
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].map({"NO": 0, "YES": 1}).astype(int)
    _coerce_numeric(dataset, NUMERIC_FEATURE_COLUMNS)

    X = dataset.loc[:, list(FEATURE_COLUMNS)].copy()
    y = dataset[TARGET_COLUMN].copy()
    class_counts = y.value_counts().sort_index()
    if len(class_counts) != 2:
        raise DataContractError(
            f"LUNG_CANCER must contain both classes; found {class_counts.to_dict()}"
        )
    if class_counts.min() < min_class_count:
        raise DataContractError(
            f"Each target class needs at least {min_class_count} rows; found {class_counts.to_dict()}"
        )
    return dataset, X, y


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Read and validate an authorized CSV file without inventing fallback data."""
    csv_path = Path(path).expanduser()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path.resolve()}. Provide the authorized CSV "
            "or set LUNG_CANCER_DATA to its path; no training or metrics were produced."
        )
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, UnicodeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DataContractError(f"Could not read CSV dataset at {csv_path}: {exc}") from exc
    return validate_dataset(frame, min_class_count=2)


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    cv_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified split and verify both partitions support the protocol."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_counts = y_train.value_counts()
    test_counts = y_test.value_counts()
    if len(train_counts) != 2 or train_counts.min() < cv_splits:
        raise DataContractError(
            f"Training split needs at least {cv_splits} rows per class for stratified CV; "
            f"found {train_counts.to_dict()}. Supply more rows or change the split."
        )
    if len(test_counts) != 2:
        raise DataContractError(
            f"Held-out test split must contain both classes; found {test_counts.to_dict()}"
        )
    return X_train, X_test, y_train, y_test


def validate_feature_values(
    feature_values: Mapping[str, Any],
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Validate one prediction mapping and return a model-ready one-row frame."""
    if not isinstance(feature_values, Mapping):
        raise TypeError("feature_values must be a mapping")
    missing = sorted(set(feature_columns) - set(feature_values))
    extra = sorted(set(feature_values) - set(feature_columns))
    if missing or extra:
        raise DataContractError(
            f"Feature keys must match the training schema; missing={missing}, extra={extra}"
        )
    row = pd.DataFrame([dict(feature_values)], columns=list(feature_columns))
    if row.isna().any().any():
        raise DataContractError("Prediction features must not be missing")
    row["GENDER"] = row["GENDER"].astype(str).str.strip().str.upper().map({"M": 0, "F": 1})
    if row["GENDER"].isna().any():
        raise DataContractError("Prediction GENDER must be M or F")
    _coerce_numeric(row, tuple(column for column in feature_columns if column != "GENDER"))
    return row


def save_artifact(model: Any, path: str | Path, *, random_state: int = RANDOM_STATE) -> Path:
    """Persist a versioned model envelope only after the caller has fit the model."""
    if not hasattr(model, "predict"):
        raise TypeError("model must provide predict()")
    artifact_path = Path(path).expanduser()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": _ARTIFACT_VERSION,
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "random_state": random_state,
        "selected_by": "5-fold stratified CV on training partition",
    }
    temporary_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    joblib.dump(payload, temporary_path)
    temporary_path.replace(artifact_path)
    return artifact_path


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a trusted versioned model envelope."""
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {artifact_path.resolve()}")
    payload = joblib.load(artifact_path)
    if not isinstance(payload, dict):
        raise DataContractError("Model artifact must contain a dictionary envelope")
    required = {"artifact_version", "model", "feature_columns", "target_column", "random_state"}
    missing = sorted(required - set(payload))
    if missing:
        raise DataContractError(f"Model artifact is missing keys: {missing}")
    if payload["artifact_version"] != _ARTIFACT_VERSION:
        raise DataContractError(
            f"Unsupported model artifact version: {payload['artifact_version']}"
        )
    if tuple(payload["feature_columns"]) != FEATURE_COLUMNS:
        raise DataContractError("Model artifact feature schema does not match the current schema")
    if payload["target_column"] != TARGET_COLUMN or not hasattr(payload["model"], "predict"):
        raise DataContractError("Model artifact contains an incompatible model envelope")
    return payload
