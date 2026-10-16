"""Leakage-safe training utilities for the hazardous NEO classifier.

The repository intentionally does not include the Kaggle CSV.  Call
``run_experiment`` with ``NEO_DATA_PATH`` set to a local copy of that file.
"""

from __future__ import annotations

import os
import math
import numbers
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "is_hazardous"
IDENTIFIER_COLUMNS = ("neo_id", "name", "orbiting_body")
REQUIRED_FEATURES = (
    "absolute_magnitude",
    "estimated_diameter_min",
    "estimated_diameter_max",
    "relative_velocity",
    "miss_distance",
)
DEFAULT_RANDOM_STATE = 42


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    """Return a usable dataset path or fail with an actionable message."""
    explicit_path = data_path if data_path is not None else os.environ.get("NEO_DATA_PATH")
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"NEO data file not found at {path}. "
                "Download the Kaggle CSV and set NEO_DATA_PATH to its path."
            )
        return path

    repo_root = Path(__file__).resolve().parent
    candidates = (
        repo_root / "data" / "nearest-earth-objects-1910-2024.csv",
        repo_root / "nearest-earth-objects-1910-2024.csv",
        repo_root / "data" / "neo.csv",
        repo_root / "neo.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "NEO dataset is not included in this repository. "
        "Download the Kaggle CSV referenced in NASA_Data.txt and either "
        "place it under data/nearest-earth-objects-1910-2024.csv or set "
        "NEO_DATA_PATH. No model training was run."
    )


def _normalise_target(target: pd.Series) -> pd.Series:
    """Convert common binary target representations to integer 0/1 values."""
    if target.isna().any():
        raise ValueError(
            f"{TARGET_COLUMN!r} contains missing values; remove or label them before training."
        )
    if pd.api.types.is_bool_dtype(target):
        return target.astype("int64")
    if pd.api.types.is_numeric_dtype(target):
        if not target.map(math.isfinite).all():
            raise ValueError(f"{TARGET_COLUMN!r} must contain only finite values.")
        values = set(target.dropna().unique())
        if not values.issubset({0, 1}):
            raise ValueError(
                f"{TARGET_COLUMN!r} must be binary 0/1 or boolean; found {sorted(values)}."
            )
        return target.astype("int64")

    normalised = target.astype("string").str.strip().str.lower()
    mapped = normalised.map(
        {"0": 0, "1": 1, "false": 0, "true": 1, "no": 0, "yes": 1}
    )
    if mapped.isna().any():
        unknown = sorted(normalised[mapped.isna()].dropna().unique().tolist())
        raise ValueError(
            f"{TARGET_COLUMN!r} contains unsupported values: {unknown}. "
            "Use binary 0/1 or boolean values."
        )
    return mapped.astype("int64")


def load_dataset(data_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load, validate, and minimally clean the source CSV.

    Identifier/text columns are excluded before modelling.  Feature imputation
    is deliberately performed inside each estimator pipeline after the split.
    """
    path = resolve_data_path(data_path)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pandas exposes several parser/IO exception types
        raise ValueError(f"Unable to read NEO CSV at {path}: {exc}") from exc

    required = (*REQUIRED_FEATURES, TARGET_COLUMN)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing required columns: {missing}. "
            f"Expected at least {list(required)}."
        )

    frame = frame.drop_duplicates().copy()
    target = _normalise_target(frame.pop(TARGET_COLUMN))
    feature_frame = frame.drop(
        columns=[column for column in IDENTIFIER_COLUMNS if column in frame.columns]
    )
    if feature_frame.empty:
        raise ValueError("No usable feature columns remain after removing identifiers.")

    non_numeric = [
        column
        for column in feature_frame.columns
        if not pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    if non_numeric:
        raise ValueError(
            f"Non-numeric feature columns are not supported: {non_numeric}. "
            "Remove or encode them before training."
        )
    non_finite = [
        column
        for column in feature_frame.columns
        if (~feature_frame[column].dropna().map(math.isfinite)).any()
    ]
    if non_finite:
        raise ValueError(
            f"Feature columns contain non-finite values: {non_finite}. "
            "Replace +/-inf with a missing value or remove those rows."
        )
    all_missing = [column for column in feature_frame if feature_frame[column].notna().sum() == 0]
    if all_missing:
        raise ValueError(
            f"Feature columns contain no observed values: {all_missing}. "
            "Provide at least one finite value per feature."
        )
    if target.nunique() < 2:
        raise ValueError(f"{TARGET_COLUMN!r} must contain both classes for training.")
    return feature_frame, target


def _preprocessor(feature_names: list[str], *, scale: bool) -> ColumnTransformer:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps)
    return ColumnTransformer(
        [("numeric", numeric_pipe, feature_names)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(
    feature_names: list[str], *, random_state: int = DEFAULT_RANDOM_STATE
) -> dict[str, Pipeline]:
    """Build estimators whose preprocessing is fitted only on training data."""
    if not feature_names:
        raise ValueError("At least one feature is required to build models.")
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor(feature_names, scale=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _preprocessor(feature_names, scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        class_weight="balanced",
                        n_estimators=300,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def _metrics(y_true: pd.Series, y_pred: Any, y_probability: Any) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
    }


def run_experiment(
    data_path: str | Path | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train both baseline models and return test metrics without persistence."""
    X, y = load_dataset(data_path)
    if isinstance(random_state, bool) or not isinstance(random_state, numbers.Integral):
        raise ValueError("random_state must be an integer for reproducible experiments.")
    if isinstance(test_size, bool) or not isinstance(test_size, numbers.Real):
        raise ValueError("test_size must be a number between 0 and 1.")
    if not 0 < float(test_size) < 1:
        raise ValueError("test_size must be strictly between 0 and 1.")
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Each class needs at least two rows for a stratified train/test split."
        )
    n_test = math.ceil(len(X) * float(test_size))
    n_train = len(X) - n_test
    if n_test < 2 or n_train < 2:
        raise ValueError(
            "test_size must leave at least two rows in both train and test splits."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Both train and test splits must contain both classes; adjust test_size."
        )

    results: dict[str, Any] = {
        "rows_after_deduplication": len(X),
        "features": list(X.columns),
        "class_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "test_size": float(test_size),
        "random_state": random_state,
        "metrics": {},
    }
    for name, estimator in build_models(
        list(X.columns), random_state=int(random_state)
    ).items():
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        probabilities = estimator.predict_proba(X_test)[:, 1]
        results["metrics"][name] = _metrics(y_test, predictions, probabilities)
    return results
