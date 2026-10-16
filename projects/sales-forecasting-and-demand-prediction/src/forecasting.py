"""Leakage-aware, reproducible sales and demand forecasting pipeline.

The module deliberately requires a real CSV supplied by the caller; it never
creates fallback or synthetic data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGETS = ("Units Sold", "Demand Forecast")
DATE_COLUMN = "Date"
LEAKY_FEATURES = {
    "units sold",
    "demand forecast",
    "inventory_demand",
    "unitssold_price",
}
REQUIRED_COLUMNS = frozenset((*TARGETS, DATE_COLUMN))


def load_dataset(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load and validate a real CSV, failing clearly when it is unavailable."""
    data_path = Path(path).expanduser()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Provide --data PATH or "
            "set SALES_DATA_PATH; no data was generated."
        )
    if data_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV dataset, got: {data_path}")
    frame = pd.read_csv(data_path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty; training was not attempted.")
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="coerce")
    if frame[DATE_COLUMN].isna().any():
        raise ValueError("Dataset contains invalid or missing Date values.")
    for column in TARGETS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"Target column {column!r} contains non-numeric/missing values.")
    return frame.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only information available before the target date."""
    features = frame.drop(columns=list(TARGETS)).copy()
    date = pd.to_datetime(features.pop(DATE_COLUMN))
    features["date_year"] = date.dt.year
    features["date_month"] = date.dt.month
    features["date_dayofweek"] = date.dt.dayofweek
    features["date_day"] = date.dt.day
    forbidden = {
        str(column).strip().lower().replace(" ", "_")
        for column in features.columns
        if str(column).strip().lower().replace(" ", "_") in LEAKY_FEATURES
    }
    if forbidden:
        features = features.drop(columns=[
            column for column in features.columns
            if str(column).strip().lower().replace(" ", "_") in forbidden
        ])
    return features


def build_pipeline(features: pd.DataFrame) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    if not transformers:
        raise ValueError("No usable feature columns remain after validation.")
    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers=transformers)),
        ("regressor", ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_and_evaluate(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[Any, dict[str, Any]]:
    """Use a chronological holdout and fit preprocessing on training rows only."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    split = int(len(frame) * (1 - test_fraction))
    if split < 2 or len(frame) - split < 1:
        raise ValueError("Dataset needs enough rows for a chronological train/test split.")
    train, test = frame.iloc[:split], frame.iloc[split:]
    x_train = prepare_features(train)
    x_test = prepare_features(test)
    y_train = train[list(TARGETS)]
    y_test = test[list(TARGETS)]
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model = build_pipeline(x_train)
    model.fit(x_train, y_train)
    predictions = np.asarray(model.predict(x_test))
    metrics: dict[str, Any] = {
        "rows": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_end": train[DATE_COLUMN].max().date().isoformat(),
        "test_start": test[DATE_COLUMN].min().date().isoformat(),
        "random_state": 42,
        "targets": {},
    }
    for index, target in enumerate(TARGETS):
        actual = y_test[target].to_numpy()
        predicted = predictions[:, index]
        metrics["targets"][target] = {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
            "r2": float(r2_score(actual, predicted)),
        }
    return model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.getenv("SALES_DATA_PATH"))
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if not args.data:
        parser.error("--data PATH is required (or set SALES_DATA_PATH); no data was generated.")
    model, metrics = train_and_evaluate(load_dataset(args.data))
    print(json.dumps(metrics, indent=2))
    if args.artifact:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError(
                "joblib is required only when --artifact is requested; install requirements.txt."
            ) from exc
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.artifact)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
