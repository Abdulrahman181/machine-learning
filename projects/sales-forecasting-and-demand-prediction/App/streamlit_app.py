"""Streamlit UI for the committed two-output Keras model.

The model and its feature contract are external runtime dependencies. The app
fails with an actionable message instead of silently using a different file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:  # Allows static checks without installing the UI dependency.
    st = None  # type: ignore[assignment]

FEATURE_COLUMNS = (
    "Inventory Level",
    "Price",
    "Seasonality_Summer",
    "Inventory_Demand",
    "UnitsSold_Price",
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "best_demand_forecast_model.keras"


def _require_streamlit() -> Any:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install requirements.txt before starting the app."
        )
    return st


def load_model(model_path: Path) -> Any:
    """Load the artifact lazily and cache only a verified existing path."""
    ui = _require_streamlit()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Supply MODEL_PATH; "
            "the app will not invent a model."
        )

    @ui.cache_resource
    def _cached(path: str) -> Any:
        try:
            from tensorflow.keras.models import load_model as keras_load_model
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is not installed. Install requirements.txt to load the model."
            ) from exc
        return keras_load_model(path)

    return _cached(str(model_path))


class DemandForecaster:
    """Validate the feature contract and return the model's two outputs."""

    def __init__(self, model_path: Path) -> None:
        self.model = load_model(model_path)

    def predict(self, input_data: pd.DataFrame) -> tuple[int, int]:
        missing = [column for column in FEATURE_COLUMNS if column not in input_data]
        if missing:
            raise ValueError(f"Prediction input is missing columns: {missing}")
        values = input_data.loc[:, FEATURE_COLUMNS]
        expected = getattr(self.model, "input_shape", None)
        if expected and expected[-1] != len(FEATURE_COLUMNS):
            raise ValueError(
                "Model/input contract mismatch: model expects "
                f"{expected[-1]} features, UI provides {len(FEATURE_COLUMNS)}."
            )
        prediction = self.model.predict(values, verbose=0)
        if getattr(prediction, "shape", None) != (1, 2):
            raise ValueError(
                f"Expected one prediction with two outputs, got shape {getattr(prediction, 'shape', None)}."
            )
        return round(float(prediction[0][0])), round(float(prediction[0][1]))


def create_sidebar_inputs() -> pd.DataFrame:
    ui = _require_streamlit()
    with ui.sidebar:
        ui.header("Model parameters")
        inventory_level = ui.number_input("Inventory Level", min_value=0.0, value=0.0)
        price = ui.number_input("Price (USD)", min_value=0.0, value=0.0, format="%.2f")
        seasonality = ui.selectbox("Summer Season", options=[0, 1])
        inventory_demand = ui.number_input("Demand input", min_value=0.0, value=0.0)
        units_sold_price = ui.slider(
            "Units-sold price feature", min_value=0.0, max_value=10.0, value=0.0, step=0.5
        )
    return pd.DataFrame({
        "Inventory Level": [inventory_level],
        "Price": [price],
        "Seasonality_Summer": [seasonality],
        "Inventory_Demand": [inventory_demand],
        "UnitsSold_Price": [units_sold_price],
    })


def main() -> None:
    ui = _require_streamlit()
    ui.set_page_config(page_title="Sales & Demand Forecasting", page_icon="📈")
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
    try:
        forecaster = DemandForecaster(model_path)
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        ui.error(str(exc))
        ui.stop()
    inputs = create_sidebar_inputs()
    ui.subheader("Input parameters")
    ui.dataframe(inputs, hide_index=True, use_container_width=True)
    if ui.button("Run forecast", type="primary"):
        try:
            units_sold, demand = forecaster.predict(inputs)
        except (ValueError, RuntimeError, OSError) as exc:
            ui.error(str(exc))
            return
        left, right = ui.columns(2)
        left.metric("Units Sold", f"{units_sold:,}")
        right.metric("Demand Forecast", f"{demand:,}")


if __name__ == "__main__":
    main()
