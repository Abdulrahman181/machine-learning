"""Streamlit UI for the diabetes classifier artifact.

The model path is configurable with ``DIABETES_MODEL_PATH`` and defaults to
``model.pkl`` next to this file.  The application intentionally does not train
or evaluate a model at request time.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from diabetes_contracts import (
    FEATURES,
    resolve_model_path,
    validate_input_frame,
    validate_model_contract,
    validate_prediction,
)


def model_path() -> Path:
    """Return a model path independent of Streamlit's current working directory."""
    return resolve_model_path(os.getenv("DIABETES_MODEL_PATH"), Path(__file__).resolve().parent)


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> Any:
    """Load and minimally validate a trusted, local pickle model artifact."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact}. "
            "Set DIABETES_MODEL_PATH or place model.pkl beside diabetes.py."
        )
    with artifact.open("rb") as handle:
        model = pickle.load(handle)
    return validate_model_contract(model)


def input_features() -> pd.DataFrame:
    """Collect one row of numeric features in the exact training-column order."""
    values = {
        "Pregnancies": st.number_input("Pregnancies", min_value=0.0, value=1.0, step=1.0),
        "Glucose": st.number_input("Glucose", min_value=0.0, value=120.0, step=1.0),
        "BloodPressure": st.number_input("BloodPressure", min_value=0.0, value=70.0, step=1.0),
        "SkinThickness": st.number_input("SkinThickness", min_value=0.0, value=20.0, step=1.0),
        "Insulin": st.number_input("Insulin", min_value=0.0, value=80.0, step=1.0),
        "BMI": st.number_input("BMI", min_value=0.0, value=30.0, step=0.1),
        "DiabetesPedigreeFunction": st.number_input(
            "DiabetesPedigreeFunction", min_value=0.0, value=0.5, step=0.001, format="%.3f"
        ),
        "Age": st.number_input("Age", min_value=0.0, value=30.0, step=1.0),
    }
    return validate_input_frame(pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES))


def main() -> None:
    """Render the prediction page."""
    st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
    st.title("Diabetes Prediction Web App")
    st.info("Enter clinical measurements to obtain a model prediction. This is not a diagnosis.")

    try:
        model = load_model(str(model_path()))
    except (
        AttributeError,
        EOFError,
        FileNotFoundError,
        ImportError,
        ModuleNotFoundError,
        OSError,
        pickle.UnpicklingError,
        TypeError,
        ValueError,
    ) as exc:
        st.error(
            f"Unable to load a compatible model from {model_path()}: {exc} "
            "Use a trusted artifact trained with a compatible scikit-learn version."
        )
        st.stop()

    features = input_features()
    if st.button("Predict", type="primary"):
        try:
            prediction = validate_prediction(model.predict(features))
        except Exception as exc:  # Streamlit boundary: show model errors without crashing the page.
            st.error(f"Prediction failed; verify model compatibility with the {len(FEATURES)} features: {exc}")
            st.stop()
        label = "positive (model predicts diabetes)" if prediction == 1 else "negative (model predicts no diabetes)"
        st.success(f"Prediction: {label}")


if __name__ == "__main__":
    main()
