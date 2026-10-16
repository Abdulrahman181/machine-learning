"""Dependency-light contracts shared by the Streamlit app and tests."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any, Sequence

FEATURES = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)


def resolve_model_path(configured: str | None, base_dir: Path) -> Path:
    """Resolve an optional model path consistently, independent of process cwd."""
    value = configured.strip() if configured else ""
    path = Path(value).expanduser() if value else base_dir / "model.pkl"
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def validate_model_contract(model: Any) -> Any:
    """Validate the model properties that the inference UI relies on."""
    if not callable(getattr(model, "predict", None)):
        raise TypeError("The model artifact does not expose a callable predict method.")

    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None and int(feature_count) != len(FEATURES):
        raise TypeError(
            f"The model expects {feature_count} features; this app supplies {len(FEATURES)}."
        )

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and tuple(feature_names) != FEATURES:
        raise TypeError(
            "The model feature names/order do not match the app contract: "
            f"expected {FEATURES}, got {tuple(feature_names)}."
        )
    return model


def validate_input_frame(frame: Any) -> Any:
    """Validate the one-row input frame before sending it to the artifact."""
    if tuple(getattr(frame, "columns", ())) != FEATURES:
        raise ValueError(f"Input columns must be exactly {FEATURES} in this order.")
    if getattr(frame, "shape", None) != (1, len(FEATURES)):
        raise ValueError(f"Input must contain exactly one row and {len(FEATURES)} features.")
    return frame


def validate_prediction(predictions: Sequence[Any]) -> int:
    """Return a strict binary prediction, rejecting malformed model outputs."""
    try:
        values = list(predictions)
    except TypeError as exc:
        raise ValueError("The model returned a non-iterable prediction.") from exc
    if len(values) != 1:
        raise ValueError(f"The model must return exactly one prediction, got {len(values)}.")
    value = values[0]
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in (0, 1):
        raise ValueError(f"The model returned invalid class label {value!r}; expected 0 or 1.")
    return int(value)
