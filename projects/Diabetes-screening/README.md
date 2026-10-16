# Diabetes screening

A small Streamlit inference app for the Pima-style diabetes classifier in
`model.pkl`. The repository contains the application and an exploratory/training
notebook; it does **not** contain the source CSV dataset.

> This model is an educational screening aid, not a medical diagnosis. Do not
> use it for clinical decisions without appropriate validation and oversight.

## Run the app

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run diabetes.py
```

By default the app loads `model.pkl` from the repository directory. To use a model stored elsewhere, set a portable path explicitly:

```bash
DIABETES_MODEL_PATH=/absolute/path/to/model.pkl streamlit run diabetes.py
```

Relative `DIABETES_MODEL_PATH` values are resolved relative to the repository directory rather than Streamlit's current working directory. The model must expose `predict()` and, when the estimator publishes them, its feature count and names must match these columns in this exact order:

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`,
`DiabetesPedigreeFunction`, `Age`.

The app rejects malformed non-binary predictions and displays a clear error for
missing, incompatible, or corrupted artifacts. Pickle files can execute code
while being loaded. Load only a trusted artifact.

**Artifact compatibility warning:** the checked-in `model.pkl` embeds
scikit-learn `1.0.2` metadata and cannot be loaded by the modern scikit-learn
version available in this review environment (the tree serialization schema
changed). It was retained unchanged because the source dataset is absent, so
it cannot be safely retrained or converted here. Retrain/export the model with
the versions in `requirements.txt` before using the app; no inference metrics
are claimed.

## Notebook and data

`diabetes.ipynb` is a cleaned, reproducible template. It expects the original
CSV at `data/diabetes.csv` (or a path supplied through `DIABETES_DATA_PATH`) and
requires a target column named `Outcome`. No dataset or new metrics are claimed
in this repository. If the file is missing, the notebook raises a clear error
instead of silently fabricating data.

The notebook splits data before fitting any preprocessing or balancing step,
uses a stratified fixed random seed, and reports accuracy, recall, F1, and ROC
AUC only after a held-out evaluation. Oversampling, if enabled, is performed
inside the training pipeline and never on the test set.

## Static checks

```bash
python -m pytest -q
ruff check diabetes.py tests
python -m json.tool diabetes.ipynb >/dev/null
```

These checks validate syntax and repository structure; they do not constitute a
full model evaluation. A full run additionally requires the CSV dataset and the
Python dependencies above.
