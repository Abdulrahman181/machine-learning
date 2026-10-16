# Lung Cancer Classification

This repository contains a reproducible scikit-learn notebook and reusable validation helpers for binary lung-cancer classification from tabular risk-factor data.

## Data contract

The notebook expects a CSV file at `data.csv` by default. Set `LUNG_CANCER_DATA` to use another path. The file must contain these columns (column names are normalized by trimming whitespace, upper-casing, and replacing whitespace with underscores):

- `GENDER`, `AGE`, `SMOKING`, `YELLOW_FINGERS`, `ANXIETY`, `PEER_PRESSURE`, `CHRONIC_DISEASE`
- `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL_CONSUMING`, `COUGHING`, `SHORTNESS_OF_BREATH`
- `SWALLOWING_DIFFICULTY`, `CHEST_PAIN`, and target `LUNG_CANCER`

`GENDER` must contain `M`/`F` and `LUNG_CANCER` must contain `YES`/`NO`. Missing values, non-finite values, duplicate canonical column names, and unknown categorical values are rejected rather than silently imputed or fabricated. Extra columns are ignored by the model after validation.

The dataset is intentionally **not committed**. If it is absent, execution stops with an actionable `FileNotFoundError`; no metrics or sample predictions are invented.

## Setup and use

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# Put the authorized dataset at data.csv, or set LUNG_CANCER_DATA=/path/to/data.csv
jupyter nbconvert --execute --to notebook --inplace "Lung Cancer.ipynb"
```

The notebook creates `artifacts/lc.pkl` by default, or the path in `LUNG_CANCER_ARTIFACT`, only after successful training. The artifact is a trusted joblib pickle and must not be loaded from an untrusted source. It includes a version and feature-schema envelope; `lung_cancer.load_artifact` rejects incompatible artifacts.

It uses a fixed-seed stratified hold-out test set. Model selection is performed by seeded stratified cross-validation on the training partition; the test partition is evaluated once at the end. The split helper checks that both partitions and all CV folds have the required classes. Reported metrics are therefore only available when the real dataset is supplied.

The reusable helpers in `lung_cancer.py` expose `validate_dataset`, `split_dataset`, `validate_feature_values`, `save_artifact`, and `load_artifact` for callers that do not use the notebook.

## Validation

Run the repository checks without data or heavyweight ML execution:

```bash
python validate_repo.py
python -m unittest discover -s tests -v
```

These checks parse every notebook code cell with Python AST, reject stale notebook outputs and execution counts, flag local absolute paths, exercise the schema/split/prediction/artifact contracts, and verify required files. They do not claim a completed training run.
