# Mental Health Prediction

This repository contains a reproducible baseline notebook for predicting the `treatment` response in the OSMI Mental Health in Tech survey. It is an educational/research artifact, **not** a medical decision system.

## Data

The dataset is intentionally not committed because it is not part of this repository and may have licensing/privacy constraints. The notebook expects a CSV with a `treatment` column containing only `Yes`/`No` (case and surrounding whitespace are normalized). The other survey columns are used as features when present.

Set `MENTAL_HEALTH_DATA` to an existing CSV file. If it is unset, the notebook looks for `data/data.csv` relative to the repository root:

```bash
export MENTAL_HEALTH_DATA=/absolute/path/to/data.csv
```

If the file is missing, empty, malformed, has duplicate/non-string columns, lacks `treatment`, contains invalid target values, or has no usable feature after cleaning, execution stops with an explicit error. Categorical missing values are normalized to scikit-learn-compatible `NaN` values; no data or metrics are fabricated in this repository.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook "Mental Health Prediction.ipynb"
```

Run cells from top to bottom. The notebook removes free-text/location/timestamp columns when present, drops all-missing feature columns, stratifies a fixed holdout split, and fits imputation, scaling, and one-hot encoding **inside each model pipeline**. This prevents preprocessing statistics from being learned from the test set. Models use fixed random seeds.

Reported holdout metrics include accuracy, balanced accuracy, precision, recall, F1, and ROC AUC. They are dataset- and split-dependent. The notebook reports both baselines and uses a pre-declared reference model; production model selection needs a validation scheme and a separately untouched test set.

## Validation

The repository includes static checks that parse every notebook code cell, check for syntax errors and stale notebook outputs, reject common non-portable absolute paths, and run a small synthetic loader/pipeline smoke test when runtime ML dependencies are installed:

```bash
python -m unittest discover -s tests -v
ruff check --select E9,F63,F7,F82 tests
```

The validation suite does not train a model. A full training run requires the external CSV and the packages in `requirements.txt`.

## Privacy and limitations

Do not commit datasets, generated models, or notebook caches. Predictions must not be used to diagnose, triage, or make decisions about a person. Validate schema, class balance, data provenance, and fairness for any legitimate research use.
