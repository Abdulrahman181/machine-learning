# Chronic Kidney Disease

This repository contains a reproducible **educational** machine-learning baseline for Chronic Kidney Disease classification. The notebook is intentionally data-free: no patient records or fabricated metrics are committed.

## Data contract

The notebook expects `kidney_disease.csv` in a local data directory. The commonly distributed UCI file may use abbreviated headers (`bp`, `sg`, `class`, etc.); the notebook maps those names to canonical feature names. It accepts the following target labels after trimming: `ckd` and `notckd` (the target is encoded as 1 and 0 respectively).

The dataset is not present in this repository because it is not part of the supplied checkout and may have distribution/licensing constraints. If the file is absent, the notebook stops with an actionable `FileNotFoundError` rather than inventing data or results.

Set the path explicitly:

```bash
export CKD_DATA_DIR=/absolute/path/to/data
# /absolute/path/to/data/kidney_disease.csv must exist
```

The optional `data_description.txt` is not required at runtime because the raw-column mapping is defined in the notebook.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export CKD_DATA_DIR=/absolute/path/to/data
jupyter lab Chronical_kidney_disease.ipynb
```

Run cells from top to bottom. A fitted pipeline is written only when the training cells are run, to `artifacts/kidney_model.joblib` by default. Override it with `CKD_MODEL_PATH` if needed. The artifact is ignored by Git.

## Engineering safeguards

- The test set is created with a fixed seed and stratification before fitting anything.
- Numeric imputation and categorical one-hot encoding are inside a `ColumnTransformer`/`Pipeline`, so hold-out rows do not influence preprocessing.
- Cross-validation runs only on the training partition; the hold-out metrics include ROC AUC, balanced accuracy, F1, precision, recall, and accuracy.
- The notebook contains no fabricated inference row and no checked-in model or metric output.
- `validate_notebook.py` performs dependency-free AST/static checks and rejects stale outputs, hard-coded absolute paths, and the former pre-split feature-selection/random-imputation patterns.

Run the static check from the repository root:

```bash
python validate_notebook.py
```

## Limitations

Metrics depend on the supplied data, class balance, split seed, and package versions. The project has not been fully trained in this checkout because the dataset is unavailable; therefore this review makes no claim about model quality. This notebook is not a medical device and must not be used for diagnosis or treatment decisions.
