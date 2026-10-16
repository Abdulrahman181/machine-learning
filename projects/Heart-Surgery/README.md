# Cardiac Surgery Analysis

This repository contains a reproducible exploratory-analysis notebook and a leakage-safe machine-learning baseline for cardiac-surgery data. It is an educational/research artifact, **not a clinical decision system**.

## Data (not included)

The source CSV is intentionally not committed because of size/licensing constraints. No metrics can be reproduced until a licensed copy is supplied. The notebook expects these columns:

- `Region`, `Procedure`, `Year of Hospital Discharge`
- `Number of Cases`, `Number of Deaths`
- `Expected Mortality Rate`, `Observed Mortality Rate`
- `Comparison Results`

Optional columns such as `Facility ID`, `Hospital Name`, and `Detailed Region` are ignored when present. Set `CARDIAC_SURGERY_DATA` to the CSV path. If it is unset, the notebook checks `data/cardiac-surgery.csv` and raises a clear error when that file is absent. The optional banner path is controlled by `CARDIAC_SURGERY_BANNER` and is never required.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export CARDIAC_SURGERY_DATA=/absolute/path/to/cardiac-surgery.csv
jupyter lab "cardiac-surgery-and-percutaneous (1).ipynb"
```

Run cells from top to bottom. `data_validation.py` validates required columns, coerces the count/rate fields, parses numeric years without the pandas nanosecond-date trap, and rejects impossible counts/rates. The baseline classifier uses a fixed seed (`42`), fits imputation and one-hot encoding inside a scikit-learn `Pipeline` on the training split only, and reports accuracy, balanced accuracy, a classification report, and a confusion matrix. Region/Procedure groups remain together across the holdout; a stratified grouped split is used when feasible, with a deterministic grouped fallback for rare classes.

## Static validation

The notebook is the executable artifact, with `data_validation.py` providing the tested data contract. Run these checks without the dataset:

```bash
python validate_notebook.py
python -m unittest discover -s tests -v
ruff check .
```

These checks parse every code cell with Python AST, verify notebook structure and cleared outputs, and catch obvious regression conditions. They do not replace a full notebook execution with the licensed dataset.

## Limitations

- The dataset and any generated model artifacts are not included, so full execution and numerical results are intentionally not claimed here.
- Metrics depend on the data split, class balance, missingness, and installed library versions.
- The grouped holdout is a single baseline evaluation; it is not temporal validation, cross-validation, or a clinical validation study.
- The validator intentionally rejects non-numeric count/rate values and impossible ranges; inspect and correct source data rather than silently imputing malformed records.
- The model is a baseline for analysis and must not be used for patient-level decisions.
