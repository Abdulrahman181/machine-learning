# Black Friday Sales Prediction

A reproducible scikit-learn baseline for predicting the `Purchase` amount in the Black Friday dataset.

## What is included

- `Black Friday Sales Prediction.ipynb`: leakage-aware exploratory checks, grouped holdout evaluation, and Decision Tree/Random Forest baselines.
- `requirements.txt`: runtime dependencies.
- `tests/test_static_validation.py`: dependency-light checks for notebook structure and portability.
- `ENGINEERING_REVIEW.md`: engineering audit and validation record.

## Data setup

The dataset is **not committed** to this repository. Download the Kaggle Black Friday dataset separately and either:

1. place the file at `data/Black Friday Dataset.csv`, or
2. set `BLACK_FRIDAY_DATA` to the CSV path before opening/running the notebook.

Example:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export BLACK_FRIDAY_DATA=/path/to/Black\ Friday\ Dataset.csv
jupyter notebook "Black Friday Sales Prediction.ipynb"
```

If the file is missing, the first configuration cell raises a clear `FileNotFoundError` with the checked path. The notebook also rejects empty/duplicate-column data, invalid or non-finite targets, missing customer groups, insufficient groups for a split, unsupported feature dtypes, and empty partitions before fitting. No model run or score is claimed without the data.

## Methodology

- `Purchase` is the regression target.
- `User_ID` and `Product_ID` are excluded as identifiers.
- `GroupShuffleSplit` keeps customers out of both train and test partitions, reducing customer-level leakage from repeated rows.
- Imputation and one-hot encoding are fitted inside a scikit-learn `Pipeline` on training data only.
- Metrics are **MAE**, **RMSE**, and **R²**. Optional grouped three-fold cross-validation is disabled by default because it can be expensive on the full dataset.
- All random operations use seed `42`.

The repository intentionally does not contain generated predictions, cached outputs, or fabricated benchmark scores. See `ENGINEERING_REVIEW.md` for the audit details and the limitations of the available validation.

## Static validation

Run the repository checks without the dataset (the tests include notebook schema, leakage, cleanliness, and edge-case guard checks):

```bash
python -m unittest discover -s tests -v
ruff check tests
```

A full notebook execution requires the dataset and the packages in `requirements.txt`.
