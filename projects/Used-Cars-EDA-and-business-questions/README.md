# Used Cars EDA and Business Questions

This repository contains a reproducible exploratory analysis of Craigslist used-car listings. The notebook answers descriptive questions about vehicle type, title status, condition, manufacture year, mileage/price, and fuel type.

## Data

The source dataset is **not committed** because it is large and may contain listing URLs, free-text descriptions, and other identifying or unstable fields. Obtain the authorized `vehicles.csv` file separately, then either:

- place it at `data/vehicles.csv` (the recommended local layout), or
- set `USED_CARS_DATA` to the CSV path before starting Jupyter.

The notebook stops with a clear `FileNotFoundError` when the data is absent; it does not create synthetic data or claim results without the source file. It validates the required columns before analysis.

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook vehicles-EDA.ipynb
```

For a non-default location:

```bash
USED_CARS_DATA=/path/to/vehicles.csv jupyter notebook vehicles-EDA.ipynb
```

## Repository layout

```text
.
├── vehicles-EDA.ipynb
├── tests/test_repository.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── ENGINEERING_REVIEW.md
```

## Data quality and interpretation

The notebook coerces only numeric/time fields used by the charts, treats mixed date formats and non-finite numeric values safely, reports missingness, and excludes invalid values from the relevant comparison rather than filling them with global means. Duplicate non-missing listing IDs are reduced to the first occurrence while rows without IDs are retained. Manufacture years are restricted to integer values in a plausible historical range. There is no predictive model or train/test split in this repository; if modeling is added, preprocessing must be fitted on the training split only to avoid leakage.

## Validation

Run the repository checks from the project root:

```bash
pytest -q
ruff check tests
```

The tests validate notebook JSON, Python syntax, absence of stale notebook outputs, portability of the data-loading path, and execution against a small malformed-value fixture. They do not replace running the full notebook with the separately obtained dataset.

## Author

Abdul Rahman Ahmed — <abdulrahmannassar@gmail.com>
