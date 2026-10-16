# Hybrid Bank Customer Segmentation (ML + RFM)

This repository contains a reproducible, exploratory customer-segmentation notebook based on bank transaction data. It combines customer-level behavioral features, KMeans/DBSCAN clustering, PCA visualization, and interpretable RFM labels.

## Repository status

The repository contains the notebook and source documentation only; **the transaction dataset is not committed**. The notebook therefore fails fast with an actionable `FileNotFoundError` until a local dataset is supplied. No metric values in this README are presented as verified results.

## Data contract

Provide a CSV with at least these columns:

- `CustomerID`
- `TransactionID`
- `TransactionAmount (INR)`
- `CustAccountBalance`
- `CustomerDOB`
- `TransactionDate`

`CustGender`, `CustLocation`, and `TransactionTime` are optional. Dates are parsed with `dayfirst=True`, and invalid key rows are excluded with counts available in the notebook. The notebook does not invent replacement records or data.

Set the path explicitly (recommended):

```bash
export BANK_TRANSACTIONS_PATH=/absolute/path/to/bank_transactions.csv
```

Alternatively place the file at `bank_transactions.csv` or `data/bank_transactions.csv` relative to the repository. Local data files are ignored by Git.

## Setup and execution

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook Bank_Customer_Segmentation.ipynb
```

The notebook has been cleared of saved outputs and execution counts. Run it only after configuring the dataset path. It uses `RANDOM_STATE = 42`; the DBSCAN sample is drawn with a seeded NumPy generator. The number of PCA components is selected by the 95% variance rule, and KMeans uses a fixed initialization seed and `n_init=10`.

## Method and validation notes

- This is an **unsupervised** workflow: there is no target variable and no valid supervised train/test split. Preprocessing and clustering are fit on the available historical customer population, so silhouette scores are in-sample descriptive diagnostics rather than production generalization estimates.
- Feature engineering is performed at customer level before clustering. No cluster labels are used to construct the features, so target/label leakage is not present.
- If this is promoted to a predictive production service, use a time-based holdout and fit preprocessing/model parameters on the earlier period only; the current notebook does not claim that validation.
- DBSCAN is evaluated on a deterministic sample for runtime control. The notebook reports the sample scope and does not pretend that unsampled rows have DBSCAN labels.
- Silhouette metrics are guarded for the single-cluster/noise-only cases and are not fabricated when undefined.

## Static validation

Run the repository checks without data:

```bash
python scripts/validate_notebook.py
python -m unittest discover -s tests -v
ruff check .
```

These checks parse every notebook code cell with Python AST, reject saved outputs/execution counts and machine-specific `/content` paths, and run Ruff diagnostics. A full notebook execution is intentionally not claimed when the dataset or optional heavy runtime dependencies are unavailable.
