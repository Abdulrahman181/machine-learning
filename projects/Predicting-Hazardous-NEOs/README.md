# Predicting Hazardous Near-Earth Objects

This repository contains a reproducible scikit-learn baseline for classifying whether a near-Earth object (NEO) is marked hazardous in the NASA/Kaggle source dataset referenced by [`NASA_Data.txt`](NASA_Data.txt).

> **Data and results status:** The CSV is not committed to this repository, and no model metrics are committed. Training stops with an actionable error when the CSV is absent; this audit therefore does not claim a completed training run.

## Repository contents

- `neos_pipeline.py` — validated data loading, leakage-safe preprocessing, two deterministic baseline pipelines, and evaluation.
- `Project_NEOs.ipynb` — a clean, output-free walkthrough that calls the same pipeline.
- `tests/test_engineering.py` — static/notebook and missing-data guard tests.
- `NASA_Data.txt` — the original source URL; it is not the dataset itself.
- `ENGINEERING_REVIEW.md` — audit findings and validation record.

## Requirements

Python 3.10+ is supported. Install the pinned dependency ranges in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Obtain the data

Download the CSV from the source URL in `NASA_Data.txt` (the Kaggle dataset commonly named `nearest-earth-objects(1910-2024).csv`). Do not commit the downloaded data. Either set an explicit path:

```bash
export NEO_DATA_PATH=/absolute/path/to/nearest-earth-objects-1910-2024.csv
```

or place it at `data/nearest-earth-objects-1910-2024.csv` (the `data/` directory is ignored by Git).

 The loader requires these columns: `absolute_magnitude`, `estimated_diameter_min`, `estimated_diameter_max`, `relative_velocity`, `miss_distance`, and binary `is_hazardous`. Identifier columns (`neo_id`, `name`, and `orbiting_body`) are excluded from modelling. Missing feature values may be imputed by the training pipelines; missing targets, infinite values, and all-missing features are rejected.

## Run

The command below trains the logistic-regression and random-forest baselines and prints a JSON result. It performs no run when the data file is missing; instead it explains how to provide it.

```bash
python -c "from neos_pipeline import run_experiment; import json; print(json.dumps(run_experiment(), indent=2))"
```

To open the notebook:

```bash
jupyter notebook Project_NEOs.ipynb
```

## Methodology safeguards

- A stratified, fixed-seed train/test split is created before any learned preprocessing. The split contract rejects invalid proportions and datasets too small to retain both classes in both partitions.
- Median imputation and (for logistic regression) standardisation are fitted inside scikit-learn `Pipeline` objects on training data only.
- Class imbalance is handled with `class_weight='balanced'`; oversampling is not applied to the full dataset.
- Metrics include accuracy, balanced accuracy, precision, recall, F1, and ROC AUC. No metric is reported here because the source CSV is not present in this checkout.
- Random seeds are explicit (`42` by default) for the split and estimators; a caller-provided integer is propagated to both. Reproducibility still depends on library versions and hardware.

## Validation

Run static checks and tests from the repository root:

```bash
ruff check .
pytest -q
```

The notebook is intentionally saved without stale execution outputs. Tests verify that every code cell parses, that missing data and invalid schema values fail clearly, and that a small synthetic fixture exercises both model pipelines. A full model run requires the real CSV and installed runtime dependencies.

## Limitations

This is a baseline classification experiment, not a planetary-defence decision system. The source labels, feature definitions, temporal coverage, and possible duplicate observations should be independently verified before scientific or operational use. The current split is stratified and row-based; it does not claim object-grouped or time-aware generalisation when repeated objects or temporal drift are present. A future study should use a time-aware or object-grouped holdout and tune thresholds using a validation set rather than the test set. No serialized model or production inference API is provided.
