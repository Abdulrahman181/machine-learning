# Sales Forecasting and Demand Prediction

This repository contains a **leakage-aware, reproducible baseline** for jointly forecasting `Units Sold` and `Demand Forecast`, plus a Streamlit UI for the committed Keras artifact.

## Data is required

The dataset is **not committed**. The pipeline never creates synthetic data or silently falls back to another file. Supply the original Retail Store Inventory Forecasting CSV yourself:

```bash
python "ML flow.py" --data /absolute/path/to/retail_store_inventory.csv \
  --metrics artifacts/metrics.json --artifact artifacts/forecasting.joblib
```

The CSV must contain `Date`, `Units Sold`, and `Demand Forecast`. Other columns are treated as candidate predictors. The loader validates dates and numeric targets, sorts chronologically, and fails with an actionable message when the path or schema is invalid.

## Reproducible training design

- Uses a **chronological holdout** (last 20% of rows) rather than a random split.
- Fits imputation, scaling, and one-hot encoding on the training partition only.
- Removes target columns and known target-derived fields (`Inventory_Demand`, `UnitsSold_Price`) from features.
- Uses a fixed `random_state=42` and reports MAE, RMSE, R², row counts, and train/test date boundaries.
- Does not claim the historical near-perfect scores in older project documents: those results were not reproducible here and the original notebook had leakage and random-split issues.

## Streamlit application

The UI expects five features matching the committed artifact's historical contract: `Inventory Level`, `Price`, `Seasonality_Summer`, `Inventory_Demand`, and `UnitsSold_Price`. The latter two are user-supplied inputs and should be treated as potentially target-derived; retraining with a documented, future-available feature contract is recommended.

```bash
pip install -r requirements.txt
streamlit run App/streamlit_app.py
# Optional artifact override:
MODEL_PATH=/absolute/path/to/model.keras streamlit run App/streamlit_app.py
```

The app resolves the default model relative to `App/`, checks the artifact before loading, loads TensorFlow lazily, validates input/output shapes, and stops with a clear error instead of guessing. The committed artifact has not been executed in this audit because TensorFlow is not installed in the sandbox.

`Deployment` remains a compatibility entry point (`streamlit run Deployment`).

## Notebook

`notebooks/Sales_Forecasting_and_Demand_Prediction.ipynb` is a clean, output-free companion demonstrating the same data checks, chronological split, and train-only preprocessing. It requires a real dataset via `SALES_DATA_PATH` (or `--data` in the CLI); it intentionally stops when the dataset is absent.

## Validation

```bash
ruff check .
python -m unittest discover -s tests -v
python -m compileall -q .
python "ML flow.py"   # expected to stop because no data path was provided
```

The PDFs and BI/archive files are retained as historical deliverables and were not executed or regenerated. Do not commit local datasets, credentials, MLflow runs, or generated model outputs; `.gitignore` covers these classes.
