# Engineering Review

**Repository:** `Abdulrahman181/sales-forecasting-and-demand-prediction`  
**Review date:** 2026-10-14  
**Scope:** Python files, Jupyter notebook, README, requirements/configuration, Streamlit artifact boundary, static validation, and lightweight tests.

## Executive summary

The repository was not runnable as committed. The main ML-flow script depended on undefined notebook globals (`Xf_train`, `Yt_train`, `Xf_test`, `Yt_test`), used `xgb` without importing it, pointed MLflow at a local service, and mixed model implementations and indentation. The notebook contained Windows-only paths, a missing-image dependency, an AST syntax error caused by notebook magic when treated as ordinary Python, random train/test splits for time-ordered data, preprocessing fitted before the split, and target-derived features. The app used a working-directory-relative model path and did not verify its model/input/output contract.

A small, reproducible baseline and a clean notebook were added. No data was present in the repository, so **no model was retrained and no metrics were invented**. TensorFlow, Streamlit, scikit-learn, and joblib were not installed in this sandbox; the relevant runtime paths were therefore not claimed as fully executed.

## Findings and fixes

### Critical / high impact

1. **Training entry point was non-self-contained.** `ML flow.py` referenced undefined notebook state, missing imports, and inconsistent estimators. It now delegates to `src/forecasting.py`, a CLI that validates inputs, uses deterministic chronological holdout, and emits metrics only from a supplied dataset.
2. **Time-series evaluation leakage.** The old notebook used `train_test_split` with `random_state=42` for forecasting and fitted `MinMaxScaler`/encodings before the split. The new baseline sorts by `Date`, holds out the final 20%, and fits imputation, scaling, and one-hot encoding inside a pipeline on training rows only.
3. **Target leakage.** The notebook constructed `Inventory_Demand` from `Demand Forecast` and `UnitsSold_Price` from `Units Sold`, then used both as predictors. The new baseline explicitly removes these known target-derived fields as well as both target columns.
4. **Local and Windows-only paths.** Paths such as `D:\desctop\retail_store_inventory.CSV`, `D:\desctop\mmm.jpg`, and a `C:/Users/...` model path were removed. Data is supplied with `--data`/`SALES_DATA_PATH`; the app resolves its default artifact relative to `App/` and supports an explicit `MODEL_PATH`.
5. **Notebook stale state and outputs.** The original notebook contained 199 cells with stale outputs and a large stateful chain. It was replaced with a five-cell, output-free companion that stops clearly when real data is absent and mirrors the corrected split/preprocessing design.
6. **Dependency file was invalid/incomplete.** `src/requirements` included the built-in `datetime` package and a brittle `pystan` pin. A root `requirements.txt` now declares runtime dependencies; `src/requirements` is a compatibility include of that canonical file.

### API / application quality

- Streamlit model loading is lazy, checks artifact existence, and reports missing TensorFlow/artifacts instead of silently falling back.
- Input columns are ordered and checked against the historical five-feature contract; model input shape must match five features and output shape must be `(1, 2)`.
- The app still exposes `Inventory_Demand` and `UnitsSold_Price` because the committed artifact was trained with that historical contract. These are explicitly called out in the README as potentially target-derived and requiring retraining with future-available features.
- The app does not accept arbitrary file paths from users, use shell execution, or embed credentials.

## Added or changed files

- `src/forecasting.py` — validated, leakage-aware CLI pipeline with deterministic seed and MAE/RMSE/R² metrics.
- `ML flow.py` — self-contained compatibility entry point.
- `notebooks/Sales_Forecasting_and_Demand_Prediction.ipynb` — clean output-free companion notebook.
- `App/streamlit_app.py` — robust artifact path, lazy dependency loading, input/output shape validation.
- `Deployment` — compatibility wrapper for the Streamlit app.
- `requirements.txt`, `src/requirements` — canonical and compatibility dependency declarations.
- `.gitignore` — ignores local data, credentials, caches, MLflow runs, and generated artifacts.
- `tests/test_engineering.py` — tests missing-data failure and target/proxy exclusion.
- `pyproject.toml` — Ruff configuration.
- `App/__init__.py`, `src/__init__.py` — package markers.
- `ENGINEERING_REVIEW.md` — this report.

## Validation performed

- `ruff check .` — **passed**.
- `python3 -m compileall -q .` — **passed** for Python files.
- `python3 -m unittest discover -s tests -v` — **2 tests passed**.
- Notebook JSON parsing and AST parsing of every code cell — **passed**; `outputs=0`.
- `git diff --check` — **passed**.
- `python "ML flow.py"` without a dataset — stopped with an actionable argparse message: `--data PATH is required (or set SALES_DATA_PATH); no data was generated.`
- Direct missing-path loader check — raised an actionable `FileNotFoundError`.

## Limitations and follow-up

- The repository has no source CSV. Full fitting, metric verification, artifact regeneration, and end-to-end Streamlit inference were intentionally not run.
- TensorFlow, Streamlit, scikit-learn, and joblib were unavailable in the sandbox. They are declared in `requirements.txt`; installing them and supplying the real dataset is required for runtime validation.
- The committed `.keras` file is a historical binary artifact. Its exact training preprocessing metadata is not committed, so the app can verify dimensionality/output shape but cannot prove semantic feature scaling. Retrain and ship a serialized preprocessing-plus-model pipeline with a versioned feature schema before production use.
- Historical PDFs/Power BI/RAR deliverables were not regenerated or treated as executable source. Their reported near-perfect metrics are not accepted as validated results.
- No changes were pushed to GitHub. A local commit was created only after validation.

## Second Senior Review (2026-10-16)

A second manual pass rechecked the forecasting module, Streamlit boundary, artifact contract, notebook state, and tests. No new correctness defect was found in the repaired workflow. The source dataset and compatible model artifact remain required for a real prediction run.

Validation: `pytest -q` (2 passed), `compileall`, and `git diff --check` passed.
