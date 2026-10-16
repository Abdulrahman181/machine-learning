# Engineering Review — `mental-health`

**Repository:** `Abdulrahman181/Mental-Health-Prediction`
**Working tree:** `/home/ubuntu/github_priority/Mental-Health-Prediction`
**Scope:** Python/Notebook/README/dependency audit and local-only repair
**Review status:** Repaired and statically validated; **full model training was not run** because the repository contains no dataset and the sandbox does not have the project dependencies installed.

## Executive summary

The repository consisted of one notebook plus minimal documentation and dependencies. The original notebook was not portable or safely runnable from a clean checkout: it assumed a CSV and a fixed set of columns, encoded categories before the split, did not stratify the split, reported only accuracy, imported an optional heavy XGBoost dependency, used an incorrectly shaped hard-coded prediction vector, compared a classifier object to `0`, and wrote pickle files to the current directory without an explicit model contract. Its outputs were already absent, but the workflow was still difficult to reproduce and could fail with opaque exceptions.

The notebook was replaced with a smaller, dependency-aware baseline. It now validates the dataset and target, performs the split before fitting preprocessing, puts imputation/scaling/one-hot encoding in each scikit-learn pipeline, fixes seeds, reports multiple classification metrics, and fails clearly when the external data file is absent. The unused XGBoost dependency and unused plotting stack were removed. No dataset, predictions, or fabricated metric values were added.

## Findings and disposition

### Critical/high-impact findings fixed

1. **Missing-data handling was implicit.** `pd.read_csv` failed with an opaque file error and there was no schema check. The notebook now supports `MENTAL_HEALTH_DATA`, defaults to `data/data.csv`, and raises an actionable `FileNotFoundError` when the file is absent. It also checks for an empty file, the `treatment` target, valid `Yes`/`No` labels, and enough rows in both classes.
2. **Schema assumptions caused avoidable `KeyError`s.** The old code unconditionally dropped `Country`, `state`, `Timestamp`, and `comments`. The repaired version drops those columns only when present and documents the required target contract.
3. **Preprocessing was not isolated to training data.** The old notebook performed category encoding before the split. The repaired version splits with `stratify=y` and fits imputation, scaling, and one-hot encoding inside a `Pipeline`/`ColumnTransformer` on the training partition only. `handle_unknown="ignore"` keeps inference robust to unseen categories.
4. **Split and evaluation were not sufficiently stable or informative.** The old split was not stratified and the comparison printed accuracy only. The repaired version uses explicit `RANDOM_STATE=42`, a fixed stratified holdout, and reports accuracy, balanced accuracy, precision, recall, F1, and ROC AUC. Both baseline models are reported; a pre-declared logistic reference is used for the example holdout inspection rather than selecting a model on the holdout.
5. **The hard-coded inference example was unsafe.** The old 63-value list depended on an incidental one-hot column order and then compared the estimator object (`classifier`) to `0`. That example was removed; the notebook uses the fitted pipeline on the held-out dataframe instead.
6. **Dependency surface was unnecessarily heavy and unpinned.** XGBoost was imported but not needed by the repaired workflow. Requirements now contain bounded ranges for NumPy, pandas, scikit-learn, Jupyter, and nbformat. Ruff is isolated in `requirements-dev.txt`.

### Medium/maintainability findings fixed

- Replaced in-place, order-dependent label and dummy-column manipulation with explicit cleaning and pipeline preprocessing.
- Removed the broad correlation heatmap step, which did not contribute to a validated model and could be misread as a result.
- Removed automatic pickle writes; model persistence needs an explicit schema/version contract and was not required for this educational notebook.
- Added a dependency-free `unittest` static validation suite for notebook JSON, syntax, stale outputs, portability, leakage-safe split patterns, and metric coverage.
- Expanded README instructions, data contract, limitations, validation commands, and privacy/medical-use caution.
- Expanded `.gitignore` to exclude CSV data, generated model artifacts, Python caches, notebook checkpoints, and local tool caches.

## Reproducibility and leakage review

- **Data path:** `MENTAL_HEALTH_DATA` if set; otherwise `Path("data") / "data.csv"` relative to the repository working directory.
- **Target:** `treatment`, normalized to integer `1/0` only after validating `Yes/No` values.
- **Excluded fields:** `Country`, `state`, `Timestamp`, and `comments`, only when present. This avoids free text, location, and timestamp features in the baseline.
- **Age:** values outside 18–100 are converted to missing and are imputed within the training-fitted numeric pipeline; no rows or target labels are invented.
- **Split:** `train_test_split(..., stratify=y, random_state=42, test_size=0.25)`.
- **Preprocessing:** median imputation and scaling for numeric fields; most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")` for categorical fields. All are fitted inside each model pipeline after the split.
- **Models:** deterministic logistic regression and a seeded, class-balanced random forest. No model selection is performed using the holdout metrics.
- **Metrics:** accuracy, balanced accuracy, precision, recall, F1, and ROC AUC. Values are intentionally absent until a real external dataset is supplied and the dependencies are installed.

## Files changed/added

- `Mental Health Prediction.ipynb` — rewritten into a clean, output-free, leakage-safe baseline with explicit data validation and metrics.
- `README.md` — documented data contract, execution, validation, reproducibility, privacy, and limitations.
- `requirements.txt` — bounded runtime dependency ranges; removed unused XGBoost/plotting dependencies.
- `requirements-dev.txt` — optional Ruff validation dependency.
- `.gitignore` — excludes data, generated artifacts, caches, and local environments.
- `tests/test_static_validation.py` — stdlib-only static checks for notebook structure, syntax, paths, leakage safeguards, and metrics.
- `ENGINEERING_REVIEW.md` — this report.

## Validation performed

All commands were run locally in the specified repository and did not push to GitHub:

```text
python3 -m unittest discover -s tests -v
# Ran 4 tests ... OK

ruff check --select E9,F63,F7,F82 .
# All checks passed!

ruff check --fix .
ruff check .
# All checks passed!

python3 -m json.tool "Mental Health Prediction.ipynb"
# passed

git diff --check
# passed
```

A focused missing-data smoke check executed the import and loader cells without requiring scikit-learn and produced the expected message:

```text
Mental-health dataset not found. Put data.csv under data/ or set
MENTAL_HEALTH_DATA to an existing CSV file; no training was run.
```

## Not run / caveats

- **No full training run:** no `data.csv` was present in the repository, and `sklearn` was not installed in the sandbox. Consequently, no accuracy, ROC AUC, confusion matrix, or model artifact is claimed.
- **No data-quality result:** class balance, missingness, and feature cardinality can only be assessed after the real dataset is supplied.
- The dependency ranges improve portability but are not a lockfile; production use should pin a tested environment and retain a separately untouched test set.
- This project must not be used to diagnose, triage, or make decisions about a person.

## Commit policy

A local commit will be created after this report is written. No push or remote mutation is performed.

## Second Senior Review

### Scope and method

The second review re-read every tracked file, notebook code cell, dependency declaration, test, README, Git status/history, and current diff. It also compiled every notebook code cell and exercised the loader and both model pipelines with a small synthetic fixture created only in the test process. No dataset, metric, model artifact, or external result was added.

### Findings fixed in this pass

1. **Nullable string runtime failure:** the first repair converted categorical columns to pandas' nullable `string` dtype. With scikit-learn 1.5, `SimpleImputer` can raise `TypeError: boolean value of NA is ambiguous` when it receives `pd.NA`. String normalization now converts missing values to ordinary `numpy.nan`/object values before the pipeline.
2. **Opaque malformed-input failures:** CSV empty-file and parser errors now become actionable `ValueError`s. Non-string and duplicate column names are rejected explicitly.
3. **Degenerate feature schemas:** all-missing feature columns are removed, and a dataset with no remaining feature is rejected before splitting. The split also gives explicit errors for fewer than four rows or insufficient class support.
4. **Training-time empty columns:** numeric and categorical imputers use `keep_empty_features=True`, preventing a feature that happens to be empty in the training partition from silently changing the transformed schema or failing downstream.
5. **Holdout presentation ambiguity:** metrics no longer sort models by holdout F1. They preserve the pre-declared order, avoiding an implicit appearance of model selection on the test set.
6. **Regression coverage:** tests now execute the loader and both pipelines on a synthetic fixture with whitespace, missing categories, invalid ages, and an all-missing column. Static checks remain dependency-light and skip only runtime checks when optional runtime packages are unavailable.

### Second-pass validation

```text
python3 -m unittest discover -s tests -v
python3 -m compileall -q tests
python3 -m json.tool "Mental Health Prediction.ipynb"
ruff check .
git diff --check
```

The runtime smoke test uses synthetic rows solely to exercise code paths; it is not a performance evaluation and no metric is reported in this review. The real CSV is still intentionally absent.

### Remaining risks

- No full run on the real OSMI CSV was possible because it is not committed and no provenance-verified copy was supplied.
- Dependency ranges are compatible bounds, not a lockfile; CI should test a pinned Python/dependency matrix before deployment.
- The notebook is a research baseline with a single holdout, not a production model, fairness assessment, clinical instrument, or API.
- Column-level semantic leakage (for example, a future-derived survey field) cannot be determined without dataset provenance and should be reviewed before any legitimate research use.
