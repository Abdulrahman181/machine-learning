# Engineering review: Chronic Kidney Disease

**Repository:** `Abdulrahman181/chronic_kidney_disease`  
**Review scope:** Python/Notebook/documentation/dependencies in this checkout  
**Review status:** repaired locally; no GitHub push performed

## Executive summary

The checkout contained one large Jupyter notebook and minimal project metadata. The original notebook had no tracked dataset, stale exploratory structure, repeated imports, fragile notebook-state dependencies, random-value imputation performed before splitting, chi-square feature ranking performed before splitting, a hard-coded model artifact path, a fabricated inference row whose column order was not declared, and a hard-coded accuracy claim. Those patterns made the reported evaluation optimistic or non-reproducible and made a clean restart difficult.

The notebook was rewritten as a compact, executable baseline. It now fails clearly when `kidney_disease.csv` is unavailable, accepts the standard abbreviated UCI headers, validates the target schema, splits before fitting preprocessing, keeps imputation/encoding inside a pipeline, uses fixed seeds, reports multiple metrics without claiming results, and saves a single portable fitted pipeline only when the user explicitly runs the training cells.

## Findings and dispositions

| Severity | Finding | Disposition |
|---|---|---|
| Critical | Feature selection (`SelectKBest`) was fit on all rows before the train/test split, leaking target information into test-time feature choice. | Removed feature-selection stage; all preprocessing is fitted inside the training pipeline. |
| Critical | Random-value imputation and category encoding were performed before the split and depended on mutable notebook state. | Replaced with `SimpleImputer` and `OneHotEncoder` inside `ColumnTransformer`/`Pipeline`; split is stratified and happens first. |
| High | Notebook had a fixed `kidney_Model.joblib` path and an inference example with an undeclared feature order/fabricated values. | Model path is configurable via `CKD_MODEL_PATH`; inference helper requires named feature mappings and no fabricated patient is included. |
| High | Original notebook asserted `Accuracy : XGBoost 98%` without a reproducible metric calculation. | Removed the claim; metrics are calculated at runtime for cross-validation and the hold-out set. |
| High | Missing data was not included in the repository, while the notebook attempted to read it with a relative path and assumed a description file. | Added `CKD_DATA_DIR`, a clear missing-file error, schema validation, canonical abbreviation mapping, and documentation. |
| Medium | Notebook had stale outputs and 160+ exploratory cells with repeated imports/useless expressions; Ruff identified 42 issues in an extracted source view. | Replaced with 16 focused cells and removed all outputs/execution counts. |
| Medium | Dependency file omitted the notebook runtime and had no compatibility floor. | Added minimum versions for the runtime, ML, plotting, notebook, and serialization dependencies. |
| Low | No static validation or explicit artifact/cache policy was present. | Added `validate_notebook.py` and expanded `.gitignore` for data/artifacts/caches. |

## Reproducibility and leakage review

- `SEED = 42` is used by `train_test_split`, `StratifiedKFold`, and `XGBClassifier`.
- `train_test_split(..., stratify=y)` creates the hold-out partition before model fitting.
- Numeric median imputation and categorical most-frequent imputation/one-hot encoding are learned only within each training fold or final training fit.
- Cross-validation is restricted to `X_train, y_train`. The test partition is scored once after fitting.
- The test report includes accuracy, balanced accuracy, F1, precision, recall, ROC AUC, a classification report, and a confusion matrix. No result is stated in advance.
- The model uses `n_jobs=1` to reduce nondeterminism and resource surprises. Exact reproducibility can still vary with OS, BLAS, XGBoost, and package versions.

## Data and execution status

No CSV, description file, or other dataset was present under the repository at review time. Heavy dependencies were not installed solely to fabricate a run, and a full model fit was **not** claimed. The performed checks are static/structural only. To run the notebook, install `requirements.txt`, provide `CKD_DATA_DIR`, and execute cells from top to bottom.

## Validation performed

- `python3 -m json.tool Chronical_kidney_disease.ipynb` — passed.
- `python validate_notebook.py` — passed after the rewrite (AST parse, no outputs/counts, required safeguards, and forbidden-pattern checks).
- Ruff was run against the original extracted notebook code and reported 42 issues, including repeated imports, unused expressions, and import redefinitions. Ruff now reports `All checks passed!` for the rewritten notebook source.
- A full data-dependent training/metric run was not performed because the dataset is absent. This report intentionally contains no invented metric values.
