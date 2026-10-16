# Engineering Review — Black Friday Sales Prediction

## Scope

Reviewed the complete tracked repository at the time of the audit, including the original Jupyter notebook and README. The repository initially contained only `Black Friday Sales Prediction.ipynb` and `README.md`; there were no Python modules, dependency manifest, tests, or dataset files.

This review makes only local working-tree changes. Nothing was pushed to GitHub.

## Critical findings in the original repository

1. **The notebook could not run outside the author's machine.** Its first data-loading cell used a hard-coded Windows path containing a local username and Arabic desktop directory.
2. **Dependencies were undocumented.** There was no `requirements.txt` or equivalent, so the notebook's pandas, NumPy, plotting, and scikit-learn requirements could not be recreated reliably.
3. **Preprocessing was performed before the train/test split.** Global `LabelEncoder` fitting and global missing-value replacement made the modeling flow harder to reproduce and could leak distribution information into evaluation. Label encoding also imposed arbitrary ordinal relationships on categorical values.
4. **The row-wise split did not address repeated customers.** The source dataset contains multiple records per `User_ID`; a customer could therefore appear in both partitions. The original notebook also encoded identifiers instead of explicitly excluding them.
5. **Evaluation was incomplete.** It reported only R² and did not report MAE/RMSE, grouped validation, or a reproducibility check. The unbounded default decision tree was also prone to overfitting.
6. **The committed notebook contained many stale execution counts and outputs.** Those outputs were not a reliable benchmark and made it unclear whether the notebook could be rerun from a clean kernel.

## Repairs implemented

### `Black Friday Sales Prediction.ipynb`

Rebuilt as a clean 15-cell notebook with all outputs and execution counts cleared.

- Added `BLACK_FRIDAY_DATA` configuration with a portable default of `data/Black Friday Dataset.csv`.
- Added a clear `FileNotFoundError` explaining where to put the data or which environment variable to set.
- Added required-column and numeric-target validation.
- Kept descriptive data checks separate from model fitting.
- Excluded `User_ID` and `Product_ID` from features.
- Replaced row-wise splitting with deterministic `GroupShuffleSplit` by `User_ID`; assertions verify disjoint row indexes and customer groups.
- Put numeric imputation, categorical imputation, and one-hot encoding inside a `ColumnTransformer`/`Pipeline`, so preprocessing is fitted only on training data and refit per cross-validation fold.
- Added deterministic Decision Tree and Random Forest baselines with bounded tree complexity and fixed seed.
- Added MAE, RMSE, and R² evaluation.
- Added optional grouped three-fold cross-validation, disabled by default because it can be expensive on the full dataset.
- Added limitations that explicitly prevent interpreting absent results as benchmark scores.

### Supporting files

- `requirements.txt`: Python 3.10+ dependency ranges for NumPy, pandas, scikit-learn, matplotlib, and Jupyter.
- `.gitignore`: excludes local CSV data, notebooks checkpoints, environments, caches, and generated model artifacts.
- `tests/test_static_validation.py`: dependency-light unittest checks for notebook JSON/AST validity, cleared outputs, portable paths, leakage-aware pipeline tokens, and required repository files.
- `README.md`: documents data setup, execution, methodology, metrics, validation commands, and the fact that no data or fabricated results are committed.

## Validation performed

The following checks passed:

- `python3 -m unittest discover -s tests -v` — 5 tests passed.
- `ruff check tests` — passed.
- JSON parsing and `ast.parse` for all 8 code cells — passed.
- Notebook cleanliness check — all code cells have empty `outputs` and `execution_count: null`.
- `ruff check --select E9` on a combined Python extraction of all notebook code — passed.
- `git diff --check` — run before commit (must remain clean).

## Execution limitation

The repository does not contain the Kaggle CSV. At the time of the first review, this Sandbox also lacked scikit-learn, so a full notebook execution, model fitting, and numerical metric report were **not** claimed. The static checks validate syntax, structure, portability, and the intended leakage controls only. After installing `requirements.txt` and providing the dataset, run the notebook to obtain real metrics.

## Second Senior Review

The second pass re-read every tracked file, the complete notebook, the previous review, and the commit history/diff. It also checked for syntax/runtime hazards, function and pipeline contracts, empty-input behavior, schema validation, leakage, grouped splitting/evaluation, reproducibility, dependency compatibility, paths/secrets, notebook state, artifacts, tests, documentation, and missing application/API files. The repository contains no Python package, service, UI, API, secret, or generated model artifact to review; its only executable workload is the notebook.

### Findings and repairs

1. **Invalid or degenerate input was not rejected early.** An empty CSV, duplicate column names, missing/non-finite `Purchase`, missing `User_ID`, or a single customer group could otherwise fail later with an opaque pandas/scikit-learn error or produce invalid evaluation. The notebook now validates all of these cases with actionable errors.
2. **The grouped holdout could theoretically be empty.** The notebook now checks both row partitions and both group partitions after `GroupShuffleSplit`.
3. **Unsupported feature dtypes were silently dropped by `remainder="drop"`.** String extension dtypes are now treated as categorical, while any remaining unsupported dtype and the no-feature case fail explicitly. Entirely missing training features are also rejected instead of relying on estimator-specific behavior.
4. **Optional grouped CV lacked a group-count contract.** `RUN_CV=True` now requires at least three distinct customers before constructing `GroupKFold`.
5. **Regression tests did not protect these edge-case contracts.** Static tests now assert that the validation guards remain present, in addition to the existing notebook JSON/AST, cleanliness, portability, leakage, and repository-file checks.

No data leakage, hard-coded secret, stale notebook output, execution count, fabricated metric, or path outside the documented configuration was found in the second pass. No API/UI behavior or artifact-loading code exists in this repository, so those areas are not applicable rather than untested.

### Second-pass validation

- `python3 -m unittest discover -s tests -v` and `pytest -q` passed after the changes.
- Notebook JSON parsing, AST parsing of every code cell, clean outputs/counts, `compileall`, `ruff`, and `git diff --check` passed.
- `pip install --dry-run --ignore-installed -r requirements.txt` resolved a clean dependency set. `pip check` in the pre-existing shared environment reports `contourpy 1.4.0` requiring NumPy >=2 while NumPy 1.26.4 is installed; this is outside the repository and a clean requirements solve selects a compatible NumPy version.
- A real end-to-end model run remains conditional on the absent Kaggle CSV and was not claimed.

## Reproduction checklist

```bash
python -m pip install -r requirements.txt
export BLACK_FRIDAY_DATA=/path/to/Black\ Friday\ Dataset.csv
jupyter notebook "Black Friday Sales Prediction.ipynb"
python -m unittest discover -s tests -v
ruff check tests
```
