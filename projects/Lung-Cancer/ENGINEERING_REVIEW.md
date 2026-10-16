# Engineering Review — Lung-Cancer

## Scope and baseline

- Repository audited: `Abdulrahman181/Lung-Cancer` only.
- Baseline contained one notebook, `Lung Cancer.ipynb`; there were no Python modules, README, requirements file, tests, or committed dataset.
- The notebook referenced `data.csv`, but that file is not present in the repository. No data, model artifact, or performance metric was fabricated.

## Findings and fixes

### Critical / blocking

1. **Missing dataset** — the original first data cell assumed `data.csv` existed and failed unclearly depending on the working directory. The notebook now resolves `LUNG_CANCER_DATA` or `data.csv` and raises an actionable `FileNotFoundError` before training when absent.
2. **Weak reproducibility and split design** — the original split was unstratified and only seeded once. The revised notebook uses a fixed seed and stratified 80/20 split, then performs seeded 5-fold stratified CV on the training partition only.
3. **Test-set model selection risk** — the original notebook printed several test-set scores without a selection protocol. The revised flow selects the model by training-fold CV F1, refits that model on training rows, and evaluates the untouched test partition once.
4. **Unstable/partial metrics** — the original flow primarily reported accuracy and reused a mutable `y_pred` for plotting. The revised flow computes accuracy, balanced accuracy, precision, recall, F1, ROC-AUC when probabilities are available, a classification report, and a held-out confusion matrix from explicit variables.
5. **Implicit schema and encoding assumptions** — the original code depended on positional columns and trailing spaces such as `FATIGUE `. The revised flow normalizes column names, validates the complete named schema, validates `M/F` and `YES/NO`, rejects missing/unknown/non-finite values, and uses named feature columns.
6. **Heavy undeclared dependency** — the original notebook imported XGBoost without a requirements file. XGBoost was removed from the baseline candidate set in favor of declared scikit-learn estimators; dependencies are now explicit and installable.

### Quality and safety

- Removed arbitrary hard-coded sample prediction input; `predict_one` now requires caller-provided feature values matching the validated schema.
- Removed stale notebook outputs, execution counts, and the original exploratory state.
- Model preprocessing (standardization for logistic regression/KNN) is inside pipelines, so scaling is fit within each CV fold and cannot leak test information.
- Artifacts are written only after a successful data-backed run to `artifacts/lc.pkl`, which is ignored by Git.
- Added a standard-library static validator that parses every notebook code cell with AST, rejects stale outputs/counts, flags local absolute path markers, and checks for the data-missing guard.

## Validation performed

- `python3 validate_repo.py` — passed.
- `python3 -m unittest discover -s tests -v` — 3 tests passed.
- `python3 -m compileall -q validate_repo.py tests` — passed.
- `ruff check validate_repo.py tests` — passed (Ruff was available).
- `git diff --check` — passed.
- Notebook inspection confirms 17 cells, zero saved outputs, and zero execution counts.

## Not run / limitations

A full notebook execution and trustworthy ML metric run were **not performed** because the repository does not contain `data.csv` and no authorized replacement dataset was supplied. Installing or exercising the full ML stack was not used to manufacture a result. Once the real dataset is supplied, execute the documented notebook command and review the resulting metrics for domain appropriateness.

## Second Senior Review

### Scope and baseline

- Re-reviewed every tracked file, the complete notebook (all code cells and saved state), Python/config/test/README/requirements files, Git history, and the working-tree diff.
- The repository still contains no authorized `data.csv`, trained model, or trustworthy metrics. No data, prediction, or metric was fabricated.
- Baseline checks passed, but the first review left the notebook logic duplicated and its reusable contracts largely untested.

### Findings and fixes

1. **Notebook split implementation was not centralized or sufficiently guarded.** A second review found that the notebook still used a direct `train_test_split` path and did not verify that the resulting training partition could support five stratified folds or that the test partition contained both classes. Added `split_dataset` with deterministic stratification and explicit train/test class-capacity checks; the notebook now uses it.
2. **Schema validation was notebook-local and missed ambiguous columns.** Added `lung_cancer.py` with reusable validation. It rejects duplicate names after canonicalization, empty canonical names, missing/unknown categories, missing/non-finite values, non-numeric risk factors, empty data, and single-class targets. Extra CSV columns are retained but never used as features.
3. **Prediction API accepted invalid numeric values.** The former `predict_one` could allow `None`/`NaN` or infinity through numeric coercion. It now validates exact keys, categorical values, missingness, numeric conversion, and finiteness through `validate_feature_values`.
4. **Artifact contract and loading path were incomplete.** The former notebook wrote an unversioned dictionary and had no loader validation. Added versioned, schema-bound `save_artifact`/`load_artifact` helpers with atomic replacement and notebook reload verification. Joblib remains a trusted-file-only format because deserialization can execute code.
5. **Static validator assumed well-formed cell structures and had limited assertions.** Hardened `validate_repo.py` to validate the notebook root, format, cell types, source fields, AST syntax, clean outputs/counts, path markers, data guard, prediction guard, artifact reload, and required project files.
6. **Tests covered only notebook cleanliness.** Expanded tests to nine cases covering normalization, duplicate columns, schema errors, finite values, deterministic stratified splitting, CV capacity, prediction inputs, artifact round-trip/schema rejection, notebook AST/state, and required files.
7. **Documented notebook execution dependencies were incomplete.** Added bounded `nbconvert` and `ipykernel` requirements and documented the artifact environment variable, trusted-pickle caveat, reusable helpers, and no-data limitation.
8. **Notebook stale/duplicate state.** Restored the descriptive correlation cell, removed the stale direct split implementation, and cleared all outputs and execution counts.

### Second-review validation

- `python3 validate_repo.py` — passed.
- `python3 -m unittest discover -s tests -v` — 9 tests passed.
- `python3 -m compileall -q lung_cancer.py validate_repo.py tests` — passed.
- `git diff --check` — passed.
- Notebook inspection confirms all code cells parse, have no saved outputs or execution counts, and use the reusable split/prediction/artifact contracts.
- Full notebook training was intentionally **not run**: no authorized dataset is present, so an end-to-end model result would be misleading.

### Remaining risks and limitations

- The model candidate set and metric choice (CV F1 and one held-out split) are engineering defaults, not clinical validation; domain review, external validation, calibration, and prospective evaluation remain necessary.
- A single hold-out split can still have high variance on a small dataset; no confidence intervals are claimed.
- Joblib model loading is unsafe for untrusted files and is not a cross-version interchange format; pin the environment and load only trusted artifacts.
- The notebook still requires the caller to run from a directory where relative `data.csv`/artifact paths are intended, or to set the explicit environment variables.
- No dataset quality, label provenance, class imbalance, or leakage audit beyond the declared tabular contract can be completed without the authorized data.
