# Engineering Review — Heart Failure Prediction

## Scope and evidence

- Repository audited: `Abdulrahman181/Heart_Failure_Prediction`
- Audit path: `/home/ubuntu/github_priority/Heart_Failure_Prediction`
- Audited files: `Heart_Failure_Prediction.ipynb`, `README.md`, `requirements.txt`, `.gitignore`, and the added `validate_repository.py`.
- The remote baseline contained one notebook with 98 cells and 74 saved outputs. The working tree already had a local documentation/notebook-repair commit before this review; this review adds a further local commit and does not push to GitHub.
- No Python source files existed in the baseline. The notebook was extracted to Python temporarily for AST/Ruff checks and was not executed end-to-end.

## Findings in the baseline

### Critical / high-risk

1. **Data leakage in preprocessing:** categorical encoders and `Age` quantile bins were fitted/transformed before the train/test split. This lets validation/test distributions influence training transformations.
2. **Non-stratified split:** the two `train_test_split` calls did not preserve the target class ratio, which is risky for a small binary medical dataset.
3. **Test-set selection bias:** every model was evaluated on the test set and the displayed “best” models were selected from those test metrics. This makes the final comparison optimistic and invalidates the holdout as an unbiased estimate.
4. **Weak reproducibility:** several stochastic estimators and the CV splitter had no explicit seed/shuffle configuration.
5. **Missing-data failure mode:** `pd.read_csv(...)` had no clear missing-file/schema/target validation and did not explain how to provide the excluded dataset.
6. **Fragile persistence:** models/scalers/encoders were dumped to the current working directory with open file handles and inconsistent preprocessing state.

### Medium / maintainability

- The baseline notebook repeated imports, included unused imports, had a stray `from re import X`, used misspelled/ambiguous names, and mixed raw and scaled feature variables.
- Hyperparameter grids were large relative to the dataset and were not wrapped in preprocessing pipelines.
- Saved notebook outputs made the repository noisy and could imply results from an unavailable or different dataset.
- Runtime dependencies were unpinned and omitted Jupyter/tooling declarations.
- The baseline had no repository-level static validation.

## Repairs made

1. Rebuilt `Heart_Failure_Prediction.ipynb` as a concise nine-cell workflow with zero saved outputs and empty execution counts.
2. Added an explicit `HEART_FAILURE_DATA` path with a portable `data/heart.csv` fallback and a readable `FileNotFoundError` when the dataset is unavailable.
3. Added required-column, non-empty, target-null, target-type, and binary-target checks.
4. Replaced full-data encoding/binning with a `ColumnTransformer` and `Pipeline`: numeric median imputation + scaling and categorical most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`). These are fit only on training data.
5. Added stratified 70/15/15 train/validation/test splits with `random_state=42`.
6. Selects models using validation recall (F1 and accuracy tie-breakers), then refits only the selected pipeline on train+validation and evaluates the untouched test set once. No test leaderboard is used for selection.
7. Set deterministic seeds and constrained estimator parallelism (`n_jobs=1`) for stochastic models. XGBoost is imported optionally so static inspection remains possible without its native package.
8. Stores the optional local model artifact under ignored `artifacts/best_model.pkl` using a context manager.
9. Added `validate_repository.py`, a dependency-free static check for notebook JSON, Python AST, stale outputs/execution counts, required reproducibility markers, and non-portable paths.
10. Updated README, requirements, and `.gitignore` with dataset schema, run instructions, limitations, minimum dependency versions, and generated-artifact rules.

## Validation performed

- `python3 validate_repository.py` — **passed**: notebook JSON, AST, output cleanup, path checks, and all Python files.
- `ruff check validate_repository.py /tmp/heart_failure_notebook.py` — **passed**.
- Notebook code cells were parsed with `ast.parse` — **passed**.
- Notebook has 9 cells, 5 code cells, 0 outputs, and all code execution counts are `None`.
- `git diff --check` — **passed**.
- No hard-coded `/content/`, `/Users/`, Windows drive, or other personal absolute dataset paths remain.
- Full model training was **not run**: `heart.csv` is intentionally absent, and this sandbox lacks `scikit-learn` and `xgboost`. No metrics or model-quality claims are reported.

## Remaining caveats

- Install `requirements.txt` and provide the real dataset before executing the notebook.
- The common source dataset is small; a single holdout is only a baseline. Clinical use would require external validation, calibration, uncertainty intervals, fairness analysis, and prospective governance.
- Dependency minimums improve portability but do not guarantee identical results across all future library versions.

## Second Senior Review

### Scope and additional findings

- Re-audited the complete tracked tree, including the notebook JSON, all Python/config/documentation/dependency files, repository tests, Git history/diff, output/execution state, and ignore rules.
- The first repair still had a schema-validation gap: `astype("int64")` could truncate a fractional target such as `0.5` to `0`, and numeric feature columns were inferred from whatever dtypes happened to be present in the CSV. It also required XGBoost in `requirements.txt` while treating it as optional at runtime.
- The first repair wrote only a pickle artifact with a working-directory-relative path, did not emit artifact metadata, and evaluated the test predictions more than once for the final report.
- The sandbox has scikit-learn/pandas installed but no XGBoost; `heart.csv` remains intentionally absent. No real-data metrics are claimed.

### Repairs made in this review

1. Added `heart_failure_schema.py` with explicit numeric/categorical column contracts, duplicate/missing-column checks, strict binary target validation (including fractional-value rejection), malformed numeric-value rejection, usable-feature checks, class-count checks for the requested stratified split, and stable categorical normalization.
2. Updated the notebook to use the shared validator, fixed feature-column selection to the declared schema, removed broad warning suppression, made final test prediction computation single-pass, and retained preprocessing inside each model pipeline.
3. Made the optional XGBoost dependency explicit by removing it from mandatory installation requirements and documenting the opt-in package. Added conservative major-version upper bounds for the supported stack.
4. Added trusted-local artifact metadata (`best_model_metadata.json`) beside the pickle and an explicit `HEART_FAILURE_ARTIFACT_DIR` override; the README now warns that pickle loading is unsafe for untrusted files.
5. Added `tests/test_repository.py` with schema normalization/rejection tests and repository static checks. Updated README run instructions and this report.

### Second-review validation

- `python3 validate_repository.py` — **passed**.
- `python3 -m unittest discover -s tests -v` — **passed** (4 tests).
- `ruff check .` — **passed**.
- Notebook code cells parsed with `ast.parse` — **passed**; notebook remains 9 cells, 0 outputs, and all code execution counts are `None`.
- End-to-end notebook smoke run on an isolated synthetic fixture — **passed**, including pipeline fitting and model/metadata artifact creation. The fixture is not repository data and its metrics are not reported as project results.
- `git diff --check` — **passed**.

### Remaining risks after the second review

- The real dataset is absent, so real-data training, model quality, and compatibility with the dataset owner’s exact CSV cannot be verified here.
- The baseline still uses one validation split and one final holdout; it is not external validation, calibrated clinical evidence, or a substitute for cross-validated model development.
- Pickle artifacts execute code when deserialized and must only be loaded from trusted sources. The metadata file documents the selected run but does not provide cryptographic integrity or environment locking.
- Categorical values are accepted as arbitrary strings and unknown categories are ignored by design; domain-level allowed-value validation and clinical data governance remain out of scope.
