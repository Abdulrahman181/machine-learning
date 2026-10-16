# Engineering review — Diabetes-screening

**Repository:** `Abdulrahman181/Diabetes-screening`  
**Scope:** Python, Jupyter notebook, README, dependency/configuration files, and the checked-in model artifact.  
**Review outcome:** code and reproducibility improvements committed locally; no push was performed.

## Executive summary

The repository was a minimal Streamlit app plus an exploratory notebook. The app used a hard-coded Windows path, passed text strings to the estimator, had no dependency manifest, and produced an incorrect/garbled negative/positive message. The notebook contained stale rendered outputs, an in-notebook `pip install`, non-portable relative assumptions, an oversampling operation before the train/test split, non-stratified splitting, missing random seeds for several models, and reversed arguments for recall/F1 calculations.

The app now uses a path relative to `diabetes.py` by default or `DIABETES_MODEL_PATH`, collects numeric values in a fixed feature order, validates the loaded object, and reports clear missing/incompatible-artifact errors. The notebook was replaced with a clean, unexecuted, reproducible template that fails clearly when the source CSV is absent, splits with stratification and a fixed seed, fits preprocessing only on training data, and evaluates held-out predictions with correctly ordered metric arguments. No data or metrics were invented.

## Critical findings and disposition

| Priority | Finding | Disposition |
|---|---|---|
| High | `diabetes.py` loaded `C:\\Users\\TREIKA\\...\\model.pkl`; it could not run outside one Windows machine. | Fixed with `Path(__file__)` default and `DIABETES_MODEL_PATH` override. |
| High | The app passed `st.text_input` strings to the estimator and did not guarantee the training feature order. | Fixed with numeric `st.number_input` controls and the explicit `FEATURES` tuple. |
| High | `model.pkl` could not be validated in this sandbox because scikit-learn is not installed; the source training CSV is also absent. | Artifact was retained unchanged; requirements and clear runtime/documentation errors were added. No full inference/training claim is made. |
| High | The original notebook resampled the complete dataset before splitting, which can leak duplicated minority examples into the test set. | Removed the pre-split oversampling path; the replacement split occurs before any learned transform. |
| High | Original `recall_score(pred, y_test)` and `f1_score(pred, y_test)` reversed ground truth and predictions. | Replacement evaluates `recall_score(y_test, predicted)` and `f1_score(y_test, predicted)`. |
| Medium | Original split was not stratified and multiple estimators lacked fixed seeds. | Replacement uses `stratify=y` and `RANDOM_STATE = 42`; model pipeline is deterministic for a given environment. |
| Medium | Notebook contained old HTML/image/text outputs and an executable `pip install imbalanced-learn` cell. | Notebook rebuilt with zero code-cell outputs/execution counts and dependencies moved to `requirements.txt`. |
| Medium | No README setup, data location, model compatibility, or safety caveats. | README now documents installation, paths, required columns, pickle trust, missing data, and the no-diagnosis limitation. |
| Low | No automated repository-level static validation. | Added dependency-light tests for Python AST parsing, notebook cleanliness/JSON, and portable model-path configuration. |

## Data and model integrity

- There is **no `diabetes.csv`** (or equivalent source dataset) in the repository. The notebook expects `data/diabetes.csv` or the path in `DIABETES_DATA_PATH` and raises a clear `FileNotFoundError` when it is absent.
- The existing `model.pkl` was not retrained or rewritten. Attempting to inspect it in this sandbox failed with `ModuleNotFoundError: No module named 'sklearn'`; this is an environment/dependency limitation, not evidence that the artifact itself is corrupt.
- No accuracy, recall, F1, ROC-AUC, or clinical performance result is claimed in this review.
- The application loads a pickle only from a configured local path. Pickle loading is code execution; only trusted artifacts should be used.
- The app preserves the eight columns used by the original notebook: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`.
- The notebook's preprocessing is inside a scikit-learn pipeline and is fitted only on `X_train`; the test partition is held out until evaluation. The checked-in inference artifact is intentionally not silently replaced by the notebook's illustrative model.

## Files changed

- `diabetes.py` — portable, typed Streamlit inference entry point with model/error handling.
- `diabetes_contracts.py` — dependency-light shared feature, path, model, and prediction contracts.
- `diabetes.ipynb` — clean reproducible training/evaluation template with outputs removed.
- `README.md` — setup, data/model configuration, limitations, and validation instructions.
- `requirements.txt` — runtime and review dependencies with bounded major versions.
- `.gitignore` — local environments, caches, datasets, and generated artifacts (while retaining `model.pkl`).
- `tests/test_static_validation.py` — AST, JSON/notebook, and path portability checks.
- `ENGINEERING_REVIEW.md` — this review.

## Validation performed

The following checks passed in the sandbox:

- `ruff check diabetes.py tests` — passed.
- `python3 -m compileall -q diabetes.py tests` — passed.
- `python3 -m json.tool diabetes.ipynb` — passed.
- Direct execution of all three dependency-light test functions — passed.
- Explicit AST parse of every Python file — passed.
- Notebook structural check: 7 cells, 0 code-cell outputs, all code-cell execution counts unset — passed.
- `git diff --check` — passed.

A full `pytest` run and model inference/training run were **not claimed**: pytest, Streamlit, and scikit-learn were not installed in the starting sandbox, and the source CSV is absent. Install `requirements.txt` and provide the dataset before running the full notebook or app smoke test.

## Local delivery

Changes are committed locally with no remote push. The final local commit SHA is recorded by the task result and can also be obtained with `git rev-parse HEAD`.

## Second Senior Review

### Scope and findings

This follow-up inspected every tracked file, the full notebook JSON/code state,
the existing Git history and clean diff, the checked-in pickle metadata, and the
Python/config/test/documentation contracts. The repository was clean at the
start of this review. The following residual defects were found:

| Priority | Finding | Disposition |
|---|---|---|
| Critical | `model.pkl` embeds scikit-learn `1.0.2` tree state and raises a tree-node dtype `ValueError` under the installed 1.5.1 runtime; the app did not catch this error. | Fixed the UI error boundary and documented that the artifact must be retrained/exported with a compatible dependency set. The binary was not rewritten without source data. |
| High | The notebook's final cells contained literal `\\n` text rather than real newlines, and the metrics dictionary closed with `]`; JSON validity did not imply executable Python. | Repaired both cells and added AST parsing for every notebook code cell to the tests. |
| High | The app only checked for a callable `predict()` and coerced any non-`1` result to negative, allowing feature-contract drift and malformed outputs to pass silently. | Added shared model feature-count/name validation, exact one-row input validation, and strict binary prediction validation. |
| Medium | Notebook input validation allowed non-numeric feature values, non-binary labels, duplicate headers, and non-finite values to fail later or produce ambiguous training behavior. | Added explicit schema, numeric, finite-value, binary-target, minimum-size, and duplicate-header checks while preserving missing values for training imputation. |
| Medium | Duplicate feature vectors with conflicting labels could be split across train/test, and identical feature vectors were only deduplicated when the target also matched. | Reject conflicting feature/label duplicates and deduplicate by feature vector before the split. |
| Medium | Model paths supplied relatively could depend on Streamlit's process working directory. | Centralized path resolution relative to the application directory and covered it with tests. |

### Second-review validation

- `python3 -m pytest -q` — **6 passed**.
- `ruff check diabetes.py diabetes_contracts.py tests` — **passed**.
- `python3 -m compileall -q diabetes.py diabetes_contracts.py tests` — **passed**.
- `python3 -m json.tool diabetes.ipynb` — **passed**.
- AST parsing of every Python source and every notebook code cell — **passed**.
- Notebook state check: all code-cell outputs empty and execution counts unset — **passed**.
- `git diff --check` — **passed**.
- The checked-in model was inspected without rewriting it; loading under the
  available scikit-learn 1.5.1 failed with the known legacy tree dtype error.

### Remaining risks and explicit non-claims

- The source CSV is still absent, so no retraining, conversion, held-out
  evaluation, or clinical performance claim is possible.
- The checked-in pickle remains a trusted-code-execution risk and a legacy
  scikit-learn compatibility risk. The app now fails clearly rather than
  silently using an incompatible model, but the artifact itself still needs a
  supervised rebuild/export.
- Streamlit was not installed in the sandbox, so no browser/UI smoke test was
  claimed. Dependency-free contracts and all static checks were exercised.
