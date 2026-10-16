# Engineering review — Predicting-Hazardous-NEOs

**Date:** 2026-10-14
**Scope:** all tracked Python/notebook/documentation/dependency files in this checkout
**Repository:** `Abdulrahman181/Predicting-Hazardous-NEOs`

## Executive summary

The original repository contained one output-heavy notebook, one source-URL text file, and a README. It did not contain the CSV used by the notebook, a dependency manifest, tests, or a Python module. The notebook's cells were individually parseable, but it was not reproducible: it used a developer-local Windows path, relied on execution order and stale variables, reported metrics from the wrong prediction variable in places, and included resampling on the complete dataset (a leakage risk).

The notebook is now a small, output-free walkthrough backed by `neos_pipeline.py`. The pipeline validates the real CSV, removes identifier columns, performs a stratified fixed-seed split, and fits imputation/scaling inside model pipelines after the split. It evaluates logistic regression and a class-weighted random forest using accuracy, balanced accuracy, precision, recall, F1, and ROC AUC. No fabricated data or metrics were added.

## Findings in the original checkout

- **Data path:** a developer-local Windows absolute path made execution machine-specific. `NASA_Data.txt` contains only a Kaggle URL, not data.
- **Undefined/order-dependent state:** cells reused `model`, `y_pred`, `AccScore`, `results`, and other variables across multiple models. The random-forest section evaluated a decision-tree prediction in some print statements. A fresh, linear execution was not dependable.
- **Schema fragility:** `X = df.iloc[:, :-1]` assumed the last column was the target instead of checking `is_hazardous` and required features.
- **Leakage risk:** the notebook defined oversampling/undersampling and applied resampling to `X, y` before the train/test split. Any fitted imputation or resampling must be restricted to training data.
- **Preprocessing inconsistency:** logistic regression was trained on scaled arrays but its train/test `.score()` calls used unscaled data. Missing values were filled using medians computed over the full dataset before splitting.
- **Reproducibility:** the split had a seed, but estimators and resampling had inconsistent or absent seeds; class stratification was not requested.
- **Dependency hygiene:** `imblearn` imports were unused in the final model path and were not declared anywhere. Required notebook/runtime packages were also undeclared.
- **Tracked artefacts:** the notebook contained 183 KB of old outputs, including tables, plots, and historical metric text. These could not be treated as current results.
- **Documentation:** the README claimed a best model and results without reproducible values in the checkout and described a repository layout that did not exist.

## Implemented changes

- Added `neos_pipeline.py` with explicit path resolution, schema checks, binary-target normalisation, duplicate removal, and actionable missing-data errors.
- Added two deterministic scikit-learn pipelines. Imputation and scaling are learned after a stratified split; class imbalance uses `class_weight="balanced"` rather than full-dataset resampling.
- Replaced `Project_NEOs.ipynb` with an output-free, two-cell execution path that delegates to the module.
- Added `requirements.txt` for runtime, notebook, and validation dependencies. Unused `imbalanced-learn` was not added.
- Added `.gitignore` so downloaded data and generated artefacts remain local.
- Added `tests/test_engineering.py` for notebook AST/output checks, required-file checks, and the missing-data guard.
- Rewrote `README.md` to state the data/metrics status, setup, schema, methodology safeguards, validation commands, and limitations.
- Added this audit report.

## Validation performed

The following checks were run in the sandbox:

- Parsed every original notebook code cell with Python AST; no cell-level syntax error was found, but Ruff reported 8 import/redefinition findings.
- Ran Ruff after the changes: `ruff check .` passes.
- Ran the repository tests: `pytest -q` passes.
- Parsed the final notebook as JSON and AST-checked each code cell; all code cells have no outputs and `execution_count: null`.
- Inspected Git status and tracked files; only this repository was modified.

A full model training run was **not** claimed: the external CSV is absent from the repository. The installed sandbox happened to provide pandas/scikit-learn, but the declared environment should still be installed from `requirements.txt` for reproducibility. No metric values were generated or copied into this report.

## Remaining scientific/engineering considerations

The source dataset may contain repeated observations of the same object. Before operational or scientific use, assess object-level grouping and temporal drift; a time-aware or grouped holdout may be more appropriate than a random split. Threshold selection, calibration, error analysis, and independent validation remain future work.

## Second Senior Review

**Date:** 2026-10-16
**Reviewer scope:** all tracked files, the full notebook, Python pipeline, tests, requirements, README, Git history, and working-tree diff.

### Findings and implemented fixes

- **Target validation was incomplete:** a numeric target containing missing values reached an integer cast and raised an opaque pandas `IntCastingNaNError`. Missing targets now fail with an actionable `ValueError`; non-finite numeric targets are rejected as well.
- **Feature validation allowed estimator-level failures:** `+/-inf` values and columns with no observed values were not checked before the scikit-learn pipeline. The loader now rejects non-finite values and all-missing features while continuing to allow ordinary missing feature values for training-time median imputation.
- **Split arguments were under-specified:** invalid `test_size` values and tiny datasets relied on downstream scikit-learn errors. The experiment now validates the numeric range, requires enough rows/classes for both partitions, and verifies both resulting partitions contain both classes.
- **Seed propagation was inconsistent:** `run_experiment(random_state=...)` changed only the split while model estimators remained fixed at `42`. The caller-provided integer is now passed to both the split and model estimators.
- **Path handling was ambiguous:** an explicitly supplied empty path could silently fall back to `NEO_DATA_PATH`. Explicit paths now take precedence whenever `data_path` is not `None`.
- **Regression coverage was too narrow:** tests now cover missing required columns, missing targets, non-finite features, split validation, and a real synthetic run through both model pipelines with metric-schema/range assertions.

No dataset, metric, serialized model, API, or UI was fabricated or added. Notebook code cells remain parseable, output-free, and with null execution counts.

### Validation performed in this review

- `ruff check .` — passed.
- `pytest -q` — passed (6 tests).
- Targeted edge-case probes confirmed actionable errors for missing targets, non-finite features, and invalid/tiny split requests.
- Inspected `git status`, `git log`, tracked paths, Git diff, notebook JSON/state, dependency manifest, and secret/path patterns. No credentials or developer-local absolute paths were found in active project code.
- A full source-data training run remains intentionally unclaimed because the external CSV is absent from the checkout.

### Risks remaining

The current evaluation is a row-level random stratified holdout. Repeated observations of one NEO can therefore cross the split, and temporal drift is not evaluated; grouped or time-aware validation is required for scientific claims. The source label semantics and feature provenance were not independently verified here. Metrics are threshold-dependent and there is no calibration, uncertainty analysis, external validation, persisted-model loading contract, or production inference API. Dependency ranges are bounded but not lock-file reproducible, and results can vary across supported library versions/hardware.
