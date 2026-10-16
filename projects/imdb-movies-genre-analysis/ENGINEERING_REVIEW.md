# Engineering review

Date: 2026-10-14

## Scope

Reviewed the complete repository at this commit. The original repository contained one large Jupyter notebook and no Python package, README, dependency manifest, tests, or ignore file. The notebook was inspected cell by cell, including imports, data loading, preprocessing, feature engineering, regression, clustering, and persisted outputs.

## Changes made

- Replaced the Windows-only `D:\\IMDb_All_Genres_etf_clean1.csv` path with portable `IMDB_DATA_PATH`/`data/` lookup and an explicit missing-file error. Required columns are validated before analysis.
- Removed all 68 stale notebook outputs and execution counters. The committed notebook is source-only and does not present old metrics as current results.
- Made text cleaning null-safe and changed target missing-value handling to drop missing `Total_Gross` rows rather than imputing the target with a global mean (which would leak information across a split).
- Created one deterministic holdout split (`random_state=42`). Training-only feature selection, scaling, PCA visualization, cross-validation, and grid-search fitting now avoid using `X_test`.
- Removed target-derived `Total_Gross_x_Rating`, `Revenue_per_minute`, and interaction leakage from the supervised feature matrix. They remain available for exploratory plots only.
- Removed `Total_Gross` from clustering inputs and made KMeans deterministic with explicit `n_init=10`.
- Made the legacy `cufflinks` import optional and added a requirements manifest; the notebook remains dependent on the original interactive plotting cells when cufflinks is installed.
- Added `README.md`, `.gitignore`, `requirements.txt`, and `tests/test_repository.py`.

## Critical issues found and status

1. **Critical — hard-coded local path:** the original notebook could only load a Windows `D:` path. Fixed with portable lookup and a clear stop message.
2. **Critical — target leakage:** the original regression matrix used `Total_Gross_x_Rating` and `Revenue_per_minute`, both computed from the target. Fixed by excluding all target-derived columns from `X`.
3. **High — preprocessing leakage:** the original selector, transformer, and some models were fit before/against all rows. Fixed for the supervised path by fitting on `X_train`/`y_train`; CV and grid search use training data.
4. **High — stale claims:** 68 old outputs included metrics and a Windows warning from an unknown historical environment. Removed; no results are claimed here because the dataset is absent.
5. **Medium — missing target imputation:** the original filled missing target values with a dataset-wide mean. Fixed by dropping rows with missing target.
6. **Medium — reproducibility:** several estimators lacked seeds or explicit KMeans `n_init`. Seeds and stable settings were added where applicable.
7. **Low — project hygiene:** no README, requirements, ignore rules, or tests existed. Added all four.

## Validation performed

- Notebook JSON parsed successfully after edits (`137` cells, `0` outputs).
- Python AST parsing was applied to every notebook code cell after removing notebook magic lines.
- Ruff was run directly on the notebook with `E9,F821,F811`; no syntax, undefined-name, or redefinition diagnostics remained.
- Repository tests validate AST parseability, output cleanup, portable data loading, and leakage guards.

## Limitations

- The Kaggle dataset is not present in the repository or sandbox, so the notebook was **not** executed end to end. No new model metrics, plots, or artifacts were generated.
- Heavy runtime validation (all Plotly/cufflinks rendering, model fitting, and clustering) therefore remains pending until the user supplies the CSV and installs the requirements.
- The legacy notebook is still a long stateful exploratory workflow rather than a packaged Python API; no public API was present to review. The new tests are intentionally static and do not invent data.

## Second Senior Review (2026-10-16)

A second manual pass rechecked the 137-cell notebook, portable data lookup, target-derived feature exclusions, train-only preprocessing, regression tests, and clean execution state. It found and removed the remaining `%matplotlib inline` IPython magic so every code cell is valid Python for static analysis and automation. The source CSV is absent, so no real-data metrics or end-to-end training claim is made.

Validation: `pytest -q` (2 passed), `compileall`, notebook AST/output-state audit, and `git diff --check` passed.
