# Engineering Review

## Scope

Reviewed the complete repository at `Abdulrahman181/Used-Cars-EDA-and-business-questions` on 2026-10-14. The repository contained one README and one Jupyter notebook (`vehicles-EDA.ipynb`); no Python package, service/API, model artifact, or source dataset was present.

## Findings in the original revision

- The notebook hard-coded `/kaggle/input/craigslist-carstrucks-data`, which made it non-portable and caused an immediate failure outside Kaggle.
- The variable `df` first held a directory string and was later replaced by a DataFrame, making the workflow confusing and fragile.
- `os.listdir` and `pd.read_csv` had no clear missing-data handling or schema validation.
- Missing values were filled with global means/modes before analysis. This can distort descriptive results and would cause leakage if the notebook were extended into modeling. `posting_date` was first filled with the string `Unknown` and later coerced to datetime, silently converting those entries to `NaT`.
- `year.astype(int)` could fail on malformed values and did not handle invalid numeric input robustly.
- The mileage/price chart grouped on every unique odometer value, producing a noisy and potentially very large line plot.
- The notebook had 22 stale, large outputs and old execution counts committed to source control. Those outputs included numerical results that could not be reproduced without the absent dataset.
- There was no dependency manifest, `.gitignore`, test/validator, or reproducible validation command.
- Static checks on the extracted original notebook found unsorted imports, an unused NumPy import, semicolons, and useless expression statements.
- No API or model endpoint was present to review.

## Changes made

- Rebuilt `vehicles-EDA.ipynb` as a portable, data-gated EDA workflow:
  - Reads `USED_CARS_DATA`, `data/vehicles.csv`, or `vehicles.csv` only.
  - Raises a clear `FileNotFoundError` when the source CSV is unavailable.
  - Validates required columns before analysis.
  - Coerces only numeric/time fields used by the analysis and reports missingness.
  - Uses listing ID de-duplication when IDs are available.
  - Excludes invalid/missing values from the relevant chart rather than inventing replacements.
  - Uses a deterministic bounded sample for the mileage/price scatter plot.
  - Reports mean, median, and count for fuel comparisons.
  - Includes explicit interpretation and future-model leakage guardrails.
  - Removes all saved outputs and execution counts so results cannot become stale.
- Replaced the README with accurate setup, data, layout, validation, and limitation instructions.
- Added `requirements.txt` with bounded runtime and validation dependencies.
- Added `pyproject.toml` for pytest and Ruff configuration.
- Added `.gitignore` for local data, generated outputs, notebooks checkpoints, caches, and environments.
- Added `tests/test_repository.py` to validate notebook JSON/syntax, output cleanliness, data gating/portability, and dependency declarations.

## Validation performed

- `pytest -q`: **4 passed**.
- `ruff check tests /tmp/vehicles_current.py`: **all checks passed**. The temporary file contains all current notebook code cells for linting and is outside the repository.
- Python AST parsing of every current notebook code cell: **passed**.
- Notebook JSON validation: **passed**; 14 code cells, zero outputs, and all execution counts set to `null`.
- Missing-data gate exercised without synthetic data: raised the expected clear `FileNotFoundError`.
- Existing notebook static baseline was recorded before changes; its Ruff issues were corrected in the rewritten notebook/tests.

## Limitations and residual risks

- The source `vehicles.csv` is absent from the repository and was not downloaded or fabricated. Therefore the full notebook was not executed end-to-end and no numerical EDA results are claimed.
- Heavy optional execution tooling (`nbformat`/`nbconvert`) was not installed in the sandbox. JSON parsing, AST compilation, unit tests, and code-cell linting were used instead.
- The notebook is descriptive EDA only. There is no train/test split, predictive model, metric evaluation, or model artifact to audit. If modeling is added, all imputation, encoding, scaling, and feature selection must be fitted on training data only.
- External Kaggle licensing/provenance and the freshness of the Craigslist scrape require confirmation by the data owner before redistribution or production use.

## Second Senior Review (2026-10-16)

### Scope and additional checks

The tracked repository, complete notebook JSON, all Python/config/test/README/dependency files, current Git history, and working-tree diff were reviewed again. Static AST compilation, notebook-state checks, Ruff, the existing tests, and an execution test using a deliberately malformed-value fixture were performed. No API, UI, model artifact, secret, source dataset, or untracked application code was present.

### Additional findings fixed

- **Fractional manufacture years could crash the notebook.** The previous `astype("Int64")` failed on values such as `2018.5`. Year plotting now keeps only integer years in a plausible range (1886 through the current UTC year), so malformed or future values are excluded rather than silently rounded.
- **Rows with missing IDs could be silently collapsed.** When any ID existed, `drop_duplicates(subset=["id"])` treated all missing IDs as one duplicate group. De-duplication now applies only to non-missing IDs and retains every row without an ID, preserving evidence that cannot be proven duplicate. Original row order is retained.
- **Mixed date formats and non-finite numerics were not handled explicitly.** Date parsing now uses pandas' supported `format="mixed"` mode; numeric analysis columns convert infinities to missing values before chart filters.
- **Configured paths were less defensive than needed.** Blank environment values are treated as unset, user-home expansion is supported, and successful paths are normalized to absolute paths. Default candidates are anchored to the current working directory used by the documented command.
- **Runtime coverage was too static.** Tests now execute all notebook code cells with a temporary CSV containing duplicate IDs, missing IDs, malformed numeric/date values, negative mileage, and a fractional year. This is a contract test only; it does not create or claim production EDA results.

### Validation after the second review

- `pytest -q`: **5 passed** (including the fixture-backed notebook execution test).
- `ruff check tests /tmp/vehicles_second_review.py`: **all checks passed**; the temporary file contains extracted notebook code and is outside the repository.
- Python AST parsing of every code cell: **passed**.
- Notebook JSON validation: **passed**; all code-cell outputs are empty and execution counts are `null`.
- No source dataset was downloaded or fabricated, so no numerical business findings or end-to-end run are claimed.

### Remaining risks

- Full execution with the authorized real `vehicles.csv` remains unverified because the repository intentionally does not contain that dataset.
- The schema contract still assumes the nine required column names; a differently structured authorized export needs an explicit adapter rather than implicit inference.
- The plausible manufacture-year bounds are pragmatic EDA filters, not a domain-certified data-quality policy; they should be reviewed if the source covers antique vehicles or future-dated listings.
- This repository remains descriptive EDA only: there is no train/test split, predictive metric, API/UI, or artifact-loading path to validate. If modeling is added, fit all preprocessing on training data only.
