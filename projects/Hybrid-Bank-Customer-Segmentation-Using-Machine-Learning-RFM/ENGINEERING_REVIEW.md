# Engineering Review — Hybrid Bank Customer Segmentation

**Repository:** `Abdulrahman181/Hybrid-Bank-Customer-Segmentation-Using-Machine-Learning-RFM`
**Review date:** 2026-10-13
**Scope:** committed Python/notebook/documentation/dependency surface in the requested repository only

## Executive summary

The repository originally contained only `Bank_Customer_Segmentation.ipynb` and `README.md`. The notebook was not portable or statically executable: it included a Jupyter magic that is invalid Python AST, a hard-coded Colab path, saved outputs/execution counts, an unseeded DBSCAN sample, fragile silhouette evaluation, and a customer-demographic merge that could duplicate customer rows. The README also presented specific dataset sizes and metric values as if verified, although the dataset was not committed.

The notebook has been rewritten as a clean, output-free, deterministic unsupervised workflow. It now has an explicit data contract, environment/path configuration, validation errors, customer-level aggregation, deterministic sampling, guarded metrics, and documented validation boundaries. Dependencies, ignore rules, and dependency-free static tests were added. No data was invented and no results were generated or claimed.

## Critical findings and remediation

### 1. Missing data and non-portable path — fixed

- Original load used `/content/bank_transactions.csv`, which only works in a particular Colab environment.
- The dataset is not present in the repository after inspection.
- The notebook now checks `BANK_TRANSACTIONS_PATH`, then `bank_transactions.csv` and `data/bank_transactions.csv` relative to the working directory.
- Missing data raises a clear `FileNotFoundError` describing the required environment variable and checked locations.
- Local data extensions are ignored by Git to reduce accidental disclosure of customer data.

### 2. Syntax and static-analysis defects — fixed

- Original cell 2 contained `%matplotlib inline`, which caused `ast.parse` to fail.
- Imports were reordered and the unused import was removed.
- Every current code cell parses as Python AST.
- Ruff reports no diagnostics across the repository.

### 3. Customer-level integrity risk — fixed

- The original notebook merged `drop_duplicates()` demographics back to customer aggregates. If a customer had more than one observed age/gender value, this could create multiple rows per customer and misalign features with clustering labels.
- The replacement aggregates `Age` by customer median and optional gender with a first non-null value in the same groupby operation. This keeps one row per `CustomerID`.
- Required identifiers, transaction amount, and transaction date are validated before aggregation. Exact duplicate rows and dropped required-field rows are counted, not silently hidden.

### 4. Reproducibility — fixed

- `RANDOM_STATE = 42` is defined once and used for NumPy sampling and KMeans.
- DBSCAN samples with `np.random.default_rng(RANDOM_STATE)` rather than the global unseeded random state.
- PCA uses a deterministic full SVD solver.
- KMeans uses a fixed seed and `n_init=10`.
- RFM quartile scoring ranks with `method="first"` before `qcut`, making tie handling deterministic.

### 5. DBSCAN labeling and metric validity — fixed

- The original code assigned DBSCAN labels only to a sample, filled all other customers with `-1`, and then evaluated that synthetic full-length labeling against all PCA rows. This could misrepresent unsampled rows as noise and could fail when fewer than two non-noise clusters existed.
- The replacement keeps DBSCAN labels scoped to the sampled rows, reports the sample size, excludes noise only for the silhouette calculation, and reports `NaN`/an explanatory message when a silhouette is undefined. No unsampled customer is assigned a fabricated DBSCAN result.
- KMeans silhouette is similarly guarded for the single-cluster case.

### 6. Splitting and leakage — assessed and documented

- This repository performs unsupervised clustering and has no target variable. A supervised train/test split would not be meaningful for the current objective.
- Feature engineering uses transaction history to form customer-level descriptors; cluster labels are not used to build those descriptors, so target-label leakage is not introduced.
- Imputation, scaling, PCA, and clustering are fit on the complete observed population. This is explicitly documented as an in-sample descriptive analysis, not holdout validation.
- The README and notebook state that a production predictive extension should use an earlier-period fit and later time-based holdout. The review does **not** claim such a holdout was run.

### 7. Metrics and README claims — corrected

- The original README stated dataset sizes, PCA dimensions, and KMeans/DBSCAN scores without a committed dataset or reproducible run in this checkout.
- Those unverified numeric claims were removed. Cluster names are now explicitly treated as arbitrary identifiers until profiling is performed on supplied data.
- No new metric values were invented.

## Files changed

- `Bank_Customer_Segmentation.ipynb` — rewritten portable workflow; removed outputs and execution counts; fixed syntax, paths, aggregation, reproducibility, and metric guards.
- `README.md` — accurate data contract, setup, unsupervised validation caveats, and static-check commands.
- `requirements.txt` — declared NumPy, pandas, scikit-learn, matplotlib, seaborn, Jupyter, and nbformat version ranges.
- `.gitignore` — excludes local/private datasets, virtual environments, notebook checkpoints, caches, and generated output directories.
- `scripts/validate_notebook.py` — dependency-free AST/JSON/portability/output validator.
- `tests/test_static_validation.py` — standard-library tests for notebook presence and static validity.
- `ENGINEERING_REVIEW.md` — this report.

## Validation performed

Passed:

```text
ruff check .                         All checks passed
python scripts/validate_notebook.py  OK: Bank_Customer_Segmentation.ipynb passed static notebook validation
python -m unittest discover -s tests -v
  Ran 2 tests in 0.00s — OK
git diff --check                     passed
python -m py_compile scripts/validate_notebook.py tests/test_static_validation.py  passed
```

Static validation confirmed 20 notebook cells, zero saved outputs, and zero execution counts. The validator also rejects `/content/` and Windows/Colab-style machine-specific paths.

## Execution caveat

A full notebook run was **not claimed**. The supplied checkout contains no transaction CSV. The current Sandbox has the declared ML dependencies available and was used only for the synthetic smoke execution documented below; a real run still requires the user-supplied dataset. Consequently, no production clustering or RFM result values are included in this review.

## Local Git state

Changes are intended to be committed locally only. No push or remote mutation is part of this review.

## Second Senior Review

**Review date:** 2026-10-15
**Scope:** complete tracked checkout, including the notebook, Python validator/tests, README, requirements, ignore rules, Git history/diff, and repository state.

### Findings and fixes

1. **Pandas runtime incompatibility — fixed.** The notebook called `DataFrame.describe(datetime_is_numeric=True)`. The declared pandas range includes pandas 2.3, where this keyword is no longer accepted; a dependency-compatible synthetic execution failed with `TypeError`. The call now uses the portable `df.describe(include="all")` form.
2. **Silhouette edge cases — fixed.** The previous guard only checked for more than one label. Silhouette is undefined when every row has a distinct label (`n_labels == n_samples`), and DBSCAN can provide zero or one non-noise rows. A shared guard now requires `2 <= n_labels <= n_samples - 1` and reports `NaN` with an explanation instead of raising or inventing a score.
3. **Blank identifiers — fixed.** Whitespace is stripped from `CustomerID` and `TransactionID`; blank identifiers are now included in the invalid-row count and actually excluded before grouping. Missing/invalid numeric transaction amounts and non-finite values are also rejected from required transaction rows.
4. **Transaction frequency and balance quality — fixed.** Customer frequency now counts distinct `TransactionID` values rather than duplicate rows. `ZeroBalanceRatio` excludes missing account balances from its denominator and returns undefined when no balance is observed, allowing the existing imputer to handle it honestly.
5. **Notebook validation coverage — fixed.** The validator now discovers every repository notebook, validates notebook structure/source types/cell types/duplicate cell IDs, parses every code cell, rejects saved outputs and execution counts, and checks forbidden machine-specific paths. Tests now cover all-notebook validation and a deliberately invalid notebook fixture.

### Second-review validation

Passed in this checkout:

```text
python scripts/validate_notebook.py       OK: 1 notebook(s) passed static validation
python -m unittest discover -s tests -v  Ran 3 tests — OK
ruff check .                              All checks passed
python -m py_compile scripts/validate_notebook.py tests/test_static_validation.py
                                           passed
git diff --check                         passed
```

A dependency-backed smoke execution was also run with a four-customer, contract-valid temporary CSV created only for testing. It completed all notebook code without exception and correctly left KMeans/DBSCAN silhouette as undefined for mathematically invalid label cardinalities. This is not a production dataset run and no metrics are claimed from it.

### Remaining risks and boundaries

- The real transaction dataset is absent, so no full-data clustering, data-distribution review, or production metric validation is possible from this checkout.
- The notebook is an in-sample unsupervised descriptive analysis. It does not implement a time-based holdout, trained artifact export, serving API, or UI.
- The fixed four-cluster KMeans choice and DBSCAN parameters remain analysis choices requiring domain/data validation; cluster IDs have no inherent business meaning.
- `dayfirst=True` is retained as an explicit input contract for the supplied CSV. Ambiguous date formats or semantically incorrect source values still require upstream data stewardship.
