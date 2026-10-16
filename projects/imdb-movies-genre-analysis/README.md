# IMDb Movies Genre Analysis

Exploratory analysis and regression/clustering experiments for the [IMDb 5000+ Movies Multiple Genres dataset](https://www.kaggle.com/datasets/rakkesharv/imdb-5000-movies-multiple-genres-dataset).

## Requirements

Python 3.10+ is recommended. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Dataset setup

The dataset is **not included** in this repository. Download the Kaggle CSV and either:

- place it at `data/IMDb_All_Genres_etf_clean1.csv`, or
- set `IMDB_DATA_PATH` to the CSV path.

The notebook fails early with a clear `FileNotFoundError` when the file is absent; it does not create or substitute data.

## Run

Open `IMDb 5000+ Movies & Multiple Genres .ipynb` from the repository root and run it after installing dependencies and providing the data. The supervised target is `Total_Gross`. The holdout split is deterministic (`random_state=42`), and preprocessing, feature selection, and grid search are fitted on training data only. Target-derived exploratory columns are deliberately excluded from the regression feature matrix to prevent leakage.

## Validation

Run the repository checks from the root:

```bash
pytest -q
ruff check --select E9,F821,F811 '*.ipynb'
```

The full notebook is not executed by CI because it requires the external dataset and can be computationally expensive. No model metrics in this repository should be treated as freshly reproduced unless the notebook is run with the dataset.
