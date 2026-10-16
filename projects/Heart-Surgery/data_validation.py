"""Data-contract and normalization helpers for the cardiac-surgery notebook."""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

REQUIRED_COLUMNS = frozenset(
    {
        "Region",
        "Procedure",
        "Year of Hospital Discharge",
        "Number of Cases",
        "Number of Deaths",
        "Expected Mortality Rate",
        "Observed Mortality Rate",
        "Comparison Results",
    }
)
OPTIONAL_IDENTIFIER_COLUMNS = ("Facility ID", "Hospital Name", "Detailed Region")
YEAR_COLUMN = "Year of Hospital Discharge"
COUNT_COLUMNS = ("Number of Cases", "Number of Deaths")
RATE_COLUMNS = ("Expected Mortality Rate", "Observed Mortality Rate")


def _missing_columns(columns: Iterable[str]) -> list[str]:
    return sorted(REQUIRED_COLUMNS.difference(columns))


def parse_year(values: pd.Series) -> pd.Series:
    """Parse numeric years and date-like years without treating 2015 as nanoseconds."""
    numeric = pd.to_numeric(values, errors="coerce")
    date_year = pd.to_datetime(values, errors="coerce").dt.year
    parsed = numeric.where(numeric.between(1900, 2100), date_year)
    return parsed.astype("Float64")


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the input data before exploratory analysis or modeling."""
    missing = _missing_columns(data.columns)
    if missing:
        raise ValueError("The input dataset is missing required columns: " + ", ".join(missing))

    result = data.drop(columns=list(OPTIONAL_IDENTIFIER_COLUMNS), errors="ignore").copy()
    result[YEAR_COLUMN] = parse_year(result[YEAR_COLUMN])
    for column in (*COUNT_COLUMNS, *RATE_COLUMNS):
        converted = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(
                f"Column {column!r} contains {int(invalid.sum())} non-numeric value(s)."
            )
        result[column] = converted

    for column in COUNT_COLUMNS:
        if (result[column].dropna() < 0).any():
            raise ValueError(f"Column {column!r} cannot contain negative values.")
    for column in RATE_COLUMNS:
        if ((result[column].dropna() < 0) | (result[column].dropna() > 100)).any():
            raise ValueError(f"Column {column!r} must be between 0 and 100 percent.")
    both_counts = result[list(COUNT_COLUMNS)].dropna()
    if (both_counts["Number of Deaths"] > both_counts["Number of Cases"]).any():
        raise ValueError("Number of Deaths cannot exceed Number of Cases.")
    if result[YEAR_COLUMN].notna().sum() == 0:
        raise ValueError(f"Column {YEAR_COLUMN!r} has no parseable year values.")
    return result
